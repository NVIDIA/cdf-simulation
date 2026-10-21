#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Solder Fillet Board Placer — Script Editor script.

Finds every instance of the configured component types in camera view and
places two solder fillet meshes (one per pad end) for each. The component
types and the board root come from the caller — this module names no board.

UI controls: shape, noise, material, camera path.
Buttons: "Generate Fillets" (places/regenerates) | "Clear" (removes all).

Paste into Omniverse Composer Script Editor with temp_scene.usd open.
"""

import os

import numpy as np
import omni.usd
from omni import ui
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

# ── Fractal noise ─────────────────────────────────────────────────────────────


def _fractal_noise_2d(x, y, scale, octaves, seed=42):
    noise = np.zeros_like(x, dtype=float)
    amplitude = 1.0
    frequency = float(scale)
    np.random.seed(seed)
    for _ in range(max(1, int(octaves))):
        px = np.random.random() * 2 * np.pi
        py = np.random.random() * 2 * np.pi
        ang = np.random.random() * 2 * np.pi
        xr = x * np.cos(ang) - y * np.sin(ang)
        yr = x * np.sin(ang) + y * np.cos(ang)
        noise += amplitude * (
            np.sin(2 * np.pi * frequency * xr + px)
            * np.cos(2 * np.pi * frequency * yr + py)
        )
        amplitude *= 0.5
        frequency *= 2.0
    return noise


# ── Warp GPU acceleration ────────────────────────────────────────────────────
# The fillet's per-grid-point profile (sigmoid plateau × exp decay × fractal
# noise) is embarrassingly parallel — each (i, j) is independent. Move the
# whole grid kernel onto cuda:0 via warp; falls back to NumPy when warp/cuda
# isn't available, so the function is safe to call on CPU-only test rigs.
#
# Set FILLET_BACKEND=cpu in the environment to force the NumPy fallback path
# (useful for debugging or perf comparison).

try:  # warp is bundled with Kit; outside Kit it may be missing.
    import warp as _wp

    _WP_OK = True
except Exception:  # noqa: BLE001
    _wp = None
    _WP_OK = False

_WP_MAX_OCTAVES = 16  # bumped from PARAMS.noise_octaves default of 4; covers any
# reasonable user setting without dynamic shape compilation
_WP_INIT_DONE = False


def _wp_init() -> bool:
    """Initialize warp once. Returns True if a CUDA device is available; False
    otherwise (caller should fall back to NumPy)."""
    global _WP_INIT_DONE
    if not _WP_OK:
        return False
    if _WP_INIT_DONE:
        return True
    try:
        _wp.init()
        # If there's no CUDA device, get_preferred_device returns "cpu" — in
        # that case there's no point routing through warp (its overhead beats
        # NumPy on CPU-bound problems this small).
        dev = str(_wp.get_preferred_device())
        if not dev.startswith("cuda"):
            return False
        _WP_INIT_DONE = True
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[Fillet/Warp] init failed, falling back to CPU: {exc!r}")
        return False


if _WP_OK:

    @_wp.kernel
    def _wp_fillet_z_kernel(
        z_out: _wp.array2d(dtype=_wp.float32),
        long_e: _wp.float32,
        short: _wp.float32,
        x_len: _wp.float32,
        decay: _wp.float32,
        smooth_y: _wp.float32,
        plat_start: _wp.float32,
        plat_end: _wp.float32,
        bump: _wp.float32,
        noise_amp: _wp.float32,
        n_octaves: _wp.int32,
        noise_freq0: _wp.float32,
        rng_px: _wp.array(dtype=_wp.float32),
        rng_py: _wp.array(dtype=_wp.float32),
        rng_ang: _wp.array(dtype=_wp.float32),
        r: _wp.int32,
    ):
        """One thread per grid cell. Writes z (height) into z_out[i, j].

        Mirrors the NumPy expression in :func:`_compute_geometry`:
          xn = x / long_e
          yn = y / short
          dx = exp(-decay * xn)
          lr = sigmoid(smooth_y * (yn - plat_start))
          rf = sigmoid(smooth_y * (yn - plat_end))
          delta = long_e * bump * dx * lr * (1 - rf)
          noise = sum_octaves( amp_k · sin(2πf_k·xr+px) · cos(2πf_k·yr+py) )
          z = delta + noise_amp * noise
        """
        i, j = _wp.tid()
        if i >= r or j >= r:
            return

        # Linear x/y across the grid: x in [0, x_len], y in [0, short].
        # The Python code uses ``np.linspace(0, x_len, r)`` — last sample is
        # exactly x_len, so spacing is x_len / (r - 1) when r >= 2.
        denom = _wp.float32(_wp.max(r - 1, 1))
        x = _wp.float32(j) * (x_len / denom)
        y = _wp.float32(i) * (short / denom)

        xn = x / long_e
        yn = y / short

        dx = _wp.exp(-decay * xn)
        lr = _wp.float32(1.0) / (
            _wp.float32(1.0) + _wp.exp(-smooth_y * (yn - plat_start))
        )
        rf = _wp.float32(1.0) / (
            _wp.float32(1.0) + _wp.exp(-smooth_y * (yn - plat_end))
        )
        delta = long_e * bump * dx * lr * (_wp.float32(1.0) - rf)

        # Fractal noise (mirror of _fractal_noise_2d): per-octave rotation +
        # phase offsets are sampled CPU-side and passed in; here we only
        # consume them deterministically.
        noise = _wp.float32(0.0)
        amp = _wp.float32(1.0)
        freq = noise_freq0
        TWO_PI = _wp.float32(6.2831853)
        for k in range(n_octaves):
            ang = rng_ang[k]
            px = rng_px[k]
            py = rng_py[k]
            xr = xn * _wp.cos(ang) - yn * _wp.sin(ang)
            yr = xn * _wp.sin(ang) + yn * _wp.cos(ang)
            noise += amp * (
                _wp.sin(TWO_PI * freq * xr + px) * _wp.cos(TWO_PI * freq * yr + py)
            )
            amp = amp * _wp.float32(0.5)
            freq = freq * _wp.float32(2.0)

        z_out[i, j] = delta + noise_amp * noise


def _compute_z_warp(p: "FilletParams") -> "np.ndarray | None":
    """GPU path. Returns the (r, r) z-grid as a NumPy array, or None if warp
    isn't usable (caller should fall back to NumPy)."""
    if not _wp_init():
        return None

    r = max(4, int(p.resolution))
    long_e = float(p.long_edge)
    short = long_e * 0.6
    x_len = long_e * float(p.x_scale)
    n_oct = max(1, min(_WP_MAX_OCTAVES, int(p.noise_octaves)))

    # Reproduce CPU side's fractal-noise RNG draw order so warp output bit
    # matches the NumPy reference (same seed → same shape).
    np.random.seed(42)
    rng_px = np.zeros(_WP_MAX_OCTAVES, dtype=np.float32)
    rng_py = np.zeros(_WP_MAX_OCTAVES, dtype=np.float32)
    rng_ang = np.zeros(_WP_MAX_OCTAVES, dtype=np.float32)
    for k in range(n_oct):
        rng_px[k] = float(np.random.random() * 2 * np.pi)
        rng_py[k] = float(np.random.random() * 2 * np.pi)
        rng_ang[k] = float(np.random.random() * 2 * np.pi)

    try:
        z_dev = _wp.zeros((r, r), dtype=_wp.float32, device="cuda:0")
        rng_px_d = _wp.array(rng_px, dtype=_wp.float32, device="cuda:0")
        rng_py_d = _wp.array(rng_py, dtype=_wp.float32, device="cuda:0")
        rng_ang_d = _wp.array(rng_ang, dtype=_wp.float32, device="cuda:0")
        _wp.launch(
            kernel=_wp_fillet_z_kernel,
            dim=(r, r),
            inputs=[
                z_dev,
                np.float32(long_e),
                np.float32(short),
                np.float32(x_len),
                np.float32(p.decay),
                np.float32(p.smooth_y),
                np.float32(p.plat_start),
                np.float32(p.plat_end),
                np.float32(p.bump),
                np.float32(p.noise_amp),
                np.int32(n_oct),
                np.float32(p.noise_scale),
                rng_px_d,
                rng_py_d,
                rng_ang_d,
                np.int32(r),
            ],
            device="cuda:0",
        )
        z_host = z_dev.numpy().astype(np.float64, copy=False)
        return z_host
    except Exception as exc:  # noqa: BLE001
        print(f"[Fillet/Warp] launch failed, falling back to CPU: {exc!r}")
        return None


# ── Parameters ────────────────────────────────────────────────────────────────


class FilletParams:
    # Shape (map to editor_solder_fillet.py fields)
    long_edge = 9.0  # long_edge_length
    x_scale = 0.2  # x_length_scale  → x_length = long_edge * x_scale
    plat_start = 0.2  # platform_start
    plat_end = 0.8  # platform_end
    # Defaults match the production reference run's tuned single-shot
    # values (board_0505 fixed-camera config). YAML pipeline pulls
    # ±10 % randomization ranges from ``randomize_fillet:`` — the
    # editor stays at the midpoint scalars below.
    smooth_y = 6.44  # smoothness_y
    bump = 0.38  # delta_max_scale
    decay = 10.8  # delta_x_decay
    # Decay used by PAD-AWARE fillets only (bbox/production fillets keep
    # ``decay``). The production decay (10.8, i.e. k = decay * x_scale =
    # 2.16 across the mesh) drops the tapered profile below 10 % of the
    # crest by ~75 % of the run: on low parts (0603 crest ≈ 0.32 mm) the
    # outer quarter of the meniscus is < 0.03 mm — a flat metallic sheet
    # that reads as BARE PAD at oblique angles even though the mesh
    # geometrically reaches the pad edge. 5.0 (k = 1.0) keeps a visible
    # slope over the whole exposed pad while staying clearly concave.
    pad_decay = 5.0
    # Solid-body fillets (opt-in). When True the open heightfield sheet is
    # closed into a watertight solid: the boundary ring is skirted down to
    # ``-solid_depth_frac * long_edge * bump`` and capped with one bottom
    # polygon, so the mesh reads as a volume from grazing angles and the
    # default catmullClark subdivision rounds the lower edges slightly.
    # Default False so existing output stays byte-identical.
    solid = False
    solid_depth_frac = 0.15
    # Noise
    noise_scale = 4.0
    noise_octaves = 4.0
    noise_amp = 0.01  # absolute amplitude (NOT scaled by long_edge)
    resolution = 20.0
    # USD scale applied to every fillet prim (xformOp:transform bakes this in)
    fillet_sx = 0.14
    fillet_sy = 0.14
    fillet_sz = 0.24
    # Material (UsdPreviewSurface). mat_color is the baseColor shown on
    # every fillet prim — defaults to mid-grey silver (0.85) to mimic
    # real solder. Override via yaml ``solder_fillet.mat_color`` /
    # editor "Fillet color" slider for vantablack-style black fillets
    # (e.g. when using a fully black PCBA reference).
    mat_color = (0.85, 0.85, 0.85)
    mat_metallic = 1.0
    mat_roughness = 0.2
    # Position fine-tune
    inward_offset_mm = (
        2.6  # absolute inward offset (mm); used when adaptive_scale=False
    )
    inward_frac = 0.15  # fraction of each component's half-length used as inward offset
    # when adaptive_scale=True (replaces inward_offset_mm per-component)
    z_offset_mm = 10.0  # move both fillets upward (along Z, mm)
    # Adaptive sizing: scale fillet sx/sy to match each component's world-space bbox width.
    # Also applies inward_frac instead of inward_offset_mm.
    # When True, fillet_sx/sy from PARAMS are ignored; sz and z_offset_mm are still used.
    # Default is False so the editor's "Scale X/Y" and "Inward (mm)" sliders take
    # effect directly. The pipeline overrides this to True via yaml (see
    # sdg_pipeline.py: `_sf_cfg.get("adaptive_scale", True)`), so production
    # behaviour is unchanged.
    adaptive_scale = False
    # Scale fillet height with the adaptive width factor (keeps design
    # aspect ratio). Off by default: the shipped tuning uses the absolute
    # fillet_sz.
    adaptive_sz = False
    # height calibration for adaptive_sz (fraction of the aspect-true height;
    # ~0.3 gives realistic meniscus height on mm-unit CAD boards)
    adaptive_sz_factor = 1.0
    # Scene — must match the PCBA assembly being scanned (top side).
    # ``pcba_root`` and ``comp_types`` name geometry inside ONE board, so they
    # carry no default: the pipeline sets them from the pcba target config and
    # the Script Editor sets them from the UI fields. A baked-in default is a
    # silent wrong answer on every other board — it finds nothing and reports
    # zero fillets as if the scene had none. Never hardcode board paths.
    camera_path = "/World/camera_light/Camera"  # capital C matches config
    pcba_root = ""
    comp_types = []  # noqa: RUF012
    # Pad-aware placement (opt-in). When True AND a component-instance
    # sidecar (<board>.components.json) is available, each fillet is shaped
    # from the component's own solder-paste pad geometry instead of the
    # coarse component bbox: footprint spans termination face → pad outer
    # edge, width follows the pad, base sits on the pad surface (correct on
    # both board sides), height is proportioned to the termination height.
    # Default False so existing behaviour is untouched.
    pad_aware = False
    # Explicit sidecar path (set at runtime by the caller/harness). When
    # empty, the path is derived from the stage root layer:
    # <root_layer_dir>/<stem>.components.json. Never hardcode board paths.
    sidecar_path = ""
    # Pad-local sizing prevents package-height fins beside gull-wing pins by
    # measuring the contact foot that actually touches each paste pad.
    lead_local_sizing = False


PARAMS = FilletParams()

FILLET_ROOT = "/World/SolderFillets"
MAT_PATH = "/World/SolderFilletMat"


# ── Mesh geometry ─────────────────────────────────────────────────────────────

# Cache for the per-call invariants: when ``FilletParams`` doesn't change
# between successive calls (the common case — every fillet placed in one
# ``generate_fillets()`` pass uses the same global PARAMS), reuse the
# previous geometry instead of recomputing the grid + Vt arrays.
_GEOM_CACHE: "dict[tuple, tuple]" = {}
_GEOM_CACHE_MAX = 4


def _params_cache_key(p: "FilletParams") -> tuple:
    return (
        float(p.long_edge),
        float(p.x_scale),
        float(p.plat_start),
        float(p.plat_end),
        float(p.smooth_y),
        float(p.bump),
        float(p.decay),
        float(p.noise_scale),
        int(p.noise_octaves),
        float(p.noise_amp),
        int(p.resolution),
        bool(getattr(p, "solid", False)),
        float(getattr(p, "solid_depth_frac", 0.15)),
    )


def _compute_geometry(
    p: FilletParams,
    backend: "str | None" = None,
    use_cache: bool = True,
    taper_to_zero: bool = False,
    decay_override: "float | None" = None,
):
    """Compute the per-grid-point fillet profile and triangulation.

    ``backend`` selects the math kernel:
      * ``None`` (default) — pick warp+CUDA if available, else NumPy.
        Override globally via env var ``FILLET_BACKEND=cpu|warp``.
      * ``"cpu"`` — force the NumPy fallback path.
      * ``"warp"`` — force the warp path; if warp/CUDA isn't usable,
        falls back to NumPy and prints a one-line warning.

    ``use_cache`` (default True): when set, identical successive calls
    skip the recompute entirely. The cache is keyed on every
    geometry-affecting param (long_edge / x_scale / plat_start / plat_end
    / smooth_y / bump / decay / noise_* / resolution / taper_to_zero), so
    as soon as any of them change the cache invalidates correctly.

    ``taper_to_zero`` (default False): the raw exp-decay profile still
    holds ``exp(-decay * x_scale)`` (~11 % of the crest with production
    params) at the far mesh edge, which reads as a cliff when the mesh
    edge must land exactly on a paste-pad boundary. When set, the decay
    is renormalized to ``(dx - dx_end) / (1 - dx_end)`` so the profile
    keeps its crest (factor 1 at x = 0) and its concavity but feathers to
    exactly zero at the far edge — used by pad-aware placement, where the
    fillet foot ends precisely at the pad's outer edge.

    ``decay_override``: alternative decay for the tapered profile (only
    meaningful together with ``taper_to_zero``; ignored otherwise). Lets
    pad-aware placement use a fuller meniscus (``PARAMS.pad_decay``)
    without touching the production ``decay`` shared with the bbox path.
    """
    if backend is None:
        backend = os.environ.get("FILLET_BACKEND", "auto").lower()

    eff_decay = float(p.decay if decay_override is None else decay_override)
    key = (
        (_params_cache_key(p) + (bool(taper_to_zero), eff_decay)) if use_cache else None
    )
    if key is not None and key in _GEOM_CACHE:
        return _GEOM_CACHE[key]

    # Pad-aware fillets share their stage-free reference geometry with the
    # declarative `ProceduralSolderSolver`. Keep the legacy bbox/production
    # path below unchanged, including its optional Warp acceleration.
    if taper_to_zero:
        from pcb_aoi.solder_generation.procedural_meniscus_geometry import (
            MeniscusParameters,
            compute_meniscus_geometry,
        )

        shared_parameters = MeniscusParameters(
            long_edge=float(p.long_edge),
            x_scale=float(p.x_scale),
            platform_start=float(p.plat_start),
            platform_end=float(p.plat_end),
            smoothness_y=float(p.smooth_y),
            bump=float(p.bump),
            decay=float(p.decay),
            pad_decay=float(
                p.decay if decay_override is None else decay_override,
            ),
            noise_scale=float(p.noise_scale),
            noise_octaves=int(p.noise_octaves),
            noise_amplitude=float(p.noise_amp),
            resolution=max(4, int(p.resolution)),
            solid=bool(getattr(p, "solid", False)),
            solid_depth_fraction=float(
                getattr(p, "solid_depth_frac", 0.15),
            ),
        )
        numeric_points, counts, idx = compute_meniscus_geometry(
            shared_parameters,
        )
        try:
            from pxr import Vt

            pts = Vt.Vec3fArray.FromNumpy(
                np.asarray(numeric_points, dtype=np.float32),
            )
        except Exception:  # noqa: BLE001
            pts = [Gf.Vec3f(*point) for point in numeric_points]
        out = (pts, list(counts), list(idx))
        if key is not None:
            if len(_GEOM_CACHE) >= _GEOM_CACHE_MAX:
                _GEOM_CACHE.pop(next(iter(_GEOM_CACHE)))
            _GEOM_CACHE[key] = out
        return out

    long_e = float(p.long_edge)
    short = long_e * 0.6
    x_len = long_e * float(p.x_scale)
    r = max(4, int(p.resolution))

    # Build the (r, r) X / Y grids once on CPU — used both for the
    # ``pts`` array (regardless of compute backend) and for the NumPy
    # fallback. Indexing convention matches np.meshgrid(..., indexing='xy'),
    # i.e. X[i, j] varies with j (column), Y[i, j] varies with i (row),
    # to keep the warp kernel addressing scheme consistent.
    x_lin = np.linspace(0.0, x_len, r, dtype=np.float64)
    y_lin = np.linspace(0.0, short, r, dtype=np.float64)
    X, Y = np.meshgrid(x_lin, y_lin)

    z = None
    if backend in ("auto", "warp"):
        z = _compute_z_warp(p)
        if z is None and backend == "warp":
            print("[Fillet/Warp] warp unavailable — using NumPy")

    if z is None:
        # CPU fallback (the original NumPy implementation).
        xn = X / long_e
        yn = Y / short
        dx = np.exp(-float(p.decay) * xn)
        lr = 1.0 / (1.0 + np.exp(-float(p.smooth_y) * (yn - float(p.plat_start))))
        rf = 1.0 / (1.0 + np.exp(-float(p.smooth_y) * (yn - float(p.plat_end))))
        dy = lr * (1.0 - rf)
        delta = long_e * float(p.bump) * dx * dy
        noise = _fractal_noise_2d(xn, yn, p.noise_scale, p.noise_octaves)
        z = delta + float(p.noise_amp) * noise

    if taper_to_zero:
        # Replace the raw exp-decay term with its renormalized (and, with
        # ``decay_override``, reshaped) form so the profile is exactly zero
        # at x = x_len. Done as a post-pass on the z grid (works identically
        # for the warp and NumPy backends): the deterministic delta
        # component the backend baked in (always ``p.decay``) is recomputed
        # on CPU and subtracted, then the tapered profile at the effective
        # decay is added; the (tiny, ±noise_amp) fractal noise floor is
        # left untouched.
        xn_g = X / long_e
        yn_g = Y / short
        lr_g = 1.0 / (1.0 + np.exp(-float(p.smooth_y) * (yn_g - float(p.plat_start))))
        rf_g = 1.0 / (1.0 + np.exp(-float(p.smooth_y) * (yn_g - float(p.plat_end))))
        dy_g = lr_g * (1.0 - rf_g)
        dx_raw = np.exp(-float(p.decay) * xn_g)
        dx_eff = np.exp(-eff_decay * xn_g)
        dx_end = np.exp(-eff_decay * float(p.x_scale))
        if dx_end < 1.0:
            delta_raw = long_e * float(p.bump) * dx_raw * dy_g
            delta_tapered = (
                long_e * float(p.bump) * dy_g * (dx_eff - dx_end) / (1.0 - dx_end)
            )
            z = z - delta_raw + delta_tapered

    # Bulk-build a Vt.Vec3fArray straight from the (r, r) X/Y/z grids —
    # an order-of-magnitude faster than a Python list comp constructing
    # r*r Gf.Vec3f objects one at a time.
    flat = np.empty((r * r, 3), dtype=np.float32)
    flat[:, 0] = X.reshape(-1).astype(np.float32, copy=False)
    flat[:, 1] = Y.reshape(-1).astype(np.float32, copy=False)
    flat[:, 2] = z.reshape(-1).astype(np.float32, copy=False)
    try:
        from pxr import Vt

        pts = Vt.Vec3fArray.FromNumpy(flat)
    except Exception:  # noqa: BLE001
        # Older USD without FromNumpy — fall back to list comp.
        pts = [
            Gf.Vec3f(float(flat[k, 0]), float(flat[k, 1]), float(flat[k, 2]))
            for k in range(flat.shape[0])
        ]

    # Triangulation indices: for each (r-1) × (r-1) quad, two triangles.
    # Build with vectorised arange + reshape in O(quads), no Python loop.
    n_quads = (r - 1) * (r - 1)
    qi, qj = np.meshgrid(np.arange(r - 1), np.arange(r - 1), indexing="ij")
    a = (qi * r + qj).reshape(-1)
    b = (qi * r + qj + 1).reshape(-1)
    c = ((qi + 1) * r + qj).reshape(-1)
    d = ((qi + 1) * r + qj + 1).reshape(-1)
    idx = np.empty(n_quads * 6, dtype=np.int32)
    idx[0::6], idx[1::6], idx[2::6] = a, b, c
    idx[3::6], idx[4::6], idx[5::6] = b, d, c
    idx = idx.tolist()
    counts = [3] * (n_quads * 2)

    if getattr(p, "solid", False):
        # Close the open sheet into a watertight solid: skirt the boundary
        # ring down to a shallow base plane and cap it with one bottom
        # polygon. With the default catmullClark subdivision the lower
        # edges/corners round off slightly.
        depth = long_e * float(p.bump) * float(getattr(p, "solid_depth_frac", 0.15))
        ring = np.concatenate(
            [
                np.arange(r),  # row i=0, +x
                np.arange(1, r) * r + (r - 1),  # col j=r-1, +y
                (r - 1) * r + np.arange(r - 2, -1, -1),  # row i=r-1, -x
                np.arange(r - 2, 0, -1) * r,  # col j=0, -y
            ]
        ).astype(np.int32)
        n_ring = int(ring.shape[0])
        bflat = np.empty((n_ring, 3), dtype=np.float32)
        bflat[:, 0] = flat[ring, 0]
        bflat[:, 1] = flat[ring, 1]
        bflat[:, 2] = -np.float32(depth)
        flat = np.concatenate([flat, bflat], axis=0)
        base = r * r
        skirt = []
        for k in range(n_ring):
            k1 = (k + 1) % n_ring
            skirt.extend([int(ring[k]), base + k, base + k1, int(ring[k1])])
        idx = idx + skirt + [base + k for k in range(n_ring - 1, -1, -1)]
        counts = counts + [4] * n_ring + [n_ring]
        try:
            from pxr import Vt

            pts = Vt.Vec3fArray.FromNumpy(flat)
        except Exception:  # noqa: BLE001
            pts = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in flat]

    out = (pts, counts, idx)
    if key is not None:
        if len(_GEOM_CACHE) >= _GEOM_CACHE_MAX:
            _GEOM_CACHE.pop(next(iter(_GEOM_CACHE)))
        _GEOM_CACHE[key] = out
    return out


# ── Material ──────────────────────────────────────────────────────────────────

_shader_prim = [None]  # mutable ref so callbacks can update it


def _ensure_material(stage, p: FilletParams):
    base_rgb = (
        Gf.Vec3f(*[float(x) for x in p.mat_color])
        if not isinstance(p.mat_color, Gf.Vec3f)
        else p.mat_color
    )
    existing = stage.GetPrimAtPath(MAT_PATH)
    if existing.IsValid() and _shader_prim[0] is not None:
        s = _shader_prim[0]
        s.GetInput("metallic").Set(float(p.mat_metallic))
        s.GetInput("roughness").Set(float(p.mat_roughness))
        # baseColor input may not exist if first call set it; guard.
        bc = s.GetInput("baseColor")
        if bc:
            bc.Set(base_rgb)
        else:
            s.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set(base_rgb)
        return UsdShade.Material(existing)

    mat = UsdShade.Material.Define(stage, MAT_PATH)
    shader = UsdShade.Shader.Define(stage, f"{MAT_PATH}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(p.mat_metallic))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        float(p.mat_roughness)
    )
    shader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set(base_rgb)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _shader_prim[0] = shader
    return mat


# ── Single fillet mesh ────────────────────────────────────────────────────────


def _write_mesh(
    stage,
    path,
    p: FilletParams,
    mat,
    taper_to_zero: bool = False,
    decay_override: "float | None" = None,
):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        mesh = UsdGeom.Mesh.Define(stage, path)
        UsdShade.MaterialBindingAPI(mesh).Bind(mat)
    else:
        mesh = UsdGeom.Mesh(prim)
    pts, counts, idx = _compute_geometry(
        p, taper_to_zero=taper_to_zero, decay_override=decay_override
    )
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(idx)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return mesh


def _set_transform(stage, path, mat4: Gf.Matrix4d):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(mat4)


def _fillet_world_matrix(
    origin: Gf.Vec3d,
    x_dir: Gf.Vec3d,  # fillet local X in world space (unit vector)
    y_dir: Gf.Vec3d,  # fillet local Y in world space (unit vector)
    z_dir: Gf.Vec3d,  # fillet local Z in world space (unit vector)
    sx: float,
    sy: float,
    sz: float,
) -> Gf.Matrix4d:
    """Row-vector convention: p_world = p_local * M."""
    return Gf.Matrix4d(
        x_dir[0] * sx,
        x_dir[1] * sx,
        x_dir[2] * sx,
        0.0,
        y_dir[0] * sy,
        y_dir[1] * sy,
        y_dir[2] * sy,
        0.0,
        z_dir[0] * sz,
        z_dir[1] * sz,
        z_dir[2] * sz,
        0.0,
        origin[0],
        origin[1],
        origin[2],
        1.0,
    )


# ── Camera frustum ────────────────────────────────────────────────────────────


def _mm_to_scene(stage, mm: float) -> float:
    """Convert physical mm to scene units using stage metersPerUnit."""
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    return mm * 0.001 / mpu


def _get_frustum(stage, camera_path):
    """
    Return GfFrustum in world space (scene units), or None.
    Compatible with pxr v0.25 / USD 21.x where GfCamera.GetFrustum() may not exist.
    """
    import math as _math

    cam_prim = stage.GetPrimAtPath(camera_path)
    if not cam_prim.IsValid():
        print(f"[Fillet] Camera not found: {camera_path} — including all instances.")
        return None

    cam_schema = UsdGeom.Camera(cam_prim)
    xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_mat = xfc.GetLocalToWorldTransform(cam_prim)

    # Try GetCamera().GetFrustum() first (USD ≥ 22.03)
    try:
        gf_cam = cam_schema.GetCamera(Usd.TimeCode.Default())
        frustum = gf_cam.GetFrustum()
        frustum.Transform(world_mat)
        return frustum
    except AttributeError:
        pass

    # Fallback: build GfFrustum from USD camera attributes.
    # USD stores hap/vap/fl in *tenths of scene units* (film-format compat).
    # Divide by 10 to convert to scene units before building the frustum.
    fl = cam_schema.GetFocalLengthAttr().Get()
    hap = cam_schema.GetHorizontalApertureAttr().Get()
    vap = cam_schema.GetVerticalApertureAttr().Get()
    crng = cam_schema.GetClippingRangeAttr().Get()
    proj = cam_schema.GetProjectionAttr().Get()

    fl = float(fl) / 10.0 if fl is not None else 5.0
    hap = float(hap) / 10.0 if hap is not None else 3.6
    vap = float(vap) / 10.0 if vap is not None else 2.4
    near = float(crng[0]) if crng is not None else 1.0
    far = float(crng[1]) if crng is not None else 2_000_000.0

    try:
        frustum = Gf.Frustum()
        if proj == "orthographic":
            frustum.SetOrthographic(-hap / 2, hap / 2, -vap / 2, vap / 2, near, far)
        else:
            fov_v = _math.degrees(2.0 * _math.atan(vap / (2.0 * fl)))
            aspect = hap / vap
            try:
                frustum.SetPerspective(fov_v, True, aspect, near, far)
            except TypeError:
                frustum.SetPerspective(fov_v, aspect, near, far)
        frustum.Transform(world_mat)
        return frustum
    except Exception as e:  # noqa: BLE001
        print(f"[Fillet] Frustum build failed: {e} — including all instances.")
        return None


def _in_frustum(frustum, world_pos: Gf.Vec3d, margin_scene: float = 0.0) -> bool:
    if frustum is None:
        return True
    rng = Gf.Range3d(
        world_pos - Gf.Vec3d(margin_scene, margin_scene, margin_scene),
        world_pos + Gf.Vec3d(margin_scene, margin_scene, margin_scene),
    )
    return frustum.Intersects(Gf.BBox3d(rng))


# ── Instance discovery ────────────────────────────────────────────────────────


def _find_instances(
    stage, camera_path, pcba_root, margin_mm: float = 5.0, comp_types=None
):
    """
    Return list of (prim, world_mat) for all instance prims whose USD path
    contains any string in comp_types, are within the camera frustum, and
    belong to pcba_root.

    comp_types: list of substrings matched against each prim path, naming this
                board's component families. Required — there is no default.
    margin_mm:  physical bbox expansion in mm around each component centre.

    Raises:
        ValueError: If ``comp_types`` is missing or empty.
    """
    if not comp_types:
        raise ValueError(
            "_find_instances needs comp_types; set solder_fillet component "
            "types from the pcba target config (component_types) or the "
            "editor's field. Matching nothing would report zero components "
            "on a board that has them."
        )
    frustum = _get_frustum(stage, camera_path)
    margin_scene = _mm_to_scene(stage, margin_mm)
    xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
    result = []
    for prim in stage.TraverseAll():
        path_str = str(prim.GetPath())
        if not any(ct in path_str for ct in comp_types):
            continue
        if pcba_root and not path_str.startswith(pcba_root):
            continue
        # Instance check: USD native instances (IsInstance()) OR
        # `tn__`-prefixed Xformable instance prims (newer assets that
        # use plain references instead of `instanceable=true`). Without
        # the fallback, a referenced-but-not-instanceable board reports 0
        # in-view instances even though component Xforms exist.
        name = prim.GetName()
        # also accept plain (non-instanceable) component Xforms directly under
        # a family scope matching comp_types -- some CAD conversions author
        # certain families without instanceable=true (e.g. PASSIVE_SMD_1210)
        parent_name = prim.GetParent().GetName() if prim.GetParent() else ""
        plain_component = prim.GetTypeName() == "Xform" and any(
            ct in parent_name for ct in comp_types
        )
        if not (
            prim.IsInstance()
            or (name.startswith("tn__") and UsdGeom.Xformable(prim))
            or plain_component
        ):
            continue
        world_mat = xfc.GetLocalToWorldTransform(prim)
        pos = world_mat.ExtractTranslation()
        if _in_frustum(frustum, pos, margin_scene=margin_scene):
            result.append((prim, world_mat))
    return result


# ── Component axis extraction ─────────────────────────────────────────────────


def _component_axes(world_mat: Gf.Matrix4d, bbox_range: Gf.Range3d):
    """
    Return (long_dir, width_dir, up_dir, half_len, half_width) in world space.
    Identifies long axis from world-space bbox (works for 0°/90° rotations).
    """
    # Local axes from world matrix (row-vector convention → rows = local axes)
    lx = Gf.Vec3d(world_mat[0][0], world_mat[0][1], world_mat[0][2])
    ly = Gf.Vec3d(world_mat[1][0], world_mat[1][1], world_mat[1][2])
    lz = Gf.Vec3d(world_mat[2][0], world_mat[2][1], world_mat[2][2])

    # Determine which world dimension is the long one from bbox extents
    size = bbox_range.GetSize()
    if size[0] >= size[1]:
        long_world = Gf.Vec3d(1, 0, 0)
    else:
        long_world = Gf.Vec3d(0, 1, 0)

    lxn = lx.GetNormalized()
    lyn = ly.GetNormalized()
    lzn = lz.GetNormalized()

    # World-axis frame: the local-rows heuristic assumed local Z == board
    # normal (authored AOI scenes). CAD imports permute local
    # axes (local Y = normal), which mapped the fillet width to world-up and
    # rendered fillets as vertical fins. The bbox already told us the long
    # world direction; derive width as the other horizontal axis and take
    # the board normal from whichever local axis points most vertically
    # (sign kept, so bottom-side components get downward fillets).
    long_dir = long_world
    width_dir = Gf.Vec3d(0, 1, 0) if long_world[0] == 1.0 else Gf.Vec3d(1, 0, 0)
    vertical = max((lxn, lyn, lzn), key=lambda v: abs(v[2]))
    up_dir = Gf.Vec3d(0, 0, 1.0 if vertical[2] >= 0 else -1.0)
    half_len = max(size[0], size[1]) / 2.0
    half_width = min(size[0], size[1]) / 2.0

    return long_dir, width_dir, up_dir, half_len, half_width


# ── Bbox (native-instance safe) ───────────────────────────────────────────────


def _compute_world_bbox(inst_prim, world_mat):
    """
    World-space GfRange3d for a native instance prim.

    BBoxCache.ComputeWorldBound on a native instance (IsInstance()==True) in
    USD 21.x returns the prototype's LOCAL bbox — coordinates near the prototype
    origin, not the instance's world position.

    Fix: compute the prototype's local bbox with ComputeLocalBound, then
    transform its 8 corners through the instance's world_mat.
    """
    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=True,
    )

    # For native instances use the prototype prim; otherwise use prim itself
    compute_target = inst_prim
    if inst_prim.IsInstance():
        proto = inst_prim.GetPrototype()
        if proto and proto.IsValid():
            compute_target = proto

    try:
        local_bbox = bc.ComputeLocalBound(compute_target)
        local_rng = local_bbox.GetRange()
    except Exception:  # noqa: BLE001
        local_rng = Gf.Range3d()

    if local_rng.IsEmpty():
        return Gf.Range3d()

    # Transform 8 corners of local bbox to world space via instance world_mat
    lmin, lmax = local_rng.GetMin(), local_rng.GetMax()
    world_pts = [
        world_mat.Transform(Gf.Vec3d(x, y, z))
        for x in (lmin[0], lmax[0])
        for y in (lmin[1], lmax[1])
        for z in (lmin[2], lmax[2])
    ]
    xs = [pt[0] for pt in world_pts]
    ys = [pt[1] for pt in world_pts]
    zs = [pt[2] for pt in world_pts]
    return Gf.Range3d(
        Gf.Vec3d(min(xs), min(ys), min(zs)),
        Gf.Vec3d(max(xs), max(ys), max(zs)),
    )


# ── Component-instance sidecar (pad-aware placement) ─────────────────────────
#
# The path/index/lookup conventions come from board_sidecar.py; the functions
# below are thin adapters.

# Lazy module-level cache: sidecar json path → {component prim path: entry}.
_SIDECAR_CACHE: "dict[str, dict]" = {}


def _board_sidecar_mod():
    """Import the shared sidecar helper (board_sidecar.py, same directory).

    In the pipeline context the editor dir is on sys.path; a bare Script-Editor
    paste without it degrades LOUDLY to the existing
    no-sidecar bbox fallback instead of crashing the whole editor.
    """
    import sys

    try:
        import board_sidecar

        return board_sidecar
    except ImportError:
        pass
    try:
        if "__file__" in globals():
            import importlib.util

            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "board_sidecar.py"
            )
            if os.path.isfile(path):
                spec = importlib.util.spec_from_file_location("board_sidecar", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules["board_sidecar"] = mod
                return mod
    except Exception as exc:  # noqa: BLE001
        print(f"[Fillet] board_sidecar load failed: {exc!r}")
    print(
        "[Fillet] board_sidecar.py not importable — sidecar lookups "
        "disabled (bbox fallback); add scripts/sdg/editor to sys.path"
    )
    return None


def _sidecar_path_for_stage(stage, p: FilletParams) -> str:
    """Resolve the component-instance sidecar path.

    Priority: explicit ``PARAMS.sidecar_path`` (set at runtime by the
    pipeline/harness) → ``<root_layer_dir>/<stem>.components.json`` next to
    the stage's root layer. Returns "" when neither exists (bbox fallback).
    """
    mod = _board_sidecar_mod()
    if mod is None:
        return ""
    try:
        real = stage.GetRootLayer().realPath or ""
    except Exception:  # noqa: BLE001 — anonymous/in-memory layers
        real = ""
    return mod.resolve_sidecar_path(str(getattr(p, "sidecar_path", "") or ""), (real,))


def _load_sidecar_index(stage, p: FilletParams) -> "dict | None":
    """Load (once) and return {component path: instance entry} or None."""
    path = _sidecar_path_for_stage(stage, p)
    if not path:
        return None
    idx = _SIDECAR_CACHE.get(path)
    if idx is None:
        mod = _board_sidecar_mod()
        idx = mod.load_component_index(path) if mod is not None else None
        if idx is None:
            print(f"[Fillet] sidecar load failed ({path})")
            idx = {}
        _SIDECAR_CACHE[path] = idx
    return idx or None


def _sidecar_lookup(index: dict, prim_path: str):
    """Find the sidecar entry for a composed-stage prim path (shared logic:
    ``board_sidecar.lookup`` — ancestor walk + /World prefix remap; see
    that module for the convention)."""
    mod = _board_sidecar_mod()
    if mod is None:
        return None, ""
    return mod.lookup(index, prim_path)


# ── Per-instance fillet placement ─────────────────────────────────────────────


def _range_bounds(rng):
    """Convert a USD range to transport-safe six-value bounds."""
    minimum, maximum = rng.GetMin(), rng.GetMax()
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(minimum[2]),
        float(maximum[0]),
        float(maximum[1]),
        float(maximum[2]),
    )


def _entry_world_points(stage, prim_paths, prefix, cache):
    """Return cached world-space mesh points below sidecar prim paths."""
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    chunks = []
    for path in dict.fromkeys(prim_paths):
        key = prefix + path
        if key not in cache:
            arrays = []
            prim = stage.GetPrimAtPath(key)
            if prim.IsValid():
                for mesh_prim in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                    if not mesh_prim.IsA(UsdGeom.Mesh):
                        continue
                    points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
                    if not points:
                        continue
                    matrix = np.asarray(
                        xform_cache.GetLocalToWorldTransform(mesh_prim),
                        dtype=float,
                    )
                    arrays.append(
                        np.asarray(points, dtype=float) @ matrix[:3, :3] + matrix[3, :3]
                    )
            cache[key] = np.vstack(arrays) if arrays else None
        if cache[key] is not None:
            chunks.extend(
                (float(point[0]), float(point[1]), float(point[2]))
                for point in cache[key]
            )
    return tuple(chunks)


def _place_fillets_pad_aware(
    stage, inst_prim, world_mat, idx, p: FilletParams, mat, entry, prefix
) -> bool:
    """Place one fillet per solder-paste pad using the sidecar's pad prims.

    Geometry per pad (all in world space, Z = board normal):
      * outward axis   — the horizontal axis/sign along which the pad
        sticks out furthest beyond the termination bbox (fallback:
        dominant offset of the pad centre from the component centre);
      * termination    — union bbox of ``body_prims`` AND
        ``solder_contact_prims``: on chip passives whose CAD splits the
        metallized end caps into separate lead prims (e.g. 0805, where
        each lead extends 0.5 mm beyond the ceramic), the body union
        alone puts the "termination face" at the ceramic and buries the
        whole fillet inside the lead;
      * footprint      — a meniscus wedge spanning the FULL exposed pad:
        from the termination face all the way to the pad's outer edge,
        where the tapered profile (see ``taper_to_zero``) feathers to
        exactly zero; short-protrusion pads simply get a correspondingly
        short fillet. A small wetting overlap (~11 % of the length) sinks
        the crest into the termination so it sits flush with no gap;
      * width          — the termination/body width, centred on the pad.
        Only parts with MORE than two pads (one body spanning all leads,
        e.g. resistor networks) clamp it to the pad width for per-lead
        sizing; two-pad molded packages (tantalum caps) keep the full
        body width — their lead frame and pads are much narrower than
        the slug, and the meniscus must be framed by the body block;
      * base plane     — the pad surface: pad-bbox top for Top-side parts,
        pad-bbox bottom for Bottom-side parts (fixes floating fillets on
        bottom-mounted components), meniscus growing away from the board;
      * height         — ~90 % of the termination height above the pad:
        the heightfield's crest (mesh x=0, the exp-decay maximum) sits at
        the termination face — sunk into it by the wetting overlap so the
        crest is flush against the face with no gap — and the surface
        falls concavely outward to a low foot feathering onto the pad,
        the classic solder wetting profile.

    The pad provides the base plane, the outward direction, and the foot's
    LENGTH (its exposed protrusion); the fillet's WIDTH and HEIGHT derive
    from the termination cuboid — paste pads are wider than the part on
    small passives (0201/0402/0805), so pad-wide fillets would render as
    oversized flat sheets.

    Returns True when at least one fillet was placed; False → caller falls
    back to the bbox-based path.
    """
    comp_rng = _compute_world_bbox(inst_prim, world_mat)
    if comp_rng.IsEmpty():
        return False

    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True
    )

    pads = []
    for pad_path in entry.get("solder_paste_prims", []):
        prim = stage.GetPrimAtPath(prefix + pad_path)
        if not prim.IsValid():
            continue
        rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
        if not rng.IsEmpty():
            pads.append(rng)
    if not pads:
        return False

    # Union bbox of the body AND lead (solder-contact) prims — the volume
    # the fillet must not enter beyond the wetting overlap, and the
    # reference for the termination face. The leads MUST be included:
    # some families (0805) author the metallized end caps as separate
    # pkg_lead prims protruding well beyond the pkg_body union, and a
    # body-only face would sink the fillet inside the lead.
    body_rng = Gf.Range3d()
    for body_path in entry.get("body_prims", []) + entry.get(
        "solder_contact_prims", []
    ):
        prim = stage.GetPrimAtPath(prefix + body_path)
        if prim.IsValid():
            body_rng.UnionWith(bc.ComputeWorldBound(prim).ComputeAlignedRange())
    if body_rng.IsEmpty():
        body_rng = comp_rng

    from pcb_aoi.solder_generation.procedural_meniscus_geometry import (
        MeniscusParameters,
        build_pad_meniscus,
        is_area_array,
        measure_lead_contact,
        resolve_pad_orientation,
    )

    scene_units_per_mm = _mm_to_scene(stage, 1.0)
    parameters = MeniscusParameters(
        long_edge=float(p.long_edge),
        x_scale=float(p.x_scale),
        platform_start=float(p.plat_start),
        platform_end=float(p.plat_end),
        smoothness_y=float(p.smooth_y),
        bump=float(p.bump),
        decay=float(p.decay),
        pad_decay=float(getattr(p, "pad_decay", p.decay)),
        noise_scale=float(p.noise_scale),
        noise_octaves=int(p.noise_octaves),
        noise_amplitude=float(p.noise_amp),
        resolution=max(4, int(p.resolution)),
        solid=bool(getattr(p, "solid", False)),
        solid_depth_fraction=float(getattr(p, "solid_depth_frac", 0.15)),
        termination_height_ratio=0.9,
        wetting_overlap_ratio=0.12,
        z_offset=_mm_to_scene(stage, p.z_offset_mm),
        lead_local_sizing=bool(getattr(p, "lead_local_sizing", False)),
        scene_units_per_mm=scene_units_per_mm,
    )
    pad_bounds = tuple(_range_bounds(pad) for pad in pads)
    if parameters.lead_local_sizing and is_area_array(pad_bounds):
        print(
            f"  [Fillet] {inst_prim.GetPath().name}  → 0 fillets "
            f"(area-array package, {len(pads)} pads)"
        )
        return True
    body_bounds = _range_bounds(body_rng)
    component_bounds = _range_bounds(comp_rng)
    side = str(entry.get("side", "Top"))
    lead_points = ()
    if parameters.lead_local_sizing:
        point_paths = tuple(
            list(entry.get("solder_contact_prims") or [])
            + list(entry.get("body_prims") or [])
        )
        lead_points = _entry_world_points(stage, point_paths, prefix, {})

    placed = 0
    for k, bounds in enumerate(pad_bounds):
        contact = None
        if parameters.lead_local_sizing:
            _, axis, sign = resolve_pad_orientation(
                bounds,
                body_bounds,
                component_bounds,
            )
            contact = measure_lead_contact(
                bounds,
                lead_points,
                side,
                axis,
                sign,
                0.5 * scene_units_per_mm,
                2.5 * scene_units_per_mm,
            )
        generated = build_pad_meniscus(
            bounds,
            body_bounds,
            component_bounds,
            side,
            len(pads),
            parameters,
            contact,
        )
        if generated is None:
            continue
        path = f"{FILLET_ROOT}/fillet_{idx:04d}_P{k}"
        _write_mesh(
            stage,
            path,
            p,
            mat,
            taper_to_zero=True,
            decay_override=getattr(p, "pad_decay", None),
        )
        _set_transform(stage, path, Gf.Matrix4d(*generated.transform))
        placed += 1

    if placed:
        print(
            f"  [Fillet] {inst_prim.GetPath().name}  → {placed} pad-aware fillet(s) at idx {idx}"
        )
    elif parameters.lead_local_sizing:
        print(
            f"  [Fillet] {inst_prim.GetPath().name}  → 0 fillets "
            f"(lead-local: all {len(pads)} pads are covered or have no contact)"
        )
    return placed > 0 or parameters.lead_local_sizing


def _place_fillets(stage, inst_prim, world_mat, idx, p: FilletParams, mat):
    if getattr(p, "pad_aware", False):
        index = _load_sidecar_index(stage, p)
        if index:
            entry, prefix = _sidecar_lookup(index, str(inst_prim.GetPath()))
            if entry is not None:
                if _place_fillets_pad_aware(
                    stage, inst_prim, world_mat, idx, p, mat, entry, prefix
                ):
                    return
                # Silent fallbacks hid real coverage gaps (e.g. multi-pad
                # parts whose paste prims don't resolve on the composed
                # stage) — name the part and the pad count.
                refdes = entry.get("refdes") or inst_prim.GetPath().name
                n_paste = len(entry.get("solder_paste_prims") or [])
                print(
                    f"  [Fillet] WARNING: {refdes}: pad-aware placement "
                    f"produced no fillets ({n_paste} paste prim(s) in the "
                    f"sidecar) — falling back to bbox placement"
                )
    _place_fillets_bbox(stage, inst_prim, world_mat, idx, p, mat)


def _place_fillets_bbox(stage, inst_prim, world_mat, idx, p: FilletParams, mat):
    rng = _compute_world_bbox(inst_prim, world_mat)
    if rng.IsEmpty():
        print(f"  [Fillet] empty bbox for {inst_prim.GetPath()} — skipping")
        return

    min_pt = Gf.Vec3d(rng.GetMin())
    max_pt = Gf.Vec3d(rng.GetMax())
    center = (min_pt + max_pt) * 0.5

    long_dir, width_dir, up_dir, half_len, half_width = _component_axes(world_mat, rng)

    # Fillet scale: adaptive (fit to component bbox) or fixed from PARAMS
    if getattr(p, "adaptive_scale", False):
        mesh_width = float(p.long_edge) * 0.6
        sy = (half_width * 2.0) / mesh_width if mesh_width > 0 else p.fillet_sy
        sx = sy  # keep X/Y scale equal (square pad footprint)
        # adaptive_sz: keep the design aspect ratio so small-scene boards
        # (e.g. mm-unit CAD imports) get proportionate fillet height
        # instead of the absolute sz
        if getattr(p, "adaptive_sz", False) and p.fillet_sy:
            sz = (
                sy
                * (p.fillet_sz / p.fillet_sy)
                * float(getattr(p, "adaptive_sz_factor", 1.0))
            )
        else:
            sz = p.fillet_sz
    else:
        sx, sy, sz = p.fillet_sx, p.fillet_sy, p.fillet_sz
    y_world = float(p.long_edge) * 0.6 * sy  # scaled width of fillet

    # Center fillet across component width
    width_offset = -y_world / 2.0

    # Base Z: bottom of bbox (PCB surface level)
    base_z = rng.GetMin()[2]

    for side, sign in (("L", -1.0), ("R", +1.0)):
        path = f"{FILLET_ROOT}/fillet_{idx:04d}_{side}"
        _write_mesh(stage, path, p, mat)

        if getattr(p, "adaptive_scale", False):
            inward = half_len * float(getattr(p, "inward_frac", 0.15))
        else:
            inward = _mm_to_scene(stage, p.inward_offset_mm)
        inner_edge = center + long_dir * (sign * (half_len - inward))

        origin_xy = inner_edge + width_dir * width_offset
        z_off = _mm_to_scene(stage, p.z_offset_mm)
        origin = Gf.Vec3d(origin_xy[0], origin_xy[1], base_z + z_off)

        # For right (+1): fillet x axis points rightward (outward from body)
        # For left  (-1): fillet x axis points leftward (outward from body)
        fx = long_dir * sign  # outward direction
        fy = width_dir
        fz = up_dir

        mat4 = _fillet_world_matrix(origin, fx, fy, fz, sx, sy, sz)
        _set_transform(stage, path, mat4)

    print(f"  [Fillet] {inst_prim.GetPath().name}  → L+R placed at idx {idx}")


# ── Top-level actions ─────────────────────────────────────────────────────────


def generate_fillets():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[Fillet] No stage.")
        return

    if not stage.GetPrimAtPath(FILLET_ROOT).IsValid():
        UsdGeom.Xform.Define(stage, FILLET_ROOT)

    mat = _ensure_material(stage, PARAMS)
    comp_types = list(getattr(PARAMS, "comp_types", None) or [])
    if not comp_types:
        print(
            "[Fillet] No component types set. Fill the editor's component-type "
            "field, or pass component_types in the pcba target config."
        )
        return
    instances = _find_instances(
        stage, PARAMS.camera_path, PARAMS.pcba_root, comp_types=comp_types
    )
    type_label = f"{len(comp_types)} types" if len(comp_types) > 1 else comp_types[0]
    print(
        f"[Fillet] {len(instances)} {type_label} instances in view → generating {len(instances) * 2} fillets…"
    )

    for idx, (prim, wmat) in enumerate(instances):
        _place_fillets(stage, prim, wmat, idx, PARAMS, mat)

    print("[Fillet] Done.")


def clear_fillets():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    root = stage.GetPrimAtPath(FILLET_ROOT)
    if root.IsValid():
        stage.RemovePrim(FILLET_ROOT)
        print(f"[Fillet] Cleared {FILLET_ROOT}")
    mat = stage.GetPrimAtPath(MAT_PATH)
    if mat.IsValid():
        stage.RemovePrim(MAT_PATH)
        _shader_prim[0] = None


def _get_viewport_camera_path():
    """Return the USD prim path of the active viewport camera, or None."""
    try:
        import omni.kit.viewport.utility as vp_util

        vp = vp_util.get_active_viewport()
        if vp is not None:
            return str(vp.camera_path)
    except Exception:  # noqa: BLE001, S110
        pass
    return None


def list_inview_components():
    """Print the configured component types visible in the viewport camera."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[Fillet] No stage.")
        return

    cam_path = _get_viewport_camera_path()
    if cam_path:
        print(f"[Fillet] Viewport camera: {cam_path}")
        margin_mm = 1.0
    else:
        cam_path = PARAMS.camera_path
        print(f"[Fillet] (viewport cam unavailable, falling back to: {cam_path})")
        margin_mm = 5.0

    comp_types = list(getattr(PARAMS, "comp_types", None) or [])
    if not comp_types:
        print("[Fillet] No component types set; nothing to list.")
        return
    instances = _find_instances(
        stage, cam_path, PARAMS.pcba_root, margin_mm=margin_mm, comp_types=comp_types
    )
    label = ", ".join(comp_types)
    print(f"[Fillet] {len(instances)} {label} instances in view:")
    for prim, _ in instances:
        print(f"  {prim.GetPath()}")


# ── UI ────────────────────────────────────────────────────────────────────────

_auto_regen = [False]  # mutable so closures can read it


def _auto_update_shape():
    """Compute geometry once, push to every existing fillet mesh — O(N) writes, 0 stage traversals."""
    stage = omni.usd.get_context().get_stage()
    if not stage:
        return
    root = stage.GetPrimAtPath(FILLET_ROOT)
    if not root.IsValid():
        return
    pts, counts, idx = _compute_geometry(PARAMS)
    for child in root.GetChildren():
        if child.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(child)
            m.GetPointsAttr().Set(pts)
            m.GetFaceVertexCountsAttr().Set(counts)
            m.GetFaceVertexIndicesAttr().Set(idx)


def _auto_update_material():
    stage = omni.usd.get_context().get_stage()
    if stage and stage.GetPrimAtPath(MAT_PATH).IsValid():
        _ensure_material(stage, PARAMS)


def _auto_update_full():
    """Position/scale changed — need to recompute transforms, so full regeneration."""
    stage = omni.usd.get_context().get_stage()
    if stage and stage.GetPrimAtPath(FILLET_ROOT).IsValid():
        generate_fillets()


def _slider(label, key, lo, hi, default, fmt="{:.3f}", regen="shape"):
    """regen: 'shape' | 'material' | 'full'"""
    lbl_ref = [None]
    with ui.HStack(height=20, spacing=4):
        ui.Label(label, width=120)
        s = ui.FloatSlider(min=lo, max=hi)
        s.model.set_value(float(default))
        lbl_ref[0] = ui.Label(fmt.format(default), width=48)

    def _on(_=None):
        v = s.model.get_value_as_float()
        setattr(PARAMS, key, v)
        lbl_ref[0].text = fmt.format(v)
        if _auto_regen[0]:
            if regen == "shape":
                _auto_update_shape()
            elif regen == "material":
                _auto_update_material()
            else:
                _auto_update_full()

    s.model.add_value_changed_fn(_on)
    return s


def _section(title, collapsed=False):
    return ui.CollapsableFrame(
        title, collapsed=collapsed, style={"font_size": 13, "margin": 0}
    )


def build_ui():
    win = ui.Window("Solder Fillet Placer", width=440, height=520)
    win.visible = True

    with win.frame:  # noqa: SIM117
        with ui.ScrollingFrame(
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
        ):
            with ui.VStack(spacing=2, style={"margin": 4}):
                with _section("Camera / PCBA", collapsed=True):  # noqa: SIM117
                    with ui.VStack(spacing=2, style={"margin": 4}):
                        with ui.HStack(height=20, spacing=4):
                            ui.Label("Camera", width=70)
                            cam_f = ui.StringField()
                            cam_f.model.set_value(PARAMS.camera_path)
                            cam_f.model.add_value_changed_fn(
                                lambda m: setattr(
                                    PARAMS, "camera_path", m.get_value_as_string()
                                )
                            )
                        with ui.HStack(height=20, spacing=4):
                            ui.Label("PCBA root", width=70)
                            root_f = ui.StringField()
                            root_f.model.set_value(PARAMS.pcba_root)
                            root_f.model.add_value_changed_fn(
                                lambda m: setattr(
                                    PARAMS, "pcba_root", m.get_value_as_string()
                                )
                            )

                with _section("Shape"):  # noqa: SIM117
                    with ui.VStack(spacing=2, style={"margin": 4}):
                        _slider(
                            "Long edge", "long_edge", 2, 20, PARAMS.long_edge, "{:.1f}"
                        )
                        _slider("X scale", "x_scale", 0.05, 0.8, PARAMS.x_scale)
                        _slider("Plat start", "plat_start", 0.0, 1.0, PARAMS.plat_start)
                        _slider("Plat end", "plat_end", 0.0, 1.0, PARAMS.plat_end)
                        _slider(
                            "Smooth Y", "smooth_y", 1.0, 30.0, PARAMS.smooth_y, "{:.1f}"
                        )
                        _slider("Bump", "bump", 0.0, 1.0, PARAMS.bump)
                        _slider("Decay", "decay", 1.0, 40.0, PARAMS.decay, "{:.1f}")

                with _section("Noise"):  # noqa: SIM117
                    with ui.VStack(spacing=2, style={"margin": 4}):
                        _slider("Scale", "noise_scale", 0.0, 1.5, PARAMS.noise_scale)
                        _slider(
                            "Octaves",
                            "noise_octaves",
                            1,
                            8,
                            PARAMS.noise_octaves,
                            "{:.0f}",
                        )
                        _slider(
                            "Amplitude",
                            "noise_amp",
                            0.0,
                            0.05,
                            PARAMS.noise_amp,
                            "{:.4f}",
                        )
                        _slider(
                            "Resolution",
                            "resolution",
                            10,
                            80,
                            PARAMS.resolution,
                            "{:.0f}",
                        )

                with _section("Scale & Position"):  # noqa: SIM117
                    with ui.VStack(spacing=2, style={"margin": 4}):
                        _slider(
                            "Scale X/Y",
                            "fillet_sx",
                            0.01,
                            0.5,
                            PARAMS.fillet_sx,
                            "{:.5f}",
                            regen="full",
                        )
                        _slider(
                            "Scale Z",
                            "fillet_sz",
                            0.01,
                            1.0,
                            PARAMS.fillet_sz,
                            "{:.4f}",
                            regen="full",
                        )
                        _slider(
                            "Inward (mm)",
                            "inward_offset_mm",
                            -3.0,
                            6.0,
                            PARAMS.inward_offset_mm,
                            "{:.3f}",
                            regen="full",
                        )
                        _slider(
                            "Z offset (mm)",
                            "z_offset_mm",
                            -2.0,
                            15.0,
                            PARAMS.z_offset_mm,
                            "{:.3f}",
                            regen="full",
                        )

                with _section("Material", collapsed=True):  # noqa: SIM117
                    with ui.VStack(spacing=2, style={"margin": 4}):
                        _slider(
                            "Metallic",
                            "mat_metallic",
                            0.0,
                            1.0,
                            PARAMS.mat_metallic,
                            regen="material",
                        )
                        _slider(
                            "Roughness",
                            "mat_roughness",
                            0.0,
                            1.0,
                            PARAMS.mat_roughness,
                            regen="material",
                        )

                ui.Spacer(height=4)
                with ui.HStack(height=28, spacing=4):
                    ui.Button(
                        "Generate", clicked_fn=generate_fillets, width=ui.Fraction(3)
                    )
                    ui.Button("Clear", clicked_fn=clear_fillets, width=ui.Fraction(1))
                with ui.HStack(height=28, spacing=4):
                    ui.Button(
                        "List In-View",
                        clicked_fn=list_inview_components,
                        width=ui.Fraction(1),
                    )
                ui.Spacer(height=2)
                with ui.HStack(height=22, spacing=6):
                    cb = ui.CheckBox(width=18)
                    cb.model.set_value(False)
                    ui.Label("Auto-update on slider change")

                    def _on_auto(m):
                        _auto_regen[0] = m.get_value_as_bool()

                    cb.model.add_value_changed_fn(_on_auto)

    return win


# Keep Scale Y in sync with Scale X (fillet_sx slider sets both)
def _patched_setattr(self, key, val):
    object.__setattr__(self, key, val)
    if key == "fillet_sx":
        object.__setattr__(self, "fillet_sy", val)


FilletParams.__setattr__ = _patched_setattr


# ── Execute ───────────────────────────────────────────────────────────────────

if not os.getenv("FILLET_PIPELINE_MODE"):
    try:
        _fillet_win = build_ui()
        globals()["_fillet_win"] = _fillet_win
        print(
            "[Fillet] UI ready. Set the board root and component types, then "
            "press 'Generate Fillets' to place fillets for in-view components."
        )
    except Exception as _e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
