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

"""Procedural component body texture maps (capability:
``appearance.body_texture``).

Pure numpy — importable without Kit/omni, so the generator math is unit
tested on CI (``tests/unittests/core/editor/test_body_texture.py``); only
``write_png``/``generate_maps`` need PIL.

Noise pedigree: this is hash-permutation Perlin GRADIENT noise composed
into fBm with per-octave random rotation + offset and jittered lacunarity
(1.9–2.1). It is deliberately NOT the repo's sin/cos value noise
(``_fractal_noise_2d`` in editor_solder_fillet_board.py /
tin_noise_patch.py): at body-texture scale that one reads plaid/repetitive,
while this one was visually approved by the user. Do NOT swap either
implementation for the other — the production fillet/tin noise must stay
byte-identical.

Per body the op builds two 1024² maps:

* a NORMAL map — fBm heightfield relief (``relief_strength``);
* a layered ALBEDO map over the body's display colour: dark blotches
  (smoothstep of a coarse fBm field), dark contour rings at 2–3 random
  level-sets of that field (gaussian bands), and TWO off-colour tint
  layers — one LIGHT palette (off-white / yellow / orange, pops by
  brightness under the red-dominant inner ring) and one COOL palette
  (teal / blue / green, pops by hue under the green/blue rings).

All layer strengths/coverages accept a scalar or a ``[lo, hi]`` range
sampled once from the seeded rng (defaults are the approved reference
values).
"""

import numpy as np

# Off-colour tint palettes (linear RGB). One entry of each palette is
# picked per albedo map.
TINTS_LIGHT = (
    (0.80, 0.75, 0.65),  # off-white
    (0.78, 0.62, 0.15),  # bright yellow
    (0.85, 0.45, 0.20),  # bright orange
)
TINTS_COOL = (
    (0.12, 0.55, 0.50),  # teal
    (0.22, 0.38, 0.75),  # bright blue
    (0.15, 0.60, 0.22),  # green
)

DEFAULT_BASE_COLOR = (0.35, 0.2, 0.15)


def _sample(rng, value):
    """Scalar passthrough, or one uniform draw from a [lo, hi] range."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(rng.uniform(float(value[0]), float(value[1])))
    return float(value)


def perlin2d(x, y, perm):
    """Hash-permutation gradient (Perlin) noise at coordinates x, y.

    ``perm`` is a permutation of range(256); gradients are unit vectors at
    angles hashed from the lattice corners. Zero at integer lattice
    points, range roughly [-0.7, 0.7].
    """
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi

    def grad(ix, iy):
        h = perm[(perm[ix & 255] + iy) & 255]
        ang = h * (2.0 * np.pi / 256.0)
        return np.cos(ang), np.sin(ang)

    gx00, gy00 = grad(xi, yi)
    gx10, gy10 = grad(xi + 1, yi)
    gx01, gy01 = grad(xi, yi + 1)
    gx11, gy11 = grad(xi + 1, yi + 1)
    n00 = gx00 * xf + gy00 * yf
    n10 = gx10 * (xf - 1) + gy10 * yf
    n01 = gx01 * xf + gy01 * (yf - 1)
    n11 = gx11 * (xf - 1) + gy11 * (yf - 1)
    u = xf**3 * (xf * (xf * 6 - 15) + 10)  # quintic fade
    v = yf**3 * (yf * (yf * 6 - 15) + 10)
    return (n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v


def fbm(rng, freq_base, octaves, size, persistence=0.55):
    """fBm over ``perlin2d``, normalized to [0, 1], shape (size, size).

    Each octave gets a random rotation + offset and a JITTERED lacunarity
    (uniform 1.9–2.1) — this is what kills the axis-aligned/plaid reading
    of naive octave stacking.
    """
    perm = rng.permutation(256).astype(np.int64)
    ax = np.linspace(0.0, 1.0, size)
    X, Y = np.meshgrid(ax, ax)
    h = np.zeros((size, size))
    amp, freq = 1.0, float(freq_base)
    for _o in range(int(octaves)):
        th = rng.uniform(0, 2 * np.pi)
        ox, oy = rng.uniform(0, 1000, 2)
        xr = (X * np.cos(th) - Y * np.sin(th)) * freq + ox
        yr = (X * np.sin(th) + Y * np.cos(th)) * freq + oy
        h += amp * perlin2d(xr, yr, perm)
        amp *= persistence
        freq *= rng.uniform(1.9, 2.1)
    h -= h.min()
    h /= max(h.max(), 1e-9)
    return h


def smoothstep(a, lo, hi):
    """Hermite smoothstep of array ``a`` between edges lo and hi."""
    t = np.clip((a - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def normal_map(rng, *, size=1024, freq=4.0, octaves=6, relief_strength=0.75):
    """Tangent-space normal map from an fBm heightfield.

    Returns float array (size, size, 3) in [0, 1] (0.5-biased encoding;
    decode with *2-1). ``relief_strength`` scales the height gradient.
    """
    h = fbm(rng, _sample(rng, freq), int(octaves), int(size))
    hc = h * 2.0 - 1.0
    gy, gx = np.gradient(hc)
    strength = _sample(rng, relief_strength)
    nx = -gx * strength * size / 8.0
    ny = -gy * strength * size / 8.0
    nz = np.ones_like(hc)
    ln = np.sqrt(nx**2 + ny**2 + nz**2)
    return np.dstack((nx / ln * 0.5 + 0.5, ny / ln * 0.5 + 0.5, nz / ln * 0.5 + 0.5))


def albedo_map(
    rng,
    base_color,
    *,
    size=1024,
    blotch_freq=(2.0, 3.5),
    blotch_lo=0.52,
    blotch_hi=0.72,
    blotch_strength=(0.65, 0.85),
    ring_count=(2, 3),
    ring_level=(0.30, 0.62),
    ring_width=(0.012, 0.028),
    ring_strength=(0.55, 0.75),
    tint_light_freq=(3.0, 5.5),
    tint_light_lo=0.58,
    tint_light_hi=0.74,
    tint_cool_freq=(2.5, 4.5),
    tint_cool_lo=0.60,
    tint_cool_hi=0.76,
    tint_blend=(0.9, 1.0),
    tints_light=TINTS_LIGHT,
    tints_cool=TINTS_COOL,
):
    """Layered albedo over ``base_color``; float (size, size, 3) in [0, 1].

    Layers (defaults = the approved reference values): dark blotches, dark
    contour rings at random level-sets of the blotch field, one LIGHT-tint
    layer, one COOL-tint layer.
    """
    size = int(size)
    base = np.asarray(base_color, dtype=float)
    img = np.tile(base, (size, size, 1))
    hb = fbm(rng, _sample(rng, blotch_freq), 4, size)  # blotch field
    ht = fbm(rng, _sample(rng, tint_light_freq), 3, size)  # light-tint field
    # dark blotches (top of the field, soft edged)
    mdark = smoothstep(hb, blotch_lo, blotch_hi) * _sample(rng, blotch_strength)
    img *= (1.0 - mdark)[:, :, None]
    # dark contour RINGS: thin gaussian bands where the blotch field
    # crosses random levels
    ring = np.zeros((size, size))
    n_rings = int(
        rng.integers(int(ring_count[0]), int(ring_count[1]) + 1)
        if isinstance(ring_count, (list, tuple))
        else int(ring_count)
    )
    for _r in range(n_rings):
        lev = _sample(rng, ring_level)
        wid = _sample(rng, ring_width)
        ring += np.exp(-(((hb - lev) / wid) ** 2))
    img *= (1.0 - np.clip(ring, 0, 1) * _sample(rng, ring_strength))[:, :, None]
    # off-colour tint patches: one LIGHT layer (pops by brightness under
    # the red-dominant inner ring) + one COOL layer (pops by hue via the
    # green/blue rings)
    tint_l = np.asarray(
        tints_light[int(rng.integers(0, len(tints_light)))], dtype=float
    )
    mt_l = smoothstep(ht, tint_light_lo, tint_light_hi) * _sample(rng, tint_blend)
    img = img * (1.0 - mt_l[:, :, None]) + tint_l[None, None, :] * mt_l[:, :, None]
    hc2 = fbm(rng, _sample(rng, tint_cool_freq), 3, size)  # cool-tint field
    tint_c = np.asarray(tints_cool[int(rng.integers(0, len(tints_cool)))], dtype=float)
    mt_c = smoothstep(hc2, tint_cool_lo, tint_cool_hi) * _sample(rng, tint_blend)
    img = img * (1.0 - mt_c[:, :, None]) + tint_c[None, None, :] * mt_c[:, :, None]
    return np.clip(img, 0.0, 1.0)


def box_project_uvs(points, face_vertex_indices, face_vertex_counts):
    """faceVarying box-projection UVs for a mesh, one GLOBAL scale.

    Per face, the dominant axis of the geometric normal picks which two
    point coordinates become (u, v); every face divides by the same
    global scale (the mesh's largest bbox extent) so texel density is
    uniform across faces. This is the projection that survives CAD side
    walls — PLANAR vertex UVs degenerate to a 1-D gradient there (CAD
    local Y is the board normal, so a wall's planar footprint is a line).

    Returns float array (sum(face_vertex_counts), 2) in [0, 1].
    """
    pts = np.asarray(points, dtype=float)
    idx = np.asarray(face_vertex_indices, dtype=np.int64)
    cnt = np.asarray(face_vertex_counts, dtype=np.int64)
    pmin = pts.min(axis=0)
    scale = float((pts.max(axis=0) - pmin).max())
    scale = max(scale, 1e-9)
    uvs = np.empty((int(cnt.sum()), 2), dtype=float)
    off = 0
    for n in cnt:
        face = idx[off : off + n]
        p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
        nrm = np.cross(p1 - p0, p2 - p0)
        dom = int(np.argmax(np.abs(nrm)))
        a1, a2 = [a for a in (0, 1, 2) if a != dom]
        uvs[off : off + n, 0] = (pts[face, a1] - pmin[a1]) / scale
        uvs[off : off + n, 1] = (pts[face, a2] - pmin[a2]) / scale
        off += n
    return uvs


def write_png(path, array):
    """Write a float [0, 1] HxWx3 array as an 8-bit PNG (needs PIL)."""
    from PIL import Image

    Image.fromarray((np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)).save(str(path))
    return str(path)


def generate_maps(
    out_dir,
    tag,
    seed,
    base_color,
    *,
    size=1024,
    normal_freq=4.0,
    normal_octaves=6,
    relief_strength=0.75,
    **albedo_params,
):
    """Generate + write the normal/albedo PNG pair for one body.

    One rng seeded from ``seed`` flows through normal THEN albedo (same
    order as the approved reference), so a seed fully determines both
    maps. Returns ``(normal_path, albedo_path)``.
    """
    import os

    rng = np.random.default_rng(seed)
    os.makedirs(str(out_dir), exist_ok=True)
    nrm = normal_map(
        rng,
        size=size,
        freq=normal_freq,
        octaves=normal_octaves,
        relief_strength=relief_strength,
    )
    alb = albedo_map(rng, base_color, size=size, **albedo_params)
    normal_path = write_png(f"{out_dir}/{tag}_n.png", nrm)
    albedo_path = write_png(f"{out_dir}/{tag}_c.png", alb)
    return normal_path, albedo_path
