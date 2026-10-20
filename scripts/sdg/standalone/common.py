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

"""Shared utilities for the unified SDG pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

try:
    from omni.replicator.core.functional import modify as rep_modify
except ImportError:  # Kit --exec init-timing: lazy sys.modules alias not yet set
    try:
        from omni.replicator.core.scripts.functional import modify as rep_modify
    except ImportError:

        class _ReplicatorModifyCompatibility:
            """Adapt the public Replicator modifier to the functional API."""

            @staticmethod
            def semantics(prim: object, *, value: dict[str, object], mode="replace"):
                """Apply semantic labels through the public Replicator API."""
                return rep.modify.semantics(
                    list(value.items()),
                    input_prims=[str(prim.GetPath())],
                    mode=mode,
                )

        rep_modify = _ReplicatorModifyCompatibility()

from pcba_target import DEFAULT_COMPONENT_CLASS, classify_components

from cdf.execution.isaac.guard_band import render_aperture
from pcb_aoi.lighting import (
    lighting_mode,
)
from pcb_aoi.geometry.fov import check_square_pixels, mm_to_su
from pcb_aoi.geometry.fov import scan_region_su as fov_scan_region_su
from pcb_aoi.geometry.scan_grid import (
    auto_complete_grid_nums,
    compute_scan_grid_geometry,
)
from pcb_aoi.modus_light import (
    DEFAULT_ARC_RADIUS_SCALE,
    DEFAULT_COLORS,
    DEFAULT_ELEVATIONS,
    DEFAULT_HEIGHT_SCALE,
    DEFAULT_INTENSITIES,
    DEFAULT_WIDTH_FRAC,
    compute_modus_layout,
    modus_bounds_from_prims,
)

logger = logging.getLogger(__name__)


def set_writer_start_frame_index(writer: object, frame_index: int) -> None:
    """BasicWriter names outputs rgb_{frame_index:0padding}; align with config seed + global offset.

    On the first ``write()``, BasicWriter compares ``sequence_id`` (often ``""`` when there is no
    ``on_time`` trigger) to ``_sequence_id``. The writer starts with ``_sequence_id == 0``, so
    ``"" != 0`` is true and it **resets** ``_frame_id`` to 0, wiping a custom start index. Match
    the empty-string sequence so the first write keeps ``_frame_id``.
    """
    writer._frame_id = int(frame_index)
    if getattr(writer, "_sequence_id", None) == 0:
        writer._sequence_id = ""


def seed_trigger_numpy(random_seed: int, trigger_idx: int) -> None:
    """NumPy RNG at start of each trigger (lighting then augmentation); +1 skips defect-prep stream."""
    np.random.seed(int(random_seed) + 1 + int(trigger_idx))


def set_replicator_seed_for_output_frame(image_number: int) -> None:
    """Seed Replicator per written frame (IRO-style seed + index).

    Pins what Replicator itself draws -- in this pipeline that is the camera
    rotation, the one ``rep.distribution`` call we make. It does NOT pin
    path-traced sampling, despite what this docstring used to claim. Measured
    over four runs of one commit at fixed ``seed`` / ``random_seed``: the
    bounding boxes, segmentation, labels and prim paths come back byte
    identical -- so the scene state really is fully determined -- while the RGB
    differs run to run by ~1.66 mean absolute, over about half the pixels. The
    residual lives in the renderer's own accumulation, below anything seeded
    here.

    Consequence for anyone diffing two runs: compare the annotations, and treat
    RGB statistically against a same-commit control. A bare RGB diff between two
    branches reads as a regression when nothing changed.

    Note also that this does not advance ``numpy.random`` -- see the per-sample
    ``np.random.seed`` in ``sdg_pipeline`` and :func:`seed_trigger_numpy`. The
    two RNG domains are separate and both have to be seeded.

    Args:
        image_number: Global index of the frame being written.

    """
    rep.set_global_seed(int(image_number))


def build_scan_positions(grid: dict[str, Any]) -> list[tuple[float, float, float]]:
    """Build the (x, y, z) camera positions for one scan pass.

    Two input shapes are supported:

    * Auto/explicit list — ``x_centers`` and ``y_centers`` lists (e.g. produced
      by :func:`resolve_scan_grid_from_stage`). Iterated y-outer, x-inner.
    * Older descending loop — ``x_start``, ``x_end``, ``y_start``, ``y_end``
      with either separate ``x_step`` / ``y_step`` or a single ``step``.
    """
    z = float(grid["z"])
    if "x_centers" in grid and "y_centers" in grid:
        positions: list[tuple[float, float, float]] = []
        for y in grid["y_centers"]:
            for x in grid["x_centers"]:
                positions.append((float(x), float(y), z))
        return positions

    x_step = float(grid.get("x_step", grid.get("step")))
    y_step = float(grid.get("y_step", grid.get("step")))
    x_eps = 1e-6 * max(abs(x_step), 1.0)
    y_eps = 1e-6 * max(abs(y_step), 1.0)
    positions = []
    y = float(grid["y_start"])
    while y >= float(grid["y_end"]) - y_eps:
        x = float(grid["x_start"])
        while x >= float(grid["x_end"]) - x_eps:
            positions.append((x, y, z))
            if x_step <= 0:
                break
            x -= x_step
        if y_step <= 0:
            break
        y -= y_step
    return positions


def resolve_scan_grid_from_stage(stage: object, cfg: dict[str, Any]) -> dict[str, Any]:
    """If ``cfg["scan_grid"]`` uses ``x_num`` / ``y_num``, fill in the rest.

    Reads:
      * the camera at ``cfg["camera_path"]`` (projection / hap / vap / fl —
        all stored in tenths of scene units, divided by 10 here);
      * the world-space AABB of ``cfg["pcba_root"]``.

    Returns a dict (also assigned back into ``cfg["scan_grid"]``) suitable for
    :func:`build_scan_positions`.

    No-op when the config is in older form (``x_start`` / ``x_end`` / …).
    """
    sg = cfg.get("scan_grid", {})
    if "x_num" not in sg and "y_num" not in sg:
        return sg
    # Either (or both) of x_num / y_num may be provided. The missing dim is
    # auto-filled below from bbox + apertures so cells are aspect-matched.
    x_num_in = int(sg["x_num"]) if "x_num" in sg and sg["x_num"] is not None else None
    y_num_in = int(sg["y_num"]) if "y_num" in sg and sg["y_num"] is not None else None

    cam_prim = stage.GetPrimAtPath(cfg["camera_path"])
    if not cam_prim.IsValid():
        raise RuntimeError(f"Camera not found at {cfg['camera_path']}")
    cam = UsdGeom.Camera(cam_prim)
    proj_attr = cam.GetProjectionAttr().Get() or "perspective"
    proj = "orthographic" if proj_attr == "orthographic" else "perspective"

    hap_usd = float(cam.GetHorizontalApertureAttr().Get() or 0.0)
    vap_usd = float(cam.GetVerticalApertureAttr().Get() or 0.0)
    fl_usd_raw = cam.GetFocalLengthAttr().Get()
    fl_usd = float(fl_usd_raw) if fl_usd_raw is not None else None

    # USD stores hap / vap / fl in tenths of scene units (film-format compat).
    hap_su = hap_usd / 10.0
    vap_su = vap_usd / 10.0
    fl_su = fl_usd / 10.0 if fl_usd is not None else None

    # Square-pixel correction: derive vap from render resolution × hap.
    res = cfg.get("resolution")
    aspect_h_over_w = None
    if res and len(res) == 2 and float(res[0]) > 0:
        aspect_h_over_w = float(res[1]) / float(res[0])
        vap_su = hap_su * aspect_h_over_w

    # Camera Z yaw maps image axes onto world axes. For ±90°/±270° the camera's
    # horizontal aperture covers world Y (not X) and vertical covers world X.
    # The grid in this module stays world-axis-aligned, so we need to know
    # which aperture binds which world axis when sizing the orthographic
    # auto-fit and when calling compute_scan_grid_geometry.
    cam_rot = cfg.get("camera_rotation") or {}
    cam_z_rot_deg = float(cam_rot.get("z_fixed", 0.0))
    swap_aperture_xy = round(cam_z_rot_deg / 90.0) % 2 == 1

    pcba_root = cfg.get("pcba_root")
    if not pcba_root:
        raise RuntimeError("scan_grid auto mode requires pcba_root in config")
    pcba_prim = stage.GetPrimAtPath(pcba_root)
    if not pcba_prim.IsValid():
        raise RuntimeError(f"PCBA root not found: {pcba_root}")

    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=True,
    )
    world_bbox = bc.ComputeWorldBound(pcba_prim).GetRange()
    if world_bbox.IsEmpty():
        raise RuntimeError(f"World bbox is empty for {pcba_root}")
    bb_min = (
        float(world_bbox.GetMin()[0]),
        float(world_bbox.GetMin()[1]),
        float(world_bbox.GetMin()[2]),
    )
    bb_max = (
        float(world_bbox.GetMax()[0]),
        float(world_bbox.GetMax()[1]),
        float(world_bbox.GetMax()[2]),
    )

    # Resolve missing x_num/y_num from bbox + apertures so we have a concrete
    # cell count before sizing the orthographic auto-aperture (which depends
    # on the cell count). For perspective the same auto-fill drops naturally
    # into compute_scan_grid_geometry below, but doing it here keeps a single
    # source of truth.
    bbox_x = bb_max[0] - bb_min[0]
    bbox_y = bb_max[1] - bb_min[1]

    # The millimetres are converted HERE, not at configuration time, because
    # only the stage knows its own scale. A constant guessed upstream is off by
    # a factor of ten on a millimetre stage, which is not a rounding error: it
    # is the difference between framing the product and framing a tenth of it.
    scan_area_mm = cfg.get("scan_area_mm")
    guard_px = int(cfg.get("guard_px") or 0)
    if guard_px and not scan_area_mm:
        raise ValueError(
            "scan_grid guard_px widens the scan_area_mm aperture; no scan_area_mm is set"
        )
    scan_region_su = None
    scan_region_center_su = None
    meters_per_unit = None
    if scan_area_mm:
        declared = float(UsdGeom.GetStageMetersPerUnit(stage) or 0.01)
        # Both spellings. The JobOrder path writes `meters_per_unit_override`;
        # a flow YAML is written by hand, and the name a hand reaches for is
        # the one the declarative config uses, `scene.meters_per_unit`. Reading
        # only one of them makes the other silently fall back to the stage's
        # own declaration -- which is the exact failure this override exists to
        # correct, arriving with no symptom but a mis-framed capture.
        override = cfg.get("meters_per_unit_override") or cfg.get("meters_per_unit")
        meters_per_unit = float(override) if override else declared
        # A stage that declares the wrong scale is framed wrong by exactly that
        # factor, and nothing else in this pipeline would notice: every other
        # consumer works in scene units and never converts. So state both, and
        # state the board's size under each, because a 1.8 m PCB is the shape
        # the mistake takes and it is obvious on sight.
        print(
            f"[Pipeline] Stage scale: declared metersPerUnit={declared}"
            + (f", OVERRIDDEN to {meters_per_unit}" if override else "")
            + f" | board bbox {bbox_x:.3f} x {bbox_y:.3f} su"
            f" = {bbox_x * meters_per_unit * 1000.0:.1f} x "
            f"{bbox_y * meters_per_unit * 1000.0:.1f} mm"
        )
        # Refuses a perspective camera rather than ignoring the rectangle; the
        # rule lives beside the rest of the fov arithmetic.
        scan_region_su = fov_scan_region_su(
            scan_area_mm,
            projection=proj,
            stage_meters_per_unit=meters_per_unit,
        )
        # A field of view stated as corners is placed, not centered on the
        # board: its center travels in millimetres and is converted here with
        # the same scale as the area, for the same reason.
        scan_center_mm = cfg.get("scan_center_mm")
        if scan_center_mm:
            scan_region_center_su = (
                mm_to_su(float(scan_center_mm[0]), meters_per_unit),
                mm_to_su(float(scan_center_mm[1]), meters_per_unit),
            )
            print(
                f"[Pipeline] Scan area centered on ({scan_center_mm[0]:.3f}, "
                f"{scan_center_mm[1]:.3f}) mm = ({scan_region_center_su[0]:.4f}, "
                f"{scan_region_center_su[1]:.4f}) su, not the board midpoint"
            )

    # Auto-fill from what the grid has to span -- the scan rectangle when one
    # is stated, the board otherwise. Matching the cell aspect to the board
    # while tiling a smaller rectangle sizes the count for the wrong shape.
    span_x, span_y = scan_region_su or (bbox_x, bbox_y)
    x_num, y_num = auto_complete_grid_nums(
        span_x,
        span_y,
        horizontal_aperture_su=hap_su,
        vertical_aperture_su=vap_su,
        x_num=x_num_in,
        y_num=y_num_in,
        camera_z_rotation_deg=cam_z_rot_deg,
    )
    if (x_num_in, y_num_in) != (x_num, y_num):
        print(
            f"[Pipeline] scan_grid auto-num: x_num={x_num_in}->{x_num}, "
            f"y_num={y_num_in}->{y_num} (aspect-matched to camera footprint)"
        )

    # Orthographic auto-fit: footprint = aperture (z-independent), so the only
    # way to make ``x_num=y_num=1`` "just cover the board" — and larger
    # ``x_num/y_num`` zoom in proportionally — is to override the camera's
    # aperture to ``bbox / num`` (with square-pixel correction). Symmetric to
    # how perspective auto-derives z. We grow the aperture only; never shrink
    # below what the user authored.
    #
    # ``scan_area_mm`` opts out: the caller stated the footprint in
    # millimetres, and growing it back to the board is exactly wrong for them —
    # scanning a rectangle SMALLER than the board is what a field of view IS.
    # It was converted to scene units above, before the cell counts, because
    # the counts have to be aspect-matched to the rectangle being tiled.
    if scan_region_su is not None:
        region_x, region_y = scan_region_su
        # World-axis footprints per cell, then back into camera axes: at +-90
        # deg of yaw the camera's horizontal aperture is what covers world Y.
        world_x_per_cell = region_x / x_num
        world_y_per_cell = region_y / y_num
        hap_su = world_y_per_cell if swap_aperture_xy else world_x_per_cell
        vap_su = world_x_per_cell if swap_aperture_xy else world_y_per_cell
        # The guard band: the camera renders guard_px more pixels beyond every
        # edge at the same pitch, and the writer crops them off. hap_su/vap_su
        # stay the delivered footprint, so the cell centres, the step between
        # them and the pixel layout are exactly those of an unguarded grid.
        render_hap_su, render_vap_su = hap_su, vap_su
        if guard_px:
            if not aspect_h_over_w:
                raise ValueError(
                    "scan_grid guard_px needs the delivered resolution to widen the "
                    "aperture by whole pixels; cfg['resolution'] is not set"
                )
            render_hap_su = render_aperture(hap_su, int(res[0]), guard_px)
            render_vap_su = render_aperture(vap_su, int(res[1]), guard_px)
        cam.GetHorizontalApertureAttr().Set(render_hap_su * 10.0)
        cam.GetVerticalApertureAttr().Set(render_vap_su * 10.0)
        print(
            f"[Pipeline] Orthographic aperture from scan_area_mm "
            f"{scan_area_mm[0]:.1f}x{scan_area_mm[1]:.1f} mm "
            f"(mpu={meters_per_unit}, z_yaw={cam_z_rot_deg:.1f} deg): "
            f"hap_su={hap_su:.4f}, vap_su={vap_su:.4f} | "
            f"auto-fit would have used bbox_x/x_num={bbox_x / x_num:.3f}, "
            f"bbox_y/y_num={bbox_y / y_num:.3f}"
            + (
                f" | guard {guard_px} px: rendered hap_su={render_hap_su:.4f}, "
                f"vap_su={render_vap_su:.4f}"
                if guard_px
                else ""
            )
        )
    elif proj == "orthographic":
        # Effective world-X aperture is vap when yaw swaps axes (±90°), else hap.
        # Effective world-Y aperture is the opposite. Grow hap so both effective
        # apertures cover bbox/num, while honoring the square-pixel link
        # (vap = hap * aspect_h_over_w).
        if aspect_h_over_w is not None:
            if swap_aperture_xy:
                need_hap_for_world_y = bbox_y / y_num
                need_hap_for_world_x = (bbox_x / x_num) / aspect_h_over_w
            else:
                need_hap_for_world_x = bbox_x / x_num
                need_hap_for_world_y = (bbox_y / y_num) / aspect_h_over_w
            new_hap_su = max(hap_su, need_hap_for_world_x, need_hap_for_world_y)
            new_vap_su = new_hap_su * aspect_h_over_w
        else:
            if swap_aperture_xy:
                new_hap_su = max(hap_su, bbox_y / y_num)  # hap covers world Y
                new_vap_su = max(vap_su, bbox_x / x_num)  # vap covers world X
            else:
                new_hap_su = max(hap_su, bbox_x / x_num)
                new_vap_su = max(vap_su, bbox_y / y_num)
        if new_hap_su > hap_su + 1e-9 or new_vap_su > vap_su + 1e-9:
            cam.GetHorizontalApertureAttr().Set(new_hap_su * 10.0)
            cam.GetVerticalApertureAttr().Set(new_vap_su * 10.0)
            print(
                f"[Pipeline] Orthographic auto-aperture "
                f"(z_yaw={cam_z_rot_deg:.1f} deg): "
                f"hap_su {hap_su:.3f}->{new_hap_su:.3f}, "
                f"vap_su {vap_su:.3f}->{new_vap_su:.3f} "
                f"(bbox_x/x_num={bbox_x / x_num:.3f}, bbox_y/y_num={bbox_y / y_num:.3f})"
            )
            hap_su = new_hap_su
            vap_su = new_vap_su

    # The renderer keeps hap / width on both image axes and ignores vap, so a
    # vap that disagrees is not a different frame shape, it is a frame that
    # misses the cell it was placed on: a step at every row seam and a sawtooth
    # inside each tile. capture.fov never asks for that; a hand-authored
    # scan_grid with scan_area_mm beside an authored resolution can. Refuse it
    # here, where the apertures and the resolution first meet.
    if proj == "orthographic" and aspect_h_over_w is not None:
        check_square_pixels(
            hap_su,
            vap_su,
            (int(res[0]), int(res[1])),
            stage_meters_per_unit=meters_per_unit,
        )

    # z is derived entirely from camera + bbox; ``z_min`` is an optional safety
    # floor (e.g. when the user wants the camera at least N scene units above
    # the origin regardless of the auto value).
    z_min = sg.get("z_min")
    z_min_su = float(z_min) if z_min is not None else None

    geom = compute_scan_grid_geometry(
        bb_min,
        bb_max,
        projection=proj,
        horizontal_aperture_su=hap_su,
        vertical_aperture_su=vap_su,
        focal_length_su=fl_su,
        x_num=x_num,
        y_num=y_num,
        camera_z_rotation_deg=cam_z_rot_deg,
        z_min_camera_su=z_min_su,
        scan_region_su=scan_region_su,
        scan_region_center_su=scan_region_center_su,
    )
    camera_side = cfg.get("camera_side", "top")
    if camera_side not in {"top", "bottom"}:
        raise ValueError(f"camera_side must be 'top' or 'bottom', got {camera_side!r}")
    if camera_side == "bottom":
        clearance = geom["z"] - bb_max[2]
        geom["z"] = bb_min[2] - clearance
    geom["bbox_min"] = list(bb_min)
    geom["bbox_max"] = list(bb_max)
    geom["guard_px"] = guard_px
    cfg["scan_grid"] = geom
    print(
        f"[Pipeline] scan_grid auto ({proj}): "
        f"x=[{geom['x_end']:.2f},{geom['x_start']:.2f}] x_step={geom['x_step']:.2f} | "
        f"y=[{geom['y_end']:.2f},{geom['y_start']:.2f}] y_step={geom['y_step']:.2f} | "
        f"side={camera_side} z={geom['z']:.2f} | "
        f"footprint=({geom['footprint_x']:.2f}x{geom['footprint_y']:.2f}) | "
        f"bbox=({bb_min[0]:.2f},{bb_min[1]:.2f},{bb_min[2]:.2f})->"
        f"({bb_max[0]:.2f},{bb_max[1]:.2f},{bb_max[2]:.2f}) | "
        f"x_num={x_num} y_num={y_num}"
    )
    return geom


def grid_index_for_position(
    x: float,
    y: float,
    scan_grid: dict[str, Any],
) -> tuple[int, int] | None:
    """Map a world-space position back to its scan-grid indices.

    Use ``x_centers`` and ``y_centers``. Convention follows the **iteration order** so the first
    frame written is ``(0, 0)``: ``x_idx=0`` is the first-iterated x (largest
    world x = rightmost; the scan walks left from there) and ``y_idx=0`` is
    the first-iterated y (largest world y = topmost). Returns ``None`` if
    the scan_grid was authored in older form (no centres list).
    """
    if "x_centers" not in scan_grid or "y_centers" not in scan_grid:
        return None
    x_centers = scan_grid["x_centers"]  # descending: largest x first
    y_centers = scan_grid["y_centers"]  # descending: largest y first
    # Closest centre, in case of float drift.
    x_idx = min(range(len(x_centers)), key=lambda i: abs(x_centers[i] - x))
    y_idx = min(range(len(y_centers)), key=lambda i: abs(y_centers[i] - y))
    return (x_idx, y_idx)


def rename_outputs_to_grid_index(
    trigger_dir: str,
    frame_to_grid: list[tuple[int, int, int]],
    frame_padding: int = 4,
    flush_timeout_s: float = 15.0,
    poll_interval_s: float = 0.05,
    quiescence_s: float = 0.4,
) -> int:
    """Rename writer outputs from frame numbers to grid indices.

    Change ``*_<frame_num>.<ext>`` to ``*_x{x_idx}_y{y_idx}.<ext>``.

    Replicator's BasicWriter flushes asynchronously; even after
    ``writer.detach()`` the last frame's outputs may still be encoding when
    this runs, and different annotators (rgb / bbox JSON / npy) finish at
    different times. We rename in repeated passes until no rename has
    happened and no numeric-suffix file remains on disk for ``quiescence_s``
    seconds, or ``flush_timeout_s`` elapses.

    ``frame_to_grid`` — list of ``(frame_num, x_idx, y_idx)`` tuples covering
    every frame written this trigger. Index width is sized from the largest
    x/y index seen so files sort sensibly (e.g. x00 < x01 < x10).
    """
    if not frame_to_grid or not os.path.isdir(trigger_dir):
        return 0
    pad_x = max(1, len(str(max(t[1] for t in frame_to_grid))))
    pad_y = max(1, len(str(max(t[2] for t in frame_to_grid))))
    frame_to_idx = {int(fn): (int(xi), int(yi)) for fn, xi, yi in frame_to_grid}
    sfx_to_new = {
        f"_{fn:0{frame_padding}d}": f"_x{xi:0{pad_x}d}_y{yi:0{pad_y}d}"
        for fn, (xi, yi) in frame_to_idx.items()
    }

    deadline = time.monotonic() + max(0.0, float(flush_timeout_s))
    last_activity = time.monotonic()
    seen_suffixes: set[str] = set()
    renamed = 0
    while True:
        had_activity = False
        for fname in os.listdir(trigger_dir):
            base, ext = os.path.splitext(fname)
            for sfx, new_sfx in sfx_to_new.items():
                if base.endswith(sfx):
                    new_base = base[: -len(sfx)] + new_sfx
                    src = os.path.join(trigger_dir, fname)
                    dst = os.path.join(trigger_dir, new_base + ext)
                    try:
                        os.replace(src, dst)
                        renamed += 1
                        seen_suffixes.add(sfx)
                        had_activity = True
                    except OSError:
                        # Rename failed (e.g. file still being written). Treat
                        # the still-present file as activity so we keep polling.
                        had_activity = True
                    break
        if had_activity:
            last_activity = time.monotonic()

        # Done when we've been quiet long enough — gives slow annotators time
        # to surface even after the fastest-annotator rename pass completes.
        if time.monotonic() - last_activity >= max(0.0, float(quiescence_s)):
            break
        if time.monotonic() >= deadline:
            stragglers = sorted(
                fname
                for fname in os.listdir(trigger_dir)
                if any(os.path.splitext(fname)[0].endswith(sfx) for sfx in sfx_to_new)
            )
            missing_frames = sorted(set(sfx_to_new) - seen_suffixes)
            if stragglers or missing_frames:
                logger.warning(
                    "rename_outputs_to_grid_index: timed out after %.1fs. "
                    "Stragglers on disk: %s. Frames that never surfaced: %s.",
                    flush_timeout_s,
                    stragglers,
                    missing_frames,
                )
            break
        time.sleep(max(0.001, float(poll_interval_s)))
    return renamed


def scan_positions_for_trigger(
    positions_full: list[tuple[float, float, float]],
    trigger_idx: int,
    image_seed: int,
) -> list[tuple[float, float, float]]:
    """Trigger 0: skip the first ``image_seed`` grid points (output frame id == scan order index)."""
    skip = max(0, int(image_seed))
    if int(trigger_idx) == 0:
        return list(positions_full[skip:])
    return list(positions_full)


def cap_positions_for_max_outputs(
    positions: list[tuple[float, float, float]],
    pipeline_type: str,
    max_image_count: int,
    outputs_written: int,
) -> list[tuple[float, float, float]]:
    """Trim positions so total writer frames stay at or below ``max_image_count`` (-1 = unlimited)."""
    if max_image_count < 0:
        return list(positions)
    rem = int(max_image_count) - int(outputs_written)
    if rem <= 0:
        return []
    if pipeline_type == "missing":
        return list(positions[: rem // 2])
    return list(positions[:rem])


def _apply_to_all_lights(
    layer_prim: object,
    intensity: float,
    color: Gf.Vec3f,
    exposure: float,
    cone_angle: float,
    cone_softness: float,
) -> None:
    for child in layer_prim.GetChildren():
        if child.IsA(UsdLux.DiskLight):
            light = UsdLux.DiskLight(child)
            light.GetIntensityAttr().Set(intensity)
            light.GetColorAttr().Set(color)
            light.GetExposureAttr().Set(exposure)
            shaping = UsdLux.ShapingAPI(child)
            shaping.GetShapingConeAngleAttr().Set(cone_angle)
            shaping.GetShapingConeSoftnessAttr().Set(cone_softness)
        elif child.GetChildren():
            _apply_to_all_lights(
                child, intensity, color, exposure, cone_angle, cone_softness
            )


def _ensure_dome_light(
    stage, intensity: float, color, exposure: float, *, visible: bool = False
):
    """Create or update the dome light for white-light AOI rendering.

    Dome modes use a single light at `/World/_aoi_dome_light`.
    """
    path = "/World/_aoi_dome_light"
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        UsdLux.DomeLight.Define(stage, path)
        prim = stage.GetPrimAtPath(path)
    dome = UsdLux.DomeLight(prim)
    dome.CreateIntensityAttr().Set(float(intensity))
    dome.CreateColorAttr().Set(color)
    dome.CreateExposureAttr().Set(float(exposure))
    # Illuminate without rendering as a visible backdrop: a visible DomeLight
    # fills the off-board area with a bright grey, washing the scene and lifting
    # the mean. An authored AOI dome uses ``visibility = invisible`` (it still
    # emits in RTX) so the background stays black. Match that.
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def _set_lights_white(prim) -> int:
    """Recursively set color=(1,1,1) on all light prims; return count changed."""
    white = Gf.Vec3f(1.0, 1.0, 1.0)
    count = 0
    if prim.GetTypeName() in (
        "DiskLight",
        "SphereLight",
        "RectLight",
        "CylinderLight",
        "DistantLight",
        "DomeLight",
    ):
        attr = prim.GetAttribute("inputs:color")
        if not attr.IsValid():
            attr = prim.GetAttribute("color")
        if attr.IsValid():
            attr.Set(white)
            count += 1
    for child in prim.GetChildren():
        count += _set_lights_white(child)
    return count


def _sample_range(value):
    """Sample a ``[lo, hi]`` range (uniform) or pass a scalar through."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(np.random.uniform(float(value[0]), float(value[1])))
    return float(value)


def _randomize_modus_lights(
    stage: object, cfg: dict[str, Any], light_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Per-trigger Modus-light randomization.

    The rig geometry is authored once by :func:`build_modus_lights` -- the
    tunnel is fixed hardware, so there is nothing to re-stamp there. Exposure
    always varies here, sampled and written onto every RectLight in the rig,
    matching how the ring path applies it (``_apply_light_params`` sets the
    same attribute).

    Exposure used to be sampled, reported into the metadata, and never
    applied, so ``lighting.exposure_range`` did nothing to the pixels and the
    record claimed a brightness the image did not have. That is worse than
    ignoring the setting, because a downstream consumer trusts the metadata.

    A group's intensity and colour stay at their built-in fixed value unless
    ``lighting.modus.layers`` names that group -- then it is re-sampled here
    per trigger from the given ranges, same as a ring layer, and every
    RectLight in the group is stamped with the sampled value so multi-light
    groups do not drift apart from each other over triggers.
    """
    modus_root = cfg.get("modus_light_root", "/World/_aoi_modus_light")
    modus_cfg = light_cfg.get("modus") or {}
    layer_ranges = modus_cfg.get("layers") or {}
    exposure = _sample_range(light_cfg.get("exposure_range", [0.0, 0.0]))
    metadata: dict[str, Any] = {
        "mode": "modus_light",
        "global": {"exposure": exposure},
    }

    for group in ("white", "red", "blue"):
        group_prim = stage.GetPrimAtPath(f"{modus_root}/{group.capitalize()}")
        if not group_prim.IsValid():
            logger.warning("Modus light group not found: %s/%s", modus_root, group)
            continue
        rect_children = [c for c in group_prim.GetChildren() if c.IsA(UsdLux.RectLight)]
        if not rect_children:
            continue

        ranges = layer_ranges.get(group)
        intensity = None
        color = None
        if ranges is not None:
            intensity = _sample_range(ranges["intensity"])
            color = Gf.Vec3f(
                _sample_range(ranges["color_r"]),
                _sample_range(ranges["color_g"]),
                _sample_range(ranges["color_b"]),
            )

        for child in rect_children:
            rect = UsdLux.RectLight(child)
            rect.GetExposureAttr().Set(float(exposure))
            if ranges is not None:
                rect.GetIntensityAttr().Set(float(intensity))
                rect.GetColorAttr().Set(color)

        first = UsdLux.RectLight(rect_children[0])
        reported_color = color if color is not None else first.GetColorAttr().Get()
        reported_intensity = (
            intensity
            if intensity is not None
            else float(first.GetIntensityAttr().Get())
        )
        metadata[group] = {
            "intensity": float(reported_intensity),
            "exposure": float(exposure),
            "color": [
                float(reported_color[0]),
                float(reported_color[1]),
                float(reported_color[2]),
            ],
        }
    dome_fill = light_cfg.get("dome_fill")
    if dome_fill:
        intensity = _sample_range(dome_fill["intensity"])
        _ensure_dome_light(
            stage, intensity, Gf.Vec3f(1.0, 1.0, 1.0), exposure, visible=True
        )
        # Keep the emitter enabled; suppress only its camera-visible backdrop.
        dome_prim = stage.GetPrimAtPath("/World/_aoi_dome_light")
        dome_prim.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(
            False
        )
        metadata["dome_fill"] = {
            "intensity": intensity,
            "color": [1.0, 1.0, 1.0],
        }
    logger.info("Lighting randomized (modus): exposure=%.2f", exposure)
    return metadata


def randomize_lighting(stage: object, cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply the configured lighting mode and return the sampled metadata.

    Args:
        stage: USD stage whose lights should be updated.
        cfg: Pipeline configuration containing the lighting mode and ranges.

    Returns:
        Lighting metadata describing the mode and applied parameters.

    Raises:
        ValueError: If the mode or its required lighting configuration is invalid.

    Example:

    .. code-block:: python

        print(randomize_lighting(stage, cfg))

    """
    mode = lighting_mode(cfg)
    light_cfg = cfg.get("lighting") or {}
    if mode == "usd_light":
        cam = stage.GetPrimAtPath("/World/camera_light")
        if cam.IsValid() and not light_cfg.get("preserve_scene_light_color", False):
            n = _set_lights_white(cam)
            logger.info(
                "[Lighting] %d light(s) set to white under /World/camera_light", n
            )
        else:
            logger.info(
                "[Lighting] usd_light with preserve_scene_light_color "
                "- authored colors untouched"
            )
        return {"mode": mode}
    if mode == "modus_light":
        return _randomize_modus_lights(stage, cfg, light_cfg)
    ring_light_root = cfg["ring_light_root"]
    use_ring_light = mode == "rgb_light"

    # Validate up-front so users get a clean error instead of a deep KeyError.
    if use_ring_light and "layers" not in light_cfg:
        raise ValueError("lighting.mode='rgb_light' requires lighting.layers")
    if not use_ring_light and "white_light" not in light_cfg:
        raise ValueError(
            f"lighting.mode={mode!r} requires the lighting.white_light dome block"
        )

    metadata: dict[str, Any] = {"mode": mode}

    exposure = float(np.random.uniform(*light_cfg["exposure_range"]))
    cone_angle = float(np.random.uniform(*light_cfg["cone_angle_range"]))
    cone_softness = float(np.random.uniform(*light_cfg["cone_softness_range"]))
    metadata["global"] = {
        "exposure": exposure,
        "cone_angle": cone_angle,
        "cone_softness": cone_softness,
    }

    if "layers" in light_cfg:
        layer_names = list(light_cfg["layers"].keys())
    else:
        root_prim = stage.GetPrimAtPath(ring_light_root)
        layer_names = (
            [c.GetName() for c in root_prim.GetChildren()]
            if root_prim.IsValid()
            else []
        )

    if use_ring_light:
        for layer_name in layer_names:
            layer_path = f"{ring_light_root}/{layer_name}"
            layer_prim = stage.GetPrimAtPath(layer_path)
            if not layer_prim.IsValid():
                logger.warning("Layer not found: %s", layer_path)
                continue
            ranges = light_cfg["layers"][layer_name]
            intensity = float(np.random.uniform(*ranges["intensity"]))
            color_r = float(np.random.uniform(*ranges["color_r"]))
            color_g = float(np.random.uniform(*ranges["color_g"]))
            color_b = float(np.random.uniform(*ranges["color_b"]))
            color = Gf.Vec3f(color_r, color_g, color_b)
            metadata[layer_name] = {
                "intensity": intensity,
                "color": [color_r, color_g, color_b],
            }
            _apply_to_all_lights(
                layer_prim, intensity, color, exposure, cone_angle, cone_softness
            )
        # Optional neutral white dome fill paired with the colored ring. A
        # typical authored AOI rig is an RGB ring PLUS a white DomeLight; the
        # synthetic ring omits the dome. On a board with vantablack substrate/bodies the
        # dome lifts only the white meshes (silkscreen, package markings) to
        # white while the dark board absorbs it — so legend/markings read white
        # without washing out the dark look. lighting.dome_fill enables it;
        # absent → unchanged behavior (ring only, no dome).
        df = light_cfg.get("dome_fill")
        if df:
            d_intensity = float(np.random.uniform(*df["intensity"]))
            _ensure_dome_light(stage, d_intensity, Gf.Vec3f(1.0, 1.0, 1.0), exposure)
            metadata["dome_fill"] = {
                "intensity": d_intensity,
                "color": [1.0, 1.0, 1.0],
            }
    else:
        # White light mode → single DomeLight (HDRI-style hemisphere) instead of
        # stamping each ring layer white. cone_angle / cone_softness are
        # spotlight-only and have no effect on a DomeLight; recorded in
        # metadata as None for clarity.
        wl = light_cfg["white_light"]
        d_intensity = float(np.random.uniform(*wl["intensity"]))
        d_r = float(np.random.uniform(*wl["color_r"]))
        d_g = float(np.random.uniform(*wl["color_g"]))
        d_b = float(np.random.uniform(*wl["color_b"]))
        d_color = Gf.Vec3f(d_r, d_g, d_b)
        _ensure_dome_light(
            stage,
            d_intensity,
            d_color,
            exposure,
            visible=bool(wl.get("visible", False)),
        )
        # Make sure no leftover ring rig contributes if a previous trigger used rgb_light.
        for layer_name in layer_names:
            layer_prim = stage.GetPrimAtPath(f"{ring_light_root}/{layer_name}")
            if layer_prim.IsValid():
                _apply_to_all_lights(
                    layer_prim,
                    0.0,
                    Gf.Vec3f(0, 0, 0),
                    exposure,
                    cone_angle,
                    cone_softness,
                )
        metadata["dome_light"] = {"intensity": d_intensity, "color": [d_r, d_g, d_b]}

    logger.info(
        "Lighting randomized (%s): exposure=%.2f, cone=%.0f", mode, exposure, cone_angle
    )
    for name in layer_names:
        if name in metadata:
            m = metadata[name]
            logger.info(
                "  %s: intensity=%.0f, color=(%.2f,%.2f,%.2f)",
                name,
                m["intensity"],
                m["color"][0],
                m["color"][1],
                m["color"][2],
            )
    return metadata


def setup_augmentation(render_product: object, cfg: dict[str, Any]) -> dict[str, Any]:
    """Conditionally attach motion blur and return augmentation metadata.

    Args:
        render_product: Render product to receive the augmented color annotator.
        cfg: Pipeline configuration with motion-blur probability and ranges.

    Returns:
        Applied flag and, when enabled, the sampled motion-blur parameters.

    Example:

    .. code-block:: python

        print(setup_augmentation(render_product, cfg))

    """
    aug_cfg = cfg["augmentation"]["motion_blur"]
    aug_meta: dict[str, Any] = {"applied": False}

    if np.random.random() < aug_cfg["probability"]:
        alpha = float(np.random.uniform(*aug_cfg["alpha_range"]))
        kernel_size = int(np.random.choice(aug_cfg["kernel_choices"]))
        ldr_color = rep.annotators.get("LdrColor", device="cuda")
        ldr_color = ldr_color.augment(
            "MotionBlur",
            motionAngle=np.random.uniform(0, 360),
            strength=alpha,
            kernelSize=kernel_size,
        )
        ldr_color.attach(render_product)
        aug_meta = {
            "applied": True,
            "type": "MotionBlur",
            "strength": alpha,
            "kernelSize": kernel_size,
        }
        logger.info(
            "Augmentation: MotionBlur strength=%.2f, kernel=%d", alpha, kernel_size
        )
    else:
        logger.info("Augmentation: none")
    return aug_meta


async def open_usd_stage(app: object, usd_path: str) -> None:
    """Open a USD stage and wait for the application and assets to be ready.

    Args:
        app: Running Kit application providing asynchronous updates.
        usd_path: USD asset path to open.

    Raises:
        RuntimeError: If the USD context cannot open the requested stage.

    Example:

    .. code-block:: python

        await open_usd_stage(app, usd_path)

    """
    ctx = omni.usd.get_context()
    ctx.disable_save_to_recent_files()
    result, error = await ctx.open_stage_async(usd_path)
    ctx.enable_save_to_recent_files()
    if not result:
        raise RuntimeError(f"Cannot open USD file: {usd_path} ({error})")

    while not app.is_app_ready() or not ctx.get_stage():
        await app.next_update_async()

    while ctx.get_stage_loading_status()[2] > 0:
        await app.next_update_async()

    logger.info("Opened stage: %s", usd_path)


# YAML key -> carb path (RTX Interactive Path Tracing). Only keys present in YAML are applied.
# Ref: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html
_PATHTRACING_OPTIONAL_RTX = (
    ("max_bounces", "/rtx/pathtracing/maxBounces"),
    (
        "max_specular_and_transmission_bounces",
        "/rtx/pathtracing/maxSpecularAndTransmissionBounces",
    ),
    ("max_volume_bounces", "/rtx/pathtracing/maxVolumeBounces"),
    ("ptfog_max_bounces", "/rtx/pathtracing/ptfog/maxBounces"),
    ("ptvol_max_bounces", "/rtx/pathtracing/ptvol/maxBounces"),
    ("adaptive_sampling_enabled", "/rtx/pathtracing/adaptiveSampling/enabled"),
    ("adaptive_sampling_target_error", "/rtx/pathtracing/adaptiveSampling/targetError"),
    ("cached_enabled", "/rtx/pathtracing/cached/enabled"),
    ("lightcache_cached_enabled", "/rtx/pathtracing/lightcache/cached/enabled"),
    ("ris_mesh_lights", "/rtx/pathtracing/ris/meshLights"),
    ("optix_denoiser_enabled", "/rtx/pathtracing/optixDenoiser/enabled"),
    (
        "optix_denoiser_temporal_enabled",
        "/rtx/pathtracing/optixDenoiser/temporalMode/enabled",
    ),
    ("optix_denoiser_blend_factor", "/rtx/pathtracing/optixDenoiser/blendFactor"),
    ("optix_denoiser_denoise_aovs", "/rtx/pathtracing/optixDenoiser/AOV"),
    ("firefly_filter_enabled", "/rtx/pathtracing/fireflyFilter/enabled"),
    ("aa_op", "/rtx/pathtracing/aa/op"),
    ("aa_filter_radius", "/rtx/pathtracing/aa/filterRadius"),
    ("reset_pt_accum_on_anim_time_change", "/rtx/resetPtAccumOnAnimTimeChange"),
    ("fractional_cutout_opacity", "/rtx/pathtracing/fractionalCutoutOpacity"),
)


def configure_pathtracing(cfg: dict[str, Any]) -> int:
    """Configure the renderer from *cfg* and return the rt_subframes value.

    ``cfg["render_mode"]`` selects:
      * ``"pathtracing"`` (default) — RTX PathTracing; returns
        ``int(cfg["pathtracing"]["total_spp"])``.
      * ``"realtime"`` — RTX RaytracedLighting; returns
        ``int(cfg.get("realtime", {}).get("subframes", 1))``.

    The return value should be passed directly as ``rt_subframes`` to
    ``rep.orchestrator.step_async``.
    """
    mode = cfg.get("render_mode", "pathtracing").lower()
    settings = carb.settings.get_settings()

    if mode == "realtime":
        settings.set("/rtx/rendermode", "RaytracedLighting")
        subframes = int(cfg.get("realtime", {}).get("subframes", 1))
        print(
            f"[Pipeline] Renderer: RealTime (RaytracedLighting), rt_subframes={subframes}"
        )
        return subframes

    # PathTracing (default)
    pt = cfg["pathtracing"]
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", pt["spp"])
    settings.set("/rtx/pathtracing/totalSpp", pt["total_spp"])
    logger.info("PathTracing: spp=%d, totalSpp=%d", pt["spp"], pt["total_spp"])

    _extras = []
    for yk, path in _PATHTRACING_OPTIONAL_RTX:
        if yk not in pt:
            continue
        val = pt[yk]
        settings.set(path, val)
        _extras.append(f"{yk}={val}")
    if _extras:
        print(f"[Pipeline] PathTracing RTX options: {', '.join(_extras)}")

    return int(pt["total_spp"])


def _find_scope_instances(
    root_prim: object, scope_name: str, result: list[str]
) -> None:
    """Recursively find all Xform instances under a named Scope."""
    for child in root_prim.GetChildren():
        if child.GetName() == scope_name and child.GetTypeName() == "Scope":
            for instance in child.GetChildren():
                if instance.GetTypeName() == "Xform":
                    result.append(str(instance.GetPath()))
        elif child.GetChildren():
            _find_scope_instances(child, scope_name, result)


def build_component_pool(
    stage: object,
    pcba_root: str,
    component_types: list[str],
) -> list[str]:
    """Build a list of all component prim paths from the PCBA hierarchy."""
    pcba_prim = stage.GetPrimAtPath(pcba_root)
    if not pcba_prim.IsValid():
        logger.error("PCBA root not found: %s", pcba_root)
        return []

    all_paths: list[str] = []
    for scope_name in component_types:
        paths: list[str] = []
        _find_scope_instances(pcba_prim, scope_name, paths)
        all_paths.extend(paths)

    logger.info(
        "Component pool: %d components from %d types",
        len(all_paths),
        len(component_types),
    )
    return all_paths


def refresh_authored_semantics(stage: object, prim_paths: list[str]) -> int:
    """Register authored USD semantic labels with Replicator.

    Args:
        stage: Open USD stage containing the labeled prims.
        prim_paths: Prim paths whose authored semantic attributes should be
            registered.

    Returns:
        Number of prims registered with at least one semantic label.

    """
    # `rep.functional.modify.semantics` reaches for
    # `...scripts.utils.utils.process_semantics_value`, but that submodule is
    # NOT in the `from . import (...)` list of `scripts/utils/__init__.py` --
    # checked in both 1.13.4 (Isaac 6.0) and 1.13.32 (6.1). The attribute only
    # exists if something else imported the submodule first, so whether this
    # call works depends on import order elsewhere in the session: it succeeds
    # on the render host and raises `has no attribute
    # 'process_semantics_value'` on a clean local boot. Importing it here makes
    # the dependency explicit instead of inherited.
    import importlib

    importlib.import_module("omni.replicator.core.scripts.utils.utils")

    registered = 0
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        values = {}
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if not name.startswith("semantic:") or not name.endswith(":semanticData"):
                continue
            semantic_prefix = name.removesuffix(":semanticData")
            semantic_type = prim.GetAttribute(f"{semantic_prefix}:semanticType").Get()
            semantic_data = attribute.Get()
            if semantic_type and semantic_data is not None:
                values[str(semantic_type)] = semantic_data
        if values:
            rep_modify.semantics(prim, value=values)
            registered += 1
    return registered


def apply_semantics(
    stage: object,
    pcba_root: str,
    component_types: list[str],
    component_classes: dict | None = None,
    *,
    component_paths: list[str] | None = None,
    default_class: str = DEFAULT_COMPONENT_CLASS,
) -> int:
    """Apply ``class:`` semantic labels to every component instance.

    When ``component_classes`` is provided (a ``{class: [refdes, ...]}`` map, as
    supplied by the board configuration) each instance is labeled
    with its per-type class, keyed by the instance prim's reference designator
    (its path leaf). Any instance the map does not name, and every instance when
    there is no map, gets ``default_class`` -- the neutral
    ``pcba_target.DEFAULT_COMPONENT_CLASS`` unless the pcba target's
    ``default_component_class`` says otherwise. A map that leaves instances
    unnamed is reported with their designators, so a mislabel is visible in the
    log rather than discovered in the dataset.

    Args:
        stage: Open USD stage containing the components.
        pcba_root: Absolute board-root prim path.
        component_types: Component scope names used by legacy discovery.
        component_classes: Optional class-to-refdes mapping.
        component_paths: Explicit catalog-resolved component paths. If omitted,
            components are discovered through the legacy scope hierarchy.
        default_class: Class for instances the map does not name.

    Returns:
        Number of component prims labeled.

    """
    pcba_prim = stage.GetPrimAtPath(pcba_root)
    if not pcba_prim.IsValid():
        logger.error("PCBA root not found: %s", pcba_root)
        return 0

    paths_by_type = []
    if component_paths is not None:
        paths_by_type.append(component_paths)
    else:
        for scope_name in component_types:
            paths: list[str] = []
            _find_scope_instances(pcba_prim, scope_name, paths)
            paths_by_type.append(paths)

    # Every instance's reference designator is its path leaf. Classes are
    # decided in one place (pcba_target.classify_components) so the rule --
    # and what falls back -- is the same one the unit tests exercise.
    instances: list[tuple[object, str]] = []
    for paths in paths_by_type:
        for prim_path in paths:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                instances.append((prim, prim_path.rsplit("/", 1)[-1]))
    classes, unmapped = classify_components(
        [refdes for _, refdes in instances], component_classes, default_class
    )

    total = 0
    applied: dict[str, int] = {}
    for prim, refdes in instances:
        cls = classes[refdes]
        prim.CreateAttribute(
            "semantic:Semantics:class:semanticData",
            Sdf.ValueTypeNames.String,
        ).Set(cls)
        prim.CreateAttribute(
            "semantic:Semantics:class:semanticType",
            Sdf.ValueTypeNames.String,
        ).Set("class")
        rep_modify.semantics(prim, value={"class": cls})
        applied[cls] = applied.get(cls, 0) + 1
        total += 1

    if component_classes:
        logger.info(
            "Applied per-type semantic labels to %d components (%s)",
            total,
            ", ".join(
                f"{c}:{n}" for c, n in sorted(applied.items(), key=lambda x: -x[1])
            ),
        )
        if unmapped:
            # A map that misses parts is the case the old `capacitor` fallback
            # hid: ICs and connectors labelled as passives, and a dataset that
            # looked right. Say which designators, so the target gets fixed.
            shown = ", ".join(unmapped[:20])
            if len(unmapped) > 20:
                shown += f", ... ({len(unmapped) - 20} more)"
            logger.warning(
                "component_classes names no class for %d of %d components; "
                "labeled them %r. Add them to the pcba target's component_classes "
                "or set default_component_class: %s",
                len(unmapped),
                total,
                default_class,
                shown,
            )
    else:
        logger.info(
            "Applied baseline semantic label %r to %d components (no "
            "component_classes in the pcba target)",
            default_class,
            total,
        )
    return total


def find_translate_op(stage: object, camera_xform_path: str) -> UsdGeom.XformOp:
    """Find the translate xformOp on the camera xform."""
    xform = UsdGeom.Xformable.Get(stage, camera_xform_path)
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            return op
    raise RuntimeError(f"{camera_xform_path} missing xformOp:translate")


def save_metadata(
    trigger_dir: str, metadata: dict[str, Any], filename: str = "metadata.json"
) -> None:
    """Write trigger metadata as indented JSON.

    Args:
        trigger_dir: Existing output directory for the trigger.
        metadata: JSON-serializable metadata to save.
        filename: Name of the output file within the trigger directory.

    Raises:
        OSError: If the output file cannot be written.
        TypeError: If a metadata value cannot be serialized as JSON.

    Example:

    .. code-block:: python

        save_metadata(trigger_dir, {"applied": False})

    """
    with open(os.path.join(trigger_dir, filename), "w") as f:
        json.dump(metadata, f, indent=2)


# ── Light helpers ─────────────────────────────────────────────────────────────

_LIGHT_TYPES = {
    "DiskLight",
    "SphereLight",
    "RectLight",
    "CylinderLight",
    "DistantLight",
    "DomeLight",
    "PortalLight",
}


def disable_all_lights(stage) -> int:
    """Make every existing light prim in the stage invisible.

    Returns the number of lights that were disabled.
    """
    count = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() in _LIGHT_TYPES:
            UsdGeom.Imageable(prim).MakeInvisible()
            count += 1
    if count:
        print(f"[Pipeline] Disabled {count} existing light(s) found in scene.")
    return count


def build_ring_lights(
    stage,
    ring_light_root: str,
    camera_side: str = "top",
    dome_radius: float = 100.0,
    light_radius: float = 4.0,
    light_intensity: float = 5000.0,
    cone_angle: float = 120.0,
    cone_softness: float = 1.0,
) -> None:
    """Build a half-dome AOI ring light rig under *ring_light_root*.

    The rig is made of DiskLights arranged in three color layers
    (Inner_Red / Middle_Green / Outer_Blue), each layer split across
    NUM_RINGS_PER_LAYER rings of LIGHTS_PER_RING lights.

    All positions are in the LOCAL space of *ring_light_root*'s parent,
    centred at (0, 0, 0).  If ring_light_root is a child of the camera
    xform, the whole rig translates with the camera automatically.

    ``camera_side="bottom"`` mirrors the dome below the camera so the lights
    point upward toward the bottom of the board.

    Geometry logic is taken from ring_light_controller.py.
    Pitch range: 0° (horizontal) → 90° (straight up).

    dome_radius / light_radius are read from config YAML (rig.dome_radius /
    rig.light_radius_world) and passed in by the pipeline; defaults match the
    canonical values in the template configs.
    """
    NUM_RINGS_PER_LAYER = 5
    LIGHTS_PER_RING = 60
    DOME_RADIUS = float(dome_radius)
    LIGHT_RADIUS = float(light_radius)
    LIGHT_INTENSITY = float(light_intensity)
    CONE_ANGLE = float(cone_angle)
    CONE_SOFTNESS = float(cone_softness)
    if camera_side not in {"top", "bottom"}:
        raise ValueError(f"camera_side must be 'top' or 'bottom', got {camera_side!r}")
    z_sign = -1.0 if camera_side == "bottom" else 1.0
    MIN_PITCH = 0.0  # degrees — horizontal ring at dome equator
    MAX_PITCH = 90.0  # degrees — ring directly overhead

    COLOR_LAYERS = [
        ("Outer_Blue", Gf.Vec3f(0.0, 0.0, 1.0)),  # low pitch → outer rings → blue
        ("Middle_Green", Gf.Vec3f(0.0, 1.0, 0.0)),
        ("Inner_Red", Gf.Vec3f(1.0, 0.0, 0.0)),  # high pitch → inner rings → red
    ]

    # Ensure root xform exists; clear any pre-existing children
    UsdGeom.Xform.Define(stage, ring_light_root)
    root_prim = stage.GetPrimAtPath(ring_light_root)
    for child in list(root_prim.GetChildren()):
        stage.RemovePrim(child.GetPath())

    total_rings = len(COLOR_LAYERS) * NUM_RINGS_PER_LAYER
    total_lights = 0

    for layer_idx, (layer_name, color) in enumerate(COLOR_LAYERS):
        layer_path = f"{ring_light_root}/{layer_name}"
        UsdGeom.Xform.Define(stage, layer_path)

        for ring_idx in range(NUM_RINGS_PER_LAYER):
            global_ring_idx = layer_idx * NUM_RINGS_PER_LAYER + ring_idx

            pitch_ratio = (
                global_ring_idx / (total_rings - 1) if total_rings > 1 else 0.0
            )
            pitch_deg = MIN_PITCH + pitch_ratio * (MAX_PITCH - MIN_PITCH)
            pitch_rad = np.radians(pitch_deg)

            ring_radius_xy = DOME_RADIUS * np.cos(pitch_rad)
            z_pos = z_sign * DOME_RADIUS * np.sin(pitch_rad)

            ring_path = f"{layer_path}/Ring_{ring_idx:02d}_p{int(pitch_deg)}"
            UsdGeom.Xform.Define(stage, ring_path)

            for light_idx in range(LIGHTS_PER_RING):
                azimuth = 2.0 * np.pi * light_idx / LIGHTS_PER_RING
                x = ring_radius_xy * np.cos(azimuth)
                y = ring_radius_xy * np.sin(azimuth)
                z = z_pos

                light = UsdLux.DiskLight.Define(stage, f"{ring_path}/L_{light_idx:03d}")
                light.CreateIntensityAttr(LIGHT_INTENSITY)
                light.CreateColorAttr(color)
                light.CreateRadiusAttr(LIGHT_RADIUS)

                shaping = UsdLux.ShapingAPI(light)
                shaping.CreateShapingConeAngleAttr(CONE_ANGLE)
                shaping.CreateShapingConeSoftnessAttr(CONE_SOFTNESS)

                xformable = UsdGeom.Xformable(light)
                xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

                # Rotate DiskLight's -Z axis to point toward local origin
                horizontal_dist = np.sqrt(x * x + y * y)
                if horizontal_dist > 0:
                    actual_pitch = np.degrees(np.arctan2(-z, horizontal_dist)) + 90.0
                else:
                    actual_pitch = 0.0 if z > 0 else 180.0
                xformable.AddRotateZYXOp().Set(
                    Gf.Vec3f(0.0, actual_pitch, np.degrees(azimuth))
                )

                total_lights += 1

    print(
        f"[Pipeline] Built ring light rig: {total_lights} DiskLights "
        f"({total_rings} rings, pitch {MIN_PITCH}-{MAX_PITCH} deg) "
        f"under {ring_light_root}"
    )


#: Knobs :func:`compute_modus_layout` accepts from a ``lighting.modus`` block.
_MODUS_KNOBS = (
    "elevations",
    "arc_radius_scale",
    "width",
    "width_frac",
    "height_scale",
    "colors",
    "intensities",
)


def build_modus_lights(
    stage: object,
    modus_root: str,
    pcba_root: str,
    *,
    modus: dict | None = None,
    exposure: float = 0.0,
    normalize: bool = False,
    camera_side: str = "top",
    bounds_paths: Sequence[str] | None = None,
) -> None:
    """Build the **Modus light** rig — a 2D-dome linear AOI illuminator.

    Six ``UsdLux.RectLight`` prims sit on an arc in the world XZ plane (at
    ``Y = board-centre-Y``), 3 per side of the top-down camera, all aiming at
    the board centre (see :mod:`pcb_aoi.modus_light` for the geometry).

    Unlike :func:`build_ring_lights` (authored in the *local* frame of a
    camera-parented ``ring_light_root`` so it scans with the camera), the Modus
    rig is a **fixed tunnel**: it is authored in WORLD space under *modus_root*,
    which must be an identity xform near the world origin (e.g.
    ``/World/_aoi_modus_light``), so the light positions computed from the board
    bbox land where intended.

    Args:
        stage: Open USD stage to modify.
        modus_root: Absolute prim path for the world-space light rig.
        pcba_root: Board prim whose world bounds size the rig.
        modus: Validated ``lighting.modus`` settings. Omitted values use the
            defaults from :mod:`pcb_aoi.modus_light`.
        exposure: Exposure authored on every RectLight.
        normalize: Whether RectLight power is normalized by emitter area.
        camera_side: Inspected board side. Bottom inspection mirrors and aims
            the lights below the minimum Z face without changing their colors.
        bounds_paths: Prims whose union of world bounds sizes and centers the
            rig, such as every board placement of a step-repeated panel. When
            None, the rig is sized from ``pcba_root`` alone.

    Raises:
        RuntimeError: If a sizing prim or its world bounds are unavailable.
        ValueError: If ``camera_side`` is not supported.

    Example:

    .. code-block:: python

        build_modus_lights(
            stage,
            "/World/_aoi_modus_light",
            "/World/Board",
            camera_side="bottom",
        )

    """
    knobs = {
        key: value
        for key, value in (modus or {}).items()
        if key in _MODUS_KNOBS and value is not None
    }
    center, top_z, x_extent, y_extent = modus_bounds_from_prims(
        stage,
        tuple(bounds_paths) if bounds_paths is not None else (pcba_root,),
    )
    lights = compute_modus_layout(
        board_center=center,
        board_top_z=top_z,
        board_x_extent=x_extent,
        board_y_extent=y_extent,
        camera_side=camera_side,
        **knobs,
    )

    UsdGeom.Xform.Define(stage, modus_root)
    root_prim = stage.GetPrimAtPath(modus_root)
    for child in list(root_prim.GetChildren()):
        stage.RemovePrim(child.GetPath())

    # One Xform per colour group so the per-trigger randomizer can re-stamp
    # intensity/colour by group (mirrors build_ring_lights' layer Xforms).
    group_paths = {}
    meta: list[dict] = []
    for spec in lights:
        group_path = f"{modus_root}/{spec.color_group.capitalize()}"
        if group_path not in group_paths:
            UsdGeom.Xform.Define(stage, group_path)
            group_paths[group_path] = True

        light = UsdLux.RectLight.Define(stage, f"{group_path}/{spec.name}")
        light.CreateWidthAttr(float(spec.width))
        light.CreateHeightAttr(float(spec.height))
        light.CreateIntensityAttr(float(spec.intensity))
        light.CreateColorAttr(Gf.Vec3f(*spec.color))
        light.CreateExposureAttr(float(exposure))
        light.CreateNormalizeAttr(bool(normalize))

        # Row-major (row-vector) world transform: rows are the local basis
        # vectors, last row the translation. Local -Z (RectLight emission) then
        # points from the light at the board centre.
        x_ax, y_ax, z_ax, pos = spec.x_axis, spec.y_axis, spec.z_axis, spec.position
        matrix = Gf.Matrix4d(
            x_ax[0],
            x_ax[1],
            x_ax[2],
            0.0,
            y_ax[0],
            y_ax[1],
            y_ax[2],
            0.0,
            z_ax[0],
            z_ax[1],
            z_ax[2],
            0.0,
            pos[0],
            pos[1],
            pos[2],
            1.0,
        )
        xformable = UsdGeom.Xformable(light)
        transform_op = next(
            (
                op
                for op in xformable.GetOrderedXformOps()
                if op.GetOpName() == "xformOp:transform"
            ),
            None,
        )
        if transform_op is None:
            transform_op = xformable.AddTransformOp()
        transform_op.Set(matrix)

    elev_by_group = {lt.color_group: round(lt.elevation_deg, 1) for lt in lights}
    print(
        f"[Pipeline] Built Modus light rig: {len(lights)} RectLights "
        f"(2D dome arc, elevations {elev_by_group}) under {modus_root}"
    )
