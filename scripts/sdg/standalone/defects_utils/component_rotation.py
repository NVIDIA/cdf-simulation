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

"""Internal geometry implementations for component rotation defects."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from defects_utils.transform_utils import (
    _bbox_projection_range,
    _bbox_range_corners,
    _component_rotation_frame,
    _min_support_projection_after_rotation,
    _normalized,
    _pivot_rotation_matrix,
    _pivot_rotation_transform,
    _project_point_to_plane,
)

logger = logging.getLogger(__name__)

# Stratified shift sampling. All bounds are FRACTIONS of the configured
# translate_range / rotate_z_range so the env caps keep their meaning.
# The variant index (deterministic per scene) round-robins mode, direction
# sector, and magnitude bin so a large run covers the whole space evenly.
_SHIFT_MODES = ("translate", "rotate", "combined")
_SHIFT_DIRECTION_SECTORS = 8
_SHIFT_TRANSLATE_BINS = ((0.15, 0.30), (0.30, 0.60), (0.60, 1.00))
_SHIFT_ROTATE_BINS = ((0.20, 0.60), (0.60, 1.00))
_SHIFT_TRANSLATE_MODE_ROT_JITTER = 0.12
_SHIFT_ROTATE_MODE_TRANS_JITTER = 0.10
_SHIFT_COMBINED_TRANSLATE = (0.20, 0.50)
_SHIFT_COMBINED_ROTATE = (0.20, 0.60)


def _yawed_hinge(pivot, axis, yaw_matrix):
    """Carry a hinge pivot and axis through a yaw, so it stays the part's edge.

    The flip must rotate the part about its OWN support edge. A yaw applied
    before the flip moves that edge, so the hinge has to move with it; hinging
    about the un-yawed pad axis rotates the part about a line that is no longer
    an edge of it.

    Args:
        pivot: Hinge pivot in component-local space, before the yaw.
        axis: Hinge direction in component-local space, before the yaw.
        yaw_matrix: The 4x4 row-vector yaw applied before the flip.

    Returns:
        ``(pivot, axis)`` in the yawed frame.

    Example:

    .. code-block:: python

        pivot, axis = _yawed_hinge(pivot, axis, yaw)  # doctest: +SKIP
    """
    rotation3 = np.asarray(yaw_matrix)[:3, :3]
    moved_axis = np.asarray(axis, dtype=float) @ rotation3
    moved_pivot = (np.asarray(pivot, dtype=float) @ rotation3
                   + np.asarray(yaw_matrix)[3, :3])
    return moved_pivot, moved_axis


def _shift_variant_plan(variant_index: int) -> dict[str, Any]:
    """Decode a deterministic shift stratification plan from a variant index.

    Args:
        variant_index: Deterministic per-scene index.

    Returns:
        Mode, direction sector, magnitude-bin, and sign assignments.
    """
    # A mixed-radix odometer: each axis gets its own digit, and a digit only
    # advances once every axis below it has wrapped. That is what makes the axes
    # independent.
    #
    # An earlier revision reused strides (rotate_bin every 6, rotate_sign every
    # 3) and the result was that every direction sector was permanently locked to
    # one rotation bin and one sign -- sector 0 could never receive a large
    # rotation. Only 72 of the 3 x 8 x 3 x 2 x 2 = 288 combinations were
    # reachable, which is the clumping this whole mechanism exists to prevent.
    v = int(variant_index)
    stride_sector = 3
    stride_translate = stride_sector * _SHIFT_DIRECTION_SECTORS          # 24
    stride_rotate_bin = stride_translate * len(_SHIFT_TRANSLATE_BINS)    # 72
    stride_rotate_sign = stride_rotate_bin * len(_SHIFT_ROTATE_BINS)     # 144
    return {
        "mode": _SHIFT_MODES[v % 3],
        "sector": (v // stride_sector) % _SHIFT_DIRECTION_SECTORS,
        "translate_bin": (v // stride_translate) % len(_SHIFT_TRANSLATE_BINS),
        "rotate_bin": (v // stride_rotate_bin) % len(_SHIFT_ROTATE_BINS),
        "rotate_sign": 1 if (v // stride_rotate_sign) % 2 == 0 else -1,
    }


def _world_planar_direction(
    axis_local: np.ndarray,
    M_orig: np.ndarray,
) -> np.ndarray | None:
    """Map a component-local direction to a unit board-plane world direction.

    Args:
        axis_local: Direction in component-local space.
        M_orig: Component transform in row-major (row-vector) form.

    Returns:
        Unit world direction projected to the global board plane (z=0), or
        None when the projection is degenerate.
    """
    world = np.asarray(axis_local, dtype=float) @ M_orig[:3, :3]
    world[2] = 0.0
    return _normalized(world)


def _compute_board_normal_rotation_transform(
    prim: object,
    M_orig: np.ndarray,
    angle_deg: float,
    component_instance: dict[str, Any] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Rotate a component about its center along the local board normal.

    Args:
        prim: Component USD prim whose geometry defines the rotation frame.
        M_orig: Original component transform in row-major form.
        angle_deg: Signed rotation angle in degrees.
        component_instance: Optional component metadata with solder prim paths.

    Returns:
        Flattened absolute transform and board-normal rotation parameters.
    """
    frame = _component_rotation_frame(prim, M_orig, component_instance)
    pivot = frame["center"]
    angle = float(angle_deg)
    final_flat = _pivot_rotation_transform(
        M_orig,
        pivot,
        frame["board_normal_local"],
        angle,
    )
    params = {
        "angle_deg": angle,
        "axis_source": frame["axis_source"],
        "pivot_local": pivot.tolist(),
        "rot_axis": frame["board_normal_local"].tolist(),
        "long_axis": frame["long_axis"].tolist(),
        "width_axis": frame["width_axis"].tolist(),
        "board_normal_local": frame["board_normal_local"].tolist(),
    }
    logger.info(
        "Board-normal rotation (%s): angle=%.1f deg, board_normal=%s",
        frame["axis_source"],
        angle,
        np.array2string(frame["board_normal_local"], precision=3),
    )
    return final_flat, params


def _compute_shift_transform(
    prim: object,
    M_orig: np.ndarray,
    translate_range: float,
    rotate_z_range: float,
    component_instance: dict[str, Any] | None = None,
    variant_index: int | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Apply an in-plane offset and board-normal rotation to a component.

    With a ``variant_index`` the defect is sampled from a deterministic
    stratification: the scene round-robins one of three modes
    (translate-dominant / rotate-dominant / combined), a component-frame
    direction sector, and a magnitude bin, so a large run covers the whole
    parameter space evenly and every sample stays clearly above the visual
    no-defect floor. Without an index the legacy uniform sampling is used.

    Args:
        prim: Component USD prim whose geometry defines the rotation frame.
        M_orig: Original component transform in row-major form.
        translate_range: Symmetric in-plane translation bound (mm).
        rotate_z_range: Symmetric board-normal rotation bound (deg).
        component_instance: Optional component metadata with solder prim paths.
        variant_index: Deterministic per-scene stratification index.

    Returns:
        Flattened absolute transform and sampled shift parameters.

    Raises:
        ValueError: If either range is negative.
    """
    if translate_range < 0.0 or rotate_z_range < 0.0:
        raise ValueError("Shift ranges must be non-negative")

    if variant_index is None:
        translation = np.random.uniform(
            [-translate_range, -translate_range, 0.0],
            [translate_range, translate_range, 0.0],
        )
        angle_deg = float(np.random.uniform(-rotate_z_range, rotate_z_range))
        extra_params: dict[str, Any] = {"mode": "legacy_uniform"}
    else:
        plan = _shift_variant_plan(variant_index)
        frame = _component_rotation_frame(prim, M_orig, component_instance)
        # Direction sampled inside the assigned sector, expressed in the
        # component frame (0 = slide along the length axis, 90 deg = slide
        # across the pads) then mapped to the global board plane.
        sector_width = 2.0 * np.pi / _SHIFT_DIRECTION_SECTORS
        theta = (plan["sector"] + float(np.random.uniform(0.0, 1.0))) * sector_width
        dir_local = np.cos(theta) * frame["long_axis"] + np.sin(theta) * frame["width_axis"]
        dir_world = _world_planar_direction(dir_local, M_orig)
        if dir_world is None:
            fallback = float(np.random.uniform(0.0, 2.0 * np.pi))
            dir_world = np.array([np.cos(fallback), np.sin(fallback), 0.0])

        if plan["mode"] == "translate":
            lo, hi = _SHIFT_TRANSLATE_BINS[plan["translate_bin"]]
            magnitude = float(np.random.uniform(lo, hi)) * translate_range
            angle_deg = float(
                np.random.uniform(-1.0, 1.0)
                * _SHIFT_TRANSLATE_MODE_ROT_JITTER
                * rotate_z_range
            )
        elif plan["mode"] == "rotate":
            lo, hi = _SHIFT_ROTATE_BINS[plan["rotate_bin"]]
            angle_deg = (
                plan["rotate_sign"] * float(np.random.uniform(lo, hi)) * rotate_z_range
            )
            magnitude = (
                float(np.random.uniform(0.0, _SHIFT_ROTATE_MODE_TRANS_JITTER))
                * translate_range
            )
        else:  # combined
            lo, hi = _SHIFT_COMBINED_TRANSLATE
            magnitude = float(np.random.uniform(lo, hi)) * translate_range
            rlo, rhi = _SHIFT_COMBINED_ROTATE
            angle_deg = (
                plan["rotate_sign"] * float(np.random.uniform(rlo, rhi)) * rotate_z_range
            )
        translation = dir_world * magnitude
        extra_params = {
            "mode": plan["mode"],
            "variant_index": int(variant_index),
            "direction_sector": plan["sector"],
            "direction_theta_deg": float(np.degrees(theta)),
            "translate_magnitude_mm": float(magnitude),
        }

    final_flat, rotation_params = _compute_board_normal_rotation_transform(
        prim,
        M_orig,
        angle_deg,
        component_instance=component_instance,
    )
    final_matrix = np.asarray(final_flat).reshape(4, 4)
    final_matrix[3, :3] += translation

    params = {
        **rotation_params,
        **extra_params,
        "translate": translation.tolist(),
        "translate_range": float(translate_range),
        "rotate_z_deg": angle_deg,
        "rotate_z_range": float(rotate_z_range),
    }
    logger.info(
        "Shift (%s, %s): translation=%s, board-normal rotation=%.1f deg",
        rotation_params["axis_source"],
        extra_params.get("mode"),
        np.array2string(np.asarray(translation), precision=3),
        angle_deg,
    )
    return final_matrix.flatten().tolist(), params


def _compute_sideflip_transform(
    prim: object,
    M_orig: np.ndarray,
    angle_min: float,
    angle_max: float,
    component_instance: dict[str, Any] | None = None,
    variant_index: int | None = None,
    rotate_z_jitter_deg: float = 0.0,
    slide_range_mm: float = 0.0,
) -> tuple[list[float], dict[str, Any]]:
    """Rotate a component about a board-contacting edge along its length.

    With a ``variant_index`` the flip angle is drawn from one of three equal
    sub-bins of ``[angle_min, angle_max]`` (round-robin) and the hinge edge
    alternates deterministically between the two long support edges, so both
    edges and the whole angle range are covered evenly. An optional yaw
    jitter about the board normal and a slide along the hinge break the
    perfect axis-aligned look of a pure geometric flip.

    Args:
        prim: Component USD prim whose geometry defines the rotation frame.
        M_orig: Original component transform in row-major form.
        angle_min: Minimum sideflip magnitude.
        angle_max: Maximum sideflip magnitude.
        component_instance: Optional component metadata with solder prim paths.
        variant_index: Deterministic per-scene stratification index.
        rotate_z_jitter_deg: Symmetric board-normal pre-rotation bound (deg).
        slide_range_mm: Symmetric slide bound along the hinge axis (mm).

    Returns:
        Flattened absolute transform and sampled sideflip parameters.
    """
    frame = _component_rotation_frame(prim, M_orig, component_instance)
    center = frame["center"]
    width_axis = frame["width_axis"]
    normal_axis = frame["board_normal_local"]

    # Pick a long support edge as the hinge. The rotation sign that gives the
    # greatest board clearance is selected, as in the tombstone algorithm.
    angle_bin = None
    if variant_index is None:
        edge_sign = int(np.random.choice([-1, 1]))
        angle_magnitude = float(np.random.uniform(angle_min, angle_max))
    else:
        v = int(variant_index)
        angle_bin = v % 3
        edge_sign = 1 if (v // 3) % 2 == 0 else -1
        bin_width = (float(angle_max) - float(angle_min)) / 3.0
        bin_lo = float(angle_min) + angle_bin * bin_width
        angle_magnitude = float(np.random.uniform(bin_lo, bin_lo + bin_width))

    min_width, max_width = _bbox_projection_range(
        frame["vmin"], frame["vmax"], width_axis
    )
    edge_projection = max_width if edge_sign > 0 else min_width
    pivot = _project_point_to_plane(center, normal_axis, frame["support_projection"])
    pivot += width_axis * (edge_projection - float(np.dot(pivot, width_axis)))

    angle_rad = np.radians(angle_magnitude)
    positive_clearance = _min_support_projection_after_rotation(
        frame["vmin"], frame["vmax"], pivot, frame["long_axis"], angle_rad, normal_axis
    )
    negative_clearance = _min_support_projection_after_rotation(
        frame["vmin"], frame["vmax"], pivot, frame["long_axis"], -angle_rad, normal_axis
    )
    angle_sign = 1 if positive_clearance >= negative_clearance else -1
    angle_deg = angle_magnitude * angle_sign

    # Pre-perturbation: yaw about the board normal at the component center and
    # slide along the hinge, applied in component-local space before the flip,
    # so the fallen part is not perfectly parallel to its pads.
    #
    # The hinge travels with the yaw -- see _yawed_hinge below. It did not
    # always: the flip was built from a pivot and axis measured on the un-yawed
    # frame, so a non-zero jitter made the part rotate about a line that was no
    # longer one of its edges, and the lift then pushed it back onto the board.
    # The pose looked plausible and was not the one the parameters described.
    yaw_deg = (
        float(np.random.uniform(-rotate_z_jitter_deg, rotate_z_jitter_deg))
        if rotate_z_jitter_deg > 0.0
        else 0.0
    )
    slide_mm = (
        float(np.random.uniform(-slide_range_mm, slide_range_mm))
        if slide_range_mm > 0.0
        else 0.0
    )
    local = np.eye(4)
    if slide_mm != 0.0:
        local[3, :3] = frame["long_axis"] * slide_mm

    # The flip hinges on the part's OWN support edge. Once a yaw is applied that
    # edge has moved with it, so the hinge pivot and axis are carried through the
    # same rotation before the flip is built. Hinging about the un-yawed pad axis
    # instead made the part rotate about a line that was no longer its edge: the
    # lift below then had to push it back onto the board, which produced a
    # plausible pose but not the one the parameters describe.
    hinge_pivot = pivot
    hinge_axis = frame["long_axis"]
    if yaw_deg != 0.0:
        yaw_matrix = _pivot_rotation_matrix(center, normal_axis, yaw_deg)
        local = local @ yaw_matrix
        hinge_pivot, hinge_axis = _yawed_hinge(pivot, hinge_axis, yaw_matrix)
    local = local @ _pivot_rotation_matrix(hinge_pivot, hinge_axis, angle_deg)

    # The yaw perturbation can swing a corner of the support edge below the
    # board plane; lift the whole part along the board normal so the lowest
    # bbox corner stays on the original support plane.
    corners = _bbox_range_corners(frame["vmin"], frame["vmax"])
    corners_h = np.hstack([corners, np.ones((len(corners), 1))])
    moved = (corners_h @ local)[:, :3]
    lift_mm = max(0.0, frame["support_projection"] - float(np.min(moved @ normal_axis)))
    if lift_mm > 0.0:
        local[3, :3] += normal_axis * lift_mm
    final_flat = (local @ M_orig).flatten().tolist()

    params = {
        "angle_deg": angle_deg,
        "sign": angle_sign,
        "edge_sign": edge_sign,
        "angle_bin": angle_bin,
        "variant_index": None if variant_index is None else int(variant_index),
        "rotate_z_jitter_deg": yaw_deg,
        "slide_along_hinge_mm": slide_mm,
        "board_clearance_lift_mm": lift_mm,
        "axis_source": frame["axis_source"],
        "pivot_local": pivot.tolist(),
        "rot_axis": frame["long_axis"].tolist(),
        "width_axis": width_axis.tolist(),
        "board_normal_local": normal_axis.tolist(),
        "support_projection": frame["support_projection"],
        "positive_clearance": positive_clearance,
        "negative_clearance": negative_clearance,
    }
    logger.info(
        "Sideflip (%s): angle=%.1f deg (bin=%s edge=%+d yaw=%.1f slide=%.2f), length_axis=%s",
        frame["axis_source"],
        angle_deg,
        angle_bin,
        edge_sign,
        yaw_deg,
        slide_mm,
        np.array2string(frame["long_axis"], precision=3),
    )
    return final_flat, params
