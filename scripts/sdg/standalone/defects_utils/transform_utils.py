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

"""Shared component geometry and row-major transform helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import omni.usd
from pxr import UsdGeom


def _local_transform_matrix(prim: object) -> np.ndarray:
    """Return a component's composed local transform as a 4x4 array.

    Args:
        prim: USD transformable prim.

    Returns:
        Composed local transform in row-major form.

    Raises:
        ValueError: If the prim has no usable local transform.
    """
    transform_attr = prim.GetAttribute("xformOp:transform")
    matrix = (
        transform_attr.Get()
        if transform_attr is not None and transform_attr.IsValid()
        else None
    )
    if matrix is None:
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError("prim is not UsdGeom.Xformable")
        matrix = xformable.GetLocalTransformation()
    if matrix is None:
        raise ValueError("prim has no composed local transform")

    result = np.asarray([[float(value) for value in row] for row in matrix])
    if result.shape != (4, 4):
        raise ValueError(f"expected a 4x4 local transform, got {result.shape}")
    return result


def _normalized(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    """Return a unit vector when the input has a stable direction.

    Args:
        vec: Vector to normalize.
        eps: Minimum accepted vector magnitude.

    Returns:
        Unit vector, or None when the magnitude does not exceed the threshold.
    """
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return None
    return vec / norm


def _bbox_wide_and_normal_axes(
    half_extents: np.ndarray,
    long_axis_vec: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Infer component-width and height axes from local bbox geometry.

    Args:
        half_extents: Local bounding-box half extents.
        long_axis_vec: Candidate component length direction.

    Returns:
        Width and normal axes, or two None values for degenerate geometry.
    """
    long_axis_norm = _normalized(long_axis_vec)
    if long_axis_norm is None:
        return None, None

    basis = np.eye(3)
    perp_lengths = np.linalg.norm(
        basis - np.outer(basis @ long_axis_norm, long_axis_norm),
        axis=1,
    )
    scores = half_extents * perp_lengths
    if float(np.max(scores)) <= 1e-6:
        return None, None

    wide_idx = int(np.argmax(scores))
    normal_idx = int(np.argmin(scores + (perp_lengths <= 1e-6) * 1e9))
    wide_axis = basis[wide_idx]
    normal_hint = basis[normal_idx]
    normal_axis = _normalized(np.cross(long_axis_norm, wide_axis))

    if normal_axis is not None and float(np.dot(normal_axis, normal_hint)) < 0.0:
        wide_axis = -wide_axis
        normal_axis = -normal_axis

    return _normalized(wide_axis), normal_axis


def _bbox_range_corners(vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    """Return the eight corners of an axis-aligned bbox range.

    Args:
        vmin: Minimum bounding-box coordinates.
        vmax: Maximum bounding-box coordinates.

    Returns:
        Array containing the eight corner coordinates.
    """
    return np.array(
        [
            [x, y, z]
            for x in (vmin[0], vmax[0])
            for y in (vmin[1], vmax[1])
            for z in (vmin[2], vmax[2])
        ],
        dtype=float,
    )


def _bbox_projection_range(
    vmin: np.ndarray,
    vmax: np.ndarray,
    axis: np.ndarray,
) -> tuple[float, float]:
    """Return the bounding-box projection range along an axis.

    Args:
        vmin: Minimum bounding-box coordinates.
        vmax: Maximum bounding-box coordinates.
        axis: Axis onto which the corners are projected.

    Returns:
        Minimum and maximum scalar projections.
    """
    projections = _bbox_range_corners(vmin, vmax) @ axis
    return float(np.min(projections)), float(np.max(projections))


def _support_axis_and_projection(
    vmin: np.ndarray,
    vmax: np.ndarray,
    normal_axis: np.ndarray,
    reference_points: list[np.ndarray],
) -> tuple[np.ndarray | None, float | None]:
    """Find and orient the support face nearest solder-interface points.

    Args:
        vmin: Minimum bounding-box coordinates.
        vmax: Maximum bounding-box coordinates.
        normal_axis: Candidate component normal.
        reference_points: Solder-interface points in component-local space.

    Returns:
        Oriented normal and support-plane projection, or two None values when
        the support face cannot be determined.
    """
    normal = _normalized(normal_axis)
    if normal is None or not reference_points:
        return None, None

    min_proj, max_proj = _bbox_projection_range(vmin, vmax, normal)
    ref_proj = float(np.mean([float(np.dot(p, normal)) for p in reference_points]))
    support_proj = (
        min_proj if abs(ref_proj - min_proj) <= abs(ref_proj - max_proj) else max_proj
    )

    # Orient the normal from the support face through the component body.
    center_proj = float(np.dot((vmin + vmax) / 2.0, normal))
    if center_proj < support_proj:
        normal = -normal
        support_proj = -support_proj

    return normal, support_proj


def _project_point_to_plane(
    point: np.ndarray,
    plane_normal: np.ndarray,
    plane_projection: float,
) -> np.ndarray:
    """Move a point along a plane normal onto a projected plane.

    Args:
        point: Point to project.
        plane_normal: Unit plane normal.
        plane_projection: Scalar projection identifying the plane.

    Returns:
        Projected point.
    """
    projection_delta = plane_projection - float(np.dot(point, plane_normal))
    return point + plane_normal * projection_delta


def _project_point_to_support_end(
    point: np.ndarray,
    vmin: np.ndarray,
    vmax: np.ndarray,
    support_normal: np.ndarray,
    support_projection: float,
    long_axis: np.ndarray,
) -> np.ndarray:
    """Project a solder point onto the support face and nearest body end.

    Args:
        point: Solder-interface point in component-local space.
        vmin: Minimum bounding-box coordinates.
        vmax: Maximum bounding-box coordinates.
        support_normal: Oriented component support normal.
        support_projection: Scalar projection identifying the support plane.
        long_axis: Component length direction.

    Returns:
        Point on the nearest support-plane body end.
    """
    projected = _project_point_to_plane(point, support_normal, support_projection)
    long_norm = _normalized(long_axis)
    if long_norm is None:
        return projected

    min_long, max_long = _bbox_projection_range(vmin, vmax, long_norm)
    point_long = float(np.dot(projected, long_norm))
    end_long = (
        min_long
        if abs(point_long - min_long) <= abs(point_long - max_long)
        else max_long
    )
    return projected + long_norm * (end_long - point_long)


def _row_major_axis_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Build a 4x4 row-vector Rodrigues rotation matrix.

    Args:
        axis: Rotation axis.
        angle_rad: Rotation angle in radians.

    Returns:
        Homogeneous row-vector rotation matrix.
    """
    rot_axis = _normalized(axis)
    if rot_axis is None:
        return np.eye(4)

    c_r, s_r = np.cos(angle_rad), np.sin(angle_rad)
    t_r = 1.0 - c_r
    rx, ry, rz = rot_axis

    rotation = np.eye(4)
    rotation[0, 0] = t_r * rx * rx + c_r
    rotation[0, 1] = t_r * rx * ry + s_r * rz
    rotation[0, 2] = t_r * rx * rz - s_r * ry
    rotation[1, 0] = t_r * rx * ry - s_r * rz
    rotation[1, 1] = t_r * ry * ry + c_r
    rotation[1, 2] = t_r * ry * rz + s_r * rx
    rotation[2, 0] = t_r * rx * rz + s_r * ry
    rotation[2, 1] = t_r * ry * rz - s_r * rx
    rotation[2, 2] = t_r * rz * rz + c_r
    return rotation


def _min_support_projection_after_rotation(
    vmin: np.ndarray,
    vmax: np.ndarray,
    pivot_local: np.ndarray,
    rot_axis: np.ndarray,
    angle_rad: float,
    support_normal: np.ndarray,
) -> float:
    """Return the lowest rotated bbox projection along the support normal.

    Args:
        vmin: Minimum bounding-box coordinates.
        vmax: Maximum bounding-box coordinates.
        pivot_local: Rotation pivot in component-local space.
        rot_axis: Rotation axis in component-local space.
        angle_rad: Rotation angle in radians.
        support_normal: Oriented support normal.

    Returns:
        Minimum rotated-corner projection.
    """
    corners = _bbox_range_corners(vmin, vmax)
    rotation = _row_major_axis_rotation(rot_axis, angle_rad)[:3, :3]
    rotated = (corners - pivot_local) @ rotation + pivot_local
    return float(np.min(rotated @ support_normal))


def _world_bbox_points_in_local(
    bbox_cache: object,
    prim: object,
    M_orig: np.ndarray,
) -> np.ndarray:
    """Transform a prim's world bbox corners into component-local space.

    Args:
        bbox_cache: USD bounding-box cache.
        prim: USD prim whose world bound is measured.
        M_orig: Component transform in row-major form.

    Returns:
        World bounding-box corners expressed in component-local space.
    """
    bbox_range = bbox_cache.ComputeWorldBound(prim).GetRange()
    world_corners = _bbox_range_corners(
        np.array(bbox_range.GetMin()),
        np.array(bbox_range.GetMax()),
    )
    try:
        matrix_inverse = np.linalg.inv(M_orig)
    except np.linalg.LinAlgError:
        matrix_inverse = np.linalg.pinv(M_orig)
    return np.array(
        [
            (np.array([point[0], point[1], point[2], 1.0]) @ matrix_inverse)[:3]
            for point in world_corners
        ],
        dtype=float,
    )


def _paste_contact_geometry(
    stage: object,
    bbox_cache: object,
    M_orig: np.ndarray,
    paste_prim_paths: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return two paste centroids plus aggregate local half extents.

    Args:
        stage: Open USD stage containing the paste prims.
        bbox_cache: USD bounding-box cache.
        M_orig: Component transform in row-major form.
        paste_prim_paths: Candidate solder-paste prim paths.

    Returns:
        Two centroids and aggregate half extents, or None unless exactly two
        valid paste prims are available.
    """
    if len(paste_prim_paths) != 2:
        return None

    centers: list[np.ndarray] = []
    all_points: list[np.ndarray] = []
    for prim_path in paste_prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        points = _world_bbox_points_in_local(bbox_cache, prim, M_orig)
        pmin = np.min(points, axis=0)
        pmax = np.max(points, axis=0)
        centers.append((pmin + pmax) / 2.0)
        all_points.extend(points)

    aggregate = np.array(all_points)
    half = (np.max(aggregate, axis=0) - np.min(aggregate, axis=0)) / 2.0
    return centers[0], centers[1], half


def _contact_centroid_local(bbox_cache: object, prim: object) -> np.ndarray:
    """Return a solder-contact centroid in component/prototype local space.

    Args:
        bbox_cache: USD bounding-box cache.
        prim: Solder-contact prim or instance proxy.

    Returns:
        Contact centroid in component/prototype local space.
    """
    bbox_range = bbox_cache.ComputeLocalBound(prim).GetRange()
    center = (
        np.array(bbox_range.GetMin(), dtype=float)
        + np.array(bbox_range.GetMax(), dtype=float)
    ) / 2.0

    # Instance-proxy contact geometry must be placed in prototype local space.
    xform_attr = prim.GetAttribute("xformOp:transform")
    if xform_attr and xform_attr.IsValid():
        xform = xform_attr.Get()
        if xform is not None:
            matrix = np.array([[float(v) for v in row] for row in xform])
            return (np.append(center, 1.0) @ matrix)[:3]
    return center


def _component_rotation_frame(
    prim: object,
    M_orig: np.ndarray,
    component_instance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer component length, width, board normal, and support geometry.

    Prefer solder-paste geometry, then solder-contact geometry, and finally a
    local bounding-box fallback.

    Args:
        prim: Component USD prim.
        M_orig: Component transform in row-major form.
        component_instance: Optional component metadata with solder prim paths.

    Returns:
        Component-local axes, bounds, support geometry, and source diagnostics.
    """
    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    bbox_range = bbox_cache.ComputeLocalBound(prim).GetRange()
    vmin = np.array(bbox_range.GetMin(), dtype=float)
    vmax = np.array(bbox_range.GetMax(), dtype=float)
    center = (vmin + vmax) / 2.0
    half = (vmax - vmin) / 2.0

    source = "bbox"
    reference_points: list[np.ndarray] = []
    long_axis = None
    width_axis = None
    normal_axis = None
    paste_paths: list[str] = []
    contact_paths: list[str] = []

    if component_instance is not None:
        stage = omni.usd.get_context().get_stage()
        paste_paths = list(component_instance.get("solder_paste_prims") or [])
        contact_paths = list(component_instance.get("solder_contact_prims") or [])

        paste_geometry = _paste_contact_geometry(stage, bbox_cache, M_orig, paste_paths)
        if paste_geometry is not None:
            point_0, point_1, paste_half = paste_geometry
            long_axis = _normalized(point_1 - point_0)
            if long_axis is not None:
                width_axis, normal_axis = _bbox_wide_and_normal_axes(
                    paste_half, long_axis
                )
                reference_points = [point_0, point_1]
                source = "paste"

        if width_axis is None and len(contact_paths) == 2:
            contacts = [stage.GetPrimAtPath(path) for path in contact_paths]
            if all(contact.IsValid() for contact in contacts):
                point_0, point_1 = (
                    _contact_centroid_local(bbox_cache, contact) for contact in contacts
                )
                long_axis = _normalized(point_1 - point_0)
                if long_axis is not None:
                    width_axis, normal_axis = _bbox_wide_and_normal_axes(
                        half, long_axis
                    )
                    reference_points = [point_0, point_1]
                    source = "contact"

    if long_axis is None or width_axis is None or normal_axis is None:
        long_axis_index = int(np.argmax(half))
        long_axis = np.eye(3)[long_axis_index]
        width_axis, normal_axis = _bbox_wide_and_normal_axes(half, long_axis)
        if width_axis is None or normal_axis is None:
            remaining = [idx for idx in range(3) if idx != long_axis_index]
            width_axis = np.eye(3)[remaining[0]]
            normal_axis = np.eye(3)[remaining[1]]
        reference_points = []
        source = "bbox"

    support_projection = None
    if reference_points:
        aligned_normal, support_projection = _support_axis_and_projection(
            vmin, vmax, normal_axis, reference_points
        )
        if aligned_normal is not None:
            normal_axis = aligned_normal

    if support_projection is None:
        support_projection, _ = _bbox_projection_range(vmin, vmax, normal_axis)

    # Keep a right-handed frame after orienting the board normal.
    aligned_width = _normalized(np.cross(normal_axis, long_axis))
    if aligned_width is not None:
        width_axis = aligned_width

    return {
        "vmin": vmin,
        "vmax": vmax,
        "center": center,
        "half": half,
        "long_axis": long_axis,
        "width_axis": width_axis,
        "board_normal_local": normal_axis,
        "support_projection": float(support_projection),
        "axis_source": source,
        "reference_points": reference_points,
        "paste_prim_paths": paste_paths,
        "contact_prim_paths": contact_paths,
    }


def _pivot_rotation_matrix(
    pivot_local: np.ndarray,
    axis_local: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Return a 4x4 row-vector rotation about a component-local pivot.

    Args:
        pivot_local: Rotation pivot in component-local space.
        axis_local: Rotation axis in component-local space.
        angle_deg: Rotation angle in degrees.

    Returns:
        Homogeneous row-vector pivot-rotation matrix (component-local).
    """
    rotation = _row_major_axis_rotation(axis_local, np.radians(angle_deg))
    translate_to_origin = np.eye(4)
    translate_to_origin[3, :3] = -pivot_local
    translate_back = np.eye(4)
    translate_back[3, :3] = pivot_local
    return translate_to_origin @ rotation @ translate_back


def _pivot_rotation_transform(
    M_orig: np.ndarray,
    pivot_local: np.ndarray,
    axis_local: np.ndarray,
    angle_deg: float,
) -> list[float]:
    """Return a flattened row-major local pivot rotation.

    Args:
        M_orig: Original component transform in row-major form.
        pivot_local: Rotation pivot in component-local space.
        axis_local: Rotation axis in component-local space.
        angle_deg: Rotation angle in degrees.

    Returns:
        Flattened absolute transform in row-major order.
    """
    local = _pivot_rotation_matrix(pivot_local, axis_local, angle_deg)
    return (local @ M_orig).flatten().tolist()
