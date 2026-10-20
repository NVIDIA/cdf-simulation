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

"""Prepare numeric THT pin and pad inputs from a live USD stage."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pcb_aoi.geometry.planar import convex_hull_xy


def _aligned_bounds(
    stage: object,
    prim_path: str,
    cache: object,
) -> tuple[float, ...] | None:
    """Return six-value aligned world bounds for one prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if bounds.IsEmpty():
        return None
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(minimum[2]),
        float(maximum[0]),
        float(maximum[1]),
        float(maximum[2]),
    )


def _face_components(
    point_count: int,
    face_counts: Sequence[int],
    face_indices: Sequence[int],
) -> list[list[int]]:
    """Return connected vertex-index groups from mesh face topology."""
    if point_count < 1:
        return []
    parent = list(range(point_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    offset = 0
    for count in face_counts:
        face = face_indices[offset : offset + int(count)]
        offset += int(count)
        if not face:
            continue
        for index in face[1:]:
            union(int(face[0]), int(index))

    groups: dict[int, list[int]] = {}
    for index in range(point_count):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def _point_bounds(
    points: Sequence[object],
    indices: Sequence[int],
) -> tuple[float, ...]:
    """Return aligned bounds for selected points."""
    selected = [points[index] for index in indices]
    return (
        min(float(point[0]) for point in selected),
        min(float(point[1]) for point in selected),
        min(float(point[2]) for point in selected),
        max(float(point[0]) for point in selected),
        max(float(point[1]) for point in selected),
        max(float(point[2]) for point in selected),
    )


def _merge_overlapping_xy(
    bounds: Sequence[tuple[float, ...]],
    tolerance: float,
) -> tuple[tuple[float, ...], ...]:
    """Merge disconnected mesh shells occupying one XY pin footprint."""
    groups = [list(item) for item in bounds]
    changed = True
    while changed:
        changed = False
        for first_index, first in enumerate(groups):
            for second_index in range(first_index + 1, len(groups)):
                second = groups[second_index]
                x_gap = max(0.0, first[0] - second[3], second[0] - first[3])
                y_gap = max(0.0, first[1] - second[4], second[1] - first[4])
                if x_gap > tolerance or y_gap > tolerance:
                    continue
                groups[first_index] = [
                    min(first[0], second[0]),
                    min(first[1], second[1]),
                    min(first[2], second[2]),
                    max(first[3], second[3]),
                    max(first[4], second[4]),
                    max(first[5], second[5]),
                ]
                groups.pop(second_index)
                changed = True
                break
            if changed:
                break
    return tuple(tuple(item) for item in groups)


def _mesh_world_topology(
    stage: object,
    prim_path: str,
) -> tuple[tuple[object, ...], tuple[int, ...], tuple[int, ...]] | None:
    """Return world points and face topology for one mesh prim."""
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        return None
    mesh = UsdGeom.Mesh(prim)
    local_points = tuple(mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or ())
    if not local_points:
        return None
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    return (
        tuple(matrix.Transform(Gf.Vec3d(point)) for point in local_points),
        tuple(int(value) for value in (mesh.GetFaceVertexCountsAttr().Get() or ())),
        tuple(int(value) for value in (mesh.GetFaceVertexIndicesAttr().Get() or ())),
    )


def _bounded_pad_contour(
    stage: object,
    pad_prim_path: str,
) -> tuple[tuple[float, float], ...]:
    """Return a pad's XY contour from its bounds when it carries no points.

    A pad is usually an exported mesh, but it does not have to be: a board
    authored by hand, or a fixture, may use `UsdGeom.Cube` or any other
    boundable gprim, which has an extent and no points at all. Reading the
    world bounds gives the same rectangle the mesh would have given, so the
    caller does not have to care which one it was handed.

    Args:
        stage: Live USD stage containing the associated pad.
        pad_prim_path: Absolute prim path of that pad.

    Returns:
        The bounding rectangle in world XY, or an empty tuple when the prim is
        absent, not boundable, or empty.
    """
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(pad_prim_path)
    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Boundable):
        return ()
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=True,
    ).ComputeWorldBound(prim)
    box = bounds.ComputeAlignedRange()
    if box.IsEmpty():
        return ()
    low, high = box.GetMin(), box.GetMax()
    return convex_hull_xy(
        (
            (float(low[0]), float(low[1])),
            (float(high[0]), float(low[1])),
            (float(high[0]), float(high[1])),
            (float(low[0]), float(high[1])),
        ),
    )


def fallback_pad_contour_from_mesh(
    stage: object,
    pad_prim_path: str,
) -> tuple[tuple[float, float], ...]:
    """Recover a pad's outer XY contour from its USD mesh.

    This geometry fallback is used only when the metadata catalog does not
    provide ``pcba:board:padContour``. It derives the contour of the already
    associated pad and does not search for, replace, or repair that metadata
    association.

    Args:
        stage: Live USD stage containing the associated pad.
        pad_prim_path: Absolute prim path of that pad.

    Returns:
        Convex world-space XY contour, or an empty tuple when unavailable.

    Example:

    .. code-block:: python

        contour = fallback_pad_contour_from_mesh(stage, "/Board/Pads/P1")
        print(len(contour) >= 3)
        # True

    """
    topology = _mesh_world_topology(stage, pad_prim_path)
    if topology is not None:
        points, _face_counts, _face_indices = topology
        return convex_hull_xy(
            tuple((float(point[0]), float(point[1])) for point in points),
        )
    return _bounded_pad_contour(stage, pad_prim_path)


def _pad_shape_offsets(
    stage: object,
    pad_prim_path: str,
    raw_contour: object,
    pad_center: tuple[float, float],
    pad_edge_inset: float,
) -> tuple[tuple[float, float], ...]:
    """Return a convex pad boundary as offsets from its center."""
    contour_points: tuple[tuple[float, float], ...] = ()
    if isinstance(raw_contour, Sequence) and not isinstance(
        raw_contour,
        (str, bytes),
    ):
        parsed_points = []
        for point in raw_contour:
            if (
                isinstance(point, Sequence)
                and not isinstance(point, (str, bytes))
                and len(point) >= 2
            ):
                parsed_points.append((float(point[0]), float(point[1])))
        if len(parsed_points) >= 3:
            absolute_hull = convex_hull_xy(tuple(parsed_points))
            translated_hull = convex_hull_xy(
                tuple(
                    (pad_center[0] + point[0], pad_center[1] + point[1])
                    for point in parsed_points
                ),
            )

            def centroid_distance(points: Sequence[tuple[float, float]]) -> float:
                center_x = sum(point[0] for point in points) / len(points)
                center_y = sum(point[1] for point in points) / len(points)
                return math.hypot(center_x - pad_center[0], center_y - pad_center[1])

            contour_points = min(
                (absolute_hull, translated_hull),
                key=centroid_distance,
            )
    if not contour_points:
        contour_points = fallback_pad_contour_from_mesh(stage, pad_prim_path)

    offsets = []
    for point in contour_points:
        offset_x = point[0] - pad_center[0]
        offset_y = point[1] - pad_center[1]
        radius = math.hypot(offset_x, offset_y)
        if radius <= pad_edge_inset:
            continue
        scale = (radius - pad_edge_inset) / radius
        offsets.append((offset_x * scale, offset_y * scale))
    return convex_hull_xy(tuple(offsets))


def fallback_connected_pin_cross_section_hull(
    points_xy: Sequence[tuple[float, float]],
    focus_center: tuple[float, float],
    focus_radius: float,
) -> tuple[tuple[float, float], ...]:
    """Localize one connected multi-pin mesh to the current pad.

    This geometry-containment fallback is invoked only after the normal
    cross-section exceeds its associated pad neighborhood. It does not repair
    or reinterpret incorrect pin-to-pad metadata.

    Args:
        points_xy: Cross-section points from one connected mesh region.
        focus_center: Center of the associated solder pad.
        focus_radius: Maximum local distance retained around the pad.

    Returns:
        Local convex hull, or an empty tuple when isolation is impossible.

    Example:

    .. code-block:: python

        hull = fallback_connected_pin_cross_section_hull(
            ((-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)),
            (0.0, 0.0),
            0.7,
        )
        print(len(hull))
        # 4

    """
    local_points = tuple(
        point
        for point in points_xy
        if math.hypot(
            point[0] - focus_center[0],
            point[1] - focus_center[1],
        )
        <= focus_radius
    )
    if len(local_points) < 3:
        return ()
    return convex_hull_xy(local_points)


def fallback_pad_centered_throat(
    pad_center: tuple[float, float],
    pin_radius: float,
) -> tuple[
    tuple[float, float],
    float,
    tuple[tuple[float, float], ...],
]:
    """Return a conservative throat when contact geometry cannot be isolated.

    This final containment fallback prevents a joint from bridging neighboring
    pads. It deliberately does not infer or correct metadata associations.

    Args:
        pad_center: Center of the associated solder pad.
        pin_radius: Conservative pin radius inferred from contact geometry.

    Returns:
        Circular pad-centered throat description.

    Example:

    .. code-block:: python

        throat = fallback_pad_centered_throat((1.0, 2.0), 0.2)
        print(throat[0])
        # (1.0, 2.0)

    """
    return (pad_center, pin_radius, ())


def _pin_cross_sections_at_z(
    stage: object,
    contact_prim_path: str,
    target_z: float,
    scene_units_per_mm: float,
    *,
    focus_center: tuple[float, float] | None = None,
    focus_radius: float | None = None,
) -> tuple[
    tuple[
        tuple[float, float],
        float,
        tuple[tuple[float, float], ...],
    ],
    ...,
]:
    """Return pin cross-sections at a world-space Z plane.

    A focus region localizes a connected multi-pin contact mesh to the pad
    currently being prepared. Normal isolated contacts take the unchanged fast
    path because their complete cross-section already fits inside that region.
    """
    topology = _mesh_world_topology(stage, contact_prim_path)
    if topology is None:
        return ()
    world_points, face_counts, face_indices = topology

    node_xy: list[tuple[float, float]] = []
    node_by_key: dict[tuple[float, float], int] = {}
    adjacency: dict[int, set[int]] = {}
    epsilon = 1e-6

    def node_index(x_value: float, y_value: float) -> int:
        key = (round(float(x_value), 7), round(float(y_value), 7))
        index = node_by_key.get(key)
        if index is None:
            index = len(node_xy)
            node_xy.append((float(x_value), float(y_value)))
            node_by_key[key] = index
            adjacency[index] = set()
        return index

    def connect(first: int, second: int) -> None:
        if first == second:
            return
        adjacency[first].add(second)
        adjacency[second].add(first)

    offset = 0
    for count in face_counts:
        face = face_indices[offset : offset + count]
        offset += count
        if len(face) < 2:
            continue
        hits = []
        for edge_index, current_index in enumerate(face):
            previous_index = face[edge_index - 1]
            previous = world_points[previous_index]
            current = world_points[current_index]
            previous_distance = float(previous[2]) - target_z
            current_distance = float(current[2]) - target_z
            if abs(previous_distance) <= epsilon and abs(current_distance) <= epsilon:
                connect(
                    node_index(float(previous[0]), float(previous[1])),
                    node_index(float(current[0]), float(current[1])),
                )
                continue
            if abs(previous_distance) <= epsilon:
                hits.append(node_index(float(previous[0]), float(previous[1])))
            if previous_distance * current_distance < 0.0:
                fraction = previous_distance / (previous_distance - current_distance)
                hits.append(
                    node_index(
                        float(previous[0])
                        + (float(current[0]) - float(previous[0])) * fraction,
                        float(previous[1])
                        + (float(current[1]) - float(previous[1])) * fraction,
                    ),
                )
            elif abs(current_distance) <= epsilon:
                hits.append(node_index(float(current[0]), float(current[1])))
        unique_hits = tuple(dict.fromkeys(hits))
        if len(unique_hits) == 2:
            connect(unique_hits[0], unique_hits[1])

    sections = []
    unseen = set(adjacency)
    while unseen:
        start = unseen.pop()
        stack = [start]
        indices = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in unseen:
                    continue
                unseen.remove(neighbor)
                stack.append(neighbor)
                indices.append(neighbor)
        component_points = tuple(node_xy[index] for index in indices)
        hull = convex_hull_xy(component_points)
        if (
            focus_center is not None
            and focus_radius is not None
            and any(
                math.hypot(
                    point[0] - focus_center[0],
                    point[1] - focus_center[1],
                )
                > focus_radius
                for point in hull
            )
        ):
            hull = fallback_connected_pin_cross_section_hull(
                component_points,
                focus_center,
                focus_radius,
            )
        if len(hull) < 3:
            continue
        minimum_x = min(point[0] for point in hull)
        minimum_y = min(point[1] for point in hull)
        maximum_x = max(point[0] for point in hull)
        maximum_y = max(point[1] for point in hull)
        center_xy = (
            (minimum_x + maximum_x) * 0.5,
            (minimum_y + maximum_y) * 0.5,
        )
        width = maximum_x - minimum_x
        height = maximum_y - minimum_y
        pin_radius = min(width, height) * 0.5
        if pin_radius <= 1e-6:
            continue
        offsets = tuple(
            (point[0] - center_xy[0], point[1] - center_xy[1]) for point in hull
        )
        radii = tuple(math.hypot(*point) for point in offsets)
        radial_band = max(radii) - min(radii)
        roundish = len(hull) >= 8 and radial_band <= max(
            0.03 * scene_units_per_mm,
            pin_radius * 0.18,
        )
        aspect_ratio = max(width, height) / max(min(width, height), 1e-6)
        radius_ratio = max(radii) / pin_radius
        compact = aspect_ratio <= 2.5 and radius_ratio <= 2.75
        sections.append(
            (
                center_xy,
                pin_radius,
                offsets if not roundish and compact else (),
            ),
        )
    return tuple(sections)


def _cross_section_fits_pad(
    section: tuple[
        tuple[float, float],
        float,
        tuple[tuple[float, float], ...],
    ],
    pad_center: tuple[float, float],
    usable_pad_radius: float,
) -> bool:
    """Return whether a pin throat remains inside its associated pad."""
    throat_center, pin_radius, throat_shape = section
    if throat_shape:
        return all(
            math.hypot(
                throat_center[0] + offset[0] - pad_center[0],
                throat_center[1] + offset[1] - pad_center[1],
            )
            <= usable_pad_radius
            for offset in throat_shape
        )
    return (
        math.hypot(
            throat_center[0] - pad_center[0],
            throat_center[1] - pad_center[1],
        )
        + pin_radius
        <= usable_pad_radius
    )


def _contact_region_bounds(
    stage: object,
    contact_prim_path: str,
    pin_center: tuple[float, float],
    scene_units_per_mm: float,
) -> tuple[float, ...] | None:
    """Return the disconnected contact region nearest a declared pin center."""
    topology = _mesh_world_topology(stage, contact_prim_path)
    if topology is None:
        return None
    points, face_counts, face_indices = topology
    components = _face_components(
        len(points),
        face_counts,
        face_indices,
    )
    bounds = tuple(_point_bounds(points, indices) for indices in components)
    merged = _merge_overlapping_xy(
        bounds,
        tolerance=0.02 * scene_units_per_mm,
    )
    if not merged:
        return None
    return min(
        merged,
        key=lambda item: math.hypot(
            pin_center[0] - (item[0] + item[3]) * 0.5,
            pin_center[1] - (item[1] + item[4]) * 0.5,
        ),
    )


def _contact_region_mesh(
    stage: object,
    contact_prim_path: str,
    pin_center: tuple[float, float],
    scene_units_per_mm: float,
) -> dict[str, tuple[object, ...]] | None:
    """Return mesh topology for the disconnected contact region nearest a pin."""
    topology = _mesh_world_topology(stage, contact_prim_path)
    if topology is None:
        return None
    points, face_counts, face_indices = topology
    components = _face_components(len(points), face_counts, face_indices)
    if not components:
        return None
    component_bounds = tuple(_point_bounds(points, indices) for indices in components)
    tolerance = 0.02 * scene_units_per_mm
    target_bounds = min(
        _merge_overlapping_xy(component_bounds, tolerance),
        key=lambda item: math.hypot(
            pin_center[0] - (item[0] + item[3]) * 0.5,
            pin_center[1] - (item[1] + item[4]) * 0.5,
        ),
    )
    selected = {
        index
        for indices, bounds in zip(components, component_bounds, strict=True)
        if max(0.0, bounds[0] - target_bounds[3], target_bounds[0] - bounds[3])
        <= tolerance
        and max(0.0, bounds[1] - target_bounds[4], target_bounds[1] - bounds[4])
        <= tolerance
        for index in indices
    }
    if not selected:
        return None
    old_to_new = {old: new for new, old in enumerate(sorted(selected))}
    selected_counts = []
    selected_indices = []
    offset = 0
    for count in face_counts:
        face = face_indices[offset : offset + count]
        offset += count
        if not face or not all(index in selected for index in face):
            continue
        selected_counts.append(count)
        selected_indices.extend(old_to_new[index] for index in face)
    if not selected_counts:
        return None
    return {
        "points": tuple(
            tuple(float(value) for value in points[index]) for index in sorted(selected)
        ),
        "face_vertex_counts": tuple(selected_counts),
        "face_vertex_indices": tuple(selected_indices),
    }


def prepare_through_hole_site(
    stage: object,
    association: Mapping[str, object],
    side: str,
    scene_units_per_mm: float,
    cache: object,
    *,
    height_ratio: float = 0.9,
    z_offset: float = 0.0,
    pad_edge_inset: float = 0.0,
    solder_bath: bool = False,
) -> dict[str, object] | None:
    """Prepare one serializable occupied through-hole site.

    Args:
        stage: Live USD stage used only during solver preparation.
        association: Catalog pin-to-pad association record.
        side: Component insertion side.
        scene_units_per_mm: Conversion factor from millimeters to scene units.
        cache: Shared USD bounding-box cache.
        height_ratio: Crest height relative to exposed annulus width.
        z_offset: Placement offset along the outward board normal.
        pad_edge_inset: Radial clearance between solder and the pad edge.
        solder_bath: Whether to capture the pin mesh for the primary solder-bath
            behavior of enveloping the exposed solder-side pin.

    Returns:
        Serializable numeric site data, or None for invalid source geometry.

    Example:

    .. code-block:: python

        site = prepare_through_hole_site(
            stage,
            association,
            "Top",
            1.0,
            bbox_cache,
        )
        print(site is not None)

    """
    contact_prim = str(association.get("solder_contact_prim") or "")
    pad_prim = str(association.get("solder_pad_prim") or "")
    hole_prim = str(association.get("solder_hole_prim") or "")
    raw_center = association.get("pin_center")
    if (
        not contact_prim
        or not pad_prim
        or not isinstance(raw_center, (list, tuple))
        or len(raw_center) != 2
    ):
        return None
    pad_center = (float(raw_center[0]), float(raw_center[1]))
    contact_bounds = _contact_region_bounds(
        stage,
        contact_prim,
        pad_center,
        scene_units_per_mm,
    )
    pad_bounds = _aligned_bounds(stage, pad_prim, cache)
    if contact_bounds is None or pad_bounds is None:
        return None
    pad_shape_xy = _pad_shape_offsets(
        stage,
        pad_prim,
        association.get("pad_contour"),
        pad_center,
        pad_edge_inset,
    )

    throat_center = (
        (contact_bounds[0] + contact_bounds[3]) * 0.5,
        (contact_bounds[1] + contact_bounds[4]) * 0.5,
    )
    pin_width = min(
        contact_bounds[3] - contact_bounds[0],
        contact_bounds[4] - contact_bounds[1],
    )
    pad_width = min(
        pad_bounds[3] - pad_bounds[0],
        pad_bounds[4] - pad_bounds[1],
    )
    pin_radius = pin_width * 0.5
    maximum_pad_radius = pad_width * 0.5
    minimum_annulus = 0.10 * scene_units_per_mm
    solder_pad_radius = maximum_pad_radius - pad_edge_inset
    if solder_pad_radius <= 0.0:
        return None
    hole_bounds = _aligned_bounds(stage, hole_prim, cache) if hole_prim else None
    if hole_bounds is not None:
        hole_width = min(
            hole_bounds[3] - hole_bounds[0],
            hole_bounds[4] - hole_bounds[1],
        )
        maximum_pin_radius = hole_width * 0.5 - minimum_annulus
        if maximum_pin_radius > 0.0:
            pin_radius = min(pin_radius, maximum_pin_radius)
    available_annulus = solder_pad_radius - pin_radius
    if pin_radius <= 0.0 or available_annulus < minimum_annulus:
        return None
    pad_radius = solder_pad_radius
    grow_negative = side.lower().startswith("t")
    direction = -1.0 if grow_negative else 1.0
    base_z = pad_bounds[2] if grow_negative else pad_bounds[5]
    throat_z = base_z + direction * (
        z_offset + (pad_radius - pin_radius) * height_ratio
    )
    cross_sections = _pin_cross_sections_at_z(
        stage,
        contact_prim,
        throat_z,
        scene_units_per_mm,
        focus_center=pad_center,
        focus_radius=max(0.0, solder_pad_radius - minimum_annulus),
    )
    throat_shape: tuple[tuple[float, float], ...] = ()
    if cross_sections:
        nearest_section = min(
            cross_sections,
            key=lambda item: math.hypot(
                item[0][0] - pad_center[0],
                item[0][1] - pad_center[1],
            ),
        )
        if _cross_section_fits_pad(
            nearest_section,
            pad_center,
            max(0.0, solder_pad_radius - minimum_annulus),
        ):
            throat_center, pin_radius, throat_shape = nearest_section
        else:
            throat_center, pin_radius, throat_shape = fallback_pad_centered_throat(
                pad_center,
                pin_radius,
            )
    elif not _cross_section_fits_pad(
        (throat_center, pin_radius, ()),
        pad_center,
        max(0.0, solder_pad_radius - minimum_annulus),
    ):
        throat_center, pin_radius, throat_shape = fallback_pad_centered_throat(
            pad_center,
            pin_radius,
        )
    if hole_bounds is not None:
        hole_width = min(
            hole_bounds[3] - hole_bounds[0],
            hole_bounds[4] - hole_bounds[1],
        )
        maximum_pin_radius = hole_width * 0.5 - minimum_annulus
        if maximum_pin_radius > 0.0:
            pin_radius = min(pin_radius, maximum_pin_radius)
    available_annulus = solder_pad_radius - pin_radius
    if pin_radius <= 0.0 or available_annulus < minimum_annulus:
        return None
    pad_radius = solder_pad_radius
    site = {
        "center_xy": pad_center,
        "pad_shape_xy": pad_shape_xy,
        "throat_center_xy": throat_center,
        "throat_shape_xy": throat_shape,
        "pin_radius": pin_radius,
        "pad_radius": pad_radius,
        "base_z": base_z,
        "grow_toward_negative_z": grow_negative,
        "solder_contact_prim": contact_prim,
        "solder_pad_prim": pad_prim,
    }
    if solder_bath:
        pin_mesh = _contact_region_mesh(
            stage,
            contact_prim,
            pad_center,
            scene_units_per_mm,
        )
        if pin_mesh is not None:
            site["pin_points"] = pin_mesh["points"]
            site["pin_face_vertex_counts"] = pin_mesh["face_vertex_counts"]
            site["pin_face_vertex_indices"] = pin_mesh["face_vertex_indices"]
    return site
