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

"""Generate annotation-only proxies for through-hole lead tip surfaces."""

from __future__ import annotations

import math
from collections.abc import Sequence

Point3 = tuple[float, float, float]

#: Largest angle, in degrees, by which a face's outward normal may deviate from
#: the seed patch's normal and still be pulled into a tip selection. Seed faces
#: themselves are exempt, so a lead cut off at a steep angle still yields its
#: whole end face. The bound only constrains how far the selection may crawl
#: across a curved surface: on a lead whose end is rounded rather than flat,
#: the chamfer flanking the tip tilts away past this limit and stays part of
#: the lead (class ``pin``) instead of being absorbed into the tip proxy.
#: Without it, a finely tessellated formed lead can hand its entire surface to
#: the proxy, because every facet along the bend keeps some outward component.
MAX_TIP_NORMAL_DEVIATION_DEG = 30.0


def _face_normal(points: Sequence[Point3], face: Sequence[int]) -> Point3 | None:
    """Return one polygon's Newell unit normal, or None when degenerate."""
    normal = [0.0, 0.0, 0.0]
    for position, index in enumerate(face):
        first = points[index]
        second = points[face[(position + 1) % len(face)]]
        normal[0] += (first[1] - second[1]) * (first[2] + second[2])
        normal[1] += (first[2] - second[2]) * (first[0] + second[0])
        normal[2] += (first[0] - second[0]) * (first[1] + second[1])
    length = math.sqrt(sum(value * value for value in normal))
    if length < 1e-12:
        return None
    return tuple(value / length for value in normal)


def _mean_normal(
    records: Sequence[tuple[tuple[int, ...], Point3]],
    indices: set[int],
) -> Point3 | None:
    """Return the unit mean of the given faces' normals, or None if degenerate."""
    total = tuple(
        sum(records[index][1][coordinate] for index in indices)
        for coordinate in range(3)
    )
    length = math.sqrt(sum(value * value for value in total))
    if length < 1e-12:
        return None
    return tuple(value / length for value in total)


def select_pin_tip_faces(
    points: Sequence[Point3],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    *,
    direction: float,
) -> tuple[tuple[tuple[int, ...], ...], Point3] | None:
    """Select the connected polygon patch forming a lead's outward tip.

    A flat cap whose vertices all lie on the terminal's extreme plane takes
    precedence over adjacent chamfers. When no such cap exists, seed faces
    touch the terminal's extreme point and selection follows edge-connected
    faces whose outward normals retain a component toward that side. This
    fallback includes steeply angled and curved tips while stopping at vertical
    lead walls. Face winding does not matter because selected normals are
    oriented toward the requested side.

    Expansion beyond the seed faces is bounded by
    :data:`MAX_TIP_NORMAL_DEVIATION_DEG`, measured against the seed patch's own
    normal. A tessellated flat or slanted cut is unaffected because its facets
    share that normal, but a rounded tip stops at its chamfer rather than
    crawling down the lead.

    Args:
        points: Mesh points in world space.
        face_vertex_counts: Number of corners in each polygon.
        face_vertex_indices: Flattened polygon corner indices.
        direction: Positive for a top-side tip and negative for a bottom-side
            tip.

    Returns:
        Selected faces and representative outward unit normal, or None when no
        tip surface can be identified.

    Example:

    .. code-block:: python

        result = select_pin_tip_faces(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
            [4],
            [0, 1, 2, 3],
            direction=1.0,
        )
        print(result is not None)
        # True

    """
    if direction == 0.0 or not points or not face_vertex_counts:
        return None
    faces: list[tuple[int, ...]] = []
    cursor = 0
    for count in face_vertex_counts:
        face = tuple(face_vertex_indices[cursor : cursor + count])
        cursor += count
        if len(face) >= 3:
            faces.append(face)
    side_sign = 1.0 if direction > 0.0 else -1.0
    records: list[tuple[tuple[int, ...], Point3]] = []
    for face in faces:
        normal = _face_normal(points, face)
        if normal is None:
            continue
        if side_sign * normal[2] < 0.0:
            normal = tuple(-value for value in normal)
        records.append((face, normal))
    if not records:
        return None

    extent = math.sqrt(
        sum(
            (
                max(point[coordinate] for point in points)
                - min(point[coordinate] for point in points)
            )
            ** 2
            for coordinate in range(3)
        )
    )
    tolerance = max(1e-8, extent * 1e-7)
    extreme = max(side_sign * point[2] for point in points)
    candidate_indices = {
        index
        for index, (_face, normal) in enumerate(records)
        if side_sign * normal[2] > 1e-6
    }
    flat_cap_indices = {
        index
        for index in candidate_indices
        if all(
            extreme - side_sign * points[vertex][2] <= tolerance
            for vertex in records[index][0]
        )
    }
    if flat_cap_indices:
        candidate_indices = flat_cap_indices
        selected_indices = set(flat_cap_indices)
    else:
        selected_indices = {
            index
            for index in candidate_indices
            if any(
                extreme - side_sign * points[vertex][2] <= tolerance
                for vertex in records[index][0]
            )
        }
    if not selected_indices:
        return None

    seed_normal = _mean_normal(records, selected_indices)
    if seed_normal is None:
        return None
    deviation_limit = math.cos(math.radians(MAX_TIP_NORMAL_DEVIATION_DEG))

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for index in candidate_indices:
        face = records[index][0]
        for start, end in zip(face, face[1:] + face[:1], strict=True):
            edge = tuple(sorted((start, end)))
            edge_to_faces.setdefault(edge, []).append(index)
    pending = list(selected_indices)
    while pending:
        index = pending.pop()
        face = records[index][0]
        for start, end in zip(face, face[1:] + face[:1], strict=True):
            for neighbour in edge_to_faces.get(tuple(sorted((start, end))), ()):
                if neighbour in selected_indices:
                    continue
                normal = records[neighbour][1]
                alignment = sum(
                    normal[coordinate] * seed_normal[coordinate]
                    for coordinate in range(3)
                )
                if alignment < deviation_limit:
                    continue
                selected_indices.add(neighbour)
                pending.append(neighbour)

    selected = tuple(
        record[0] for index, record in enumerate(records) if index in selected_indices
    )
    outward = _mean_normal(records, selected_indices)
    if outward is None:
        return None
    return selected, outward


def create_pin_tip_proxy(
    stage: object,
    lead_prim_path: str,
    *,
    output_root_path: str,
    class_name: str,
    side: str,
    offset_mm: float,
    meters_per_unit: float | None,
) -> str | None:
    """Copy a terminal's outward tip surface into a semantic proxy mesh.

    The proxy is authored under a dedicated, default-hidden board-level output
    scope with a diagnostic magenta display color. Its vertices are displaced
    outward in world space before being transformed into that scope's coordinates.
    This gives the proxy stable depth precedence without changing the physical
    lead mesh or relying on render traversal of children beneath a geometric prim.
    An annotation pass reveals it with
    ``visibility.show: ["pin_tip"]``.

    Args:
        stage: Open USD stage to author on.
        lead_prim_path: Terminal mesh whose tip surface is copied.
        output_root_path: Board-level scope that receives generated proxies.
        class_name: Semantic class authored on the proxy.
        side: Board side carrying the visible tip.
        offset_mm: Outward displacement used to prevent z-fighting.
        meters_per_unit: Stage metres-per-unit used for unit conversion.

    Returns:
        Created proxy path, or None when the terminal has no identifiable tip
        surface.

    Raises:
        ValueError: If ``side`` is not top or bottom.

    Example:

    .. code-block:: python

        path = create_pin_tip_proxy(
            stage,
            "/World/Board/Components/J1/pkg_lead_1",
            output_root_path="/World/Board/SegmentationProxies/PinTips",
            class_name="pin_tip",
            side="bottom",
            offset_mm=0.002,
            meters_per_unit=0.001,
        )  # doctest: +SKIP
        print(path)  # doctest: +SKIP

    """
    from pxr import Gf, Sdf, Tf, UsdGeom

    from pcb_aoi.board_metadata.usd_schema_metadata_keys import MESH_ROLE_ATTR
    from pcb_aoi.geometry_labels import (
        SEGMENTATION_MASK_DISPLAY_COLOR,
        ensure_segmentation_proxies_parent,
    )

    if side not in ("top", "bottom"):
        raise ValueError("side must be 'top' or 'bottom'")
    lead_prim = stage.GetPrimAtPath(lead_prim_path)
    if not lead_prim or not lead_prim.IsValid() or not lead_prim.IsA(UsdGeom.Mesh):
        return None
    lead_mesh = UsdGeom.Mesh(lead_prim)
    local_points = lead_mesh.GetPointsAttr().Get()
    counts = lead_mesh.GetFaceVertexCountsAttr().Get()
    indices = lead_mesh.GetFaceVertexIndicesAttr().Get()
    if not local_points or not counts or not indices:
        return None

    cache = UsdGeom.XformCache()
    to_world = cache.GetLocalToWorldTransform(lead_prim)
    world_points = [
        tuple(float(value) for value in to_world.Transform(Gf.Vec3d(*point)))
        for point in local_points
    ]
    direction = 1.0 if side == "top" else -1.0
    selected = select_pin_tip_faces(
        world_points,
        tuple(int(value) for value in counts),
        tuple(int(value) for value in indices),
        direction=direction,
    )
    if selected is None:
        return None
    faces, normal = selected

    mpu = meters_per_unit if meters_per_unit and meters_per_unit > 0.0 else 0.001
    offset_stage = (offset_mm / 1000.0) / mpu
    ensure_segmentation_proxies_parent(stage, output_root_path)
    output_root_schema = UsdGeom.Xform.Define(stage, output_root_path)
    output_root_schema.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    output_root = output_root_schema.GetPrim()
    output_to_world = cache.GetLocalToWorldTransform(output_root)
    world_to_output = output_to_world.GetInverse()
    proxy_points: list[object] = []
    proxy_counts: list[int] = []
    proxy_indices: list[int] = []
    remapped: dict[int, int] = {}
    for face in faces:
        proxy_counts.append(len(face))
        for source_index in face:
            if source_index not in remapped:
                world = world_points[source_index]
                displaced = Gf.Vec3d(
                    *(world[axis] + offset_stage * normal[axis] for axis in range(3))
                )
                local = world_to_output.Transform(displaced)
                remapped[source_index] = len(proxy_points)
                proxy_points.append(Gf.Vec3f(*local))
            proxy_indices.append(remapped[source_index])

    component_marker = "/Components/"
    component_tail = lead_prim_path.split(component_marker, 1)[-1]
    component_ref, _, lead_name = component_tail.partition("/")
    component_ref = Tf.MakeValidIdentifier(component_ref or "unknown_component")
    lead_name = Tf.MakeValidIdentifier(lead_name.rsplit("/", 1)[-1] or "pin_tip")
    components_path = f"{output_root_path}/Components"
    component_path = f"{components_path}/{component_ref}"
    UsdGeom.Scope.Define(stage, components_path)
    UsdGeom.Scope.Define(stage, component_path)
    proxy_path = f"{component_path}/{lead_name}"
    proxy_mesh = UsdGeom.Mesh.Define(stage, proxy_path)
    proxy_mesh.CreatePointsAttr(proxy_points)
    proxy_mesh.CreateFaceVertexCountsAttr(proxy_counts)
    proxy_mesh.CreateFaceVertexIndicesAttr(proxy_indices)
    proxy_mesh.CreateDoubleSidedAttr(True)
    proxy_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    proxy_mesh.CreateDisplayColorAttr([Gf.Vec3f(*SEGMENTATION_MASK_DISPLAY_COLOR)])

    proxy_prim = proxy_mesh.GetPrim()
    proxy_prim.CreateAttribute(MESH_ROLE_ATTR, Sdf.ValueTypeNames.Token).Set(
        "board_pin_tip"
    )
    proxy_prim.CreateAttribute(
        "semantic:Semantics:class:semanticData",
        Sdf.ValueTypeNames.String,
    ).Set(class_name)
    proxy_prim.CreateAttribute(
        "semantic:Semantics:class:semanticType",
        Sdf.ValueTypeNames.String,
    ).Set("class")
    return proxy_path
