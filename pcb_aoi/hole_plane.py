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

"""Runtime generation of a board-sized hole-reveal plane for segmentation."""

from __future__ import annotations


def _substrate_xy_bounds(
    stage: object,
    board_root: str,
) -> tuple[float, float, float, float]:
    """Return world-space (x_min, y_min, x_max, y_max) of board_substrate prims.

    Args:
        stage: Open USD stage.
        board_root: Absolute path of the board root prim to scan.

    Returns:
        (x_min, y_min, x_max, y_max) in world-space stage units.

    Raises:
        ValueError: If the board root is invalid or no board_substrate prims
            are found under it.

    """
    from pxr import UsdGeom

    from pcb_aoi.board_metadata.usd_schema.stage_traversal import iter_prim_subtree
    from pcb_aoi.board_metadata.usd_schema_metadata_keys import MESH_ROLE_ATTR
    from pcb_aoi.geometry.transform import transform_points

    board_prim = stage.GetPrimAtPath(board_root)
    if not board_prim.IsValid():
        raise ValueError(f"board root {board_root!r} is not a valid prim")

    # Use iter_prim_subtree (which handles abstract/class prims via
    # TraverseInstanceProxies) + XformCache vertex iteration rather than
    # BBoxCache, which returns empty for USD class prototype prims.
    xform_cache = UsdGeom.XformCache()
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    substrate_found = False

    for prim in iter_prim_subtree(board_prim):
        attr = prim.GetAttribute(MESH_ROLE_ATTR)
        if not attr or attr.Get() != "board_substrate":
            continue
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        world = transform_points(matrix, points)
        x_min = min(x_min, float(world[:, 0].min()))
        y_min = min(y_min, float(world[:, 1].min()))
        x_max = max(x_max, float(world[:, 0].max()))
        y_max = max(y_max, float(world[:, 1].max()))
        substrate_found = True

    if substrate_found:
        return x_min, y_min, x_max, y_max

    # Legacy boards without pcba:mesh:role metadata — scan all mesh geometry.
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    found = False
    for prim in iter_prim_subtree(board_prim):
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        # The legacy branch has no metadata to narrow it, so it walks every
        # mesh on the board -- the largest point count in this module.
        world = transform_points(matrix, points)
        x_min = min(x_min, float(world[:, 0].min()))
        y_min = min(y_min, float(world[:, 1].min()))
        x_max = max(x_max, float(world[:, 0].max()))
        y_max = max(y_max, float(world[:, 1].max()))
        found = True

    if not found:
        raise ValueError(
            f"no geometry found under {board_root!r}; cannot compute XY footprint "
            "for the hole-reveal plane"
        )
    return x_min, y_min, x_max, y_max


def create_hole_plane(
    stage: object,
    board_root: str,
    output_prim_path: str,
    substrate_z_range: tuple[float, float],
    *,
    side: str,
    z_offset_mm: float,
    meters_per_unit: float | None,
    class_name: str,
) -> str:
    """Create a board-sized quad mesh labeled for hole-reveal segmentation.

    The plane is placed at ``z_offset_mm`` inside the board substrate (FR4),
    measured from the ``side`` face of ``substrate_z_range``.

    The z reference is the ``board_substrate`` (FR4) face as measured by
    ``BoardRecord.substrate_z_range`` — not the solder-mask or paste layers.

    Substrate geometry naturally occludes the plane everywhere except at
    through-hole voids, so the segmentation annotator sees it only where
    drilled holes penetrate the board.

    The synthetic mesh has a magenta display color for stage inspection, but
    its parent Xform is hidden by default. Setting ``pcba:mesh:role =
    "board_hole"`` on the prim lets an annotation pass reveal it with
    ``capture.visibility(show=["hole"])``.

    Args:
        stage: Open USD stage to author on.
        board_root: Absolute path of the board root prim, used to scan
            ``board_substrate`` meshes for the XY footprint.
        output_prim_path: Absolute path at which to create the Xform parent.
        substrate_z_range: World-space ``(z_lo, z_hi)`` of ``board_substrate``
            meshes, as stored in ``BoardRecord.substrate_z_range``.
        side: ``"bottom"`` places the plane at ``z_lo + z_offset_stage``;
            ``"top"`` places it at ``z_hi - z_offset_stage``.
        z_offset_mm: Distance in millimetres to offset inward from the chosen
            substrate face.  Measured from the FR4 substrate face.
        meters_per_unit: Stage metres-per-unit for mm-to-stage-unit
            conversion.  Defaults to ``0.001`` (1 mm per stage unit) when
            ``None`` or zero.
        class_name: Semantic class name authored on the generated prim.

    Returns:
        Absolute prim path of the created child mesh.

    Raises:
        ValueError: If the board root is invalid or no board_substrate prims
            are found under it.

    """
    from pxr import Gf, Sdf, UsdGeom

    from pcb_aoi.board_metadata.usd_schema_metadata_keys import MESH_ROLE_ATTR
    from pcb_aoi.geometry_labels import (
        SEGMENTATION_MASK_DISPLAY_COLOR,
        ensure_segmentation_proxies_parent,
    )

    mpu = meters_per_unit if (meters_per_unit and meters_per_unit > 0) else 0.001
    z_offset_stage = (z_offset_mm / 1000.0) / mpu

    z_lo, z_hi = substrate_z_range
    z = z_lo + z_offset_stage if side == "bottom" else z_hi - z_offset_stage

    x_min, y_min, x_max, y_max = _substrate_xy_bounds(stage, board_root)

    points = [
        Gf.Vec3f(x_min, y_min, z),
        Gf.Vec3f(x_max, y_min, z),
        Gf.Vec3f(x_max, y_max, z),
        Gf.Vec3f(x_min, y_max, z),
    ]

    ensure_segmentation_proxies_parent(stage, output_prim_path)
    output_root = UsdGeom.Xform.Define(stage, output_prim_path)
    output_root.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    mesh_path = f"{output_prim_path.rstrip('/')}/plane"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*SEGMENTATION_MASK_DISPLAY_COLOR)])

    prim = mesh.GetPrim()
    prim.CreateAttribute(MESH_ROLE_ATTR, Sdf.ValueTypeNames.Token).Set("board_hole")
    prim.CreateAttribute(
        "semantic:Semantics:class:semanticData",
        Sdf.ValueTypeNames.String,
    ).Set(class_name)
    prim.CreateAttribute(
        "semantic:Semantics:class:semanticType",
        Sdf.ValueTypeNames.String,
    ).Set("class")

    return mesh_path


def create_hole_plane_at_position(
    stage: object,
    output_prim_path: str,
    source_prim_path: str,
    substrate_z_range: tuple[float, float],
    *,
    side: str,
    z_offset_mm: float,
    meters_per_unit: float | None,
    class_name: str,
) -> str | None:
    """Create a small quad plane at a specific prim's XY position.

    Used when component filtering is active: one magenta plane per selected
    pad prim, sized to the pad's world-space geometry, instead of a single
    board-wide plane. Returns ``None`` and skips creation when the prim is
    missing or carries no mesh geometry.

    Args:
        stage: Open USD stage.
        output_prim_path: Absolute path at which to create the mesh prim.
        source_prim_path: Absolute path of the source prim (typically a pad)
            whose XY bounds determine the plane's position and size.
        substrate_z_range: World-space ``(z_lo, z_hi)`` of board_substrate
            meshes, as stored in ``BoardRecord.substrate_z_range``.
        side: ``"bottom"`` or ``"top"``.
        z_offset_mm: Distance in millimetres to offset inward from the chosen
            substrate face.
        meters_per_unit: Stage metres-per-unit; defaults to ``0.001``.
        class_name: Semantic class name authored on the generated prim.

    Returns:
        Absolute prim path of the created mesh, or ``None`` if the source prim
        has no usable geometry.

    """
    from pxr import Gf, Sdf, UsdGeom

    from pcb_aoi.board_metadata.usd_schema_metadata_keys import MESH_ROLE_ATTR
    from pcb_aoi.geometry.transform import transform_points
    from pcb_aoi.geometry_labels import (
        SEGMENTATION_MASK_DISPLAY_COLOR,
        ensure_segmentation_proxies_parent,
    )

    prim = stage.GetPrimAtPath(source_prim_path)
    if not prim.IsValid():
        return None
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        return None
    points = mesh.GetPointsAttr().Get()
    if not points:
        return None

    xform_cache = UsdGeom.XformCache()
    matrix = xform_cache.GetLocalToWorldTransform(prim)
    world_pts = transform_points(matrix, points)
    x_min = float(world_pts[:, 0].min())
    x_max = float(world_pts[:, 0].max())
    y_min = float(world_pts[:, 1].min())
    y_max = float(world_pts[:, 1].max())

    mpu = meters_per_unit if (meters_per_unit and meters_per_unit > 0) else 0.001
    z_offset_stage = (z_offset_mm / 1000.0) / mpu
    z_lo, z_hi = substrate_z_range
    z = z_lo + z_offset_stage if side == "bottom" else z_hi - z_offset_stage

    quad_points = [
        Gf.Vec3f(x_min, y_min, z),
        Gf.Vec3f(x_max, y_min, z),
        Gf.Vec3f(x_max, y_max, z),
        Gf.Vec3f(x_min, y_max, z),
    ]

    parent_path = str(Sdf.Path(output_prim_path).GetParentPath())
    ensure_segmentation_proxies_parent(stage, output_prim_path)
    output_root = UsdGeom.Xform.Define(stage, parent_path)
    output_root.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    mesh_out = UsdGeom.Mesh.Define(stage, output_prim_path)
    mesh_out.CreatePointsAttr(quad_points)
    mesh_out.CreateFaceVertexCountsAttr([4])
    mesh_out.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh_out.CreateDoubleSidedAttr(True)
    mesh_out.CreateDisplayColorAttr([Gf.Vec3f(*SEGMENTATION_MASK_DISPLAY_COLOR)])

    out_prim = mesh_out.GetPrim()
    out_prim.CreateAttribute(MESH_ROLE_ATTR, Sdf.ValueTypeNames.Token).Set("board_hole")
    out_prim.CreateAttribute(
        "semantic:Semantics:class:semanticData",
        Sdf.ValueTypeNames.String,
    ).Set(class_name)
    out_prim.CreateAttribute(
        "semantic:Semantics:class:semanticType",
        Sdf.ValueTypeNames.String,
    ).Set("class")

    return output_prim_path
