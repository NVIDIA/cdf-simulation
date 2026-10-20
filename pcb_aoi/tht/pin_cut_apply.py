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

"""Apply one trim plane to a lead and to the joint grown on it.

A lead is trimmed by a blade, and a blade does not stop at the boundary between
copper and solder. Doing this as a stage edit AFTER the joint exists means the
solder generator never has to know a cut is coming: it wraps the whole lead as
if untrimmed, and the blade then takes lead and joint together, leaving one
planar section across both.

That also removes a special case rather than adding one. Leaving the lead's cut
face bare while the bath wrapped around it existed only because the bath was
built before the trim; built after, there is nothing to leave bare.

The plane is placed the way `pin_variation` has always placed it: anchored so
its DEEPEST point sits at the requested protrusion, because anchoring the
centroid reports a protrusion no part of the lead actually has.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pcb_aoi.geometry.plane_cut import (
    close_open_boundaries,
    cut_mesh_by_plane,
)
from pcb_aoi.geometry.transform import (
    transform_points,
    transform_points_as_tuples,
)


def trim_plane(
    *,
    board_far_face_z: float,
    protrusion_mm: float,
    cut_deg: float,
    cut_azimuth_deg: float,
    grow_toward_negative_z: bool,
    span_radius_mm: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the trim plane for one lead.

    Args:
        board_far_face_z: Board face the protrusion is measured from.
        protrusion_mm: Depth the deepest point of the section must land at.
        cut_deg: Trim angle off perpendicular.
        cut_azimuth_deg: Direction the trim plane tips toward.
        grow_toward_negative_z: Whether the lead protrudes toward -z.
        span_radius_mm: Half-width of the lead, used to lift the plane so its
            deepest point rather than its centre lands at the protrusion.

    Returns:
        ``(point_on_plane, unit_normal)``, the normal pointing at the material
        the blade removes.
    """
    direction = -1.0 if grow_toward_negative_z else 1.0
    cut = math.radians(cut_deg)
    azimuth = math.radians(cut_azimuth_deg)
    normal = (
        math.sin(cut) * math.cos(azimuth),
        math.sin(cut) * math.sin(azimuth),
        math.cos(cut) * direction,
    )
    # A plane tipped by `cut` through the centre reaches `span * tan(cut)`
    # deeper on one side. Pull it back by that much so the DEEPEST point, not
    # the centre, sits at the requested protrusion.
    overhang = span_radius_mm * math.tan(cut)
    z = board_far_face_z + direction * (protrusion_mm - overhang)
    return (0.0, 0.0, z), normal


def offset_plane(
    plane: tuple[tuple[float, float, float], tuple[float, float, float]],
    distance: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the same plane moved along its normal.

    Args:
        plane: ``(point_on_plane, unit_normal)``.
        distance: How far to move it. Positive leaves MORE material, because
            the normal points at what the blade takes away.

    Returns:
        The moved plane.
    """
    origin, normal = plane
    return (
        tuple(origin[axis] + distance * normal[axis] for axis in range(3)),
        normal,
    )


def cut_prims(
    stage: object,
    prim_paths: Sequence[str],
    plane: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> list[Mapping[str, object]]:
    """Trim every named mesh with one plane, in world space.

    Args:
        stage: Open USD stage.
        prim_paths: Mesh prims to trim. The lead and its joint are passed
            together so one blade produces one section across both.
        plane: ``(point_on_plane, unit_normal)`` in WORLD space.

    Returns:
        One record per prim: what it was, what it became, and how many faces
        the new section took. A prim the plane misses is reported with
        ``cut_faces`` zero and is left untouched.
    """
    import numpy as np
    from pxr import Gf, UsdGeom, Vt

    records: list[Mapping[str, object]] = []
    cache = UsdGeom.XformCache()
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            records.append({"path": str(path), "skipped": "not a mesh"})
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts:
            records.append({"path": str(path), "skipped": "no geometry"})
            continue

        # Work in world space. The lead and its joint are authored under
        # different parents -- the joint lives beside the board root in the
        # solder scope -- so a plane expressed in either one's local frame is
        # the wrong plane for the other.
        to_world = cache.GetLocalToWorldTransform(prim)
        to_local = to_world.GetInverse()
        world_points = transform_points_as_tuples(to_world, points)
        cut_points, cut_counts, cut_indices, cap_faces = cut_mesh_by_plane(
            world_points, tuple(counts), tuple(indices), plane)
        # Close whatever is still open, cut or not.
        #
        # The joint arrives with a rim already: the solder generator leaves it
        # where the joint meets the plated hole, which is 24 edges before any
        # blade touches it. A trimmed lead is then a solid sitting inside a
        # surface with a hole in it, and nothing downstream that wants a closed
        # body -- a volume, a signed distance, a physical sim -- can use it.
        #
        # The cap is built from the rim's OWN vertices, so no point is added
        # and none moves: the silhouette is exactly what it was.
        cut_points, cut_counts, cut_indices, sealed = close_open_boundaries(
            cut_points, cut_counts, cut_indices)

        before = (len(points), len(counts))
        if not cut_counts:
            records.append({
                "path": str(path), "before_points": before[0],
                "before_faces": before[1], "after_faces": 0,
                "cut_faces": 0, "note": "entirely removed",
            })
            mesh.GetFaceVertexCountsAttr().Set([])
            mesh.GetFaceVertexIndicesAttr().Set([])
            continue

        # One Vt array rather than one Gf.Vec3f per point. On a mesh the size a
        # real board carries that loop was a quarter of the whole call. The
        # values are the same: both round float64 to float32 to nearest, and
        # `test_pin_cut_apply` compares the authored points against the
        # per-point Gf path coordinate for coordinate.
        mesh.GetPointsAttr().Set(
            Vt.Vec3fArray.FromNumpy(
                transform_points(to_local, cut_points).astype(np.float32),
            ),
        )
        mesh.GetFaceVertexCountsAttr().Set(list(cut_counts))
        mesh.GetFaceVertexIndicesAttr().Set(list(cut_indices))
        # Authored normals describe the mesh before the cut, and the cut adds
        # points -- measured on a production THT panel, all eighteen joints ended up
        # carrying 432 normals for 472 points. `Clear()` only drops the opinion
        # in the current edit target, so a weaker layer's stale array survives
        # it and the renderer shades from an array that no longer matches the
        # mesh. Block it outright, which is authoritative across layers.
        normals = mesh.GetNormalsAttr()
        if normals:
            normals.Block()
        records.append({
            "path": str(path),
            "before_points": before[0], "before_faces": before[1],
            "after_points": len(cut_points), "after_faces": len(cut_counts),
            "cut_faces": cap_faces,
            "sealed_holes": sealed,
        })
    return records
