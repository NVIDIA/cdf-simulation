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

"""Trim a lead on a real stage and check what came back.

`cut_prims` had no test, which is how a per-point transform loop sat on the
per-lead path unmeasured. It needs only an in-memory stage, so there is no
excuse: the lead is a cylinder under a parent transform, so the world-space
blade and the local-space write-back are both exercised, and a mesh authored at
the origin would have hidden a mistake in either.

The trimmed geometry is compared against the same cut performed entirely with
`Gf.Matrix4d.Transform`, coordinate for coordinate. That is the property the
batched transform has to preserve: it is a speed change, and a speed change
that moves a vertex is a defect.
"""

from __future__ import annotations


import math
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from pxr import Gf, Sdf, Usd, UsdGeom
except ImportError:  # the CI unit-test image installs numpy only
    Gf = Sdf = Usd = UsdGeom = None

from pcb_aoi.geometry.plane_cut import (  # noqa: E402
    close_open_boundaries,
    cut_mesh_by_plane,
)
from pcb_aoi.tht.pin_cut_apply import cut_prims, offset_plane, trim_plane  # noqa: E402


def _cylinder(segments: int, stacks: int, radius: float, height: float):
    """Build a closed cylinder standing on the local origin, as a lead is."""
    points = []
    for stack in range(stacks + 1):
        z = height * stack / stacks
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            points.append(
                Gf.Vec3f(radius * math.cos(angle), radius * math.sin(angle), z),
            )
    counts, indices = [], []
    for stack in range(stacks):
        base = stack * segments
        for segment in range(segments):
            following = (segment + 1) % segments
            counts.append(4)
            indices.extend(
                (
                    base + segment,
                    base + following,
                    base + segments + following,
                    base + segments + segment,
                ),
            )
    counts.append(segments)
    indices.extend(range(segments - 1, -1, -1))
    top = stacks * segments
    counts.append(segments)
    indices.extend(range(top, top + segments))
    return points, counts, indices


def _stage_with_lead(
    *,
    # z stays at 0 so the world-space blade actually crosses the barrel;
    # x and y are far from the origin so a dropped translation would show.
    translate=(12.0, -7.5, 0.0),
    rotate_deg=23.0,
    scale=1.0,
    segments=16,
    stacks=6,
):
    """Return a stage carrying one lead under a non-identity parent xform."""
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, Sdf.Path("/Board/Component"))
    parent.AddTranslateOp().Set(Gf.Vec3d(*translate))
    parent.AddRotateZOp().Set(rotate_deg)
    if scale != 1.0:
        parent.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path("/Board/Component/Lead"))
    points, counts, indices = _cylinder(segments, stacks, 0.3, 3.0)
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    return stage, "/Board/Component/Lead"


def _scalar_reference(stage, path, plane):
    """Cut one prim the way the per-point Gf loop used to, for comparison."""
    prim = stage.GetPrimAtPath(path)
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = tuple(mesh.GetFaceVertexCountsAttr().Get())
    indices = tuple(mesh.GetFaceVertexIndicesAttr().Get())
    to_world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    to_local = to_world.GetInverse()

    world_points = [
        tuple(to_world.Transform(Gf.Vec3d(*point))) for point in points
    ]
    cut_points, cut_counts, cut_indices, _ = cut_mesh_by_plane(
        world_points, counts, indices, plane,
    )
    cut_points, cut_counts, cut_indices, _ = close_open_boundaries(
        cut_points, cut_counts, cut_indices,
    )
    local = [
        Gf.Vec3f(*to_local.Transform(Gf.Vec3d(*point))) for point in cut_points
    ]
    return local, list(cut_counts), list(cut_indices)


@unittest.skipUnless(UsdGeom is not None, "usd-core is not installed")
class CutPrimsTest(unittest.TestCase):
    """What a trimmed lead looks like after the blade."""

    def _plane(self):
        return trim_plane(
            board_far_face_z=0.0,
            protrusion_mm=1.4,
            cut_deg=0.0,
            cut_azimuth_deg=0.0,
            grow_toward_negative_z=False,
            span_radius_mm=0.3,
        )

    def test_a_trim_removes_material_and_caps_the_opening(self):
        stage, path = _stage_with_lead()
        before = len(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get())

        records = cut_prims(stage, [path], self._plane())

        self.assertEqual(len(records), 1)
        self.assertGreater(records[0]["cut_faces"], 0)
        self.assertEqual(records[0]["before_points"], before)
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        self.assertGreater(len(mesh.GetFaceVertexCountsAttr().Get()), 0)

    def test_the_result_is_identical_to_the_per_point_gf_loop(self):
        """The batched transform is a speed change, not a geometry change."""
        for rotate_deg, scale in ((0.0, 1.0), (23.0, 1.0), (137.0, 2.5)):
            with self.subTest(rotate_deg=rotate_deg, scale=scale):
                plane = self._plane()
                reference_stage, path = _stage_with_lead(
                    rotate_deg=rotate_deg, scale=scale,
                )
                expected_points, expected_counts, expected_indices = (
                    _scalar_reference(reference_stage, path, plane)
                )

                stage, path = _stage_with_lead(rotate_deg=rotate_deg, scale=scale)
                cut_prims(stage, [path], plane)
                mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))

                actual_points = list(mesh.GetPointsAttr().Get())
                self.assertEqual(len(actual_points), len(expected_points))
                for index, (got, want) in enumerate(
                    zip(actual_points, expected_points, strict=True),
                ):
                    self.assertEqual(tuple(got), tuple(want), "point %d" % index)
                self.assertEqual(
                    list(mesh.GetFaceVertexCountsAttr().Get()), expected_counts,
                )
                self.assertEqual(
                    list(mesh.GetFaceVertexIndicesAttr().Get()), expected_indices,
                )

    def test_a_plane_the_lead_never_reaches_takes_nothing(self):
        """No cap, no lost faces.

        The points ARE rewritten even so -- a miss still round-trips them
        through world and back, which lands some of them a float32 step from
        where they started. That is the shipped behaviour, not something this
        change introduced, so the assertion describes it rather than pretending
        the array is untouched.
        """
        stage, path = _stage_with_lead()
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        before_points = list(mesh.GetPointsAttr().Get())
        before_counts = list(mesh.GetFaceVertexCountsAttr().Get())

        records = cut_prims(
            stage,
            [path],
            offset_plane(self._plane(), 500.0),
        )

        self.assertEqual(records[0]["cut_faces"], 0)
        self.assertEqual(
            list(mesh.GetFaceVertexCountsAttr().Get()), before_counts,
        )
        after_points = list(mesh.GetPointsAttr().Get())
        self.assertEqual(len(after_points), len(before_points))
        for index, (got, want) in enumerate(
            zip(after_points, before_points, strict=True),
        ):
            for axis in range(3):
                self.assertAlmostEqual(
                    got[axis], want[axis], places=5, msg="point %d" % index,
                )

    def test_a_plane_that_takes_everything_empties_the_mesh(self):
        stage, path = _stage_with_lead()

        records = cut_prims(
            stage,
            [path],
            ((0.0, 0.0, -900.0), (0.0, 0.0, 1.0)),
        )

        self.assertEqual(records[0]["after_faces"], 0)
        self.assertEqual(records[0]["note"], "entirely removed")
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        self.assertEqual(list(mesh.GetFaceVertexCountsAttr().Get()), [])

    def test_a_prim_that_is_not_a_mesh_is_reported_not_raised(self):
        stage, path = _stage_with_lead()
        UsdGeom.Xform.Define(stage, Sdf.Path("/Board/NotAMesh"))

        records = cut_prims(stage, ["/Board/NotAMesh", path], self._plane())

        self.assertEqual(records[0]["skipped"], "not a mesh")
        self.assertEqual(len(records), 2)

    def test_a_mesh_with_no_geometry_is_reported_not_raised(self):
        stage, path = _stage_with_lead()
        UsdGeom.Mesh.Define(stage, Sdf.Path("/Board/Empty"))

        records = cut_prims(stage, ["/Board/Empty"], self._plane())

        self.assertEqual(records[0]["skipped"], "no geometry")

    def test_every_point_a_face_still_uses_is_on_the_kept_side(self):
        """A cut that leaves material on the blade side is not a cut.

        Points are checked through the face indices, not through the whole
        array: the cutter compacts faces but not points, so the array still
        holds the orphans of the faces it removed. They are unreferenced, and
        measuring them would measure the wrong thing.
        """
        stage, path = _stage_with_lead()
        plane = self._plane()
        origin, normal = plane

        cut_prims(stage, [path], plane)

        prim = stage.GetPrimAtPath(path)
        mesh = UsdGeom.Mesh(prim)
        to_world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        points = mesh.GetPointsAttr().Get()
        referenced = sorted(set(mesh.GetFaceVertexIndicesAttr().Get()))
        self.assertGreater(len(referenced), 0)

        for index in referenced:
            world = to_world.Transform(Gf.Vec3d(*points[index]))
            distance = sum(
                (world[axis] - origin[axis]) * normal[axis] for axis in range(3)
            )
            self.assertLessEqual(distance, 1.0e-5, "point %d" % index)


if __name__ == "__main__":
    unittest.main()
