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

"""Pin down the board's xy footprint before making it faster.

`_substrate_xy_bounds` has two branches and both walk every point of every mesh
through a `Gf.Matrix4d.Transform` call of its own. The legacy branch walks the
WHOLE board, not just the substrate, which makes it the largest scalar loop in
this module. Neither branch had a test, so there was nothing to hold a batched
replacement to.

These assert what has to keep being true across that change: which branch runs,
that the answer is world space, that a rotation is applied rather than ignored,
and that the scan reaches the end of the points rather than sampling them.
"""

from __future__ import annotations

import unittest

try:
    from pxr import Gf, Sdf, Usd, UsdGeom

    from pcb_aoi.hole_plane import _substrate_xy_bounds
except ModuleNotFoundError:
    Gf = None
    _substrate_xy_bounds = None


@unittest.skipUnless(Gf is not None, "usd-core is not installed")
class SubstrateXyBoundsTest(unittest.TestCase):
    """The world-space xy extent a hole plane is sized from."""

    def _stage(self):
        """Return an in-memory stage with a board root."""
        stage = Usd.Stage.CreateInMemory()
        stage.SetDefaultPrim(stage.DefinePrim("/Board"))
        return stage

    def _mesh(self, stage, path, points, *, role=None):
        """Author one mesh, optionally tagged with a substrate role."""
        mesh = UsdGeom.Mesh.Define(stage, path)
        if role is not None:
            mesh.GetPrim().CreateAttribute(
                "pcba:mesh:role", Sdf.ValueTypeNames.Token,
            ).Set(role)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        return mesh.GetPrim()

    def _bounds(self, stage):
        """Run the function under test against the whole board."""
        return _substrate_xy_bounds(stage, "/Board")

    def test_a_tagged_substrate_gives_its_own_extent(self):
        stage = self._stage()
        self._mesh(stage, "/Board/S",
                   [(-3.0, -2.0, 0.0), (5.0, 7.0, 0.0)], role="board_substrate")

        self.assertEqual(self._bounds(stage), (-3.0, -2.0, 5.0, 7.0))

    def test_a_tagged_substrate_excludes_everything_not_tagged(self):
        """This is the whole point of the metadata branch."""
        stage = self._stage()
        self._mesh(stage, "/Board/S",
                   [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)], role="board_substrate")
        self._mesh(stage, "/Board/Tall", [(-99.0, -99.0, 0.0), (99.0, 99.0, 0.0)])

        self.assertEqual(self._bounds(stage), (0.0, 0.0, 1.0, 1.0))

    def test_without_any_tag_the_legacy_branch_scans_every_mesh(self):
        stage = self._stage()
        self._mesh(stage, "/Board/A", [(-4.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
        self._mesh(stage, "/Board/B", [(0.0, -6.0, 0.0), (8.0, 2.0, 0.0)])

        self.assertEqual(self._bounds(stage), (-4.0, -6.0, 8.0, 2.0))

    def test_the_extent_is_world_space_not_local(self):
        stage = self._stage()
        parent = UsdGeom.Xform.Define(stage, "/Board/G")
        parent.AddTranslateOp().Set(Gf.Vec3d(100.0, -50.0, 0.0))
        self._mesh(stage, "/Board/G/S",
                   [(-1.0, -1.0, 0.0), (1.0, 1.0, 0.0)], role="board_substrate")

        self.assertEqual(self._bounds(stage), (99.0, -51.0, 101.0, -49.0))

    def test_a_rotation_is_applied_rather_than_ignored(self):
        """Rotating a square 45 degrees widens its axis-aligned extent."""
        stage = self._stage()
        parent = UsdGeom.Xform.Define(stage, "/Board/G")
        parent.AddRotateZOp().Set(45.0)
        self._mesh(stage, "/Board/G/S",
                   [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0),
                    (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)], role="board_substrate")

        x_min, y_min, x_max, y_max = self._bounds(stage)

        for value, expected in ((x_min, -(2 ** 0.5)), (y_min, -(2 ** 0.5)),
                                (x_max, 2 ** 0.5), (y_max, 2 ** 0.5)):
            self.assertAlmostEqual(value, expected, places=5)

    def test_every_point_is_considered_not_just_the_corners(self):
        """A scan that sampled instead of walking would pass everything above."""
        stage = self._stage()
        points = [(0.0, 0.0, 0.0)] * 50
        points[23] = (-11.0, 0.0, 0.0)
        points[41] = (0.0, 13.0, 0.0)
        self._mesh(stage, "/Board/S", points, role="board_substrate")

        self.assertEqual(self._bounds(stage), (-11.0, 0.0, 0.0, 13.0))

    def test_meshes_with_no_points_do_not_break_the_scan(self):
        stage = self._stage()
        UsdGeom.Mesh.Define(stage, "/Board/Empty").GetPrim().CreateAttribute(
            "pcba:mesh:role", Sdf.ValueTypeNames.Token,
        ).Set("board_substrate")
        self._mesh(stage, "/Board/S",
                   [(2.0, 3.0, 0.0), (4.0, 5.0, 0.0)], role="board_substrate")

        self.assertEqual(self._bounds(stage), (2.0, 3.0, 4.0, 5.0))

    def test_the_values_are_plain_floats(self):
        stage = self._stage()
        self._mesh(stage, "/Board/S",
                   [(0.0, 0.0, 0.0), (1.0, 2.0, 0.0)], role="board_substrate")

        for value in self._bounds(stage):
            self.assertIs(type(value), float)

    def test_an_invalid_board_root_is_rejected(self):
        stage = self._stage()

        with self.assertRaises(ValueError):
            _substrate_xy_bounds(stage, "/NotThere")


if __name__ == "__main__":
    unittest.main()
