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

"""Pin down the substrate z extent before making it faster.

`_substrate_z_range` walks every point of every substrate mesh through a
`Gf.Matrix4d.Transform` call to keep two scalars, which is the shape this tree
has already replaced elsewhere with `transform_points`. It had no test, so
there was nothing to make that replacement safe against.

These assert what the function must keep saying: world space rather than local,
the extent across several meshes rather than one, and the exact values a
non-trivial matrix produces -- a rotation, so a change that quietly dropped the
transform would move the answer instead of leaving it alone.
"""

from __future__ import annotations

import math
import unittest

try:
    from pxr import Gf, Sdf, Usd, UsdGeom

    from pcb_aoi.board_metadata.usd_schema.metadata_provider import (
        _substrate_z_range,
    )
except ModuleNotFoundError:
    Gf = None
    _substrate_z_range = None


@unittest.skipUnless(Gf is not None, "usd-core is not installed")
class SubstrateZRangeTest(unittest.TestCase):
    """The world-space z extent of the board_substrate meshes."""

    def _mesh(self, stage, path, points, *, role="board_substrate"):
        """Author one mesh prim carrying `role` and `points`."""
        mesh = UsdGeom.Mesh.Define(stage, path)
        prim = mesh.GetPrim()
        prim.CreateAttribute("pcba:mesh:role", Sdf.ValueTypeNames.Token).Set(role)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        return prim

    def _stage(self):
        """Return an empty in-memory stage with a board root."""
        stage = Usd.Stage.CreateInMemory()
        stage.SetDefaultPrim(stage.DefinePrim("/Board"))
        return stage

    def test_a_single_mesh_reports_its_lowest_and_highest_point(self):
        stage = self._stage()
        prim = self._mesh(stage, "/Board/Substrate",
                          [(0.0, 0.0, -1.5), (1.0, 0.0, 0.25), (0.0, 1.0, 2.0)])

        self.assertEqual(_substrate_z_range((prim,)), (-1.5, 2.0))

    def test_the_extent_spans_every_substrate_mesh(self):
        stage = self._stage()
        low = self._mesh(stage, "/Board/A", [(0.0, 0.0, -4.0), (1.0, 0.0, -3.0)])
        high = self._mesh(stage, "/Board/B", [(0.0, 0.0, 7.0), (1.0, 0.0, 6.0)])

        self.assertEqual(_substrate_z_range((low, high)), (-4.0, 7.0))

    def test_meshes_that_are_not_substrate_do_not_count(self):
        stage = self._stage()
        substrate = self._mesh(stage, "/Board/Substrate",
                               [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
        pad = self._mesh(stage, "/Board/Pad", [(0.0, 0.0, 99.0)], role="board_pad")

        self.assertEqual(_substrate_z_range((substrate, pad)), (0.0, 1.0))

    def test_the_answer_is_world_space_not_local(self):
        """A parent translation must move the extent with it."""
        stage = self._stage()
        parent = UsdGeom.Xform.Define(stage, "/Board/Group")
        parent.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 10.0))
        prim = self._mesh(stage, "/Board/Group/Substrate",
                          [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0)])

        self.assertEqual(_substrate_z_range((prim,)), (9.0, 11.0))

    def test_a_rotation_mixes_x_into_z_and_the_answer_follows(self):
        """The z extent of a rotated plate is not the z extent of its points."""
        stage = self._stage()
        parent = UsdGeom.Xform.Define(stage, "/Board/Group")
        parent.AddRotateYOp().Set(90.0)
        # Flat in z; only the rotation can produce a non-zero extent.
        prim = self._mesh(stage, "/Board/Group/Substrate",
                          [(-2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 1.0, 0.0)])

        low, high = _substrate_z_range((prim,))

        # Rotating +90 about y sends +x to -z, so x in [-2, 3] becomes z in [-3, 2].
        self.assertAlmostEqual(low, -3.0, places=5)
        self.assertAlmostEqual(high, 2.0, places=5)

    def test_a_mesh_with_no_points_is_ignored_rather_than_crashing(self):
        stage = self._stage()
        empty = UsdGeom.Mesh.Define(stage, "/Board/Empty").GetPrim()
        empty.CreateAttribute("pcba:mesh:role", Sdf.ValueTypeNames.Token).Set(
            "board_substrate",
        )
        real = self._mesh(stage, "/Board/Substrate",
                          [(0.0, 0.0, 5.0), (0.0, 0.0, 6.0)])

        self.assertEqual(_substrate_z_range((empty, real)), (5.0, 6.0))

    def test_no_substrate_at_all_is_None_rather_than_an_empty_range(self):
        stage = self._stage()
        pad = self._mesh(stage, "/Board/Pad", [(0.0, 0.0, 1.0)], role="board_pad")

        self.assertIsNone(_substrate_z_range((pad,)))

    def test_the_values_are_plain_floats(self):
        """Callers write these into a record and on to JSON."""
        stage = self._stage()
        prim = self._mesh(stage, "/Board/Substrate",
                          [(0.0, 0.0, 0.5), (0.0, 0.0, 1.5)])

        low, high = _substrate_z_range((prim,))

        self.assertIs(type(low), float)
        self.assertIs(type(high), float)

    def test_every_point_is_considered_not_just_the_first_and_last(self):
        """A partial scan would pass the tests above and fail this one."""
        stage = self._stage()
        points = [(0.0, 0.0, 0.0)] * 40
        points[17] = (0.0, 0.0, -8.0)
        points[29] = (0.0, 0.0, 9.0)
        prim = self._mesh(stage, "/Board/Substrate", points)

        self.assertEqual(_substrate_z_range((prim,)), (-8.0, 9.0))

    def test_a_projective_matrix_is_divided_through(self):
        """Gf.Transform is a projective transform, not a bare matmul."""
        stage = self._stage()
        prim = self._mesh(stage, "/Board/Substrate",
                          [(0.0, 0.0, 1.0), (0.0, 0.0, 3.0)])
        # USD is row-vector, so the projective column is [*][3]: w becomes
        # 0.5z + 1 and every z is divided by it.
        matrix = Gf.Matrix4d(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.5,
            0.0, 0.0, 0.0, 1.0,
        )
        expected = sorted(matrix.Transform(Gf.Vec3d(0.0, 0.0, z))[2]
                          for z in (1.0, 3.0))
        # The plain matmul answers would be 1.0 and 3.0; the divide changes both.
        self.assertNotAlmostEqual(expected[0], 1.0, places=6)

        parent = UsdGeom.Xform.Define(stage, "/Board/Group")
        UsdGeom.Xformable(parent).AddTransformOp().Set(matrix)
        moved = self._mesh(stage, "/Board/Group/Substrate",
                           [(0.0, 0.0, 1.0), (0.0, 0.0, 3.0)])

        low, high = _substrate_z_range((moved,))

        self.assertAlmostEqual(low, expected[0], places=9)
        self.assertAlmostEqual(high, expected[1], places=9)


if __name__ == "__main__":
    unittest.main()
