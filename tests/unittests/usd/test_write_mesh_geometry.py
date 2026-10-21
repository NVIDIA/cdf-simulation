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

"""Pin down how a solved mesh is authored before making it faster.

Profiling the `surface_evolver` path in-process showed 97% of
`solder_joints.generate` is `select.poll` -- waiting on the external solver --
and that the largest PYTHON cost left in it is `_write_mesh_geometry`, which
walks the points five times in Python: once to build `Gf.Vec3f` objects, once
more for normals, and three times to find the extent.

It had no test. These fix the observable contract -- what lands on the prim,
in what type, with what extent -- so the walks can be replaced without the
authored stage changing.
"""

from __future__ import annotations

import unittest

from pxr import Usd, UsdGeom

from pcb_aoi.solder_generation.data_types import SolderMesh
from pcb_aoi.solder_generation.solder_generation_extension import (
    SolderGenerationExtension,
)


class WriteMeshGeometryTest(unittest.TestCase):
    """What `_write_mesh_geometry` leaves on the stage."""

    #: A triangle whose three axes each have a different min and max.
    POINTS = ((-1.0, 0.5, 3.0), (2.0, -4.0, 0.25), (0.0, 7.0, -6.5))

    def _write(self, mesh, path="/Solder"):
        """Author `mesh` onto a fresh stage and return the UsdGeom.Mesh.

        The stage is held on the test case: dropping the last reference to it
        invalidates the prim, and every assertion then fails with
        `Accessed schema on invalid prim` rather than with what it checks.
        """
        self._stage = Usd.Stage.CreateInMemory()
        return SolderGenerationExtension._write_mesh_geometry(
            self._stage,
            path,
            mesh,
        )

    def _mesh(self, **overrides):
        """Build a SolderMesh with the triangle above unless overridden."""
        fields = {
            "name": "joint",
            "points": self.POINTS,
            "face_vertex_counts": (3,),
            "face_vertex_indices": (0, 1, 2),
        }
        fields.update(overrides)
        return SolderMesh(**fields)

    def test_the_points_land_in_order_and_unchanged(self):
        usd_mesh = self._write(self._mesh())

        self.assertEqual(
            [tuple(point) for point in usd_mesh.GetPointsAttr().Get()],
            [tuple(point) for point in self.POINTS],
        )

    def test_the_points_are_authored_as_single_precision(self):
        """Vec3f, not Vec3d -- consumers and file size both depend on it."""
        usd_mesh = self._write(self._mesh())

        self.assertEqual(
            usd_mesh.GetPointsAttr().GetTypeName().type.typeName,
            "VtArray<GfVec3f>",
        )

    def test_the_topology_lands_as_authored(self):
        usd_mesh = self._write(
            self._mesh(
                face_vertex_counts=(3, 3),
                face_vertex_indices=(0, 1, 2, 2, 1, 0),
            )
        )

        self.assertEqual(list(usd_mesh.GetFaceVertexCountsAttr().Get()), [3, 3])
        self.assertEqual(
            list(usd_mesh.GetFaceVertexIndicesAttr().Get()),
            [0, 1, 2, 2, 1, 0],
        )

    def test_subdivision_is_off_so_the_solved_shape_is_the_shape(self):
        usd_mesh = self._write(self._mesh())

        self.assertEqual(
            usd_mesh.GetSubdivisionSchemeAttr().Get(),
            UsdGeom.Tokens.none,
        )

    def test_the_extent_is_the_per_axis_min_and_max(self):
        """Each axis takes its bound from a DIFFERENT point here."""
        usd_mesh = self._write(self._mesh())

        low, high = usd_mesh.GetExtentAttr().Get()

        self.assertEqual(tuple(low), (-1.0, -4.0, -6.5))
        self.assertEqual(tuple(high), (2.0, 7.0, 3.0))

    def test_normals_are_written_per_vertex_when_present(self):
        normals = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
        usd_mesh = self._write(self._mesh(normals=normals))

        self.assertEqual(
            [tuple(normal) for normal in usd_mesh.GetNormalsAttr().Get()],
            [tuple(normal) for normal in normals],
        )
        self.assertEqual(
            usd_mesh.GetNormalsInterpolation(),
            UsdGeom.Tokens.vertex,
        )

    def test_no_normals_authors_no_normals_attribute(self):
        usd_mesh = self._write(self._mesh())

        self.assertFalse(usd_mesh.GetNormalsAttr().HasAuthoredValue())

    def test_a_mesh_with_no_points_authors_no_extent(self):
        usd_mesh = self._write(
            self._mesh(
                points=(),
                face_vertex_counts=(),
                face_vertex_indices=(),
            )
        )

        self.assertFalse(usd_mesh.GetExtentAttr().HasAuthoredValue())

    def test_every_point_reaches_the_extent_not_just_the_ends(self):
        """A partial scan passes the small cases above and fails this one."""
        points = [(0.0, 0.0, 0.0)] * 60
        points[19] = (-30.0, 0.0, 0.0)
        points[44] = (0.0, 0.0, 25.0)
        usd_mesh = self._write(
            self._mesh(
                points=tuple(points),
                face_vertex_counts=(3,),
                face_vertex_indices=(0, 1, 2),
            )
        )

        low, high = usd_mesh.GetExtentAttr().Get()

        self.assertEqual(tuple(low), (-30.0, 0.0, 0.0))
        self.assertEqual(tuple(high), (0.0, 0.0, 25.0))

    def test_values_that_do_not_survive_float32_are_narrowed_the_same_way(self):
        """The authored array is float32; the narrowing must not change."""
        points = (
            (0.1, 0.2, 0.3),
            (1e-8, 123456.789, -0.7),
            (3.14159265358979, 2.718281828459045, 1.4142135623730951),
        )
        usd_mesh = self._write(self._mesh(points=points))

        import struct

        authored_points = usd_mesh.GetPointsAttr().Get()
        self.assertEqual(len(authored_points), len(points))
        for authored, original in zip(authored_points, points, strict=True):
            self.assertEqual(len(authored), len(original))
            for got, want in zip(authored, original, strict=True):
                self.assertEqual(
                    struct.pack("<f", got),
                    struct.pack("<f", want),
                )


if __name__ == "__main__":
    unittest.main()
