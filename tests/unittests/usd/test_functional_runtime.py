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

"""Test USD-backed functional output-quality checks."""

from __future__ import annotations

import unittest

from pxr import Usd, UsdGeom

from tests.functional import runtime


class GeneratedMeshQualityTest(unittest.TestCase):
    """Verify generated solder mesh topology before rendered output is accepted."""

    def test_accepts_finite_nonempty_mesh_geometry(self) -> None:
        """Accept a finite triangle with internally consistent topology."""
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/Generated/Solder")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

        paths = runtime._validate_generated_meshes(stage, ["/Generated"])

        self.assertEqual(("/Generated/Solder",), paths)

    def test_rejects_an_empty_generated_mesh(self) -> None:
        """Reject a mesh prim that carries no renderable geometry."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Mesh.Define(stage, "/Generated/Solder")

        with self.assertRaisesRegex(RuntimeError, "has no points"):
            runtime._validate_generated_meshes(stage, ["/Generated"])


if __name__ == "__main__":
    unittest.main()
