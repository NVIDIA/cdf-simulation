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

"""Tests for grouped, visibly synthetic hole-plane proxy geometry."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import cdf.execution.isaac.job_order_runtime as runtime_module
from cdf.execution.isaac.execution_context import IsaacExecutionContext
from cdf.execution.isaac.job_order_parameters import HolePlaneParameters
from cdf.execution.isaac.job_order_runtime import IsaacSdgRuntime
from pcb_aoi.hole_plane import create_hole_plane, create_hole_plane_at_position

try:
    from pxr import Gf, Sdf, Usd, UsdGeom
except ModuleNotFoundError:
    Gf = Sdf = Usd = UsdGeom = None


class HolePlaneSelectionTest(unittest.TestCase):
    """Verify explicit component filters never widen their generation scope."""

    def test_empty_filter_warns_without_board_wide_fallback(self) -> None:
        context = IsaacExecutionContext(runtime=object(), stage=object())
        context.scene = {"root_path": "/Board", "board_root_path": "/Board"}
        catalog = SimpleNamespace(
            boards=(SimpleNamespace(substrate_z_range=(-0.2, 0.2)),),
            capabilities=frozenset(),
        )
        parameters = HolePlaneParameters(side="bottom", components=())

        with (
            mock.patch.object(runtime_module, "_board_metadata", return_value=catalog),
            mock.patch.object(runtime_module, "_standalone_module") as standalone,
            self.assertWarnsRegex(RuntimeWarning, "components is empty"),
        ):
            output = IsaacSdgRuntime().generate_hole_plane(context, parameters)

        self.assertEqual(output["prim_path"], [])
        self.assertTrue(output["warnings"])
        standalone.assert_not_called()

    def test_missing_filter_metadata_warns_without_board_wide_fallback(self) -> None:
        context = IsaacExecutionContext(runtime=object(), stage=object())
        context.scene = {"root_path": "/Board", "board_root_path": "/Board"}
        catalog = SimpleNamespace(
            boards=(SimpleNamespace(substrate_z_range=(-0.2, 0.2)),),
            capabilities=frozenset(),
        )
        parameters = HolePlaneParameters(
            side="bottom",
            components=("X205",),
        )

        with (
            mock.patch.object(runtime_module, "_board_metadata", return_value=catalog),
            self.assertWarnsRegex(RuntimeWarning, "never fall back"),
        ):
            output = IsaacSdgRuntime().generate_hole_plane(context, parameters)

        self.assertEqual(output["prim_path"], [])
        self.assertEqual(output["unmatched_components"], ["X205"])
        self.assertTrue(output["warnings"])


@unittest.skipIf(Usd is None, "usd-core not installed (CI installs numpy only)")
class HolePlaneAuthoringTest(unittest.TestCase):
    """Verify hole meshes use a groupable parent and diagnostic color."""

    def setUp(self) -> None:
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(self.stage, "/World/Board")
        self.substrate = UsdGeom.Mesh.Define(self.stage, "/World/Board/Substrate")
        self.substrate.CreatePointsAttr(
            [
                Gf.Vec3f(-2.0, -1.0, 0.0),
                Gf.Vec3f(2.0, -1.0, 0.0),
                Gf.Vec3f(2.0, 1.0, 0.0),
                Gf.Vec3f(-2.0, 1.0, 0.0),
            ]
        )
        self.substrate.GetPrim().CreateAttribute(
            "pcba:mesh:role",
            Sdf.ValueTypeNames.Token,
        ).Set("board_substrate")

    def test_board_plane_is_magenta_child_of_xform(self) -> None:
        path = create_hole_plane(
            self.stage,
            "/World/Board",
            "/World/Board/SegmentationProxies/HolePlane",
            (-0.2, 0.2),
            side="bottom",
            z_offset_mm=0.1,
            meters_per_unit=0.001,
            class_name="hole",
        )

        self.assertEqual(path, "/World/Board/SegmentationProxies/HolePlane/plane")
        self.assertEqual(
            UsdGeom.Imageable(
                self.stage.GetPrimAtPath("/World/Board/SegmentationProxies")
            )
            .GetVisibilityAttr()
            .Get(),
            UsdGeom.Tokens.invisible,
        )
        self.assertTrue(
            self.stage.GetPrimAtPath("/World/Board/SegmentationProxies/HolePlane").IsA(
                UsdGeom.Xform
            )
        )
        self.assertEqual(
            UsdGeom.Imageable(
                self.stage.GetPrimAtPath("/World/Board/SegmentationProxies/HolePlane")
            )
            .GetVisibilityAttr()
            .Get(),
            UsdGeom.Tokens.invisible,
        )
        color = UsdGeom.Mesh(self.stage.GetPrimAtPath(path)).GetDisplayColorAttr().Get()
        self.assertEqual(tuple(color[0]), (1.0, 0.0, 1.0))

    def test_filtered_plane_gets_xform_parent(self) -> None:
        path = create_hole_plane_at_position(
            self.stage,
            "/World/Board/SegmentationProxies/HolePlane/pad_0",
            "/World/Board/Substrate",
            (-0.2, 0.2),
            side="bottom",
            z_offset_mm=0.1,
            meters_per_unit=0.001,
            class_name="hole",
        )

        self.assertEqual(path, "/World/Board/SegmentationProxies/HolePlane/pad_0")
        self.assertTrue(
            self.stage.GetPrimAtPath("/World/Board/SegmentationProxies/HolePlane").IsA(
                UsdGeom.Xform
            )
        )
        self.assertEqual(
            UsdGeom.Imageable(
                self.stage.GetPrimAtPath("/World/Board/SegmentationProxies/HolePlane")
            )
            .GetVisibilityAttr()
            .Get(),
            UsdGeom.Tokens.invisible,
        )
        color = UsdGeom.Mesh(self.stage.GetPrimAtPath(path)).GetDisplayColorAttr().Get()
        self.assertEqual(tuple(color[0]), (1.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
