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

"""Size the Modus rig from the placed boards of an instanced panel."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from pxr import Gf, Sdf, Usd, UsdGeom

import cdf.execution.isaac.capture_execution as capture_module
from cdf.execution.isaac.execution_context import IsaacExecutionContext
from pcb_aoi.execution.board_instances import resolve_board_instance_cohort
from pcb_aoi.lighting import lighting_mode

PROTOTYPE = "/World/_class_Boards/Board"
PANEL = "/World/Panel"
INSTANCES = (f"{PANEL}/SR_0_0", f"{PANEL}/SR_0_1")


def _panel_stage() -> Usd.Stage:
    """Return two instanced placements of a board defined under a class prim.

    The board is a 2 x 2 x 2 cube; the placements sit at X = 0 and X = 10, so
    the panel spans X -1..11, Y -1..1, Z -1..1. The prototype source is
    abstract, exactly like the ``_class_`` scopes of a panel export.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    stage.DefinePrim("/World/_class_Boards", "Scope").SetSpecifier(
        Sdf.SpecifierClass,
    )
    UsdGeom.Xform.Define(stage, PROTOTYPE)
    body = UsdGeom.Cube.Define(stage, f"{PROTOTYPE}/Body")
    body.CreateSizeAttr(2.0)
    body.CreateExtentAttr([(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)])
    UsdGeom.Xform.Define(stage, PANEL)
    for path, x in zip(INSTANCES, (0.0, 10.0), strict=True):
        placement = UsdGeom.Xform.Define(stage, path)
        placement.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.0))
        placement.GetPrim().GetReferences().AddInternalReference(PROTOTYPE)
        placement.GetPrim().SetInstanceable(True)
    return stage


def _world_bounds(stage: Usd.Stage, paths: tuple[str, ...]) -> Gf.Range3d:
    """Return the union world bounds, raising like the shipped Modus sizing."""
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=True,
    )
    union = Gf.Range3d()
    for path in paths:
        bounds = cache.ComputeWorldBound(
            stage.GetPrimAtPath(path)
        ).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise RuntimeError(f"Modus light: world bbox empty for {path}")
        union.UnionWith(bounds)
    return union


class ModusPanelLightingTest(unittest.TestCase):
    """Verify ``appearance.lighting.apply`` sizes Modus over every placement."""

    def _context(self, stage: Usd.Stage) -> IsaacExecutionContext:
        context = IsaacExecutionContext(runtime=object(), stage=stage)
        context.board_instance_cohort = resolve_board_instance_cohort(
            stage,
            PANEL,
            INSTANCES,
        )
        context.scene = {
            "root_path": PANEL,
            "board_root_path": context.board_instance_cohort.prototype_source_path,
        }
        context.state["appearance.lighting"] = {
            "mode": "modus_light",
            "preserve_scene_light_color": False,
            "modus": {},
        }
        return context

    def test_apply_sizes_rig_from_placed_instances(self) -> None:
        stage = _panel_stage()
        context = self._context(stage)
        self.assertEqual(context.scene["board_root_path"], PROTOTYPE)
        sized = []

        def build_modus_lights(stage, modus_root, pcba_root, *, bounds_paths=None, **_):
            paths = tuple(bounds_paths) if bounds_paths is not None else (pcba_root,)
            sized.append((paths, _world_bounds(stage, paths)))

        common = SimpleNamespace(
            lighting_mode=lighting_mode,
            disable_all_lights=lambda _stage: None,
            build_modus_lights=build_modus_lights,
        )
        with mock.patch.object(capture_module, "_script_module", return_value=common):
            result = capture_module._apply_lighting_appearance(context, "bottom")

        self.assertEqual(result["mode"], "modus_light")
        self.assertEqual(len(sized), 1)
        paths, bounds = sized[0]
        self.assertEqual(paths, INSTANCES)
        self.assertEqual(tuple(bounds.GetMin()), (-1.0, -1.0, -1.0))
        self.assertEqual(tuple(bounds.GetMax()), (11.0, 1.0, 1.0))


class ModusBoundsFromPrimsTest(unittest.TestCase):
    """Verify the shared Modus bounds helper stays loud on empty bounds."""

    def test_union_of_instances(self) -> None:
        from pcb_aoi.modus_light import modus_bounds_from_prims

        center, top_z, x_extent, y_extent = modus_bounds_from_prims(
            _panel_stage(),
            INSTANCES,
        )

        self.assertEqual(center, (5.0, 0.0, 0.0))
        self.assertEqual((top_z, x_extent, y_extent), (1.0, 12.0, 2.0))

    def test_abstract_prototype_is_rejected(self) -> None:
        from pcb_aoi.modus_light import modus_bounds_from_prims

        with self.assertRaisesRegex(RuntimeError, "world bbox empty for .*Board"):
            modus_bounds_from_prims(_panel_stage(), (PROTOTYPE,))

    def test_one_empty_prim_fails_the_union(self) -> None:
        from pcb_aoi.modus_light import modus_bounds_from_prims

        stage = _panel_stage()
        UsdGeom.Xform.Define(stage, f"{PANEL}/Empty")
        with self.assertRaisesRegex(RuntimeError, "world bbox empty for .*Empty"):
            modus_bounds_from_prims(stage, (*INSTANCES, f"{PANEL}/Empty"))

    def test_missing_prim_and_no_prims_are_rejected(self) -> None:
        from pcb_aoi.modus_light import modus_bounds_from_prims

        stage = _panel_stage()
        with self.assertRaisesRegex(RuntimeError, "not found"):
            modus_bounds_from_prims(stage, (f"{PANEL}/Nope",))
        with self.assertRaisesRegex(RuntimeError, "no prims"):
            modus_bounds_from_prims(stage, ())


if __name__ == "__main__":
    unittest.main()
