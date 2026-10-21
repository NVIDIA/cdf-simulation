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

"""Re-applying the AOI light rig once per capture pass.

A plan applies the lighting once for every capture pass, and each apply builds
the rig into a fresh anonymous layer. ``RemovePrim`` only edits the current
edit target, so wherever the previous pass's rig is still authored -- its own
layer, or the root layer of the flattened stage a capture reopens -- its lights
would stay composed under the rebuilt rig, and authoring their xform ops again
raises.
"""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from cdf import IsaacExecutionContext
from cdf.execution.isaac.capture_execution import _apply_lighting_appearance

try:
    from pxr import Usd, UsdGeom
except ImportError:  # pragma: no cover - exercised by the usd lane only
    Usd = None

_RING_LIGHT_ROOT = "/World/camera_light/aoi_ring_light"
_LIGHT = f"{_RING_LIGHT_ROOT}/L_000"


def _build_ring_lights(stage, root, **_kwargs):
    """Build a one-light rig the way ``common.build_ring_lights`` does."""
    UsdGeom.Xform.Define(stage, root)
    for child in stage.GetPrimAtPath(root).GetChildren():
        stage.RemovePrim(child.GetPath())
    light = UsdGeom.Xform.Define(stage, f"{root}/L_000")
    light.AddTranslateOp().Set((1.0, 2.0, 3.0))


def _context(stage):
    """Return a context configured for rgb_light on ``stage``."""
    context = IsaacExecutionContext(stage=stage, scene={"root_path": "/Board"})
    context.state["appearance.lighting"] = {
        "mode": "rgb_light",
        "cone_angle_range": (120.0, 120.0),
        "cone_softness_range": (1.0, 1.0),
    }
    return context


def _common_module():
    """Patch the standalone ``common`` module with the stand-in rig builder."""
    return mock.patch(
        "cdf.execution.isaac.capture_execution._script_module",
        return_value=types.SimpleNamespace(
            lighting_mode=lambda config: "rgb_light",
            disable_all_lights=lambda stage: None,
            build_ring_lights=_build_ring_lights,
        ),
    )


@unittest.skipIf(Usd is None, "usd-core is not installed")
class LightingRigLayerTest(unittest.TestCase):
    """Each pass's rig must replace the last one, not stack on top of it."""

    def test_rgb_lighting_reapplied_per_pass_replaces_previous_rig_layer(self):
        """A second rgb_light apply rebuilds the rig over a single layer."""
        stage = Usd.Stage.CreateInMemory()
        context = _context(stage)

        with _common_module():
            _apply_lighting_appearance(context, "top")
            _apply_lighting_appearance(context, "top")

        self.assertEqual(len(context.state["appearance.lighting.rig_layers"]), 1)
        self.assertEqual(len(stage.GetRootLayer().subLayerPaths), 1)
        self.assertTrue(stage.GetPrimAtPath(_LIGHT).IsValid())

    def test_rgb_lighting_reapplied_after_a_capture_reopens_a_flattened_stage(self):
        """The old rig baked into a reopened stage's root layer is cleared too.

        A capture reopens a flattened copy of the stage when an earlier step
        changed instanced geometry, which bakes the previous pass's rig into
        the new root layer. This is what a real Isaac run does between the
        component_focus and component_focus_roi passes.
        """
        context = _context(Usd.Stage.CreateInMemory())
        with tempfile.TemporaryDirectory() as directory:
            # Text USD avoids a crate file mapping blocking Windows cleanup.
            flattened = str(Path(directory) / "flattened-stage.usda")

            with _common_module():
                _apply_lighting_appearance(context, "top")
                self.assertTrue(context.stage.Flatten().Export(flattened))
                context.stage = Usd.Stage.Open(flattened)
                _apply_lighting_appearance(context, "top")

            light = context.stage.GetPrimAtPath(_LIGHT)
            self.assertEqual(
                [spec.layer for spec in light.GetPrimStack()],
                context.state["appearance.lighting.rig_layers"],
            )
            self.assertEqual(
                [
                    op.GetOpName()
                    for op in UsdGeom.Xformable(light).GetOrderedXformOps()
                ],
                ["xformOp:translate"],
            )


if __name__ == "__main__":
    unittest.main()
