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

"""Test which prims size the Modus rig, without USD or Isaac Sim."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from cdf.errors import ValidationError
from cdf.execution.isaac import capture_execution
from cdf.execution.isaac.execution_context import IsaacExecutionContext


class ModusBoundsPathsTest(unittest.TestCase):
    """Verify panels size Modus from placements and single boards from the root."""

    def test_single_board_uses_board_root(self):
        context = IsaacExecutionContext(runtime=object(), stage=object())
        context.scene = {"root_path": "/World", "board_root_path": "/World/Board"}

        self.assertEqual(
            capture_execution._modus_bounds_paths(context),
            ("/World/Board",),
        )

    def test_panel_uses_every_placement_not_the_prototype(self):
        context = IsaacExecutionContext(runtime=object(), stage=object())
        context.scene = {
            "root_path": "/World/Panel",
            "board_root_path": "/World/_class_Boards/Board",
        }
        context.board_instance_cohort = SimpleNamespace(
            instance_paths=("/World/Panel/SR_0_0", "/World/Panel/SR_0_1"),
        )

        self.assertEqual(
            capture_execution._modus_bounds_paths(context),
            ("/World/Panel/SR_0_0", "/World/Panel/SR_0_1"),
        )

    def test_unloaded_scene_is_rejected(self):
        context = IsaacExecutionContext(runtime=object(), stage=object())
        context.scene = {}

        with self.assertRaisesRegex(ValidationError, "scene.load"):
            capture_execution._modus_bounds_paths(context)

    def test_rig_build_forwards_bounds_paths(self):
        common = Mock()
        config = {
            "modus_light_root": "/World/Modus",
            "pcba_root": "/World/_class_Boards/Board",
            "modus_bounds_paths": ("/World/Panel/SR_0_0", "/World/Panel/SR_0_1"),
            "camera_side": "bottom",
        }

        capture_execution._build_modus_rig(object(), config, {"modus": {}}, common)

        self.assertEqual(
            common.build_modus_lights.call_args.kwargs["bounds_paths"],
            ("/World/Panel/SR_0_0", "/World/Panel/SR_0_1"),
        )


if __name__ == "__main__":
    unittest.main()
