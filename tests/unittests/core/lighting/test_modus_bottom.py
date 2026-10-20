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

"""Test bottom-side geometry and capture-side forwarding in the core lane."""

import math
import unittest
from unittest.mock import Mock

from cdf.execution.isaac.capture_execution import _build_modus_rig
from pcb_aoi.modus_light import compute_modus_layout


class ModusBottomTest(unittest.TestCase):
    """Verify the bottom rig without launching Isaac Sim."""

    def test_bottom_aim_and_right_handed_basis(self):
        for side in ("top", "bottom"):
            with self.subTest(side=side):
                lights = compute_modus_layout(
                    board_center=(15, 25, 8),
                    board_top_z=11,
                    board_x_extent=100,
                    board_y_extent=200,
                    camera_side=side,
                )
                self.assertEqual(len(lights), 6)
                surface = 11 if side == "top" else 5
                sign = 1 if side == "top" else -1
                for light in lights:
                    self.assertEqual(light.aim_point, (15, 25, surface))
                    delta = sign * (light.position[2] - surface)
                    self.assertGreater(delta, 0)
                    angle = math.degrees(math.atan2(delta, abs(light.position[0] - 15)))
                    self.assertAlmostEqual(angle, light.elevation_deg)
                    self.assertLess(sign * light.normal[2], 0)
                    x, y, z = light.x_axis, light.y_axis, light.z_axis
                    cross = (
                        y[1] * z[2] - y[2] * z[1],
                        y[2] * z[0] - y[0] * z[2],
                        y[0] * z[1] - y[1] * z[0],
                    )
                    for actual, expected in zip(x, cross, strict=True):
                        self.assertAlmostEqual(actual, expected)

    def test_invalid_side_rejected(self):
        with self.assertRaisesRegex(ValueError, "camera_side"):
            compute_modus_layout(
                board_center=(0, 0, 0),
                board_top_z=1,
                board_x_extent=100,
                board_y_extent=100,
                camera_side="sideways",
            )

    def test_capture_forwards_side(self):
        for side in ("top", "bottom"):
            with self.subTest(side=side):
                common = Mock()
                stage = object()
                config = {
                    "modus_light_root": "/World/Modus",
                    "pcba_root": "/World/Board",
                    "camera_side": side,
                }
                _build_modus_rig(stage, config, {"modus": {}}, common)
                self.assertEqual(
                    common.build_modus_lights.call_args.kwargs["camera_side"],
                    side,
                )


if __name__ == "__main__":
    unittest.main()
