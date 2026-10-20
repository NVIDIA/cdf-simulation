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

"""Tests for the environment-sensor defect-render example."""

from __future__ import annotations

import unittest

from pcb_aoi.examples.environment_sensor_defects import build_job


class EnvironmentSensorDefectExampleTest(unittest.TestCase):
    """Verify the example compiles into the intended sidecar-free workflow."""

    def test_plan_contains_six_defects_capture_and_closeups(self) -> None:
        """Compile exact defect counts without solder or sidecar orders."""
        plan = build_job(resolution=(320, 180), total_spp=2).compile_orders(
            output_dir="output",
        )
        orders = {order.order_id: order for order in plan.orders}

        self.assertNotIn("scene.sidecar.components", orders)
        self.assertNotIn("operation.solder_joints", orders)
        for operation in ("tombstone", "sideflip", "shift"):
            parameters = orders[f"operation.{operation}"].parameters
            self.assertEqual(parameters["count"], 2)
            self.assertNotIn("ratio", parameters)
        self.assertEqual(
            orders["operation.tombstone"].parameters["component_types"],
            ("C8", "C5"),
        )
        self.assertEqual(orders["capture.scan_grid"].parameters["x_count"], 1)
        self.assertEqual(orders["capture.scan_grid"].parameters["y_count"], 1)
        self.assertEqual(
            orders["postprocess.crop"].parameters["class_filter"],
            ("tombstone", "sideflip", "shift"),
        )
