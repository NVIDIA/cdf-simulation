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

"""Validate the shared SMT functional-scenario plans without launching Isaac."""

from __future__ import annotations

import unittest
from pathlib import Path

from tests.functional import (
    test_smt_reverse_polarity,
    test_smt_shift,
    test_smt_sideflip,
    test_smt_tombstone,
    test_tht,
)


class SmtFunctionalScenarioTest(unittest.TestCase):
    """Keep defect-specific scripts thin while preserving their plan semantics."""

    def test_local_defaults_use_sdg_test_output(self) -> None:
        """Keep every local functional output beneath the ignored output root."""
        repo_root = Path(__file__).resolve().parents[3]
        output_root = repo_root / "sdg_test_output" / "functional"

        self.assertEqual(
            test_smt_tombstone.SCENARIO.default_output_dir,
            output_root / "smt_tombstone",
        )
        self.assertEqual(
            test_smt_shift.SCENARIO.default_output_dir,
            output_root / "smt_shift",
        )
        self.assertEqual(
            test_smt_sideflip.SCENARIO.default_output_dir,
            output_root / "smt_sideflip",
        )
        self.assertEqual(
            test_smt_reverse_polarity.SCENARIO.default_output_dir,
            output_root / "smt_reverse_polarity",
        )
        self.assertEqual(test_tht.DEFAULT_OUTPUT_DIR, output_root / "tht")

    def _orders(self, module: object) -> dict[str, object]:
        job, eligible_count, selection_ratio = module.build_job(
            resolution=(320, 180),
            total_spp=2,
        )
        self.assertEqual(eligible_count, 1)
        self.assertEqual(selection_ratio, 1.0)
        plan = job.compile_orders(output_dir=Path("functional-output"))
        orders = {order.order_id: order for order in plan.orders}
        self.assertNotIn("scene.sidecar.attach", orders)
        self.assertEqual(orders["scene.components.select"].parameters["source"], "usd")
        return orders

    def test_tombstone_injects_only_tombstone_specific_work(self) -> None:
        """Keep tombstone observable through its own functional workflow."""
        orders = self._orders(test_smt_tombstone)

        self.assertIn("operation.tombstone", orders)
        self.assertFalse(
            any(order_id.startswith("solder_joints.") for order_id in orders),
        )
        crop = orders["postprocess.crop"]
        self.assertEqual(crop.parameters["class_filter"], ("tombstone",))
        self.assertEqual(crop.parameters["output_subdir"], "tombstone_closeups")

    def test_shift_generates_procedural_solder_after_the_defect(self) -> None:
        """Keep shifted solder generation ordered after the pose defect."""
        orders = self._orders(test_smt_shift)

        self.assertIn("operation.shift", orders)
        self.assertEqual(
            orders["solder_joints.collect"].depends_on,
            ("solder_joints.discover", "operation.shift"),
        )
        self.assertEqual(
            orders["solder_joints.generate"].depends_on,
            ("solder_joints.collect",),
        )
        self.assertEqual(
            orders["solder_joints.attach"].depends_on,
            ("solder_joints.generate",),
        )
        self.assertIn(
            "solder_joints.attach",
            orders["capture.render"].depends_on,
        )
        parameters = orders["solder_joints.generate"].parameters
        self.assertEqual(parameters["solver"], "procedural")
        self.assertEqual(parameters["component_types"], ("C_0805_2012Metric",))
        self.assertTrue(parameters["solver_options"]["solid"])
        crop = orders["postprocess.crop"]
        self.assertEqual(crop.parameters["class_filter"], ("shift",))
        self.assertEqual(crop.parameters["output_subdir"], "shift_closeups")

    def test_sideflip_is_a_standalone_structural_scenario(self) -> None:
        """Keep sideflip observable through its own functional workflow."""
        orders = self._orders(test_smt_sideflip)

        self.assertIn("operation.sideflip", orders)
        self.assertFalse(
            any(order_id.startswith("solder_joints.") for order_id in orders),
        )
        crop = orders["postprocess.crop"]
        self.assertEqual(crop.parameters["class_filter"], ("sideflip",))
        self.assertEqual(crop.parameters["output_subdir"], "sideflip_closeups")

    def test_reverse_polarity_is_a_standalone_structural_scenario(self) -> None:
        """Keep reverse polarity observable through its own functional workflow."""
        orders = self._orders(test_smt_reverse_polarity)

        self.assertIn("operation.reverse_polarity", orders)
        self.assertEqual(
            orders["scene.components.select"].parameters["types"],
            ("LED_0603_1608Metric_G",),
        )
        self.assertFalse(
            any(order_id.startswith("solder_joints.") for order_id in orders),
        )
        crop = orders["postprocess.crop"]
        self.assertEqual(crop.parameters["class_filter"], ("reverse_polarity",))
        self.assertEqual(
            crop.parameters["output_subdir"],
            "reverse_polarity_closeups",
        )


if __name__ == "__main__":
    unittest.main()
