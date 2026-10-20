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

"""Unit tests for the public component-defect transform API."""

from __future__ import annotations

import unittest

import numpy as np

from tests.unittests.core.defects.mock_utils import DefectsApiMockEnvironment


class DefectsApiTest(unittest.TestCase):
    """Validate the public defect API without an Omniverse runtime."""

    def setUp(self) -> None:
        """Create an isolated defects API fixture."""
        self._environment = DefectsApiMockEnvironment()
        self.addCleanup(self._environment.stop)
        self.defects_api, self.component = self._environment.start()

    def test_all_public_calls_return_consistent_results(self) -> None:
        """Verify all public calls return the shared result contract."""
        original = np.eye(4)
        np.random.seed(2)

        results = (
            self.defects_api.shift(
                self.component,
                original,
                translate_range=0.2,
                rotate_z_range=15.0,
            ),
            self.defects_api.sideflip(
                self.component,
                original,
                angle_min=70.0,
                angle_max=90.0,
            ),
            self.defects_api.reverse_polarity(self.component, original),
            self.defects_api.tombstone(
                self.component,
                original,
                angle_min=70.0,
                angle_max=90.0,
            ),
        )

        for result in results:
            with self.subTest(result=result):
                self.assertIsInstance(result, self.defects_api.DefectTransform)
                self.assertEqual(len(result.flat_transform), 16)
                self.assertIn("angle_deg", result.parameters)

    def test_shift_respects_ranges_and_keeps_pivot_translation(self) -> None:
        """Verify shift bounds and pivot-relative translation."""
        original = np.eye(4)
        original[3, :3] = (10.0, 20.0, 30.0)
        original_copy = original.copy()
        np.random.seed(123)

        result = self.defects_api.shift(
            self.component,
            original,
            translate_range=0.2,
            rotate_z_range=15.0,
        )
        parameters = result.parameters
        translation = np.asarray(parameters["translate"])
        final = np.asarray(result.flat_transform).reshape(4, 4)
        pivot = np.append(parameters["pivot_local"], 1.0)

        self.assertTrue(np.all(np.abs(translation[:2]) <= 0.2))
        self.assertEqual(translation[2], 0.0)
        self.assertLessEqual(abs(parameters["rotate_z_deg"]), 15.0)
        np.testing.assert_allclose(
            pivot @ final,
            pivot @ original + np.append(translation, 0.0),
        )
        np.testing.assert_array_equal(original, original_copy)

    def test_zero_shift_is_identity(self) -> None:
        """Verify a zero-range shift preserves the original transform."""
        original = np.eye(4)
        original[3, :3] = (10.0, 20.0, 30.0)

        result = self.defects_api.shift(
            self.component,
            original,
            translate_range=0.0,
            rotate_z_range=0.0,
        )

        np.testing.assert_allclose(
            np.asarray(result.flat_transform).reshape(4, 4),
            original,
        )

    def test_sideflip_uses_length_axis_and_support_edge(self) -> None:
        """Verify sideflip rotates around the length-axis support edge."""
        original = np.eye(4)
        np.random.seed(7)

        result = self.defects_api.sideflip(
            self.component,
            original,
            angle_min=80.0,
            angle_max=80.0,
        )
        parameters = result.parameters
        pivot = np.asarray(parameters["pivot_local"])
        final = np.asarray(result.flat_transform).reshape(4, 4)

        np.testing.assert_allclose(
            np.abs(parameters["rot_axis"]),
            (1.0, 0.0, 0.0),
        )
        self.assertAlmostEqual(abs(pivot[1]), 1.0)
        self.assertAlmostEqual(pivot[2], -0.2)
        np.testing.assert_allclose(
            np.append(pivot, 1.0) @ final,
            np.append(pivot, 1.0),
        )

    def test_reverse_polarity_uses_board_normal_and_fixed_angle(self) -> None:
        """Verify reverse polarity uses a fixed board-normal half-turn."""
        original = np.eye(4)

        result = self.defects_api.reverse_polarity(self.component, original)
        parameters = result.parameters
        pivot = np.append(parameters["pivot_local"], 1.0)
        final = np.asarray(result.flat_transform).reshape(4, 4)

        self.assertEqual(parameters["angle_deg"], 180.0)
        np.testing.assert_allclose(
            np.abs(parameters["rot_axis"]),
            (0.0, 0.0, 1.0),
        )
        np.testing.assert_allclose(pivot @ final, pivot @ original)

    def test_tombstone_prefers_paste_geometry(self) -> None:
        """Verify tombstone prefers solder-paste contact geometry."""
        metadata = {
            "side": "Top",
            "solder_paste_prims": ["/paste_0", "/paste_1"],
        }
        np.random.seed(11)

        result = self.defects_api.tombstone(
            self.component,
            np.eye(4),
            angle_min=80.0,
            angle_max=80.0,
            component_instance=metadata,
        )
        parameters = result.parameters

        self.assertTrue(parameters["use_contacts"])
        self.assertEqual(parameters["contact_source"], "paste")
        np.testing.assert_allclose(
            np.abs(parameters["rot_axis"]),
            (0.0, 1.0, 0.0),
        )
        self.assertAlmostEqual(parameters["pivot_local"][2], -0.2)

    def test_public_api_rejects_invalid_inputs(self) -> None:
        """Verify public calls reject invalid ranges and transforms."""
        invalid_calls = (
            lambda: self.defects_api.shift(
                self.component,
                np.eye(4),
                translate_range=-0.1,
                rotate_z_range=15.0,
            ),
            lambda: self.defects_api.sideflip(
                self.component,
                np.eye(4),
                angle_min=90.0,
                angle_max=70.0,
            ),
            lambda: self.defects_api.tombstone(
                self.component,
                np.eye(4),
                angle_min=90.0,
                angle_max=70.0,
            ),
            lambda: self.defects_api.reverse_polarity(self.component, np.eye(3)),
        )

        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()
