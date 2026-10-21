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

"""Tests for randomised material-override appearance."""

from __future__ import annotations

import unittest

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi domain
from cdf import CDFJob
from cdf.errors import ValidationError
from cdf.execution.isaac.job_order_parameters import MaterialOverrideParameters
from cdf.execution.isaac.job_order_runtime import sample_material_variation


class MaterialVariationDeclarationTest(unittest.TestCase):
    """Verify how variation ranges are declared and validated."""

    def _job(self):
        job = CDFJob(domain="pcb-aoi", random_seed=7)
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        return job

    def test_variation_ranges_are_recorded_on_the_material_spec(self):
        job = self._job()
        job.appearance.material_override(
            "carrier",
            material_path="/World/Looks/Carrier",
            variation={"roughness": (0.3, 0.7), "texture_rotation": (0.0, 180.0)},
        )

        spec = job.appearance.material_specs["carrier"]

        # ranges are normalised to lists so the compiled plan stays JSON-friendly
        self.assertEqual(
            spec["variation"],
            {"roughness": [0.3, 0.7], "texture_rotation": [0.0, 180.0]},
        )

    def test_material_without_variation_records_no_variation_key(self):
        job = self._job()
        job.appearance.material_override(
            "carrier",
            material_path="/World/Looks/Carrier",
        )

        self.assertNotIn("variation", job.appearance.material_specs["carrier"])

    def test_unknown_variation_control_is_rejected(self):
        job = self._job()

        with self.assertRaisesRegex(ValidationError, "unknown variation controls"):
            job.appearance.material_override(
                "carrier",
                material_path="/World/Looks/Carrier",
                variation={"diffuse_color": (0.0, 1.0)},
            )

    def test_parameters_reject_unordered_bounds(self):
        with self.assertRaisesRegex(ValueError, "bounds must be ordered"):
            MaterialOverrideParameters(
                material_path="/World/Looks/Carrier",
                variation={"roughness": (0.9, 0.1)},
            )

    def test_opacity_is_recorded_and_transport_validated(self):
        job = self._job()
        job.appearance.material_override(
            "solder_mask",
            material_path="/World/Looks/SolderMask",
            opacity=0.5,
        )

        spec = job.appearance.material_specs["solder_mask"]
        self.assertEqual(spec["opacity"], 0.5)
        self.assertEqual(
            MaterialOverrideParameters.model_validate(spec).opacity,
            0.5,
        )
        with self.assertRaisesRegex(ValueError, "less than or equal to 1"):
            MaterialOverrideParameters(
                material_path="/World/Looks/SolderMask",
                opacity=1.1,
            )


class MaterialVariationSamplingTest(unittest.TestCase):
    """Verify the execution-time draw."""

    RANGES = {"roughness": (0.2, 0.8), "albedo_brightness": (0.5, 1.5)}

    def test_sample_is_reproducible_for_a_seed_and_material(self):
        first = sample_material_variation(11, "/World/Looks/Carrier", self.RANGES)
        second = sample_material_variation(11, "/World/Looks/Carrier", self.RANGES)

        self.assertEqual(first, second)
        self.assertEqual(sorted(first), ["albedo_brightness", "roughness"])

    def test_materials_and_seeds_vary_independently(self):
        carrier = sample_material_variation(11, "/World/Looks/Carrier", self.RANGES)
        mask = sample_material_variation(11, "/World/Looks/SolderMask", self.RANGES)
        reseeded = sample_material_variation(12, "/World/Looks/Carrier", self.RANGES)

        self.assertNotEqual(carrier, mask)
        self.assertNotEqual(carrier, reseeded)

    def test_samples_stay_inside_the_declared_bounds(self):
        for seed in range(25):
            sample = sample_material_variation(seed, "/World/Looks/Carrier", self.RANGES)
            for control, (low, high) in self.RANGES.items():
                self.assertGreaterEqual(sample[control], low)
                self.assertLessEqual(sample[control], high)

    def test_no_declared_ranges_yields_no_sample(self):
        self.assertEqual(sample_material_variation(1, "/World/Looks/Carrier", {}), {})


if __name__ == "__main__":
    unittest.main()
