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

"""Declarative-API tests for the Modus lighting mode.

``appearance.lighting(modus=...)`` must validate the block and compile it into
the flow YAML ``lighting.modus`` key that ``common.randomize_lighting`` /
``sdg_pipeline`` read. CI-safe: stdlib unittest, no omni/USD.
"""

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: F401 -- registers the "pcb-aoi" domain
from cdf import CDFJob, ValidationError
from pcb_aoi.config.appearance import ModusLightingConfig


def _base_job():
    job = CDFJob(domain="pcb-aoi")
    job.scene.configure(usd_path="/tmp/board.usd", root_path="/Board")
    # compile() refuses a job whose component types it cannot make concrete;
    # these tests are about the lighting block, so name one type and move on.
    job.scene.select_components(types=["C_0805"])
    job.capture.renderer(
        mode="pathtracing", resolution=(64, 64), spp=1, total_spp=1, max_bounces=1
    )
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.camera(rotation_range={"x": (0, 0), "y": (0, 0), "z": -90})
    job.capture.outputs(rgb=True)
    return job


class ModusLightingApiTest(unittest.TestCase):
    def test_modus_compiles_into_flow_lighting_block(self):
        job = _base_job()
        job.appearance.lighting(
            mode="modus_light",
            modus={
                "arc_radius_scale": 1.5,
                "height_scale": 2.0,
                "width_frac": 0.12,
                "elevations": {"white": 75, "red": 60, "blue": 45},
                "intensities": {"white": 9000, "red": 7000, "blue": 6000},
                "colors": {
                    "white": [1.0, 1.0, 1.0],
                    "red": [1.0, 0.02, 0.01],
                    "blue": [0.03, 0.05, 0.98],
                },
            },
        )
        flow = job.compile().flow
        self.assertIn("modus", flow["lighting"])
        modus = flow["lighting"]["modus"]
        self.assertEqual(modus["arc_radius_scale"], 1.5)
        self.assertEqual(modus["elevations"]["blue"], 45)
        self.assertEqual(modus["colors"]["blue"], [0.03, 0.05, 0.98])
        # modus does not set ring_light
        self.assertFalse(flow["lighting"].get("ring_light"))

    def test_modus_minimal_block_uses_pipeline_defaults(self):
        job = _base_job()
        job.appearance.lighting(mode="modus_light", modus={})
        flow = job.compile().flow
        self.assertEqual(flow["lighting"]["modus"], {})

    def test_modus_mode_rejects_the_ring_payload(self):
        job = _base_job()
        with self.assertRaisesRegex(ValidationError, "not part of mode"):
            job.appearance.lighting(
                mode="modus_light",
                layers={"Inner_Red": {"intensity": [1, 2]}},
                modus={},
            )

    def test_ring_mode_rejects_the_modus_payload(self):
        job = _base_job()
        with self.assertRaisesRegex(ValidationError, "not part of mode"):
            job.appearance.lighting(
                mode="rgb_light", layers={"Inner_Red": {"intensity": [1, 2]}}, modus={}
            )

    def test_unknown_modus_key_rejected(self):
        job = _base_job()
        with self.assertRaises(ValidationError):
            job.appearance.lighting(mode="modus_light", modus={"bogus": 1})

    def test_unknown_modus_group_rejected(self):
        job = _base_job()
        with self.assertRaises(ValidationError):
            job.appearance.lighting(
                mode="modus_light", modus={"elevations": {"green": 50}}
            )

    def test_bad_colour_triple_rejected(self):
        job = _base_job()
        with self.assertRaises(ValidationError):
            job.appearance.lighting(
                mode="modus_light", modus={"colors": {"white": [1.0, 1.0]}}
            )

    def test_width_and_width_frac_together_rejected(self):
        job = _base_job()
        with self.assertRaisesRegex(ValidationError, "width_frac"):
            job.appearance.lighting(
                mode="modus_light", modus={"width": 20.0, "width_frac": 0.12}
            )

    def test_layers_compile_into_the_flow_lighting_block(self):
        job = _base_job()
        job.appearance.lighting(
            mode="modus_light",
            modus={
                "intensities": {"white": 9000, "red": 7000, "blue": 6000},
                "colors": {
                    "white": [1.0, 1.0, 1.0],
                    "red": [1.0, 0.0, 0.0],
                    "blue": [0.0, 0.0, 1.0],
                },
                "layers": {
                    "white": {
                        "intensity": [80000, 100000],
                        "color_r": [0.95, 1.0],
                        "color_g": [0.95, 1.0],
                        "color_b": [0.95, 1.0],
                    },
                    "red": {
                        "intensity": [60000, 80000],
                        "color_r": [0.95, 1.0],
                        "color_g": [0.0, 0.05],
                        "color_b": [0.0, 0.05],
                    },
                },
            },
        )
        flow = job.compile().flow
        layers = flow["lighting"]["modus"]["layers"]
        self.assertEqual(set(layers), {"white", "red"})
        self.assertEqual(layers["white"]["intensity"], [80000, 100000])
        self.assertEqual(layers["red"]["color_r"], [0.95, 1.0])
        # A group left out of layers is not required to appear here; its
        # fixed intensities/colors value is what randomize_lighting falls
        # back to.
        self.assertNotIn("blue", layers)

    def test_unknown_layers_group_rejected(self):
        job = _base_job()
        with self.assertRaises(ValidationError):
            job.appearance.lighting(
                mode="modus_light", modus={"layers": {"green": {"intensity": [1, 2]}}}
            )


class ModusLightingConfigYamlTest(unittest.TestCase):
    """``ModusLightingConfig`` must catch what ``_modus_block`` catches --
    but at YAML parse time, before a pipeline builds anything.
    """

    def test_width_and_width_frac_together_rejected_at_parse_time(self):
        with self.assertRaises(PydanticValidationError):
            ModusLightingConfig.model_validate({"width": 20.0, "width_frac": 0.12})

    def test_an_unknown_group_is_rejected_at_parse_time(self):
        for field in ("elevations", "intensities", "colors", "layers"):
            with self.subTest(field=field):
                payload = {
                    field: {
                        "green": ({"intensity": [1, 2]} if field == "layers" else 50)
                    }
                }
                with self.assertRaises(PydanticValidationError):
                    ModusLightingConfig.model_validate(payload)

    def test_a_non_positive_geometry_knob_is_rejected_at_parse_time(self):
        for field in ("arc_radius_scale", "width", "height_scale"):
            with self.subTest(field=field), self.assertRaises(PydanticValidationError):
                ModusLightingConfig.model_validate({field: 0.0})

    def test_layers_reuse_the_ring_layer_schema(self):
        config = ModusLightingConfig.model_validate(
            {
                "layers": {
                    "white": {
                        "intensity": [80000, 100000],
                        "color_r": [0.95, 1.0],
                        "color_g": [0.95, 1.0],
                        "color_b": [0.95, 1.0],
                    },
                },
            }
        )
        self.assertEqual(config.layers["white"].intensity, (80000, 100000))


if __name__ == "__main__":
    unittest.main()
