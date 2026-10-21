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

"""``appearance.lighting(mode=...)`` selects one rig and one payload.

The five modes replace the ``use_scene_lights`` / ``ring_light`` boolean pair,
which could only express three rigs and gave no name to the Modus illuminator
or to the pure-white dome. Two rigs are never active at once, so one required
selector is the honest shape and the booleans are gone -- from the API, from
the compiled flow, and from the shipped configs.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_STANDALONE = REPO_ROOT / "scripts" / "sdg" / "standalone"
if str(_STANDALONE) not in sys.path:
    sys.path.insert(0, str(_STANDALONE))

import pcb_aoi  # noqa: E402,F401 -- registers the `pcb-aoi` domain
from cdf import CDFJob  # noqa: E402
from cdf.errors import ValidationError  # noqa: E402
from cdf.execution.isaac.capture_execution import _lighting_config  # noqa: E402
from cdf.execution.isaac.job_order_parameters import (  # noqa: E402
    LightingParameters,
)
from pcb_aoi.lighting import lighting_mode
from cdf.sections import LIGHTING_MODES  # noqa: E402

RING_LAYERS = {
    "Inner_Red": {
        "intensity": (6500, 6500),
        "color_r": (1.0, 1.0),
        "color_g": (0.0, 0.0),
        "color_b": (0.0, 0.0),
    },
}


def _job():
    """Return a job whose scene is declared, ready for a lighting call."""
    job = CDFJob(domain="pcb-aoi", random_seed=7)
    job.scene.configure(
        usd_path="board.usd",
        root_path="/Board",
        camera_path="/World/camera_light/Camera",
        camera_xform_path="/World/camera_light",
        ring_light_root="/World/camera_light/aoi_ring_light",
    )
    return job


class LightingModeSelectionTest(unittest.TestCase):
    """Each mode records itself plus the booleans the pipeline still reads."""

    def test_every_mode_records_itself(self):
        payloads = {
            "rgb_light": {"layers": RING_LAYERS},
            "white_light": {"dome": {"intensity": (700, 700)}},
            "dome_light": {
                "dome": {
                    "intensity": (700, 700),
                    "color_r": (0.9, 1.0),
                    "color_g": (0.9, 1.0),
                    "color_b": (0.8, 0.9),
                },
            },
            "modus_light": {"modus": {}},
            "usd_light": {},
        }
        self.assertEqual(set(payloads), set(LIGHTING_MODES))
        for mode, payload in payloads.items():
            with self.subTest(mode=mode):
                job = _job()
                job.appearance.lighting(mode=mode, **payload)
                spec = job.appearance.lighting_spec
                self.assertEqual(spec["mode"], mode)
                self.assertNotIn("use_scene_lights", spec)
                self.assertNotIn("ring_light", spec)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "lighting.mode must be one of"):
            _job().appearance.lighting(mode="disco_light")

    def test_mode_is_required(self):
        with self.assertRaisesRegex(ValidationError, "lighting requires mode="):
            _job().appearance.lighting()


class LightingPayloadTest(unittest.TestCase):
    """A mode accepts its own payload and rejects every other rig's."""

    def test_ring_requires_layers(self):
        with self.assertRaisesRegex(ValidationError, "layers is required"):
            _job().appearance.lighting(mode="rgb_light")

    def test_ring_accepts_dome_fill(self):
        job = _job()
        job.appearance.lighting(
            mode="rgb_light",
            layers=RING_LAYERS,
            dome_fill={"intensity": (700, 700)},
        )
        self.assertEqual(
            job.appearance.lighting_spec["dome_fill"]["intensity"], [700, 700]
        )

    def test_modus_requires_its_block(self):
        with self.assertRaisesRegex(ValidationError, "modus is required"):
            _job().appearance.lighting(mode="modus_light")

    def test_usd_light_takes_no_rig_payload(self):
        for key, value in (
            ("layers", RING_LAYERS),
            ("dome", {"intensity": (700, 700)}),
            ("modus", {}),
            ("dome_fill", {"intensity": (700, 700)}),
        ):
            with self.subTest(payload=key):
                with self.assertRaisesRegex(ValidationError, "not part of mode"):
                    _job().appearance.lighting(mode="usd_light", **{key: value})

    def test_dome_fill_belongs_to_the_ring_only(self):
        with self.assertRaisesRegex(ValidationError, "not part of mode"):
            _job().appearance.lighting(
                mode="dome_light",
                dome={
                    "intensity": (700, 700),
                    "color_r": (1.0, 1.0),
                    "color_g": (1.0, 1.0),
                    "color_b": (1.0, 1.0),
                },
                dome_fill={"intensity": (700, 700)},
            )


class DomeModesTest(unittest.TestCase):
    """``white_light`` pins the dome white; ``dome_light`` authors its colour."""

    def test_white_light_pins_the_colour(self):
        job = _job()
        job.appearance.lighting(mode="white_light", dome={"intensity": (500, 900)})
        block = job.appearance.lighting_spec["white_light"]
        self.assertEqual(block["intensity"], [500, 900])
        for channel in ("color_r", "color_g", "color_b"):
            self.assertEqual(block[channel], [1.0, 1.0])

    def test_white_light_rejects_an_authored_colour(self):
        with self.assertRaisesRegex(ValidationError, "pins the dome to white"):
            _job().appearance.lighting(
                mode="white_light",
                dome={"intensity": (500, 900), "color_r": (0.2, 0.4)},
            )

    def test_dome_light_requires_all_three_channels(self):
        with self.assertRaisesRegex(ValidationError, "needs color_r, color_g"):
            _job().appearance.lighting(
                mode="dome_light",
                dome={"intensity": (500, 900), "color_r": (0.2, 0.4)},
            )

    def test_dome_requires_an_intensity(self):
        with self.assertRaisesRegex(ValidationError, "requires an intensity"):
            _job().appearance.lighting(mode="white_light", dome={})


class RetiredSpellingTest(unittest.TestCase):
    """The boolean pair is gone from the API and from the pipeline config."""

    def test_retired_kwargs_are_no_longer_accepted(self):
        for kwarg in ("use_scene_lights", "ring_light", "white_light"):
            with self.subTest(kwarg=kwarg):
                with self.assertRaises(TypeError):
                    _job().appearance.lighting(mode="rgb_light", **{kwarg: True})

    def test_pipeline_config_without_a_mode_says_what_to_write(self):
        with self.assertRaisesRegex(ValueError, "lighting.mode is required"):
            lighting_mode({"lighting": {"layers": {}}})

    def test_legacy_flow_yaml_still_resolves_through_the_booleans(self):
        # Historical configurations are the compatibility surface formerly reached
        # through legacy_router; it keeps the pre-enum spelling and must go on
        # resolving to the same rig.
        self.assertEqual(
            lighting_mode({"use_scene_lights": True, "lighting": {}}), "usd_light"
        )
        self.assertEqual(
            lighting_mode({"lighting": {"ring_light": True}}), "rgb_light"
        )
        self.assertEqual(
            lighting_mode({"lighting": {"ring_light": False}}), "dome_light"
        )

    def test_a_config_naming_no_rig_at_all_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lighting.mode is required"):
            lighting_mode({"lighting": {}})

    def test_pipeline_config_rejects_an_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            lighting_mode({"lighting": {"mode": "disco_light"}})


class CompiledFlowTest(unittest.TestCase):
    """The compiled pipeline flow carries the mode into ``lighting:``."""

    def _flow(self, **kwargs):
        job = _job()
        job.scene.select_components(types=["PASSIVE_SMD_0805"])
        job.appearance.lighting(**kwargs)
        job.capture.output_start(0)
        job.capture.renderer(
            mode="pathtracing", resolution=(320, 180), spp=1, total_spp=2, max_bounces=4
        )
        job.capture.camera(rotation_range={"x": (0, 0), "y": (0, 0), "z": -90})
        job.capture.scan_grid(x_count=1, y_count=1)
        job.capture.limits(triggers=1, max_images=1)
        job.capture.outputs(rgb=True)
        return job.compile().flow

    def test_modus_mode_reaches_the_flow(self):
        flow = self._flow(mode="modus_light", modus={"arc_radius_scale": 1.5})
        # The retired boolean lives inside the lighting block, so asserting its
        # absence at the top level passed even when it was still being emitted.
        self.assertNotIn("use_scene_lights", flow)
        self.assertNotIn("use_scene_lights", flow["lighting"])
        self.assertEqual(flow["lighting"]["mode"], "modus_light")
        self.assertEqual(flow["lighting"]["modus"]["arc_radius_scale"], 1.5)

    def test_white_light_mode_reaches_the_flow_as_a_white_dome(self):
        flow = self._flow(mode="white_light", dome={"intensity": (400, 400)})
        self.assertEqual(flow["lighting"]["mode"], "white_light")
        self.assertNotIn("ring_light", flow["lighting"])
        self.assertEqual(flow["lighting"]["white_light"]["color_r"], [1.0, 1.0])

    def test_usd_light_mode_reaches_the_flow(self):
        flow = self._flow(mode="usd_light")
        self.assertNotIn("use_scene_lights", flow)
        self.assertNotIn("use_scene_lights", flow["lighting"])
        self.assertEqual(flow["lighting"]["mode"], "usd_light")


class JobOrderContractTest(unittest.TestCase):
    """Every mode has to survive the JobOrder transport boundary.

    ``LightingParameters`` is ``extra="forbid"``, so a field the declarative
    API emits but the model does not declare fails at execution time rather
    than at compile time -- invisible to any test that only inspects
    ``job.compile().flow``. Validate the compiled order itself.
    """

    def _lighting_order(self, **kwargs):
        job = _job()
        job.scene.select_components(types=["PASSIVE_SMD_0805"])
        job.appearance.lighting(**kwargs)
        job.capture.output_start(0)
        job.capture.renderer(
            mode="pathtracing", resolution=(320, 180), spp=1, total_spp=2, max_bounces=4
        )
        job.capture.camera(rotation_range={"x": (0, 0), "y": (0, 0), "z": -90})
        job.capture.scan_grid(x_count=1, y_count=1)
        job.capture.limits(triggers=1, max_images=1)
        job.capture.outputs(rgb=True)
        orders = [
            order
            for order in job.compile_orders().orders
            if order.order_id == "appearance.lighting"
        ]
        self.assertEqual(len(orders), 1)
        return orders[0]

    def test_every_mode_validates_against_the_contract(self):
        payloads = {
            "rgb_light": {"layers": RING_LAYERS},
            "white_light": {"dome": {"intensity": (700, 700)}},
            "dome_light": {
                "dome": {
                    "intensity": (700, 700),
                    "color_r": (0.9, 1.0),
                    "color_g": (0.9, 1.0),
                    "color_b": (0.8, 0.9),
                },
            },
            "modus_light": {"modus": {"arc_radius_scale": 1.5}},
            "usd_light": {},
        }
        for mode, payload in payloads.items():
            with self.subTest(mode=mode):
                order = self._lighting_order(mode=mode, **payload)
                validated = LightingParameters.model_validate(dict(order.parameters))
                self.assertEqual(validated.mode, mode)

    def test_contract_rejects_an_unknown_mode(self):
        with self.assertRaises(Exception):
            LightingParameters.model_validate({"mode": "disco_light"})

    def test_unselected_rig_payloads_are_dropped_before_the_pipeline(self):
        # `randomize_lighting` tests membership, not truth, so a None-valued
        # `white_light` would pass its guard and then fail on subscript.
        order = self._lighting_order(mode="rgb_light", layers=RING_LAYERS)
        state = LightingParameters.model_validate(
            dict(order.parameters)
        ).model_dump(mode="python")
        config = _lighting_config(state)
        self.assertIn("layers", config)
        for absent in ("white_light", "dome_fill"):
            self.assertNotIn(absent, config)


if __name__ == "__main__":
    unittest.main()
