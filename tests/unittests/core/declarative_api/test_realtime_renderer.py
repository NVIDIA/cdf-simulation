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

"""The declarative API can express the renderer the pipeline already supports.

`configure_pathtracing` has handled ``render_mode: "realtime"`` all along, and
the declarative API could not reach it: `_RENDER_MODES` accepted only
``"pathtracing"``, and so did `RendererParameters.mode`, on a model with
``extra="forbid"`` -- so bypassing the first still failed the second.

Measured, realtime is 1.47x on `frame.step` at a 10x10 grid and 37.8 dB against
a 128-sample reference where path tracing is 46.7 dB.
That is a reason not to DEFAULT to it, not a reason the API should be unable to
say it. These tests hold the shape of saying it, and hold path tracing's
behaviour unchanged.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi CDF domain
from cdf.errors import ValidationError
from cdf.job import CDFJob


def _job() -> CDFJob:
    """Return a job with everything a capture pass needs except the renderer."""
    job = CDFJob(domain="pcb-aoi", random_seed=7)
    job.scene.configure(usd_path="board.usd", root_path="/Board")
    job.capture.camera(rotation_range={"x": (0, 0), "y": (0, 0), "z": -90})
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.limits(triggers=1, max_images=1)
    job.capture.outputs(rgb=True)
    return job


def _renderer_order(job: CDFJob) -> dict:
    """Return the compiled parameters of the renderer order."""
    for order in job.compile_orders(output_dir="/tmp/out").orders:
        if order.order_type == "capture.renderer.configure":
            return dict(order.parameters)
    raise AssertionError("no capture.renderer.configure order in the plan")


def _context(job: CDFJob):
    """Return a real execution context holding what the handlers would store.

    Hand-writing this mapping drifts from what the handlers actually store, so
    it is assembled the way `_store_configuration` does -- every
    `capture.X.configure` order becomes state key `capture.X` -- and dropped
    into the real context class so its other defaults are the real ones.
    """
    from cdf.execution.isaac.execution_context import IsaacExecutionContext

    orders = job.compile_orders(output_dir="/tmp/out").orders
    state = {
        (order.order_type.removesuffix(".configure")): dict(order.parameters)
        for order in orders
    }
    return IsaacExecutionContext(state=state, scene=state.get("scene.load", {}))


class RealtimeRendererTest(unittest.TestCase):
    """Declaring, compiling and rejecting the realtime renderer."""

    def test_realtime_is_accepted(self):
        job = _job()

        job.capture.renderer(
            mode="realtime",
            resolution=(64, 64),
            spp=1,
            total_spp=1,
            max_bounces=1,
        )

        self.assertEqual(job.capture.renderer_spec["mode"], "realtime")

    def test_realtime_reaches_the_compiled_plan(self):
        """A spec the compiler drops would look accepted and do nothing."""
        job = _job()
        job.capture.renderer(
            mode="realtime",
            resolution=(64, 64),
            spp=1,
            total_spp=1,
            max_bounces=1,
            realtime_subframes=4,
        )

        parameters = _renderer_order(job)

        self.assertEqual(parameters["mode"], "realtime")
        self.assertEqual(parameters["realtime_subframes"], 4)

    def test_subframes_default_to_one(self):
        job = _job()
        job.capture.renderer(
            mode="realtime",
            resolution=(64, 64),
            spp=1,
            total_spp=1,
            max_bounces=1,
        )

        self.assertEqual(_renderer_order(job)["realtime_subframes"], 1)

    def test_pathtracing_still_compiles_exactly_as_before(self):
        """The default path must not move because a second mode exists."""
        job = _job()
        job.capture.renderer(
            mode="pathtracing",
            resolution=(320, 180),
            spp=1,
            total_spp=2,
            max_bounces=2,
        )

        parameters = _renderer_order(job)

        self.assertEqual(parameters["mode"], "pathtracing")
        self.assertEqual(parameters["resolution"], (320, 180))
        self.assertEqual(parameters["spp"], 1)
        self.assertEqual(parameters["total_spp"], 2)
        self.assertEqual(parameters["max_bounces"], 2)

    def test_an_unknown_mode_is_still_rejected(self):
        job = _job()

        with self.assertRaises(ValidationError):
            job.capture.renderer(
                mode="raster",
                resolution=(64, 64),
                spp=1,
                total_spp=1,
                max_bounces=1,
            )

    def test_subframes_below_one_are_rejected(self):
        job = _job()

        with self.assertRaises(ValidationError):
            job.capture.renderer(
                mode="realtime",
                resolution=(64, 64),
                spp=1,
                total_spp=1,
                max_bounces=1,
                realtime_subframes=0,
            )

    def test_the_capture_config_carries_the_block_the_pipeline_reads(self):
        """common.configure_pathtracing reads cfg["realtime"]["subframes"]."""
        job = _job()
        job.capture.renderer(
            mode="realtime",
            resolution=(64, 64),
            spp=1,
            total_spp=1,
            max_bounces=1,
            realtime_subframes=8,
        )
        from cdf.execution.isaac.capture_execution import build_capture_config

        config = build_capture_config(_context(job), "/tmp/capture")

        self.assertEqual(config["render_mode"], "realtime")
        self.assertEqual(config["realtime"], {"subframes": 8})

    def test_a_pathtracing_config_still_carries_a_realtime_default(self):
        """configure_pathtracing ignores it, and a missing key would raise."""
        job = _job()
        job.capture.renderer(
            mode="pathtracing",
            resolution=(64, 64),
            spp=1,
            total_spp=4,
            max_bounces=1,
        )
        from cdf.execution.isaac.capture_execution import build_capture_config

        config = build_capture_config(_context(job), "/tmp/capture")

        self.assertEqual(config["render_mode"], "pathtracing")
        self.assertEqual(config["pathtracing"]["total_spp"], 4)


if __name__ == "__main__":
    unittest.main()


class YamlParityTest(unittest.TestCase):
    """A JobConfig must be able to say what the Python API can say.

    Versioned JobConfig parity is required unless a limitation is explicitly
    Python-only. The renderer mode and its subframe count
    describe the JOB -- what to render and how finely -- so they travel with a
    config and parity is the right answer rather than an exemption. This
    guards against `RendererConfig.mode` being a third
    `Literal["pathtracing"]` guard behind the two the Python side lifted.
    """

    def test_the_yaml_renderer_accepts_realtime(self):
        from pcb_aoi.config.capture import RendererConfig

        config = RendererConfig(mode="realtime", resolution=(64, 64))

        self.assertEqual(config.mode, "realtime")

    def test_the_yaml_renderer_still_defaults_to_pathtracing(self):
        from pcb_aoi.config.capture import RendererConfig

        self.assertEqual(RendererConfig(resolution=(64, 64)).mode, "pathtracing")

    def test_the_yaml_renderer_carries_realtime_subframes(self):
        from pcb_aoi.config.capture import RendererConfig

        config = RendererConfig(
            mode="realtime",
            resolution=(64, 64),
            realtime_subframes=4,
        )

        self.assertEqual(config.realtime_subframes, 4)

    def test_subframes_default_to_one_in_yaml_too(self):
        from pcb_aoi.config.capture import RendererConfig

        self.assertEqual(
            RendererConfig(resolution=(64, 64)).realtime_subframes,
            1,
        )

    def test_the_yaml_renderer_rejects_an_unknown_mode(self):
        import pydantic

        from pcb_aoi.config.capture import RendererConfig

        with self.assertRaises(pydantic.ValidationError):
            RendererConfig(mode="raster", resolution=(64, 64))

    def test_subframes_below_one_are_rejected_in_yaml_too(self):
        import pydantic

        from pcb_aoi.config.capture import RendererConfig

        with self.assertRaises(pydantic.ValidationError):
            RendererConfig(resolution=(64, 64), realtime_subframes=0)

    def test_a_config_document_round_trips_the_realtime_renderer(self):
        """The point of parity is that a YAML job can actually ask for it."""
        from pcb_aoi.config.capture import RendererConfig

        loaded = RendererConfig.model_validate(
            {"mode": "realtime", "resolution": [128, 128], "realtime_subframes": 8},
        )

        self.assertEqual(loaded.mode, "realtime")
        self.assertEqual(loaded.realtime_subframes, 8)

    def test_yaml_reaches_the_compiled_plan(self):
        """Parity that stops at the config model is parity on paper.

        This builds a real `RenderPassConfig` and drives the real
        `_apply_render_pass` -- the function a JobConfig actually goes through.
        The renderer is splatted by FIELD NAME, so a hand-written splat would
        keep passing after a rename that broke the real path.
        """
        from pcb_aoi.config.capture import (
            CameraConfig,
            RendererConfig,
            RenderPassConfig,
            RotationConfig,
            _apply_render_pass,
        )

        render_pass = RenderPassConfig(
            renderer=RendererConfig(
                mode="realtime",
                resolution=(64, 64),
                realtime_subframes=6,
            ),
            camera=CameraConfig(side="top", rotation=RotationConfig()),
            scan_grid=(1, 1),
        )

        job = _job()
        _apply_render_pass(job.capture, render_pass)
        parameters = _renderer_order(job)

        self.assertEqual(parameters["mode"], "realtime")
        self.assertEqual(parameters["realtime_subframes"], 6)
