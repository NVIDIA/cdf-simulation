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

"""Closing openings depends on the stitch that produced the image it reads.

Declaration order is not a semantic dependency. The two orders name the same
stitched image -- same capture pass, same directory, same filename -- and that
naming is what the plan is built from, so a job may declare them in either
order and a job that names an image nothing produces fails to compile.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import ClassVar

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi CDF domain
from cdf import CDFJob, ValidationError
from pcb_aoi.config.postprocess import (
    PostprocessConfig,
    apply_postprocess,
)

PASS_NAME = "full_board"


def _capture_job() -> CDFJob:
    """Return a job with one named capture pass and nothing else declared."""
    job = CDFJob(domain="pcb-aoi", random_seed=0)
    job.scene.configure(usd_path="board.usd", root_path="/Board")
    with job.capture.render_pass(PASS_NAME) as render_pass:
        render_pass.renderer(
            mode="pathtracing",
            resolution=(64, 64),
            spp=1,
            total_spp=1,
            max_bounces=1,
        )
        render_pass.camera(side="bottom", rotation_range={"z": -90})
        render_pass.scan_grid(x_count=1, y_count=1)
        render_pass.outputs(rgb=True, semantic_segmentation=True)
    return job


def _postprocess(job: CDFJob, blocks: dict) -> None:
    """Apply a YAML postprocess block to a job, as a config-driven run does."""
    apply_postprocess(job, PostprocessConfig.model_validate(blocks))


def _order(plan, order_type: str):
    """Return the single order of one type in a compiled plan."""
    matches = [order for order in plan.orders if order.order_type == order_type]
    if len(matches) != 1:
        message = f"expected exactly one {order_type}, found {len(matches)}"
        raise AssertionError(message)
    return matches[0]


class OpeningClosurePlanTest(unittest.TestCase):
    """The compiled plan wires closure to its producer, not to its neighbour."""

    STITCH: ClassVar[dict[str, object]] = {
        "source_pass": PASS_NAME,
        "image_type": "semantic_segmentation",
        "output_subdir": "stitched",
        "output_filename": "mask.png",
    }
    CLOSURE: ClassVar[dict[str, object]] = {
        "source_pass": PASS_NAME,
        "class_name": "hole",
        "output_subdir": "stitched",
        "output_filename": "mask.png",
    }

    def test_closure_depends_on_the_stitch_that_writes_its_image(self):
        """The dependency is the producing order, not every render."""
        job = _capture_job()
        _postprocess(job, {"stitch": self.STITCH, "close_openings": self.CLOSURE})

        plan = job.compile_orders(output_dir="output")

        stitch = _order(plan, "postprocess.stitch")
        closure = _order(plan, "postprocess.close_openings")
        self.assertEqual(closure.depends_on, (stitch.order_id,))

    def test_declaring_closure_first_compiles_to_the_same_plan(self):
        """Order of declaration carries no meaning, so it must not change one."""
        after = _capture_job()
        _postprocess(after, {"stitch": self.STITCH, "close_openings": self.CLOSURE})
        before = _capture_job()
        _postprocess(before, {"close_openings": self.CLOSURE, "stitch": self.STITCH})

        wired = [
            _order(
                job.compile_orders(output_dir="output"), "postprocess.close_openings"
            ).depends_on
            for job in (after, before)
        ]

        self.assertEqual(wired[0], wired[1])

    def test_naming_an_image_nothing_produces_is_rejected(self):
        """A plan that reads a file no order writes is not a plan."""
        job = _capture_job()
        _postprocess(
            job,
            {
                "stitch": self.STITCH,
                "close_openings": {**self.CLOSURE, "output_filename": "other.png"},
            },
        )

        with self.assertRaises(ValidationError) as raised:
            job.compile_orders(output_dir="output")

        message = str(raised.exception)
        self.assertIn("nothing", message)
        self.assertIn("other.png", message)

    def test_two_stitches_writing_one_image_are_rejected(self):
        """Ambiguity has to be refused; picking one silently would be a guess."""
        job = _capture_job()
        _postprocess(
            job,
            {
                "stitch": [self.STITCH, dict(self.STITCH)],
                "close_openings": self.CLOSURE,
            },
        )

        with self.assertRaises(ValidationError) as raised:
            job.compile_orders(output_dir="output")

        self.assertIn("more than one", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
