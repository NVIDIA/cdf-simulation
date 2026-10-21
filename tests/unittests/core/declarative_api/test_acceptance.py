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

"""Tests for the captured-scene acceptance rule."""

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cdf.errors import ValidationError  # noqa: E402
from cdf.execution.acceptance import (  # noqa: E402
    AcceptancePolicy,
    ViewObservation,
)


class AcceptancePolicyTest(unittest.TestCase):

    def test_rejects_out_of_range_threshold(self):
        with self.assertRaises(ValidationError):
            AcceptancePolicy(occlusion_threshold=1.5)

    def test_rejects_negative_minimum(self):
        with self.assertRaises(ValidationError):
            AcceptancePolicy(min_visible_views=-1)

    def test_invisible_view_is_never_fully_visible(self):
        verdict = AcceptancePolicy().classify(
            ViewObservation("top_down", visible=False, occlusion_ratio=0.0),
        )
        self.assertTrue(verdict.occluded)
        self.assertFalse(verdict.fully_visible)

    def test_occlusion_threshold_is_exclusive(self):
        policy = AcceptancePolicy(occlusion_threshold=0.5)
        at_threshold = policy.classify(
            ViewObservation("v", visible=True, occlusion_ratio=0.5),
        )
        above_threshold = policy.classify(
            ViewObservation("v", visible=True, occlusion_ratio=0.51),
        )
        self.assertFalse(at_threshold.occluded)
        self.assertTrue(above_threshold.occluded)

    def test_missing_occlusion_ratio_does_not_occlude(self):
        verdict = AcceptancePolicy().classify(
            ViewObservation("v", visible=True, occlusion_ratio=None),
        )
        self.assertFalse(verdict.occluded)
        self.assertTrue(verdict.fully_visible)

    def test_default_policy_accepts_one_clean_view(self):
        report = AcceptancePolicy().evaluate([ViewObservation("top_down", visible=True)])
        self.assertTrue(report.accepted)
        self.assertEqual(report.reasons, ())
        self.assertEqual(report.fully_visible_views, 1)

    def test_default_policy_rejects_a_truncated_scene(self):
        report = AcceptancePolicy().evaluate(
            [ViewObservation("top_down", visible=True, truncated=True)],
        )
        self.assertFalse(report.accepted)
        self.assertEqual(len(report.reasons), 2)
        self.assertEqual(report.truncated_views, 1)

    def test_truncation_can_be_tolerated(self):
        policy = AcceptancePolicy(reject_truncated=False, min_fully_visible_views=0)
        report = policy.evaluate(
            [ViewObservation("top_down", visible=True, truncated=True)],
        )
        self.assertTrue(report.accepted)

    def test_minimum_invisible_views_is_enforced(self):
        policy = AcceptancePolicy(min_invisible_views=1)
        observations = [
            ViewObservation("a", visible=True),
            ViewObservation("b", visible=True),
        ]
        self.assertFalse(policy.evaluate(observations).accepted)
        observations.append(ViewObservation("c", visible=False))
        self.assertTrue(policy.evaluate(observations).accepted)

    def test_minimum_visible_views_is_enforced(self):
        policy = AcceptancePolicy(min_visible_views=2, min_fully_visible_views=0)
        report = policy.evaluate([ViewObservation("a", visible=True)])
        self.assertFalse(report.accepted)
        self.assertIn("visible views 1 < 2", report.reasons)

    def test_counts_summarize_the_scene(self):
        policy = AcceptancePolicy(min_fully_visible_views=0)
        report = policy.evaluate(
            [
                ViewObservation("a", visible=True),
                ViewObservation("b", visible=True, occlusion_ratio=0.9),
                ViewObservation("c", visible=False),
            ],
        )
        self.assertEqual(
            report.counts(),
            {
                "visible_views": 2,
                "invisible_views": 1,
                "occluded_views": 2,
                "truncated_views": 0,
                "fully_visible_views": 1,
            },
        )

    def test_empty_scene_is_rejected_by_default(self):
        self.assertFalse(AcceptancePolicy().evaluate([]).accepted)

    def test_expected_absence_policy_accepts_a_fully_invisible_scene(self):
        # Expected absence semantics: the `missing` defect pass
        # succeeds precisely when the component is gone from every view, which
        # is the opposite of what the default (>=1 fully visible view) policy
        # accepts. Scoring a missing pass against the default policy made
        # defect_ok permanently False.
        views = ["top_down", "oblique_N_45", "oblique_S_45", "oblique_E_45", "oblique_W_45"]
        policy = AcceptancePolicy(
            min_visible_views=0,
            min_invisible_views=len(views),
            min_fully_visible_views=0,
            reject_truncated=False,
        )
        all_absent = [ViewObservation(v, visible=False) for v in views]
        self.assertTrue(policy.evaluate(all_absent).accepted)

    def test_expected_absence_policy_rejects_a_still_visible_component(self):
        views = ["top_down", "oblique_N_45", "oblique_S_45", "oblique_E_45", "oblique_W_45"]
        policy = AcceptancePolicy(
            min_visible_views=0,
            min_invisible_views=len(views),
            min_fully_visible_views=0,
            reject_truncated=False,
        )
        one_still_visible = [ViewObservation(v, visible=(v == "top_down")) for v in views]
        self.assertFalse(policy.evaluate(one_still_visible).accepted)


if __name__ == "__main__":
    unittest.main()
