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

"""The shared schedule and acceptance rule reproduce the old behaviour.

``capture_3d_aoi.py`` used to carry private copies of three decisions -- which
sample gets which operation and component, where the camera aims, and whether a
captured scene is acceptable. Those copies now live in ``cdf.execution`` so
they can be unit tested and shared with the JobOrder runtime.

This file is the safety net for that move. Each ``_legacy_*`` function below is
the pre-refactor implementation, copied verbatim, and each test asserts the new
module agrees with it exactly. If one of these ever fails, the answer is not to
adjust a bound -- the refactor was supposed to be behaviour-preserving, so a
divergence means something really changed.
"""

import itertools
import math
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cdf.execution.acceptance import (  # noqa: E402
    AcceptancePolicy,
    ViewObservation,
)
from cdf.execution.sample_schedule import (  # noqa: E402
    SampleSchedule,
    TargetOffsetPolicy,
)

_DEFECT_TYPES = ("tombstone", "shift", "sideflip", "missing", "reverse_polarity")

_POOL = (
    "/Board/Top/PASSIVE_SMD_0402/R1",
    "/Board/Top/PASSIVE_SMD_0402/R2",
    "/Board/Top/PASSIVE_SMD_0402/R3",
    "/Board/Top/PASSIVE_SMD_0402/R4",
    "/Board/Top/PASSIVE_SMD_0402/R5",
    "/Board/Top/PASSIVE_SMD_0603/R6",
    "/Board/Top/CAP_TANT_SMD/C1",
    "/Board/Top/CAP_TANT_SMD/C2",
    "/Board/Top/IC_QFN/U1",
)


# --------------------------------------------------------------------------
# The pre-refactor implementations, copied verbatim from capture_3d_aoi.py.
# --------------------------------------------------------------------------

def _legacy_schedule_index(base, sample_idx):
    return max(0, base + int(sample_idx))


def _legacy_scheduled_defect_type(base, sample_idx):
    return _DEFECT_TYPES[_legacy_schedule_index(base, sample_idx) % len(_DEFECT_TYPES)]


def _legacy_balanced_component_candidates(pool, base, sample_idx):
    groups = {}
    for path in sorted(set(pool)):
        family = path.rstrip("/").rsplit("/", 2)[-2]
        groups.setdefault(family, []).append(path)
    families = sorted(groups)
    if not families:
        return
    idx = _legacy_schedule_index(base, sample_idx)
    family_start = idx % len(families)
    member_round = idx // len(families)
    max_members = max(len(paths) for paths in groups.values())
    seen = set()
    for member_delta in range(max_members):
        for family_delta in range(len(families)):
            family = families[(family_start + family_delta) % len(families)]
            members = groups[family]
            path = members[(member_round + member_delta) % len(members)]
            if path not in seen:
                seen.add(path)
                yield path


def _legacy_sample_target_offset(sequence_idx, qa_attempt, lo_mm, hi_mm, radial_bins):
    lo, hi = sorted((max(0.0, lo_mm), max(0.0, hi_mm)))
    sequence_index = max(0, int(sequence_idx)) + max(0, int(qa_attempt) - 1) * 997
    radial_bin = sequence_index % radial_bins
    area_fraction = (radial_bin + 0.5) / radial_bins
    radius = math.sqrt(lo * lo + area_fraction * (hi * hi - lo * lo))
    theta = sequence_index * math.pi * (3.0 - math.sqrt(5.0))
    return radius * math.cos(theta), radius * math.sin(theta)


def _legacy_scene_qa(views, threshold, min_visible, min_invisible, min_fully, reject_truncated):
    visible = sum(1 for v in views if v["visible"])
    invisible = len(views) - visible
    fully_visible = 0
    truncated_count = 0
    occluded_count = 0
    for view in views:
        ratio = view["occlusion_ratio"]
        is_truncated = bool(view["truncated"])
        is_visible = bool(view["visible"])
        is_occluded = bool(
            not is_visible or (ratio is not None and ratio > threshold))
        truncated_count += int(is_truncated)
        occluded_count += int(is_occluded)
        fully_visible += int(is_visible and not is_occluded and not is_truncated)
    accepted = (
        visible >= min_visible
        and invisible >= min_invisible
        and fully_visible >= min_fully
        and (not reject_truncated or truncated_count == 0)
    )
    return accepted, visible, invisible, fully_visible, occluded_count, truncated_count


# --------------------------------------------------------------------------


class ScheduleEquivalenceTest(unittest.TestCase):
    """``SampleSchedule`` reproduces the script's private scheduling."""

    def test_operation_and_component_order_match_over_200_samples(self):
        for base in (0, 7, 20260713):
            schedule = SampleSchedule(
                operations=_DEFECT_TYPES,
                index_base=base,
            )
            for sample in range(200):
                with self.subTest(base=base, sample=sample):
                    self.assertEqual(
                        schedule.schedule_index(sample),
                        _legacy_schedule_index(base, sample),
                    )
                    self.assertEqual(
                        schedule.operation_for(sample),
                        _legacy_scheduled_defect_type(base, sample),
                    )
                    self.assertEqual(
                        tuple(schedule.component_order(sorted(set(_POOL)), sample)),
                        tuple(_legacy_balanced_component_candidates(
                            _POOL, base, sample)),
                    )


class TargetOffsetEquivalenceTest(unittest.TestCase):
    """``TargetOffsetPolicy`` reproduces the script's aim-point sampling."""

    def test_offsets_are_bit_identical_over_8000_draws(self):
        cases = 0
        for lo, hi, bins in ((0.0, 0.75, 4), (0.25, 2.0, 3), (0.0, 1.0, 8)):
            policy = TargetOffsetPolicy(min_mm=lo, max_mm=hi, radial_bins=bins)
            for attempt in (1, 2, 3, 4):
                for sequence_index in range(667):
                    got = policy.sample(sequence_index, attempt=attempt)
                    want = _legacy_sample_target_offset(
                        sequence_index, attempt, lo, hi, bins)
                    if got != want:
                        self.fail(
                            "offset diverged at lo=%s hi=%s bins=%s seq=%s attempt=%s: "
                            "%r != %r" % (lo, hi, bins, sequence_index, attempt, got, want))
                    cases += 1
        self.assertGreaterEqual(cases, 8000)


class AcceptanceEquivalenceTest(unittest.TestCase):
    """``AcceptancePolicy`` reproduces the script's inline scene QA."""

    _VIEW_STATES = (
        {"visible": False, "occlusion_ratio": None, "truncated": False},
        {"visible": True, "occlusion_ratio": None, "truncated": False},
        {"visible": True, "occlusion_ratio": 0.1, "truncated": False},
        {"visible": True, "occlusion_ratio": 0.8, "truncated": False},
        {"visible": True, "occlusion_ratio": 0.1, "truncated": True},
        {"visible": False, "occlusion_ratio": 0.8, "truncated": True},
    )

    def test_verdict_and_counts_match_over_every_combination(self):
        thresholds = (0.2, 0.5, 0.9)
        minima = ((0, 0, 1), (1, 0, 1), (2, 1, 0), (0, 2, 2))
        cases = 0
        for views in itertools.product(self._VIEW_STATES, repeat=3):
            for threshold in thresholds:
                for min_visible, min_invisible, min_fully in minima:
                    for reject_truncated in (True, False):
                        policy = AcceptancePolicy(
                            occlusion_threshold=threshold,
                            min_visible_views=min_visible,
                            min_invisible_views=min_invisible,
                            min_fully_visible_views=min_fully,
                            reject_truncated=reject_truncated,
                        )
                        report = policy.evaluate(
                            ViewObservation(
                                view_id="v%d" % index,
                                visible=view["visible"],
                                occlusion_ratio=view["occlusion_ratio"],
                                truncated=view["truncated"],
                            )
                            for index, view in enumerate(views)
                        )
                        want = _legacy_scene_qa(
                            views, threshold, min_visible, min_invisible,
                            min_fully, reject_truncated)
                        got = (
                            report.accepted,
                            report.visible_views,
                            report.invisible_views,
                            report.fully_visible_views,
                            report.occluded_views,
                            report.truncated_views,
                        )
                        if got != want:
                            self.fail(
                                "acceptance diverged for views=%r threshold=%s "
                                "minima=%r reject_truncated=%s: %r != %r"
                                % (views, threshold, (min_visible, min_invisible,
                                                      min_fully),
                                   reject_truncated, got, want))
                        cases += 1
        self.assertGreaterEqual(cases, 2916)


if __name__ == "__main__":
    unittest.main()
