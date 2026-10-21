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

"""Tests for the executor-owned sample schedule."""

import math
import sys
import unittest
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cdf.errors import ValidationError  # noqa: E402
from cdf.execution.sample_schedule import (  # noqa: E402
    SampleSchedule,
    TargetOffsetPolicy,
    default_group_key,
)

_POOL = (
    "/Board/Top/PASSIVE_SMD_0402/R1",
    "/Board/Top/PASSIVE_SMD_0402/R2",
    "/Board/Top/PASSIVE_SMD_0402/R3",
    "/Board/Top/PASSIVE_SMD_0402/R4",
    "/Board/Top/PASSIVE_SMD_0402/R5",
    "/Board/Top/CAP_TANT_SMD/C1",
    "/Board/Top/IC_QFN/U1",
)


class GroupKeyTest(unittest.TestCase):

    def test_parent_prim_names_the_group(self):
        self.assertEqual(
            default_group_key("/Board/Top/PASSIVE_SMD_0402/R12"),
            "PASSIVE_SMD_0402",
        )

    def test_trailing_slash_is_ignored(self):
        self.assertEqual(default_group_key("/Board/Top/CAP_TANT_SMD/C1/"), "CAP_TANT_SMD")

    def test_short_path_falls_back_to_itself(self):
        self.assertEqual(default_group_key("R12"), "R12")


class TargetOffsetPolicyTest(unittest.TestCase):

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValidationError):
            TargetOffsetPolicy(mode="spiral")

    def test_rejects_inverted_bounds(self):
        with self.assertRaises(ValidationError):
            TargetOffsetPolicy(min_mm=2.0, max_mm=1.0)

    def test_rejects_negative_radius(self):
        with self.assertRaises(ValidationError):
            TargetOffsetPolicy(min_mm=-1.0)

    def test_rejects_zero_radial_bins(self):
        with self.assertRaises(ValidationError):
            TargetOffsetPolicy(radial_bins=0)

    def test_balanced_disk_is_deterministic_and_bounded(self):
        policy = TargetOffsetPolicy(mode="balanced_disk", max_mm=1.5)
        for index in range(64):
            x, y = policy.sample(index)
            self.assertEqual((x, y), policy.sample(index))
            self.assertLessEqual(math.hypot(x, y), 1.5 + 1e-9)

    def test_balanced_disk_covers_every_radial_bin(self):
        policy = TargetOffsetPolicy(max_mm=4.0, radial_bins=4)
        radii = sorted({round(math.hypot(*policy.sample(i)), 6) for i in range(4)})
        self.assertEqual(len(radii), 4)

    def test_balanced_disk_spreads_directions(self):
        policy = TargetOffsetPolicy(max_mm=1.0)
        quadrants = Counter()
        for index in range(64):
            x, y = policy.sample(index)
            quadrants[(x >= 0, y >= 0)] += 1
        self.assertEqual(len(quadrants), 4)
        self.assertGreater(min(quadrants.values()), 8)

    def test_retry_moves_the_aim_point(self):
        policy = TargetOffsetPolicy(max_mm=1.0)
        self.assertNotEqual(policy.sample(3, attempt=1), policy.sample(3, attempt=2))

    def test_min_radius_is_respected(self):
        policy = TargetOffsetPolicy(min_mm=1.0, max_mm=2.0)
        for index in range(32):
            self.assertGreaterEqual(math.hypot(*policy.sample(index)), 1.0 - 1e-9)

class SampleScheduleConstructionTest(unittest.TestCase):

    def test_requires_an_operation(self):
        with self.assertRaises(ValidationError):
            SampleSchedule(operations=())

    def test_rejects_duplicate_operations(self):
        with self.assertRaises(ValidationError):
            SampleSchedule(operations=("shift", "shift"))

    def test_rejects_blank_operation(self):
        with self.assertRaises(ValidationError):
            SampleSchedule(operations=("shift", "  "))

    def test_rejects_unknown_sampling_policy(self):
        with self.assertRaises(ValidationError):
            SampleSchedule(operations=("shift",), sampling="balanced")

    def test_rejects_negative_index_base(self):
        with self.assertRaises(ValidationError):
            SampleSchedule(operations=("shift",), index_base=-1)


class SampleScheduleSamplingTest(unittest.TestCase):

    def setUp(self):
        self.schedule = SampleSchedule(operations=("shift", "tombstone", "missing"))

    def test_schedule_index_offsets_from_the_base(self):
        schedule = SampleSchedule(operations=("shift",), index_base=100)
        self.assertEqual(schedule.schedule_index(0), 100)
        self.assertEqual(schedule.schedule_index(7), 107)

    def test_variant_index_tracks_the_schedule_index(self):
        schedule = SampleSchedule(operations=("shift",), index_base=5)
        self.assertEqual(schedule.variant_index(2), schedule.schedule_index(2))

    def test_stratified_operations_cycle_evenly(self):
        drawn = Counter(self.schedule.operation_for(index) for index in range(30))
        self.assertEqual(set(drawn), {"shift", "tombstone", "missing"})
        self.assertEqual(set(drawn.values()), {10})

    def test_stratified_operations_are_reproducible(self):
        first = [self.schedule.operation_for(index) for index in range(12)]
        second = [self.schedule.operation_for(index) for index in range(12)]
        self.assertEqual(first, second)

    def test_index_base_shifts_the_operation_phase(self):
        shifted = SampleSchedule(
            operations=("shift", "tombstone", "missing"),
            index_base=1,
        )
        self.assertEqual(shifted.operation_for(0), self.schedule.operation_for(1))

    def test_component_order_is_a_permutation_of_the_pool(self):
        order = self.schedule.component_order(_POOL, 0)
        self.assertEqual(sorted(order), sorted(_POOL))

    def test_component_order_deduplicates_candidates(self):
        order = self.schedule.component_order([*_POOL, _POOL[0]], 0)
        self.assertEqual(len(order), len(set(_POOL)))

    def test_component_order_is_empty_without_candidates(self):
        self.assertEqual(self.schedule.component_order([], 0), ())

    def test_stratified_order_alternates_groups_first(self):
        order = self.schedule.component_order(_POOL, 0)
        leading_groups = [default_group_key(path) for path in order[:3]]
        self.assertEqual(len(set(leading_groups)), 3)

    def test_stratified_first_pick_balances_small_groups(self):
        picks = Counter(
            default_group_key(self.schedule.component_order(_POOL, index)[0])
            for index in range(30)
        )
        self.assertEqual(set(picks), {"PASSIVE_SMD_0402", "CAP_TANT_SMD", "IC_QFN"})
        self.assertEqual(set(picks.values()), {10})

    def test_stratified_first_pick_rotates_within_a_group(self):
        picks = {
            self.schedule.component_order(_POOL, index)[0]
            for index in range(30)
            if default_group_key(self.schedule.component_order(_POOL, index)[0])
            == "PASSIVE_SMD_0402"
        }
        self.assertEqual(len(picks), 5)

    def test_custom_group_key_is_honored(self):
        schedule = SampleSchedule(
            operations=("shift",),
            group_key=lambda path: path.rsplit("/", 1)[-1][0],
        )
        order = schedule.component_order(_POOL, 0)
        self.assertEqual(sorted(order), sorted(_POOL))
        self.assertNotEqual(order[0][-2], order[1][-2])

    def test_camera_target_offset_follows_the_schedule_index(self):
        schedule = SampleSchedule(operations=("shift",), index_base=9)
        self.assertEqual(
            schedule.camera_target_offset(1),
            schedule.target_offset.sample(10),
        )

    def test_camera_target_offset_retry_differs(self):
        self.assertNotEqual(
            self.schedule.camera_target_offset(2, attempt=1),
            self.schedule.camera_target_offset(2, attempt=2),
        )


class SampleSchedulePreviewTest(unittest.TestCase):

    def test_preview_describes_each_sample(self):
        schedule = SampleSchedule(operations=("shift", "tombstone"), index_base=4)
        preview = schedule.preview(3)
        self.assertEqual(len(preview), 3)
        self.assertEqual(
            [entry["operation"] for entry in preview],
            ["shift", "tombstone", "shift"],
        )
        self.assertEqual(
            [entry["schedule_index"] for entry in preview],
            [4, 5, 6],
        )
        self.assertEqual(len(preview[0]["camera_target_offset_mm"]), 2)

    def test_preview_of_zero_samples_is_empty(self):
        self.assertEqual(SampleSchedule(operations=("shift",)).preview(0), ())

