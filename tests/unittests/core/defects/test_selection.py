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

"""Tests for exact-count and ratio defect targeting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_STANDALONE = Path(__file__).resolve().parents[4] / "scripts" / "sdg" / "standalone"
if str(_STANDALONE) not in sys.path:
    sys.path.insert(0, str(_STANDALONE))

from defects_utils.selection import (
    describe_targeting,
    selection_count,
)


class SelectionCountTest(unittest.TestCase):
    """Verify low-level structural-defect targeting semantics."""

    def test_exact_count_is_returned(self) -> None:
        self.assertEqual(selection_count({"count": 2}, 5), 2)

    def test_exact_count_rejects_insufficient_eligibility(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 1.*eligible"):
            selection_count({"count": 2}, 1, operation_name="shift")

    def test_ratio_preserves_legacy_minimum_and_clamping(self) -> None:
        self.assertEqual(selection_count({"ratio": 0.0}, 4), 1)
        self.assertEqual(
            selection_count({"ratio": 1.0}, 2, ratio_base_count=8),
            2,
        )

    def test_targeting_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            selection_count({"count": 2, "ratio": 0.5}, 5)



class DescribeTargetingTest(unittest.TestCase):
    """Reporting must survive both targeting modes.

    ``prepare_defects`` logs what an operation asked for AFTER authoring the
    transforms. Reading ``cfg["ratio"]`` there raised ``KeyError`` for every
    count-targeted operation, which surfaced as a failed job order over an
    already-modified stage -- the defects had been applied and the caller
    was told the run failed.
    """

    def test_count_targeting_is_described(self) -> None:
        self.assertEqual(describe_targeting({"count": 3}), "count=3 requested")

    def test_ratio_targeting_is_described_as_a_percentage(self) -> None:
        self.assertEqual(describe_targeting({"ratio": 0.25}), "25.0% requested")

    def test_a_zero_ratio_is_not_mistaken_for_absent(self) -> None:
        # 0.0 is falsy, and the operation still ran: selection_count floors
        # a ratio request at one component.
        self.assertEqual(describe_targeting({"ratio": 0.0}), "0.0% requested")

    def test_neither_mode_is_reported_rather_than_raising(self) -> None:
        # This runs on the failure path of an operation that already
        # changed the stage; it must never be the thing that raises.
        self.assertEqual(describe_targeting({}), "no targeting given")

    def test_every_mode_selection_count_accepts_is_describable(self) -> None:
        # The two functions read the same keys, so they must not disagree
        # about which configurations are meaningful.
        for config in ({"count": 2}, {"ratio": 0.5}):
            with self.subTest(config=config):
                selection_count(config, 5)
                self.assertNotIn("no targeting", describe_targeting(config))

if __name__ == "__main__":
    unittest.main()
