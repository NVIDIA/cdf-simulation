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

"""Verify handlers can attribute their own time to named sub-phases.

Per-order timing found that one order, ``capture.render``, holds most of a
run. Splitting that order further is the only way to tell a one-time setup cost
-- which a persistent session amortizes -- from a per-frame cost, which it does
not.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cdf.execution.phase_timer import PhaseTimer  # noqa: E402


class PhaseTimerTest(unittest.TestCase):
    """The measurement primitive every long handler reports through."""

    def _timer(self, *readings: float) -> tuple[PhaseTimer, mock.Mock]:
        clock = mock.Mock(side_effect=list(readings))
        timer = PhaseTimer(clock=clock)
        return timer, clock

    def test_a_phase_reports_its_seconds_and_one_call(self):
        timer, _ = self._timer(0.0, 4.0)

        with timer.phase("warmup"):
            pass

        self.assertEqual(timer.report(), {"warmup": {"seconds": 4.0, "calls": 1}})

    def test_repeated_phases_accumulate_and_count(self):
        timer, _ = self._timer(0.0, 4.0, 10.0, 12.5)

        for _ in range(2):
            with timer.phase("warmup"):
                pass

        self.assertEqual(timer.report(), {"warmup": {"seconds": 6.5, "calls": 2}})

    def test_phases_are_reported_costliest_first(self):
        timer, _ = self._timer(0.0, 1.0, 1.0, 9.0, 9.0, 12.0)

        with timer.phase("cheap"):
            pass
        with timer.phase("expensive"):
            pass
        with timer.phase("middling"):
            pass

        self.assertEqual(
            list(timer.report()),
            ["expensive", "middling", "cheap"],
        )

    def test_a_raising_phase_is_still_recorded(self):
        """A failed render must still say how far it got."""
        timer, _ = self._timer(0.0, 3.0)

        with self.assertRaises(RuntimeError), timer.phase("warmup"):
            raise RuntimeError("boom")

        self.assertEqual(timer.report()["warmup"]["seconds"], 3.0)

    def test_a_phase_spanning_an_await_measures_the_whole_wait(self):
        timer = PhaseTimer()

        async def run() -> None:
            with timer.phase("step"):
                await asyncio.sleep(0.02)

        asyncio.run(run())

        self.assertGreaterEqual(timer.report()["step"]["seconds"], 0.015)

    def test_seconds_are_rounded_for_a_readable_report(self):
        timer, _ = self._timer(0.0, 1.0 / 3.0)

        with timer.phase("step"):
            pass

        self.assertEqual(timer.report()["step"]["seconds"], 0.333)

    def test_an_unused_timer_reports_nothing(self):
        self.assertEqual(PhaseTimer().report(), {})

    def test_total_sums_every_phase(self):
        timer, _ = self._timer(0.0, 1.0, 1.0, 2.5)

        with timer.phase("a"):
            pass
        with timer.phase("b"):
            pass

        self.assertEqual(timer.total_s, 2.5)

    def test_nested_names_are_kept_distinct(self):
        timer, _ = self._timer(0.0, 1.0, 3.0, 5.0)

        with timer.phase("render"), timer.phase("render.step"):
            pass

        self.assertEqual(
            timer.report(),
            {
                "render": {"seconds": 5.0, "calls": 1},
                "render.step": {"seconds": 2.0, "calls": 1},
            },
        )


if __name__ == "__main__":
    unittest.main()
