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

"""The core budget: solver_options["workers"] as a hard "use at most X cores".

The budget is enforced by a semaphore installed before the footprint pool
forks, so it must hold across BOTH fan-out levels combined -- footprint
processes and within-footprint joint threads. The concurrency test exercises
the exact mechanism generate() relies on (fork-inherited gate, one slot held
per solve) with sleeps standing in for Evolver subprocesses -- no binary and
no geometry deps beyond the solver module import.
"""

import multiprocessing
import os
import sys
import time
import unittest

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Same guard as the sibling modules: where the solver's deps are absent (the
# generic deps-starved test job), skip the module rather than error discovery.
try:
    from pcb_aoi.solder_generation import surface_evolver_solver as ses
except (Exception, SystemExit):  # noqa: BLE001
    raise unittest.SkipTest("surface_evolver_solver not importable here")

# Shared with the forked pool children BY INHERITANCE (module globals set
# before the fork), exactly like the solve gate itself -- multiprocessing
# sync objects cannot be pickled into pool tasks.
_peak = None
_live = None
_lock = None


def _arm_counters(ctx):
    global _peak, _live, _lock
    _peak = ctx.Value("i", 0)
    _live = ctx.Value("i", 0)
    _lock = ctx.Lock()


def _hold_slot(_i):
    """One 'solve': take a budget slot, bump the live count, linger, drop."""
    with ses._solve_slot():
        with _lock:
            _live.value += 1
            _peak.value = max(_peak.value, _live.value)
        time.sleep(0.15)
        with _lock:
            _live.value -= 1


class TestResolveCoreBudget(unittest.TestCase):
    def test_auto_is_machine_minus_headroom(self):
        want = max(1, (os.cpu_count() or 2) - 2)
        self.assertEqual(ses._resolve_core_budget({}), want)
        self.assertEqual(ses._resolve_core_budget({"workers": 0}), want)
        self.assertEqual(ses._resolve_core_budget({"workers": None}), want)
        self.assertEqual(ses._resolve_core_budget({"workers": "junk"}), want)

    def test_positive_is_taken_verbatim(self):
        # The budget is NOT clamped to the footprint count -- a one-footprint
        # board with a big connector still spends the whole budget on the
        # joint fan.
        self.assertEqual(ses._resolve_core_budget({"workers": 3}), 3)
        self.assertEqual(ses._resolve_core_budget({"workers": 999}), 999)


class TestJointFan(unittest.TestCase):
    def test_clamped_to_distinct_solves(self):
        self.assertEqual(ses._joint_fan({}, 1), 1)
        self.assertLessEqual(ses._joint_fan({}, 4), 4)

    def test_internal_override_wins(self):
        self.assertEqual(ses._joint_fan({"joint_workers": 2}, 100), 2)

    def test_auto_uses_installed_budget(self):
        old_gate, old_budget = ses._solve_gate, ses._core_budget
        try:
            ses._install_solve_gate(3)
            self.assertEqual(ses._joint_fan({}, 100), 3)
        finally:
            ses._solve_gate, ses._core_budget = old_gate, old_budget


class TestGateCapsConcurrency(unittest.TestCase):
    def test_budget_holds_across_forked_processes(self):
        if not hasattr(os, "fork"):
            self.skipTest("fork-only mechanism (Linux runtime)")
        old_gate, old_budget = ses._solve_gate, ses._core_budget
        ctx = multiprocessing.get_context("fork")
        _arm_counters(ctx)
        try:
            ses._install_solve_gate(2)
            # Same shape as generate(): gate armed first, THEN the pool forks,
            # wider than the budget on purpose.
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
                list(pool.map(_hold_slot, range(6)))
        finally:
            ses._solve_gate, ses._core_budget = old_gate, old_budget
        self.assertEqual(_live.value, 0)
        self.assertGreater(_peak.value, 1)   # it did overlap...
        self.assertLessEqual(_peak.value, 2)  # ...but never past the budget

    def test_ungated_is_passthrough(self):
        old_gate, old_budget = ses._solve_gate, ses._core_budget
        try:
            # Direct engine callers (anything that never runs generate())
            # see no gate and must not block.
            ses._solve_gate = None
            ses._core_budget = None
            with ses._solve_slot():
                pass
        finally:
            ses._solve_gate, ses._core_budget = old_gate, old_budget


if __name__ == "__main__":
    unittest.main()
