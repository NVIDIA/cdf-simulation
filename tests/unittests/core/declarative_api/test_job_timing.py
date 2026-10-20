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

"""Verify every executed JobOrder reports how long it took.

A pipeline that cannot say where its wall-clock went cannot be optimized, and
until now a CDF run reported only its total. The timing is part of the result
contract, so it is transported across the Isaac process boundary and survives a
handler that raises.
"""

import asyncio
import sys
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cdf import (  # noqa: E402
    IsaacInProcessExecutor,
    JobExecutionError,
    JobOrder,
    JobOrderPlan,
    JobOrderResult,
    JobResult,
)
from cdf.execution.isaac.subprocess_executor import _decode_result  # noqa: E402


def _plan(*order_types: str) -> JobOrderPlan:
    """Build a plan with one order per supplied type."""
    return JobOrderPlan(
        orders=tuple(JobOrder(order_id=name, order_type=name) for name in order_types),
    )


class JobOrderResultContractTest(unittest.TestCase):
    """The added field must not disturb the existing result contract."""

    def test_duration_defaults_to_zero_and_keeps_positional_construction(self):
        result = JobOrderResult("load", "scene.load", "succeeded")

        self.assertEqual(result.duration_s, 0.0)
        self.assertEqual(result.output, None)
        self.assertEqual(result.error, "")

    def test_duration_round_trips_through_the_transport_encoding(self):
        encoded = asdict(
            JobOrderResult("load", "scene.load", "succeeded", duration_s=1.25),
        )

        self.assertEqual(encoded["duration_s"], 1.25)
        self.assertEqual(JobOrderResult(**encoded).duration_s, 1.25)

    def test_decode_accepts_a_payload_from_an_older_child(self):
        """An Isaac child that predates the field must still decode."""
        decoded = _decode_result(
            {
                "success": True,
                "order_results": [
                    {
                        "order_id": "load",
                        "order_type": "scene.load",
                        "status": "succeeded",
                        "output": None,
                        "error": "",
                    },
                ],
            },
        )

        self.assertTrue(decoded.success)
        self.assertEqual(decoded.order_results[0].duration_s, 0.0)
        self.assertEqual(decoded.startup_s, 0.0)

    def test_decode_carries_startup_and_order_durations(self):
        decoded = _decode_result(
            {
                "success": True,
                "startup_s": 42.0,
                "order_results": [
                    {
                        "order_id": "load",
                        "order_type": "scene.load",
                        "status": "succeeded",
                        "output": None,
                        "error": "",
                        "duration_s": 3.0,
                    },
                ],
            },
        )

        self.assertEqual(decoded.startup_s, 42.0)
        self.assertEqual(decoded.timings, {"load": 3.0})


class JobResultTimingViewTest(unittest.TestCase):
    """The aggregate views a profile is read through."""

    def _result(self) -> JobResult:
        return JobResult(
            success=True,
            order_results=(
                JobOrderResult("load", "scene.load", "succeeded", duration_s=4.0),
                JobOrderResult(
                    "light",
                    "appearance.lighting",
                    "succeeded",
                    duration_s=0.5,
                ),
                JobOrderResult(
                    "solder",
                    "solder_joints.generate",
                    "succeeded",
                    duration_s=9.0,
                ),
            ),
            startup_s=100.0,
        )

    def test_timings_are_keyed_by_order_id(self):
        self.assertEqual(
            self._result().timings,
            {"load": 4.0, "light": 0.5, "solder": 9.0},
        )

    def test_total_includes_startup(self):
        self.assertEqual(self._result().total_duration_s, 113.5)

    def test_slowest_orders_rank_by_duration(self):
        self.assertEqual(
            self._result().slowest_orders(limit=2),
            (("solder", "solder_joints.generate", 9.0), ("load", "scene.load", 4.0)),
        )

    def test_slowest_orders_ignores_an_order_that_was_not_timed(self):
        result = JobResult(
            success=False,
            order_results=(
                JobOrderResult("load", "scene.load", "failed", error="boom"),
            ),
        )

        self.assertEqual(result.slowest_orders(), ())
        self.assertEqual(result.total_duration_s, 0.0)


class InProcessExecutorTimingTest(unittest.TestCase):
    """The executor is the one choke point every order passes through."""

    def test_every_order_is_timed(self):
        clock = iter([0.0, 2.0, 2.0, 5.5])
        executor = IsaacInProcessExecutor(
            {
                "scene.load": lambda order, context: None,
                "capture.render": lambda order, context: None,
            },
        )

        with mock.patch(
            "cdf.execution.isaac.in_process_executor.perf_counter",
            side_effect=lambda: next(clock),
        ):
            result = executor.execute(_plan("scene.load", "capture.render"))

        self.assertEqual(result.timings, {"scene.load": 2.0, "capture.render": 3.5})

    def test_a_failed_order_still_reports_its_duration(self):
        clock = iter([0.0, 7.0])

        def explode(order, context):
            raise RuntimeError("boom")

        executor = IsaacInProcessExecutor({"scene.load": explode})

        with (
            mock.patch(
                "cdf.execution.isaac.in_process_executor.perf_counter",
                side_effect=lambda: next(clock),
            ),
            self.assertRaises(JobExecutionError) as caught,
        ):
            executor.execute(_plan("scene.load"))

        partial = caught.exception.result
        self.assertEqual(partial.order_results[0].duration_s, 7.0)

    def test_async_execution_is_timed_across_the_await(self):
        clock = iter([0.0, 3.0])

        async def slow(order, context):
            return "done"

        executor = IsaacInProcessExecutor({"scene.load": slow})

        with mock.patch(
            "cdf.execution.isaac.in_process_executor.perf_counter",
            side_effect=lambda: next(clock),
        ):
            result = asyncio.run(executor.execute_async(_plan("scene.load")))

        self.assertEqual(result.timings, {"scene.load": 3.0})

    def test_measured_durations_are_real_and_ordered(self):
        """Guard against a stubbed clock hiding a wiring mistake."""
        from time import sleep

        executor = IsaacInProcessExecutor(
            {
                "scene.load": lambda order, context: sleep(0.02),
                "appearance.lighting": lambda order, context: None,
            },
        )
        timings = executor.execute(
            _plan("scene.load", "appearance.lighting"),
        ).timings

        self.assertGreaterEqual(timings["scene.load"], 0.015)
        self.assertLess(timings["appearance.lighting"], timings["scene.load"])


class _SimulationApp:
    """Drive the host event loop while recording application lifecycle."""

    def __init__(self, settings: dict[str, object]) -> None:
        self.settings = settings
        self.closed = False

    def is_running(self) -> bool:
        """Report that the test application remains active."""
        return True

    def update(self) -> None:
        """Advance the asyncio task by one turn."""
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0))

    def close(self) -> None:
        """Record application shutdown."""
        self.closed = True


class AppRunnerStartupTimingTest(unittest.TestCase):
    """Booting Kit is the largest single cost in a batch run; name it."""

    def test_startup_is_measured_around_the_application_construction(self):
        clock = iter([1.0, 61.0])
        executor = mock.Mock()
        executor.execute_async = mock.AsyncMock(return_value=JobResult(success=True))
        isaacsim = types.ModuleType("isaacsim")
        isaacsim.SimulationApp = mock.Mock(return_value=_SimulationApp({}))

        from cdf.execution.isaac import run_plan_in_isaac

        with (
            mock.patch.dict(sys.modules, {"isaacsim": isaacsim}),
            mock.patch(
                "cdf.execution.isaac.app_runner.create_default_isaac_executor",
                return_value=executor,
            ),
            mock.patch(
                "cdf.execution.isaac.app_runner.perf_counter",
                side_effect=lambda: next(clock),
            ),
        ):
            result = run_plan_in_isaac(object(), resolution=(64, 64))

        self.assertEqual(result.startup_s, 60.0)
        self.assertTrue(result.success)

    def test_startup_is_reported_to_the_completion_callback(self):
        """The child writes the result from the callback, before Kit exits."""
        clock = iter([0.0, 5.0])
        executor = mock.Mock()
        executor.execute_async = mock.AsyncMock(return_value=JobResult(success=True))
        isaacsim = types.ModuleType("isaacsim")
        isaacsim.SimulationApp = mock.Mock(return_value=_SimulationApp({}))
        seen: list[JobResult | None] = []

        from cdf.execution.isaac import run_plan_in_isaac

        with (
            mock.patch.dict(sys.modules, {"isaacsim": isaacsim}),
            mock.patch(
                "cdf.execution.isaac.app_runner.create_default_isaac_executor",
                return_value=executor,
            ),
            mock.patch(
                "cdf.execution.isaac.app_runner.perf_counter",
                side_effect=lambda: next(clock),
            ),
        ):
            run_plan_in_isaac(
                object(),
                resolution=(64, 64),
                on_complete=lambda result, error: seen.append(result),
            )

        self.assertEqual(seen[0].startup_s, 5.0)


if __name__ == "__main__":
    unittest.main()


class SlowestOrdersLimitTest(unittest.TestCase):
    """A negative limit is a caller error, not a slice.

    `timed[:limit]` with a negative limit silently returns almost the whole
    list -- `slowest_orders(-1)` drops the last entry rather than returning
    nothing or complaining. Reported in review of MR !142.
    """

    def _result(self, count=4):
        """Return a result with `count` timed orders."""
        return JobResult(
            success=True,
            order_results=tuple(
                JobOrderResult(
                    "order_%d" % index, "scene.load", "succeeded",
                    duration_s=float(index + 1),
                )
                for index in range(count)
            ),
        )

    def test_a_limit_selects_the_costliest(self):
        self.assertEqual(
            [entry[0] for entry in self._result().slowest_orders(2)],
            ["order_3", "order_2"],
        )

    def test_a_zero_limit_returns_nothing(self):
        self.assertEqual(self._result().slowest_orders(0), ())

    def test_a_negative_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            self._result().slowest_orders(-1)

    def test_a_limit_larger_than_the_list_returns_all_of_it(self):
        self.assertEqual(len(self._result(3).slowest_orders(99)), 3)
