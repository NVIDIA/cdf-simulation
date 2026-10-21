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

"""Verify many plans can share one Isaac process.

Measured on a KiCad board: a render costs 101.5s in a fresh process and 4.8s as
the second render in the same one, because the ~98s inside
``orchestrator.wait_until_complete_async`` is RTX pipeline and shader
compilation paid once per process. A seven-variant dataset therefore pays that
seven times today for no reason.

What the tests below can check is the orchestration: that the plans run in
order, that each gets a context of its own, that the boot cost is attributed to
the run that paid it, and that one bad plan does not throw away the plans after
it. That the renders themselves agree was established by measurement -- two
frames from one process matched to a mean absolute difference of 0.36 with
identical annotation counts -- and no Kit-free test can restate it.
"""

import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cdf import (  # noqa: E402
    JobOrder,
    JobOrderPlan,
    JobOrderResult,
    JobResult,
    ValidationError,
)
import pcb_aoi  # noqa: E402, F401  (registers the "pcb-aoi" domain)
from cdf.errors import JobExecutionError  # noqa: E402


def _plan(order_id: str) -> JobOrderPlan:
    """Build a one-order plan identifiable by its order id."""
    return JobOrderPlan(
        orders=(JobOrder(order_id=order_id, order_type="scene.load"),),
    )


def _result(order_id: str, *, success: bool = True) -> JobResult:
    """Build a result identifiable by the plan it came from."""
    return JobResult(
        success=success,
        order_results=(
            JobOrderResult(order_id, "scene.load", "succeeded", duration_s=1.0),
        ),
    )


class _SimulationApp:
    """Drive the host event loop while recording application lifecycle."""

    def __init__(self, settings: dict[str, object]) -> None:
        self.settings = settings
        self.closed = False
        self.updates = 0

    def is_running(self) -> bool:
        """Report that the test application remains active."""
        return True

    def update(self) -> None:
        """Advance the asyncio task by one turn."""
        self.updates += 1
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0))

    def close(self) -> None:
        """Record application shutdown."""
        self.closed = True


class _Harness:
    """Stand in for Isaac so the batch loop can be exercised without Kit."""

    def __init__(self, outcomes: dict[str, object]) -> None:
        self.outcomes = outcomes
        self.application = _SimulationApp({})
        self.isaacsim = types.ModuleType("isaacsim")
        self.isaacsim.SimulationApp = mock.Mock(return_value=self.application)
        self.executors: list[mock.Mock] = []
        self.executed: list[str] = []

    def _executor(self) -> mock.Mock:
        """Return a fresh executor, as the real factory does per call."""
        executor = mock.Mock()

        async def execute_async(plan):
            order_id = plan.orders[0].order_id
            self.executed.append(order_id)
            outcome = self.outcomes[order_id]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        executor.execute_async = execute_async
        self.executors.append(executor)
        return executor

    def patches(self, clock=None):
        """Return the context managers the batch runner needs patched."""
        entered = [
            mock.patch.dict(sys.modules, {"isaacsim": self.isaacsim}),
            mock.patch(
                "cdf.execution.isaac.app_runner.create_default_isaac_executor",
                side_effect=lambda: self._executor(),
            ),
        ]
        if clock is not None:
            entered.append(
                mock.patch(
                    "cdf.execution.isaac.app_runner.perf_counter",
                    side_effect=clock,
                ),
            )
        return entered


class RunPlansInIsaacTest(unittest.TestCase):
    """The batch runner: one boot, many plans."""

    def _run(self, plans, outcomes, *, clock=None):
        from cdf.execution.isaac import run_plans_in_isaac

        harness = _Harness(outcomes)
        managers = harness.patches(clock)
        for manager in managers:
            manager.start()
        try:
            results = run_plans_in_isaac(plans, resolution=(64, 64))
        finally:
            for manager in reversed(managers):
                manager.stop()
        return harness, results

    def test_the_application_is_constructed_once_for_every_plan(self):
        plans = [_plan("a"), _plan("b"), _plan("c")]
        outcomes = {name: _result(name) for name in ("a", "b", "c")}

        harness, results = self._run(plans, outcomes)

        harness.isaacsim.SimulationApp.assert_called_once()
        self.assertEqual(len(results), 3)
        self.assertTrue(harness.application.closed)

    def test_plans_run_in_order(self):
        plans = [_plan("a"), _plan("b"), _plan("c")]
        outcomes = {name: _result(name) for name in ("a", "b", "c")}

        harness, results = self._run(plans, outcomes)

        self.assertEqual(harness.executed, ["a", "b", "c"])
        self.assertEqual(
            [result.order_results[0].order_id for result in results],
            ["a", "b", "c"],
        )

    def test_each_plan_gets_its_own_executor_and_context(self):
        """A reused context would carry one plan's stage state into the next."""
        plans = [_plan("a"), _plan("b")]
        outcomes = {name: _result(name) for name in ("a", "b")}

        harness, _ = self._run(plans, outcomes)

        self.assertEqual(len(harness.executors), 2)
        self.assertIsNot(harness.executors[0], harness.executors[1])

    def test_startup_is_charged_to_the_first_plan_only(self):
        plans = [_plan("a"), _plan("b")]
        outcomes = {name: _result(name) for name in ("a", "b")}

        _, results = self._run(plans, outcomes, clock=iter([10.0, 55.0]).__next__)

        self.assertEqual(results[0].startup_s, 45.0)
        self.assertEqual(results[1].startup_s, 0.0)

    def test_a_failing_plan_does_not_discard_the_plans_after_it(self):
        """The whole point of a batch is that the boot is already paid."""
        plans = [_plan("a"), _plan("b"), _plan("c")]
        failure = JobExecutionError(
            plans[1].orders[0],
            _result("b", success=False),
            RuntimeError("boom"),
        )
        outcomes = {"a": _result("a"), "b": failure, "c": _result("c")}

        harness, results = self._run(plans, outcomes)

        self.assertEqual(harness.executed, ["a", "b", "c"])
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
        self.assertTrue(results[2].success)

    def test_a_failed_plan_keeps_the_partial_result_it_got_to(self):
        plans = [_plan("a")]
        partial = JobResult(
            success=False,
            order_results=(
                JobOrderResult("a", "scene.load", "failed", error="boom",
                               duration_s=2.0),
            ),
        )
        failure = JobExecutionError(
            plans[0].orders[0],
            partial,
            RuntimeError("boom"),
        )

        _, results = self._run(plans, {"a": failure})

        self.assertEqual(results[0].order_results[0].error, "boom")
        self.assertEqual(results[0].order_results[0].duration_s, 2.0)

    def test_a_plan_that_raises_something_else_still_yields_a_result(self):
        plans = [_plan("a"), _plan("b")]
        outcomes = {"a": RuntimeError("no stage"), "b": _result("b")}

        harness, results = self._run(plans, outcomes)

        self.assertFalse(results[0].success)
        self.assertIn("no stage", results[0].order_results[0].error)
        self.assertEqual(harness.executed, ["a", "b"])

    def test_an_empty_batch_never_boots_the_application(self):
        from cdf.execution.isaac import run_plans_in_isaac

        harness = _Harness({})
        managers = harness.patches()
        for manager in managers:
            manager.start()
        try:
            results = run_plans_in_isaac([], resolution=(64, 64))
        finally:
            for manager in reversed(managers):
                manager.stop()

        self.assertEqual(results, ())
        harness.isaacsim.SimulationApp.assert_not_called()

    def test_the_completion_callback_receives_every_result(self):
        from cdf.execution.isaac import run_plans_in_isaac

        harness = _Harness({name: _result(name) for name in ("a", "b")})
        managers = harness.patches()
        seen = []
        for manager in managers:
            manager.start()
        try:
            run_plans_in_isaac(
                [_plan("a"), _plan("b")],
                resolution=(64, 64),
                on_complete=lambda results, error: seen.append((results, error)),
            )
        finally:
            for manager in reversed(managers):
                manager.stop()

        self.assertEqual(len(seen), 1)
        self.assertEqual(len(seen[0][0]), 2)
        self.assertIsNone(seen[0][1])

    def test_a_non_plan_in_the_batch_is_rejected_before_booting(self):
        from cdf.execution.isaac import run_plans_in_isaac

        harness = _Harness({})
        managers = harness.patches()
        for manager in managers:
            manager.start()
        try:
            with self.assertRaisesRegex(ValidationError, "JobOrderPlan"):
                run_plans_in_isaac([_plan("a"), "not a plan"], resolution=(64, 64))
        finally:
            for manager in reversed(managers):
                manager.stop()

        harness.isaacsim.SimulationApp.assert_not_called()


class SubprocessBatchTransportTest(unittest.TestCase):
    """The batch must survive the process boundary like a single plan does."""

    def test_execute_batch_transports_one_result_per_plan(self):
        from cdf.execution.isaac.subprocess_executor import (
            IsaacSubprocessExecutor,
            _decode_batch_result,
        )

        decoded = _decode_batch_result(
            {
                "success": True,
                "results": [
                    {
                        "success": True,
                        "startup_s": 40.0,
                        "order_results": [
                            {
                                "order_id": "a",
                                "order_type": "scene.load",
                                "status": "succeeded",
                                "output": None,
                                "error": "",
                                "duration_s": 100.0,
                            },
                        ],
                    },
                    {
                        "success": True,
                        "order_results": [
                            {
                                "order_id": "b",
                                "order_type": "scene.load",
                                "status": "succeeded",
                                "output": None,
                                "error": "",
                                "duration_s": 5.0,
                            },
                        ],
                    },
                ],
            },
        )

        self.assertEqual(len(decoded), 2)
        self.assertEqual(decoded[0].startup_s, 40.0)
        self.assertEqual(decoded[1].startup_s, 0.0)
        self.assertEqual(decoded[1].timings, {"b": 5.0})
        self.assertTrue(hasattr(IsaacSubprocessExecutor, "execute_batch"))

    def test_a_batch_envelope_without_results_is_rejected(self):
        from cdf.errors import CDFError
        from cdf.execution.isaac.subprocess_executor import _decode_batch_result

        with self.assertRaises(CDFError):
            _decode_batch_result({"success": True})


class RenderBatchApiTest(unittest.TestCase):
    """The authoring-side entry point a dataset script actually calls."""

    def _job(self, resolution=(256, 256)):
        from cdf import CDFJob

        job = CDFJob(domain="pcb-aoi", random_seed=1)
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.capture.output_start(0)
        job.capture.renderer(
            mode="pathtracing",
            resolution=resolution,
            spp=1,
            total_spp=4,
            max_bounces=2,
        )
        job.capture.camera(rotation_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": 0.0})
        job.capture.scan_grid(x_count=1, y_count=1)
        job.capture.limits(triggers=1, max_images=1)
        job.capture.outputs(rgb=True, image_format="png")
        return job

    def test_every_job_is_compiled_and_handed_over_as_one_batch(self):
        from cdf import BatchJob, render_batch

        captured = {}

        class _Executor:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def execute_batch(self, plans):
                captured["plans"] = plans
                return tuple(JobResult(success=True) for _ in plans)

        with mock.patch(
            "cdf.execution.isaac.IsaacSubprocessExecutor",
            _Executor,
        ):
            results = render_batch(
                [
                    BatchJob(self._job(), output_dir="out/a"),
                    BatchJob(self._job(), output_dir="out/b"),
                ],
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(len(captured["plans"]), 2)
        for plan in captured["plans"]:
            self.assertIsInstance(plan, JobOrderPlan)

    def test_the_application_resolution_covers_the_largest_job(self):
        """One process, one window: a small job must not shrink a large one."""
        from cdf import BatchJob, render_batch

        captured = {}

        class _Executor:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def execute_batch(self, plans):
                return tuple(JobResult(success=True) for _ in plans)

        with mock.patch(
            "cdf.execution.isaac.IsaacSubprocessExecutor",
            _Executor,
        ):
            render_batch(
                [
                    BatchJob(self._job((512, 256)), output_dir="out/a"),
                    BatchJob(self._job((256, 1024)), output_dir="out/b"),
                ],
            )

        self.assertEqual(captured["kwargs"]["resolution"], (512, 1024))

    def test_a_batch_job_carries_the_same_destinations_as_execute(self):
        """`BatchJob` is `job.execute`'s keyword set, one per job."""
        from cdf import BatchJob

        job = self._job()
        item = BatchJob(job, output_dir="out/a", serialization_path="out/s.usda")

        self.assertIs(item.job, job)
        self.assertEqual(item.output_dir, "out/a")
        self.assertEqual(item.serialization_path, "out/s.usda")
        self.assertIsNone(BatchJob(job).output_dir)
        self.assertIsNone(BatchJob(job).serialization_path)

    def test_a_job_with_capture_and_no_destination_is_rejected(self):
        from cdf import BatchJob, render_batch

        with self.assertRaises(ValidationError):
            render_batch([BatchJob(self._job())])

    def test_an_empty_batch_returns_nothing_without_launching_isaac(self):
        from cdf import render_batch

        self.assertEqual(render_batch([]), ())


if __name__ == "__main__":
    unittest.main()
