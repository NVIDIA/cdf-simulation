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

"""Run a CDF JobOrder plan inside a locally managed Isaac Sim application."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING

from ...errors import ValidationError
from ..job_result_types import JobOrderResult, JobResult
from .executor_factory import create_default_isaac_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..job_order_types import JobOrderPlan


def run_plan_in_isaac(
    plan: JobOrderPlan,
    *,
    resolution: tuple[int, int],
    windowed: bool = False,
    experience: str | None = None,
    on_complete: Callable[[JobResult | None, BaseException | None], None] | None = None,
) -> JobResult:
    """Boot Isaac Sim and execute a compiled plan through the in-process API.

    The plan has already crossed the process boundary and is executed directly
    by the default JobOrder executor. This function never reconstructs a
    mutable job, compiles legacy YAML, or invokes ``sdg_pipeline.py``.

    Args:
        plan: Validated, immutable JobOrder plan to execute.
        resolution: Width and height supplied to the Isaac application window.
        windowed: Whether to display the Isaac application window.
        experience: Optional Kit experience file. `apps/pcb_aoi.sdg.kit` drops
            the 29 extensions a board cannot use, for a bit-identical frame.
        on_complete: Optional child-transport callback invoked before Kit
            shutdown with either the result or execution error.

    Returns:
        Structured result from the in-process JobOrder executor, carrying the
        seconds the application took to boot in ``startup_s``.

    Raises:
        RuntimeError: If Isaac Sim exits before execution finishes.

    Example:

    .. code-block:: python

        result = run_plan_in_isaac(
            plan,
            resolution=(1920, 1080),
        )  # doctest: +SKIP
        print(result.success)  # doctest: +SKIP

    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "run_plan_in_isaac cannot run inside an active event loop",
        )

    # Establish the loop before `SimulationApp` starts so Kit and the JobOrder
    # task share one explicitly owned event loop on Python 3.12 and later.
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    application = None
    completion_sent = False

    async def execute() -> JobResult:
        return await create_default_isaac_executor().execute_async(plan)

    try:
        from isaacsim import SimulationApp

        boot_started = perf_counter()
        application = SimulationApp(_application_settings(resolution, windowed, experience))
        startup_s = perf_counter() - boot_started
        task = event_loop.create_task(execute())
        while application.is_running() and not task.done():
            application.update()
            event_loop.run_until_complete(asyncio.sleep(0))
        if not task.done():
            task.cancel()
            raise RuntimeError("Isaac Sim stopped before the job completed")
        result = replace(task.result(), startup_s=startup_s)
        if on_complete is not None:
            on_complete(result, None)
            completion_sent = True
        return result
    except BaseException as error:
        if on_complete is not None and not completion_sent:
            on_complete(None, error)
            completion_sent = True
        raise
    finally:
        if application is not None:
            application.close()
        asyncio.set_event_loop(None)
        event_loop.close()


def run_plans_in_isaac(
    plans: Sequence[JobOrderPlan],
    *,
    resolution: tuple[int, int],
    windowed: bool = False,
    experience: str | None = None,
    on_complete: (
        Callable[[tuple[JobResult, ...], BaseException | None], None] | None
    ) = None,
) -> tuple[JobResult, ...]:
    """Boot Isaac Sim once and execute several compiled plans in sequence.

    Booting Kit is not the reason to do this. Measured on a KiCad board, a
    render costs 101.5 s in a fresh process and 4.8 s as the second render in
    the same one: the ~98 s inside the replicator orchestrator's completion
    wait is RTX pipeline and shader compilation, and it is paid once per
    process. Booting costs a further ~40 s. A seven-variant dataset run one
    process at a time therefore pays roughly 140 s seven times over.

    Each plan is executed by its own executor, and therefore its own
    `IsaacExecutionContext`: a context reused across plans would carry one
    plan's stage and operation state into the next.

    A plan that fails does NOT end the batch. The boot and the compilation the
    remaining plans need have already been paid for, and throwing them away is
    the opposite of the point; the failure is reported in that plan's own
    result, in the position it ran.

    Args:
        plans: Validated, immutable JobOrder plans to execute in order.
        resolution: Width and height supplied to the Isaac application window.
            One process has one window, so this must cover the largest plan.
        windowed: Whether to display the Isaac application window.
        experience: Optional Kit experience file. `apps/pcb_aoi.sdg.kit` drops
            the 29 extensions a board cannot use, for a bit-identical frame.
        on_complete: Optional child-transport callback invoked before Kit
            shutdown with every result, or with the error that prevented them.

    Returns:
        One result per plan, in the order the plans were given.

    Raises:
        ValidationError: If any entry is not a JobOrderPlan.
        RuntimeError: If Isaac Sim exits before the batch finishes.

    Example:

    .. code-block:: python

        results = run_plans_in_isaac(
            [first_plan, second_plan],
            resolution=(1920, 1080),
        )  # doctest: +SKIP
        print(len(results))  # doctest: +SKIP

    """
    from ..job_order_types import JobOrderPlan as _JobOrderPlan

    ordered = tuple(plans)
    for position, plan in enumerate(ordered):
        if not isinstance(plan, _JobOrderPlan):
            raise ValidationError(
                "run_plans_in_isaac expects JobOrderPlan at position %d" % position,
            )
    if not ordered:
        # Nothing to run is not a reason to spend four minutes booting Kit.
        if on_complete is not None:
            on_complete((), None)
        return ()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "run_plans_in_isaac cannot run inside an active event loop",
        )

    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    application = None
    completion_sent = False

    try:
        from isaacsim import SimulationApp

        boot_started = perf_counter()
        application = SimulationApp(_application_settings(resolution, windowed, experience))
        startup_s = perf_counter() - boot_started

        results: list[JobResult] = []
        for position, plan in enumerate(ordered):
            async def execute(plan: JobOrderPlan = plan) -> JobResult:
                return await create_default_isaac_executor().execute_async(plan)

            task = event_loop.create_task(execute())
            while application.is_running() and not task.done():
                application.update()
                event_loop.run_until_complete(asyncio.sleep(0))
            if not task.done():
                task.cancel()
                raise RuntimeError(
                    "Isaac Sim stopped before plan %d of %d completed"
                    % (position + 1, len(ordered)),
                )
            result = _plan_result(task)
            # The boot is charged to the plan that waited for it, and to no
            # other: a later plan reading a nonzero startup would be reading
            # a cost it did not pay.
            results.append(
                replace(result, startup_s=startup_s if position == 0 else 0.0),
            )

        completed = tuple(results)
        if on_complete is not None:
            on_complete(completed, None)
            completion_sent = True
        return completed
    except BaseException as error:
        if on_complete is not None and not completion_sent:
            on_complete((), error)
            completion_sent = True
        raise
    finally:
        if application is not None:
            application.close()
        asyncio.set_event_loop(None)
        event_loop.close()


def _plan_result(task: asyncio.Task) -> JobResult:
    """Return one plan's result, turning its failure into a reported one.

    A batch reports a failure rather than raising it, so that the plans behind
    it still run. Everything the executor accumulated before the failure is
    kept: a `JobExecutionError` already carries the partial result, and any
    other exception -- one raised before the first handler ran -- is recorded
    as a single failed order so the caller always has something to read.
    """
    from ...errors import JobExecutionError

    error = task.exception()
    if error is None:
        return task.result()
    if isinstance(error, JobExecutionError):
        return error.result
    return JobResult(
        success=False,
        order_results=(
            JobOrderResult(
                order_id="",
                order_type="",
                status="failed",
                error="%s: %s" % (type(error).__name__, error),
            ),
        ),
    )


def _application_settings(
    resolution: tuple[int, int],
    windowed: bool,
    experience: object = None,
) -> dict[str, object]:
    """Return the SimulationApp settings both runners boot with.

    Args:
        resolution: Width and height for the Isaac application window.
        windowed: Whether to display that window.
        experience: Optional Kit experience (``.kit``) file. Omitted entirely
            when falsy, which leaves Isaac on its own default -- an unset
            environment variable arrives as ``""``, not as ``None``.

    Returns:
        Settings to hand to ``SimulationApp``.

    """
    settings: dict[str, object] = {
        "headless": not windowed,
        "hide_ui": not windowed,
        "width": resolution[0],
        "height": resolution[1],
        # The capture handler switches RTX to PathTracing after the
        # composed stage and render product exist. Boot with the
        # lightweight renderer and no empty startup stage so hidden
        # Windows runs do not wait on a nonexistent UI viewport.
        "renderer": "RaytracedLighting",
        "create_new_stage": False,
        "multi_gpu": False,
    }
    if experience:
        settings["experience"] = str(experience)
    return settings
