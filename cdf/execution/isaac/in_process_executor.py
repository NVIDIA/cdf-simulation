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

"""Execute normalized CDF job orders inside an Isaac process."""

from __future__ import annotations

import contextlib
import inspect
import warnings
from collections.abc import Mapping
from time import perf_counter

from ...errors import JobExecutionError, ValidationError
from ..executor_base import JobExecutor
from ..handler_registry import JobOrderHandler, JobOrderHandlerRegistry
from ..job_order_types import JobOrderPlan
from ..job_result_types import JobOrderResult, JobResult


def _begin_provenance() -> None:
    """Start a fresh override record for this render.

    ``_READS`` in :mod:`cdf.provenance.env` is process-global and accumulates,
    so without this the second render in a batch inherits the first one's reads
    and reports overrides that its own run never consulted. reset_overrides()
    existed for this and had no caller.
    """
    try:
        from ...provenance import reset_overrides

        reset_overrides()
    except Exception as error:  # noqa: BLE001 -- provenance must not fail a render
        warnings.warn(
            f"provenance overrides were not reset: {type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )


def _record_provenance(plan, result, context) -> None:
    """Record the run's full provenance into the dataset manifest, never raising.

    Both execution paths converge here: ``capture.render_async`` calls this
    executor directly, and the synchronous ``capture.render`` serializes the
    plan into an Isaac child that runs this same executor. Recording at the
    authoring section would have missed the second, and with it every example
    and the YAML config path.

    The manifest is the sink: ``context.manifest.record_provenance`` puts the
    record into ``dataset_index.json`` and a summary into every trigger
    ``metadata.json``. There is deliberately no sibling file -- two records of
    the same run would be two things to disagree.
    """
    try:
        from ...provenance import record_for_plan

        record_for_plan(plan, result, manifest=getattr(context, "manifest", None))
    except Exception as error:  # noqa: BLE001 -- provenance must not fail a render
        # Swallowing this silently is how a provenance system quietly stops
        # producing provenance, which is worse than the failure itself.
        warnings.warn(
            f"provenance record was not recorded: {type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )


class IsaacInProcessExecutor(JobExecutor):
    """Validate, sequence, and execute JobOrders in the current process.

    The class itself does not import Isaac. A caller already running inside
    Isaac supplies an execution ``context`` and a registry whose handlers use
    the available Isaac/SDG APIs.

    Args:
        handlers: Registry or mapping containing supported order handlers.
        context: Isaac-side state passed unchanged to every handler.

    Raises:
        ValidationError: If the handler collection is invalid.

    Example:

    .. code-block:: python

        from cdf import IsaacInProcessExecutor, JobOrder, JobOrderPlan

        executor = IsaacInProcessExecutor({
            "scene.load": lambda order, context: {"loaded": True},
        })
        result = executor.execute(JobOrderPlan(orders=(
            JobOrder(order_id="load", order_type="scene.load"),
        )))
        print(result.success)
        # True

    """

    def __init__(
        self,
        handlers: JobOrderHandlerRegistry | Mapping[str, JobOrderHandler],
        context: object | None = None,
    ) -> None:
        if isinstance(handlers, JobOrderHandlerRegistry):
            self.handlers = handlers
        elif isinstance(handlers, Mapping):
            self.handlers = JobOrderHandlerRegistry(handlers)
        else:
            raise ValidationError(
                "IsaacInProcessExecutor handlers must be a registry or mapping",
            )
        self.context = context

    def execute(self, plan: JobOrderPlan) -> JobResult:
        """Execute a plan after dependency and handler preflight.

        Args:
            plan: Immutable job-order plan to execute.

        Returns:
            Structured results for every successfully executed order.

        Raises:
            ValidationError: If dependencies or handler support are invalid.
            JobExecutionError: If a handler fails after execution begins.

        Example:

        .. code-block:: python

            result = executor.execute(JobOrderPlan())
            print(result.success)
            # True

        """
        _begin_provenance()
        if not isinstance(plan, JobOrderPlan):
            raise ValidationError(
                "IsaacInProcessExecutor.execute expects JobOrderPlan",
            )

        ordered = plan.ordered()
        # Resolve every handler before running the first order. Unsupported
        # plans therefore fail without partially mutating the stage.
        resolved = tuple(
            (order, self.handlers.handler(order.order_type)) for order in ordered
        )
        asynchronous = [
            order.order_type
            for order, handler in resolved
            if inspect.iscoroutinefunction(handler)
        ]
        if asynchronous:
            raise ValidationError(
                "asynchronous JobOrder handlers require execute_async(): %s"
                % ", ".join(asynchronous),
            )
        storage_scope = getattr(self.context, "storage_scope", None)
        scope = storage_scope() if callable(storage_scope) else contextlib.nullcontext()
        with scope:
            prepare = getattr(self.context, "prepare", None)
            if callable(prepare):
                prepare(plan)

            results = []
            for order, handler in resolved:
                started = perf_counter()
                try:
                    output = handler(order, self.context)
                except Exception as exc:
                    results.append(
                        JobOrderResult(
                            order_id=order.order_id,
                            order_type=order.order_type,
                            status="failed",
                            error="%s: %s" % (type(exc).__name__, exc),
                            duration_s=perf_counter() - started,
                        ),
                    )
                    partial = JobResult(success=False, order_results=tuple(results))
                    # A failed render is exactly when provenance is most wanted:
                    # the artifacts that did land need to be attributable too.
                    _record_provenance(plan, partial, self.context)
                    raise JobExecutionError(order, partial, exc) from exc
                results.append(
                    JobOrderResult(
                        order_id=order.order_id,
                        order_type=order.order_type,
                        status="succeeded",
                        output=output,
                        duration_s=perf_counter() - started,
                    ),
                )
            completed = JobResult(success=True, order_results=tuple(results))
            _record_provenance(plan, completed, self.context)
            return completed

    async def execute_async(self, plan: JobOrderPlan) -> JobResult:
        """Execute synchronous and asynchronous handlers inside Kit.

        Args:
            plan: Immutable job-order plan to execute.

        Returns:
            Structured results for every successfully executed order.

        Raises:
            ValidationError: If dependencies or handler support are invalid.
            JobExecutionError: If a handler fails after execution begins.

        Example:

        .. code-block:: python

            result = await executor.execute_async(JobOrderPlan())
            print(result.success)
            # True

        """
        _begin_provenance()
        if not isinstance(plan, JobOrderPlan):
            raise ValidationError(
                "IsaacInProcessExecutor.execute_async expects JobOrderPlan",
            )

        ordered = plan.ordered()
        resolved = tuple(
            (order, self.handlers.handler(order.order_type)) for order in ordered
        )
        storage_scope = getattr(self.context, "storage_scope", None)
        scope = storage_scope() if callable(storage_scope) else contextlib.nullcontext()
        with scope:
            prepare = getattr(self.context, "prepare", None)
            if callable(prepare):
                prepare(plan)
            results = []
            for order, handler in resolved:
                started = perf_counter()
                try:
                    output = handler(order, self.context)
                    if inspect.isawaitable(output):
                        output = await output
                except Exception as exc:
                    results.append(
                        JobOrderResult(
                            order_id=order.order_id,
                            order_type=order.order_type,
                            status="failed",
                            error="%s: %s" % (type(exc).__name__, exc),
                            duration_s=perf_counter() - started,
                        ),
                    )
                    partial = JobResult(success=False, order_results=tuple(results))
                    _record_provenance(plan, partial, self.context)
                    raise JobExecutionError(order, partial, exc) from exc
                results.append(
                    JobOrderResult(
                        order_id=order.order_id,
                        order_type=order.order_type,
                        status="succeeded",
                        output=output,
                        duration_s=perf_counter() - started,
                    ),
                )
            completed = JobResult(success=True, order_results=tuple(results))
            _record_provenance(plan, completed, self.context)
            return completed
