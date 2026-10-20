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

"""Return structured results from CDF job-order executors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class JobOrderResult:
    """Outcome from one JobOrder handler invocation.

    Args:
        order_id: Identifier of the executed order.
        order_type: Semantic type used to resolve the handler.
        status: Execution status for the order.
        output: Value returned by a successful handler.
        error: Diagnostic produced by a failed handler.
        duration_s: Wall-clock seconds the handler ran for.

    """

    order_id: str
    #: Identifier of the executed order.

    order_type: str
    #: Semantic type used to resolve the handler.

    status: str
    #: Execution status for the order.

    output: object = None
    #: Value returned by a successful handler.

    error: str = ""
    #: Diagnostic produced by a failed handler.

    duration_s: float = 0.0
    #: Wall-clock seconds the handler ran for. Recorded for a failed order too.


@dataclass(frozen=True)
class JobResult:
    """Structured result returned by a JobOrder executor.

    Args:
        success: Whether every planned order completed successfully.
        order_results: Results accumulated in execution order.
        startup_s: Wall-clock seconds spent booting the application before the
            first order ran. Zero when the caller supplied a running one.

    """

    success: bool
    #: Whether every planned order completed successfully.

    order_results: tuple[JobOrderResult, ...] = field(default_factory=tuple)
    #: Results accumulated in execution order.

    startup_s: float = 0.0
    #: Seconds spent booting the application before the first order ran.

    @property
    def timings(self) -> dict[str, float]:
        """Return the seconds each order took, keyed by order identifier.

        Returns:
            Mapping from order identifier to wall-clock seconds.

        Example:

        .. code-block:: python

            result = JobResult(success=True)
            print(result.timings)
            # {}

        """
        return {item.order_id: item.duration_s for item in self.order_results}

    @property
    def total_duration_s(self) -> float:
        """Return startup plus every order duration.

        Returns:
            Wall-clock seconds attributable to this run.

        Example:

        .. code-block:: python

            result = JobResult(success=True)
            print(result.total_duration_s)
            # 0.0

        """
        return self.startup_s + sum(item.duration_s for item in self.order_results)

    def slowest_orders(
        self,
        limit: int = 5,
    ) -> tuple[tuple[str, str, float], ...]:
        """Return the costliest orders, longest first.

        Orders that were not timed are omitted: reporting them as zero-cost
        reads as a measurement rather than the absence of one.

        Args:
            limit: Maximum number of orders to return.

        Returns:
            Tuple of ``(order_id, order_type, duration_s)``, longest first.

        Raises:
            ValueError: If ``limit`` is negative. A negative slice bound would
                quietly return nearly everything -- ``slowest_orders(-1)`` drops
                one entry rather than returning none.

        Example:

        .. code-block:: python

            result = JobResult(success=True)
            print(result.slowest_orders())
            # ()

        """
        if limit < 0:
            raise ValueError("slowest_orders limit must be >= 0, got %r" % (limit,))
        timed = [
            (item.order_id, item.order_type, item.duration_s)
            for item in self.order_results
            if item.duration_s > 0.0
        ]
        timed.sort(key=lambda entry: entry[2], reverse=True)
        return tuple(timed[:limit])

    @property
    def outputs(self) -> dict[str, object]:
        """Return successful handler outputs keyed by order identifier.

        Returns:
            Mapping from successful order identifiers to handler outputs.

        Example:

        .. code-block:: python

            result = JobResult(success=True)
            print(result.outputs)
            # {}

        """
        return {
            item.order_id: item.output
            for item in self.order_results
            if item.status == "succeeded"
        }

    def output_for_type(self, order_type: str) -> object | None:
        """Return the latest successful output for an order type.

        Args:
            order_type: Semantic JobOrder type to find.

        Returns:
            Handler output, or ``None`` when the type did not succeed.

        Example:

        .. code-block:: python

            capture = result.output_for_type("capture.render")
            print(capture["output_dir"])

        """
        for item in reversed(self.order_results):
            if item.order_type == order_type and item.status == "succeeded":
                return item.output
        return None

    @property
    def capture_output(self) -> Mapping[str, object] | None:
        """Return structured capture artifacts when capture succeeded.

        Returns:
            Capture output mapping, or ``None`` if capture did not succeed.

        Example:

        .. code-block:: python

            print(result.capture_output["output_dir"])

        """
        output = self.output_for_type("capture.render")
        return output if isinstance(output, Mapping) else None

    @property
    def crop_output(self) -> Mapping[str, object] | None:
        """Return structured crop output when postprocessing succeeded.

        Returns:
            Crop output mapping, or ``None`` if crop did not succeed.

        Example:

        .. code-block:: python

            if result.crop_output:
                print(result.crop_output["output_dir"])

        """
        output = self.output_for_type("postprocess.crop")
        return output if isinstance(output, Mapping) else None

    @property
    def stitch_output(self) -> Mapping[str, object] | None:
        """Return structured stitch output when postprocessing succeeded.

        Returns:
            Stitch output mapping, or ``None`` if stitching did not succeed.

        Example:

        .. code-block:: python

            if result.stitch_output:
                print(result.stitch_output["artifacts"])

        """
        output = self.output_for_type("postprocess.stitch")
        return output if isinstance(output, Mapping) else None

    @property
    def failed_order(self) -> JobOrderResult | None:
        """Return the failed order result, if one was recorded.

        Returns:
            First failed order result, or ``None`` on success.

        Example:

        .. code-block:: python

            if result.failed_order:
                print(result.failed_order.error)

        """
        return next(
            (item for item in self.order_results if item.status == "failed"),
            None,
        )
