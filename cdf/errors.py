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

"""Errors raised by the declarative CDF API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .execution.job_order_types import JobOrder
    from .execution.job_result_types import JobResult


class CDFError(Exception):
    """Base error for the cdf package."""


class ValidationError(CDFError, ValueError):
    """A job description is invalid or incomplete."""


class JobExecutionError(CDFError):
    """Report a JobOrder handler failure after execution begins.

    Args:
        order: Order whose handler raised the underlying exception.
        result: Partial result accumulated through the failed order.
        cause: Original exception raised by the handler.

    """

    order: JobOrder
    #: Order whose handler failed.

    result: JobResult
    #: Partial execution result including the failed order.

    cause: Exception
    #: Original exception raised by the handler.

    def __init__(
        self,
        order: JobOrder,
        result: JobResult,
        cause: Exception,
    ) -> None:
        self.order = order
        self.result = result
        self.cause = cause
        super().__init__(
            "JobOrder %r (%s) failed: %s: %s"
            % (
                order.order_id,
                order.order_type,
                type(cause).__name__,
                cause,
            ),
        )
