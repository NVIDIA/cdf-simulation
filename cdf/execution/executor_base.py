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

"""Define the common interface for CDF job-order executors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .job_order_types import JobOrderPlan
from .job_result_types import JobResult


class JobExecutor(ABC):
    """Interface implemented by in-process and launcher executors."""

    @abstractmethod
    def execute(self, plan: JobOrderPlan) -> JobResult:
        """Execute a normalized job-order plan.

        Args:
            plan: Validated plan to execute.

        Returns:
            Structured aggregate execution result.

        Raises:
            ValidationError: If the plan cannot be executed.
            JobExecutionError: If a handler fails after execution begins.

        Example:

        .. code-block:: python

            result = executor.execute(plan)
            print(result.success)

        """

    async def execute_async(self, plan: JobOrderPlan) -> JobResult:
        """Execute a plan through an asynchronous public boundary.

        Synchronous executors inherit this adapter. Executors that need to
        await Kit APIs or process I/O override it.

        Args:
            plan: Validated plan to execute.

        Returns:
            Structured aggregate execution result.

        Raises:
            ValidationError: If the plan cannot be executed.
            JobExecutionError: If a handler fails after execution begins.

        Example:

        .. code-block:: python

            result = await executor.execute_async(plan)
            print(result.success)

        """
        return self.execute(plan)
