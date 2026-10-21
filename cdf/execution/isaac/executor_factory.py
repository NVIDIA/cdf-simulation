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

"""Construct public Isaac executor configurations."""

from __future__ import annotations

from .default_handler_registry import create_default_isaac_handler_registry
from .execution_context import IsaacExecutionContext
from .in_process_executor import IsaacInProcessExecutor


def create_default_isaac_executor(
    *,
    context: IsaacExecutionContext | None = None,
) -> IsaacInProcessExecutor:
    """Create an in-process executor with all built-in Isaac handlers.

    Args:
        context: Optional execution context for dependency injection or reuse.

    Returns:
        Executor ready to run a compiled CDF JobOrder plan.

    Example:

    .. code-block:: python

        executor = create_default_isaac_executor()
        result = await job.capture.render_async(
            output_dir="output",
            executor=executor,
        )

    """
    execution_context = context if context is not None else IsaacExecutionContext()
    return IsaacInProcessExecutor(
        create_default_isaac_handler_registry(),
        context=execution_context,
    )
