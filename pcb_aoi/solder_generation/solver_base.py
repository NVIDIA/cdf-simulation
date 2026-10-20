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

"""Define the common interface for solder-generation solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from .data_types import (
    SolderGenerationRequest,
    SolderSolverRequest,
    SolderSolverResult,
    SolderSolverType,
    SolderWorkItem,
)


def _component_allowlist(
    solver_options: Mapping[str, object],
) -> frozenset[str]:
    """Return the validated component refdes or path allowlist.

    Args:
        solver_options: Solver-specific options containing ``components``.

    Returns:
        Exact component identifiers, or an empty set when unrestricted.

    Raises:
        ValueError: If ``components`` is not a list of non-empty strings.

    """
    configured = solver_options.get("components")
    if configured is None:
        return frozenset()
    if not isinstance(configured, (list, tuple)) or not all(
        isinstance(identifier, str) and identifier for identifier in configured
    ):
        raise ValueError("components must be a list of non-empty names or paths")
    return frozenset(configured)


def _component_allowlist_warning(
    solver_options: Mapping[str, object],
    work_items: Sequence[SolderWorkItem],
) -> str | None:
    """Return a warning for allowlist entries absent from discovered work.

    Args:
        solver_options: Solver-specific options containing ``components``.
        work_items: Discovered solver-neutral work items.

    Returns:
        Warning text when configured identifiers matched nothing, otherwise None.

    """
    allowlist = _component_allowlist(solver_options)
    if not allowlist:
        return None
    discovered = {
        identifier
        for work_item in work_items
        for component in work_item.components
        for identifier in (component.refdes, component.instance_path)
    }
    unmatched = sorted(allowlist - discovered)
    if not unmatched:
        return None
    formatted = ", ".join(repr(identifier) for identifier in unmatched)
    return f"components allowlist entries matched no discovered component: {formatted}"


class SolderSolverBase(ABC):
    """Define a registered solver that never modifies the live stage."""

    solver_id: str
    #: Stable identifier accepted by the public API.

    solver_type: SolderSolverType
    #: Solver implementation type.

    available: bool
    #: Whether runtime selection is currently supported.

    unavailable_reason: str | None
    #: User-facing explanation when the solver is unavailable.

    requires_source_paste: bool
    #: Whether discovery must exclude components without source paste.

    @abstractmethod
    def prepare(
        self,
        request: SolderGenerationRequest,
        work_items: Sequence[SolderWorkItem],
    ) -> SolderSolverRequest:
        """Adapt extension inputs into a stage-free solver request.

        Args:
            request: Extension-level request containing the live stage.
            work_items: Solver-neutral work selected by the extension.

        Returns:
            Stage-free request suitable for local or external execution.

        """

    @abstractmethod
    def generate(self, request: SolderSolverRequest) -> SolderSolverResult:
        """Generate solder geometry without modifying the live stage.

        Args:
            request: Stage-free, solver-conditioned request.

        Returns:
            Geometry artifacts, coverage, warnings, and backend metadata.

        """
