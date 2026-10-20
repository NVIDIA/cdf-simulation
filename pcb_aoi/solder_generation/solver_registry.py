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

"""Register solder-generation solvers by stable identifier and type."""

from __future__ import annotations

import re

from cdf.errors import ValidationError

from .data_types import SolderSolverType
from .solver_base import SolderSolverBase

_SOLVER_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_SOLVERS: dict[str, SolderSolverBase] = {}


def _solver_type_value(solver_type: SolderSolverType | str) -> str:
    """Return the string value of a solver type.

    Args:
        solver_type: Solver type enum or compatible string.

    Returns:
        Stable solver-type value.

    """
    return (
        solver_type.value
        if isinstance(solver_type, SolderSolverType)
        else str(solver_type)
    )


def register_solver(solver: SolderSolverBase) -> None:
    """Register one solder-generation solver.

    Args:
        solver: Solver delegate with a stable identifier and lifecycle methods.

    Raises:
        ValidationError: If the identifier or delegate is invalid, or another
            solver already owns the identifier.

    """
    solver_id = getattr(solver, "solver_id", None)
    if not isinstance(solver_id, str) or not _SOLVER_ID_PATTERN.fullmatch(solver_id):
        raise ValidationError(
            "solder solver id must use lowercase letters, digits, '.', '_', "
            "or '-' separators",
        )
    for method_name in ("prepare", "generate"):
        if not callable(getattr(solver, method_name, None)):
            raise ValidationError(
                "solder solver %r must implement %s()" % (solver_id, method_name),
            )
    solver_type = _solver_type_value(getattr(solver, "solver_type", ""))
    if solver_type not in {item.value for item in SolderSolverType}:
        raise ValidationError(
            "solder solver %r must identify type 'none', 'procedural', "
            "or 'simulated'" % solver_id,
        )
    available = getattr(solver, "available", None)
    if not isinstance(available, bool):
        raise ValidationError(
            "solder solver %r must declare boolean availability" % solver_id,
        )
    reason = getattr(solver, "unavailable_reason", None)
    if not available and (not isinstance(reason, str) or not reason):
        raise ValidationError(
            "unavailable solder solver %r must provide a reason" % solver_id,
        )
    if not isinstance(getattr(solver, "requires_source_paste", None), bool):
        raise ValidationError(
            "solder solver %r must declare requires_source_paste" % solver_id,
        )

    existing = _SOLVERS.get(solver_id)
    if existing is solver:
        return
    if existing is not None:
        raise ValidationError("solder solver %r is already registered" % solver_id)
    _SOLVERS[solver_id] = solver


def registered_solver_ids(
    *,
    available_only: bool = False,
) -> tuple[str, ...]:
    """Return registered solver identifiers in deterministic order.

    Args:
        available_only: Exclude unavailable registry entries when true.

    Returns:
        Registered solver identifiers.

    """
    return tuple(
        solver_id
        for solver_id, solver in sorted(_SOLVERS.items())
        if not available_only or solver.available
    )


def registered_solvers(
    solver_type: SolderSolverType | str | None = None,
    *,
    available_only: bool = False,
) -> tuple[SolderSolverBase, ...]:
    """Return registered solvers, optionally filtered by type and availability.

    Args:
        solver_type: Optional ``none``, ``procedural``, or ``simulated`` filter.
        available_only: Exclude unavailable registry entries when true.

    Returns:
        Registered solver delegates in deterministic ID order.

    Raises:
        ValidationError: If the filter is unsupported.

    """
    supported = {item.value for item in SolderSolverType}
    normalized_type = None if solver_type is None else _solver_type_value(solver_type)
    if normalized_type is not None and normalized_type not in supported:
        raise ValidationError(
            "solver type filter must be 'none', 'procedural', 'simulated', or None",
        )
    return tuple(
        solver
        for _, solver in sorted(_SOLVERS.items())
        if (
            (normalized_type is None or solver.solver_type.value == normalized_type)
            and (not available_only or solver.available)
        )
    )


def resolve_solver(
    solver_id: str,
    *,
    require_available: bool = True,
) -> SolderSolverBase:
    """Resolve a registered solver by identifier.

    Args:
        solver_id: Stable solver identifier supplied by the public API.
        require_available: Reject known but unavailable solvers when true.

    Returns:
        Registered solver delegate.

    Raises:
        ValidationError: If the identifier is unknown or unavailable.

    """
    solver = _SOLVERS.get(solver_id)
    if solver is None:
        raise ValidationError(
            "unknown solder solver %r (registered: %s)"
            % (solver_id, list(registered_solver_ids())),
        )
    if require_available and not solver.available:
        raise ValidationError(
            "solder solver %r is unavailable: %s"
            % (solver_id, solver.unavailable_reason),
        )
    return solver
