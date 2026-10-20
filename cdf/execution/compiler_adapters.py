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

"""Let domains extend JobOrder validation, dependencies, and lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ..errors import CDFError, ValidationError
from ..job import _JobDeclaration
from .job_order_types import JobOrder

PrepareDeclarations = Callable[
    [tuple[_JobDeclaration, ...]],
    tuple[_JobDeclaration, ...],
]
ValidateDeclarations = Callable[
    [tuple[_JobDeclaration, ...], Mapping[str, int]],
    Sequence[str],
]
DeclarationDependencies = Callable[
    [_JobDeclaration, Mapping[str, int], tuple[_JobDeclaration, ...]],
    Sequence[str],
]
ProducesOutput = Callable[[tuple[_JobDeclaration, ...]], bool]
ExpandOrders = Callable[
    [list[JobOrder], tuple[_JobDeclaration, ...], Mapping[str, int]],
    list[JobOrder],
]


def _unchanged(
    declarations: tuple[_JobDeclaration, ...],
) -> tuple[_JobDeclaration, ...]:
    """Return declarations without domain preparation."""
    return declarations


def _valid(
    declarations: tuple[_JobDeclaration, ...],
    positions: Mapping[str, int],
) -> tuple[str, ...]:
    """Return no domain validation problems."""
    return ()


def _independent(
    declaration: _JobDeclaration,
    positions: Mapping[str, int],
    declarations: tuple[_JobDeclaration, ...],
) -> tuple[str, ...]:
    """Return no domain prerequisites for a declaration."""
    return ()


def _no_output(declarations: tuple[_JobDeclaration, ...]) -> bool:
    """Report that the domain contributes no persisted output."""
    return False


def _no_expansion(
    orders: list[JobOrder],
    declarations: tuple[_JobDeclaration, ...],
    positions: Mapping[str, int],
) -> list[JobOrder]:
    """Return lowered orders without domain expansion."""
    return orders


@dataclass(frozen=True)
class DomainCompilerAdapter:
    """Domain-owned hooks used while compiling a JobOrder plan.

    Args:
        prepare_declarations: Normalize or synthesize declarations before
            validation.
        validate_declarations: Return domain validation problems.
        declaration_dependencies: Return domain prerequisites for one order.
        configuration_order_types: Declaration types consumed during lowering
            rather than emitted directly.
        produces_output: Report whether domain declarations persist an output.
        expand_orders: Replace configuration declarations with executable
            domain orders.

    """

    prepare_declarations: PrepareDeclarations = _unchanged
    validate_declarations: ValidateDeclarations = _valid
    declaration_dependencies: DeclarationDependencies = _independent
    configuration_order_types: frozenset[str] = field(default_factory=frozenset)
    produces_output: ProducesOutput = _no_output
    expand_orders: ExpandOrders = _no_expansion


_ADAPTERS: dict[str, DomainCompilerAdapter] = {}


def register_compiler_adapter(
    domain: str,
    adapter: DomainCompilerAdapter,
) -> None:
    """Register one domain's compiler hooks.

    Args:
        domain: Stable domain identifier.
        adapter: Compiler hooks owned by that domain.

    Raises:
        ValidationError: If the arguments are invalid.
        CDFError: If another adapter already owns the domain.

    """
    if not isinstance(domain, str) or not domain.strip():
        raise ValidationError("compiler domain must be a non-empty string")
    if not isinstance(adapter, DomainCompilerAdapter):
        raise ValidationError(
            f"compiler adapter for {domain!r} must be a DomainCompilerAdapter",
        )
    existing = _ADAPTERS.get(domain)
    if existing is not None and existing != adapter:
        raise CDFError(
            f"a different compiler adapter is already registered for {domain!r}",
        )
    _ADAPTERS[domain] = adapter


def compiler_adapter(domain: str) -> DomainCompilerAdapter:
    """Return the compiler hooks registered for a domain.

    Args:
        domain: Stable domain identifier.

    Returns:
        Registered compiler adapter.

    Raises:
        CDFError: If the selected domain has no compiler adapter.

    """
    adapter = _ADAPTERS.get(domain)
    if adapter is None:
        raise CDFError(
            f"domain {domain!r} has no registered compiler adapter; import its "
            "CDF domain plugin",
        )
    return adapter
