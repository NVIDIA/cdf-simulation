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

"""Let a domain contribute its own Isaac handlers to the executor.

The Isaac executor machinery is domain-neutral; the orders a domain invents are
not. Without an extension point every new domain order has to be added to
CDF's own contract module, handler registry and runtime adapter, which is the
boundary violation `DEVELOPMENT.md` names:

    Keep the Isaac executor machinery here, but migrate PCB-specific contracts
    and handlers to domain-owned Isaac adapters that register with it.

A domain registers a factory; the default registry asks every registered domain
for its handlers after installing the built-in ones. The contract, the
validation and the implementation all stay in the domain package.

Registration is idempotent for the same factory so importing a domain package
twice is harmless, and a *different* factory for a name already taken is an
error rather than a silent replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cdf.errors import CDFError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cdf.execution.handler_registry import JobOrderHandler

#: Registered domain adapter factories, keyed by domain name.
_ADAPTERS: dict[str, Callable[[], Mapping[str, JobOrderHandler]]] = {}

#: Registered geometry-role factories, keyed by domain name.
_GEOMETRY_ROLES: dict[str, Callable[[], Mapping[str, "GeometryRole"]]] = {}

#: The owning component is the nearest ancestor prim that is a component.
OWNERSHIP_ANCESTOR = "ancestor"

#: The owning component is the target of a component relationship on the prim.
OWNERSHIP_RELATIONSHIP = "relationship"

_OWNERSHIP_KINDS = frozenset({OWNERSHIP_ANCESTOR, OWNERSHIP_RELATIONSHIP})


@dataclass(frozen=True)
class GeometryRole:
    """How one geometry target is resolved by stage traversal.

    The executor labels geometry it finds by a role token on the prim. Which
    token means what, and how the owning component is reached from a prim
    carrying it, is domain vocabulary -- so a domain declares it here and the
    runtime consumes targets it does not have to recognise.

    Args:
        role: Mesh-role token the prims of this target carry.
        ownership: How to reach the owning component from a matching prim,
            either :data:`OWNERSHIP_ANCESTOR` or :data:`OWNERSHIP_RELATIONSHIP`.
        projected: Whether the target is prototype geometry, so an instanced
            cohort reads its class through the prototype projection.

    Raises:
        ValidationError: If the role token or ownership kind is unusable.

    """

    role: str
    #: Mesh-role token the prims of this target carry.

    ownership: str = OWNERSHIP_RELATIONSHIP
    #: How the owning component is reached from a matching prim.

    projected: bool = False
    #: Whether an instanced cohort reads this class through the projection.

    def __post_init__(self) -> None:
        """Reject a role declaration the runtime could not act on.

        Raises:
            ValidationError: If the role token or ownership kind is unusable.

        """
        if not isinstance(self.role, str) or not self.role.strip():
            message = "geometry role token must be a non-empty string"
            raise ValidationError(message)
        if self.ownership not in _OWNERSHIP_KINDS:
            message = (
                f"geometry role {self.role!r} has unknown ownership "
                f"{self.ownership!r}; expected one of {sorted(_OWNERSHIP_KINDS)}"
            )
            raise ValidationError(message)


def register_geometry_roles(
    domain: str,
    factory: Callable[[], Mapping[str, GeometryRole]],
) -> None:
    """Register one domain's traversal-resolved geometry targets.

    ``annotations.geometry`` labels geometry the domain names. Registering the
    resolution here keeps the role vocabulary in the domain package and leaves
    the generic runtime consuming domain-neutral targets.

    Args:
        domain: Domain name, matching its ``DomainSpec``.
        factory: Callable returning target name to :class:`GeometryRole`.
            Called when the roles are read, not at registration time, so a
            domain may import heavy dependencies lazily.

    Raises:
        ValidationError: If the domain name or factory is unusable.
        CDFError: If a different factory is already registered for the name.

    """
    if not isinstance(domain, str) or not domain.strip():
        message = "domain name must be a non-empty string"
        raise ValidationError(message)
    if not callable(factory):
        message = f"geometry-role factory for {domain!r} must be callable"
        raise ValidationError(message)
    existing = _GEOMETRY_ROLES.get(domain)
    if existing is not None and existing is not factory:
        message = (
            f"a different geometry-role factory is already registered for "
            f"{domain!r}"
        )
        raise CDFError(message)
    _GEOMETRY_ROLES[domain] = factory


def geometry_roles() -> dict[str, GeometryRole]:
    """Return every registered domain's traversal geometry targets, merged.

    Returns:
        Target name mapped to how it is resolved. Empty when no domain has
        registered any, in which case the runtime labels no traversal geometry.

    Raises:
        ValidationError: If a factory returns an unusable target name or a
            spec that is not a :class:`GeometryRole`.
        CDFError: If two domains claim the same target, which would make
            execution depend on import order.

    """
    merged: dict[str, GeometryRole] = {}
    owners: dict[str, str] = {}
    for domain in sorted(_GEOMETRY_ROLES):
        for target, spec in dict(_GEOMETRY_ROLES[domain]()).items():
            # Check the shape here rather than letting a bad entry through to
            # the runtime, where it would surface as an attribute error on a
            # target name with no hint of which domain declared it.
            if not isinstance(target, str) or not target.strip():
                message = (
                    f"geometry-role factory for {domain!r} returned an invalid "
                    f"target name {target!r}; expected a non-empty string"
                )
                raise ValidationError(message)
            if not isinstance(spec, GeometryRole):
                message = (
                    f"geometry-role factory for {domain!r} returned an invalid "
                    f"spec for {target!r}: expected GeometryRole, got "
                    f"{type(spec).__name__}"
                )
                raise ValidationError(message)
            if target in merged:
                message = (
                    f"geometry target {target!r} is claimed by both "
                    f"{owners[target]!r} and {domain!r}"
                )
                raise CDFError(message)
            merged[target] = spec
            owners[target] = domain
    return merged


def register_isaac_adapter(
    domain: str,
    factory: Callable[[], Mapping[str, JobOrderHandler]],
) -> None:
    """Register one domain's Isaac handlers.

    Args:
        domain: Domain name, matching its ``DomainSpec``.
        factory: Callable returning order type to handler. Called when a
            registry is built, not at registration time, so a domain may import
            heavy dependencies lazily.

    Raises:
        ValidationError: If the domain name or factory is unusable.
        CDFError: If a different factory is already registered for the name.

    """
    if not isinstance(domain, str) or not domain.strip():
        message = "domain name must be a non-empty string"
        raise ValidationError(message)
    if not callable(factory):
        message = f"Isaac adapter factory for {domain!r} must be callable"
        raise ValidationError(message)
    existing = _ADAPTERS.get(domain)
    if existing is not None and existing is not factory:
        message = (
            f"a different Isaac adapter is already registered for {domain!r}"
        )
        raise CDFError(message)
    _ADAPTERS[domain] = factory


def registered_domains() -> tuple[str, ...]:
    """Return the domains that have contributed Isaac handlers.

    Returns:
        Domain names, sorted.
    """
    return tuple(sorted(_ADAPTERS))


def domain_handlers() -> dict[str, JobOrderHandler]:
    """Return every registered domain's handlers, merged.

    Returns:
        Order type mapped to handler.

    Raises:
        CDFError: If two domains claim the same order type, which would make
            execution depend on import order.

    """
    merged: dict[str, JobOrderHandler] = {}
    owners: dict[str, str] = {}
    for domain in sorted(_ADAPTERS):
        for order_type, handler in dict(_ADAPTERS[domain]()).items():
            if order_type in merged:
                message = (
                    f"order type {order_type!r} is claimed by both "
                    f"{owners[order_type]!r} and {domain!r}"
                )
                raise CDFError(message)
            merged[order_type] = handler
            owners[order_type] = domain
    return merged
