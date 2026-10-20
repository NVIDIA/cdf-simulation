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

"""Register domain capability catalogs without coupling CDF to a domain."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import metadata

_ENTRY_POINT_GROUP = "cdf.domains"
_CAPABILITIES_BY_DOMAIN: dict[str, tuple[Capability, ...]] = {}
_DOCUMENTATION_BY_DOMAIN: dict[str, str] = {}
_PLUGINS_LOADING = False
_PLUGINS_LOADED = False


@dataclass(frozen=True)
class Capability:
    """Describe one workflow capability across the supported batch surfaces."""

    id: str
    title: str
    config: str = ""
    api: str = ""
    skill: str = ""
    preview: str = ""
    docs: tuple[str, ...] = ()
    modes: frozenset[str] = field(default_factory=frozenset)


def register_capabilities(
    domain: str,
    capabilities: Iterable[Capability],
    *,
    documentation: str = "",
) -> None:
    """Register a domain's capability records.

    Args:
        domain: Stable domain identifier.
        capabilities: Capability records contributed by the domain.
        documentation: Domain-specific documentation appended to its matrix.

    Example:
        >>> register_capabilities("example", ())

    """
    records = tuple(capabilities)
    existing = _CAPABILITIES_BY_DOMAIN.get(domain)
    if existing is not None and existing != records:
        raise ValueError(f"capabilities already registered for domain {domain!r}")
    existing_documentation = _DOCUMENTATION_BY_DOMAIN.get(domain)
    if existing_documentation is not None and existing_documentation != documentation:
        raise ValueError(f"capability documentation already registered for {domain!r}")
    _CAPABILITIES_BY_DOMAIN[domain] = records
    _DOCUMENTATION_BY_DOMAIN[domain] = documentation


def load_installed_domains() -> None:
    """Load installed domain entry points once.

    Example:
        >>> load_installed_domains()

    """
    global _PLUGINS_LOADED, _PLUGINS_LOADING
    if _PLUGINS_LOADED or _PLUGINS_LOADING:
        return
    _PLUGINS_LOADING = True
    try:
        entry_points = metadata.entry_points()
        if hasattr(entry_points, "select"):
            candidates = entry_points.select(group=_ENTRY_POINT_GROUP)
        else:  # pragma: no cover - Python 3.9 compatibility.
            candidates = entry_points.get(_ENTRY_POINT_GROUP, ())
        for entry_point in candidates:
            entry_point.load()
    finally:
        _PLUGINS_LOADING = False
    _PLUGINS_LOADED = True


def get_capabilities(
    *,
    domains: Iterable[str] | None = None,
    load_plugins: bool = True,
) -> tuple[Capability, ...]:
    """Return registered capabilities in deterministic domain order.

    Args:
        domains: Domain identifiers to include, or all registered domains.
        load_plugins: Load installed CDF domain entry points first.

    Returns:
        All registered capability records.

    Example:
        >>> isinstance(get_capabilities(load_plugins=False), tuple)
        True

    """
    if load_plugins:
        load_installed_domains()
    domain_names = (
        sorted(_CAPABILITIES_BY_DOMAIN) if domains is None else sorted(set(domains))
    )
    return tuple(
        capability
        for domain in domain_names
        for capability in _CAPABILITIES_BY_DOMAIN.get(domain, ())
    )


def get_capability_documentation(
    *,
    domains: Iterable[str] | None = None,
    load_plugins: bool = True,
) -> tuple[str, ...]:
    """Return domain documentation in deterministic domain order.

    Args:
        domains: Domain identifiers to include, or all registered domains.
        load_plugins: Load installed CDF domain entry points first.

    Returns:
        Non-empty documentation blocks contributed by the selected domains.

    """
    if load_plugins:
        load_installed_domains()
    domain_names = (
        sorted(_DOCUMENTATION_BY_DOMAIN) if domains is None else sorted(set(domains))
    )
    return tuple(
        documentation
        for domain in domain_names
        if (documentation := _DOCUMENTATION_BY_DOMAIN.get(domain))
    )
