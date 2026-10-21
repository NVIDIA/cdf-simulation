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

"""Domain registry.

A domain module (e.g. ``pcb_aoi``) registers a :class:`DomainSpec` on import,
declaring which operations and sidecar kinds its pipeline understands. The
core ``cdf`` package stays domain-agnostic (Appearance/Scene describe
portable state; Operations carry the domain vocabulary).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import entry_points

from .errors import CDFError

_REGISTRY: dict = {}


@dataclass(frozen=True)
class DomainSpec:
    """Vocabulary and section factories contributed by a domain.

    Args:
        name: Stable domain identifier.
        operations: Runtime operation names supported by the domain.
        job_operations: Declarative-only operation names supported by the domain.
        sidecar_kinds: Sidecar kinds accepted by the domain.
        geometry_targets: Geometry names ``annotations.label_geometry`` accepts.
        operations_section_factory: Optional factory for the domain's public
            operations section.
        capture_section_factory: Optional factory for the domain's public
            capture section.

    """

    name: str
    #: Stable domain identifier.

    operations: frozenset[str] = field(default_factory=frozenset)
    #: Runtime operation names supported by the domain.

    job_operations: frozenset[str] = field(default_factory=frozenset)
    #: Declarative-only operation names supported by the domain.

    sidecar_kinds: frozenset[str] = field(default_factory=frozenset)
    #: Sidecar kinds accepted by the domain.

    geometry_targets: frozenset[str] = field(default_factory=frozenset)
    #: Geometry a job may label with a class of its own.

    board_geometry_roles: frozenset[str] = field(default_factory=frozenset)
    #: USD mesh-role shortnames (without the ``board_`` prefix) that may be
    #: hidden per-pass via ``capture.visibility(hide=...)``.  The runtime maps
    #: each name to the USD attribute value ``pcba:mesh:role == f"board_{name}"``
    #: and hides every matching prim under the board root for the pass duration.

    operations_section_factory: Callable[[object], object] | None = field(
        default=None,
        repr=False,
    )
    #: Factory for the domain's public operations section, if provided.

    capture_section_factory: Callable[[object], object] | None = field(
        default=None,
        repr=False,
    )
    #: Factory for the domain's public capture section, if provided. Capture is
    #: otherwise domain-neutral -- a camera, a resolution and a grid of cells --
    #: so a domain contributes one only to add a way of stating a capture in its
    #: own terms, the way ``pcb-aoi`` states one as a scan rectangle in
    #: millimetres at a dpi. The factory receives the job and must return a
    #: :class:`~cdf.sections.CaptureSection`, so named render passes and
    #: ``render()`` keep working unchanged.


def register_domain(spec: DomainSpec) -> None:
    """Register a domain's vocabulary (idempotent for an identical spec)."""
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing != spec:
        raise CDFError(
            f"domain {spec.name!r} is already registered with a different spec",
        )
    _REGISTRY[spec.name] = spec


def get_domain(name: str) -> DomainSpec:
    """Look up or explicitly discover a :class:`DomainSpec` by name.

    An installed plugin is resolved through the ``cdf.domains`` entry-point
    group. A source checkout without installed package metadata falls back to
    the normalized module name (for example, ``example-domain`` becomes
    ``example_domain``). The fallback occurs only after a caller explicitly
    selects the domain; importing :mod:`cdf` alone remains domain-neutral.

    Args:
        name: Stable domain identifier.

    Returns:
        Registered domain specification.

    Raises:
        CDFError: If the selected domain cannot be discovered or registered.

    """
    spec = _REGISTRY.get(name)
    if spec is not None:
        return spec
    matches = tuple(entry_points(group="cdf.domains", name=name))
    if len(matches) > 1:
        providers = ", ".join(sorted(item.value for item in matches))
        raise CDFError(
            f"multiple installed plugins advertise CDF domain {name!r}: {providers}",
        )
    if matches:
        matches[0].load()
    else:
        module_name = name.replace("-", "_")
        try:
            import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
    spec = _REGISTRY.get(name)
    if spec is not None:
        return spec
    registered = sorted(_REGISTRY) or "none"
    raise CDFError(
        f"unknown domain {name!r} (registered: {registered}); install its "
        "CDF plugin or make its normalized module importable",
    )
