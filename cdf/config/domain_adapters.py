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

"""Register domain-owned JobConfig models and interpreters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from ..domain import get_domain
from ..errors import CDFError, ValidationError

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ..job import CDFJob

ApplySection = Callable[..., object]


@dataclass(frozen=True)
class DomainConfigAdapter:
    """Domain-owned configuration model and interpretation hooks.

    Args:
        scene_model: Strict model for the domain's ``scene`` block.
        apply_scene: Interpreter that records scene declarations.
        appearance_model: Strict model for the domain's ``appearance`` block.
        apply_appearance: Interpreter that records appearance declarations.
        operations_model: Strict model for the domain's ``operations`` block.
        apply_operations: Interpreter that records operation declarations.
        annotations_model: Strict model for the domain's ``annotations`` block.
        apply_annotations: Interpreter that records annotation declarations.
        capture_model: Strict model for the domain's ``capture`` block.
        apply_capture: Interpreter that records capture declarations.
        postprocess_model: Strict model for the domain's ``postprocess`` block.
        apply_postprocess: Interpreter that records postprocess declarations.

    """

    scene_model: type[BaseModel]
    apply_scene: ApplySection
    appearance_model: type[BaseModel]
    apply_appearance: ApplySection
    annotations_model: type[BaseModel]
    apply_annotations: ApplySection
    operations_model: type[BaseModel]
    apply_operations: ApplySection
    capture_model: type[BaseModel]
    apply_capture: ApplySection
    postprocess_model: type[BaseModel]
    apply_postprocess: ApplySection


_ADAPTERS: dict[str, DomainConfigAdapter] = {}


def register_domain_config(domain: str, adapter: DomainConfigAdapter) -> None:
    """Register one domain's JobConfig extension.

    Args:
        domain: Stable domain identifier.
        adapter: Domain-owned configuration adapter.

    Raises:
        ValidationError: If the registration arguments are invalid.
        CDFError: If another adapter already owns the domain.

    """
    if not isinstance(domain, str) or not domain.strip():
        raise ValidationError("config domain must be a non-empty string")
    if not isinstance(adapter, DomainConfigAdapter):
        raise ValidationError(
            f"config adapter for {domain!r} must be a DomainConfigAdapter",
        )
    existing = _ADAPTERS.get(domain)
    if existing is not None and existing != adapter:
        raise CDFError(
            f"a different JobConfig adapter is already registered for {domain!r}",
        )
    _ADAPTERS[domain] = adapter


def _load_entry_point(domain: str) -> None:
    """Load the installed or source-checkout plugin for one CDF domain.

    Args:
        domain: Domain identifier to discover.

    """
    get_domain(domain)


def get_domain_config(domain: str) -> DomainConfigAdapter:
    """Return a registered domain configuration adapter.

    Args:
        domain: Stable domain identifier.

    Returns:
        Registered adapter for the domain.

    Raises:
        CDFError: If no installed or imported plugin owns the domain.

    """
    adapter = _ADAPTERS.get(domain)
    if adapter is None:
        _load_entry_point(domain)
        adapter = _ADAPTERS.get(domain)
    if adapter is None:
        raise CDFError(
            f"domain {domain!r} has no registered JobConfig adapter; install "
            "or import its CDF domain plugin",
        )
    return adapter


def registered_domain_configs() -> Mapping[str, DomainConfigAdapter]:
    """Return a snapshot of registered domain configuration adapters.

    Returns:
        Mapping from domain identifier to adapter.

    """
    return dict(_ADAPTERS)


def load_installed_domain_configs() -> None:
    """Load every installed CDF domain plugin for schema generation."""
    for entry_point in entry_points(group="cdf.domains"):
        entry_point.load()


def apply_domain_operations(
    domain: str,
    job: CDFJob,
    operations: BaseModel,
    **context: object,
) -> None:
    """Apply a domain-owned operations configuration.

    Args:
        domain: Stable domain identifier.
        job: Job receiving declarations.
        operations: Validated domain operations model.
        **context: Scene-derived values required by the domain interpreter.

    """
    adapter = get_domain_config(domain)
    if not isinstance(operations, adapter.operations_model):
        raise ValidationError(
            f"operations for {domain!r} must be "
            f"{adapter.operations_model.__name__}, got {type(operations).__name__}",
        )
    adapter.apply_operations(job, operations, **context)


def apply_domain_annotations(
    domain: str,
    job: CDFJob,
    annotations: BaseModel,
    **context: object,
) -> None:
    """Apply a domain-owned annotations configuration.

    Args:
        domain: Stable domain identifier.
        job: Job receiving declarations.
        annotations: Validated domain annotations model.
        **context: Scene-derived values required by the domain interpreter.

    """
    adapter = get_domain_config(domain)
    if not isinstance(annotations, adapter.annotations_model):
        raise ValidationError(
            f"annotations for {domain!r} must be "
            f"{adapter.annotations_model.__name__}, got "
            f"{type(annotations).__name__}",
        )
    adapter.apply_annotations(job, annotations, **context)
