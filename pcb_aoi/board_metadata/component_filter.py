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

"""Resolve normalized component filters against board metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from cdf.component_filter import ComponentFilter
from cdf.errors import ValidationError

from .metadata_catalog import BoardMetadataCatalog, ComponentRecord

_SCALAR_FIELDS = (
    "refdes",
    "component_type",
    "package_type",
    "footprint_id",
    "side",
    "process_class",
)


def select_components(
    catalog: BoardMetadataCatalog,
    component_filter: ComponentFilter | Mapping[str, object] | None = None,
    *,
    parent_paths: Iterable[str] | None = None,
    board_root: str | None = None,
    where: str = "component_filter",
) -> tuple[ComponentRecord, ...]:
    """Resolve one filter within an optional parent component population.

    Args:
        catalog: Source-neutral component metadata.
        component_filter: Explicit metadata-field filter, or None for no
            additional restriction.
        parent_paths: Parent population to intersect with this filter.
        board_root: Optional owning board root restriction.
        where: Configuration location used in validation messages.

    Returns:
        Matching components in stable catalog order.

    Raises:
        ValidationError: If a requested field is unavailable or the effective
            filter matches no components.

    Example:

    .. code-block:: python

        selected = select_components(
            catalog,
            {"side": ["Top"], "process_classes": ["SMT"]},
        )
        print([component.refdes for component in selected])

    """
    model = (
        component_filter
        if isinstance(component_filter, ComponentFilter)
        else ComponentFilter.model_validate(component_filter or {})
    )
    parent = None if parent_paths is None else {str(path) for path in parent_paths}
    candidates = tuple(
        component
        for component in catalog.components
        if (parent is None or component.prim_path in parent)
        and (board_root is None or component.board_root == board_root)
    )
    requested = model.model_dump(exclude_none=True)
    for field in requested:
        if field == "process_classes":
            available = any(component.process_classes for component in candidates)
        else:
            available = any(
                getattr(component, field) is not None for component in candidates
            )
        if not available:
            raise ValidationError(
                f"{where} requires component metadata field {field!r}, "
                "but the parent population does not provide it",
            )

    selected = candidates
    for field in _SCALAR_FIELDS:
        values = getattr(model, field)
        if values is not None:
            allowed = set(values)
            selected = tuple(
                component
                for component in selected
                if getattr(component, field) in allowed
            )
    if model.process_classes is not None:
        allowed = set(model.process_classes)
        selected = tuple(
            component
            for component in selected
            if allowed.intersection(component.process_classes)
        )
    if requested and not selected:
        raise ValidationError(f"{where} matched no components in its parent population")
    return selected
