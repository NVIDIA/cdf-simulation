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

"""Define the domain-neutral transport shape for component filters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
)

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ComponentFilter(BaseModel):
    """Select components by explicit metadata fields.

    Values within one field are alternatives. Populated fields are combined
    as an intersection. A child filter is also intersected with its resolved
    parent component population by the executing domain.

    Args:
        refdes: Exact component reference designators.
        component_type: Exact semantic component types.
        package_type: Exact package classifications.
        footprint_id: Exact source footprint identities.
        side: Exact board-side values.
        process_class: Exact scalar manufacturing process classifications.
        process_classes: Manufacturing classifications contained in the
            component's plural process-class property.

    Example:

    .. code-block:: python

        from cdf import ComponentFilter

        component_filter = ComponentFilter(
            side=("Top",),
            process_classes=("SMT",),
            package_type=("0603",),
        )
        print(component_filter.model_dump(exclude_none=True))

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    refdes: tuple[NonEmptyString, ...] | None = None
    #: Exact component reference designators.
    component_type: tuple[NonEmptyString, ...] | None = None
    #: Exact semantic component types.
    package_type: tuple[NonEmptyString, ...] | None = None
    #: Exact package classifications.
    footprint_id: tuple[NonEmptyString, ...] | None = None
    #: Exact source footprint identities.
    side: tuple[NonEmptyString, ...] | None = None
    #: Exact board-side values.
    process_class: tuple[NonEmptyString, ...] | None = None
    #: Exact scalar manufacturing process classifications.
    process_classes: tuple[NonEmptyString, ...] | None = None
    #: Alternative members of the plural manufacturing process classifications.

    @field_validator("*")
    @classmethod
    def validate_values(
        cls,
        values: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        """Require every populated field to contain unique values.

        Args:
            values: Normalized values for one filter field.

        Returns:
            Validated values, or None when the field is absent.

        Raises:
            ValueError: If a populated field is empty or contains duplicates.

        """
        if values is None:
            return None
        if not values:
            raise ValueError("component filter fields cannot be empty")
        if len(set(values)) != len(values):
            raise ValueError("component filter field values must be unique")
        return values

    def to_transport(self) -> dict[str, list[str]]:
        """Return a JSON-compatible filter mapping.

        Returns:
            Mapping containing only populated fields.

        Example:

        .. code-block:: python

            from cdf import ComponentFilter

            print(ComponentFilter(refdes=("R1",)).to_transport())
            {'refdes': ['R1']}

        """
        return {
            name: list(values)
            for name, values in self.model_dump(exclude_none=True).items()
        }


def normalize_component_filter(
    value: ComponentFilter | Mapping[str, object] | None,
    *,
    where: str,
) -> dict[str, list[str]] | None:
    """Validate and serialize a public component-filter argument.

    Args:
        value: Filter model, mapping, or None.
        where: Public API location used in validation messages.

    Returns:
        JSON-compatible filter, or None when omitted.

    Raises:
        ValueError: If the supplied filter does not satisfy the contract.

    Example:

    .. code-block:: python

        normalized = normalize_component_filter(
            {"side": ["Top"], "process_classes": ["SMT"]},
            where="scene.components",
        )
        print(normalized)

    """
    if value is None:
        return None
    try:
        model = (
            value
            if isinstance(value, ComponentFilter)
            else ComponentFilter.model_validate(value)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: invalid component_filter: {exc}") from exc
    return model.to_transport()
