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

"""Transport contract for ``operation.pin_variation.apply``.

Owned by the domain that invented the order, not by CDF's generic contract
module. This is the second gate: an order can arrive from a file or another
process, so it is the only validation that has definitely run.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from cdf.component_filter import ComponentFilter

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class PinVariationGroup(BaseModel):
    """One selected set of parts and the bands to vary them over."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    components: tuple[NonEmptyString, ...] | None = None
    package_types: tuple[NonEmptyString, ...] | None = None
    component_filter: ComponentFilter | None = None
    tilt_range: tuple[float, float] | None = None
    cut_angle: tuple[float, float] | None = None
    height_variation: tuple[float, float] | None = None
    offset_range: tuple[float, float] | None = None

    @field_validator(
        "tilt_range",
        "cut_angle",
        "height_variation",
        "offset_range",
        mode="before",
    )
    @classmethod
    def reject_booleans(cls, value: object) -> object:
        """Refuse a band whose ends are booleans.

        ``bool`` is a subclass of ``int``, so Pydantic coerces ``True`` to
        ``1.0`` without complaint and a typo becomes a one-degree band. Caught
        before coercion, which is the only place it is still visible.

        Args:
            value: The raw band as supplied.

        Returns:
            The band unchanged.

        Raises:
            ValueError: If either end is a bool.

        """
        if isinstance(value, (list, tuple)) and any(
            isinstance(entry, bool) for entry in value
        ):
            raise ValueError("band bounds must be numbers, not booleans")
        return value

    @model_validator(mode="after")
    def validate_group(self) -> PinVariationGroup:
        """Require one selector and at least one band, each well formed.

        Returns:
            Validated group.

        Raises:
            ValueError: If the selector is missing or ambiguous, if no band is
                given, or if a band is reversed.

        """
        supplied = sum(
            value is not None
            for value in (self.components, self.package_types, self.component_filter)
        )
        if supplied != 1:
            raise ValueError(
                "exactly one of components, package_types, or component_filter "
                "is required",
            )
        # Tilt and cut are angles off an axis, so 90 degrees is degenerate;
        # protrusion and offset are magnitudes whose direction is drawn
        # separately, so a negative one is a sign error, not a direction.
        bounds = {
            "tilt_range": (self.tilt_range, 0.0, 89.0),
            "cut_angle": (self.cut_angle, 0.0, 89.0),
            "height_variation": (self.height_variation, 0.0, None),
            "offset_range": (self.offset_range, 0.0, None),
        }
        if all(value is None for value, _low, _high in bounds.values()):
            raise ValueError("a pin_variation group must vary at least one axis")
        for name, (value, low, high) in bounds.items():
            if value is None:
                continue
            if value[0] > value[1]:
                raise ValueError(f"{name} minimum must not exceed maximum")
            for entry in value:
                if entry < low:
                    raise ValueError(f"{name} must not be below {low}")
                if high is not None and entry > high:
                    raise ValueError(f"{name} must not exceed {high}")
        return self


class PinVariationParameters(BaseModel):
    """Version-1 parameters for ``operation.pin_variation.apply``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    groups: tuple[PinVariationGroup, ...] = Field(min_length=1)
