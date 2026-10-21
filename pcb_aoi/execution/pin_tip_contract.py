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

"""Transport contract for annotation-only pin-tip proxy generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cdf.component_filter import ComponentFilter

from .pin_variation_contract import NonEmptyString


class PinTipParameters(BaseModel):
    """Version-1 parameters for ``operation.pin_tip.generate``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    class_name: NonEmptyString = "pin_tip"
    side: Literal["top", "bottom"]
    offset_mm: float = Field(default=0.002, gt=0.0)
    components: tuple[NonEmptyString, ...] | None = None
    component_filter: ComponentFilter | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> PinTipParameters:
        """Reject mixing normalized and legacy selectors.

        Returns:
            Validated pin-tip parameters.

        Raises:
            ValueError: If both selector forms are supplied.

        """
        if self.components is not None and self.component_filter is not None:
            raise ValueError("components and component_filter are mutually exclusive")
        return self

    @field_validator("components")
    @classmethod
    def reject_duplicate_components(
        cls,
        value: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        """Reject duplicate refdes entries.

        Args:
            value: Optional component allowlist.

        Returns:
            Validated allowlist or None.

        Raises:
            ValueError: If a reference designator is repeated.

        """
        if value is not None and len(set(value)) != len(value):
            raise ValueError("components must not contain duplicates")
        return value
