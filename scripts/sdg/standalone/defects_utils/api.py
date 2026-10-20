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

"""Public component-defect transform API.

Each function computes one absolute component transform without applying it to
USD or Fabric.  Selection, semantics, and scene mutation remain the caller's
responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from defects_utils.component_rotation import (
    _compute_board_normal_rotation_transform,
    _compute_shift_transform,
    _compute_sideflip_transform,
)
from defects_utils.tombstone import _compute_tombstone_transform

__all__ = [
    "DefectTransform",
    "reverse_polarity",
    "shift",
    "sideflip",
    "tombstone",
]


@dataclass(frozen=True)
class DefectTransform:
    """Absolute row-major component transform and sampled diagnostics.

    Args:
        flat_transform: Flattened 4x4 transform in row-major order.
        parameters: Sampled values and geometry diagnostics.

    Example:

    .. code-block:: python

        from defects_utils.api import DefectTransform

        result = DefectTransform([0.0] * 16, {"angle_deg": 15.0})
        print(result.parameters["angle_deg"])
        15.0
    """

    flat_transform: list[float]
    #: Flattened absolute component transform in row-major order.

    parameters: dict[str, Any]
    #: Sampled defect values and component-geometry diagnostics.


def _matrix4(original_transform: np.ndarray) -> np.ndarray:
    """Validate and normalize an API transform argument.

    Args:
        original_transform: Component transform to validate.

    Returns:
        Floating-point 4x4 matrix.

    Raises:
        ValueError: If the input does not have shape (4, 4).
    """
    matrix = np.asarray(original_transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"original_transform must have shape (4, 4); received {matrix.shape}"
        )
    return matrix


def _angle_range(angle_min: float, angle_max: float) -> tuple[float, float]:
    """Validate a closed defect angle range.

    Args:
        angle_min: Lower angle bound.
        angle_max: Upper angle bound.

    Returns:
        Validated minimum and maximum angles.

    Raises:
        ValueError: If the minimum exceeds the maximum.
    """
    minimum = float(angle_min)
    maximum = float(angle_max)
    if minimum > maximum:
        raise ValueError(
            f"angle_min ({minimum}) must be less than or equal to angle_max ({maximum})"
        )
    return minimum, maximum


def shift(
    prim: object,
    original_transform: np.ndarray,
    *,
    translate_range: float,
    rotate_z_range: float,
    component_instance: dict[str, Any] | None = None,
    variant_index: int | None = None,
) -> DefectTransform:
    """Compute a pick-and-place shift defect for one component.

    Args:
        prim: Component USD prim whose geometry defines the rotation frame.
        original_transform: Original component transform in row-major form.
        translate_range: Symmetric X/Y translation bound.
        rotate_z_range: Symmetric board-normal rotation bound.
        component_instance: Optional component metadata with solder prim paths.

    Returns:
        Absolute transform and sampled shift parameters.

    Raises:
        ValueError: If the matrix shape is invalid or a range is negative.

    Example:

    .. code-block:: python

        from defects_utils.api import shift

        result = shift(
            prim,
            original_transform,
            translate_range=0.2,
            rotate_z_range=15.0,
        )  # doctest: +SKIP
        print(len(result.flat_transform))  # doctest: +SKIP
        16
    """
    flat_transform, parameters = _compute_shift_transform(
        prim,
        _matrix4(original_transform),
        float(translate_range),
        float(rotate_z_range),
        component_instance=component_instance,
        variant_index=variant_index,
    )
    return DefectTransform(flat_transform, parameters)


def sideflip(
    prim: object,
    original_transform: np.ndarray,
    *,
    angle_min: float,
    angle_max: float,
    component_instance: dict[str, Any] | None = None,
    variant_index: int | None = None,
    rotate_z_jitter_deg: float = 0.0,
    slide_range_mm: float = 0.0,
) -> DefectTransform:
    """Compute a length-axis sideflip defect for one component.

    Args:
        prim: Component USD prim whose geometry defines the rotation frame.
        original_transform: Original component transform in row-major form.
        angle_min: Minimum sideflip magnitude.
        angle_max: Maximum sideflip magnitude.
        component_instance: Optional component metadata with solder prim paths.

    Returns:
        Absolute transform and sampled sideflip parameters.

    Raises:
        ValueError: If the matrix shape or angle range is invalid.

    Example:

    .. code-block:: python

        from defects_utils.api import sideflip

        result = sideflip(
            prim,
            original_transform,
            angle_min=70.0,
            angle_max=90.0,
        )  # doctest: +SKIP
        print(result.parameters["angle_deg"])  # doctest: +SKIP
    """
    minimum, maximum = _angle_range(angle_min, angle_max)
    if float(rotate_z_jitter_deg) < 0.0 or float(slide_range_mm) < 0.0:
        raise ValueError("Sideflip perturbation bounds must be non-negative")
    flat_transform, parameters = _compute_sideflip_transform(
        prim,
        _matrix4(original_transform),
        minimum,
        maximum,
        component_instance=component_instance,
        variant_index=variant_index,
        rotate_z_jitter_deg=float(rotate_z_jitter_deg),
        slide_range_mm=float(slide_range_mm),
    )
    return DefectTransform(flat_transform, parameters)


def reverse_polarity(
    prim: object,
    original_transform: np.ndarray,
    *,
    component_instance: dict[str, Any] | None = None,
) -> DefectTransform:
    """Compute a 180-degree board-normal polarity reversal.

    Args:
        prim: Component USD prim whose geometry defines the rotation frame.
        original_transform: Original component transform in row-major form.
        component_instance: Optional component metadata with solder prim paths.

    Returns:
        Absolute transform and polarity-reversal parameters.

    Raises:
        ValueError: If the matrix shape is invalid.

    Example:

    .. code-block:: python

        from defects_utils.api import reverse_polarity

        result = reverse_polarity(prim, original_transform)  # doctest: +SKIP
        print(result.parameters["angle_deg"])  # doctest: +SKIP
        180.0
    """
    flat_transform, parameters = _compute_board_normal_rotation_transform(
        prim,
        _matrix4(original_transform),
        180.0,
        component_instance=component_instance,
    )
    return DefectTransform(flat_transform, parameters)


def tombstone(
    prim: object,
    original_transform: np.ndarray,
    *,
    angle_min: float,
    angle_max: float,
    component_instance: dict[str, Any] | None = None,
) -> DefectTransform:
    """Compute a solder-interface hinge tombstone defect.

    Args:
        prim: Component USD prim whose geometry defines the rotation frame.
        original_transform: Original component transform in row-major form.
        angle_min: Minimum tombstone magnitude.
        angle_max: Maximum tombstone magnitude.
        component_instance: Optional component metadata with solder prim paths.

    Returns:
        Absolute transform and sampled tombstone parameters.

    Raises:
        ValueError: If the matrix shape or angle range is invalid.

    Example:

    .. code-block:: python

        from defects_utils.api import tombstone

        result = tombstone(
            prim,
            original_transform,
            angle_min=70.0,
            angle_max=90.0,
        )  # doctest: +SKIP
        print(result.parameters["pivot_local"])  # doctest: +SKIP
    """
    minimum, maximum = _angle_range(angle_min, angle_max)
    flat_transform, parameters = _compute_tombstone_transform(
        prim,
        _matrix4(original_transform),
        minimum,
        maximum,
        component_instance=component_instance,
    )
    return DefectTransform(flat_transform, parameters)
