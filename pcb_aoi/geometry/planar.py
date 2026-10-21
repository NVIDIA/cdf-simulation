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

"""Provide dependency-free planar geometry operations."""

from __future__ import annotations

from collections.abc import Sequence


def cross_xy(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Return the two-dimensional cross product of two vectors.

    Args:
        first: First XY vector.
        second: Second XY vector.

    Returns:
        Signed scalar cross product.

    Example:

    .. code-block:: python

        from pcb_aoi.geometry.planar import cross_xy

        print(cross_xy((1.0, 0.0), (0.0, 1.0)))
        # 1.0

    """
    return first[0] * second[1] - first[1] * second[0]


def signed_turn_xy(
    origin: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Return the signed turn formed by three XY points.

    Args:
        origin: Point from which both edge vectors originate.
        first: First point in traversal order.
        second: Second point in traversal order.

    Returns:
        Positive for a counter-clockwise turn, negative for clockwise, and zero
        for collinear points.

    Example:

    .. code-block:: python

        from pcb_aoi.geometry.planar import signed_turn_xy

        print(signed_turn_xy((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))
        # 1.0

    """
    first_vector = (first[0] - origin[0], first[1] - origin[1])
    second_vector = (second[0] - origin[0], second[1] - origin[1])
    return cross_xy(first_vector, second_vector)


def convex_hull_xy(
    points_xy: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Return the counter-clockwise convex hull of XY points.

    Duplicate points are removed after coordinates are normalized to seven
    decimal places, preserving the tolerance used by procedural THT analysis.
    Collinear points inside an edge are omitted.

    Args:
        points_xy: XY points from which to construct the hull.

    Returns:
        Counter-clockwise exterior vertices without a repeated closing point.

    Example:

    .. code-block:: python

        from pcb_aoi.geometry.planar import convex_hull_xy

        hull = convex_hull_xy(((0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (0.0, 1.0)))
        print(hull)
        # ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

    """
    points = sorted(
        {(round(float(point[0]), 7), round(float(point[1]), 7)) for point in points_xy},
    )
    if len(points) <= 1:
        return tuple(points)

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and signed_turn_xy(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and signed_turn_xy(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])
