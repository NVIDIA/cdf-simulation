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

"""Move a whole mesh between spaces in one pass.

``[matrix.Transform(Gf.Vec3d(*point)) for point in points]`` is the most
repeated scalar loop in this codebase. Every occurrence builds one Python object
and crosses the C++ boundary once per point, and a board with two thousand
plated holes runs it into the millions: measured, 2.3 s for 200k points, against
0.11 s here including the round trip back to tuples.

Exactness is the constraint, not an aspiration. These points feed the plane
cutter, whose on-plane test is a threshold, and they are written back onto the
stage as geometry, so a coordinate that differs in the last place is a moved
vertex. ``points @ matrix[:3, :3]`` does not qualify -- BLAS reassociates the
three terms and disagrees with Gf on about a fifth of coordinates. Accumulating
the columns explicitly, in the order Gf accumulates them, agrees on all of
them; ``test_point_transform`` asserts both halves of that sentence.

Used by ``tht/pin_cut_apply`` (both directions), ``tht/lead_discovery``,
and ``tht/pin_variation_apply``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Point = tuple[float, float, float]


def transform_points(
    matrix: object,
    points: Sequence[object],
) -> np.ndarray:
    """Return every point moved by a 4x4 row-vector matrix.

    Args:
        matrix: Anything indexable as ``matrix[row][column]`` over a 4x4 of
            floats, which a ``Gf.Matrix4d`` and a nested sequence both are.
            Kept structural so this module needs no USD import.
        points: Points to move. Each must unpack into three coordinates, so a
            ``Gf.Vec3f`` straight off a mesh attribute is accepted.

    Returns:
        An ``(N, 3)`` float64 array, matching ``Gf.Matrix4d.Transform`` exactly.

    Example:

    .. code-block:: python

        print(transform_points([[1, 0, 0, 0], [0, 1, 0, 0],
                                [0, 0, 1, 0], [1, 2, 3, 1]],
                               [(0.0, 0.0, 0.0)]).tolist())
        # [[1.0, 2.0, 3.0]]

    """
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    array = array.reshape(-1, 3)
    values = np.asarray(
        [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )

    moved = np.empty_like(array)
    for column in range(3):
        # Left to right, one term at a time: this is the association Gf uses,
        # and reassociating it costs bit-exactness. See the module docstring.
        total = array[:, 0] * values[0, column]
        total = total + array[:, 1] * values[1, column]
        total = total + array[:, 2] * values[2, column]
        moved[:, column] = total + values[3, column]

    homogeneous = array[:, 0] * values[0, 3]
    homogeneous = homogeneous + array[:, 1] * values[1, 3]
    homogeneous = homogeneous + array[:, 2] * values[2, 3]
    homogeneous = homogeneous + values[3, 3]
    # Reciprocal once then multiply, which is what Gf does. Dividing each
    # component instead is a different rounding and misses on a quarter of
    # coordinates under a projective matrix -- measured, not assumed. An affine
    # matrix leaves this exactly 1.0, where both spellings agree anyway, so the
    # projective case costs the affine one nothing.
    return moved * (1.0 / homogeneous)[:, None]


def transform_points_as_tuples(
    matrix: object,
    points: Sequence[object],
) -> tuple[Point, ...]:
    """Return the moved points as tuples, the shape the callers already pass on.

    Args:
        matrix: 4x4 row-vector matrix, as `transform_points` accepts.
        points: Points to move.

    Returns:
        One ``(x, y, z)`` tuple of floats per point, identical to what the
        per-point ``Gf`` loop returned.

    Example:

    .. code-block:: python

        print(transform_points_as_tuples([[1, 0, 0, 0], [0, 1, 0, 0],
                                          [0, 0, 1, 0], [0, 0, 0, 1]],
                                         [(1.0, 2.0, 3.0)]))
        # ((1.0, 2.0, 3.0),)

    """
    return tuple(
        tuple(row) for row in transform_points(matrix, points).tolist()
    )
