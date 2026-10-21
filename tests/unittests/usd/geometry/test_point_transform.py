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

"""Hold the batched point transform to ``Gf.Matrix4d.Transform``, exactly.

Moving a mesh between local and world space one ``Gf.Vec3d`` at a time is the
most repeated scalar loop in this codebase -- nine call sites, and on a board
with two thousand plated holes it runs into the millions of points. Batching it
is worth roughly twenty times, tuple round-trip included.

It has to be EXACT, not close. These points feed the plane cutter, whose
on-plane classification is a threshold test, and they are written back onto the
stage as geometry. ``points @ matrix[:3, :3]`` is NOT exact: BLAS reassociates
the three terms and disagrees with Gf in the last place on about a fifth of
inputs. Accumulating the columns explicitly, in Gf's own order, agrees on all of
them, which is what these tests assert.
"""

from __future__ import annotations


import sys
import unittest
from pathlib import Path
from random import Random

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

try:
    from pxr import Gf
except ImportError:  # the CI unit-test image installs numpy only
    Gf = None

from pcb_aoi.geometry.transform import (  # noqa: E402
    transform_points,
    transform_points_as_tuples,
)


def _matrix(seed: int) -> Gf.Matrix4d:
    """Build a rotation, scale and translation, as a real xform cache returns."""
    rng = Random(seed)
    rotation = Gf.Matrix4d(1.0)
    rotation.SetRotateOnly(
        Gf.Rotation(
            Gf.Vec3d(
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
            ).GetNormalized(),
            rng.uniform(0.0, 360.0),
        ),
    )
    rotation.SetTranslateOnly(
        Gf.Vec3d(
            rng.uniform(-9.0, 9.0),
            rng.uniform(-9.0, 9.0),
            rng.uniform(-9.0, 9.0),
        ),
    )
    scale = Gf.Matrix4d(1.0)
    scale.SetScale(
        Gf.Vec3d(
            rng.uniform(0.2, 3.0),
            rng.uniform(0.2, 3.0),
            rng.uniform(0.2, 3.0),
        ),
    )
    return scale * rotation


def _points(seed: int, count: int) -> list[tuple[float, float, float]]:
    """Build a point cloud spanning a board-sized volume."""
    rng = Random(10_000 + seed)
    return [
        (
            rng.uniform(-9.0, 9.0),
            rng.uniform(-9.0, 9.0),
            rng.uniform(-9.0, 9.0),
        )
        for _ in range(count)
    ]


@unittest.skipUnless(Gf is not None, "usd-core is not installed")
class TransformPointsEquivalenceTest(unittest.TestCase):
    """Every coordinate must equal what Gf would have produced."""

    def _reference(self, matrix, points) -> np.ndarray:
        return np.array(
            [list(matrix.Transform(Gf.Vec3d(*point))) for point in points],
            dtype=np.float64,
        )

    def test_matches_gf_bit_for_bit_over_many_matrices(self):
        mismatches = 0
        total = 0
        for seed in range(25):
            matrix = _matrix(seed)
            points = _points(seed, 200)
            expected = self._reference(matrix, points)
            actual = transform_points(matrix, points)
            total += expected.size
            mismatches += int((actual != expected).sum())

        self.assertEqual(mismatches, 0, "%d of %d coordinates differ" % (
            mismatches, total))

    def test_a_plain_matmul_would_not_have_passed(self):
        """Guard the reason the implementation looks the way it does."""
        differing = 0
        for seed in range(25):
            matrix = _matrix(seed)
            points = _points(seed, 200)
            expected = self._reference(matrix, points)
            values = np.array(
                [[matrix[row][column] for column in range(4)] for row in range(4)],
                dtype=np.float64,
            )
            naive = np.asarray(points, dtype=np.float64) @ values[:3, :3]
            naive = naive + values[3, :3]
            differing += int((naive != expected).sum())

        self.assertGreater(
            differing,
            0,
            "matmul now agrees with Gf; the explicit accumulation may be "
            "simplifiable, but re-measure before changing it",
        )

    def test_a_projective_matrix_divides_by_w(self):
        matrix = Gf.Matrix4d(
            1.0, 0.0, 0.0, 0.25,
            0.0, 1.0, 0.0, 0.5,
            0.0, 0.0, 1.0, -0.125,
            0.0, 0.0, 0.0, 1.0,
        )
        points = _points(3, 64)

        actual = transform_points(matrix, points)

        self.assertEqual(mismatch_count(actual, self._reference(matrix, points)), 0)

    def test_identity_returns_the_points_unchanged(self):
        points = _points(5, 32)

        actual = transform_points(Gf.Matrix4d(1.0), points)

        self.assertEqual([tuple(row) for row in actual.tolist()], points)

    def test_a_nested_sequence_stands_in_for_a_matrix(self):
        """The helper must not drag pxr into a module that has no other use for it."""
        values = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
        ]

        actual = transform_points(values, [(1.0, 1.0, 1.0)])

        self.assertEqual(actual.tolist(), [[3.0, 5.0, 7.0]])

    def test_no_points_gives_an_empty_result_of_the_right_shape(self):
        actual = transform_points(Gf.Matrix4d(1.0), [])

        self.assertEqual(actual.shape, (0, 3))
        self.assertEqual(transform_points_as_tuples(Gf.Matrix4d(1.0), []), ())

    def test_the_tuple_form_is_what_the_scalar_loop_produced(self):
        for seed in range(6):
            matrix = _matrix(seed)
            points = _points(seed, 128)
            expected = tuple(
                tuple(matrix.Transform(Gf.Vec3d(*point))) for point in points
            )
            with self.subTest(seed=seed):
                self.assertEqual(
                    transform_points_as_tuples(matrix, points),
                    expected,
                )

    def test_the_tuple_form_really_is_tuples(self):
        """A list would change the type the plane cutter hands back."""
        result = transform_points_as_tuples(Gf.Matrix4d(1.0), [(1.0, 2.0, 3.0)])

        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], tuple)
        self.assertIsInstance(result[0][0], float)

    def test_gf_vec3f_input_is_accepted_as_a_mesh_supplies_it(self):
        """`GetPointsAttr().Get()` returns Vec3f, not python tuples."""
        matrix = _matrix(1)
        points = [Gf.Vec3f(1.5, -2.25, 0.75), Gf.Vec3f(-3.0, 4.5, 6.0)]
        expected = tuple(
            tuple(matrix.Transform(Gf.Vec3d(*point))) for point in points
        )

        self.assertEqual(transform_points_as_tuples(matrix, points), expected)


def mismatch_count(actual: np.ndarray, expected: np.ndarray) -> int:
    """Return how many coordinates differ, for a readable assertion."""
    return int((actual != expected).sum())


if __name__ == "__main__":
    unittest.main()
