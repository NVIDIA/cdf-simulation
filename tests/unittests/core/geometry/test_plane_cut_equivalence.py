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

"""Hold the fast plane cutter to the scalar reference, result for result.

The cutter is on the per-lead path of every trimmed through-hole board, so it
is worth making fast -- but a cut mesh is authored straight back onto the stage
and a silently different silhouette is not something a downstream render would
report. The reference implementation therefore stays in the module, and the
optimized one must agree with it EXACTLY: same points, same order, same face
windings, same cap count.

The cases are chosen for the ways the algorithm can differ rather than for
coverage of shapes: a plane that misses, a plane that removes everything, a
plane through vertices (where a vertex is both on-plane and the end of a
crossing), a plane grazing a single face, and randomized meshes.
"""

import math
import sys
import unittest
from pathlib import Path
from random import Random

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pcb_aoi.geometry.plane_cut import (  # noqa: E402
    cut_mesh_by_plane,
    cut_mesh_by_plane_reference,
)


def _cylinder(
    segments: int,
    stacks: int,
    radius: float = 0.3,
    height: float = 3.0,
) -> tuple[list, list, list]:
    """Build a closed cylinder standing on the origin, as a lead does."""
    points = []
    for stack in range(stacks + 1):
        z = height * stack / stacks
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            points.append((radius * math.cos(angle), radius * math.sin(angle), z))
    counts, indices = [], []
    for stack in range(stacks):
        base = stack * segments
        for segment in range(segments):
            following = (segment + 1) % segments
            counts.append(4)
            indices.extend(
                (
                    base + segment,
                    base + following,
                    base + segments + following,
                    base + segments + segment,
                ),
            )
    counts.append(segments)
    indices.extend(range(segments - 1, -1, -1))
    top = stacks * segments
    counts.append(segments)
    indices.extend(range(top, top + segments))
    return points, counts, indices


def _cube() -> tuple[list, list, list]:
    """Build a unit cube centred on the origin."""
    points = [
        (x, y, z)
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ]
    counts = [4] * 6
    indices = [
        0, 1, 3, 2,
        4, 6, 7, 5,
        0, 4, 5, 1,
        2, 3, 7, 6,
        0, 2, 6, 4,
        1, 5, 7, 3,
    ]
    return points, counts, indices


def _random_mesh(seed: int) -> tuple[list, list, list]:
    """Build a jittered triangle soup with duplicate and coincident points."""
    rng = Random(seed)
    points = [
        (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        for _ in range(40)
    ]
    # Repeat a few points exactly, so the welder has something to merge.
    points.extend(points[:5])
    counts, indices = [], []
    for _ in range(60):
        counts.append(3)
        indices.extend(rng.randrange(len(points)) for _ in range(3))
    return points, counts, indices


class PlaneCutEquivalenceTest(unittest.TestCase):
    """The optimized cutter must be indistinguishable from the reference."""

    def _assert_same(self, points, counts, indices, plane, *, label: str) -> None:
        expected = cut_mesh_by_plane_reference(points, counts, indices, plane)
        actual = cut_mesh_by_plane(points, counts, indices, plane)

        self.assertEqual(len(actual[0]), len(expected[0]), "%s: point count" % label)
        for index, (got, want) in enumerate(zip(actual[0], expected[0], strict=True)):
            self.assertEqual(got, want, "%s: point %d" % (label, index))
        self.assertEqual(actual[1], expected[1], "%s: face vertex counts" % label)
        self.assertEqual(actual[2], expected[2], "%s: face vertex indices" % label)
        self.assertEqual(actual[3], expected[3], "%s: cap face count" % label)

    def test_plane_that_misses_the_mesh_entirely(self):
        points, counts, indices = _cylinder(16, 4)
        self._assert_same(
            points,
            counts,
            indices,
            ((0.0, 0.0, 99.0), (0.0, 0.0, 1.0)),
            label="miss",
        )

    def test_plane_that_removes_everything(self):
        points, counts, indices = _cylinder(16, 4)
        self._assert_same(
            points,
            counts,
            indices,
            ((0.0, 0.0, -99.0), (0.0, 0.0, 1.0)),
            label="remove-all",
        )

    def test_ordinary_trim_through_the_barrel(self):
        for segments, stacks in ((8, 2), (16, 4), (32, 12)):
            with self.subTest(segments=segments, stacks=stacks):
                points, counts, indices = _cylinder(segments, stacks)
                self._assert_same(
                    points,
                    counts,
                    indices,
                    ((0.0, 0.0, 1.4), (0.0, 0.0, 1.0)),
                    label="trim %dx%d" % (segments, stacks),
                )

    def test_plane_landing_exactly_on_a_ring_of_vertices(self):
        """Vertices already on the plane must not be classified to a side."""
        points, counts, indices = _cylinder(16, 4)
        self._assert_same(
            points,
            counts,
            indices,
            ((0.0, 0.0, 1.5), (0.0, 0.0, 1.0)),
            label="on-vertices",
        )

    def test_cube_cut_diagonally_through_its_corners(self):
        """The case that once produced no cap at all."""
        root = 1.0 / math.sqrt(2.0)
        points, counts, indices = _cube()
        self._assert_same(
            points,
            counts,
            indices,
            ((0.0, 0.0, 0.0), (root, 0.0, root)),
            label="corner-diagonal",
        )

    def test_oblique_planes_over_a_cylinder(self):
        points, counts, indices = _cylinder(24, 8)
        for index in range(6):
            angle = index * math.pi / 7.0
            normal = (math.sin(angle), 0.2, math.cos(angle))
            length = math.sqrt(sum(value * value for value in normal))
            with self.subTest(index=index):
                self._assert_same(
                    points,
                    counts,
                    indices,
                    (
                        (0.0, 0.0, 1.1),
                        tuple(value / length for value in normal),
                    ),
                    label="oblique %d" % index,
                )

    def test_randomized_meshes_with_duplicate_and_degenerate_faces(self):
        for seed in range(12):
            points, counts, indices = _random_mesh(seed)
            rng = Random(1000 + seed)
            normal = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            length = math.sqrt(sum(value * value for value in normal)) or 1.0
            with self.subTest(seed=seed):
                self._assert_same(
                    points,
                    counts,
                    indices,
                    (
                        (rng.uniform(-0.5, 0.5), 0.0, 0.0),
                        tuple(value / length for value in normal),
                    ),
                    label="random %d" % seed,
                )

    def test_faces_below_three_vertices_are_still_dropped(self):
        points, counts, indices = _cube()
        counts = [*counts, 2]
        indices = [*indices, 0, 1]
        self._assert_same(
            points,
            counts,
            indices,
            ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            label="degenerate-face",
        )

    def test_an_empty_mesh_round_trips(self):
        self._assert_same([], [], [], ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), label="empty")




class QuantizedKeysTest(unittest.TestCase):
    """The bucket a point falls in must not move when the loop is batched."""

    def _points(self, seed: int, count: int):
        rng = Random(seed)
        return [
            (
                rng.uniform(-20.0, 20.0),
                rng.uniform(-20.0, 20.0),
                rng.uniform(-20.0, 20.0),
            )
            for _ in range(count)
        ]

    def test_round_mode_matches_the_scalar_formula(self):
        from pcb_aoi.geometry.plane_cut import quantized_keys

        for quantum in (1.0e-6, 1.0e-3, 0.5, 7.0):
            points = self._points(int(quantum * 1000) + 1, 500)
            expected = [
                tuple(int(round(value / quantum)) for value in point)
                for point in points
            ]
            with self.subTest(quantum=quantum):
                self.assertEqual(quantized_keys(points, quantum), expected)

    def test_floor_mode_matches_the_scalar_formula(self):
        from pcb_aoi.geometry.plane_cut import quantized_keys

        for quantum in (1.0e-7, 1.0e-2, 3.0):
            points = self._points(int(quantum * 100) + 7, 500)
            expected = [
                tuple(int(math.floor(value / quantum)) for value in point)
                for point in points
            ]
            with self.subTest(quantum=quantum):
                self.assertEqual(
                    quantized_keys(points, quantum, mode="floor"), expected,
                )

    def test_exact_halves_round_the_way_python_rounds_them(self):
        """Both round half to EVEN; picking half-up here would move a bucket."""
        from pcb_aoi.geometry.plane_cut import quantized_keys

        points = [(0.5, 1.5, 2.5), (-0.5, -1.5, -2.5), (3.5, 4.5, 5.5)]
        expected = [
            tuple(int(round(value)) for value in point) for point in points
        ]

        self.assertEqual(quantized_keys(points, 1.0), expected)

    def test_no_points_gives_no_keys(self):
        from pcb_aoi.geometry.plane_cut import quantized_keys

        self.assertEqual(quantized_keys([], 1.0e-6), [])


class WelderBucketNeighbourhoodTest(unittest.TestCase):
    """Coincident points must weld wherever the bucket grid happens to fall.

    `_Welder` buckets points by `floor(coordinate / tolerance)` and then scans
    the neighbouring buckets. Scanning only the six AXIAL neighbours misses the
    diagonals: two points within tolerance on every axis can still differ by
    one bucket on two axes at once, and then the welder keeps both. A duplicate
    seam vertex is not cosmetic -- the cap is chained from welded indices, so it
    can leave an open loop or a degenerate edge.
    """

    def _welder(self, tolerance=1.0):
        from pcb_aoi.geometry.plane_cut import _Welder

        return _Welder([], tolerance=tolerance)

    def test_an_axial_neighbour_welds(self):
        """The case the six-offset scan already covered."""
        welder = self._welder()

        first = welder.add((0.99, 0.0, 0.0))
        second = welder.add((1.01, 0.0, 0.0))

        self.assertEqual(first, second)
        self.assertEqual(len(welder.points), 1)

    def test_a_two_axis_diagonal_neighbour_welds(self):
        """0.02 apart per axis against a tolerance of 1.0."""
        welder = self._welder()

        first = welder.add((0.99, 0.99, 0.0))
        second = welder.add((1.01, 1.01, 0.0))

        self.assertEqual(first, second)
        self.assertEqual(len(welder.points), 1)

    def test_a_three_axis_diagonal_neighbour_welds(self):
        welder = self._welder()

        first = welder.add((0.99, 0.99, 0.99))
        second = welder.add((1.01, 1.01, 1.01))

        self.assertEqual(first, second)
        self.assertEqual(len(welder.points), 1)

    def test_every_corner_of_the_surrounding_cube_is_reachable(self):
        """All 26 neighbours, not just the six with a shared face."""
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    welder = self._welder()
                    anchor = (1.0 - 0.01 * dx, 1.0 - 0.01 * dy, 1.0 - 0.01 * dz)
                    probe = (1.0 + 0.01 * dx, 1.0 + 0.01 * dy, 1.0 + 0.01 * dz)
                    with self.subTest(offset=(dx, dy, dz)):
                        self.assertEqual(welder.add(anchor), welder.add(probe))

    def test_points_beyond_the_tolerance_still_stay_apart(self):
        """The fix must not weld things that are genuinely distinct."""
        welder = self._welder(tolerance=0.001)

        first = welder.add((0.0, 0.0, 0.0))
        second = welder.add((0.5, 0.5, 0.5))

        self.assertNotEqual(first, second)
        self.assertEqual(len(welder.points), 2)

    def test_a_point_just_outside_tolerance_on_one_axis_stays_apart(self):
        welder = self._welder(tolerance=0.01)

        first = welder.add((0.0, 0.0, 0.0))
        second = welder.add((0.005, 0.005, 0.02))

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
