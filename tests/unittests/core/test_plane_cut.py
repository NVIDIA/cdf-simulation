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

"""Cover the plane cutter on shapes whose section is known in advance.

Areas rather than face counts. A cap that fans an annulus over its own hole has
the right number of faces and the wrong area, and that is precisely the failure
this cutter exists to avoid: the joint's section is an annulus and the lead's
own face sits coplanar inside it.
"""

from __future__ import annotations

import math
import unittest

from pcb_aoi.geometry.plane_cut import (
    _chain_loops,
    close_open_boundaries,
    cut_mesh_by_plane,
    weld_seams,
)

SEGMENTS = 64


def cube(half: float = 1.0):
    """Return an axis-aligned cube spanning -half..half."""
    points = [
        (-half, -half, -half),
        (half, -half, -half),
        (half, half, -half),
        (-half, half, -half),
        (-half, -half, half),
        (half, -half, half),
        (half, half, half),
        (-half, half, half),
    ]
    indices = (
        0,
        3,
        2,
        1,
        4,
        5,
        6,
        7,
        0,
        1,
        5,
        4,
        1,
        2,
        6,
        5,
        2,
        3,
        7,
        6,
        3,
        0,
        4,
        7,
    )
    return points, (4,) * 6, indices


def tube(outer: float, inner: float, z_low: float, z_high: float):
    """Return a closed tube: concentric walls with annular end caps."""
    points: list[tuple[float, float, float]] = []
    counts: list[int] = []
    indices: list[int] = []
    for radius in (outer, inner):
        for z in (z_low, z_high):
            for segment in range(SEGMENTS):
                angle = 2.0 * math.pi * segment / SEGMENTS
                points.append((radius * math.cos(angle), radius * math.sin(angle), z))

    def quad(a: int, b: int, c: int, d: int) -> None:
        """Append one quadrilateral."""
        counts.append(4)
        indices.extend((a, b, c, d))

    for segment in range(SEGMENTS):
        nxt = (segment + 1) % SEGMENTS
        quad(segment, nxt, SEGMENTS + nxt, SEGMENTS + segment)
        quad(
            3 * SEGMENTS + segment,
            3 * SEGMENTS + nxt,
            2 * SEGMENTS + nxt,
            2 * SEGMENTS + segment,
        )
        quad(segment, 2 * SEGMENTS + segment, 2 * SEGMENTS + nxt, nxt)
        quad(
            SEGMENTS + segment,
            SEGMENTS + nxt,
            3 * SEGMENTS + nxt,
            3 * SEGMENTS + segment,
        )
    return points, tuple(counts), tuple(indices)


def polygon_area(radius: float) -> float:
    """Return the area of the regular polygon the tube is actually made of."""
    return 0.5 * SEGMENTS * radius * radius * math.sin(2.0 * math.pi / SEGMENTS)


def section_area(points, counts, indices, plane, tolerance: float = 1e-7) -> float:
    """Return the total area of the faces lying in the cut plane."""
    origin, normal = plane
    total, offset = 0.0, 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        if not all(
            abs(
                sum(
                    (points[index][axis] - origin[axis]) * normal[axis]
                    for axis in range(3)
                )
            )
            <= tolerance
            for index in face
        ):
            continue
        base = points[face[0]]
        for step in range(1, count - 1):
            first = tuple(points[face[step]][axis] - base[axis] for axis in range(3))
            second = tuple(
                points[face[step + 1]][axis] - base[axis] for axis in range(3)
            )
            cross = (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            )
            total += 0.5 * math.sqrt(sum(value * value for value in cross))
    return total


class PlaneCutTest(unittest.TestCase):
    """Cover cutting and capping."""

    def test_flat_cut_through_a_solid_caps_with_its_section(self):
        """A flat blade leaves a section, and the section is closed."""
        points, counts, indices = cube()
        plane = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        cut_points, cut_counts, cut_indices, caps = cut_mesh_by_plane(
            points, counts, indices, plane
        )
        # Triangles, not one n-gon. Hydra fans an n-gon from its first vertex,
        # which is only right while it is convex, and a joint's section is not.
        self.assertEqual(caps, 2)
        self.assertAlmostEqual(
            section_area(cut_points, cut_counts, cut_indices, plane), 4.0, places=6
        )

    def test_angled_cut_through_vertices_still_caps(self):
        """A plane meeting corners exactly must not be mistaken for a miss.

        At 45 degrees through a cube's centre, four vertices land ON the plane,
        so each is both an on-plane point and the end of a crossing. Counting
        "the two things the blade opened" per face misses every one of them and
        produces no cap at all.
        """
        points, counts, indices = cube()
        normal = (math.sin(math.radians(45)), 0.0, math.cos(math.radians(45)))
        plane = ((0.0, 0.0, 0.0), normal)
        cut_points, cut_counts, cut_indices, caps = cut_mesh_by_plane(
            points, counts, indices, plane
        )
        self.assertEqual(caps, 2)
        self.assertAlmostEqual(
            section_area(cut_points, cut_counts, cut_indices, plane),
            2.0 * 2.0 * math.sqrt(2.0),
            places=6,
        )

    def test_annular_section_is_not_filled_over_its_hole(self):
        """The joint's case: a tube's section is a ring, not a disc.

        Fanning it from a centroid would paste solder over the lead's own face,
        which is coplanar with it. The area is what catches that -- the face
        count is the same either way.
        """
        points, counts, indices = tube(1.0, 0.5, -1.0, 1.0)
        plane = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        cut_points, cut_counts, cut_indices, caps = cut_mesh_by_plane(
            points, counts, indices, plane
        )
        self.assertEqual(caps, 2 * SEGMENTS)
        self.assertAlmostEqual(
            section_area(cut_points, cut_counts, cut_indices, plane),
            polygon_area(1.0) - polygon_area(0.5),
            places=4,
        )

    def test_annular_section_survives_an_angled_cut(self):
        """Tipping the blade must not turn the ring back into a disc."""
        points, counts, indices = tube(1.0, 0.5, -1.0, 1.0)
        normal = (math.sin(math.radians(30)), 0.0, math.cos(math.radians(30)))
        plane = ((0.0, 0.0, 0.0), normal)
        cut_points, cut_counts, cut_indices, _ = cut_mesh_by_plane(
            points, counts, indices, plane
        )
        self.assertAlmostEqual(
            section_area(cut_points, cut_counts, cut_indices, plane),
            (polygon_area(1.0) - polygon_area(0.5)) / math.cos(math.radians(30)),
            places=3,
        )

    def test_a_concave_section_is_triangulated_not_fanned(self):
        """The case a single n-gon gets wrong.

        An L-shaped prism cut flat has an L-shaped section. Fanning that from
        its first vertex covers ground outside the polygon, so the area comes
        out too large -- which is exactly how the joint's angled section
        rendered as see-through.
        """
        # An L in plan: six vertices, one reflex.
        plan = [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0)]
        points = [(x, y, -1.0) for x, y in plan] + [(x, y, 1.0) for x, y in plan]
        counts = [6, 6] + [4] * 6
        indices = list(range(6)) + list(range(11, 5, -1))
        for position in range(6):
            nxt = (position + 1) % 6
            indices.extend((position, nxt, 6 + nxt, 6 + position))
        plane = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        cut_points, cut_counts, cut_indices, caps = cut_mesh_by_plane(
            points, tuple(counts), tuple(indices), plane
        )
        self.assertGreaterEqual(caps, 4)
        self.assertTrue(all(count == 3 for count in cut_counts[-caps:]))
        self.assertAlmostEqual(
            section_area(cut_points, cut_counts, cut_indices, plane),
            3.0 * 1.0 + 1.0 * 2.0,
            places=6,
        )

    def test_a_plane_clear_of_the_mesh_changes_nothing(self):
        """A blade that never reaches the mesh must leave it untouched."""
        points, counts, indices = cube()
        _cut_points, cut_counts, cut_indices, caps = cut_mesh_by_plane(
            points, counts, indices, ((0.0, 0.0, 5.0), (0.0, 0.0, 1.0))
        )
        self.assertEqual(caps, 0)
        self.assertEqual(cut_counts, counts)
        self.assertEqual(cut_indices, tuple(indices))

    def test_a_plane_past_the_mesh_removes_it(self):
        """A blade beyond the mesh takes all of it, not some of it."""
        points, counts, indices = cube()
        _, cut_counts, _, caps = cut_mesh_by_plane(
            points, counts, indices, ((0.0, 0.0, -5.0), (0.0, 0.0, 1.0))
        )
        self.assertEqual(caps, 0)
        self.assertEqual(cut_counts, ())


class CloseOpenBoundariesTest(unittest.TestCase):
    """Sealing a hole must close the surface without altering the shape."""

    SEGMENTS = 16

    def _open_tube(self):
        """Return a cylinder with a bottom cap and no top cap."""
        points, counts, indices = [], [], []
        for z in (-1.0, 1.0):
            for step in range(self.SEGMENTS):
                angle = 2.0 * math.pi * step / self.SEGMENTS
                points.append((math.cos(angle), math.sin(angle), z))
        for step in range(self.SEGMENTS):
            nxt = (step + 1) % self.SEGMENTS
            counts.append(4)
            indices.extend((step, nxt, self.SEGMENTS + nxt, self.SEGMENTS + step))
        for step in range(1, self.SEGMENTS - 1):
            counts.append(3)
            indices.extend((0, step + 1, step))
        return points, tuple(counts), tuple(indices)

    @staticmethod
    def _edge_use(counts, indices):
        """Return how many times each directed edge is used."""
        used, offset = {}, 0
        for count in counts:
            face = indices[offset : offset + count]
            offset += count
            for position in range(count):
                edge = (face[position], face[(position + 1) % count])
                used[edge] = used.get(edge, 0) + 1
        return used

    def _boundary_edges(self, counts, indices):
        """Return how many edges are used by only one face."""
        used = self._edge_use(counts, indices)
        return sum(1 for (a, b) in used if used.get((b, a), 0) == 0)

    @staticmethod
    def _volume(points, counts, indices):
        """Return the enclosed volume by the divergence theorem."""
        total, offset = 0.0, 0
        for count in counts:
            face = indices[offset : offset + count]
            offset += count
            for corner in range(1, count - 1):
                a, b, c = (
                    points[face[0]],
                    points[face[corner]],
                    points[face[corner + 1]],
                )
                total += (
                    a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])
                ) / 6.0
        return abs(total)

    def test_open_tube_becomes_closed(self):
        """A tube missing its lid comes back watertight."""
        points, counts, indices = self._open_tube()
        self.assertEqual(self._boundary_edges(counts, indices), self.SEGMENTS)
        sealed = close_open_boundaries(points, counts, indices)
        self.assertEqual(self._boundary_edges(sealed[1], sealed[2]), 0)
        self.assertEqual(sealed[3], 1)

    def test_sealing_adds_no_points_and_keeps_the_volume(self):
        """The cap reuses the rim's own vertices, so the shape is unchanged."""
        points, counts, indices = self._open_tube()
        sealed = close_open_boundaries(points, counts, indices)
        self.assertEqual(len(sealed[0]), len(points))
        expected = 0.5 * self.SEGMENTS * math.sin(2.0 * math.pi / self.SEGMENTS) * 2.0
        self.assertAlmostEqual(
            self._volume(sealed[0], sealed[1], sealed[2]), expected, places=9
        )

    def test_a_closed_mesh_is_left_alone(self):
        """Sealing something already closed must change nothing."""
        points, counts, indices = self._open_tube()
        sealed = close_open_boundaries(points, counts, indices)
        again = close_open_boundaries(sealed[0], sealed[1], sealed[2])
        self.assertEqual(again[3], 0)
        self.assertEqual(again[1], sealed[1])
        self.assertEqual(again[2], sealed[2])

    def test_nested_rims_are_capped_as_an_annulus(self):
        """Two coplanar rims leave the inner opening bare, not covered.

        This is the case the lead's end face depends on: a bath clipped there
        has an outer rim and an inner one, and filling them separately would
        put a disc over the very face the dataset is about.
        """
        points, counts, indices = [], [], []
        for radius in (1.0, 0.4):
            for step in range(self.SEGMENTS):
                angle = 2.0 * math.pi * step / self.SEGMENTS
                points.append((radius * math.cos(angle), radius * math.sin(angle), 0.0))
        outer = list(range(self.SEGMENTS))
        inner = list(range(self.SEGMENTS, 2 * self.SEGMENTS))
        # A ring of side walls, open top and bottom, so the rims are nested.
        for step in range(self.SEGMENTS):
            nxt = (step + 1) % self.SEGMENTS
            counts.append(4)
            indices.extend((outer[step], outer[nxt], inner[nxt], inner[step]))
        sealed = close_open_boundaries(points, tuple(counts), tuple(indices))
        self.assertEqual(len(sealed[0]), len(points))
        area = self._triangle_area(sealed[0], sealed[1], sealed[2])
        ring = (
            0.5
            * self.SEGMENTS
            * math.sin(2.0 * math.pi / self.SEGMENTS)
            * (1.0 - 0.4**2)
        )
        self.assertAlmostEqual(area, 2.0 * ring, places=9)

    @staticmethod
    def _triangle_area(points, counts, indices):
        """Return the summed area of every face."""
        total, offset = 0.0, 0
        for count in counts:
            face = indices[offset : offset + count]
            offset += count
            for corner in range(1, count - 1):
                a, b, c = (
                    points[face[0]],
                    points[face[corner]],
                    points[face[corner + 1]],
                )
                u = tuple(b[axis] - a[axis] for axis in range(3))
                v = tuple(c[axis] - a[axis] for axis in range(3))
                cross = (
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                )
                total += 0.5 * sum(w * w for w in cross) ** 0.5
        return total

    def test_a_seam_with_a_t_junction_is_welded_not_capped(self):
        """Two patches meeting along one line must pair off, not get a lid.

        K2010's joints tear exactly like this: twenty-four boundary vertices
        lying on one straight segment 0.02 to 0.10 mm long, enclosing no area,
        so Newell's normal is 0.0 and a cap covers nothing. The boundary count
        held at 62 through two rounds of capping because there was no hole to
        cap -- one side of the seam simply had a vertex in the middle of the
        other side's edge.
        """
        # A square split into two halves along x=0. The left half meets the
        # seam with ONE edge; the right half meets it with two, because it has
        # an extra vertex at the midpoint.
        points = [
            (-1.0, -1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        ]
        counts = (4, 5)
        indices = (0, 1, 2, 3, 1, 4, 5, 2, 6)
        self.assertEqual(self._boundary_edges(counts, indices), 9)
        welded_counts, welded_indices, inserted = weld_seams(points, counts, indices)
        self.assertEqual(inserted, 1)
        # The square's outer edge is the only thing still open: six edges,
        # down from nine, and the three that went were the seam.
        self.assertEqual(self._boundary_edges(welded_counts, welded_indices), 6)
        self.assertEqual(len(points), 7)

    def test_a_collinear_rim_is_not_given_a_zero_area_lid(self):
        """A rim enclosing no area gets no cap rather than a lid over nothing.

        Fanning it would add zero-area faces, which read as a defect in every
        audit that follows and cover nothing at all.
        """
        points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        sealed = close_open_boundaries(points, (3,), (0, 1, 2))
        self.assertEqual(sealed[3], 0)
        self.assertEqual(len(sealed[0]), len(points))

    def test_a_rim_facing_either_way_still_seals(self):
        """The cap must oppose the rim whichever way the rim points.

        Ear clipping normalises to counter-clockwise in the plane it is handed,
        and each rim here is fitted its own plane -- so half of them come back
        facing the wrong way. On K2010 that read as face count rising while the
        boundary held at 62 and winding conflicts climbed: a hole, plus a
        defect where there had only been a hole.
        """
        points, counts, indices = self._open_tube()
        for flip in (False, True):
            if flip:
                # Same tube with the OTHER end open, so the rim's fitted
                # normal comes out reversed.
                mirrored = [(x, y, -z) for x, y, z in points]
                sealed = close_open_boundaries(mirrored, counts, indices)
            else:
                sealed = close_open_boundaries(points, counts, indices)
            self.assertEqual(
                self._boundary_edges(sealed[1], sealed[2]), 0, msg=f"{flip=}"
            )
            used = self._edge_use(sealed[1], sealed[2])
            self.assertFalse(
                [edge for edge, times in used.items() if times > 1],
                msg=f"cap wound the same way as the rim ({flip=})",
            )


class ChainLoopsTest(unittest.TestCase):
    """A rim that branches must still give up every one of its edges."""

    def test_a_figure_eight_gives_both_cycles(self):
        """Two cycles meeting at one vertex are two loops, not one and a loss.

        K2010's joints are the real case: their rims branch at fourteen to
        twenty vertices, and a chainer that abandons the path on revisiting a
        vertex kept 24 of 66 edges and silently dropped the rest, leaving the
        joint open with no error anywhere.
        """
        edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
        loops = _chain_loops(edges)
        self.assertEqual(len(loops), 2)
        self.assertEqual(sum(len(loop) for loop in loops), len(edges))
        self.assertEqual(
            sorted(tuple(sorted(loop)) for loop in loops), [(0, 1, 2), (0, 3, 4)]
        )

    def test_two_separate_rims_stay_separate(self):
        """Disjoint cycles are unaffected by the branching rule."""
        edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
        loops = _chain_loops(edges)
        self.assertEqual(len(loops), 2)
        self.assertEqual(sum(len(loop) for loop in loops), len(edges))


if __name__ == "__main__":
    unittest.main()
