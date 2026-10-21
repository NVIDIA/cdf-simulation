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

"""Crease-aware vertex normals: split at creases, smooth on curvature.

numpy-only on purpose -- this suite must run in the dependency-starved CI
core-tests job, exactly like ``test_planar_geometry``.
"""

import math
import unittest

import numpy as np

from pcb_aoi.solder_generation.mesh_normals import (
    crease_aware_vertex_normals,
    solder_mesh_shading,
)


def _cube():
    P = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
    )
    quads = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),  # x = 0, x = 1
        (0, 4, 5, 1),
        (2, 3, 7, 6),  # y = 0, y = 1
        (0, 2, 6, 4),
        (1, 5, 7, 3),  # z = 0, z = 1
    ]
    T = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    return P, T


class CreaseAwareNormalsTest(unittest.TestCase):
    def test_cube_corners_split_into_axis_aligned_normals(self):
        P, T = _cube()
        P2, T2, N = crease_aware_vertex_normals(P, T)

        # Every corner sits on three orthogonal planes: 8 -> 24 vertices.
        self.assertEqual(len(P2), 24)
        self.assertEqual(len(N), 24)
        # Geometry is unchanged: every triangle's corner positions survive.
        np.testing.assert_allclose(P2[T2], P[T])
        # Each vertex normal is unit and axis-aligned (a flat face's normal,
        # never a smoothed corner blend).
        np.testing.assert_allclose(np.linalg.norm(N, axis=1), 1.0)
        self.assertTrue(np.all(np.isclose(np.abs(N), 0.0) | np.isclose(np.abs(N), 1.0)))
        # And matches the face normal at every referenced corner.
        for tri in T2:
            a, b, c = P2[tri]
            face = np.cross(b - a, c - a)
            face /= np.linalg.norm(face)
            for v in tri:
                np.testing.assert_allclose(N[v], face, atol=1e-12)

    def test_smooth_arc_stays_unsplit_with_radial_normals(self):
        # A 90-degree cylinder strip, 16 segments: adjacent facets differ by
        # ~5.6 degrees, far below the crease threshold.
        segs = 16
        ang = np.linspace(0.0, math.pi / 2.0, segs + 1)
        ring = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        P = np.array([[x, y, z] for x, y in ring for z in (0.0, 1.0)])
        T = np.array(
            [
                t
                for s in range(segs)
                for t in (
                    (2 * s, 2 * s + 2, 2 * s + 1),
                    (2 * s + 1, 2 * s + 2, 2 * s + 3),
                )
            ]
        )
        P2, T2, N = crease_aware_vertex_normals(P, T)

        self.assertEqual(len(P2), len(P))  # no splits
        np.testing.assert_array_equal(T2, T)
        radial = np.array([[x, y, 0.0] for x, y, _z in P])
        agreement = np.einsum("ij,ij->i", N, radial)
        # Averaged facet normals sit within a couple of degrees of radial
        # (sign tells us they also all point outward).
        self.assertTrue(np.all(np.abs(agreement) > math.cos(math.radians(4.0))))
        self.assertTrue(np.all(agreement > 0) or np.all(agreement < 0))

    def test_degenerate_triangle_neither_crashes_nor_votes(self):
        P, T = _cube()
        P = np.vstack([P, P[0]])  # duplicate of vertex 0
        T = np.vstack([T, [(0, 1, 8)]])  # zero-area sliver on edge 0-1
        _P2, _T2, N = crease_aware_vertex_normals(P, T)
        self.assertTrue(np.all(np.isfinite(N)))
        np.testing.assert_allclose(np.linalg.norm(N, axis=1), 1.0)

    def test_solder_mesh_shading_matches_soldermesh_fields(self):
        P, T = _cube()
        points, counts, indices, normals = solder_mesh_shading(P, T)
        self.assertEqual(set(counts), {3})
        self.assertEqual(len(indices), 3 * len(counts))
        self.assertEqual(len(normals), len(points))
        self.assertLess(max(indices), len(points))
        self.assertTrue(all(len(n) == 3 for n in normals))


if __name__ == "__main__":
    unittest.main()
