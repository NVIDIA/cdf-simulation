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

"""The SDF module behind the experimental collision energy.

Rewritten against ``pcb_aoi.solder_evolver.sdf`` (the old test_sdf exercised
the deleted analytic-coupon path). Everything here runs CPU-only: the mesh
grid uses the trimesh backend explicitly, so a machine without warp or a GPU
still tests the same numbers the collider is built from.
"""

import os
import struct
import sys
import tempfile
import unittest

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    import numpy as np
    from pcb_aoi.solder_evolver import sdf
except (Exception, SystemExit) as _exc:                                 # noqa: BLE001
    raise unittest.SkipTest(f"solder_evolver deps unavailable: {_exc}")

try:
    import trimesh                                                      # noqa: F401
    _HAVE_TRIMESH = True
except Exception:                                                       # noqa: BLE001
    _HAVE_TRIMESH = False


def _cube(half=0.5):
    """A unit-ish cube mesh centred at the origin: 8 verts, 12 tris."""
    h = half
    V = np.array([(x, y, z) for z in (-h, h) for y in (-h, h) for x in (-h, h)],
                 float)
    F = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
                  [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
                  [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 5, 7]], int)
    return V, F


class TestAnalyticSdfs(unittest.TestCase):
    def test_sphere_sign_and_distance(self):
        f = sdf.sphere_sdf((0.0, 0.0, 0.0), 1.0)
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        d = f(pts)
        self.assertAlmostEqual(d[0], -1.0, places=12)   # centre: inside by r
        self.assertAlmostEqual(d[1], 0.0, places=12)    # on the surface
        self.assertAlmostEqual(d[2], 1.0, places=12)    # outside by r

    def test_box_matches_geometry(self):
        f = sdf.box_sdf((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
        pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
        d = f(pts)
        self.assertLess(d[0], 0.0)
        self.assertAlmostEqual(d[1], 0.0, places=12)
        self.assertAlmostEqual(d[2], 1.0, places=12)


class TestGridFile(unittest.TestCase):
    def test_write_grid_layout(self):
        """The .bin layout is a contract with the Evolver-side reader:
        MAGIC + version, int32 dims, float64 lo/hi corners, then the values
        transposed so x runs fastest in the file."""
        g = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
        lo, hi = np.zeros(3), np.ones(3)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "g.bin")
            sdf.write_grid(p, g, lo, hi)
            with open(p, "rb") as fh:
                raw = fh.read()
        off = len(sdf.MAGIC)
        self.assertEqual(raw[:off], sdf.MAGIC)
        (version,) = struct.unpack_from("<i", raw, off)
        self.assertEqual(version, sdf.VERSION)
        dims = struct.unpack_from("<3i", raw, off + 4)
        self.assertEqual(dims, (4, 5, 6))
        bounds = struct.unpack_from("<6d", raw, off + 16)
        self.assertEqual(bounds, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
        vals = np.frombuffer(raw[off + 16 + 48:], dtype="<f8")
        self.assertEqual(len(vals), 120)
        self.assertTrue(np.allclose(vals, g.transpose(2, 1, 0).ravel()))
        with self.assertRaises(ValueError):     # < 4 samples on an axis
            sdf.write_grid(os.devnull, np.zeros((2, 4, 4)), lo, hi)


@unittest.skipUnless(_HAVE_TRIMESH, "trimesh not installed")
class TestMeshGrid(unittest.TestCase):
    def test_cube_grid_matches_analytic(self):
        """mesh_sdf_grid on a cube mesh agrees with the analytic box SDF to
        within a grid cell -- the property the keep-out penalty rests on."""
        V, F = _cube(0.5)
        lo, hi = np.full(3, -1.0), np.full(3, 1.0)
        n = (33, 33, 33)
        g = sdf.mesh_sdf_grid(V, F, lo, hi, n, backend="trimesh")
        ref = sdf.sample(sdf.box_sdf((0, 0, 0), (0.5, 0.5, 0.5)), lo, hi, n)
        cell = float(np.max((hi - lo) / (np.array(n) - 1)))
        self.assertLess(float(np.abs(g - ref).max()), cell)
        # sign structure: centre inside, corner outside
        self.assertLess(g[16, 16, 16], 0.0)
        self.assertGreater(g[0, 0, 0], 0.0)

    def test_bake_keepout_floor(self):
        """floor_z folds the board plane into the keep-out: everything below
        the floor must read as inside (negative), whatever the mesh says."""
        V, F = _cube(0.2)
        lo, hi = np.full(3, -1.0), np.full(3, 1.0)
        with tempfile.TemporaryDirectory() as td:
            g = sdf.bake_keepout_sdf(os.path.join(td, "c.bin"), V, F, lo, hi,
                                     res=(17, 17, 17), floor_z=0.0)
        zz = np.linspace(lo[2], hi[2], 17)
        below = g[:, :, zz < 0.0]
        self.assertLess(float(below.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
