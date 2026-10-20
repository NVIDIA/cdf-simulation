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

"""Unit tests for the body-texture generator math (scripts/sdg/editor/
body_texture.py) — the pure-numpy half of the ``appearance.body_texture``
capability. No Kit/omni needed; PNG round-trip tests self-skip without PIL
(the CI image is numpy-only)."""

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EDITOR_DIR = _REPO_ROOT / "scripts" / "sdg" / "editor"


def _load():
    sys.path.insert(0, str(_EDITOR_DIR))
    try:
        return importlib.import_module("body_texture")
    finally:
        sys.path.remove(str(_EDITOR_DIR))


bt = _load()

_SIZE = 64  # small maps keep the suite fast; the math is size-agnostic


class PerlinFbmTest(unittest.TestCase):

    def test_perlin_zero_at_lattice_points(self):
        perm = np.random.default_rng(0).permutation(256).astype(np.int64)
        x, y = np.meshgrid(np.arange(5.0), np.arange(5.0))
        np.testing.assert_allclose(bt.perlin2d(x, y, perm), 0.0, atol=1e-12)

    def test_fbm_shape_range_and_determinism(self):
        a = bt.fbm(np.random.default_rng(42), 3.0, 5, _SIZE)
        b = bt.fbm(np.random.default_rng(42), 3.0, 5, _SIZE)
        c = bt.fbm(np.random.default_rng(43), 3.0, 5, _SIZE)
        self.assertEqual(a.shape, (_SIZE, _SIZE))
        self.assertGreaterEqual(a.min(), 0.0)
        self.assertLessEqual(a.max(), 1.0)
        np.testing.assert_array_equal(a, b)       # same seed -> same field
        self.assertTrue((a != c).any())           # different seed differs

    def test_fbm_not_axis_aligned(self):
        # the per-octave rotation must actually rotate: the field must not
        # be symmetric under transpose (a plaid sin/cos stack nearly is)
        a = bt.fbm(np.random.default_rng(7), 4.0, 4, _SIZE)
        self.assertGreater(np.abs(a - a.T).max(), 1e-3)

    def test_smoothstep_endpoints_and_monotonic(self):
        x = np.linspace(-1.0, 2.0, 301)
        y = bt.smoothstep(x, 0.0, 1.0)
        self.assertEqual(y[0], 0.0)
        self.assertEqual(y[-1], 1.0)
        self.assertAlmostEqual(float(bt.smoothstep(np.array(0.5), 0, 1)),
                               0.5)
        self.assertTrue((np.diff(y) >= 0).all())

    def test_sample_scalar_and_range(self):
        rng = np.random.default_rng(0)
        self.assertEqual(bt._sample(rng, 2.5), 2.5)
        v = bt._sample(rng, [1.0, 2.0])
        self.assertTrue(1.0 <= v <= 2.0)


class MapsTest(unittest.TestCase):

    def test_normal_map_shape_range_and_unit_length(self):
        nrm = bt.normal_map(np.random.default_rng(5), size=_SIZE,
                            freq=4.0, octaves=4, relief_strength=0.75)
        self.assertEqual(nrm.shape, (_SIZE, _SIZE, 3))
        self.assertGreaterEqual(nrm.min(), 0.0)
        self.assertLessEqual(nrm.max(), 1.0)
        decoded = nrm * 2.0 - 1.0                 # unit vectors, z up
        length = np.linalg.norm(decoded, axis=2)
        np.testing.assert_allclose(length, 1.0, atol=1e-6)
        self.assertGreater(decoded[:, :, 2].min(), 0.0)

    def test_normal_map_relief_strength_scales_slope(self):
        flat = bt.normal_map(np.random.default_rng(5), size=_SIZE,
                             relief_strength=0.05)
        rough = bt.normal_map(np.random.default_rng(5), size=_SIZE,
                              relief_strength=1.0)
        tilt = lambda m: np.abs(m[:, :, :2] * 2.0 - 1.0).mean()  # noqa: E731
        self.assertGreater(tilt(rough), tilt(flat) * 2)

    def test_albedo_map_shape_range_and_determinism(self):
        base = (0.35, 0.2, 0.15)
        a = bt.albedo_map(np.random.default_rng(9), base, size=_SIZE)
        b = bt.albedo_map(np.random.default_rng(9), base, size=_SIZE)
        self.assertEqual(a.shape, (_SIZE, _SIZE, 3))
        self.assertGreaterEqual(a.min(), 0.0)
        self.assertLessEqual(a.max(), 1.0)
        np.testing.assert_array_equal(a, b)
        # the layers must actually mark the surface
        self.assertTrue((np.abs(a - np.asarray(base)).max(axis=2)
                         > 0.05).any())

    def test_albedo_map_layers_can_be_disabled(self):
        # zero strengths/blends -> pure base colour (layer plumbing works)
        base = (0.4, 0.3, 0.2)
        a = bt.albedo_map(np.random.default_rng(3), base, size=_SIZE,
                          blotch_strength=0.0, ring_strength=0.0,
                          tint_blend=0.0)
        np.testing.assert_allclose(a, np.tile(base, (_SIZE, _SIZE, 1)),
                                   atol=1e-9)


class BoxProjectUvsTest(unittest.TestCase):
    # a 2x1x0.5 box: top/bottom (normal Z), front/back (normal Y),
    # left/right (normal X) quads — CAD-side-wall shaped on purpose
    _PTS = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],
                     [0, 0, .5], [2, 0, .5], [2, 1, .5], [0, 1, .5]],
                    dtype=float)
    _FACES = {  # face name -> vertex ids (outward winding not required)
        "bottom": [0, 1, 2, 3], "top": [4, 5, 6, 7],
        "front": [0, 1, 5, 4], "right": [1, 2, 6, 5],
    }

    def _uvs_for(self, names):
        idx = np.concatenate([self._FACES[n] for n in names])
        cnt = np.full(len(names), 4, dtype=np.int64)
        return bt.box_project_uvs(self._PTS, idx, cnt)

    def test_dominant_axis_selection(self):
        uvs = self._uvs_for(["top", "front", "right"])
        scale = 2.0  # ONE global scale: the largest bbox extent
        # top: normal Z -> (u, v) = (x, y) / scale
        np.testing.assert_allclose(
            uvs[0:4], self._PTS[self._FACES["top"]][:, [0, 1]] / scale)
        # front: normal Y -> (u, v) = (x, z) / scale
        np.testing.assert_allclose(
            uvs[4:8], self._PTS[self._FACES["front"]][:, [0, 2]] / scale)
        # right: normal X -> (u, v) = (y, z) / scale — a planar (x, y)
        # projection would give this wall a degenerate 1-D gradient
        np.testing.assert_allclose(
            uvs[8:12], self._PTS[self._FACES["right"]][:, [1, 2]] / scale)

    def test_facevarying_length_and_range(self):
        uvs = self._uvs_for(["bottom", "top", "front", "right"])
        self.assertEqual(uvs.shape, (16, 2))     # sum(face counts), 2
        self.assertGreaterEqual(uvs.min(), 0.0)
        self.assertLessEqual(uvs.max(), 1.0)

    def test_side_walls_not_degenerate(self):
        uvs = self._uvs_for(["right"])
        self.assertGreater(np.ptp(uvs[:, 0]), 0.0)
        self.assertGreater(np.ptp(uvs[:, 1]), 0.0)


class GenerateMapsTest(unittest.TestCase):

    def setUp(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("PIL not installed (CI image is numpy-only)")

    def test_writes_deterministic_pair(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            n1, a1 = bt.generate_maps(tmp, "x", [7, 0, 0], (0.3, 0.2, 0.1),
                                      size=_SIZE)
            n2, a2 = bt.generate_maps(tmp, "y", [7, 0, 0], (0.3, 0.2, 0.1),
                                      size=_SIZE)
            self.assertTrue(n1.endswith("x_n.png"))
            self.assertTrue(a1.endswith("x_c.png"))
            self.assertEqual(Path(n1).read_bytes(), Path(n2).read_bytes())
            self.assertEqual(Path(a1).read_bytes(), Path(a2).read_bytes())


if __name__ == "__main__":
    unittest.main()
