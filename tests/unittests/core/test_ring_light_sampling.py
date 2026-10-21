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

"""Unit tests for ``sample_ring_lighting`` in ``pcb_aoi/lighting.py``.

Photometric randomization of the AOI ring rig. Pure math, no Isaac, so the draw
order and the bounds can be pinned without a stage.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_STANDALONE = Path(__file__).resolve().parents[3] / "scripts" / "sdg" / "standalone"
if str(_STANDALONE) not in sys.path:
    sys.path.insert(0, str(_STANDALONE))

from pcb_aoi.lighting import sample_ring_lighting


class SampleRingLightingTest(unittest.TestCase):
    """Per-scene AOI ring-light sampler (photometric domain randomization)."""

    _BASE = {"Inner_Red": 2500.0, "Middle_Green": 1000.0, "Outer_Blue": 1200.0}

    def _sample(self, seed, **kw):
        params = dict(exposure_range=(0.15, 0.4), cone_angle_range=(100.0, 140.0),
                      cone_softness_range=(0.5, 1.0), intensity_jitter=0.3)
        params.update(kw)
        return sample_ring_lighting(np.random.default_rng(seed), self._BASE, **params)

    def test_deterministic_for_seed(self):
        self.assertEqual(self._sample(7), self._sample(7))

    def test_different_seeds_differ(self):
        self.assertNotEqual(self._sample(1)["exposure"], self._sample(2)["exposure"])

    def test_ranges_respected(self):
        out = self._sample(3)
        self.assertTrue(0.15 <= out["exposure"] <= 0.4)
        self.assertTrue(100.0 <= out["cone_angle"] <= 140.0)
        self.assertTrue(0.5 <= out["cone_softness"] <= 1.0)
        self.assertEqual(set(out["layer_intensity"]), set(self._BASE))

    def test_intensity_within_jitter_of_base_times_exposure(self):
        out = self._sample(11, intensity_jitter=0.3)
        exp = out["exposure"]
        for layer, base in self._BASE.items():
            got = out["layer_intensity"][layer]
            self.assertGreaterEqual(got, base * exp * 0.7 - 1e-6)
            self.assertLessEqual(got, base * exp * 1.3 + 1e-6)

    def test_zero_jitter_is_base_times_exposure(self):
        out = self._sample(5, intensity_jitter=0.0)
        for layer, base in self._BASE.items():
            self.assertAlmostEqual(out["layer_intensity"][layer], base * out["exposure"])

    def test_degenerate_range_pins_value(self):
        out = self._sample(9, exposure_range=(0.25, 0.25))
        self.assertAlmostEqual(out["exposure"], 0.25)

    def test_intensity_never_negative(self):
        out = self._sample(2, intensity_jitter=5.0)  # absurd jitter clamps at 0
        self.assertTrue(all(v >= 0.0 for v in out["layer_intensity"].values()))


if __name__ == "__main__":
    unittest.main()
