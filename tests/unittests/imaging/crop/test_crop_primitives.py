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

"""Tests for ``scripts/crop/crop_primitives.py``.

CI-safe: numpy-only (no PIL / cv2 / omni). Focus: ``mask_coverage_fraction``
must accept every semantic-mask shape a caller can feed it — 4-channel RGBA
(the in-repo replicator writer), 3-channel RGB, and true single-channel 2-D
masks (an L/P-mode PNG via ``np.array(Image.open(...))``), including the
pathological 2-D array whose WIDTH happens to be 4 (previously took the RGBA
branch and raised IndexError).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SCRIPTS_CROP = Path(__file__).resolve().parents[4] / "scripts" / "crop"
if str(_SCRIPTS_CROP) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_CROP))

from crop_primitives import mask_coverage_fraction  # noqa: E402


class MaskCoverageFractionTest(unittest.TestCase):

    def test_2d_single_channel_mask(self):
        # (h, w) uint8 label mask — colorize_semantic_segmentation=False
        arr = np.zeros((8, 10), dtype=np.uint8)
        arr[:4, :] = 7  # 40 of 80 pixels labelled
        self.assertAlmostEqual(mask_coverage_fraction(arr), 0.5)

    def test_2d_mask_with_width_four_is_not_treated_as_rgba(self):
        # shape[-1] == 4 but ndim == 2: previously raised IndexError
        arr = np.zeros((6, 4), dtype=np.uint8)
        arr[0, :] = 1  # 4 of 24 pixels labelled
        self.assertAlmostEqual(mask_coverage_fraction(arr), 4.0 / 24.0)

    def test_rgba_counts_alpha_channel(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[:, :, 0] = 255          # colour without alpha does not count
        arr[:2, :, 3] = 255         # 8 of 16 pixels alpha-on
        self.assertAlmostEqual(mask_coverage_fraction(arr), 0.5)

    def test_rgb_counts_channel_max(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[0, 0, 1] = 3
        self.assertAlmostEqual(mask_coverage_fraction(arr), 1.0 / 16.0)

    def test_empty_arrays_return_zero(self):
        self.assertEqual(mask_coverage_fraction(np.zeros((0, 5))), 0.0)
        self.assertEqual(mask_coverage_fraction(np.zeros((5, 0))), 0.0)
        self.assertEqual(mask_coverage_fraction(np.zeros((0, 0, 4))), 0.0)

    def test_all_empty_2d_mask_is_zero(self):
        self.assertEqual(mask_coverage_fraction(np.zeros((8, 8),
                                                         dtype=np.uint8)), 0.0)


if __name__ == "__main__":
    unittest.main()
