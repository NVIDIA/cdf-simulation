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

"""Tests for closing ring-shaped segmentation labels into their openings."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:  # the CI unit-test image installs numpy only
    import numpy as np
except ImportError:
    np = None
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import scipy  # noqa: F401 -- closing openings labels connected components
except ImportError:
    scipy = None

from cdf.errors import ValidationError
from cdf.postprocess.opening_closure import close_openings

HAVE_DEPENDENCIES = None not in (np, Image, scipy)

BACKGROUND, UNLABELLED, HOLE, PIN = 0, 1, 2, 3
LABELS = {
    "0": {"class": "BACKGROUND"},
    "1": {"class": "UNLABELLED"},
    "2": {"class": "hole"},
    "3": {"class": "pin"},
}


def _ring(size=41, radius=14, thickness=1.6):
    """Return a mask that is a thin circular ring, as a barrel rim projects."""
    centre = size // 2
    ys, xs = np.ogrid[:size, :size]
    distance = np.hypot(ys - centre, xs - centre)
    return np.abs(distance - radius) <= thickness


def _write_mask(directory: Path, mask, name="mask.png"):
    """Write one stitched mask and its unified label table."""
    path = directory / name
    Image.fromarray(mask.astype(np.uint8), "L").save(path)
    path.with_name(f"{path.stem}_labels.json").write_text(
        json.dumps(LABELS),
        encoding="utf-8",
    )
    return path


@unittest.skipUnless(HAVE_DEPENDENCIES,
                     "NumPy, Pillow and SciPy are not all installed")
class CloseOpeningsTest(unittest.TestCase):
    """Verify that a rendered rim recovers the opening it encloses."""

    def test_fills_a_closed_ring(self):
        """The interior of a barrel rim becomes the same class."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.full((41, 41), UNLABELLED, np.uint8)
            mask[_ring()] = HOLE
            path = _write_mask(root, mask)

            report = close_openings(path, class_name="hole")

            result = np.asarray(Image.open(path))
            self.assertEqual(report["openings_closed"], 1)
            self.assertGreater(report["pixels_added"], 100)
            self.assertEqual(int(result[20, 20]), HOLE)

    def test_preserves_a_pin_inside_the_opening(self):
        """Only unlabelled pixels are claimed, so the pin keeps its class."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.full((41, 41), UNLABELLED, np.uint8)
            mask[_ring()] = HOLE
            mask[18:23, 18:23] = PIN
            path = _write_mask(root, mask)

            close_openings(path, class_name="hole")

            result = np.asarray(Image.open(path))
            self.assertEqual(int(result[20, 20]), PIN)
            self.assertEqual(int(result[20, 27]), HOLE)

    def test_leaves_an_open_rim_alone(self):
        """A rim broken by occlusion is not guessed at."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.full((41, 41), UNLABELLED, np.uint8)
            ring = _ring()
            ring[:, 20:] = False
            mask[ring] = HOLE
            path = _write_mask(root, mask)

            report = close_openings(path, class_name="hole")

            result = np.asarray(Image.open(path))
            self.assertEqual(report["openings_closed"], 0)
            self.assertEqual(int(result[20, 20]), UNLABELLED)

    def test_rejects_an_opening_larger_than_the_bound(self):
        """A ring too big to be a hole is left alone."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.full((81, 81), UNLABELLED, np.uint8)
            mask[_ring(size=81, radius=34)] = HOLE
            path = _write_mask(root, mask)

            report = close_openings(path, class_name="hole", max_diameter_px=20)

            self.assertEqual(report["openings_closed"], 0)

    def test_absent_class_is_a_no_op(self):
        """A frame with no such class is passed over."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_mask(root, np.full((41, 41), UNLABELLED, np.uint8))

            report = close_openings(path, class_name="hole")

            self.assertEqual(report["openings_closed"], 0)

    def test_requires_the_mask_to_exist(self):
        """A missing mask is a mistake, not a silent success."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValidationError):
                close_openings(Path(directory) / "mask.png", class_name="hole")

    def test_requires_the_label_sidecar(self):
        """Identifiers are per stitch, so the unified table is mandatory."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.zeros((8, 8), np.uint8), "L").save(path)
            with self.assertRaises(ValidationError):
                close_openings(path, class_name="hole")


if __name__ == "__main__":
    unittest.main()
