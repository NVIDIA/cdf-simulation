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

"""Tests for ``scripts/crop/region.py`` (``crop_sdg``) — the
whole-frame bbox crop + component-bbox filter/translate step.

Runs end-to-end on tiny synthetic replicator-writer fixtures, matching the
structured-array schema in ``tests/unittests/imaging/crop/test_bbox_tight.py``.

Needs Pillow + OpenCV (and numpy); skips itself when either is absent,
matching the CI unit-test image which installs numpy only.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
    from PIL import Image
except ImportError:  # CI unit-test image installs numpy only
    cv2 = None
    Image = None

_SCRIPTS_CROP = Path(__file__).resolve().parents[4] / "scripts" / "crop"
if str(_SCRIPTS_CROP) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_CROP))

if cv2 is not None and Image is not None:
    from region import crop_sdg

# Matches the replicator BasicWriter bounding_box_2d_tight schema (same as
# test_bbox_tight.py) -- field names are what merge_to_components reads.
_BBOX_DTYPE = np.dtype(
    [
        ("semanticId", "<u4"),
        ("x_min", "<i4"),
        ("y_min", "<i4"),
        ("x_max", "<i4"),
        ("y_max", "<i4"),
        ("occlusionRatio", "<f4"),
    ]
)

_IMG_SIZE = 64
_PRIM_PREFIX = "/World/pcba/g/a/b"


@unittest.skipUnless(
    cv2 is not None and Image is not None,
    "cv2/Pillow not installed (CI installs numpy only)",
)
class CropSdgTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pcb_aoi_region_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = os.path.join(self.tmp, "src")
        self.dst = os.path.join(self.tmp, "dst")
        os.makedirs(self.src)

    def _write_bbox_frame(self, entries, labels, frame_idx="0000", kind="tight"):
        """``entries``: list of (component, semanticId, submesh_idx, bbox)."""
        bboxes = np.array(
            [(sid, *box, 0.0) for _comp, sid, _sub, box in entries],
            dtype=_BBOX_DTYPE,
        )
        np.save(
            os.path.join(self.src, f"bounding_box_2d_{kind}_{frame_idx}.npy"), bboxes
        )
        prim_paths = [
            f"{_PRIM_PREFIX}/{comp}/mesh_{sub}" for comp, _sid, sub, _box in entries
        ]
        with open(
            os.path.join(
                self.src, f"bounding_box_2d_{kind}_prim_paths_{frame_idx}.json"
            ),
            "w",
        ) as f:
            json.dump(prim_paths, f)
        with open(
            os.path.join(self.src, f"bounding_box_2d_{kind}_labels_{frame_idx}.json"),
            "w",
        ) as f:
            json.dump(labels, f)

    def _write_rgb(self, frame_idx="0000", value=100):
        arr = np.full((_IMG_SIZE, _IMG_SIZE, 3), value, dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(os.path.join(self.src, f"rgb_{frame_idx}.png"))

    def _write_semseg(self, frame_idx="0000"):
        arr = np.zeros((_IMG_SIZE, _IMG_SIZE, 4), dtype=np.uint8)
        Image.fromarray(arr, "RGBA").save(
            os.path.join(self.src, f"semantic_segmentation_{frame_idx}.png")
        )
        with open(
            os.path.join(self.src, f"semantic_segmentation_labels_{frame_idx}.json"),
            "w",
        ) as f:
            json.dump({}, f)

    def _entries(self):
        return [
            # compA: two submeshes merging to bbox (5,5,40,40), fully
            # inside crop bbox (5,5,50,50).
            ("compA", 1, 0, (5, 5, 20, 20)),
            ("compA", 1, 1, (15, 15, 40, 40)),
            # compB: bbox (30,10,60,20) -- x_max=60 exceeds crop x1=50.
            ("compB", 2, 0, (30, 10, 60, 20)),
        ]

    def _labels(self):
        return {"1": {"class": "capacitor"}, "2": {"class": "resistor"}}

    def test_outputs_bbox_only_skips_image_writes(self):
        self._write_bbox_frame(self._entries(), self._labels())
        self._write_rgb()
        self._write_semseg()

        stats = crop_sdg(
            self.src,
            self.dst,
            bbox=(5, 5, 50, 50),
            include_loose=False,
            outputs=("bbox",),
        )

        self.assertFalse(os.path.exists(os.path.join(self.dst, "rgb_0000.png")))
        self.assertFalse(
            os.path.exists(os.path.join(self.dst, "semantic_segmentation_0000.png"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.dst, "bounding_box_2d_tight_0000.npy"))
        )
        self.assertEqual(stats["outputs"], ["bbox"])

    def test_partially_outside_component_dropped_fully_inside_kept_and_translated(self):
        self._write_bbox_frame(self._entries(), self._labels())

        stats = crop_sdg(
            self.src,
            self.dst,
            bbox=(5, 5, 50, 50),
            include_loose=False,
            outputs=("bbox",),
        )

        self.assertEqual(stats["totals"], {"total": 2, "kept": 1, "dropped": 1})
        self.assertEqual(
            stats["per_class"]["capacitor"], {"total": 1, "kept": 1, "dropped": 0}
        )
        self.assertEqual(
            stats["per_class"]["resistor"], {"total": 1, "kept": 0, "dropped": 1}
        )

        kept = np.load(os.path.join(self.dst, "bounding_box_2d_tight_0000.npy"))
        self.assertEqual(len(kept), 1)
        row = kept[0]
        self.assertEqual(int(row["semanticId"]), 1)
        # translated into crop-local coords: (5,5,40,40) - (5,5) = (0,0,35,35)
        self.assertEqual(int(row["x_min"]), 0)
        self.assertEqual(int(row["y_min"]), 0)
        self.assertEqual(int(row["x_max"]), 35)
        self.assertEqual(int(row["y_max"]), 35)
        self.assertAlmostEqual(float(row["occlusionRatio"]), -1.0)

        prim_paths = json.loads(
            Path(
                os.path.join(self.dst, "bounding_box_2d_tight_prim_paths_0000.json")
            ).read_text()
        )
        self.assertEqual(prim_paths, [f"{_PRIM_PREFIX}/compA"])

    def test_unknown_output_raises_value_error(self):
        self._write_bbox_frame(self._entries(), self._labels())
        with self.assertRaises(ValueError):
            crop_sdg(self.src, self.dst, bbox=(5, 5, 50, 50), outputs=("bogus",))

    def test_rgb_output_is_cropped_when_included(self):
        self._write_bbox_frame(self._entries(), self._labels())
        self._write_rgb(value=77)

        crop_sdg(
            self.src,
            self.dst,
            bbox=(5, 5, 50, 50),
            include_loose=False,
            outputs=("rgb", "bbox"),
        )

        out_path = os.path.join(self.dst, "rgb_0000.png")
        self.assertTrue(os.path.isfile(out_path))
        with Image.open(out_path) as img:
            self.assertEqual(img.size, (45, 45))  # (50-5, 50-5)
            self.assertEqual(img.convert("RGB").getpixel((0, 0)), (77, 77, 77))


if __name__ == "__main__":
    unittest.main()
