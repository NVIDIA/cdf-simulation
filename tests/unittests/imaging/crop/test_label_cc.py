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

"""Tests for ``scripts/crop/label_cc.py`` (``crop_rois_by_label`` /
``crop_bridge_pairs``) — the anchor-on-target labelled-connected-component
image crop and its emitted dataset content.

Runs end-to-end on tiny synthetic semantic-mask fixtures generated in
tmpdirs, mirroring ``tests/unittests/imaging/crop/test_bbox_tight.py``'s pattern
for the sibling crop algorithm.

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
    from label_cc import crop_bridge_pairs, crop_rois_by_label


def _rgba_key(rgba):
    return "({}, {}, {}, {})".format(*(int(value) for value in rgba))


def _write_frame(aligned_dir, frame_idx, size, regions, labels):
    """Write a synthetic ``semantic_segmentation_<idx>.png`` + labels JSON.

    ``regions``: list of ``(rgba, (x1, y1, x2, y2))`` painted opaque over a
    zeroed (unlabelled) background, half-open box convention.
    """
    os.makedirs(aligned_dir, exist_ok=True)
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    for rgba, (x1, y1, x2, y2) in regions:
        arr[y1:y2, x1:x2] = rgba
    Image.fromarray(arr, "RGBA").save(
        os.path.join(aligned_dir, f"semantic_segmentation_{frame_idx}.png")
    )
    with open(
        os.path.join(aligned_dir, f"semantic_segmentation_labels_{frame_idx}.json"),
        "w",
    ) as f:
        json.dump(labels, f)


def _write_rgb(path, size, value=128):
    arr = np.full((size, size, 3), value, dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


_CAP_RGBA = (0, 255, 0, 255)
_PAD_RGBA = (0, 0, 255, 255)
_SILK_RGBA = (10, 10, 10, 255)


@unittest.skipUnless(
    cv2 is not None and Image is not None,
    "cv2/Pillow not installed (CI installs numpy only)",
)
class CropRoisByLabelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pcb_aoi_label_cc_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.aligned_dir = os.path.join(self.tmp, "aligned")
        self.output_dir = os.path.join(self.tmp, "crop")

    def _write_ref_real(self, size):
        ref = os.path.join(self.tmp, "ref_crop.png")
        real = os.path.join(self.tmp, "aligned_crop.png")
        _write_rgb(ref, size, value=200)
        _write_rgb(real, size, value=50)
        return ref, real

    def test_anchor_pulls_adjacent_support_color_into_one_roi(self):
        size = 60
        _write_frame(
            self.aligned_dir,
            "0000",
            size,
            regions=[
                (_CAP_RGBA, (10, 10, 30, 30)),  # anchor 20x20 = 400px
                (_PAD_RGBA, (30, 10, 40, 30)),  # support 10x20 = 200px, touching
            ],
            labels={
                _rgba_key(_CAP_RGBA): {"class": "capacitor"},
                _rgba_key(_PAD_RGBA): {"class": "pad"},
            },
        )
        ref, real = self._write_ref_real(size)

        stats = crop_rois_by_label(
            self.aligned_dir,
            ref,
            real,
            self.output_dir,
            target_classes=["capacitor"],
            morph_kernel=0,
            min_area=1,
            offset=2,
            edge_skip=True,
            min_coverage=0.0,
        )

        self.assertEqual(stats["n_emitted"], 1)
        roi = stats["rois"][0]
        # merged bbox includes the non-anchor support pixels
        self.assertEqual(roi["bbox_inner"], [10, 10, 39, 29])
        self.assertEqual(roi["area_pixels"], 400 + 200)
        self.assertEqual(roi["dominant_class"], "capacitor")
        self.assertTrue(
            os.path.isfile(os.path.join(self.output_dir, "normal_img", "0001.png"))
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.output_dir, "cad_mask", "0001_cad_mask.png")
            )
        )

    def test_labelled_cc_without_anchor_pixel_is_dropped(self):
        size = 40
        _write_frame(
            self.aligned_dir,
            "0000",
            size,
            regions=[(_SILK_RGBA, (10, 10, 30, 30))],
            labels={
                # capacitor is a known target color but never painted, so
                # this CC (silkscreen only) has zero anchor pixels.
                _rgba_key(_CAP_RGBA): {"class": "capacitor"},
                _rgba_key(_SILK_RGBA): {"class": "silkscreen"},
            },
        )
        ref, real = self._write_ref_real(size)

        stats = crop_rois_by_label(
            self.aligned_dir,
            ref,
            real,
            self.output_dir,
            target_classes=["capacitor"],
            morph_kernel=0,
            min_area=1,
            offset=2,
            edge_skip=False,
            min_coverage=0.0,
        )

        self.assertEqual(stats["n_emitted"], 0)
        self.assertEqual(stats["n_components_total"], 1)
        self.assertEqual(stats["skipped_no_target"], 1)
        self.assertEqual(stats["rois"], [])

    def test_min_area_filter_excludes_small_cc(self):
        size = 40
        _write_frame(
            self.aligned_dir,
            "0000",
            size,
            regions=[(_CAP_RGBA, (10, 10, 13, 13))],  # 3x3 = 9px
            labels={_rgba_key(_CAP_RGBA): {"class": "capacitor"}},
        )
        ref, real = self._write_ref_real(size)
        common = {
            "target_classes": ["capacitor"],
            "morph_kernel": 0,
            "offset": 1,
            "edge_skip": False,
            "min_coverage": 0.0,
        }

        excluded = crop_rois_by_label(
            self.aligned_dir,
            ref,
            real,
            os.path.join(self.tmp, "crop_excluded"),
            min_area=50,
            **common,
        )
        self.assertEqual(excluded["n_emitted"], 0)
        self.assertEqual(excluded["skipped_min_area"], 1)

        included = crop_rois_by_label(
            self.aligned_dir,
            ref,
            real,
            os.path.join(self.tmp, "crop_included"),
            min_area=1,
            **common,
        )
        self.assertEqual(included["n_emitted"], 1)
        self.assertEqual(included["rois"][0]["area_pixels"], 9)

    def test_max_area_filter_excludes_large_cc(self):
        size = 60
        _write_frame(
            self.aligned_dir,
            "0000",
            size,
            regions=[(_CAP_RGBA, (5, 5, 45, 45))],  # 40x40 = 1600px
            labels={_rgba_key(_CAP_RGBA): {"class": "capacitor"}},
        )
        ref, real = self._write_ref_real(size)
        common = {
            "target_classes": ["capacitor"],
            "morph_kernel": 0,
            "min_area": 1,
            "offset": 1,
            "edge_skip": False,
            "min_coverage": 0.0,
        }

        excluded = crop_rois_by_label(
            self.aligned_dir,
            ref,
            real,
            os.path.join(self.tmp, "crop_excluded"),
            max_area=100,
            **common,
        )
        self.assertEqual(excluded["n_emitted"], 0)
        self.assertEqual(excluded["skipped_max_area"], 1)

        included = crop_rois_by_label(
            self.aligned_dir,
            ref,
            real,
            os.path.join(self.tmp, "crop_included"),
            max_area=None,
            **common,
        )
        self.assertEqual(included["n_emitted"], 1)

    def test_edge_skip_filter_excludes_edge_touching_cc(self):
        size = 40
        _write_frame(
            self.aligned_dir,
            "0000",
            size,
            regions=[(_CAP_RGBA, (0, 10, 10, 20))],  # touches left edge
            labels={_rgba_key(_CAP_RGBA): {"class": "capacitor"}},
        )
        ref, real = self._write_ref_real(size)
        common = {
            "target_classes": ["capacitor"],
            "morph_kernel": 0,
            "min_area": 1,
            "offset": 1,
            "min_coverage": 0.0,
        }

        excluded = crop_rois_by_label(
            self.aligned_dir,
            ref,
            real,
            os.path.join(self.tmp, "crop_excluded"),
            edge_skip=True,
            **common,
        )
        self.assertEqual(excluded["n_emitted"], 0)
        self.assertEqual(excluded["skipped_edge"], 1)

        included = crop_rois_by_label(
            self.aligned_dir,
            ref,
            real,
            os.path.join(self.tmp, "crop_included"),
            edge_skip=False,
            **common,
        )
        self.assertEqual(included["n_emitted"], 1)

    def test_min_coverage_filter_excludes_sparse_crop(self):
        size = 100
        _write_frame(
            self.aligned_dir,
            "0000",
            size,
            regions=[(_CAP_RGBA, (40, 40, 50, 50))],  # 10x10 = 100px
            labels={_rgba_key(_CAP_RGBA): {"class": "capacitor"}},
        )
        ref, real = self._write_ref_real(size)
        # large offset dilutes coverage of the padded crop box
        common = {
            "target_classes": ["capacitor"],
            "morph_kernel": 0,
            "min_area": 1,
            "offset": 30,
            "edge_skip": False,
        }

        excluded = crop_rois_by_label(
            self.aligned_dir,
            ref,
            real,
            os.path.join(self.tmp, "crop_excluded"),
            min_coverage=0.2,
            **common,
        )
        self.assertEqual(excluded["n_emitted"], 0)
        self.assertEqual(excluded["skipped_low_coverage"], 1)

        included = crop_rois_by_label(
            self.aligned_dir,
            ref,
            real,
            os.path.join(self.tmp, "crop_included"),
            min_coverage=0.0,
            **common,
        )
        self.assertEqual(included["n_emitted"], 1)

    def test_class_dirs_single_frame_shares_counter_across_calls(self):
        size = 40
        _write_frame(
            self.aligned_dir,
            "0000",
            size,
            regions=[(_CAP_RGBA, (5, 5, 15, 15))],
            labels={_rgba_key(_CAP_RGBA): {"class": "capacitor"}},
        )
        _write_frame(
            self.aligned_dir,
            "0001",
            size,
            regions=[(_CAP_RGBA, (5, 5, 15, 15))],
            labels={_rgba_key(_CAP_RGBA): {"class": "capacitor"}},
        )
        ref, real = self._write_ref_real(size)
        from collections import Counter

        shared_counter = Counter()
        common = {
            "target_classes": ["capacitor"],
            "morph_kernel": 0,
            "min_area": 1,
            "offset": 1,
            "edge_skip": False,
            "min_coverage": 0.0,
            "class_dirs": {"capacitor": "passive"},
            "multi_frame": False,
            "class_emit_counter": shared_counter,
        }

        s1 = crop_rois_by_label(
            self.aligned_dir, ref, real, self.output_dir, frame_idx="0000", **common
        )
        s2 = crop_rois_by_label(
            self.aligned_dir, ref, real, self.output_dir, frame_idx="0001", **common
        )

        self.assertEqual(s1["rois"][0]["name"], "passive/0001")
        self.assertEqual(s2["rois"][0]["name"], "passive/0002")
        # flat layout: both frames land in the same normal_img/ dir
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.output_dir, "passive", "normal_img", "0001.png")
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.output_dir, "passive", "normal_img", "0002.png")
            )
        )

    def test_class_dirs_multi_frame_resets_counter_per_cell(self):
        size = 40
        for frame_idx in ("x0_y0", "x0_y1"):
            _write_frame(
                self.aligned_dir,
                frame_idx,
                size,
                regions=[(_CAP_RGBA, (5, 5, 15, 15))],
                labels={_rgba_key(_CAP_RGBA): {"class": "capacitor"}},
            )
        ref, real = self._write_ref_real(size)
        common = {
            "target_classes": ["capacitor"],
            "morph_kernel": 0,
            "min_area": 1,
            "offset": 1,
            "edge_skip": False,
            "min_coverage": 0.0,
            "class_dirs": {"capacitor": "passive"},
            "multi_frame": True,
        }

        s1 = crop_rois_by_label(
            self.aligned_dir, ref, real, self.output_dir, frame_idx="x0_y0", **common
        )
        s2 = crop_rois_by_label(
            self.aligned_dir, ref, real, self.output_dir, frame_idx="x0_y1", **common
        )

        # per-cell layout: each cell restarts NNNN at 0001, in its own subdir
        self.assertEqual(s1["rois"][0]["name"], "passive/x0_y0/0001")
        self.assertEqual(s2["rois"][0]["name"], "passive/x0_y1/0001")
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.output_dir, "passive", "x0_y0", "normal_img", "0001.png"
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.output_dir, "passive", "x0_y1", "normal_img", "0001.png"
                )
            )
        )

    def test_missing_seg_image_raises_file_not_found(self):
        os.makedirs(self.aligned_dir)
        with open(
            os.path.join(self.aligned_dir, "semantic_segmentation_labels_0000.json"),
            "w",
        ) as f:
            json.dump({}, f)
        ref, real = self._write_ref_real(40)
        with self.assertRaises(FileNotFoundError):
            crop_rois_by_label(
                self.aligned_dir,
                ref,
                real,
                self.output_dir,
                target_classes=["capacitor"],
            )

    def test_missing_labels_sidecar_raises_file_not_found(self):
        size = 40
        os.makedirs(self.aligned_dir)
        _write_rgb(
            os.path.join(self.aligned_dir, "semantic_segmentation_0000.png"), size
        )
        ref, real = self._write_ref_real(size)
        with self.assertRaises(FileNotFoundError):
            crop_rois_by_label(
                self.aligned_dir,
                ref,
                real,
                self.output_dir,
                target_classes=["capacitor"],
            )


@unittest.skipUnless(
    cv2 is not None and Image is not None,
    "cv2/Pillow not installed (CI installs numpy only)",
)
class CropBridgePairsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pcb_aoi_bridge_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.output_dir = os.path.join(self.tmp, "bridge")
        self.size = 60
        self.ref = os.path.join(self.tmp, "ref.png")
        self.real = os.path.join(self.tmp, "real.png")
        self.seg = os.path.join(self.tmp, "seg.png")
        _write_rgb(self.ref, self.size, value=200)
        _write_rgb(self.real, self.size, value=50)
        arr = np.zeros((self.size, self.size, 4), dtype=np.uint8)
        Image.fromarray(arr, "RGBA").save(self.seg)

    def test_two_rois_within_bridge_distance_merge(self):
        rois = [
            {
                "name": "0001",
                "bbox_inner": [10, 10, 20, 20],
                "dominant_class": "capacitor",
            },
            {
                "name": "0002",
                "bbox_inner": [25, 10, 35, 20],
                "dominant_class": "capacitor",
            },  # edge-to-edge gap = 5
        ]
        stats = crop_bridge_pairs(
            rois, self.ref, self.real, self.seg, self.output_dir, bridge_dis=10
        )
        self.assertEqual(stats["n_candidates"], 2)
        self.assertEqual(stats["n_pairs_in_range"], 1)
        self.assertEqual(stats["n_emitted"], 1)
        bridge = stats["bridges"][0]
        self.assertEqual(bridge["pair"], ["0001", "0002"])
        self.assertEqual(bridge["bbox_inner"], [10, 10, 35, 20])
        self.assertTrue(
            os.path.isfile(os.path.join(self.output_dir, "normal_img", "0001.png"))
        )

    def test_pairs_farther_than_bridge_dis_do_not_merge(self):
        rois = [
            {"name": "0001", "bbox_inner": [0, 0, 5, 5], "dominant_class": "capacitor"},
            {
                "name": "0002",
                "bbox_inner": [50, 50, 55, 55],
                "dominant_class": "capacitor",
            },
        ]
        stats = crop_bridge_pairs(
            rois, self.ref, self.real, self.seg, self.output_dir, bridge_dis=5
        )
        self.assertEqual(stats["n_pairs_in_range"], 0)
        self.assertEqual(stats["n_emitted"], 0)

    def test_bridge_classes_filters_candidates(self):
        rois = [
            {
                "name": "0001",
                "bbox_inner": [10, 10, 20, 20],
                "dominant_class": "capacitor",
            },
            {
                "name": "0002",
                "bbox_inner": [25, 10, 35, 20],
                "dominant_class": "capacitor",
            },
            {
                "name": "0003",
                "bbox_inner": [40, 10, 50, 20],
                "dominant_class": "resistor",
            },
        ]
        stats = crop_bridge_pairs(
            rois,
            self.ref,
            self.real,
            self.seg,
            self.output_dir,
            bridge_dis=10,
            bridge_classes=["capacitor"],
        )
        self.assertEqual(stats["n_candidates"], 2)  # resistor filtered out
        self.assertEqual(stats["n_pairs_in_range"], 1)
        self.assertEqual(stats["bridges"][0]["pair"], ["0001", "0002"])

    def test_fewer_than_two_candidates_yields_zero_emitted_without_io(self):
        # short-circuits before opening any image, so bogus paths are fine
        stats = crop_bridge_pairs(
            [
                {
                    "name": "0001",
                    "bbox_inner": [0, 0, 5, 5],
                    "dominant_class": "capacitor",
                }
            ],
            "/does/not/exist_ref.png",
            "/does/not/exist_real.png",
            "/does/not/exist_seg.png",
            self.output_dir,
            bridge_dis=10,
        )
        self.assertEqual(stats["n_candidates"], 1)
        self.assertEqual(stats["n_emitted"], 0)
        self.assertEqual(stats["bridges"], [])

        stats_empty = crop_bridge_pairs(
            [],
            "/does/not/exist_ref.png",
            "/does/not/exist_real.png",
            "/does/not/exist_seg.png",
            self.output_dir,
            bridge_dis=10,
        )
        self.assertEqual(stats_empty["n_candidates"], 0)
        self.assertEqual(stats_empty["n_emitted"], 0)


if __name__ == "__main__":
    unittest.main()
