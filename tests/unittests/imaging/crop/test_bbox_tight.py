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

"""Tests for ``scripts/crop/bbox_tight.py`` (``crop_components``).

Runs end-to-end on tiny synthetic replicator-writer fixtures generated in
tmpdirs: 64x64 PNGs plus ``bounding_box_2d_tight_*.npy`` structured arrays and
the parallel ``*_prim_paths_*.json`` / ``*_labels_*.json`` files.

Needs Pillow (and numpy); skips itself when Pillow is absent, matching the CI
unit-test image which installs numpy only.
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
    from PIL import Image
except ImportError:  # CI unit-test image installs numpy only
    Image = None

_SCRIPTS_CROP = Path(__file__).resolve().parents[4] / "scripts" / "crop"
if str(_SCRIPTS_CROP) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_CROP))

if Image is not None:
    from bbox_tight import crop_components

# Matches the replicator BasicWriter bounding_box_2d_tight schema (field names
# are what crop_components reads; exact widths are not load-bearing).
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
# Default XFORM_DEPTH is 7: "/World/pcba/g/a/b/<comp>/<mesh>" truncates to
# "/World/pcba/g/a/b/<comp>", so <comp> becomes the crop filename suffix.
_PRIM_PREFIX = "/World/pcba/g/a/b"

# (component, semanticId, tight bbox x_min/y_min/x_max/y_max)
_ENTRIES = [
    ("C1", 1, (10, 10, 20, 20)),  # sub-mesh 1 of C1
    ("C1", 1, (15, 15, 25, 25)),  # sub-mesh 2 of C1 -> merged bbox (10,10,25,25)
    ("R2", 2, (40, 40, 52, 52)),  # defect-labelled component
    ("C3", 3, (34, 4, 46, 14)),   # zero semantic coverage -> min_coverage filter
    ("E4", 4, (0, 30, 10, 40)),   # touches image edge -> always skipped
]

_LABELS = {
    "1": {"class": "capacitor,SMD"},
    "2": {"class": "resistor", "defect": "tombstone"},
    "3": {"class": "diode"},
    "4": {"class": "led"},
}

# Regions (x1, y1, x2, y2) painted opaque in the semantic mask: C1 + R2 only.
_SEM_REGIONS = [(10, 10, 25, 25), (40, 40, 52, 52)]

_CROPS = ["rgb", "semantic_segmentation"]


def _write_frame(anno_dir: str, rgb_dir: str | None = None, frame: str = "0000",
                 rgb_value: int = 128) -> None:
    """Write one synthetic replicator frame (annotations + images)."""
    rgb_dir = rgb_dir if rgb_dir is not None else anno_dir

    bboxes = np.array(
        [(sid, *box, 0.0) for _comp, sid, box in _ENTRIES], dtype=_BBOX_DTYPE
    )
    np.save(os.path.join(anno_dir, f"bounding_box_2d_tight_{frame}.npy"), bboxes)

    prim_paths = [f"{_PRIM_PREFIX}/{comp}/mesh_{i}" for i, (comp, _sid, _box) in enumerate(_ENTRIES)]
    with open(os.path.join(anno_dir, f"bounding_box_2d_tight_prim_paths_{frame}.json"), "w") as f:
        json.dump(prim_paths, f)
    with open(os.path.join(anno_dir, f"bounding_box_2d_tight_labels_{frame}.json"), "w") as f:
        json.dump(_LABELS, f)

    rgb = np.full((_IMG_SIZE, _IMG_SIZE, 3), rgb_value, dtype=np.uint8)
    Image.fromarray(rgb, "RGB").save(os.path.join(rgb_dir, f"rgb_{frame}.png"))

    sem = np.zeros((_IMG_SIZE, _IMG_SIZE, 4), dtype=np.uint8)
    for x1, y1, x2, y2 in _SEM_REGIONS:
        sem[y1:y2, x1:x2] = (0, 200, 0, 255)
    Image.fromarray(sem, "RGBA").save(
        os.path.join(anno_dir, f"semantic_segmentation_{frame}.png")
    )


@unittest.skipUnless(Image is not None, "Pillow not installed (CI installs numpy only)")
class CropComponentsTest(unittest.TestCase):
    """End-to-end crop_components on a synthetic single-frame trigger dir."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pcb_aoi_bbox_tight_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.input_dir = os.path.join(self.tmp, "trigger_0000")
        self.output_dir = os.path.join(self.tmp, "cropped")
        os.makedirs(self.input_dir)
        _write_frame(self.input_dir)

    def _crop(self, output_dir=None, **kwargs):
        kwargs.setdefault("offset", 2)
        kwargs.setdefault("crop_types", _CROPS)
        crop_components(self.input_dir, output_dir or self.output_dir, **kwargs)

    def test_crops_written_per_component_under_label_folders(self):
        self._crop()
        # C1: class folder with "," and " " sanitised to "_"
        for ct in _CROPS:
            self.assertTrue(
                os.path.isfile(
                    os.path.join(self.output_dir, "capacitor_SMD", ct, "frame0000_C1.png")
                ),
                f"missing capacitor_SMD/{ct} crop",
            )
        # R2: defect label wins over class as folder name
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.output_dir, "tombstone", "rgb", "frame0000_R2.png")
            )
        )
        self.assertFalse(os.path.isdir(os.path.join(self.output_dir, "resistor")))

    def test_defect_label_cannot_escape_output_directory(self):
        labels_path = os.path.join(
            self.input_dir, "bounding_box_2d_tight_labels_0000.json")
        with open(labels_path) as f:
            labels = json.load(f)
        labels["2"]["defect"] = os.path.join("..", "..", "..", "escaped")
        with open(labels_path, "w") as f:
            json.dump(labels, f)

        nested_output = os.path.join(self.tmp, "output", "deep", "nested")
        self._crop(output_dir=nested_output)

        self.assertTrue(os.path.isfile(os.path.join(
            nested_output, "escaped", "rgb", "frame0000_R2.png")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "escaped")))

    def test_submesh_bboxes_merged_into_single_component_crop(self):
        self._crop()
        rgb_dir = os.path.join(self.output_dir, "capacitor_SMD", "rgb")
        self.assertEqual(os.listdir(rgb_dir), ["frame0000_C1.png"])
        with Image.open(os.path.join(rgb_dir, "frame0000_C1.png")) as img:
            # merged bbox (10,10,25,25) + offset 2 each side, no +1 on max edge
            self.assertEqual(img.size, (19, 19))

    def test_offset_applied_to_crop_box(self):
        self._crop(offset=2)
        path = os.path.join(self.output_dir, "tombstone", "rgb", "frame0000_R2.png")
        with Image.open(path) as img:
            self.assertEqual(img.size, (52 - 40 + 4, 52 - 40 + 4))  # (16, 16)

    def test_offset_clamped_at_image_boundary(self):
        # min_coverage=0 so the heavily padded (mostly board) crop survives.
        self._crop(offset=20, min_coverage=0.0)
        path = os.path.join(self.output_dir, "tombstone", "rgb", "frame0000_R2.png")
        with Image.open(path) as img:
            # x: max(0, 40-20)=20 .. min(64, 52+20)=64 -> 44 px, same for y
            self.assertEqual(img.size, (44, 44))

    def test_min_coverage_filters_low_coverage_component(self):
        self._crop()  # default min_coverage 0.2; C3 has an empty semantic mask
        self.assertFalse(os.path.isdir(os.path.join(self.output_dir, "diode")))

        relaxed_out = os.path.join(self.tmp, "cropped_relaxed")
        self._crop(output_dir=relaxed_out, min_coverage=0.0)
        self.assertTrue(
            os.path.isfile(os.path.join(relaxed_out, "diode", "rgb", "frame0000_C3.png"))
        )

    def test_edge_touching_component_always_skipped(self):
        self._crop(min_coverage=0.0)
        self.assertFalse(os.path.isdir(os.path.join(self.output_dir, "led")))

    def test_class_filter_matches_class_substring(self):
        self._crop(class_filter=["resistor"])
        self.assertTrue(os.path.isdir(os.path.join(self.output_dir, "tombstone")))
        self.assertFalse(os.path.isdir(os.path.join(self.output_dir, "capacitor_SMD")))

    def test_class_filter_matches_defect_substring(self):
        self._crop(class_filter=["tomb"])
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.output_dir, "tombstone", "rgb", "frame0000_R2.png")
            )
        )
        self.assertFalse(os.path.isdir(os.path.join(self.output_dir, "capacitor_SMD")))

    def test_missing_mode_splits_missing_and_ok(self):
        # reference_dir holds annotations + "good" rgb; input_dir holds the
        # defective rgb only (bbox/labels/semantic are read from reference).
        ref_dir = os.path.join(self.tmp, "reference")
        defect_dir = os.path.join(self.tmp, "defect_rgb")
        out = os.path.join(self.tmp, "cropped_missing")
        os.makedirs(ref_dir)
        os.makedirs(defect_dir)
        _write_frame(ref_dir, rgb_value=200)
        _write_frame(defect_dir, rgb_value=50)  # only rgb is consumed from here
        for f in os.listdir(defect_dir):
            if not f.startswith("rgb_"):
                os.remove(os.path.join(defect_dir, f))

        crop_components(defect_dir, out, offset=2, reference_dir=ref_dir,
                        crop_types=_CROPS)

        for folder in ("missing", "ok"):
            for ct in _CROPS:
                self.assertTrue(
                    os.path.isfile(os.path.join(out, folder, ct, "frame0000_C1.png")),
                    f"missing {folder}/{ct} crop",
                )
        # "missing" rgb comes from input_dir, "ok" rgb from reference_dir
        with Image.open(os.path.join(out, "missing", "rgb", "frame0000_C1.png")) as img:
            self.assertEqual(img.convert("RGB").getpixel((0, 0)), (50, 50, 50))
        with Image.open(os.path.join(out, "ok", "rgb", "frame0000_C1.png")) as img:
            self.assertEqual(img.convert("RGB").getpixel((0, 0)), (200, 200, 200))

    def test_no_bbox_files_is_a_logged_noop(self):
        empty_in = os.path.join(self.tmp, "empty_trigger")
        out = os.path.join(self.tmp, "cropped_empty")
        os.makedirs(empty_in)
        with self.assertLogs("bbox_tight", level="ERROR"):
            crop_components(empty_in, out, crop_types=_CROPS)
        self.assertFalse(os.path.exists(out))

    def test_rgb_only_crop_types_produces_no_crops(self):
        # Documented quirk: without a "semantic*" crop type there is no
        # coverage mask, and crop_components skips every component (the
        # sem_img-is-None branch is a hard `continue`), so nothing is written.
        self._crop(crop_types=["rgb"])
        self.assertFalse(os.path.exists(self.output_dir))

    def test_frame_without_prim_paths_json_is_skipped(self):
        os.remove(
            os.path.join(self.input_dir, "bounding_box_2d_tight_prim_paths_0000.json")
        )
        with self.assertLogs("bbox_tight", level="WARNING"):
            self._crop()
        self.assertFalse(os.path.exists(self.output_dir))

    def test_frame_without_labels_json_is_skipped(self):
        os.remove(os.path.join(self.input_dir, "bounding_box_2d_tight_labels_0000.json"))
        self._crop()
        self.assertFalse(os.path.exists(self.output_dir))

    def test_frame_without_rgb_is_skipped(self):
        os.remove(os.path.join(self.input_dir, "rgb_0000.png"))
        self._crop()
        self.assertFalse(os.path.exists(self.output_dir))

    def test_default_crop_types_cover_rgb_and_segmentations(self):
        # crop_types=None -> DEFAULT_CROPS; instance_id_segmentation images are
        # absent from the fixture, so its per-label dir is created but empty.
        crop_components(self.input_dir, self.output_dir, offset=2)
        label_dir = os.path.join(self.output_dir, "capacitor_SMD")
        for ct in ("rgb", "semantic_segmentation"):
            self.assertTrue(os.path.isfile(os.path.join(label_dir, ct, "frame0000_C1.png")))
        inst_dir = os.path.join(label_dir, "instance_id_segmentation")
        self.assertTrue(os.path.isdir(inst_dir))
        self.assertEqual(os.listdir(inst_dir), [])

    def test_missing_mode_without_reference_rgb_writes_only_missing(self):
        ref_dir = os.path.join(self.tmp, "reference_no_rgb")
        out = os.path.join(self.tmp, "cropped_missing_no_ref_rgb")
        os.makedirs(ref_dir)
        _write_frame(ref_dir)
        os.remove(os.path.join(ref_dir, "rgb_0000.png"))

        crop_components(self.input_dir, out, offset=2, reference_dir=ref_dir,
                        crop_types=_CROPS)

        self.assertTrue(
            os.path.isfile(os.path.join(out, "missing", "rgb", "frame0000_C1.png"))
        )
        self.assertFalse(os.path.isdir(os.path.join(out, "ok")))

    def test_post_filter_removes_stale_low_coverage_crops(self):
        # Pre-existing crops in output_dir whose saved semantic mask is (now)
        # below min_coverage are swept out by the post-filter pass.
        for ct in _CROPS:
            os.makedirs(os.path.join(self.output_dir, "capacitor_SMD", ct))
        blank = np.zeros((8, 8, 4), dtype=np.uint8)
        Image.fromarray(blank, "RGBA").save(
            os.path.join(self.output_dir, "capacitor_SMD", "semantic_segmentation", "stale.png")
        )
        Image.fromarray(blank[:, :, :3], "RGB").save(
            os.path.join(self.output_dir, "capacitor_SMD", "rgb", "stale.png")
        )

        self._crop()

        for ct in _CROPS:
            self.assertFalse(
                os.path.exists(os.path.join(self.output_dir, "capacitor_SMD", ct, "stale.png")),
                f"stale {ct} crop not removed by post-filter",
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(self.output_dir, "capacitor_SMD", ct, "frame0000_C1.png")
                )
            )


if __name__ == "__main__":
    unittest.main()
