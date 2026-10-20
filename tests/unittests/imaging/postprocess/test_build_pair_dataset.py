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

"""CLI tests for ``scripts/postprocess/build_pair_dataset.py`` (ChangeNet-
style golden/defect/mask pair dataset builder).

Needs Pillow (and numpy); skips itself when absent, matching the CI
unit-test image which installs numpy only.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    from PIL import Image
except ImportError:  # CI unit-test image installs numpy only
    Image = None

_SCRIPTS_POSTPROCESS = (
    Path(__file__).resolve().parents[4] / "scripts" / "postprocess"
)
if str(_SCRIPTS_POSTPROCESS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_POSTPROCESS))

if Image is not None:
    import build_pair_dataset


def _write_rgb(path, size=16, value=128):
    arr = np.full((size, size, 3), value, dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


@unittest.skipUnless(Image is not None, "Pillow not installed (CI installs numpy only)")
class RequiredArgsTest(unittest.TestCase):
    def _run_expect_exit(self, argv):
        with mock.patch.object(sys, "argv", ["build_pair_dataset.py"] + argv), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_pair_dataset.main()

    def test_missing_good_arg_exits(self):
        self._run_expect_exit(["--defect", "d", "--output", "o"])

    def test_missing_defect_arg_exits(self):
        self._run_expect_exit(["--good", "g", "--output", "o"])

    def test_missing_output_arg_exits(self):
        self._run_expect_exit(["--good", "g", "--defect", "d"])


@unittest.skipUnless(Image is not None, "Pillow not installed (CI installs numpy only)")
class FrameModeHappyPathTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pcb_aoi_pair_dataset_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.good_dir = os.path.join(self.tmp, "good")
        self.defect_dir = os.path.join(self.tmp, "defect")
        self.out_dir = os.path.join(self.tmp, "out")
        os.makedirs(self.good_dir)
        os.makedirs(self.defect_dir)

        _write_rgb(os.path.join(self.good_dir, "rgb_0000.png"), value=200)
        _write_rgb(os.path.join(self.defect_dir, "rgb_0000.png"), value=50)

        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        arr[4:8, 4:8] = (255, 0, 0, 255)  # defect-colored patch
        Image.fromarray(arr, "RGBA").save(
            os.path.join(self.defect_dir, "semantic_segmentation_0000.png"))
        with open(os.path.join(
                self.defect_dir, "semantic_segmentation_labels_0000.json"),
                "w") as f:
            json.dump({"(255, 0, 0, 255)": {"defect": "tombstone"}}, f)

    def _run(self, extra=()):
        argv = ["--good", self.good_dir, "--defect", self.defect_dir,
               "--output", self.out_dir, "--light-mode", "SolderLight"]
        argv += list(extra)
        with mock.patch.object(sys, "argv", ["build_pair_dataset.py"] + argv):
            build_pair_dataset.main()

    def test_paired_frame_with_visible_defect_is_copied(self):
        self._run()

        golden = os.path.join(self.out_dir, "golden", "0000_SolderLight.png")
        defect = os.path.join(self.out_dir, "defect", "0000_SolderLight.png")
        mask = os.path.join(self.out_dir, "mask", "0000_SolderLight.png")
        mask_json = os.path.join(self.out_dir, "mask", "0000_SolderLight.json")
        for path in (golden, defect, mask, mask_json):
            self.assertTrue(os.path.isfile(path), path)

        with Image.open(golden) as img:
            self.assertEqual(img.convert("RGB").getpixel((0, 0)), (200, 200, 200))
        with Image.open(defect) as img:
            self.assertEqual(img.convert("RGB").getpixel((0, 0)), (50, 50, 50))

    def test_frame_without_visible_defect_pixels_is_skipped(self):
        # overwrite the seg mask with an all-blank frame -- defect color is
        # in labels.json but no pixel actually carries it
        blank = np.zeros((16, 16, 4), dtype=np.uint8)
        Image.fromarray(blank, "RGBA").save(
            os.path.join(self.defect_dir, "semantic_segmentation_0000.png"))

        self._run()

        self.assertFalse(os.path.exists(
            os.path.join(self.out_dir, "golden", "0000_SolderLight.png")))


if __name__ == "__main__":
    unittest.main()
