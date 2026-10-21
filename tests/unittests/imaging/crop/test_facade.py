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

"""Tests for ``scripts/crop/facade.py`` (method dispatch).

CI-safe: stdlib ``unittest`` + numpy only. The per-method strategy modules
(``bbox_tight`` needs PIL, ``label_cc`` needs cv2) are imported lazily inside
the facade, so dispatch is tested against stub modules injected into
``sys.modules`` — no third-party imaging deps are required.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

_SCRIPTS_CROP = Path(__file__).resolve().parents[4] / "scripts" / "crop"
if str(_SCRIPTS_CROP) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_CROP))

import facade  # noqa: E402
from crop_defaults import DEFAULT_CROPS, MIN_COVERAGE, OFFSET, XFORM_DEPTH  # noqa: E402


def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class UnknownMethodTest(unittest.TestCase):
    def test_unknown_method_raises_with_choices(self):
        with self.assertRaises(ValueError) as ctx:
            facade.run_crop({"method": "bogus"})
        msg = str(ctx.exception)
        self.assertIn("bogus", msg)
        for known in facade.METHODS:
            self.assertIn(known, msg)

    def test_missing_method_raises(self):
        with self.assertRaises(ValueError) as ctx:
            facade.run_crop({})
        self.assertIn("None", str(ctx.exception))


class BboxTightDispatchTest(unittest.TestCase):
    def _run(self, cfg):
        calls = []

        def fake_crop_components(**kwargs):
            calls.append(kwargs)
            return None

        stub = _stub_module("bbox_tight", crop_components=fake_crop_components)
        with mock.patch.dict(sys.modules, {"bbox_tight": stub}):
            result = facade.run_crop(cfg)
        self.assertEqual(len(calls), 1)
        return result, calls[0]

    def test_defaults_flow_through(self):
        result, kwargs = self._run({"method": "bbox_tight", "input": "in", "output": "out"})
        self.assertIsNone(result)
        self.assertEqual(kwargs["input_dir"], "in")
        self.assertEqual(kwargs["output_dir"], "out")
        self.assertEqual(kwargs["offset"], OFFSET)
        self.assertIsNone(kwargs["class_filter"])
        self.assertIsNone(kwargs["reference_dir"])
        self.assertEqual(kwargs["crop_types"], DEFAULT_CROPS)
        self.assertEqual(kwargs["component_xform_depth"], XFORM_DEPTH)
        self.assertEqual(kwargs["min_coverage"], MIN_COVERAGE)

    def test_overrides_flow_through(self):
        _, kwargs = self._run(
            {
                "method": "bbox_tight",
                "input": "in",
                "output": "out",
                "offset": 4,
                "class_filter": ["resistor"],
                "reference": "ref",
                "xform_depth": 5,
                "min_coverage": 0.5,
            }
        )
        self.assertEqual(kwargs["offset"], 4)
        self.assertEqual(kwargs["class_filter"], ["resistor"])
        self.assertEqual(kwargs["reference_dir"], "ref")
        self.assertEqual(kwargs["component_xform_depth"], 5)
        self.assertEqual(kwargs["min_coverage"], 0.5)

    def test_crops_key_takes_precedence_over_types(self):
        _, kwargs = self._run(
            {"method": "bbox_tight", "input": "i", "output": "o",
             "crops": ["rgb"], "types": ["rgb", "semantic_segmentation"]}
        )
        self.assertEqual(kwargs["crop_types"], ["rgb"])

    def test_types_key_accepted_when_crops_absent(self):
        _, kwargs = self._run(
            {"method": "bbox_tight", "input": "i", "output": "o", "types": ["rgb"]}
        )
        self.assertEqual(kwargs["crop_types"], ["rgb"])


class RegionDispatchTest(unittest.TestCase):
    def test_region_dispatch_maps_config(self):
        calls = []
        sentinel_outputs = ("rgb",)

        def fake_crop_sdg(**kwargs):
            calls.append(kwargs)
            return {"stats": 1}

        stub = _stub_module(
            "region", crop_sdg=fake_crop_sdg, DEFAULT_OUTPUTS=sentinel_outputs
        )
        with mock.patch.dict(sys.modules, {"region": stub}):
            result = facade.run_crop(
                {"method": "region", "src_dir": "s", "dst_dir": "d", "bbox": (1, 2, 3, 4)}
            )
        self.assertEqual(result, {"stats": 1})
        self.assertEqual(len(calls), 1)
        kwargs = calls[0]
        self.assertEqual(kwargs["src_dir"], "s")
        self.assertEqual(kwargs["dst_dir"], "d")
        self.assertEqual(kwargs["bbox"], (1, 2, 3, 4))
        self.assertEqual(kwargs["frame_idx"], "0000")
        self.assertEqual(kwargs["xform_depth"], XFORM_DEPTH)
        self.assertTrue(kwargs["include_loose"])
        self.assertEqual(kwargs["outputs"], sentinel_outputs)


class LabelCcDispatchTest(unittest.TestCase):
    def test_label_cc_dispatch_maps_config(self):
        calls = []

        def fake_crop_rois_by_label(**kwargs):
            calls.append(kwargs)
            return {"emitted": 3}

        stub = _stub_module("label_cc", crop_rois_by_label=fake_crop_rois_by_label)
        cfg = {
            "method": "label_cc",
            "aligned_dir": "a",
            "ref_crop_path": "r",
            "real_crop_path": "x",
            "output_dir": "o",
            "target_classes": ["cap"],
        }
        with mock.patch.dict(sys.modules, {"label_cc": stub}):
            result = facade.run_crop(cfg)
        self.assertEqual(result, {"emitted": 3})
        kwargs = calls[0]
        self.assertEqual(kwargs["aligned_dir"], "a")
        self.assertEqual(kwargs["target_classes"], ["cap"])
        self.assertEqual(kwargs["offset"], OFFSET)
        self.assertEqual(kwargs["min_coverage"], MIN_COVERAGE)
        self.assertEqual(kwargs["fallback_dir"], "component")
        self.assertFalse(kwargs["multi_frame"])


if __name__ == "__main__":
    unittest.main()
