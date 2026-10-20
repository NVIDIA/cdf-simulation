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

"""Tests for ``scripts/crop/pipeline_adapter.py`` (single-flow crop stage).

CI-safe: stdlib ``unittest`` + numpy only (numpy is pulled in transitively by
``facade`` -> ``bbox_grouping``); no PIL / cv2 / omni imports. The adapter is
pure dict/path logic, so everything runs on strings — no files are created.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Import the code under test the same way the pipeline does: scripts/crop is
# used as a flat, top-level module directory (not a package import).
_SCRIPTS_CROP = Path(__file__).resolve().parents[4] / "scripts" / "crop"
if str(_SCRIPTS_CROP) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_CROP))

import pipeline_adapter
from pipeline_adapter import build_pipeline_crop_config, run_pipeline_crop


def _out_root(*parts: str) -> str:
    """Absolute-but-portable output root string (no dirs are created)."""
    return os.path.join(tempfile.gettempdir(), "pcb_aoi_crop_adapter_test", *parts)


class BuildPipelineCropConfigDisabledTest(unittest.TestCase):
    """``None`` (crop disabled) cases."""

    def test_no_crop_key_returns_none(self):
        self.assertIsNone(build_pipeline_crop_config({"output": _out_root()}))

    def test_crop_key_none_returns_none(self):
        self.assertIsNone(
            build_pipeline_crop_config({"crop": None, "output": _out_root()})
        )

    def test_enabled_false_returns_none(self):
        cfg = {
            "output": _out_root(),
            "crop": {"enabled": False, "method": "bbox_tight"},
        }
        self.assertIsNone(build_pipeline_crop_config(cfg))

    def test_enabled_missing_defaults_to_disabled(self):
        cfg = {"output": _out_root(), "crop": {"method": "bbox_tight"}}
        self.assertIsNone(build_pipeline_crop_config(cfg))


class BuildPipelineCropConfigValidationTest(unittest.TestCase):
    """ValueError cases: method and output validation."""

    def test_non_bbox_tight_method_raises(self):
        cfg = {"output": _out_root(), "crop": {"enabled": True, "method": "label_cc"}}
        with self.assertRaises(ValueError) as ctx:
            build_pipeline_crop_config(cfg)
        self.assertIn("label_cc", str(ctx.exception))
        self.assertIn("bbox_tight", str(ctx.exception))

    def test_missing_method_raises(self):
        cfg = {"output": _out_root(), "crop": {"enabled": True}}
        with self.assertRaises(ValueError) as ctx:
            build_pipeline_crop_config(cfg)
        self.assertIn("None", str(ctx.exception))

    def test_missing_output_raises(self):
        cfg = {"crop": {"enabled": True, "method": "bbox_tight"}}
        with self.assertRaises(ValueError) as ctx:
            build_pipeline_crop_config(cfg)
        self.assertIn("output", str(ctx.exception))

    def test_output_dict_without_dir_raises(self):
        cfg = {"output": {}, "crop": {"enabled": True, "method": "bbox_tight"}}
        with self.assertRaises(ValueError):
            build_pipeline_crop_config(cfg)

    def test_boolean_trigger_index_raises(self):
        cfg = {
            "output": _out_root(),
            "trigger_index": True,
            "crop": {"enabled": True, "method": "bbox_tight"},
        }
        with self.assertRaisesRegex(
            ValueError,
            "trigger_index must be an integer >= 0",
        ):
            build_pipeline_crop_config(cfg)


class BuildPipelineCropConfigResolutionTest(unittest.TestCase):
    """Input/output path resolution for the enabled bbox_tight case."""

    def _cfg(self, output, **crop_extra):
        crop = {"enabled": True, "method": "bbox_tight"}
        crop.update(crop_extra)
        return {"output": output, "crop": crop}

    def test_string_output_resolves_trigger_input_and_cropped_output(self):
        root = _out_root("run1")
        resolved = build_pipeline_crop_config(self._cfg(root))
        self.assertEqual(resolved["input"], os.path.join(root, "trigger_0000"))
        self.assertEqual(resolved["output"], os.path.join(root, "cropped"))

    def test_dict_output_uses_dir_key(self):
        root = _out_root("run2")
        resolved = build_pipeline_crop_config(self._cfg({"dir": root}))
        self.assertEqual(resolved["input"], os.path.join(root, "trigger_0000"))
        self.assertEqual(resolved["output"], os.path.join(root, "cropped"))

    def test_output_subdir_override(self):
        root = _out_root("run3")
        resolved = build_pipeline_crop_config(self._cfg(root, output_subdir="crops_v2"))
        self.assertEqual(resolved["output"], os.path.join(root, "crops_v2"))

    def test_trigger_index_selects_the_manifest_trigger(self):
        root = _out_root("run_trigger")
        cfg = self._cfg(root, output_subdir="crops/trigger_0003")
        cfg["trigger_index"] = 3

        resolved = build_pipeline_crop_config(cfg)

        self.assertEqual(resolved["input"], os.path.join(root, "trigger_0003"))
        self.assertEqual(
            resolved["output"],
            os.path.join(root, "crops", "trigger_0003"),
        )

    def test_env_var_in_output_is_expanded(self):
        root = _out_root("run_env")
        with mock.patch.dict(os.environ, {"PCB_AOI_TEST_OUT": root}):
            resolved = build_pipeline_crop_config(self._cfg("${PCB_AOI_TEST_OUT}"))
        self.assertEqual(resolved["input"], os.path.join(root, "trigger_0000"))
        self.assertEqual(resolved["output"], os.path.join(root, "cropped"))

    def test_tilde_in_output_is_expanded(self):
        resolved = build_pipeline_crop_config(self._cfg("~/pcb_aoi_test_out"))
        expected_root = os.path.abspath(os.path.expanduser("~/pcb_aoi_test_out"))
        self.assertEqual(resolved["input"], os.path.join(expected_root, "trigger_0000"))
        self.assertNotIn("~", resolved["input"])

    def test_relative_output_is_made_absolute(self):
        resolved = build_pipeline_crop_config(self._cfg("rel_out_dir"))
        self.assertTrue(os.path.isabs(resolved["input"]))
        self.assertTrue(os.path.isabs(resolved["output"]))

    def test_crop_keys_passed_through_unchanged(self):
        root = _out_root("run4")
        resolved = build_pipeline_crop_config(
            self._cfg(
                root, offset=5, min_coverage=0.3, types=["rgb", "semantic_segmentation"]
            )
        )
        self.assertEqual(resolved["offset"], 5)
        self.assertEqual(resolved["min_coverage"], 0.3)
        self.assertEqual(resolved["types"], ["rgb", "semantic_segmentation"])
        self.assertEqual(resolved["method"], "bbox_tight")

    def test_original_cfg_not_mutated(self):
        root = _out_root("run5")
        cfg = self._cfg(root)
        crop_before = dict(cfg["crop"])
        build_pipeline_crop_config(cfg)
        self.assertEqual(cfg["crop"], crop_before)
        self.assertNotIn("input", cfg["crop"])


class BuildPipelineCropConfigTypesValidationTest(unittest.TestCase):
    """rgb-only crop.types must fail at config time (would write ZERO crops:
    bbox_tight drops every crop whose semantic coverage cannot be measured)."""

    def _cfg(self, **crop_extra):
        crop = {"enabled": True, "method": "bbox_tight"}
        crop.update(crop_extra)
        return {"output": _out_root("types_run"), "crop": crop}

    def test_rgb_only_types_raises_actionable_error(self):
        with self.assertRaises(ValueError) as ctx:
            build_pipeline_crop_config(self._cfg(types=["rgb"]))
        message = str(ctx.exception)
        self.assertIn("semantic", message)
        self.assertIn("zero crops", message)
        self.assertIn("semantic_segmentation", message)  # actionable fix

    def test_rgb_only_crops_cli_alias_raises(self):
        with self.assertRaises(ValueError):
            build_pipeline_crop_config(self._cfg(crops=["rgb"]))

    def test_single_string_type_without_semantic_raises(self):
        with self.assertRaises(ValueError):
            build_pipeline_crop_config(self._cfg(types="rgb"))

    def test_non_semantic_multi_type_list_raises(self):
        # "instance_id_segmentation" is a segmentation but NOT usable for
        # semantic coverage filtering (bbox_tight matches "semantic" in ct).
        with self.assertRaises(ValueError):
            build_pipeline_crop_config(
                self._cfg(types=["rgb", "instance_id_segmentation"])
            )

    def test_types_with_semantic_passes(self):
        resolved = build_pipeline_crop_config(
            self._cfg(types=["rgb", "semantic_segmentation"])
        )
        self.assertEqual(resolved["types"], ["rgb", "semantic_segmentation"])

    def test_omitted_types_uses_defaults_and_passes(self):
        resolved = build_pipeline_crop_config(self._cfg())
        self.assertNotIn("types", resolved)


class BuildPipelineCropConfigMissingModeTest(unittest.TestCase):
    """``missing`` mode: requires matching pipeline_type + subdir pair."""

    def _cfg(self, pipeline_type=None, **crop_extra):
        crop = {"enabled": True, "method": "bbox_tight"}
        crop.update(crop_extra)
        cfg = {"output": _out_root("missing_run"), "crop": crop}
        if pipeline_type is not None:
            cfg["pipeline_type"] = pipeline_type
        return cfg

    def test_missing_mode_resolves_input_and_reference(self):
        cfg = self._cfg(
            pipeline_type="missing",
            mode="missing",
            input_subdir="defect",
            reference_subdir="good",
        )
        resolved = build_pipeline_crop_config(cfg)
        trigger = os.path.join(_out_root("missing_run"), "trigger_0000")
        self.assertEqual(resolved["input"], os.path.join(trigger, "defect"))
        self.assertEqual(resolved["reference"], os.path.join(trigger, "good"))

    def test_pipeline_type_missing_without_crop_mode_raises(self):
        cfg = self._cfg(pipeline_type="missing")
        with self.assertRaises(ValueError) as ctx:
            build_pipeline_crop_config(cfg)
        self.assertIn("missing", str(ctx.exception))

    def test_crop_mode_missing_without_pipeline_type_raises(self):
        cfg = self._cfg(
            pipeline_type="good",
            mode="missing",
            input_subdir="defect",
            reference_subdir="good",
        )
        with self.assertRaises(ValueError):
            build_pipeline_crop_config(cfg)

    def test_missing_mode_requires_both_subdirs(self):
        for subdirs in ({}, {"input_subdir": "defect"}, {"reference_subdir": "good"}):
            with self.subTest(subdirs=subdirs):
                cfg = self._cfg(pipeline_type="missing", mode="missing", **subdirs)
                with self.assertRaises(ValueError) as ctx:
                    build_pipeline_crop_config(cfg)
                self.assertIn("input_subdir", str(ctx.exception))


class CropValidationSharedRulesTest(unittest.TestCase):
    """The crop rules live ONCE in scripts/crop/crop_validation.py; both
    pipeline_adapter and cdf/sections.py call it. Pin the third copy too:
    the omitted-types defaults must satisfy the shared semantic-type rule."""

    def test_default_crops_carry_the_canonical_semantic_type(self):
        import crop_defaults
        import crop_validation

        self.assertTrue(crop_validation.has_semantic_type(crop_defaults.DEFAULT_CROPS))
        self.assertIn(
            crop_validation.DEFAULT_SEMANTIC_TYPE, crop_defaults.DEFAULT_CROPS
        )

    def test_error_builders_return_none_for_valid_input(self):
        import crop_validation

        self.assertIsNone(crop_validation.crop_method_error("bbox_tight"))
        self.assertIsNone(
            crop_validation.crop_types_error(["rgb", "semantic_segmentation"])
        )

    def test_adapter_uses_the_shared_rules(self):
        import crop_validation

        self.assertIs(pipeline_adapter.crop_validation, crop_validation)


class RunPipelineCropTest(unittest.TestCase):
    """``run_pipeline_crop`` with an injected runner stub."""

    def test_disabled_returns_false_and_skips_runner(self):
        calls = []
        with contextlib.redirect_stdout(io.StringIO()):
            ran = run_pipeline_crop({"output": _out_root()}, runner=calls.append)
        self.assertFalse(ran)
        self.assertEqual(calls, [])

    def test_enabled_invokes_runner_with_resolved_config(self):
        root = _out_root("run_enabled")
        cfg = {
            "output": root,
            "crop": {"enabled": True, "method": "bbox_tight", "offset": 3},
        }
        calls = []
        with contextlib.redirect_stdout(io.StringIO()):
            ran = run_pipeline_crop(cfg, runner=calls.append)
        self.assertTrue(ran)
        self.assertEqual(len(calls), 1)
        resolved = calls[0]
        self.assertEqual(resolved["method"], "bbox_tight")
        self.assertEqual(resolved["offset"], 3)
        self.assertEqual(resolved["input"], os.path.join(root, "trigger_0000"))
        self.assertEqual(resolved["output"], os.path.join(root, "cropped"))

    def test_invalid_method_propagates_value_error(self):
        cfg = {"output": _out_root(), "crop": {"enabled": True, "method": "region"}}
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(ValueError),
        ):
            run_pipeline_crop(cfg, runner=lambda _cfg: None)

    def test_default_runner_is_facade_run_crop(self):
        # Guards the wiring contract: the pipeline stage dispatches through
        # the shared facade unless a runner is injected.
        defaults = run_pipeline_crop.__kwdefaults__ or {}
        import facade

        self.assertIs(defaults.get("runner"), facade.run_crop)


if __name__ == "__main__":
    unittest.main()
