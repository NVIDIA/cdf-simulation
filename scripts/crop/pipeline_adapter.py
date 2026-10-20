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

"""Adapt a single-flow pipeline config to the shared crop facade.

This module is deliberately free of Isaac Sim / Omniverse imports so the
``crop`` stage can run with ordinary Python in the SDG container.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

import crop_validation
from facade import run_crop

_FILE_UTILS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "file_utils")
)
if _FILE_UTILS_DIR not in sys.path:
    sys.path.insert(0, _FILE_UTILS_DIR)

from path_safety import confined_path


def build_pipeline_crop_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Build the flat ``run_crop`` config for one selected trigger.

    ``None`` means crop is disabled.  SDGW-796 intentionally covers only the
    single-flow ``bbox_tight`` method; ROI ``label_cc`` and registration
    ``region`` keep their existing orchestration paths.
    """
    crop_cfg = cfg.get("crop") or {}
    if not crop_cfg.get("enabled", False):
        return None

    # method + semantic-type rules are shared with the declarative API
    # (cdf/sections.py PostprocessSection) via crop_validation — one
    # source of truth, two error types (ValueError here, ValidationError
    # there).
    method_error = crop_validation.crop_method_error(crop_cfg.get("method"))
    if method_error:
        raise ValueError(f"sdg_pipeline crop stage: {method_error}")

    # "types" is the single-flow yaml key, "crops" the CLI alias (see
    # facade._run_bbox_tight). Only validated when given: the defaults
    # (crop_defaults.DEFAULT_CROPS) include a semantic type by contract.
    types_value = crop_cfg.get("types", crop_cfg.get("crops"))
    if types_value is not None:
        types_list = (
            [str(t) for t in types_value]
            if isinstance(types_value, (list, tuple))
            else [str(types_value)]
        )
        types_error = crop_validation.crop_types_error(types_list)
        if types_error:
            raise ValueError(types_error)

    output_value = cfg.get("output")
    if isinstance(output_value, dict):
        output_value = output_value.get("dir")
    if not output_value:
        raise ValueError("crop requires a top-level output path")

    output_root = os.path.abspath(
        os.path.expanduser(os.path.expandvars(str(output_value)))
    )
    trigger_index = cfg.get("trigger_index", 0)
    if (
        isinstance(trigger_index, bool)
        or not isinstance(trigger_index, int)
        or trigger_index < 0
    ):
        raise ValueError("crop trigger_index must be an integer >= 0")
    trigger_dir = os.path.join(output_root, f"trigger_{trigger_index:04d}")
    output_subdir = str(crop_cfg.get("output_subdir", "cropped"))

    resolved = dict(crop_cfg)
    resolved["input"] = trigger_dir
    resolved["output"] = confined_path(output_root, output_subdir)

    pipeline_type = cfg.get("pipeline_type", "good")
    crop_mode = crop_cfg.get("mode")
    is_missing = pipeline_type == "missing" or crop_mode == "missing"
    if is_missing:
        if pipeline_type != "missing" or crop_mode != "missing":
            raise ValueError(
                "missing crop requires both pipeline_type: missing and crop.mode: missing"
            )
        input_subdir = crop_cfg.get("input_subdir")
        reference_subdir = crop_cfg.get("reference_subdir")
        if not input_subdir or not reference_subdir:
            raise ValueError(
                "missing crop requires crop.input_subdir and crop.reference_subdir"
            )
        resolved["input"] = confined_path(trigger_dir, str(input_subdir))
        resolved["reference"] = confined_path(trigger_dir, str(reference_subdir))

    return resolved


def run_pipeline_crop(
    cfg: dict[str, Any],
    *,
    runner: Callable[[dict[str, Any]], Any] = run_crop,
) -> bool:
    """Run configured single-flow crop and report whether it was enabled."""
    resolved = build_pipeline_crop_config(cfg)
    if resolved is None:
        print("[Pipeline] Crop disabled; skipping")
        return False

    print(
        f"[Pipeline] Crop start: method={resolved['method']} "
        f"input={resolved['input']} output={resolved['output']}"
    )
    runner(resolved)
    print(f"[Pipeline] Crop done -> {resolved['output']}")
    return True


__all__ = ["build_pipeline_crop_config", "run_pipeline_crop"]
