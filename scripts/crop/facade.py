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

"""Single crop entry point — dispatch on ``crop.method``.

One interface for the three crop strategies that previously each had their own
caller:

* ``label_cc``   — semantic connected-component per-ROI crop (``label_cc.py``).
  One ``crop_rois_by_label`` call per frame.
* ``bbox_tight`` — prim-path bbox per-component crop (``bbox_tight.py``;
  single-flow good/defect/missing).
* ``region``     — clip a whole frame to one bbox (``region.py``).

Each strategy stays distinct internally; this only unifies the call surface.
``run_crop(cfg)`` maps a flat config dict onto the chosen strategy and returns
that strategy's result (stats dict, or ``None`` for ``bbox_tight``).
"""

from __future__ import annotations

from typing import Any

from bbox_grouping import component_key  # noqa: F401 (re-exported convenience)
from crop_defaults import (
    DEFAULT_CROPS,
    EDGE_SKIP,
    MIN_AREA,
    MIN_COVERAGE,
    MORPH_KERNEL,
    OFFSET,
    XFORM_DEPTH,
)

METHODS = ("label_cc", "bbox_tight", "region")


def run_crop(cfg: dict[str, Any]):
    """Dispatch a crop run by ``cfg['method']``.

    Args:
        cfg: flat config dict. Always needs ``method`` ∈ ``METHODS``; the rest
            of the keys depend on the method (see the per-method adapters).

    Returns:
        The chosen strategy's return value (label_cc / region return a stats
        dict; bbox_tight returns ``None`` and writes crops to disk).

    """
    method = cfg.get("method")
    if method == "label_cc":
        return _run_label_cc(cfg)
    if method == "bbox_tight":
        return _run_bbox_tight(cfg)
    if method == "region":
        return _run_region(cfg)
    raise ValueError(f"crop.method must be one of {METHODS}, got {method!r}")


def _run_bbox_tight(cfg: dict[str, Any]):
    from bbox_tight import crop_components

    # Accept both the single-flow yaml key ("types") and the CLI key ("crops").
    crop_types = cfg.get("crops", cfg.get("types"))
    return crop_components(
        input_dir=cfg["input"],
        output_dir=cfg["output"],
        offset=cfg.get("offset", OFFSET),
        class_filter=cfg.get("class_filter"),
        reference_dir=cfg.get("reference"),
        crop_types=crop_types if crop_types is not None else DEFAULT_CROPS,
        component_xform_depth=cfg.get("xform_depth", XFORM_DEPTH),
        min_coverage=cfg.get("min_coverage", MIN_COVERAGE),
    )


def _run_region(cfg: dict[str, Any]):
    from region import DEFAULT_OUTPUTS, crop_sdg

    return crop_sdg(
        src_dir=cfg["src_dir"],
        dst_dir=cfg["dst_dir"],
        bbox=cfg["bbox"],
        frame_idx=cfg.get("frame_idx", "0000"),
        xform_depth=cfg.get("xform_depth", XFORM_DEPTH),
        include_loose=cfg.get("include_loose", True),
        outputs=cfg.get("outputs", DEFAULT_OUTPUTS),
    )


def _run_label_cc(cfg: dict[str, Any]):
    from label_cc import crop_rois_by_label

    return crop_rois_by_label(
        aligned_dir=cfg["aligned_dir"],
        ref_crop_path=cfg["ref_crop_path"],
        real_crop_path=cfg["real_crop_path"],
        output_dir=cfg["output_dir"],
        target_classes=cfg["target_classes"],
        morph_kernel=cfg.get("morph_kernel", MORPH_KERNEL),
        min_area=cfg.get("min_area", MIN_AREA),
        max_area=cfg.get("max_area", None),
        offset=cfg.get("offset", OFFSET),
        edge_skip=cfg.get("edge_skip", EDGE_SKIP),
        min_coverage=cfg.get("min_coverage", MIN_COVERAGE),
        max_emit=cfg.get("max_emit", None),
        frame_idx=cfg.get("frame_idx", "0000"),
        class_dirs=cfg.get("class_dirs", None),
        fallback_dir=cfg.get("fallback_dir", "component"),
        class_emit_counter=cfg.get("class_emit_counter", None),
        multi_frame=cfg.get("multi_frame", False),
    )
