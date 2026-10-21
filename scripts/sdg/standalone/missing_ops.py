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

"""Select, hide, and restore missing components."""

from __future__ import annotations

import logging

import numpy as np
from defects_utils.selection import selection_count
from pxr import UsdGeom

try:
    from omni.replicator.core.functional import modify as rep_modify
except ImportError:  # Kit --exec init-timing: lazy sys.modules alias not yet set
    from omni.replicator.core.scripts.functional import modify as rep_modify

logger = logging.getLogger(__name__)


def select_missing_components(
    component_pool: list[str],
    missing_cfg: dict[str, object],
) -> list[str]:
    """Randomly select components to hide. Returns list of prim paths."""
    n = selection_count(
        missing_cfg,
        len(component_pool),
        operation_name="missing",
    )
    selected = np.random.choice(component_pool, size=n, replace=False).tolist()
    logger.info("Missing: %d components selected", n)
    for p in selected[:10]:
        logger.info("  -> %s", p.split("/")[-1])
    if len(selected) > 10:
        logger.info("  ... and %d more", len(selected) - 10)
    return selected


def apply_missing_semantics(stage: object, prim_paths: list[str]) -> None:
    """Apply semantic label defect=missing to selected components."""
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and not prim.HasAttribute("semantics:labels:defect"):
            rep_modify.semantics(prim, value={"defect": "missing"})
    logger.info("Applied defect=missing semantic to %d components", len(prim_paths))


def hide_components(stage: object, prim_paths: list[str]) -> None:
    """Make components invisible."""
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()


def restore_components(stage: object, prim_paths: list[str]) -> None:
    """Undo either hide, without the caller having to know which one ran.

    There are two hides -- :func:`hide_components` authors visibility and
    :func:`deactivate_components` prunes the subtree -- and callers that bracket
    a render pass with hide/restore should not have to track which was used.
    Reactivating first is harmless on a prim that was never deactivated, and
    ``MakeVisible`` on a reactivated prim is what actually brings it back, so
    doing both is the only form that is correct for both hides.

    Before this, restoring after a pruning hide silently did nothing: the prim
    is not traversable while inactive, so ``MakeVisible`` had nothing to act on
    and the component stayed gone for the rest of the run.
    """
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            if not prim.IsActive():
                prim.SetActive(True)
            UsdGeom.Imageable(prim).MakeVisible()


def deactivate_components(stage: object, prim_paths: list[str]) -> None:
    """Prune component subtrees so they are truly absent from the render.

    ``UsdGeom.Imageable.MakeInvisible`` (see :func:`hide_components`) authors USD
    visibility, which the Fabric Scene Delegate beauty pass does not reliably pick
    up when toggled mid-run (the segmentation annotator does, but the RGB pass
    reads Fabric). Deactivating the prim prunes it from both USD and Fabric, so it
    is genuinely gone from RGB, depth AND segmentation — the render-effective
    ``missing`` hide under FSD. Pair with :func:`activate_components` to restore.
    """
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            prim.SetActive(False)


def activate_components(stage: object, prim_paths: list[str]) -> None:
    """Re-activate component subtrees previously pruned by deactivate_components."""
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            prim.SetActive(True)


def build_writer_kwargs(writer_cfg: dict, output_dir: str) -> dict:
    """Build BasicWriter.initialize() kwargs from config dict."""
    kwargs: dict = {"output_dir": output_dir}
    for key in [
        "rgb",
        "image_output_format",
        "bounding_box_2d_tight",
        "bounding_box_2d_loose",
        "bounding_box_3d",
        "semantic_segmentation",
        "colorize_semantic_segmentation",
        "instance_id_segmentation",
        "colorize_instance_id_segmentation",
        "distance_to_camera",
        "distance_to_image_plane",
        "colorize_depth",
        "semantic_types",
        "frame_padding",
    ]:
        if key in writer_cfg:
            kwargs[key] = writer_cfg[key]
    return kwargs
