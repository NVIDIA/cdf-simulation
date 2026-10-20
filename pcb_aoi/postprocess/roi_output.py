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

"""Format per-trigger capture outputs to the PCB AOI ROI output contract."""

from __future__ import annotations

import json
import re
from pathlib import Path

from cdf.errors import ValidationError

_SEMASK_PATTERN = re.compile(r"^semantic_segmentation_(x(\d+)_y(\d+)\.png)$")
_LABELS_PATTERN = re.compile(r"^semantic_segmentation_labels_x\d+_y\d+\.json$")
_RGB_PATTERN = re.compile(r"^rgb_x(\d+)_y(\d+)\.png$")
_SEMASK_RENAMED_PATTERN = re.compile(r"^semantic_mask_x(\d+)_y(\d+)\.png$")

_ALLOWED_SEMANTIC_CLASSES: frozenset[str] = frozenset(
    {"BACKGROUND", "UNLABELLED", "pad", "hole", "pin", "pin_tip"}
)


def _rename_semantic_masks(trigger_dir: Path) -> set[tuple[int, int]]:
    """Rename ``semantic_segmentation_x*_y*.png`` to ``semantic_mask_x*_y*.png``.

    Returns the set of (x_idx, y_idx) pairs that were renamed.
    """
    renamed: set[tuple[int, int]] = set()
    for path in list(trigger_dir.iterdir()):
        match = _SEMASK_PATTERN.fullmatch(path.name)
        if match:
            path.rename(trigger_dir / f"semantic_mask_{match.group(1)}")
            renamed.add((int(match.group(2)), int(match.group(3))))
    return renamed


def _merge_semantic_labels(
    trigger_dir: Path,
    defect_classes: set[str],
) -> dict[str, object]:
    """Merge per-tile ``semantic_segmentation_labels_x*_y*.json`` into one table.

    Raises:
        ValidationError: If no label files are found, a file is unreadable, or
            two tiles assign different semantics to the same RGBA key.

    """
    merged: dict[str, object] = {}
    found_any = False
    for path in sorted(trigger_dir.iterdir()):
        if not _LABELS_PATTERN.fullmatch(path.name):
            continue
        found_any = True
        try:
            labels = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValidationError(f"could not read {path.name}: {exc}") from exc
        for rgba_key, entry in labels.items():
            existing = merged.get(rgba_key)
            if existing is not None and existing != entry:
                raise ValidationError(
                    f"conflicting semantic label for {rgba_key!r} between tiles "
                    f"in {trigger_dir}: {existing!r} vs {entry!r}",
                )
            merged[rgba_key] = entry
    if not found_any:
        raise ValidationError(
            f"no semantic_segmentation_labels_*.json files found in {trigger_dir}",
        )
    allowed_classes = _ALLOWED_SEMANTIC_CLASSES | defect_classes
    return {
        k: v
        for k, v in merged.items()
        if isinstance(v, dict) and v.get("class") in allowed_classes
    }


def _grid_indices(trigger_dir: Path, pattern: re.Pattern) -> set[tuple[int, int]]:
    """Return (x_idx, y_idx) pairs for files matching a compiled pattern."""
    indices: set[tuple[int, int]] = set()
    for path in trigger_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            indices.add((int(match.group(1)), int(match.group(2))))
    return indices


def _check_patch_dimensions(
    rgb_dir: Path,
    roi_dir: Path,
    indices: set[tuple[int, int]],
) -> None:
    """Raise ValidationError if any RGB and ROI mask tile differ in size."""
    try:
        from PIL import Image
    except ImportError:
        return
    for x, y in sorted(indices):
        rgb_path = rgb_dir / f"rgb_x{x}_y{y}.png"
        mask_path = roi_dir / f"semantic_mask_x{x}_y{y}.png"
        if not rgb_path.is_file() or not mask_path.is_file():
            continue
        with Image.open(rgb_path) as img:
            rgb_size = img.size
        with Image.open(mask_path) as img:
            mask_size = img.size
        if rgb_size != mask_size:
            raise ValidationError(
                f"dimension mismatch at tile ({x},{y}): "
                f"rgb={rgb_size} vs semantic_mask={mask_size}",
            )


def format_aoi_roi_output(
    rgb_root: str | Path,
    roi_root: str | Path,
    trigger_metadata_rgb: list[dict[str, object]],
    trigger_metadata_roi: list[dict[str, object]],
) -> dict[str, object]:
    """Format per-trigger capture outputs to the AOI ROI output contract.

    For each trigger:

    - Validates the canonical pixel layout held by the run manifest without
      replacing either pass's schema-rich ``metadata.json``.
    - Renames ``semantic_segmentation_x*_y*.png`` to ``semantic_mask_x*_y*.png``
      in the ROI trigger directory.
    - Merges per-tile ``semantic_segmentation_labels_x*_y*.json`` into a single
      ``semantic_segmentation_labels.json``, retaining the five required
      geometry classes, optional ``pin_tip``, and defect classes declared by
      the manifest.
    - Runs acceptance checks: RGB and ROI patch counts match, grid indices match,
      and corresponding tile dimensions agree.

    **Capture-pass requirement**: both capture passes must be configured with
    ``rename_to_grid_index: true`` in their ``outputs`` block.  Every file
    pattern used here (``rgb_x*_y*.png``, ``semantic_segmentation_x*_y*.png``,
    ``semantic_segmentation_labels_x*_y*.json``) depends on the
    ``x<col>_y<row>`` suffix that setting produces.

    Args:
        rgb_root: Local root of the RGB capture pass.
        roi_root: Local root of the ROI capture pass.
        trigger_metadata_rgb: Per-trigger metadata returned by the RGB capture order.
        trigger_metadata_roi: Per-trigger metadata returned by the ROI capture order.

    Returns:
        Dictionary containing acceptance counts and derived artifact paths.

    Raises:
        ValidationError: If trigger counts differ, pixel layouts are missing, patch
            counts or grid indices mismatch, or tile dimensions disagree.

    """
    rgb_root = Path(rgb_root)
    roi_root = Path(roi_root)

    if len(trigger_metadata_rgb) != len(trigger_metadata_roi):
        raise ValidationError(
            f"RGB pass has {len(trigger_metadata_rgb)} trigger(s) but ROI pass "
            f"has {len(trigger_metadata_roi)}; counts must match",
        )

    total_patches = 0
    path_mapping: dict[str, str] = {}
    artifacts: list[dict[str, object]] = []
    for meta_rgb, meta_roi in zip(trigger_metadata_rgb, trigger_metadata_roi):
        trigger_idx = int(meta_rgb["trigger_idx"])
        rgb_trigger_dir = rgb_root / f"trigger_{trigger_idx:04d}"
        roi_trigger_dir = roi_root / f"trigger_{trigger_idx:04d}"

        for meta in (meta_rgb, meta_roi):
            pixel_layout = meta.get("pixel_layout")
            if not isinstance(pixel_layout, dict) or not pixel_layout.get("supported"):
                reason = (
                    pixel_layout.get("reason")
                    if isinstance(pixel_layout, dict)
                    else None
                )
                raise ValidationError(
                    f"trigger {trigger_idx}: pixel_layout is missing or unsupported"
                    + (f": {reason}" if reason else ""),
                )

        renamed = _rename_semantic_masks(roi_trigger_dir)
        for x_idx, y_idx in sorted(renamed):
            old_path = (
                f"trigger_{trigger_idx:04d}/semantic_segmentation_x{x_idx}_y{y_idx}.png"
            )
            new_path = f"trigger_{trigger_idx:04d}/semantic_mask_x{x_idx}_y{y_idx}.png"
            path_mapping[old_path] = new_path
            artifacts.append(
                {
                    "kind": "semantic_mask",
                    "path": new_path,
                    "trigger_idx": trigger_idx,
                    "x_idx": x_idx,
                    "y_idx": y_idx,
                }
            )

        defects = meta_roi.get("defects")
        defect_classes = (
            {
                str(defect.get("type") or defect.get("defect"))
                for defect in defects
                if isinstance(defect, dict)
            }
            if isinstance(defects, list)
            else set()
        )
        label_table = _merge_semantic_labels(roi_trigger_dir, defect_classes)
        (roi_trigger_dir / "semantic_segmentation_labels.json").write_text(
            json.dumps(label_table, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts.append(
            {
                "kind": "semantic_labels",
                "path": (
                    f"trigger_{trigger_idx:04d}/semantic_segmentation_labels.json"
                ),
                "trigger_idx": trigger_idx,
            }
        )

        rgb_indices = _grid_indices(rgb_trigger_dir, _RGB_PATTERN)
        roi_indices = _grid_indices(roi_trigger_dir, _SEMASK_RENAMED_PATTERN)
        if rgb_indices != roi_indices:
            missing = sorted(rgb_indices - roi_indices)
            extra = sorted(roi_indices - rgb_indices)
            parts = []
            if missing:
                parts.append(f"missing ROI masks for tiles {missing}")
            if extra:
                parts.append(f"extra ROI masks for tiles {extra}")
            raise ValidationError(
                f"trigger {trigger_idx}: RGB/ROI grid index mismatch — "
                + "; ".join(parts),
            )

        _check_patch_dimensions(rgb_trigger_dir, roi_trigger_dir, rgb_indices)
        total_patches += len(rgb_indices)

    return {
        "trigger_count": len(trigger_metadata_rgb),
        "total_patches": total_patches,
        "accepted": True,
        "artifacts": artifacts,
        "path_mapping": path_mapping,
    }
