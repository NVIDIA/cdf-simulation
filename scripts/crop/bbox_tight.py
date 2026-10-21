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

"""bbox-tight per-component crop (single-flow ``bbox_tight`` strategy).

Groups bbox entries by component Xform (using prim_paths), merges sub-mesh
bboxes into one per-component bbox, then crops the requested image types into
``<label_or_defect>/<crop_type>/frame<NNNN>_<comp>.png`` (missing mode: split
into ``missing/`` + ``ok/``).

Was ``scripts/postprocess/crop_components.py``; the CLI front-end stays there
as a thin shell that calls ``crop.facade.run_crop``.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
from PIL import Image

from bbox_grouping import component_key, merge_to_components
from crop_defaults import DEFAULT_CROPS, MIN_COVERAGE, OFFSET, XFORM_DEPTH
from crop_primitives import mask_coverage_fraction, offset_and_clamp, touches_edge

_FILE_UTILS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "file_utils")
)
if _FILE_UTILS_DIR not in sys.path:
    sys.path.insert(0, _FILE_UTILS_DIR)

from path_safety import confined_path, safe_path_segment  # noqa: E402

logger = logging.getLogger(__name__)

# Back-compat alias: external callers / docs referenced this name.
extract_component_key = component_key

# Back-compat names used by older security tests.
_safe_path_segment = safe_path_segment
_output_path = confined_path


def crop_components(
    input_dir: str,
    output_dir: str,
    offset: int = OFFSET,
    class_filter: list[str] | None = None,
    reference_dir: str | None = None,
    crop_types: tuple[str, ...] | list[str] | None = None,
    component_xform_depth: int = XFORM_DEPTH,
    min_coverage: float = MIN_COVERAGE,
) -> None:
    """Crop components from SDG output.

    Args:
        input_dir: trigger folder containing rgb (and bbox/seg if no reference_dir)
        output_dir: output folder for cropped images
        offset: extra pixels around bbox on each side (default 10)
        class_filter: if set, only crop components whose label contains this string
        reference_dir: if set, read bbox/labels/prim_paths/semantic from here,
                       rgb from input_dir (missing mode). Also crops reference rgb as "ok".
        crop_types: list of image type prefixes to crop (e.g. ["rgb", "semantic_segmentation"])
        min_coverage: drop crops whose semantic mask covers less than this
                      fraction of the crop box (default 0.2). Applied both at
                      crop time and as a post-filter over saved masks.
    """
    if crop_types is None:
        crop_types = DEFAULT_CROPS
    validated_crop_types: list[str] = []
    for crop_type in crop_types:
        crop_type = str(crop_type)
        if safe_path_segment(crop_type, "crop_type") != crop_type:
            raise ValueError(
                f"crop type must be a portable filename segment: {crop_type!r}"
            )
        validated_crop_types.append(crop_type)
    crop_types = tuple(validated_crop_types)

    # bbox/labels/semantic come from reference_dir if provided, otherwise input_dir
    anno_dir = reference_dir if reference_dir else input_dir

    bbox_files = sorted(
        [
            f
            for f in os.listdir(anno_dir)
            if f.startswith("bounding_box_2d_tight_") and f.endswith(".npy")
        ]
    )

    if not bbox_files:
        logger.error("No bounding_box_2d_tight_*.npy files found.")
        return

    # Determine which crop type to use for semantic coverage filtering
    sem_type: str | None = None
    for ct in crop_types:
        if "semantic" in ct:
            sem_type = ct
            break

    total_crops = 0
    created_dirs: set[str] = set()

    for bbox_file in bbox_files:
        frame_idx = bbox_file.replace("bounding_box_2d_tight_", "").replace(".npy", "")

        # Load bbox + prim_paths + labels (from anno_dir)
        bboxes = np.load(os.path.join(anno_dir, bbox_file))

        prim_paths_file = os.path.join(
            anno_dir, f"bounding_box_2d_tight_prim_paths_{frame_idx}.json"
        )
        if not os.path.exists(prim_paths_file):
            logger.warning("prim_paths not found for frame %s, skipping", frame_idx)
            continue
        with open(prim_paths_file) as f:
            prim_paths = json.load(f)

        label_file = os.path.join(anno_dir, f"bounding_box_2d_tight_labels_{frame_idx}.json")
        if not os.path.exists(label_file):
            continue
        with open(label_file) as f:
            labels = json.load(f)

        # Load images for each crop type
        # rgb always from input_dir, others from anno_dir
        # (closed in the finally below — bare Image.open handles otherwise
        # stay open until GC, which piles up over thousands of frames)
        images: dict[str, Image.Image | None] = {}
        ref_rgb_img: Image.Image | None = None
        try:
            for ct in crop_types:
                if ct == "rgb":
                    path = os.path.join(input_dir, f"{ct}_{frame_idx}.png")
                else:
                    path = os.path.join(anno_dir, f"{ct}_{frame_idx}.png")
                images[ct] = Image.open(path) if os.path.exists(path) else None

            # In missing mode, also load reference rgb for "ok" crops
            if reference_dir:
                ref_rgb_path = os.path.join(reference_dir, f"rgb_{frame_idx}.png")
                ref_rgb_img = Image.open(ref_rgb_path) if os.path.exists(ref_rgb_path) else None

            if images.get("rgb") is None:
                continue

            img_w, img_h = images["rgb"].size

            # Load semantic image for coverage filtering
            sem_img = images.get(sem_type) if sem_type else None
            if sem_img is None:
                # without a semantic image every component is skipped below --
                # make that loud instead of silently writing zero crops
                print(
                    f"[bbox_tight] WARNING: no semantic image for frame "
                    f"{frame_idx} (crop types: {crop_types}); coverage "
                    f"filtering is impossible and no crops will be written. "
                    f"Include a semantic type in crop.types."
                )

            # Group bboxes by component Xform (sid as str to key the labels JSON).
            components = merge_to_components(
                bboxes, prim_paths, component_xform_depth, sid_cast=str
            )

            # Crop each component
            frame_crops = 0
            for comp_key, c in components.items():
                label_info = labels.get(c["sid"], {})
                label_class = label_info.get("class", "")
                label_defect = label_info.get("defect", "")

                if class_filter:
                    match = any(f in label_class or f in label_defect for f in class_filter)
                    if not match:
                        continue

                # Skip components whose bbox touches the image edge (incomplete/cut-off)
                if touches_edge(c["x_min"], c["y_min"], c["x_max"], c["y_max"], img_w, img_h):
                    continue

                # Apply offset with boundary clamping (bbox-tight: no +1 on max edge)
                crop_box = offset_and_clamp(
                    c["x_min"], c["y_min"], c["x_max"], c["y_max"], offset, img_w, img_h,
                    inclusive_max=False,
                )

                # Build filename from component name
                comp_name = safe_path_segment(
                    comp_key.replace("\\", "/").split("/")[-1], "component"
                )
                safe_frame_idx = safe_path_segment(frame_idx, "frame")
                prefix = f"frame{safe_frame_idx}_{comp_name}"

                # Skip component if semantic coverage below threshold
                if sem_img is not None:
                    sem_arr = np.array(sem_img.crop(crop_box))
                    if mask_coverage_fraction(sem_arr) < min_coverage:
                        continue
                else:
                    continue

                if reference_dir:
                    # Missing mode: save defective rgb as "missing", reference rgb as "ok"
                    for folder, rgb_src in [("missing", images.get("rgb")), ("ok", ref_rgb_img)]:
                        if rgb_src is None:
                            continue
                        if folder not in created_dirs:
                            for ct in crop_types:
                                os.makedirs(confined_path(output_dir, folder, ct), exist_ok=True)
                            created_dirs.add(folder)
                        rgb_src.crop(crop_box).save(
                            confined_path(output_dir, folder, "rgb", f"{prefix}.png")
                        )
                        for ct in crop_types:
                            if ct == "rgb":
                                continue
                            img = images.get(ct)
                            if img is not None:
                                img.crop(crop_box).save(
                                    confined_path(output_dir, folder, ct, f"{prefix}.png")
                                )
                else:
                    # Normal mode: use label as folder name
                    label_folder = safe_path_segment(
                        label_defect if label_defect else label_class, "unlabeled"
                    )

                    if label_folder not in created_dirs:
                        for ct in crop_types:
                            os.makedirs(
                                confined_path(output_dir, label_folder, ct), exist_ok=True
                            )
                        created_dirs.add(label_folder)

                    for ct in crop_types:
                        img = images.get(ct)
                        if img is not None:
                            img.crop(crop_box).save(
                                confined_path(
                                    output_dir, label_folder, ct, f"{prefix}.png"
                                )
                            )

                frame_crops += 1

            total_crops += frame_crops
            if frame_crops > 0:
                logger.info("  Frame %s: %d components cropped", frame_idx, frame_crops)
        finally:
            for img in images.values():
                if img is not None:
                    img.close()
            if ref_rgb_img is not None:
                ref_rgb_img.close()

    logger.info("Total: %d crops -> %s", total_crops, output_dir)

    # Post-filter: remove crops whose saved semantic image is below threshold
    if sem_type is None:
        return

    removed = 0
    for root, dirs, files in os.walk(output_dir):
        if os.path.basename(root) != sem_type:
            continue
        for fname in files:
            sem_file = os.path.join(root, fname)
            with Image.open(sem_file) as sem_open:
                sem_arr = np.array(sem_open)
            if mask_coverage_fraction(sem_arr) < min_coverage:
                label_dir = os.path.dirname(root)
                for ct in crop_types:
                    p = confined_path(label_dir, ct, fname)
                    if os.path.exists(p):
                        os.remove(p)
                removed += 1
    if removed > 0:
        logger.info("Post-filter: Removed %d crops with empty semantic masks", removed)
