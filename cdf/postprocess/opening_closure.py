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

"""Close ring-shaped segmentation labels into the opening they enclose."""

from __future__ import annotations

import json
from pathlib import Path

from ..errors import ValidationError


def _class_identifier(labels_path: Path, class_name: str) -> int | None:
    """Return the identifier a frame gave one class, or None if absent."""
    try:
        table = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValidationError(
            f"could not read segmentation labels {labels_path}: {error}",
        ) from error
    for identifier, semantics in table.items():
        if str(semantics.get("class")) == class_name:
            return int(identifier)
    return None


def _unlabelled_identifier(labels_path: Path) -> int | None:
    """Return the identifier for unlabelled pixels, or None if absent."""
    return _class_identifier(labels_path, "UNLABELLED")


def close_openings(
    mask_path: str | Path,
    *,
    class_name: str,
    max_diameter_px: int = 48,
    min_circularity: float = 0.55,
) -> dict[str, object]:
    """Fill each closed ring of one class with that same class.

    A drilled hole is a void: the barrel is a thin-walled cylinder, so viewed
    down the axis only its rim has any projected area and the opening itself
    stays unlabelled. Filling the rim recovers the opening from geometry that
    was actually rendered, rather than drawing a circle from nominal metadata.

    Only pixels that are currently unlabelled are claimed, so a pin sitting in
    the hole keeps its own class. A ring that does not close, or whose fill is
    too large or too ragged to be an opening, is left alone rather than guessed
    at -- an unclosed rim usually means the opening was partly occluded.

    This runs on a stitched mask rather than on the individual scan frames. A
    hole lying across a tile boundary has an incomplete rim in every frame that
    sees only part of it, so closing before stitching recovers a fraction of the
    openings and leaves mismatched fragments where tiles overlap.

    Args:
        mask_path: Stitched segmentation image to rewrite in place.
        class_name: Class whose rings are closed.
        max_diameter_px: Largest opening accepted, in pixels.
        min_circularity: Least filled-area-to-enclosing-disc ratio accepted.

    Returns:
        Counts of openings closed and pixels added.

    Raises:
        ValidationError: If NumPy or SciPy is unavailable, or the mask or its
            label sidecar cannot be read.

    Example:

    .. code-block:: python

        report = close_openings("output/pass/stitched/mask.png", class_name="hole")
        print(report["openings_closed"])

    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as error:
        raise ValidationError(
            "closing segmentation openings requires NumPy, Pillow and SciPy",
        ) from error

    path = Path(mask_path)
    if not path.is_file():
        raise ValidationError(f"stitched mask {path} does not exist")
    labels_path = path.with_name(f"{path.stem}_labels.json")
    if not labels_path.is_file():
        raise ValidationError(
            f"closing openings requires {labels_path.name} beside {path.name}",
        )

    identifier = _class_identifier(labels_path, class_name)
    unlabelled = _unlabelled_identifier(labels_path)
    if identifier is None or unlabelled is None:
        return {"class": class_name, "openings_closed": 0, "pixels_added": 0}

    with Image.open(path) as image:
        mask = np.asarray(image).copy()
    rings = mask == identifier
    if not rings.any():
        return {"class": class_name, "openings_closed": 0, "pixels_added": 0}

    filled = ndimage.binary_fill_holes(rings)
    interior = filled & ~rings & (mask == unlabelled)
    components, _ = ndimage.label(interior)
    accepted = np.zeros_like(interior)
    closed_total = 0
    for index, window in enumerate(ndimage.find_objects(components), 1):
        if window is None:
            continue
        blob = components[window] == index
        height = window[0].stop - window[0].start
        width = window[1].stop - window[1].start
        if height > max_diameter_px or width > max_diameter_px:
            continue
        area = int(blob.sum())
        enclosing = np.pi * (max(height, width) / 2.0) ** 2
        if area / enclosing < min_circularity:
            continue
        accepted[window][blob] = True
        closed_total += 1

    added_total = int(accepted.sum())
    if closed_total:
        mask[accepted] = identifier
        Image.fromarray(mask).save(path)

    return {
        "class": class_name,
        "openings_closed": closed_total,
        "pixels_added": added_total,
    }
