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

"""Leaf operations shared by the crop pipelines.

These are deliberately tiny, side-effect-light helpers extracted verbatim from
the three crop paths so they stop drifting. They do NOT change behaviour — each
caller keeps its exact box / coverage semantics by passing the right flag.

Consumers:
* ``component_crop.py`` (label connected-component crop) — same package.
* ``sdg_crop.py`` (region clip) — same package.
* ``scripts/postprocess/crop_components.py`` (bbox-tight crop) — different
  package; loads this module by file path (see that file's import shim).
"""

from __future__ import annotations

import numpy as np


def offset_and_clamp(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    offset: int,
    w: int,
    h: int,
    *,
    inclusive_max: bool,
) -> tuple[int, int, int, int]:
    """Pad a tight bbox by ``offset`` and clamp to the image.

    ``inclusive_max`` is load-bearing and the two callers differ on it:

    * ``True``  — label-CC crop (``component_crop``): the tight bbox is an
      inclusive pixel index, so the right/bottom edge gets ``+1`` before the
      pad (``min(w, x_max + 1 + offset)``).
    * ``False`` — bbox-tight crop (``crop_components``): no ``+1``
      (``min(w, x_max + offset)``).

    Changing this flips byte output, so it is never reconciled — only passed.
    """
    x1 = max(0, x_min - offset)
    y1 = max(0, y_min - offset)
    extra = 1 if inclusive_max else 0
    x2 = min(w, x_max + extra + offset)
    y2 = min(h, y_max + extra + offset)
    return x1, y1, x2, y2


def touches_edge(x_min: int, y_min: int, x_max: int, y_max: int, w: int, h: int) -> bool:
    """True if the tight bbox touches any image edge (partial / cut-off)."""
    return x_min <= 0 or y_min <= 0 or x_max >= w - 1 or y_max >= h - 1


def mask_coverage_fraction(arr: np.ndarray) -> float:
    """Fraction of non-empty pixels in a semantic-mask crop array (0..1).

    Matches ``crop_components``: count the alpha channel when RGBA, else the
    per-pixel channel max; divide by pixel count. Returns 0.0 for an empty
    array. NOTE: this is the *saved-image* coverage used by the bbox-tight
    path; the label-CC path measures a different domain (labelled-mask array)
    and does not use this helper.
    """
    h, w = arr.shape[0], arr.shape[1]
    if h == 0 or w == 0:
        return 0.0
    # Normalize on dimensionality BEFORE looking at channel counts: a true
    # 2-D mask (L/P-mode PNG) has arr.shape[-1] == image width, so the old
    # channel branch either raised numpy.AxisError (arr.max(axis=2) on a
    # 2-D array) or IndexError (a 2-D array whose width happens to be 4).
    if arr.ndim == 2:
        nonzero = np.count_nonzero(arr)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        nonzero = np.count_nonzero(arr[:, :, 3])
    else:
        nonzero = np.count_nonzero(arr.max(axis=2))
    return float(nonzero) / float(h * w)


def save_crop(img, box: tuple[int, int, int, int], dst_path: str) -> None:
    """Crop a PIL image to ``box`` and save to ``dst_path``."""
    img.crop(box).save(dst_path)
