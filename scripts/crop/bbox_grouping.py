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

"""Group per-mesh 2D bboxes into per-component bboxes by USD prim path.

Shared by the two bbox-based crop paths:
* ``scripts/postprocess/crop_components.py`` (bbox-tight per-component crop).
* ``sdg_crop.py`` (region clip; rebases bbox annotations).

Both previously carried their own identical ``component_key`` + merge loop.
The only real difference was the stored ``semanticId`` type (``str`` for
crop_components, which keys into the labels JSON by string; ``int`` for
sdg_crop), preserved here via ``sid_cast``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np


def component_key(prim_path: str, xform_depth: int) -> str:
    """Truncate a full prim path to its component Xform path.

    Joins the first ``xform_depth`` path segments; returns the full path
    unchanged when it is shorter than ``xform_depth``.
    """
    parts = prim_path.split("/")
    if len(parts) >= xform_depth:
        return "/".join(parts[:xform_depth])
    return prim_path


def merge_to_components(
    bboxes: np.ndarray,
    prim_paths: list[str],
    xform_depth: int,
    *,
    sid_cast: Callable[[int], Any] = str,
) -> dict[str, dict[str, Any]]:
    """Merge per-mesh tight bboxes into one bbox per component Xform.

    Args:
        bboxes: structured array of bbox entries (``x_min/y_min/x_max/y_max/
            semanticId`` fields), as loaded from ``bounding_box_2d_*_*.npy``.
        prim_paths: per-entry prim paths (parallel to ``bboxes``).
        xform_depth: hierarchy depth at which to truncate prim paths.
        sid_cast: applied to the (int) semanticId before storing — ``str`` to
            key into a labels JSON by string (crop_components), ``int`` to keep
            it numeric (sdg_crop).

    Returns:
        ``{component_key: {x_min, y_min, x_max, y_max, sid}}``. The sentinel
        seeds (``1e9`` / ``-1``) are always overwritten because a group is only
        created when at least one entry maps to it.
    """
    components: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"x_min": 10**9, "y_min": 10**9, "x_max": -1, "y_max": -1, "sid": None}
    )
    for entry, ppath in zip(bboxes, prim_paths):
        key = component_key(ppath, xform_depth)
        c = components[key]
        c["x_min"] = min(c["x_min"], int(entry["x_min"]))
        c["y_min"] = min(c["y_min"], int(entry["y_min"]))
        c["x_max"] = max(c["x_max"], int(entry["x_max"]))
        c["y_max"] = max(c["y_max"], int(entry["y_max"]))
        c["sid"] = sid_cast(int(entry["semanticId"]))
    return components
