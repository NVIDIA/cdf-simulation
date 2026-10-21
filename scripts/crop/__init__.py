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

"""Crop library — one home for all three crop strategies.

Public API (when imported as a package, e.g. ``from crop import run_crop``):
* ``run_crop(cfg)``          — single dispatch entry (``crop.facade``).
* ``crop_rois_by_label``     — label connected-component crop (``crop.label_cc``).
* ``crop_bridge_pairs``      — bridge-pair crop (``crop.label_cc``).
* ``crop_sdg``               — region clip (``crop.region``).
* ``crop_components``        — bbox-tight per-component crop (``crop.bbox_tight``).

Entry points that run modules as scripts (cwd-relative, no package) add this
directory to ``sys.path`` and import the submodules flat instead. The
submodules use flat imports (``from bbox_grouping import ...``) so they work in
that script mode; this ``__init__`` puts the package dir on ``sys.path`` first
so those same flat imports also resolve when imported as ``from crop import``.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from bbox_tight import crop_components
from facade import run_crop
from label_cc import crop_bridge_pairs, crop_rois_by_label
from region import crop_sdg

__all__ = [
    "run_crop",
    "crop_rois_by_label",
    "crop_bridge_pairs",
    "crop_sdg",
    "crop_components",
]
