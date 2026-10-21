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

"""Define defaults shared by crop helpers and CDF postprocessing.

Single source of truth for the knobs that have a sensible fallback when a
``crop:`` key is left unspecified. The crop facade and CDF pipeline adapter
share these values with ``scripts/postprocess/crop_components.py``. The CLI
loads this module by file path so offsets, coverage, hierarchy depth, and
output types stay consistent without duplicating literals.
"""

from __future__ import annotations

# Padding (px) added around each ROI/component bbox before cropping.
OFFSET = 10

# Minimum labelled / semantic pixel fraction inside a crop box; crops below
# this are dropped as mostly-blank board. 0 disables the filter.
MIN_COVERAGE = 0.2

# Label connected-component crop (`label_cc.crop_rois_by_label`) knobs.
MORPH_KERNEL = 2
MIN_AREA = 50
EDGE_SKIP = True

# bbox-tight crop (crop_components) knobs.
XFORM_DEPTH = 7
DEFAULT_CROPS = ("rgb", "semantic_segmentation", "instance_id_segmentation")
