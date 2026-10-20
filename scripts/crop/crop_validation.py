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

"""Shared in-process crop validation rules (single source of truth).

Two entry points enforce the same two business rules — 'crop.method must be
bbox_tight' (SDGW-796 covers only the single-flow method) and 'crop types
must include a semantic_* entry or bbox_tight silently writes zero crops' —
and they used to carry independently-worded copies:

  * ``cdf/sections.py`` PostprocessSection (declarative job, fails at
    description time with ValidationError);
  * ``scripts/crop/pipeline_adapter.py`` (single-flow ``--stage crop``,
    fails at config time with ValueError).

Both now call the message builders here, so loosening or extending either rule
cannot silently diverge. A Python job and its YAML counterpart must resolve to
the same plan. This module is deliberately stdlib-only and exception-free;
callers wrap the returned message in their own error type.
"""

from __future__ import annotations

# The in-process crop methods the single-flow pipeline supports today.
CROP_METHODS = ("bbox_tight",)

# What makes a crop type "semantic" for coverage filtering, and the
# canonical default semantic type (crop_defaults.DEFAULT_CROPS and
# sections.py's omitted-types default must both include it — pinned by
# unit test).
SEMANTIC_MARKER = "semantic"
DEFAULT_SEMANTIC_TYPE = "semantic_segmentation"


def has_semantic_type(types_list) -> bool:
    """Check whether an entry is usable for semantic coverage.

    filtering (bbox_tight matches the substring, e.g. also a custom
    ``semantic_segmentation_colorized``).
    """
    return any(SEMANTIC_MARKER in str(t) for t in types_list)


def crop_method_error(method) -> str | None:
    """Return an error when ``method`` is not an in-process crop method.

    Return ``None`` when valid.
    """
    if method in CROP_METHODS:
        return None
    return (
        f"crop method must be one of {CROP_METHODS} (in-process crop constraint; "
        "ROI label_cc and registration region crops keep their own "
        f"orchestration paths), got {method!r}"
    )


def crop_types_error(types_list) -> str | None:
    """Return an error when ``types_list`` has no semantic entry.

    Return ``None`` when valid. bbox_tight drops every crop whose semantic coverage
    cannot be measured, so a list without a semantic type would silently
    produce ZERO crops — fail at config/description time instead.
    """
    if has_semantic_type(types_list):
        return None
    return (
        f"crop types {list(types_list)!r} has no semantic type: bbox_tight needs a "
        "semantic image for coverage filtering and would write zero crops. "
        f"Add {DEFAULT_SEMANTIC_TYPE!r} to the types (or omit them to use the defaults)."
    )
