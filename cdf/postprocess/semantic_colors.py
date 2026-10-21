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

"""Provide stable colors for domain-neutral semantic label mappings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def semantic_color(semantics: Mapping[str, str]) -> tuple[int, int, int, int]:
    """Return a stable opaque color for one semantic-label combination.

    Args:
        semantics: Semantic type-to-label mapping.

    Returns:
        Deterministic RGBA byte tuple.

    Example:
        >>> semantic_color({"class": "example"}) == semantic_color(
        ...     {"class": "example"}
        ... )
        True

    """
    canonical = json.dumps(dict(sorted(semantics.items())), separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    color = tuple(max(32, channel) for channel in digest[:3])
    return color[0], color[1], color[2], 255
