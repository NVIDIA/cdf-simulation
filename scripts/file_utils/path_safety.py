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

"""Portable path-segment sanitization and output-root containment."""

from __future__ import annotations

import os
import re
import unicodedata

_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_path_segment(value: object, fallback: str) -> str:
    """Return an untrusted value as one safe output-path segment.

    Args:
        value: Value to sanitize.
        fallback: Segment to use when sanitization produces an empty string.

    Returns:
        Sanitized path segment.

    Example:

    .. code-block:: python

        print(safe_path_segment("../../board", "component"))
        board
    """
    normalized = unicodedata.normalize("NFKC", str(value))
    segment = re.sub(r"[^\w-]+", "_", normalized, flags=re.UNICODE).strip("_")
    if not segment:
        segment = fallback
    if os.name == "nt" and segment.upper() in _WINDOWS_RESERVED_NAMES:
        segment = f"_{segment}"
    return segment


def confined_path(root: str, *parts: object) -> str:
    """Join path parts beneath a root and reject any resolved escape.

    Args:
        root: Directory that must contain the resulting path.
        *parts: Path components to append to the root.

    Returns:
        Resolved path contained beneath the root.

    Raises:
        ValueError: If the resulting path is outside the root.

    Example:

    .. code-block:: python

        output = confined_path("output", "images", "board.png")
        print(os.path.basename(output))
        board.png
    """
    output_root = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(output_root, *(os.fspath(part) for part in parts)))
    try:
        contained = os.path.commonpath([output_root, candidate]) == output_root
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"Refusing path outside configured root: {candidate}")
    return candidate
