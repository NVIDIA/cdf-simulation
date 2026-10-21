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

"""Partition explicitly requested reference designators without widening scope."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RefdesSelection[Item]:
    """Selected records and the requested reference designators they satisfy."""

    selected: tuple[Item, ...]
    requested: tuple[str, ...]
    matched: tuple[str, ...]
    unmatched: tuple[str, ...]


def select_refdes[Item](
    items: Iterable[Item],
    requested: Sequence[str] | None,
    *,
    key: Callable[[Item], str],
) -> RefdesSelection[Item]:
    """Select records and account for every explicitly requested refdes.

    Args:
        items: Candidate records.
        requested: Explicit refdes filter, or None to select every record.
        key: Function returning one record's refdes.

    Returns:
        Selection with requested, matched, and unmatched refdes partitions.

    Example:

    .. code-block:: python

        selection = select_refdes(
            ["X204", "X205"],
            ["X205", "X999"],
            key=str,
        )
        print(selection.unmatched)
        # ('X999',)

    """
    candidates = tuple(items)
    if requested is None:
        matched = tuple(dict.fromkeys(key(item) for item in candidates))
        return RefdesSelection(candidates, (), matched, ())

    requested_items = tuple(requested)
    available = frozenset(key(item) for item in candidates)
    matched = tuple(refdes for refdes in requested_items if refdes in available)
    unmatched = tuple(refdes for refdes in requested_items if refdes not in available)
    allowed = frozenset(matched)
    selected = tuple(item for item in candidates if key(item) in allowed)
    return RefdesSelection(selected, requested_items, matched, unmatched)
