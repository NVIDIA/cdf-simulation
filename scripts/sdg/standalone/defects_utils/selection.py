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

"""Resolve exact-count or proportional defect selection requests."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real


def describe_targeting(config: Mapping[str, object]) -> str:
    """Describe what an operation ASKED for, in whichever mode it used.

    ``count`` and ``ratio`` are mutually exclusive (see
    :func:`selection_count`), so a report that reads only ``config["ratio"]``
    raises ``KeyError`` on the count path. That is worse than a bad log
    line: in ``prepare_defects`` the message is emitted after the transforms
    have been authored, so the stage is already changed when the run is
    reported as failed.

    The request is worth saying out loud even when it succeeded, because it
    is not always what was delivered -- a ratio is clamped to the eligible
    population, so "0.5% requested" beside "2 components" is the honest
    pair.

    Args:
        config: One operation's configuration.

    Returns:
        A short phrase naming the targeting mode and the value requested.

    Example:

    .. code-block:: python

        print(describe_targeting({"count": 3}))
        # count=3 requested

    """
    count = config.get("count")
    if count is not None:
        return f"count={count} requested"
    ratio = config.get("ratio")
    if isinstance(ratio, Real) and not isinstance(ratio, bool):
        return f"{float(ratio) * 100:.1f}% requested"
    return "no targeting given"


def selection_count(
    config: Mapping[str, object],
    eligible_count: int,
    *,
    ratio_base_count: int | None = None,
    operation_name: str = "defect",
) -> int:
    """Return the validated number of components to select.

    Exact ``count`` requests fail when insufficient eligible components remain.
    Ratio requests preserve the existing behavior of selecting at least one
    component and clamping the result to the eligible population.

    Args:
        config: Operation configuration containing exactly one targeting mode.
        eligible_count: Number of components currently eligible for selection.
        ratio_base_count: Population used for proportional selection. Pass None
            to use ``eligible_count``.
        operation_name: Operation label used in validation messages.

    Returns:
        Number of eligible components to select.

    Raises:
        ValueError: If targeting is invalid or an exact count cannot be met.
        TypeError: If a ratio has an invalid type.

    Example:

    .. code-block:: python

        print(selection_count({"count": 2}, 10, operation_name="shift"))
        # 2

    """
    count = config.get("count")
    ratio = config.get("ratio")
    if (count is None) == (ratio is None):
        raise ValueError(
            f"{operation_name} requires exactly one of 'count' or 'ratio'",
        )
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{operation_name}.count must be an integer >= 1")
        if count > eligible_count:
            raise ValueError(
                f"{operation_name} requested {count} component(s), but only "
                f"{eligible_count} are eligible",
            )
        return count

    if isinstance(ratio, bool) or not isinstance(ratio, Real):
        raise TypeError(f"{operation_name}.ratio must be numeric")
    normalized_ratio = float(ratio)
    if not 0.0 <= normalized_ratio <= 1.0:
        raise ValueError(f"{operation_name}.ratio must be within [0, 1]")
    base_count = eligible_count if ratio_base_count is None else ratio_base_count
    selected_count = max(1, int(base_count * normalized_ratio))
    return min(selected_count, eligible_count)
