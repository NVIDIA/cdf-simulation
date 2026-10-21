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

"""Attribute the time inside one JobOrder handler to named sub-phases.

Per-order timing answers *which* order is expensive. For a handler that holds
most of a run it does not answer the question that decides what to optimize:
how much of that cost is paid once per process -- and is therefore amortized by
a persistent session -- against how much is paid per frame, which is not.

A handler reports its phases in its own output mapping, so nothing about the
result contract changes:

.. code-block:: python

    timer = PhaseTimer()
    with timer.phase("warmup"):
        ...
    return {"output_dir": path, "phases": timer.report()}
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter


class PhaseTimer:
    """Accumulate wall-clock per named phase within one handler.

    A phase entered more than once sums its durations and counts its calls:
    a warm-up run once per pass and a warm-up run once per process cost the
    same on paper and mean opposite things, and the call count is what tells
    them apart.

    Args:
        clock: Monotonic seconds source. Injected so tests need no real time.

    Example:

    .. code-block:: python

        timer = PhaseTimer()
        with timer.phase("setup"):
            pass
        print(timer.report()["setup"]["calls"])
        # 1

    """

    def __init__(self, clock: Callable[[], float] = perf_counter) -> None:
        self._clock = clock
        self._seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure the enclosed block and record it under ``name``.

        The duration is recorded even when the block raises, so a run that
        fails still reports how far it got.

        Args:
            name: Phase label. Reusing a label accumulates into it.

        Yields:
            ``None``; the block runs inside the measurement.

        Example:

        .. code-block:: python

            timer = PhaseTimer()
            with timer.phase("setup"):
                pass
            print(timer.total_s >= 0.0)
            # True

        """
        started = self._clock()
        try:
            yield
        finally:
            elapsed = self._clock() - started
            self._seconds[name] = self._seconds.get(name, 0.0) + elapsed
            self._calls[name] = self._calls.get(name, 0) + 1

    @property
    def total_s(self) -> float:
        """Return the summed seconds across every recorded phase.

        Overlapping phases are counted once each, so a nested pair sums to more
        than the wall-clock it spans.

        Returns:
            Total recorded seconds.

        Example:

        .. code-block:: python

            print(PhaseTimer().total_s)
            # 0.0

        """
        return sum(self._seconds.values())

    def report(self) -> dict[str, dict[str, float]]:
        """Return the phases, costliest first.

        Returns:
            Mapping from phase name to its ``seconds`` and ``calls``.

        Example:

        .. code-block:: python

            print(PhaseTimer().report())
            # {}

        """
        ordered = sorted(self._seconds.items(), key=lambda item: item[1], reverse=True)
        return {
            name: {"seconds": round(seconds, 3), "calls": self._calls[name]}
            for name, seconds in ordered
        }
