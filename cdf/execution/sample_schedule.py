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

"""Executor-owned sample schedule for one-component-per-scene capture runs.

This kind of run photographs one focused component per scene: an operation is
applied to it, one or more bound views are captured, the result is checked
against an acceptance policy, and the stage is restored before the next
sample. The *executor*, not a configuration field, owns the sample index::

    for each scene or trigger:
      choose the sampling stratum
      apply the operation
      capture
      restore

This module is that owner. It is deliberately free of Kit, USD, Replicator,
NumPy, and domain vocabulary so the sampling and acceptance decisions can be
unit tested directly, and so every surface that needs them -- the JobOrder
runtime and the standalone capture script -- shares one implementation
instead of keeping private copies.

Sampling is ``"stratified"`` and fully deterministic: operations are cycled
round-robin, and component candidates are ordered by cycling groups first and
instances within a group second, so a family with many placements on the board
cannot dominate a finite run. Every decision follows from ``index_base`` plus
the sample number, which is what makes a run reproducible, resumable, and
splittable across machines.
"""

from __future__ import annotations

import numbers

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ..errors import ValidationError

SAMPLING_POLICIES = ("stratified",)
#: Supported dataset sampling policies.

TARGET_OFFSET_MODES = ("balanced_disk",)
#: Supported camera target-offset sampling modes.

_GOLDEN_ANGLE_FACTOR = 3.0 - math.sqrt(5.0)
#: Golden-angle factor applied after pi to space successive disk directions.

_ATTEMPT_STRIDE = 997
#: Sequence stride applied per retry so a retry lands on a different offset.


def default_group_key(candidate: str) -> str:
    """Return the balancing group of a candidate prim path.

    The parent prim of a component instance names its package family, so the
    default grouping balances families before instances.

    Args:
        candidate: Absolute prim path of one component instance.

    Returns:
        Group name used to balance the candidate order.

    Example:

    .. code-block:: python

        print(default_group_key("/Board/Top/PASSIVE_SMD_0402/R12"))
        # PASSIVE_SMD_0402

    """
    trimmed = candidate.rstrip("/")
    parts = trimmed.rsplit("/", 2)
    if len(parts) < 2:
        return trimmed
    return parts[-2]


@dataclass(frozen=True)
class TargetOffsetPolicy:
    """How far the camera aim point is displaced from a component centroid.

    Displacing the aim point produces the off-center framing an inspection
    camera really sees. ``balanced_disk`` walks a golden-angle sequence with
    stratified radii so a finite run covers direction and area evenly instead
    of clumping the way independent random draws do.

    Args:
        mode: Sampling mode from :data:`TARGET_OFFSET_MODES`.
        min_mm: Smallest displacement radius in millimeters.
        max_mm: Largest displacement radius in millimeters.
        radial_bins: Stratification bins used by the ``balanced_disk`` mode.

    Raises:
        ValidationError: If the mode is unknown or a bound is invalid.

    """

    mode: str = "balanced_disk"
    #: Sampling mode from :data:`TARGET_OFFSET_MODES`.

    min_mm: float = 0.0
    #: Smallest displacement radius in millimeters.

    max_mm: float = 0.75
    #: Largest displacement radius in millimeters.

    radial_bins: int = 4
    #: Stratification bins used by the ``balanced_disk`` mode.

    def __post_init__(self) -> None:
        """Reject unusable offset configuration at construction time.

        Raises:
            ValidationError: If the mode is unknown or a bound is invalid.

        """
        if self.mode not in TARGET_OFFSET_MODES:
            raise ValidationError(
                "target offset mode must be one of %s, got %r"
                % (list(TARGET_OFFSET_MODES), self.mode),
            )
        if self.min_mm < 0.0 or self.max_mm < 0.0:
            raise ValidationError("target offset radii must not be negative")
        if self.min_mm > self.max_mm:
            raise ValidationError(
                "target offset min_mm (%s) must not exceed max_mm (%s)"
                % (self.min_mm, self.max_mm),
            )
        # A float bin count makes `index % radial_bins` non-integral, so
        # area_fraction can exceed 1 and sample() returns an offset outside
        # max_mm -- a silently out-of-range camera rather than a rejected config.
        if isinstance(self.radial_bins, bool) or not isinstance(
                self.radial_bins, numbers.Integral):
            raise ValidationError(
                "target offset radial_bins must be an integer, got %r"
                % (self.radial_bins,),
            )
        if self.radial_bins < 1:
            raise ValidationError("target offset radial_bins must be >= 1")

    def sample(
        self,
        sequence_index: int,
        *,
        attempt: int = 1,
    ) -> tuple[float, float]:
        """Return the millimeter ``(x, y)`` offset for one capture attempt.

        Args:
            sequence_index: Executor-owned index of the sample being framed.
            attempt: One-based acceptance attempt for this sample.

        Returns:
            Millimeter displacement applied to the component centroid.

        Example:

        .. code-block:: python

            policy = TargetOffsetPolicy(max_mm=1.0)
            x, y = policy.sample(0)
            print(round(x * x + y * y, 6) <= 1.0)
            # True

        """
        index = max(0, int(sequence_index)) + max(0, int(attempt) - 1) * _ATTEMPT_STRIDE
        area_fraction = (index % self.radial_bins + 0.5) / self.radial_bins
        radius = math.sqrt(
            self.min_mm * self.min_mm
            + area_fraction * (self.max_mm * self.max_mm - self.min_mm * self.min_mm),
        )
        theta = index * math.pi * _GOLDEN_ANGLE_FACTOR
        return radius * math.cos(theta), radius * math.sin(theta)


@dataclass(frozen=True)
class SampleSchedule:
    """Executor-owned sample index for one capture run.

    The schedule is the single place that decides, for sample *n*, which
    operation runs, which components are tried in which order, which
    deterministic operation variant is used, and where the camera aims. Every
    decision derives from ``index_base`` plus the sample index, so a run is
    reproducible and a resumed or sharded run continues the same coverage
    instead of repeating the start of the sequence.

    Args:
        operations: Operation names cycled across samples, in declared order.
        index_base: Sequence origin, normally the job random seed.
        sampling: Sampling policy from :data:`SAMPLING_POLICIES`.
        target_offset: Camera aim-point displacement policy.
        group_key: Callable mapping a candidate to its balancing group.

    Raises:
        ValidationError: If the operations, policy, or origin are invalid.

    Example:

    .. code-block:: python

        schedule = SampleSchedule(operations=("shift", "tombstone"))
        print([schedule.operation_for(index) for index in range(4)])
        # ['shift', 'tombstone', 'shift', 'tombstone']

    """

    operations: tuple[str, ...] = ()
    #: Operation names cycled across samples, in declared order.

    index_base: int = 0
    #: Sequence origin, normally the job random seed.

    sampling: str = "stratified"
    #: Sampling policy from :data:`SAMPLING_POLICIES`.

    target_offset: TargetOffsetPolicy = field(default_factory=TargetOffsetPolicy)
    #: Camera aim-point displacement policy.

    group_key: Callable[[str], str] = default_group_key
    #: Callable mapping a candidate to its balancing group.

    def __post_init__(self) -> None:
        """Reject unusable schedule configuration at construction time.

        Raises:
            ValidationError: If the operations, policy, or origin are invalid.

        """
        if not self.operations:
            raise ValidationError("a sample schedule requires at least one operation")
        if len(set(self.operations)) != len(self.operations):
            raise ValidationError("scheduled operations must be unique")
        # str(name) would launder None into "None" and hand it on as an
        # operation name; the dispatch contract only breaks later, further from
        # the cause.
        if any(not isinstance(name, str) for name in self.operations):
            raise ValidationError(
                "scheduled operation names must be strings, got %r"
                % (self.operations,),
            )
        if any(not name.strip() for name in self.operations):
            raise ValidationError("scheduled operation names must not be empty")
        # index_base feeds schedule_index(), whose result indexes a tuple.
        if isinstance(self.index_base, bool) or not isinstance(
                self.index_base, numbers.Integral):
            raise ValidationError(
                "index_base must be an integer, got %r" % (self.index_base,),
            )
        if self.sampling not in SAMPLING_POLICIES:
            raise ValidationError(
                "sampling must be one of %s, got %r"
                % (list(SAMPLING_POLICIES), self.sampling),
            )
        if self.index_base < 0:
            raise ValidationError("index_base must not be negative")

    def schedule_index(self, sample_index: int) -> int:
        """Return the absolute sequence position of one sample.

        Args:
            sample_index: Zero-based sample counter owned by the executor.

        Returns:
            Absolute sequence position used by every sampling decision.

        """
        return max(0, self.index_base + int(sample_index))

    def variant_index(self, sample_index: int) -> int:
        """Return the deterministic operation variant for one sample.

        Operation implementations use this index to stratify their own
        internal choices, such as shift direction sectors or angle bins.

        Args:
            sample_index: Zero-based sample counter owned by the executor.

        Returns:
            Variant index handed to the operation implementation.

        """
        return self.schedule_index(sample_index)

    def operation_for(
        self,
        sample_index: int,
    ) -> str:
        """Return the operation applied to one sample.

        Args:
            sample_index: Zero-based sample counter owned by the executor.

        Returns:
            Operation name for this sample.

        """
        position = self.schedule_index(sample_index) % len(self.operations)
        return self.operations[position]

    def component_order(
        self,
        candidates: Sequence[str],
        sample_index: int,
    ) -> tuple[str, ...]:
        """Return the candidate components to try, best first.

        Selection can fail late -- a component may turn out to be ineligible
        for the scheduled operation -- so the schedule returns an order rather
        than a single choice, and the caller walks it until one succeeds.

        Under ``"stratified"`` the order cycles balancing groups first and
        instances within a group second, so a package family with many board
        placements cannot dominate a finite run.

        Args:
            candidates: Eligible component identifiers for this operation.
            sample_index: Zero-based sample counter owned by the executor.

        Returns:
            Deduplicated candidate order, empty when there are no candidates.

        Example:

        .. code-block:: python

            schedule = SampleSchedule(operations=("shift",))
            order = schedule.component_order(["/b/F/x", "/b/G/y"], 0)
            print(len(order))
            # 2

        """
        unique = sorted({str(candidate) for candidate in candidates})
        if not unique:
            return ()
        if self.sampling != "stratified":
            return tuple(unique)
        groups: dict[str, list[str]] = {}
        for candidate in unique:
            groups.setdefault(self.group_key(candidate), []).append(candidate)
        names = sorted(groups)
        index = self.schedule_index(sample_index)
        group_start = index % len(names)
        member_round = index // len(names)
        widest = max(len(members) for members in groups.values())
        ordered: list[str] = []
        seen: set[str] = set()
        for member_delta in range(widest):
            for group_delta in range(len(names)):
                members = groups[names[(group_start + group_delta) % len(names)]]
                candidate = members[(member_round + member_delta) % len(members)]
                if candidate not in seen:
                    seen.add(candidate)
                    ordered.append(candidate)
        return tuple(ordered)

    def camera_target_offset(
        self,
        sample_index: int,
        *,
        attempt: int = 1,
    ) -> tuple[float, float]:
        """Return the millimeter camera aim offset for one capture attempt.

        Args:
            sample_index: Zero-based sample counter owned by the executor.
            attempt: One-based acceptance attempt for this sample.

        Returns:
            Millimeter displacement applied to the component centroid.

        """
        return self.target_offset.sample(
            self.schedule_index(sample_index),
            attempt=attempt,
        )

    def preview(self, samples: int) -> tuple[Mapping[str, object], ...]:
        """Return the deterministic per-sample plan for the first samples.

        This makes a planned run inspectable without running Isaac: the
        operation sequence and framing offsets can be reviewed, diffed, and
        asserted on before any render happens.

        Args:
            samples: Number of leading samples to describe.

        Returns:
            One mapping per sample describing its scheduled decisions.

        """
        return tuple(
            {
                "sample_index": index,
                "schedule_index": self.schedule_index(index),
                "operation": self.operation_for(index),
                "variant_index": self.variant_index(index),
                "camera_target_offset_mm": [
                    round(value, 6) for value in self.camera_target_offset(index)
                ],
            }
            for index in range(max(0, int(samples)))
        )
