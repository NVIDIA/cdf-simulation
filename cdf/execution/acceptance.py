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

"""Whether one captured scene is good enough to keep.

Some scenes a capture run produces are unusable: the focused component is
hidden behind a neighbour, or cut off at the edge of the frame. This module is
the rule that decides, and -- when a scene is rejected -- says why, so a run
that throws most of its scenes away can be diagnosed instead of guessed at.

Deliberately free of Kit, USD, Replicator, NumPy and domain vocabulary, so the
decision is unit testable and both surfaces that need it -- the JobOrder
runtime and the standalone capture script -- share one implementation rather
than keeping private copies.

The schedule that decides *what* to capture is a separate concern and lives in
:mod:`cdf.execution.sample_schedule`; nothing here refers to it.
"""

from __future__ import annotations

import numbers

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ..errors import ValidationError

@dataclass(frozen=True)
class ViewObservation:
    """Measured quality of the focused component in one rendered view.

    Args:
        view_id: Stable identifier of the rendered view.
        visible: Whether the focused component covered any pixel.
        occlusion_ratio: Renderer-native occlusion ratio, or None if absent.
        truncated: Whether the component touched the frame margin.

    """

    view_id: str
    #: Stable identifier of the rendered view.

    visible: bool = False
    #: Whether the focused component covered any pixel.

    occlusion_ratio: float | None = None
    #: Renderer-native occlusion ratio, or None when unavailable.

    truncated: bool = False
    #: Whether the component touched the configured frame margin.


@dataclass(frozen=True)
class ViewVerdict:
    """Acceptance-policy interpretation of one :class:`ViewObservation`.

    Args:
        view_id: Stable identifier of the rendered view.
        visible: Whether the focused component covered any pixel.
        occluded: Whether the view counts as occluded under the policy.
        truncated: Whether the component touched the frame margin.
        fully_visible: Whether the view is visible, unoccluded, and untruncated.

    """

    view_id: str
    #: Stable identifier of the rendered view.

    visible: bool
    #: Whether the focused component covered any pixel.

    occluded: bool
    #: Whether the view counts as occluded under the policy.

    truncated: bool
    #: Whether the component touched the configured frame margin.

    fully_visible: bool
    #: Whether the view is visible, unoccluded, and untruncated.


@dataclass(frozen=True)
class AcceptanceReport:
    """Outcome of applying an :class:`AcceptancePolicy` to one scene.

    Args:
        accepted: Whether the scene may be written to the dataset.
        verdicts: Per-view interpretation in observation order.
        reasons: Human-readable rejection reasons, empty when accepted.

    """

    accepted: bool
    #: Whether the scene may be written to the dataset.

    verdicts: tuple[ViewVerdict, ...] = ()
    #: Per-view interpretation in observation order.

    reasons: tuple[str, ...] = ()
    #: Human-readable rejection reasons, empty when accepted.

    @property
    def visible_views(self) -> int:
        """Number of views showing at least one component pixel."""
        return sum(1 for verdict in self.verdicts if verdict.visible)

    @property
    def invisible_views(self) -> int:
        """Number of views showing no component pixel."""
        return len(self.verdicts) - self.visible_views

    @property
    def occluded_views(self) -> int:
        """Number of views counted as occluded."""
        return sum(1 for verdict in self.verdicts if verdict.occluded)

    @property
    def truncated_views(self) -> int:
        """Number of views counted as truncated."""
        return sum(1 for verdict in self.verdicts if verdict.truncated)

    @property
    def fully_visible_views(self) -> int:
        """Number of views that are visible, unoccluded, and untruncated."""
        return sum(1 for verdict in self.verdicts if verdict.fully_visible)

    def counts(self) -> dict[str, int]:
        """Return the view counts recorded alongside a captured scene.

        Returns:
            Mapping of count name to value for dataset provenance.

        """
        return {
            "visible_views": self.visible_views,
            "invisible_views": self.invisible_views,
            "occluded_views": self.occluded_views,
            "truncated_views": self.truncated_views,
            "fully_visible_views": self.fully_visible_views,
        }


@dataclass(frozen=True)
class AcceptancePolicy:
    """Whether a captured scene is good enough to enter the dataset.

    Args:
        occlusion_threshold: Occlusion ratio above which a view is occluded.
        min_visible_views: Minimum views showing the component.
        min_invisible_views: Minimum views deliberately hiding the component.
        min_fully_visible_views: Minimum clean, unoccluded, untruncated views.
        reject_truncated: Whether any truncated view rejects the scene.

    Raises:
        ValidationError: If a threshold or minimum is out of range.

    """

    occlusion_threshold: float = 0.5
    #: Occlusion ratio above which a visible view counts as occluded.

    min_visible_views: int = 0
    #: Minimum number of views that must show the component.

    min_invisible_views: int = 0
    #: Minimum number of views that must not show the component.

    min_fully_visible_views: int = 1
    #: Minimum number of clean, unoccluded, untruncated views.

    reject_truncated: bool = True
    #: Whether any truncated view rejects the whole scene.

    def __post_init__(self) -> None:
        """Reject unusable acceptance configuration at construction time.

        Raises:
            ValidationError: If a threshold or minimum is out of range.

        """
        if not 0.0 <= self.occlusion_threshold <= 1.0:
            raise ValidationError(
                "occlusion_threshold must be within [0, 1], got %r"
                % (self.occlusion_threshold,),
            )
        for name in (
            "min_visible_views",
            "min_invisible_views",
            "min_fully_visible_views",
        ):
            value = getattr(self, name)
            # These are compared against integer view counts. A float slips
            # through "must not be negative" and then silently shifts the
            # threshold: min_visible_views=1.5 rejects a single visible view as
            # though the minimum were 2, changing scene acceptance rather than
            # failing the config. bool is an int in Python, and True as a view
            # count is a mistake every time.
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValidationError(
                    "%s must be an integer view count, got %r" % (name, value),
                )
            if value < 0:
                raise ValidationError("%s must not be negative" % name)

    def classify(self, observation: ViewObservation) -> ViewVerdict:
        """Interpret one measured view under this policy.

        A view with no component pixels is never fully visible, even when an
        unavailable or stale occlusion sensor reports zero occlusion.

        Args:
            observation: Measured quality of one rendered view.

        Returns:
            Policy interpretation of the observation.

        """
        ratio = observation.occlusion_ratio
        occluded = bool(
            not observation.visible
            or (ratio is not None and ratio > self.occlusion_threshold),
        )
        return ViewVerdict(
            view_id=observation.view_id,
            visible=bool(observation.visible),
            occluded=occluded,
            truncated=bool(observation.truncated),
            fully_visible=bool(
                observation.visible and not occluded and not observation.truncated,
            ),
        )

    def evaluate(
        self,
        observations: Iterable[ViewObservation],
    ) -> AcceptanceReport:
        """Accept or reject a captured scene from its measured views.

        Args:
            observations: Measured quality of every rendered view.

        Returns:
            Acceptance decision with per-view verdicts and rejection reasons.

        Example:

        .. code-block:: python

            policy = AcceptancePolicy()
            report = policy.evaluate([ViewObservation("top_down", visible=True)])
            print(report.accepted)
            # True

        """
        verdicts = tuple(self.classify(item) for item in observations)
        report = AcceptanceReport(accepted=True, verdicts=verdicts)
        reasons = []
        if report.visible_views < self.min_visible_views:
            reasons.append(
                "visible views %d < %d" % (report.visible_views, self.min_visible_views),
            )
        if report.invisible_views < self.min_invisible_views:
            reasons.append(
                "invisible views %d < %d"
                % (report.invisible_views, self.min_invisible_views),
            )
        if report.fully_visible_views < self.min_fully_visible_views:
            reasons.append(
                "fully visible views %d < %d"
                % (report.fully_visible_views, self.min_fully_visible_views),
            )
        if self.reject_truncated and report.truncated_views:
            reasons.append("%d truncated views rejected" % report.truncated_views)
        return AcceptanceReport(
            accepted=not reasons,
            verdicts=verdicts,
            reasons=tuple(reasons),
        )
