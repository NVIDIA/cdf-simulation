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

"""Decide where every claimed lead is trimmed. No stage required.

Sampling lives apart from the stage edit for the same reason the pose solve
does: the plane a lead is cut on is a decision, and a decision that can be
tested without opening a board is one that gets tested.

Leads are processed in a stable path order and each draws from a generator
seeded by the run seed and its own path, so adding a group or reordering
discovery does not reshuffle the leads that were already planned.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LeadTrim:
    """Where one lead, and the joint on it, is cut.

    Args:
        lead_path: Prim path of the lead.
        protrusion_mm: Depth the deepest point of the section lands at,
            measured from the board's far face.
        cut_deg: Trim angle off perpendicular.
        cut_azimuth_deg: Direction the trim plane tips toward.
    """

    lead_path: str
    protrusion_mm: float
    cut_deg: float
    cut_azimuth_deg: float


def _draw(rng: random.Random, band: object) -> float | None:
    """Return a sample from a band, or None when the band is absent."""
    if band is None:
        return None
    low, high = band
    return rng.uniform(float(low), float(high))


def plan_board_trim(
    leads: Sequence[object],
    groups: Sequence[Mapping[str, object]],
    *,
    seed: int,
    select: object,
) -> tuple[list[LeadTrim], list[dict[str, object]]]:
    """Resolve and sample a trim for every lead a group claims.

    Args:
        leads: Leads discovered on the board.
        groups: Declared trim groups, in authoring order.
        seed: Run seed, owned by the job.
        select: The shared group-resolution function, passed in so this module
            does not import the selector and the selector does not import this.

    Returns:
        The trims to apply, and one record per lead skipped with the reason.

    """
    claimed = select(leads, groups)
    trims: list[LeadTrim] = []
    skipped: list[dict[str, object]] = []

    for lead in sorted(leads, key=lambda item: item.path):
        index = claimed.get(lead.path)
        if index is None:
            continue
        if getattr(lead, "hole", None) is None:
            skipped.append({"lead": lead.path, "reason": "no plated hole"})
            continue
        group = groups[index]
        # The seed is the RUN's plus the lead's own path, the same rule the
        # pose solve uses, so the two draw independently and neither reshuffles
        # when the other is added.
        rng = random.Random(f"{seed}:{lead.path}:cut")
        protrusion = _draw(rng, group.get("height_variation"))
        cut = _draw(rng, group.get("cut_angle"))
        if protrusion is None and cut is None:
            continue
        # An omitted axis must not act. A group that asks only for a cut angle
        # is asking to tip the plane where the lead already ends, not to
        # shorten it to zero; a group that asks only for a height is asking for
        # a square cut.
        if protrusion is None:
            # Cut where the lead already ends: measure how far its deepest
            # point stands beyond the board's far face. A lead record carries
            # its world points and its barrel, which is everything needed --
            # there is no authored protrusion to read.
            far_face = float(lead.hole.z_range[0])
            protrusion = max(
                (far_face - float(point[2]) for point in lead.points),
                default=0.0,
            )
            if protrusion <= 0.0:
                skipped.append({
                    "lead": lead.path,
                    "reason": "cut_angle alone, and this lead does not stand "
                              "proud of the board's far face",
                })
                continue
        trims.append(LeadTrim(
            lead_path=lead.path,
            protrusion_mm=float(protrusion),
            cut_deg=float(cut or 0.0),
            # Azimuth is drawn over the full circle: a band that says how
            # steeply to cut says nothing about which way the blade came from,
            # and cutting every lead on a board the same way would be a rig.
            cut_azimuth_deg=rng.uniform(0.0, 360.0),
        ))
    return trims, skipped


def summarise_trims(
    trims: Sequence[LeadTrim],
    skipped: Sequence[Mapping[str, object]],
    applied: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return a report of what was cut and what was not.

    Args:
        trims: The trims planned.
        skipped: Leads a group claimed but could not be cut.
        applied: Per-prim records returned by the stage edit.

    Returns:
        A summary alongside the per-lead detail.
    """
    sectioned = [record for record in applied if record.get("cut_faces")]
    return {
        "planned": len(trims),
        "skipped": len(skipped),
        "prims_cut": len(applied),
        "prims_sectioned": len(sectioned),
        "cut_deg": {
            "min": round(min((trim.cut_deg for trim in trims), default=0.0), 3),
            "max": round(max((trim.cut_deg for trim in trims), default=0.0), 3),
        },
        "protrusion_mm": {
            "min": round(min((trim.protrusion_mm for trim in trims),
                             default=0.0), 4),
            "max": round(max((trim.protrusion_mm for trim in trims),
                             default=0.0), 4),
        },
        "leads": [
            {
                "lead": trim.lead_path,
                "protrusion_mm": round(trim.protrusion_mm, 4),
                "cut_deg": round(trim.cut_deg, 3),
                "cut_azimuth_deg": round(trim.cut_azimuth_deg, 1),
            }
            for trim in trims
        ],
        "skipped_detail": list(skipped),
    }


def normal_for(cut_deg: float, cut_azimuth_deg: float,
               direction: float) -> tuple[float, float, float]:
    """Return the unit normal of a trim plane, pointing at what is removed."""
    cut = math.radians(cut_deg)
    azimuth = math.radians(cut_azimuth_deg)
    return (
        math.sin(cut) * math.cos(azimuth),
        math.sin(cut) * math.sin(azimuth),
        math.cos(cut) * direction,
    )
