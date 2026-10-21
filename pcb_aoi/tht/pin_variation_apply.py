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

"""Turn declared ``pin_variation`` groups into moved lead geometry.

The declaration names sets of parts and the bands to vary them over. This
resolves those names against the leads the board actually has, draws one
variation per lead from a seeded generator, asks
`pcb_aoi.geometry.tht_pin_pose` what the barrel allows, and writes the points
back.

Selection and sampling are separated from the USD write on purpose: everything
up to the point list is pure and testable without a stage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..geometry.transform import transform_points
from pcb_aoi.geometry.tht_pin_pose import (
    PinVariation,
    apply_pin_variation,
    densify_prism,
    wall_overrun_mm,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pcb_aoi.tht.lead_discovery import DiscoveredLead

#: Upper bound on tilt regardless of what a group asks for, from the dataset
#: description. The barrel usually binds well before this.
TILT_CAP_DEG = 15.0

#: One raster pixel of slack when gating the produced geometry: an independent
#: slice is rasterised at 10 um and a boundary sample can sit half a pixel
#: outside the true surface.
WALL_TOLERANCE_MM = 0.010


@dataclass(frozen=True)
class LeadOutcome:
    """What one lead was asked for and what it got.

    Attributes:
        path: Prim path of the lead.
        refdes: Reference designator of the owning component.
        group_index: Index of the group that claimed this lead.
        pairing_source: How the lead's barrel was resolved.
        tilt_requested_deg: Tilt drawn from the group's band.
        tilt_deg: Tilt delivered.
        tilt_limited_by: Constraint that bound the tilt.
        offset_requested_mm: Offset drawn from the group's band.
        offset_mm: Offset delivered.
        offset_limited_by: Constraint that bound the offset.
        cut_requested_deg: Cut angle drawn from the group's band.
        cut_deg: Cut angle delivered.
        cut_limited_by: Constraint that bound the cut angle.
        protrusion_mm: Protrusion delivered, or None when the lead's length was
            left alone.
        trimmed: Whether the lead's end face was moved at all.
        wall_overrun_mm: Measured escape past the bore; must not be positive.
    """

    path: str
    refdes: str
    group_index: int
    pairing_source: str
    tilt_requested_deg: float
    tilt_deg: float
    tilt_limited_by: str
    offset_requested_mm: float
    offset_mm: float
    offset_limited_by: str
    cut_requested_deg: float
    cut_deg: float
    cut_limited_by: str
    protrusion_mm: float | None
    trimmed: bool
    wall_overrun_mm: float


def select_leads(
    leads: Sequence[DiscoveredLead],
    groups: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Return which group claims each lead.

    A lead named by more than one group takes the **last** group that names it,
    so a broad ``package_types`` sweep can be narrowed by a ``components`` group
    written after it. Resolving it the other way round would make the order of
    the calls meaningless.

    Args:
        leads: Leads discovered on the board.
        groups: Declared groups, in authoring order.

    Returns:
        Lead prim path mapped to the index of the group that claims it.
    """
    claimed: dict[str, int] = {}
    for index, group in enumerate(groups):
        components = group.get("components")
        packages = group.get("package_types")
        wanted = {str(name) for name in (components or packages or ())}
        by_refdes = components is not None
        for lead in leads:
            key = lead.refdes if by_refdes else lead.package_type
            if key is not None and str(key) in wanted:
                claimed[lead.path] = index
    return claimed


def _draw(rng: random.Random, band: Sequence[float] | None) -> float | None:
    """Return one sample from a declared band, or None when it was omitted.

    **None is not zero.** An omitted axis must leave the lead alone, while an
    explicit ``(0, 0)`` is a real request -- a protrusion of zero asks for a
    lead cut flush with the board.

    Args:
        rng: Seeded generator.
        band: ``(lo, hi)`` pair, or None for an axis the group omitted.

    Returns:
        A sample, or None when the axis was not requested.
    """
    if band is None:
        return None
    return rng.uniform(float(band[0]), float(band[1]))


def plan_lead_variation(
    rng: random.Random,
    group: Mapping[str, object],
) -> PinVariation:
    """Draw one lead's requested variation from a group's bands.

    Azimuths are always drawn over the full circle: a band that says how far to
    lean says nothing about which way, and leaning every pin on a board the same
    way would be a rig, not a variation.

    Args:
        rng: Seeded generator.
        group: One declared group.

    Returns:
        The requested variation, before the barrel has its say.
    """
    # Tilt and offset are magnitudes about a nominal pose, so an omitted axis
    # is genuinely zero. Protrusion and cut act on the lead's END, where doing
    # nothing and doing zero are different geometry, so those stay None.
    return PinVariation(
        tilt_deg=_draw(rng, group.get("tilt_range")) or 0.0,
        azimuth_deg=rng.uniform(0.0, 360.0),
        offset_mm=_draw(rng, group.get("offset_range")) or 0.0,
        offset_azimuth_deg=rng.uniform(0.0, 360.0),
        protrusion_mm=_draw(rng, group.get("height_variation")),
        cut_deg=_draw(rng, group.get("cut_angle")),
        cut_azimuth_deg=rng.uniform(0.0, 360.0),
    )


def plan_board_variation(
    leads: Sequence[DiscoveredLead],
    groups: Sequence[Mapping[str, object]],
    *,
    board_far_face_z: float,
    seed: int,
) -> tuple[list[tuple[DiscoveredLead, object]], list[LeadOutcome]]:
    """Resolve, sample and solve every claimed lead. No stage required.

    Leads are processed in a **stable path order** and each draws from a
    generator seeded by the run seed and its own path, so adding a group or
    reordering discovery does not reshuffle the parts that were already varied.

    Args:
        leads: Leads discovered on the board.
        groups: Declared groups, in authoring order.
        board_far_face_z: Board face the protrusion is measured from.
        seed: Run seed.

    Returns:
        The leads to write with their solved results, one outcome record per
        varied lead, and one record per lead skipped with the reason.
    """
    claimed = select_leads(leads, groups)
    edits: list[tuple[DiscoveredLead, object]] = []
    outcomes: list[LeadOutcome] = []
    skipped: list[dict[str, object]] = []

    for lead in sorted(leads, key=lambda item: item.path):
        index = claimed.get(lead.path)
        if index is None or lead.hole is None:
            continue
        pose_lead = lead.as_pose_lead(board_far_face_z)

        # A LEAD ALREADY OUTSIDE ITS OWN BORE CANNOT BE POSED. "Stay inside the
        # plating" is not a constraint that can be satisfied when the untouched
        # geometry violates it, so such a lead is skipped and named rather than
        # failing the whole board. Measured at rest, before any pose, so this
        # can never excuse a breach the solver caused.
        rest = wall_overrun_mm(
            pose_lead,
            densify_prism(pose_lead.points, *pose_lead.barrel_z),
        )
        if rest > WALL_TOLERANCE_MM:
            skipped.append(
                {
                    "path": lead.path,
                    "refdes": lead.refdes,
                    "package_type": lead.package_type,
                    "reason": "outside its own bore at rest",
                    "rest_overrun_mm": round(rest, 5),
                    "bore_radius_mm": round(float(lead.hole.radius_mm), 5),
                },
            )
            continue

        rng = random.Random(f"{seed}:{lead.path}")  # noqa: S311 -- not crypto
        request = plan_lead_variation(rng, groups[index])
        result = apply_pin_variation(
            pose_lead, request, tilt_cap_deg=TILT_CAP_DEG,
        )
        edits.append((lead, result))
        outcomes.append(
            LeadOutcome(
                path=lead.path,
                refdes=lead.refdes,
                group_index=index,
                pairing_source=lead.pairing_source,
                tilt_requested_deg=result.tilt_requested_deg,
                tilt_deg=result.tilt_deg,
                tilt_limited_by=result.tilt_limited_by,
                offset_requested_mm=result.offset_requested_mm,
                offset_mm=result.offset_mm,
                offset_limited_by=result.offset_limited_by,
                cut_requested_deg=result.cut_requested_deg,
                cut_deg=result.cut_deg,
                cut_limited_by=result.cut_limited_by,
                protrusion_mm=result.protrusion_mm,
                trimmed=result.trimmed,
                wall_overrun_mm=wall_overrun_mm(
                    pose_lead, result.points, result.probe_points,
                ),
            ),
        )
    return edits, outcomes, skipped


def write_lead_points(
    stage: object,
    edits: Sequence[tuple[DiscoveredLead, object]],
) -> int:
    """Write solved lead geometry back onto the stage.

    Points are authored in each lead's own **local** space: the solve happens in
    world space, and writing world points onto a prim that carries a transform
    would apply that transform twice.

    Args:
        stage: Open USD stage.
        edits: Leads and their solved results.

    Returns:
        The number of leads written.

    Raises:
        ValueError: If a lead's prim is no longer on the stage.
    """
    import numpy as np
    from pxr import Gf, UsdGeom

    for lead, result in edits:
        prim = stage.GetPrimAtPath(lead.path)
        if not prim or not prim.IsValid():
            message = f"lead prim {lead.path!r} is not on the stage"
            raise ValueError(message)
        local = [
            Gf.Vec3f(*point)
            for point in transform_points(
                lead.world_to_local, result.points,
            ).tolist()
        ]
        mesh = UsdGeom.Mesh(prim)
        mesh.GetPointsAttr().Set(local)
        # The extent describes the POINTS attribute, so it is local too. Taking
        # it from the world-space solve puts the parent transform in twice and
        # the prim's bound lands wherever the board happens to sit.
        bounds = np.array([[float(v) for v in point] for point in local], float)
        mesh.CreateExtentAttr().Set(
            [
                Gf.Vec3f(*(float(v) for v in bounds.min(0))),
                Gf.Vec3f(*(float(v) for v in bounds.max(0))),
            ],
        )
    return len(edits)


def summarise(
    outcomes: Sequence[LeadOutcome],
    skipped: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Return a transport-friendly summary of one pin_variation order.

    Args:
        outcomes: Per-lead outcome records.
        skipped: Leads that could not be posed, with their reason.

    Returns:
        A JSON-compatible mapping.
    """
    if not outcomes:
        return {
            "leads": 0,
            "breaches": 0,
            "limited_by": {},
            "per_lead": [],
            "skipped": len(skipped),
            "skipped_leads": list(skipped),
        }
    limited: dict[str, int] = {}
    for outcome in outcomes:
        limited[outcome.tilt_limited_by] = limited.get(outcome.tilt_limited_by, 0) + 1
    tilts = [outcome.tilt_deg for outcome in outcomes]
    return {
        "leads": len(outcomes),
        "breaches": sum(
            1 for o in outcomes if o.wall_overrun_mm > WALL_TOLERANCE_MM
        ),
        "tilt_deg_min": min(tilts),
        "tilt_deg_max": max(tilts),
        "limited_by": limited,
        "skipped": len(skipped),
        "skipped_leads": list(skipped),
        "per_lead": [
            {
                "path": o.path,
                "refdes": o.refdes,
                "group": o.group_index,
                "pairing_source": o.pairing_source,
                "tilt_requested_deg": round(o.tilt_requested_deg, 4),
                "tilt_deg": round(o.tilt_deg, 4),
                "tilt_limited_by": o.tilt_limited_by,
                # Requested AND delivered for every axis, not just tilt.
                # The report is the only way a caller learns that a band
                # was not honoured; with the delivered value alone,
                # `cut_limited_by: "request"` reads the same whether the
                # ask was 13 deg or 30.
                "offset_requested_mm": round(o.offset_requested_mm, 5),
                "offset_mm": round(o.offset_mm, 5),
                "offset_limited_by": o.offset_limited_by,
                "cut_requested_deg": round(o.cut_requested_deg, 4),
                "cut_deg": round(o.cut_deg, 4),
                "cut_limited_by": o.cut_limited_by,
                "protrusion_mm": (
                    None if o.protrusion_mm is None
                    else round(o.protrusion_mm, 5)
                ),
                "trimmed": o.trimmed,
                "wall_overrun_mm": round(o.wall_overrun_mm, 6),
            }
            for o in outcomes
        ],
    }
