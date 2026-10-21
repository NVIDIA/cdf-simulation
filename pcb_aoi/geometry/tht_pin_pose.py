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

"""Vary a through-hole lead inside its own plated barrel.

Four axes, and every one of them is bounded by something physical rather than
by the number a caller asked for:

* **tilt** — off vertical, but a pin is inside a barrel and cannot pass through
  the copper, so the request competes with a per-pin geometric limit.
* **offset** — the lead may sit anywhere in the bore, sampled from the
  clearance that *survives* the tilt so the two never fight over it.
* **protrusion** — how far the lead stands proud of the far board face,
  delivered by **trimming the lead**, which is what happens on a real board.
* **cut plane** — the trim need not be perpendicular, but a steeply angled face
  spans a height the protrusion may not have room for.

Requested and delivered are reported for each, along with *which* constraint
bound, so a value the board could not accommodate is visible as a pair rather
than as a silent substitution.

Nothing here draws random numbers, opens a stage or authors solder. It takes a
lead's points and its barrel, and returns moved points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: A pin may legally contact the barrel wall, but coincident surfaces z-fight
#: in a render, so joints are kept just off the copper.
WALL_GAP_MM = 0.010

#: Sections sampled along the lead when testing whether a tilted pin clears the
#: bore. Leads are extruded prisms with vertices *only* at their two ends, so a
#: tilted prism touches the barrel mouth at a height where there is no vertex at
#: all: sampling puts points where the constraint actually binds.
Z_SAMPLES = 48

#: Uniform tilt azimuths probed when solving the barrel limit, on top of the
#: lead's own corner directions. The limit is direction dependent because leads
#: are not all round, so the worst direction is the one that gets reported.
AZIMUTH_PROBES = 12

#: Halvings used to solve the barrel tilt limit. Fixed rather than
#: tolerance-driven so the result is deterministic.
BISECTION_STEPS = 24


@dataclass(frozen=True)
class Lead:
    """One through-hole lead and the barrel that constrains it.

    Attributes:
        points: World-space lead vertices, one end-cap ring followed by the
            matching ring at the other end.
        hole_center: Nominal bore centre in XY.
        hole_radius_mm: Bore radius.
        barrel_z: Lower and upper end planes of the drilled hole.
        board_far_face_z: Board face the protrusion is measured from.
        tip_is_low: Whether the trimmed end is the first ring and the lower one.
    """

    points: np.ndarray
    hole_center: tuple[float, float]
    hole_radius_mm: float
    barrel_z: tuple[float, float]
    board_far_face_z: float
    tip_is_low: bool = True

    @property
    def pivot(self) -> np.ndarray:
        """Return the point the lead hinges about.

        A pin is held by the hole, so it hinges at the **barrel mid-plane**.
        Pivoting at the lead's own centroid instead swings the whole lead
        sideways through the plating.

        Returns:
            The barrel mid-plane point on the bore axis.
        """
        return np.array(
            [
                self.hole_center[0],
                self.hole_center[1],
                0.5 * (self.barrel_z[0] + self.barrel_z[1]),
            ],
            float,
        )


@dataclass(frozen=True)
class PinVariation:
    """One requested variation, before the board has had its say.

    Attributes:
        tilt_deg: Requested tilt off vertical.
        azimuth_deg: Direction the lead leans toward.
        offset_mm: Requested off-centre distance in the bore.
        offset_azimuth_deg: Direction of the offset.
        protrusion_mm: Requested lead standing proud of the far board face,
            or None to leave the lead's own length alone. **None and 0.0 mean
            different things**: 0.0 asks for a lead trimmed flush with the
            board, which is a real and quite visible request.
        cut_deg: Requested trim-plane angle off perpendicular, or None for no
            angle. A cut angle without a protrusion trims at the lead's
            existing length, changing only the angle of the face.
        cut_azimuth_deg: Direction the trim plane tips toward.
    """

    tilt_deg: float = 0.0
    azimuth_deg: float = 0.0
    offset_mm: float = 0.0
    offset_azimuth_deg: float = 0.0
    protrusion_mm: float | None = None
    cut_deg: float | None = None
    cut_azimuth_deg: float = 0.0


@dataclass(frozen=True)
class PinVariationResult:
    """What the board actually allowed, beside what was asked for.

    Attributes:
        points: Moved lead vertices.
        tilt_requested_deg: Tilt as requested.
        tilt_deg: Tilt delivered.
        tilt_limit_deg: Largest tilt this barrel permits.
        tilt_limited_by: ``"request"`` or ``"barrel"``.
        offset_requested_mm: Offset as requested.
        offset_mm: Offset delivered.
        offset_budget_mm: Clearance left after the delivered tilt.
        offset_limited_by: ``"request"`` or ``"clearance"``.
        cut_requested_deg: Trim angle as requested.
        cut_deg: Trim angle delivered.
        cut_limited_by: ``"request"``, ``"protrusion"`` or ``"omitted"``.
        protrusion_mm: Protrusion delivered, or None when the lead's length
            was left alone.
        trimmed: Whether the lead's end face was moved at all.
        probe_points: The densified clearance probe, moved with the lead. The
            lead's own vertices sit at its two ends, which are outside the
            barrel, so any measurement of the delivered pose needs this too.
    """

    points: np.ndarray = field(repr=False)
    tilt_requested_deg: float = 0.0
    tilt_deg: float = 0.0
    tilt_limit_deg: float = 0.0
    tilt_limited_by: str = "request"
    offset_requested_mm: float = 0.0
    offset_mm: float = 0.0
    offset_budget_mm: float = 0.0
    offset_limited_by: str = "request"
    cut_requested_deg: float = 0.0
    cut_deg: float = 0.0
    cut_limited_by: str = "request"
    protrusion_mm: float | None = None
    trimmed: bool = False
    probe_points: np.ndarray | None = field(default=None, repr=False)


def tilt_rotation(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    """Return the rotation that leans a lead toward an azimuth.

    Rodrigues about the horizontal axis at ``azimuth + 90``, built so the
    lead's +Z leans **towards** ``azimuth``. The tilt direction is the thing
    being varied, so the sampled azimuth has to mean what it says; a sign error
    here puts every joint on the wrong side of its pin and still renders as a
    plausible board.

    Args:
        tilt_deg: Angle off vertical, degrees.
        azimuth_deg: Direction of lean, degrees.

    Returns:
        A three-by-three rotation matrix.
    """
    theta = math.radians(float(tilt_deg))
    phi = math.radians(float(azimuth_deg))
    axis = np.array([-math.sin(phi), math.cos(phi), 0.0])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
    )
    return np.eye(3) * cos_t + sin_t * cross + (1.0 - cos_t) * np.outer(axis, axis)


def max_radius_in_barrel(
    points: np.ndarray,
    pivot: np.ndarray,
    rotation: np.ndarray,
    hole_center: tuple[float, float],
    z_low: float,
    z_high: float,
) -> float:
    """Return the furthest a rotated lead reaches from the bore axis.

    Only the part of the lead that is *inside* the barrel can hit plating, so
    the measurement is restricted to that span.

    Args:
        points: Lead points, or a densified stand-in for them.
        pivot: Point the rotation is about.
        rotation: Rotation to apply before measuring.
        hole_center: Bore centre in XY.
        z_low: Lower end plane of the drilled hole.
        z_high: Upper end plane of the drilled hole.

    Returns:
        The largest radial distance reached inside the barrel, or 0.0 when no
        part of the lead is inside it.
    """
    moved = (rotation @ (points - pivot).T).T + pivot
    inside = (moved[:, 2] >= z_low - 1e-9) & (moved[:, 2] <= z_high + 1e-9)
    if not inside.any():
        return 0.0
    inside_points = moved[inside]
    return float(
        np.hypot(
            inside_points[:, 0] - hole_center[0],
            inside_points[:, 1] - hole_center[1],
        ).max(),
    )


def densify_prism(
    points: np.ndarray,
    z_low: float,
    z_high: float,
    samples: int = Z_SAMPLES,
) -> np.ndarray:
    """Resample an extruded lead as a stack of cross-sections.

    **This is a measuring instrument, not geometry.** The leads are vertical
    prisms with vertices only at their two ends, so a tilted prism's contact
    with the barrel mouth happens at a height that carries no vertex, and a
    clearance test over the raw vertices reports a pin as safe when it is
    through the copper.

    Build it once in the lead's upright frame and then move it rigidly with the
    lead. Re-densifying already-tilted points is meaningless, because the stack
    is vertical by construction.

    Args:
        points: Lead vertices.
        z_low: Lower end plane of the drilled hole.
        z_high: Upper end plane of the drilled hole.
        samples: Sections to take across the span.

    Returns:
        A point cloud covering the lead's footprint at every sampled height.
    """
    heights = np.linspace(
        min(z_low, points[:, 2].min()),
        max(z_high, points[:, 2].max()),
        samples,
    )
    # One section is the prism's XY footprint; for these extrusions the raw
    # vertex ring already is that footprint.
    section = np.unique(np.round(points[:, :2], 6), axis=0)
    tiled = np.repeat(section, len(heights), axis=0)
    return np.column_stack([tiled, np.tile(heights, len(section))])


def probe_azimuths(lead: Lead) -> np.ndarray:
    """Return the tilt directions a barrel limit has to be safe in.

    A uniform ring is not enough on its own. Leaning toward azimuth ``phi``
    pushes the section outward hardest along ``phi``, so the worst direction is
    the one aligned with the corner that already sits furthest from the bore
    axis — and on a square lead those are the diagonals, which a 12-step ring
    (every 30 degrees) steps straight over. Measured: a square pin solved on the
    ring alone overruns its wall by 1.7 microns at 45 degrees.

    So the lead's own corner directions are probed as well, which makes the
    limit exact for a convex polygonal section and no worse for any other.

    Args:
        lead: Lead whose footprint supplies the corner directions.

    Returns:
        Azimuths in degrees, ascending and deduplicated.
    """
    ring = np.linspace(0.0, 360.0, AZIMUTH_PROBES, endpoint=False)
    footprint = np.unique(np.round(lead.points[:, :2], 6), axis=0)
    corners = np.degrees(
        np.arctan2(
            footprint[:, 1] - lead.hole_center[1],
            footprint[:, 0] - lead.hole_center[0],
        ),
    )
    return np.unique(np.round(np.concatenate([ring, corners % 360.0]), 6))


def barrel_tilt_limit(
    lead: Lead,
    cap_deg: float,
    probe: np.ndarray | None = None,
) -> float:
    """Return the largest tilt that keeps a lead off its own plating.

    Bisected on the true rotated point cloud rather than solved in closed form:
    the leads are not all round — TO-92 style pins are square and many
    connector leads are rectangular — so the clearance is a function of tilt
    azimuth and there is no single radius to put in a formula. Every probed
    azimuth must clear, so the reported limit is safe in any direction.

    Args:
        lead: Lead and barrel to solve against.
        cap_deg: Upper bound the caller will not exceed regardless.
        probe: Optional densified stand-in; built from the lead when omitted.

    Returns:
        The permitted tilt in degrees, which is 0.0 when the lead is already
        against the wall standing straight.
    """
    z_low, z_high = lead.barrel_z
    cloud = densify_prism(lead.points, z_low, z_high) if probe is None else probe
    pivot = lead.pivot
    limit = float(lead.hole_radius_mm) - WALL_GAP_MM

    if max_radius_in_barrel(
        cloud, pivot, np.eye(3), lead.hole_center, z_low, z_high,
    ) > limit:
        return 0.0

    azimuths = probe_azimuths(lead)

    def clears(tilt_deg: float) -> bool:
        """Report whether a tilt clears the bore in every probed direction.

        Args:
            tilt_deg: Tilt to test.

        Returns:
            Whether every probed azimuth stays off the plating.
        """
        return all(
            max_radius_in_barrel(
                cloud,
                pivot,
                tilt_rotation(tilt_deg, azimuth),
                lead.hole_center,
                z_low,
                z_high,
            )
            <= limit
            for azimuth in azimuths
        )

    if clears(cap_deg):
        return float(cap_deg)
    low, high = 0.0, float(cap_deg)
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if clears(middle) else (low, middle)
    return low


def _current_protrusion(points: np.ndarray, lead: Lead) -> float:
    """Return how far a lead already stands proud of the far board face.

    Used when a caller asks for a cut ANGLE without asking for a length: the
    face is re-cut where the lead already ends.

    Args:
        points: Lead points after tilt and offset.
        lead: Lead being measured.

    Returns:
        The existing protrusion in millimetres, never negative.
    """
    if lead.tip_is_low:
        return max(0.0, lead.board_far_face_z - float(points[:, 2].min()))
    return max(0.0, float(points[:, 2].max()) - lead.board_far_face_z)


def _trim_cap(
    moved: np.ndarray,
    rotation: np.ndarray,
    lead: Lead,
    protrusion_mm: float,
    cut_deg: float,
    cut_azimuth_deg: float,
) -> tuple[np.ndarray, float]:
    """Cut the lead's end face and return the points and the angle used.

    Args:
        moved: Lead points after tilt and offset.
        rotation: The tilt rotation that produced them.
        lead: Lead being trimmed.
        protrusion_mm: Depth the deepest point of the face must land at.
        cut_deg: Requested trim angle off perpendicular.
        cut_azimuth_deg: Direction the trim plane tips toward.

    Returns:
        The trimmed points and the trim angle actually used.
    """
    half = len(moved) // 2
    cap = np.zeros(len(moved), bool)
    # Select the cap BY INDEX, never by height. A z-threshold identifies a cap
    # only while the pin is upright: once it leans, the cap is an oblique plane
    # whose vertices all sit at different heights, so a threshold picks out one
    # corner and the trim silently does nothing -- leads keep their full
    # protrusion while the manifest reports the varied value.
    if lead.tip_is_low:
        cap[:half] = True
    else:
        cap[half : 2 * half] = True
    if not cap.any():
        return moved, 0.0

    sign = -1.0 if lead.tip_is_low else 1.0
    axis = rotation @ np.array([0.0, 0.0, 1.0])
    face = moved[cap]
    centre_xy = face[:, :2].mean(0)
    span_radius = float(
        np.hypot(face[:, 0] - centre_xy[0], face[:, 1] - centre_xy[1]).max(),
    )

    # LIMIT THE CUT ANGLE BY THE PROTRUSION. A plane tilted `cut` through the
    # centroid spans 2 * radius * tan(cut) in height, and protrusion is measured
    # from the board FACE -- so a steeply angled face on a short protrusion has
    # no placement that keeps its deepest corner at the requested depth and its
    # shallowest one out of the hole. A nearly flush trim cannot be steeply
    # angled physically either.
    cut = math.radians(float(cut_deg))
    if span_radius > 1e-6:
        cut = min(
            cut,
            math.atan(max(0.0, protrusion_mm - 0.02) / (2.0 * span_radius)),
        )

    normal = np.array(
        [
            math.sin(cut) * math.cos(math.radians(cut_azimuth_deg)),
            math.sin(cut) * math.sin(math.radians(cut_azimuth_deg)),
            math.cos(cut),
        ],
        float,
    )
    anchor = np.array(
        [
            centre_xy[0],
            centre_xy[1],
            lead.board_far_face_z + sign * protrusion_mm,
        ],
        float,
    )
    # Slide the cap onto the cut plane ALONG THE PIN AXIS. Sliding it vertically
    # keeps the tip's XY, which on a leaning pin is its furthest-out section,
    # and raising that into the barrel drives it through the plating. Along the
    # axis the cap stays a true section of the prism, so the trimmed lead is
    # exactly the untrimmed one with its end cut off.
    travel = ((anchor - face) @ normal) / float(normal @ axis)
    cut_face = face + np.outer(travel, axis)

    # Anchor the finished face by its DEEPEST point. Anchoring the plane's
    # centroid makes the reported protrusion a number no part of the pin has.
    if lead.tip_is_low:
        excess = cut_face[:, 2].min() - (lead.board_far_face_z - protrusion_mm)
    else:
        excess = cut_face[:, 2].max() - (lead.board_far_face_z + protrusion_mm)
    cut_face = cut_face - (excess / axis[2]) * axis

    out = moved.copy()
    out[cap] = cut_face
    return out, math.degrees(cut)


def apply_pin_variation(
    lead: Lead,
    variation: PinVariation,
    *,
    tilt_cap_deg: float,
) -> PinVariationResult:
    """Tilt, shift and trim one lead, within what its barrel allows.

    Args:
        lead: Lead and barrel to vary.
        variation: The requested variation.
        tilt_cap_deg: Upper bound on tilt regardless of the barrel.

    Returns:
        The moved points and a requested-versus-delivered record per axis.
    """
    z_low, z_high = lead.barrel_z
    probe = densify_prism(lead.points, z_low, z_high)
    limit = barrel_tilt_limit(lead, tilt_cap_deg, probe=probe)

    tilt = min(float(variation.tilt_deg), limit)
    tilt_by = "barrel" if tilt < float(variation.tilt_deg) - 1e-9 else "request"

    rotation = tilt_rotation(tilt, variation.azimuth_deg)
    pivot = lead.pivot

    # Spend what the TILT LEFT BEHIND, so the two axes never fight over one
    # budget. The tilt has already used part of the bore.
    reach = max_radius_in_barrel(
        probe, pivot, rotation, lead.hole_center, z_low, z_high,
    )
    budget = max(0.0, float(lead.hole_radius_mm) - WALL_GAP_MM - reach)
    offset = min(float(variation.offset_mm), budget)
    offset_by = "clearance" if offset < float(variation.offset_mm) - 1e-9 else "request"

    moved = (rotation @ (lead.points - pivot).T).T + pivot
    moved[:, 0] += offset * math.cos(math.radians(variation.offset_azimuth_deg))
    moved[:, 1] += offset * math.sin(math.radians(variation.offset_azimuth_deg))

    # AN OMITTED AXIS MUST NOT ACT. Mapping "no height_variation" onto
    # protrusion 0.0 and trimming anyway cuts every lead flush with the board
    # -- a dramatic change to a group that only asked for tilt.
    trimmed = not (variation.protrusion_mm is None and variation.cut_deg is None)
    if trimmed:
        # A cut angle on its own re-faces the lead where it already ends.
        protrusion = variation.protrusion_mm
        if protrusion is None:
            protrusion = _current_protrusion(moved, lead)
        moved, cut = _trim_cap(
            moved,
            rotation,
            lead,
            float(protrusion),
            float(variation.cut_deg or 0.0),
            variation.cut_azimuth_deg,
        )
        requested_cut = float(variation.cut_deg or 0.0)
        cut_by = "protrusion" if cut < requested_cut - 1e-9 else "request"
    else:
        protrusion, cut, requested_cut, cut_by = None, 0.0, 0.0, "omitted"

    moved_probe = (rotation @ (probe - pivot).T).T + pivot
    moved_probe[:, 0] += offset * math.cos(math.radians(variation.offset_azimuth_deg))
    moved_probe[:, 1] += offset * math.sin(math.radians(variation.offset_azimuth_deg))

    return PinVariationResult(
        points=moved,
        probe_points=moved_probe,
        tilt_requested_deg=float(variation.tilt_deg),
        tilt_deg=tilt,
        tilt_limit_deg=limit,
        tilt_limited_by=tilt_by,
        offset_requested_mm=float(variation.offset_mm),
        offset_mm=offset,
        offset_budget_mm=budget,
        offset_limited_by=offset_by,
        cut_requested_deg=requested_cut,
        cut_deg=cut,
        cut_limited_by=cut_by,
        protrusion_mm=None if protrusion is None else float(protrusion),
        trimmed=trimmed,
    )


def wall_overrun_mm(lead: Lead, *point_sets: np.ndarray | None) -> float:
    """Return how far the supplied geometry escapes the bore it belongs to.

    The gate takes **points**, not a request: re-running the planner's own
    arithmetic would agree with the planner by construction.

    Pass the varied lead *and* its moved probe. A lead's own vertices are at
    its two ends, both outside the barrel, so measuring the lead alone finds
    nothing inside the bore at all and reports a clean pass for any pose
    whatsoever.

    Args:
        lead: Lead whose barrel is the reference.
        *point_sets: Clouds to measure; ``None`` entries are skipped.

    Returns:
        The largest reach beyond ``hole_radius_mm``; positive means the plating
        was breached.

    Raises:
        ValueError: If no cloud is supplied.
    """
    clouds = [p for p in point_sets if p is not None and len(p)]
    if not clouds:
        message = "wall_overrun_mm needs at least one cloud to measure"
        raise ValueError(message)
    z_low, z_high = lead.barrel_z
    reach = max(
        max_radius_in_barrel(
            cloud, np.zeros(3), np.eye(3), lead.hole_center, z_low, z_high,
        )
        for cloud in clouds
    )
    return reach - float(lead.hole_radius_mm)
