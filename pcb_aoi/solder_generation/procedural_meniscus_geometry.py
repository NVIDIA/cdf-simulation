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

"""Stage-free pad-aware procedural solder-meniscus geometry.

The profile and placement calculations in this module are extracted from
``scripts/sdg/editor/editor_solder_fillet_board.py``.  They deliberately use
only numeric, serializable inputs so both the declarative solver and the editor
workflow can consume the same geometry without allowing a solver to mutate a
live USD stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import cos, exp, pi, sin
from random import Random


Bounds = tuple[float, float, float, float, float, float]
Matrix4 = tuple[float, ...]
Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class LeadContactMeasurement:
    """Stage-free measurements of one termination touching one pad.

    Args:
        foot_height: Contact-foot height above the pad surface.
        toe: Outermost contact coordinate along the resolved outward axis.
        width: Contact-foot width across the pad.
        width_center: Contact-foot center across the pad.
        wall_height: Wettable-wall height above the pad surface.
    """

    foot_height: float
    toe: float
    width: float
    width_center: float
    wall_height: float

    def as_payload(self) -> tuple[float, float, float, float, float]:
        """Return a transport-safe numeric representation."""
        return (
            self.foot_height,
            self.toe,
            self.width,
            self.width_center,
            self.wall_height,
        )

    @classmethod
    def from_payload(cls, values: tuple[float, ...]) -> "LeadContactMeasurement":
        """Restore a measurement from its transport-safe representation."""
        if len(values) != 5:
            raise ValueError("lead-contact payload must contain 5 values")
        return cls(*(float(value) for value in values))


@dataclass(frozen=True)
class MeniscusParameters:
    """Geometry and pad-placement settings for one procedural meniscus.

    Args:
        long_edge: Reference length of the unscaled meniscus mesh.
        x_scale: Fraction of the reference length occupied by the mesh run.
        platform_start: Normalized start of the lateral crest plateau.
        platform_end: Normalized end of the lateral crest plateau.
        smoothness_y: Steepness of the lateral sigmoid transitions.
        bump: Crest-height scale relative to the reference length.
        decay: Longitudinal decay used by the legacy non-pad-aware profile.
        pad_decay: Longitudinal decay used by the pad-aware tapered profile.
        noise_scale: Initial spatial frequency of deterministic surface noise.
        noise_octaves: Number of deterministic surface-noise octaves.
        noise_amplitude: Absolute amplitude of deterministic surface noise.
        resolution: Grid-point count along each mesh dimension.
        solid: Whether to close the heightfield with a skirt and bottom cap.
        solid_depth_fraction: Bottom-cap depth relative to crest scale.
        termination_height_ratio: Crest height relative to termination height.
        wetting_overlap_ratio: Fraction of the exposed run placed under the
            component termination.
        z_offset: Placement offset along the board normal in scene units.
        lead_local_sizing: Size and anchor each joint from the termination
            geometry touching its pad instead of the full package bounds.
        scene_units_per_mm: Scene-unit conversion for the internal calibrated
            lead-local placement constants.

    Raises:
        ValueError: If any parameter cannot produce valid geometry.

    Example:

    .. code-block:: python

        parameters = MeniscusParameters(resolution=20, solid=True)
        print(parameters.resolution)
        # 20
    """

    long_edge: float = 9.0
    #: Reference length of the unscaled meniscus mesh.

    x_scale: float = 0.2
    #: Fraction of `long_edge` occupied by the mesh run.

    platform_start: float = 0.2
    #: Normalized start of the lateral crest plateau.

    platform_end: float = 0.8
    #: Normalized end of the lateral crest plateau.

    smoothness_y: float = 6.44
    #: Steepness of the lateral sigmoid transitions.

    bump: float = 0.38
    #: Crest-height scale relative to `long_edge`.

    decay: float = 10.8
    #: Longitudinal decay used by the legacy profile.

    pad_decay: float = 5.0
    #: Longitudinal decay used by the pad-aware profile.

    noise_scale: float = 4.0
    #: Initial spatial frequency of deterministic surface noise.

    noise_octaves: int = 4
    #: Number of deterministic surface-noise octaves.

    noise_amplitude: float = 0.01
    #: Absolute amplitude of deterministic surface noise.

    resolution: int = 20
    #: Grid-point count along each mesh dimension.

    solid: bool = True
    #: Whether to close the heightfield with a skirt and bottom cap.

    solid_depth_fraction: float = 0.15
    #: Bottom-cap depth relative to the crest scale.

    termination_height_ratio: float = 0.9
    #: Crest height relative to the component termination height.

    wetting_overlap_ratio: float = 0.12
    #: Fraction of the exposed run placed under the termination.

    z_offset: float = 0.0
    #: Placement offset along the board normal in scene units.

    lead_local_sizing: bool = True
    #: Whether placement follows the lead foot touching each pad.

    scene_units_per_mm: float = 1.0
    #: Conversion used by internal lead-local placement constants.

    yaw_degrees: float = 0.0
    #: Rotation of the component about the board normal, in degrees.
    #:
    #: Every measurement here reasons about an **axis-aligned** pad:
    #: :func:`resolve_pad_orientation` returns an axis *index* and a sign, so it
    #: can express only four outward directions. That is exact at a quadrant
    #: rotation and wrong in both direction and size at any other, because the
    #: caller's aligned bounds are inflated by the rotation as well.
    #:
    #: The fix leaves that maths untouched and moves the *frame*: the caller
    #: measures bounds and lead points in the component's yaw frame, where the
    #: pad is axis-aligned again, and records the yaw here so
    #: :func:`_world_transform` can rotate the finished meniscus back into world
    #: space. At ``0.0`` every value produced is bit-identical to before.

    def __post_init__(self) -> None:
        """Reject settings that cannot produce a valid mesh.

        Raises:
            ValueError: If any geometry or placement parameter is invalid.
        """
        if self.long_edge <= 0.0:
            raise ValueError("long_edge must be greater than 0")
        if self.x_scale <= 0.0:
            raise ValueError("x_scale must be greater than 0")
        if not 0.0 <= self.platform_start < self.platform_end <= 1.0:
            raise ValueError(
                "platform_start and platform_end must satisfy 0 <= start < end <= 1",
            )
        if self.smoothness_y <= 0.0:
            raise ValueError("smoothness_y must be greater than 0")
        if self.bump <= 0.0:
            raise ValueError("bump must be greater than 0")
        if self.decay <= 0.0 or self.pad_decay <= 0.0:
            raise ValueError("decay and pad_decay must be greater than 0")
        if self.noise_scale < 0.0:
            raise ValueError("noise_scale must be at least 0")
        if self.noise_octaves < 1:
            raise ValueError("noise_octaves must be at least 1")
        if self.noise_amplitude < 0.0:
            raise ValueError("noise_amplitude must be at least 0")
        if self.resolution < 4:
            raise ValueError("resolution must be at least 4")
        if self.solid_depth_fraction < 0.0:
            raise ValueError("solid_depth_fraction must be at least 0")
        if self.termination_height_ratio <= 0.0:
            raise ValueError("termination_height_ratio must be greater than 0")
        if not 0.0 <= self.wetting_overlap_ratio <= 1.0:
            raise ValueError("wetting_overlap_ratio must be between 0 and 1")
        if self.scene_units_per_mm <= 0.0:
            raise ValueError("scene_units_per_mm must be greater than 0")


@dataclass(frozen=True)
class MeniscusMeshData:
    """Transport-safe mesh data and its row-major world transform.

    Args:
        points: Mesh points in local meniscus coordinates.
        face_vertex_counts: Vertex count for every mesh face.
        face_vertex_indices: Flattened vertex indices for all faces.
        transform: Row-major transform from local coordinates to world space.
    """

    points: tuple[tuple[float, float, float], ...]
    #: Mesh points in local meniscus coordinates.

    face_vertex_counts: tuple[int, ...]
    #: Vertex count for every mesh face.

    face_vertex_indices: tuple[int, ...]
    #: Flattened vertex indices for all faces.

    transform: Matrix4
    #: Row-major transform from local coordinates to world space.


def _sigmoid(value: float) -> float:
    """Return the logistic sigmoid for one scalar.

    Args:
        value: Scalar input value.

    Returns:
        Logistic sigmoid value.
    """
    return 1.0 / (1.0 + exp(-value))


def _noise_terms(parameters: MeniscusParameters) -> tuple[tuple[float, ...], ...]:
    """Build deterministic phase and rotation values for surface noise.

    Args:
        parameters: Meniscus settings controlling the octave count.

    Returns:
        Tuple containing X phase, Y phase, and rotation values per octave.
    """
    rng = Random(42)
    return tuple(
        (rng.random() * 2.0 * pi, rng.random() * 2.0 * pi, rng.random() * 2.0 * pi)
        for _ in range(parameters.noise_octaves)
    )


@lru_cache(maxsize=16)
def compute_meniscus_geometry(
    parameters: MeniscusParameters,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Compute the established tapered heightfield as plain numeric arrays.

    Args:
        parameters: Validated meniscus geometry settings.

    Returns:
        Tuple containing points, face-vertex counts, and face-vertex indices.

    Example:

    .. code-block:: python

        parameters = MeniscusParameters(resolution=4, solid=False)
        points, counts, indices = compute_meniscus_geometry(parameters)
        print(len(points), len(counts), len(indices))
        # 16 18 54
    """
    resolution = parameters.resolution
    mesh_length = parameters.long_edge * parameters.x_scale
    mesh_width = parameters.long_edge * 0.6
    noise_terms = _noise_terms(parameters)
    points: list[tuple[float, float, float]] = []

    for row in range(resolution):
        y = mesh_width * row / (resolution - 1)
        yn = y / mesh_width
        lateral = _sigmoid(
            parameters.smoothness_y * (yn - parameters.platform_start),
        ) * (
            1.0
            - _sigmoid(
                parameters.smoothness_y * (yn - parameters.platform_end),
            )
        )
        for column in range(resolution):
            x = mesh_length * column / (resolution - 1)
            xn = x / parameters.long_edge
            decay_value = exp(-parameters.pad_decay * xn)
            decay_end = exp(-parameters.pad_decay * parameters.x_scale)
            tapered = (
                (decay_value - decay_end) / (1.0 - decay_end)
                if decay_end < 1.0
                else 0.0
            )
            height = parameters.long_edge * parameters.bump * tapered * lateral

            amplitude = 1.0
            frequency = parameters.noise_scale
            noise = 0.0
            for phase_x, phase_y, angle in noise_terms:
                xr = xn * cos(angle) - yn * sin(angle)
                yr = xn * sin(angle) + yn * cos(angle)
                noise += amplitude * (
                    sin(2.0 * pi * frequency * xr + phase_x)
                    * cos(2.0 * pi * frequency * yr + phase_y)
                )
                amplitude *= 0.5
                frequency *= 2.0
            height += parameters.noise_amplitude * noise
            points.append((x, y, height))

    counts: list[int] = []
    indices: list[int] = []
    for row in range(resolution - 1):
        for column in range(resolution - 1):
            a = row * resolution + column
            b = a + 1
            c = (row + 1) * resolution + column
            d = c + 1
            counts.extend((3, 3))
            indices.extend((a, b, c, b, d, c))

    if parameters.solid:
        ring = (
            list(range(resolution))
            + [row * resolution + resolution - 1 for row in range(1, resolution)]
            + [
                (resolution - 1) * resolution + column
                for column in range(resolution - 2, -1, -1)
            ]
            + [row * resolution for row in range(resolution - 2, 0, -1)]
        )
        depth = parameters.long_edge * parameters.bump * parameters.solid_depth_fraction
        base = len(points)
        points.extend((points[index][0], points[index][1], -depth) for index in ring)
        for index, top_index in enumerate(ring):
            next_index = (index + 1) % len(ring)
            counts.append(4)
            indices.extend(
                (top_index, base + index, base + next_index, ring[next_index]),
            )
        counts.append(len(ring))
        indices.extend(base + index for index in range(len(ring) - 1, -1, -1))

    return tuple(points), tuple(counts), tuple(indices)


def is_area_array(
    pad_bounds: tuple[Bounds, ...],
    min_interior: int = 4,
) -> bool:
    """Return whether pads form a two-dimensional area array.

    Args:
        pad_bounds: World-space aligned bounds for all component sites.
        min_interior: Minimum strictly interior pad count.

    Returns:
        ``True`` for a covered BGA/LGA-style layout; otherwise ``False``.
    """
    if len(pad_bounds) < 9:
        return False
    centers = tuple(
        (
            (bounds[0] + bounds[3]) * 0.5,
            (bounds[1] + bounds[4]) * 0.5,
        )
        for bounds in pad_bounds
    )
    minimum = (
        min(center[0] for center in centers),
        min(center[1] for center in centers),
    )
    maximum = (
        max(center[0] for center in centers),
        max(center[1] for center in centers),
    )
    tolerance = (
        (maximum[0] - minimum[0]) * 0.12 + 1e-9,
        (maximum[1] - minimum[1]) * 0.12 + 1e-9,
    )
    interior = sum(
        1
        for center in centers
        if (
            minimum[0] + tolerance[0] < center[0] < maximum[0] - tolerance[0]
            and minimum[1] + tolerance[1]
            < center[1]
            < maximum[1] - tolerance[1]
        )
    )
    return interior >= min_interior


def resolve_pad_orientation(
    pad_bounds: Bounds,
    body_bounds: Bounds,
    component_bounds: Bounds,
) -> tuple[float, int, float]:
    """Resolve pad overhang, outward axis, and outward sign.

    Args:
        pad_bounds: World-space aligned pad bounds.
        body_bounds: World-space union bounds of body and terminations.
        component_bounds: World-space aligned component bounds.

    Returns:
        Overhang distance, horizontal axis index, and direction sign.
    """
    pad_min = pad_bounds[:3]
    pad_max = pad_bounds[3:]
    body_min = body_bounds[:3]
    body_max = body_bounds[3:]
    component_min = component_bounds[:3]
    component_max = component_bounds[3:]
    pad_center = tuple((pad_min[index] + pad_max[index]) * 0.5 for index in range(3))
    component_center = tuple(
        (component_min[index] + component_max[index]) * 0.5 for index in range(3)
    )
    overhangs = []
    for axis in (0, 1):
        overhangs.extend(
            (
                (pad_max[axis] - body_max[axis], axis, 1.0),
                (body_min[axis] - pad_min[axis], axis, -1.0),
            )
        )
    overhang, outward_axis, outward_sign = max(overhangs, key=lambda item: item[0])
    if overhang <= 0.0:
        offsets = (
            pad_center[0] - component_center[0],
            pad_center[1] - component_center[1],
        )
        outward_axis = 0 if abs(offsets[0]) >= abs(offsets[1]) else 1
        outward_sign = 1.0 if offsets[outward_axis] >= 0.0 else -1.0
    return overhang, outward_axis, outward_sign


def measure_lead_contact(
    pad_bounds: Bounds,
    points: tuple[Point3, ...],
    side: str,
    outward_axis: int,
    outward_sign: float,
    contact_band: float,
    wall_band: float,
    margin_fraction: float = 0.15,
) -> LeadContactMeasurement | None:
    """Measure the lead foot and wettable wall touching one pad.

    Only points close to the pad surface participate in the contact-foot
    measurement, so a gull-wing heel cannot inflate the joint height.

    Args:
        pad_bounds: World-space aligned pad bounds.
        points: World-space component termination and body mesh points.
        side: Component board side.
        outward_axis: Resolved horizontal outward axis.
        outward_sign: Resolved direction along the outward axis.
        contact_band: Pad-surface band used for the contact foot.
        wall_band: Pad-surface band used for wettable-wall height.
        margin_fraction: Fractional pad-footprint sampling margin.

    Returns:
        Lead-contact measurement, or ``None`` when no geometry touches the pad.
    """
    if not points:
        return None
    pad_min = pad_bounds[:3]
    pad_max = pad_bounds[3:]
    width_axis = 1 - outward_axis
    width_margin = (pad_max[width_axis] - pad_min[width_axis]) * margin_fraction
    outward_margin = (pad_max[outward_axis] - pad_min[outward_axis]) * margin_fraction
    bottom = side.lower().startswith("b")
    pad_surface = pad_min[2] if bottom else pad_max[2]

    def in_footprint(point: Point3, include_outward_margin: bool = True) -> bool:
        margin = outward_margin if include_outward_margin else 0.0
        return (
            pad_min[width_axis] - width_margin
            <= point[width_axis]
            <= pad_max[width_axis] + width_margin
            and pad_min[outward_axis] - margin
            <= point[outward_axis]
            <= pad_max[outward_axis] + margin
        )

    hits = tuple(
        point
        for point in points
        if (
            (
                pad_surface - contact_band
                <= point[2]
                <= pad_surface + contact_band * 0.5
                if bottom
                else pad_surface - contact_band * 0.5
                <= point[2]
                <= pad_surface + contact_band
            )
            and in_footprint(point)
        )
    )
    if not hits:
        return None
    toe = (
        max(point[outward_axis] for point in hits)
        if outward_sign > 0.0
        else min(point[outward_axis] for point in hits)
    )
    strip = max(
        (pad_max[outward_axis] - pad_min[outward_axis]) * 0.15,
        abs(contact_band) * 0.7,
    )
    toe_points = tuple(
        point
        for point in hits
        if (
            point[outward_axis] >= toe - strip
            if outward_sign > 0.0
            else point[outward_axis] <= toe + strip
        )
    )
    height_points = toe_points or hits
    foot_z = (
        min(point[2] for point in height_points)
        if bottom
        else max(point[2] for point in height_points)
    )
    foot_height = pad_surface - foot_z if bottom else foot_z - pad_surface
    width_min = min(point[width_axis] for point in hits)
    width_max = max(point[width_axis] for point in hits)
    walls = tuple(
        point
        for point in points
        if (
            in_footprint(point, include_outward_margin=False)
            and (
                pad_surface - wall_band
                <= point[2]
                <= pad_surface + contact_band * 0.5
                if bottom
                else pad_surface - contact_band * 0.5
                <= point[2]
                <= pad_surface + wall_band
            )
        )
    )
    if walls:
        wall_z = min(point[2] for point in walls) if bottom else max(point[2] for point in walls)
        wall_height = pad_surface - wall_z if bottom else wall_z - pad_surface
    else:
        wall_height = 0.0
    return LeadContactMeasurement(
        foot_height=max(0.0, float(foot_height)),
        toe=float(toe),
        width=float(width_max - width_min),
        width_center=float((width_max + width_min) * 0.5),
        wall_height=max(0.0, float(wall_height)),
    )


def _world_transform(
    origin: tuple[float, float, float],
    outward_axis: int,
    outward_sign: float,
    scale_x: float,
    scale_y: float,
    scale_z: float,
    *,
    bottom: bool,
    yaw_degrees: float = 0.0,
) -> Matrix4:
    """Build a row-major transform from local meniscus space to world space.

    Args:
        origin: Origin of the local meniscus mesh, in the yaw frame.
        outward_axis: Horizontal axis along which the meniscus extends.
        outward_sign: Direction along the selected horizontal axis.
        scale_x: Scale along the meniscus run.
        scale_y: Scale across the meniscus width.
        scale_z: Scale along the board normal.
        bottom: Whether the component is mounted on the bottom side.
        yaw_degrees: Component rotation about the board normal. The basis and
            the origin are built in that frame and rotated back here.

    Returns:
        Row-major world transform.
    """
    width_axis = 1 - outward_axis
    x_axis = [0.0, 0.0, 0.0]
    y_axis = [0.0, 0.0, 0.0]
    x_axis[outward_axis] = outward_sign * scale_x
    y_axis[width_axis] = scale_y
    z_sign = -1.0 if bottom else 1.0
    z_axis = [0.0, 0.0, z_sign * scale_z]
    place = list(origin)

    # Post-multiply by a rotation about the board normal. Every row of a
    # row-major transform is a vector in the frame being left, so each one gets
    # the same planar rotation; the Z row is unaffected by construction. Guarded
    # on exact zero so the common quadrant case stays bit-identical.
    if yaw_degrees:
        angle = yaw_degrees * pi / 180.0
        cos_a, sin_a = cos(angle), sin(angle)
        for row in (x_axis, y_axis, z_axis, place):
            rx, ry = row[0], row[1]
            row[0] = rx * cos_a - ry * sin_a
            row[1] = rx * sin_a + ry * cos_a

    return (
        x_axis[0],
        x_axis[1],
        x_axis[2],
        0.0,
        y_axis[0],
        y_axis[1],
        y_axis[2],
        0.0,
        z_axis[0],
        z_axis[1],
        z_axis[2],
        0.0,
        place[0],
        place[1],
        place[2],
        1.0,
    )


def build_pad_meniscus(
    pad_bounds: Bounds,
    body_bounds: Bounds,
    component_bounds: Bounds,
    side: str,
    pad_count: int,
    parameters: MeniscusParameters,
    lead_contact: LeadContactMeasurement | None = None,
) -> MeniscusMeshData | None:
    """Build one pad-aware meniscus using the established placement rules.

    Args:
        pad_bounds: World-space aligned bounds of one paste or contact site.
        body_bounds: World-space union bounds of the body and terminations.
        component_bounds: World-space aligned bounds of the component instance.
        side: Board side associated with the component instance.
        pad_count: Number of resolved sites for the component.
        parameters: Validated meniscus geometry and placement settings.
        lead_contact: Optional per-pad contact-foot measurement from
            :func:`measure_lead_contact`.

    Returns:
        Generated mesh and transform, or None when the bounds are degenerate.

    Example:

    .. code-block:: python

        mesh = build_pad_meniscus(
            pad_bounds=(0.5, -0.5, 0.0, 1.0, 0.5, 0.1),
            body_bounds=(-0.6, -0.4, 0.1, 0.6, 0.4, 0.8),
            component_bounds=(-0.8, -0.5, 0.1, 0.8, 0.5, 0.8),
            side="Top",
            pad_count=2,
            parameters=MeniscusParameters(),
        )
        print(mesh is not None)
        # True
    """
    pad_min = pad_bounds[:3]
    pad_max = pad_bounds[3:]
    body_min = body_bounds[:3]
    body_max = body_bounds[3:]
    pad_center = tuple((pad_min[index] + pad_max[index]) * 0.5 for index in range(3))
    overhang, outward_axis, outward_sign = resolve_pad_orientation(
        pad_bounds,
        body_bounds,
        component_bounds,
    )
    width_axis = 1 - outward_axis

    termination_face = (
        body_max[outward_axis] if outward_sign > 0.0 else body_min[outward_axis]
    )
    pad_outer = pad_max[outward_axis] if outward_sign > 0.0 else pad_min[outward_axis]
    bottom = side.lower().startswith("b")
    pad_surface = pad_min[2] if bottom else pad_max[2]
    mm = parameters.scene_units_per_mm
    lead_max_height = 2.0 * mm
    lead_generic_height = 0.3 * mm
    lead_generic_width = 1.0 * mm
    lead_run_minimum = 0.25 * mm
    termination_height = (
        pad_surface - body_min[2] if bottom else body_max[2] - pad_surface
    )
    lead_width = None
    lead_width_center = None
    run_extra = 0.0
    if parameters.lead_local_sizing:
        if lead_contact is not None:
            foot_height = lead_contact.foot_height
            toe = lead_contact.toe
            lead_width = lead_contact.width
            lead_width_center = lead_contact.width_center
            wall_height = lead_contact.wall_height
        else:
            foot_height = wall_height = 0.0
        if foot_height > 0.0:
            if pad_count <= 2:
                climb = 0.8
                if wall_height > 1.2 * mm:
                    climb = 0.45
                    run_extra = 0.4 * mm
                termination_height = min(
                    max(foot_height, climb * wall_height),
                    lead_max_height,
                )
            else:
                termination_height = min(
                    foot_height,
                    lead_max_height,
                )
            termination_face = min(
                max(toe, pad_min[outward_axis]),
                pad_max[outward_axis],
            )
        elif overhang > 0.0:
            termination_height = min(
                lead_generic_height,
                lead_max_height,
            )
            lead_width = None
        else:
            return None
    if termination_height <= 0.0:
        termination_height = body_max[2] - body_min[2]
    if termination_height <= 0.0:
        return None

    available = (pad_outer - termination_face) * outward_sign
    run = (available if available > 0.0 else 0.2 * termination_height) + run_extra
    if parameters.lead_local_sizing and pad_count > 2:
        run = min(
            run,
            max(
                2.0 * termination_height,
                lead_run_minimum,
            ),
        )
        run = max(run, lead_run_minimum)
    wetting_overlap = run * parameters.wetting_overlap_ratio
    length = run + wetting_overlap
    body_width = body_max[width_axis] - body_min[width_axis]
    pad_width = pad_max[width_axis] - pad_min[width_axis]
    width = min(body_width, pad_width) if pad_count > 2 else body_width
    if parameters.lead_local_sizing:
        if pad_count <= 2:
            width = pad_width
        elif lead_width is not None and lead_width > 0.0:
            width = min(pad_width, lead_width * 1.2)
        else:
            width = min(pad_width, lead_generic_width)
    if length <= 0.0 or width <= 0.0:
        return None

    mesh_length = parameters.long_edge * parameters.x_scale
    mesh_width = parameters.long_edge * 0.6
    lateral_peak = _sigmoid(
        parameters.smoothness_y * (0.5 - parameters.platform_start),
    ) * (
        1.0
        - _sigmoid(
            parameters.smoothness_y * (0.5 - parameters.platform_end),
        )
    )
    mesh_height = parameters.long_edge * parameters.bump * lateral_peak
    if mesh_length <= 0.0 or mesh_width <= 0.0 or mesh_height <= 0.0:
        return None

    height = parameters.termination_height_ratio * termination_height
    origin = list(pad_center)
    origin[outward_axis] = termination_face - outward_sign * wetting_overlap
    width_center = (
        lead_width_center
        if parameters.lead_local_sizing and lead_width_center is not None
        else pad_center[width_axis]
    )
    if width <= pad_width:
        width_center = min(
            max(width_center, pad_min[width_axis] + width * 0.5),
            pad_max[width_axis] - width * 0.5,
        )
    else:
        width_center = pad_center[width_axis]
    origin[width_axis] = width_center - width * 0.5
    origin[2] = pad_surface + (-parameters.z_offset if bottom else parameters.z_offset)
    points, counts, indices = compute_meniscus_geometry(parameters)
    return MeniscusMeshData(
        points=points,
        face_vertex_counts=counts,
        face_vertex_indices=indices,
        transform=_world_transform(
            tuple(origin),
            outward_axis,
            outward_sign,
            length / mesh_length,
            width / mesh_width,
            height / mesh_height,
            bottom=bottom,
            yaw_degrees=parameters.yaw_degrees,
        ),
    )
