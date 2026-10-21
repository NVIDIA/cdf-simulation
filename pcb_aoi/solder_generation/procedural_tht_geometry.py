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

"""Build stage-free procedural through-hole solder volcano meshes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import atan2, cos, pi, sin, sqrt

from pcb_aoi.geometry.planar import cross_xy
from pcb_aoi.geometry.plane_cut import (
    close_open_boundaries,
)
from pcb_aoi.pin_tip import select_pin_tip_faces

from .procedural_meniscus_geometry import Matrix4

IDENTITY_MATRIX_4X4: Matrix4 = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)
#: Bath film thickness as a fraction of the lead's half-width.
#:
#: The film has to be thick enough that a lead leaning by the amount
#: `tilt_range` allows never comes out through it. The throat is taken from the
#: lead's real cross-section, but at ONE height, and every ring of the volcano
#: is then struck from that single centre -- so the solder fits the lead exactly
#: there, with no clearance at all, and a lead that leans grazes the surface
#: everywhere else. The slivers of bare copper along that crease come back as
#: `pin` scattered through the solder mask.
#:
#: Measured on the ROI capture, which is what ships, over all four leads of
#: X204 with `tilt` at 12-15 degrees: 0.03 leaves up to 138 stray pixels in as
#: many as 13 components, 0.06 still scatters 20 to 36 on two of the four, and
#: 0.10 and 0.15 come back as one component with nothing stray on every lead
#: and every axis. 0.10 is the smallest that holds. It costs nothing visually:
#: the same frame at 0.03 and 0.10 differs nowhere the eye can find.
#:
#: Three other things were tried and are recorded so they are not tried again.
#: Standing the bore off the lead by a fixed clearance improves the count
#: monotonically and never clears it -- 138, 76, 63 stray at 0.00, 0.02, 0.05 mm
#: -- because it only opens the throat, while the lead grazes the surface above
#: it. Leaning the bore with the lead made the mask WORSE at every thickness,
#: 631, 297 and 30 against 38, 0 and 0, since the rings then no longer share a
#: centre and the surface creases where they cross. And estimating the lead's
#: axis from its point cloud tore the fillet into facets, because a lead sheared
#: at 25 to 45 degrees has an asymmetric end and reads as leaning when it is not.
PIN_BATH_THICKNESS_RATIO = 0.10
PIN_BATH_BLEND_DEPTH_RATIO = 0.25
PIN_BATH_TRANSITION_RINGS = 12
#: Additional rings on each side of the solder-bath neck transition.
#:
#: The analytic meniscus is vertical where it meets the pin sleeve, but one
#: long first polygon only approximates that tangent. These short rings and a
#: quintic smootherstep keep both ends close to the shared tangent and prevent
#: the join from rendering as a crease, including around rectangular corners.


@dataclass(frozen=True)
class ThroughHoleParameters:
    """Settings controlling one annular through-hole solder volcano.

    Args:
        resolution: Angular point count around each ring.
        radial_segments: Ring count between the pin throat and pad rim.
        height_ratio: Crest height relative to exposed annulus width.
        solid: Whether to close the annular surface into a volume.
        solid_depth_fraction: In-board depth relative to crest height.
        z_offset: Placement offset along the outward board normal.
        pad_edge_inset: Radial clearance between solder and the pad edge.
        solder_bath: Whether to enable the primary solder-bath behavior of extending
            the joint to envelop the exposed solder-side pin.
        profile_exponent: Fillet fullness control; values below one raise the
            intermediate surface without moving the throat or pad rim.
        crease_angle_deg: Optional dihedral threshold for splitting shading normals
            at geometric creases while keeping shallow surface transitions smooth.

    Raises:
        ValueError: If a setting cannot produce valid geometry.

    Example:

    .. code-block:: python

        parameters = ThroughHoleParameters(resolution=20, solid=True)
        print(parameters.radial_segments)
        # 5

    """

    resolution: int = 20
    #: Angular point count around each ring.

    radial_segments: int = 5
    #: Ring count between the pin throat and pad rim.

    height_ratio: float = 0.9
    #: Crest height relative to exposed annulus width.

    solid: bool = True
    #: Whether to close the annular surface into a volume.

    solid_depth_fraction: float = 0.15
    #: In-board depth relative to crest height.

    z_offset: float = 0.0
    #: Placement offset along the outward board normal.

    pad_edge_inset: float = 0.0
    #: Radial clearance between solder and the pad edge.

    solder_bath: bool = False
    #: Whether to enable the primary behavior of enveloping the exposed solder-side pin.

    profile_exponent: float = 1.0
    #: Exponent in (0.5, 1.0]; lower values make the fillet fuller at fixed endpoints.

    crease_angle_deg: float | None = None
    #: Optional normal-splitting threshold in degrees; None preserves legacy shading.

    def __post_init__(self) -> None:
        """Reject settings that cannot produce valid geometry.

        Raises:
            ValueError: If a geometry setting is invalid.

        """
        if self.resolution < 8:
            raise ValueError("resolution must be at least 8")
        if self.radial_segments < 2:
            raise ValueError("radial_segments must be at least 2")
        if self.height_ratio <= 0.0:
            raise ValueError("height_ratio must be greater than 0")
        if self.solid_depth_fraction < 0.0:
            raise ValueError("solid_depth_fraction must be at least 0")
        if self.pad_edge_inset < 0.0:
            raise ValueError("pad_edge_inset must be at least 0")
        if not 0.5 < self.profile_exponent <= 1.0:
            raise ValueError("profile_exponent must be greater than 0.5 and at most 1")
        if self.crease_angle_deg is not None and not (
            0.0 < self.crease_angle_deg < 180.0
        ):
            raise ValueError(
                "crease_angle_deg must be greater than 0 and less than 180"
            )


@dataclass(frozen=True)
class ThroughHoleSite:
    """Serializable numeric description of one occupied through-hole pad.

    Args:
        center_xy: World-space pad and hole center.
        pin_radius: Radius of the pin throat.
        pad_radius: Outer solderable pad radius.
        base_z: World-space conductor surface receiving solder.
        grow_toward_negative_z: Whether solder grows below the board.
        pad_shape_xy: Convex pad-boundary offsets for non-circular pads.
        throat_center_xy: World-space pin-throat center when offset from the pad.
        throat_shape_xy: Convex pin-throat boundary offsets for non-circular pins.
        pin_points: World-space points for the associated pin region.
        pin_face_vertex_counts: Vertex count for each pin polygon.
        pin_face_vertex_indices: Flattened pin polygon vertex indices.

    Raises:
        ValueError: If the radii cannot produce an annular surface.

    """

    center_xy: tuple[float, float]
    #: World-space pad and hole center.

    pin_radius: float
    #: Radius of the pin throat.

    pad_radius: float
    #: Outer solderable pad radius.

    base_z: float
    #: World-space conductor surface receiving solder.

    grow_toward_negative_z: bool
    #: Whether solder grows below the board.

    pad_shape_xy: tuple[tuple[float, float], ...] = ()
    #: Convex local pad boundary around ``center_xy`` for non-circular pads.

    throat_center_xy: tuple[float, float] | None = None
    #: World-space pin-throat center; ``None`` uses ``center_xy``.

    throat_shape_xy: tuple[tuple[float, float], ...] = ()
    #: Convex local boundary around ``throat_center_xy`` for non-circular pins.

    pin_points: tuple[tuple[float, float, float], ...] = ()
    #: World-space points for the associated pin region.

    pin_face_vertex_counts: tuple[int, ...] = ()
    #: Vertex count for each pin polygon.

    pin_face_vertex_indices: tuple[int, ...] = ()
    #: Flattened pin polygon vertex indices.

    def __post_init__(self) -> None:
        """Reject degenerate site geometry.

        Raises:
            ValueError: If the pin or pad radii are invalid.

        """
        if len(self.center_xy) != 2:
            raise ValueError("center_xy must contain two values")
        if self.throat_center_xy is not None and len(self.throat_center_xy) != 2:
            raise ValueError("throat_center_xy must contain two values")
        if self.pin_radius <= 0.0:
            raise ValueError("pin_radius must be greater than 0")
        if self.pad_radius <= self.pin_radius:
            raise ValueError("pad_radius must be greater than pin_radius")
        if self.pad_shape_xy and len(self.pad_shape_xy) < 3:
            raise ValueError("pad_shape_xy must contain at least three points")
        if self.throat_shape_xy and len(self.throat_shape_xy) < 3:
            raise ValueError("throat_shape_xy must contain at least three points")
        if bool(self.pin_face_vertex_counts) != bool(self.pin_face_vertex_indices):
            raise ValueError("pin face counts and indices must both be provided")
        if sum(self.pin_face_vertex_counts) != len(self.pin_face_vertex_indices):
            raise ValueError("pin face counts must match the pin index count")
        if self.pin_face_vertex_indices and (
            not self.pin_points
            or min(self.pin_face_vertex_indices) < 0
            or max(self.pin_face_vertex_indices) >= len(self.pin_points)
        ):
            raise ValueError("pin face indices must reference pin points")
        throat_center = self.throat_center_xy or self.center_xy
        if self.throat_shape_xy:
            throat_extent = max(
                (
                    (throat_center[0] + offset[0] - self.center_xy[0]) ** 2
                    + (throat_center[1] + offset[1] - self.center_xy[1]) ** 2
                )
                ** 0.5
                for offset in self.throat_shape_xy
            )
        else:
            throat_extent = (
                (throat_center[0] - self.center_xy[0]) ** 2
                + (throat_center[1] - self.center_xy[1]) ** 2
            ) ** 0.5 + self.pin_radius
        if throat_extent >= self.pad_radius:
            raise ValueError("pin throat must remain inside the pad boundary")


@dataclass(frozen=True)
class ThroughHoleMeshData:
    """Transport-safe through-hole mesh arrays and world transform."""

    points: tuple[tuple[float, float, float], ...]
    #: Mesh points in world space.

    face_vertex_counts: tuple[int, ...]
    #: Vertex count for every polygon.

    face_vertex_indices: tuple[int, ...]
    #: Flattened polygon vertex indices.

    normals: tuple[tuple[float, float, float], ...]
    #: Area-weighted per-point normals for smooth surface shading.

    transform: Matrix4 = IDENTITY_MATRIX_4X4
    #: Identity transform because points are authored in world space.


def _concave_meniscus_profile(value: float) -> float:
    """Return an inward-bowed height profile from pin throat to pad rim."""
    clamped = max(0.0, min(1.0, value))
    return (1.0 - clamped) ** 2


def _tangent_meniscus_profile(value: float) -> float:
    """Return a profile tangent to the solder-bath pin and pad surfaces."""
    clamped = max(0.0, min(1.0, value))
    return 1.0 - sqrt(max(0.0, 1.0 - (1.0 - clamped) ** 2))


def _smootherstep(value: float) -> float:
    """Return a quintic interpolation with flat first and second derivatives."""
    clamped = max(0.0, min(1.0, value))
    return clamped**3 * (clamped * (clamped * 6.0 - 15.0) + 10.0)


def _ray_polygon_radius(
    direction_xy: tuple[float, float],
    polygon_xy: tuple[tuple[float, float], ...],
) -> float | None:
    """Return where a ray from the origin exits a convex polygon."""
    if len(polygon_xy) < 3:
        return None

    direction_length = (direction_xy[0] ** 2 + direction_xy[1] ** 2) ** 0.5
    if direction_length <= 1e-8:
        return None
    direction = (
        direction_xy[0] / direction_length,
        direction_xy[1] / direction_length,
    )

    best_radius = None
    for index, start in enumerate(polygon_xy):
        end = polygon_xy[(index + 1) % len(polygon_xy)]
        edge = (end[0] - start[0], end[1] - start[1])
        denominator = cross_xy(direction, edge)
        if abs(denominator) <= 1e-10:
            continue

        radius = cross_xy(start, edge) / denominator
        segment_fraction = cross_xy(start, direction) / denominator
        if radius < -1e-8 or segment_fraction < -1e-8 or segment_fraction > 1.0 + 1e-8:
            continue
        if best_radius is None or radius < best_radius:
            best_radius = max(0.0, radius)
    return best_radius


def _vertex_normals(
    points: list[tuple[float, float, float]],
    counts: list[int],
    indices: list[int],
) -> tuple[tuple[float, float, float], ...]:
    """Return area-weighted normals shared by adjacent mesh faces."""
    accumulated = [[0.0, 0.0, 0.0] for _point in points]
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        if count < 3:
            continue
        origin = points[face[0]]
        face_normal = [0.0, 0.0, 0.0]
        for corner in range(1, count - 1):
            first = points[face[corner]]
            second = points[face[corner + 1]]
            first_edge = (
                first[0] - origin[0],
                first[1] - origin[1],
                first[2] - origin[2],
            )
            second_edge = (
                second[0] - origin[0],
                second[1] - origin[1],
                second[2] - origin[2],
            )
            face_normal[0] += (
                first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1]
            )
            face_normal[1] += (
                first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2]
            )
            face_normal[2] += (
                first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
            )
        for point_index in face:
            accumulated[point_index][0] += face_normal[0]
            accumulated[point_index][1] += face_normal[1]
            accumulated[point_index][2] += face_normal[2]

    normals = []
    for normal in accumulated:
        length = (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5
        if length <= 1e-12:
            normals.append((0.0, 0.0, 1.0))
            continue
        normals.append(
            (
                normal[0] / length,
                normal[1] / length,
                normal[2] / length,
            ),
        )
    return tuple(normals)


def _fan_triangulate_faces(
    face_vertex_counts: list[int],
    face_vertex_indices: list[int],
) -> tuple[tuple[int, int, int], ...]:
    """Triangulate polygon faces without changing their authored winding."""
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for count in face_vertex_counts:
        face = face_vertex_indices[cursor : cursor + count]
        cursor += count
        if count < 3:
            continue
        triangles.extend(
            (face[0], face[offset], face[offset + 1]) for offset in range(1, count - 1)
        )
    return tuple(triangles)


def _expanded_pin_bath_points(
    site: ThroughHoleSite,
) -> tuple[tuple[float, float, float], ...]:
    """Return pin points offset outward by a small solder-bath thickness."""
    if not site.pin_points or not site.pin_face_vertex_counts:
        return ()
    pin_points = list(site.pin_points)
    pin_normals = _vertex_normals(
        pin_points,
        list(site.pin_face_vertex_counts),
        list(site.pin_face_vertex_indices),
    )
    center = tuple(
        (
            min(point[axis] for point in pin_points)
            + max(point[axis] for point in pin_points)
        )
        * 0.5
        for axis in range(3)
    )
    thickness = site.pin_radius * PIN_BATH_THICKNESS_RATIO
    direction = -1.0 if site.grow_toward_negative_z else 1.0
    tip_selection = select_pin_tip_faces(
        pin_points,
        site.pin_face_vertex_counts,
        site.pin_face_vertex_indices,
        direction=direction,
    )
    tip_indices = (
        {index for face in tip_selection[0] for index in face}
        if tip_selection is not None
        else set()
    )
    expanded = []
    for index, (point, normal) in enumerate(
        zip(pin_points, pin_normals, strict=True),
    ):
        outward = normal
        if sum(normal[axis] * (point[axis] - center[axis]) for axis in range(3)) < 0.0:
            outward = tuple(-value for value in normal)
        expanded_point = tuple(
            point[axis] + thickness * outward[axis] for axis in range(3)
        )
        if index in tip_indices:
            expanded_point = (
                expanded_point[0],
                expanded_point[1],
                point[2] + direction * thickness,
            )
        expanded.append(expanded_point)
    return tuple(expanded)


def _clipped_pin_bath_mesh(
    site: ThroughHoleSite,
    direction: float,
    clip_z: float,
    expanded: tuple[tuple[float, float, float], ...] | None = None,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Return the expanded pin shell clipped to the solder side of a plane."""
    if expanded is None:
        expanded = _expanded_pin_bath_points(site)
    if not expanded:
        return (), (), ()

    points: list[tuple[float, float, float]] = []
    counts: list[int] = []
    indices: list[int] = []
    point_indices: dict[tuple[float, float, float], int] = {}

    def add_point(point: tuple[float, float, float]) -> int:
        key = tuple(round(value, 12) for value in point)
        if key not in point_indices:
            point_indices[key] = len(points)
            points.append(point)
        return point_indices[key]

    def is_inside(point: tuple[float, float, float]) -> bool:
        return direction * (point[2] - clip_z) >= -1e-9

    def intersection(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        fraction = (clip_z - start[2]) / (end[2] - start[2])
        return (
            start[0] + (end[0] - start[0]) * fraction,
            start[1] + (end[1] - start[1]) * fraction,
            clip_z,
        )

    offset = 0
    for count in site.pin_face_vertex_counts:
        face_indices = site.pin_face_vertex_indices[offset : offset + count]
        offset += count
        face = [expanded[index] for index in face_indices]
        clipped: list[tuple[float, float, float]] = []
        for start, end in zip(face, face[1:] + face[:1], strict=True):
            start_inside = is_inside(start)
            end_inside = is_inside(end)
            if start_inside:
                clipped.append(start)
            if start_inside != end_inside:
                clipped.append(intersection(start, end))

        clipped_indices = [add_point(point) for point in clipped]
        if len(set(clipped_indices)) < 3:
            continue
        counts.append(len(clipped_indices))
        indices.extend(clipped_indices)

    return tuple(points), tuple(counts), tuple(indices)


def _clip_boundary_loop(
    pin_points: tuple[tuple[float, float, float], ...],
    pin_counts: tuple[int, ...],
    pin_indices: tuple[int, ...],
    clip_z: float,
    center_xy: tuple[float, float],
) -> tuple[int, ...]:
    """Return the ordered outer boundary created by clipping a pin shell."""
    edge_counts: dict[tuple[int, int], int] = {}
    offset = 0
    for count in pin_counts:
        face = pin_indices[offset : offset + count]
        offset += count
        for first, second in zip(face, face[1:] + face[:1], strict=True):
            edge = tuple(sorted((first, second)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary = {
        index
        for edge, count in edge_counts.items()
        if count == 1
        and all(abs(pin_points[index][2] - clip_z) <= 1e-8 for index in edge)
        for index in edge
    }
    return tuple(
        sorted(
            boundary,
            key=lambda index: atan2(
                pin_points[index][1] - center_xy[1],
                pin_points[index][0] - center_xy[0],
            ),
        ),
    )


def _append_oriented_face(
    counts: list[int],
    indices: list[int],
    face: tuple[int, ...],
    direction: float,
) -> None:
    """Append an outward-facing transition polygon."""
    oriented = face if direction < 0.0 else tuple(reversed(face))
    counts.append(len(oriented))
    indices.extend(oriented)


def _stitch_boundary_loops(
    points: list[tuple[float, float, float]],
    counts: list[int],
    indices: list[int],
    near_loop: tuple[int, ...],
    far_loop: tuple[int, ...],
    direction: float,
    center_xy: tuple[float, float],
) -> None:
    """Triangulate or quad-stitch two counterclockwise boundary loops."""

    def ordered_events(loop: tuple[int, ...]) -> list[tuple[float, int]]:
        events = sorted(
            (
                atan2(
                    points[index][1] - center_xy[1],
                    points[index][0] - center_xy[0],
                )
                % (2.0 * pi),
                index,
            )
            for index in loop
        )
        return [(events[-1][0] - 2.0 * pi, events[-1][1]), *events]

    near_events = ordered_events(near_loop)
    far_events = ordered_events(far_loop)
    near_count = len(near_events) - 1
    far_count = len(far_events) - 1
    near_step = 0
    far_step = 0
    while near_step < near_count or far_step < far_count:
        near_angle = (
            near_events[near_step + 1][0] if near_step < near_count else float("inf")
        )
        far_angle = (
            far_events[far_step + 1][0] if far_step < far_count else float("inf")
        )
        near_current = near_events[near_step][1]
        far_current = far_events[far_step][1]
        if abs(near_angle - far_angle) <= 1e-12:
            face = (
                near_current,
                far_current,
                far_events[far_step + 1][1],
                near_events[near_step + 1][1],
            )
            near_step += 1
            far_step += 1
        elif near_angle < far_angle:
            face = (
                near_current,
                far_current,
                near_events[near_step + 1][1],
            )
            near_step += 1
        else:
            face = (
                near_current,
                far_current,
                far_events[far_step + 1][1],
            )
            far_step += 1
        _append_oriented_face(counts, indices, face, direction)


def _append_resampled_pin_bath(
    points: list[tuple[float, float, float]],
    counts: list[int],
    indices: list[int],
    pin_points: tuple[tuple[float, float, float], ...],
    pin_counts: tuple[int, ...],
    pin_indices: tuple[int, ...],
    angular: int,
    bath_start_z: float,
    clip_z: float,
    direction: float,
    throat_center: tuple[float, float],
) -> tuple[tuple[int, ...], ...] | None:
    """Append a pin bath through a smooth resampled neck transition."""
    boundary = _clip_boundary_loop(
        pin_points,
        pin_counts,
        pin_indices,
        clip_z,
        throat_center,
    )
    if len(boundary) < 3:
        return None

    point_offset = len(points)
    points.extend(pin_points)

    offset = 0
    for count in pin_counts:
        face = pin_indices[offset : offset + count]
        offset += count
        counts.append(count)
        indices.extend(point_offset + pin_index for pin_index in face)

    boundary_xy = tuple(
        (
            pin_points[index][0] - throat_center[0],
            pin_points[index][1] - throat_center[1],
        )
        for index in boundary
    )
    neck_loop = tuple(range(angular))
    transition_loop = neck_loop
    transition_loops = [neck_loop]
    for ring in range(1, PIN_BATH_TRANSITION_RINGS + 1):
        fraction = ring / float(PIN_BATH_TRANSITION_RINGS + 1)
        smooth_fraction = _smootherstep(fraction)
        transition_z = bath_start_z + (clip_z - bath_start_z) * fraction
        transition_start = len(points)
        for segment in range(angular):
            angle = 2.0 * pi * segment / angular
            direction_xy = (cos(angle), sin(angle))
            radius = _ray_polygon_radius(direction_xy, boundary_xy)
            if radius is None:
                radius = min(
                    (
                        (offset_xy[0] ** 2 + offset_xy[1] ** 2) ** 0.5
                        for offset_xy in boundary_xy
                    ),
                )
            neck_point = points[neck_loop[segment]]
            target_x = throat_center[0] + radius * direction_xy[0]
            target_y = throat_center[1] + radius * direction_xy[1]
            points.append(
                (
                    neck_point[0] + (target_x - neck_point[0]) * smooth_fraction,
                    neck_point[1] + (target_y - neck_point[1]) * smooth_fraction,
                    transition_z,
                ),
            )

        next_loop = tuple(transition_start + index for index in range(angular))
        _stitch_boundary_loops(
            points,
            counts,
            indices,
            transition_loop,
            next_loop,
            direction,
            throat_center,
        )
        transition_loop = next_loop
        transition_loops.append(next_loop)

    bath_boundary = tuple(point_offset + index for index in boundary)
    _stitch_boundary_loops(
        points,
        counts,
        indices,
        transition_loop,
        bath_boundary,
        direction,
        throat_center,
    )
    transition_loops.append(bath_boundary)
    return tuple(transition_loops)


def _smooth_resampled_transition_normals(
    points: list[tuple[float, float, float]],
    normals: list[tuple[float, float, float]],
    loops: tuple[tuple[int, ...], ...],
    center_xy: tuple[float, float],
) -> None:
    """Blend authored normals across a topology-resampling transition."""
    if len(loops) < 3:
        return
    neck_loop = loops[0]
    bath_loop = loops[-1]
    bath_by_angle = tuple(
        sorted(
            (
                atan2(
                    points[index][1] - center_xy[1],
                    points[index][0] - center_xy[0],
                )
                % (2.0 * pi),
                index,
            )
            for index in bath_loop
        )
    )

    def bath_normal(angle: float) -> tuple[float, float, float]:
        events = (
            (bath_by_angle[-1][0] - 2.0 * pi, bath_by_angle[-1][1]),
            *bath_by_angle,
            (bath_by_angle[0][0] + 2.0 * pi, bath_by_angle[0][1]),
        )
        for (start_angle, start_index), (end_angle, end_index) in pairwise(events):
            adjusted = angle
            if adjusted < start_angle:
                adjusted += 2.0 * pi
            if start_angle <= adjusted <= end_angle:
                span = end_angle - start_angle
                fraction = (adjusted - start_angle) / span if span > 1e-12 else 0.0
                start_normal = normals[start_index]
                end_normal = normals[end_index]
                if (
                    sum(start_normal[axis] * end_normal[axis] for axis in range(3))
                    < 0.0
                ):
                    end_normal = tuple(-value for value in end_normal)
                interpolated = tuple(
                    start_normal[axis]
                    + (end_normal[axis] - start_normal[axis]) * fraction
                    for axis in range(3)
                )
                length = sqrt(sum(value * value for value in interpolated))
                if length > 1e-12:
                    return tuple(value / length for value in interpolated)
                return normals[start_index]
        return normals[bath_by_angle[0][1]]

    denominator = len(loops) - 1
    for loop_number, loop in enumerate(loops[1:-1], 1):
        fraction = _smootherstep(loop_number / float(denominator))
        for segment, index in enumerate(loop):
            start = normals[neck_loop[segment]]
            point = points[index]
            angle = atan2(
                point[1] - center_xy[1],
                point[0] - center_xy[0],
            ) % (2.0 * pi)
            end = bath_normal(angle)
            if sum(start[axis] * end[axis] for axis in range(3)) < 0.0:
                end = tuple(-value for value in end)
            blended = tuple(
                start[axis] + (end[axis] - start[axis]) * fraction for axis in range(3)
            )
            length = sqrt(sum(value * value for value in blended))
            if length > 1e-12:
                normals[index] = tuple(value / length for value in blended)


def _crest_within_lead(
    site: ThroughHoleSite,
    direction: float,
    crest_height: float,
) -> float:
    """Return the crest height, never taller than the lead it climbs.

    The crest is derived from the pad and pin RADII alone, which is right while
    the lead is long and wrong the moment it is not. `height_variation`
    shortens the lead, and a lead shorter than the crest leaves the throat with
    nothing to close against.

    Measured with pad 0.9, pin 0.35 and height_ratio 0.9, so a crest of 0.495:
    at protrusion 0.35, 0.25 and 0.15 the solder ran 0.145, 0.245 and 0.345 mm
    past the pin's tip and closed into a TORUS -- zero boundary edges around an
    empty throat, where a joint on a long lead leaves 28 boundary edges for the
    pin's own cut face. That torus is what renders as a hole bored through the
    middle of the fillet.

    Applied in both places a crest is derived. Clamping only the first left the
    fallback branch to recompute it from the radii and silently undo the clamp,
    which measured as no change at all.

    Args:
        site: Through-hole site, whose `pin_points` give the real protrusion.
        direction: +1 or -1 along the board normal.
        crest_height: Crest height derived from the annulus.

    Returns:
        The crest height to use.

    """
    if not site.pin_points:
        return crest_height
    protrusion = max(
        0.0,
        max(direction * (point[2] - site.base_z) for point in site.pin_points),
    )
    return min(crest_height, protrusion)


def build_through_hole_volcano(
    site: ThroughHoleSite,
    parameters: ThroughHoleParameters,
) -> ThroughHoleMeshData:
    """Build an annular solder volcano around one through-hole pin.

    Args:
        site: Numeric pin, pad, and board-surface geometry.
        parameters: Validated volcano mesh settings.

    Returns:
        Closed or open annular mesh in world coordinates.

    Example:

    .. code-block:: python

        mesh = build_through_hole_volcano(
            ThroughHoleSite(
                center_xy=(0.0, 0.0),
                pin_radius=0.3,
                pad_radius=0.8,
                base_z=0.0,
                grow_toward_negative_z=True,
            ),
            ThroughHoleParameters(),
        )
        print(len(mesh.points) > 0)
        # True

    """
    angular = parameters.resolution
    radial = parameters.radial_segments
    direction = -1.0 if site.grow_toward_negative_z else 1.0
    bath_thickness = (
        site.pin_radius * PIN_BATH_THICKNESS_RATIO if parameters.solder_bath else 0.0
    )
    annulus_width = site.pad_radius - site.pin_radius - bath_thickness
    if annulus_width <= 0.0:
        raise ValueError("solder-bath pin throat must remain inside the pad boundary")
    nominal_crest_height = annulus_width * parameters.height_ratio
    nominal_crest_height = _crest_within_lead(site, direction, nominal_crest_height)
    base_z = site.base_z + direction * parameters.z_offset
    bath_clip_z = base_z + direction * nominal_crest_height
    throat_center = site.throat_center_xy or site.center_xy
    throat_shape_xy = site.throat_shape_xy
    volcano_crest_height = nominal_crest_height
    bath_join_z = bath_clip_z
    bath_points: tuple[tuple[float, float, float], ...] = ()
    bath_counts: tuple[int, ...] = ()
    bath_indices: tuple[int, ...] = ()
    bath_ready = False
    if parameters.solder_bath:
        expanded_pin_points = _expanded_pin_bath_points(site)
        outboard_depth = max(
            (direction * (point[2] - bath_clip_z) for point in expanded_pin_points),
            default=0.0,
        )
        inboard_depth = max(
            (-direction * (point[2] - bath_clip_z) for point in expanded_pin_points),
            default=0.0,
        )
        blend_depth = min(
            site.pin_radius * PIN_BATH_BLEND_DEPTH_RATIO,
            nominal_crest_height * 0.25,
            outboard_depth * 0.25,
            inboard_depth * 0.25,
        )
        if blend_depth > 1e-8:
            bath_join_z = bath_clip_z - direction * blend_depth
            bath_points, bath_counts, bath_indices = _clipped_pin_bath_mesh(
                site,
                direction,
                bath_clip_z,
                expanded_pin_points,
            )
            join_points, join_counts, join_indices = _clipped_pin_bath_mesh(
                site,
                direction,
                bath_join_z,
                expanded_pin_points,
            )
            join_boundary = _clip_boundary_loop(
                join_points,
                join_counts,
                join_indices,
                bath_join_z,
                throat_center,
            )
            if bath_points and len(join_boundary) >= 3:
                join_polygon = tuple(join_points[index] for index in join_boundary)
                throat_center = (
                    (
                        min(point[0] for point in join_polygon)
                        + max(point[0] for point in join_polygon)
                    )
                    * 0.5,
                    (
                        min(point[1] for point in join_polygon)
                        + max(point[1] for point in join_polygon)
                    )
                    * 0.5,
                )
                throat_shape_xy = tuple(
                    (
                        point[0] - throat_center[0],
                        point[1] - throat_center[1],
                    )
                    for point in join_polygon
                )
                volcano_crest_height = nominal_crest_height - blend_depth
                bath_ready = True

    if parameters.solder_bath and not bath_ready:
        annulus_width = site.pad_radius - site.pin_radius
        nominal_crest_height = _crest_within_lead(
            site, direction, annulus_width * parameters.height_ratio
        )
        volcano_crest_height = nominal_crest_height
        bath_clip_z = base_z + direction * nominal_crest_height
        bath_join_z = bath_clip_z
        throat_center = site.throat_center_xy or site.center_xy
        throat_shape_xy = site.throat_shape_xy

    outer_ring: list[tuple[float, float]] = []
    ray_samples: list[tuple[float, float, float]] = []
    for segment in range(angular):
        angle = 2.0 * pi * segment / angular
        direction_xy = (cos(angle), sin(angle))
        outer_radius = (
            _ray_polygon_radius(direction_xy, site.pad_shape_xy)
            if site.pad_shape_xy
            else site.pad_radius
        )
        if outer_radius is None:
            outer_radius = site.pad_radius
        outer_x = site.center_xy[0] + outer_radius * direction_xy[0]
        outer_y = site.center_xy[1] + outer_radius * direction_xy[1]
        delta_x = outer_x - throat_center[0]
        delta_y = outer_y - throat_center[1]
        distance = (delta_x**2 + delta_y**2) ** 0.5
        throat_radius = (
            _ray_polygon_radius(
                (delta_x, delta_y),
                throat_shape_xy,
            )
            if throat_shape_xy
            else site.pin_radius
        )
        if throat_radius is None:
            throat_radius = site.pin_radius
        if distance <= throat_radius:
            raise ValueError("pin throat must remain inside the pad boundary")
        outer_ring.append((outer_x, outer_y))
        ray_samples.append((delta_x, delta_y, throat_radius / distance))
    points: list[tuple[float, float, float]] = []

    def ring_index(ring: int, segment: int) -> int:
        return ring * angular + segment % angular

    radial_fractions = [ring / float(radial) for ring in range(radial + 1)]
    if bath_ready:
        first_fraction = radial_fractions[1]
        radial_fractions = [
            0.0,
            *(
                first_fraction * (ring / float(PIN_BATH_TRANSITION_RINGS + 1)) ** 2
                for ring in range(1, PIN_BATH_TRANSITION_RINGS + 1)
            ),
            *radial_fractions[1:],
        ]

    radial_steps = len(radial_fractions) - 1
    for fraction in radial_fractions:
        profile = (
            _tangent_meniscus_profile(fraction)
            if bath_ready
            else _concave_meniscus_profile(fraction)
        )
        # Keep the legacy arithmetic unchanged unless explicitly requested.
        # Both profiles vanish quadratically at the pad rim, so an exponent
        # greater than 0.5 preserves the horizontal tangent there.
        if parameters.profile_exponent != 1.0:
            profile **= parameters.profile_exponent
        height = volcano_crest_height * profile
        z = base_z + direction * height
        for delta_x, delta_y, throat_fraction in ray_samples:
            ring_fraction = throat_fraction + (1.0 - throat_fraction) * fraction
            points.append(
                (
                    throat_center[0] + delta_x * ring_fraction,
                    throat_center[1] + delta_y * ring_fraction,
                    z,
                ),
            )

    counts: list[int] = []
    indices: list[int] = []
    for ring in range(radial_steps):
        for segment in range(angular):
            a = ring_index(ring, segment)
            b = ring_index(ring, segment + 1)
            c = ring_index(ring + 1, segment)
            d = ring_index(ring + 1, segment + 1)
            if direction < 0.0:
                indices.extend((a, b, c, b, d, c))
            else:
                indices.extend((a, c, b, b, c, d))
            counts.extend((3, 3))

    if parameters.solid:
        depth = max(
            volcano_crest_height * parameters.solid_depth_fraction,
            annulus_width * 0.02,
        )
        bottom_z = base_z - direction * depth
        bottom_inner = len(points)
        for segment in range(angular):
            inner_point = points[ring_index(0, segment)]
            points.append(
                (
                    inner_point[0],
                    inner_point[1],
                    bottom_z,
                ),
            )
        bottom_outer = len(points)
        for outer_x, outer_y in outer_ring:
            points.append(
                (
                    outer_x,
                    outer_y,
                    bottom_z,
                ),
            )

        top_inner = 0
        top_outer = radial_steps * angular
        for segment in range(angular):
            next_segment = (segment + 1) % angular
            faces = []
            if not bath_ready:
                faces.append(
                    (
                        top_inner + segment,
                        top_inner + next_segment,
                        bottom_inner + next_segment,
                        bottom_inner + segment,
                    ),
                )
            faces.extend(
                (
                    (
                        top_outer + segment,
                        bottom_outer + segment,
                        bottom_outer + next_segment,
                        top_outer + next_segment,
                    ),
                    (
                        bottom_inner + segment,
                        bottom_inner + next_segment,
                        bottom_outer + next_segment,
                        bottom_outer + segment,
                    ),
                ),
            )
            for face in faces:
                indices.extend(face if direction > 0.0 else reversed(face))
                counts.append(4)

    transition_loops = None
    if bath_ready:
        transition_loops = _append_resampled_pin_bath(
            points,
            counts,
            indices,
            bath_points,
            bath_counts,
            bath_indices,
            angular,
            bath_join_z,
            bath_clip_z,
            direction,
            throat_center,
        )

    # A joint is a body, so hand back a closed one.
    #
    # On a panelised THT board, generated joints arrive open even when no blade
    # has touched them. The generator leaves a rim wherever the shell meets the plated hole
    # or the lead's end face, and nothing downstream that needs a solid -- a
    # volume, a signed distance, a physical sim -- can use a surface with a
    # hole in it.
    #
    # Sealing HERE rather than in the trim is what makes it a property of the
    # joint: with it on the cut path only, joints that no lead was assigned to
    # would stay open, which is exactly the wrong reason for a
    # joint to be a body or not.
    #
    # Each rim is filled from its OWN vertices, so no point is added and none
    # moves: the silhouette is untouched and `normals`, computed after, still
    # matches the points one-to-one. With solder-bath coating enabled, the
    # translated lead topology already contributes a cap beyond the physical
    # end face. Without it, coplanar rims are capped together as a ring and the
    # lead face remains bare.
    points, counts, indices, _ = close_open_boundaries(points, counts, indices)

    if parameters.crease_angle_deg is not None:
        import numpy as np

        from .mesh_normals import solder_mesh_shading

        triangles = _fan_triangulate_faces(counts, indices)
        shaded_points, shaded_counts, shaded_indices, shaded_normals = (
            solder_mesh_shading(
                np.asarray(points, dtype=float),
                np.asarray(triangles, dtype=int),
                parameters.crease_angle_deg,
            )
        )
        return ThroughHoleMeshData(
            points=tuple(
                tuple(float(value) for value in point) for point in shaded_points
            ),
            face_vertex_counts=shaded_counts,
            face_vertex_indices=shaded_indices,
            normals=shaded_normals,
        )

    normals = list(_vertex_normals(points, counts, indices))
    if transition_loops is not None:
        _smooth_resampled_transition_normals(
            points,
            normals,
            transition_loops,
            throat_center,
        )

    return ThroughHoleMeshData(
        points=tuple(points),
        face_vertex_counts=tuple(counts),
        face_vertex_indices=tuple(indices),
        normals=tuple(normals),
    )
