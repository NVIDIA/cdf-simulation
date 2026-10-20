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

"""Cut a mesh with one plane and cap the opening it leaves.

A trimmed through-hole lead is one plane through one solid: the blade does not
know whether it is passing through copper or solder, and neither does this. The
same cut is applied to the lead and to the joint that was grown on it, after
every other variation has been applied, so nothing downstream has to understand
the cut at all.

Capping is the part that needs care. A lead's section is a disc and can be
filled from its own centroid; a joint's section is an ANNULUS, and filling that
from a centroid would paste solder straight over the lead's own face, which is
coplanar with it. So loops are chained, nested by containment, and an outer
loop with a hole in it is stitched to that hole rather than fanned.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: Points closer to the plane than this count as lying on it. Vertices that a
#: previous operation placed on the cut plane -- which `pin_variation`'s trim
#: does exactly -- must not be classified as either side, or the cut spawns
#: zero-area slivers along the whole rim.
ON_PLANE_EPSILON = 1.0e-9

Point = tuple[float, float, float]
Plane = tuple[Point, Point]



def quantized_keys(
    points: Sequence[Point],
    quantum: float,
    *,
    mode: str = "round",
) -> list[tuple[int, int, int]]:
    """Return each point's integer bucket, for merging coincident points.

    Two different callers here bucket points to decide which ones are the same
    point, and both did it one ``math.floor`` or ``round`` per axis per point.
    On a lead mesh that is the single hottest loop in the cutter; batching it
    changes no bucket, because numpy rounds half to even exactly as Python's
    builtin ``round`` does and floors identically.

    Args:
        points: Points to bucket.
        quantum: Bucket edge length. Must be positive.
        mode: ``"round"`` to bucket by nearest, ``"floor"`` by lower corner.

    Returns:
        One ``(x, y, z)`` integer key per point.

    Example:

    .. code-block:: python

        print(quantized_keys([(0.26, 0.0, 0.0)], 0.1))
        # [(3, 0, 0)]

    """
    if not len(points):
        return []
    import numpy as np

    scaled = np.asarray(points, dtype=np.float64).reshape(-1, 3) / quantum
    rounded = np.floor(scaled) if mode == "floor" else np.round(scaled)
    return [tuple(key) for key in rounded.astype(np.int64).tolist()]

def signed_distance(point: Point, plane: Plane) -> float:
    """Return the signed distance from a point to a plane.

    Args:
        point: Point to measure.
        plane: ``(point_on_plane, unit_normal)``.

    Returns:
        Positive on the side the normal points to.
    """
    origin, normal = plane
    return sum(
        (point[axis] - origin[axis]) * normal[axis] for axis in range(3)
    )


def _interpolate(first: Point, second: Point, ratio: float) -> Point:
    """Return the point a given fraction along a segment."""
    return tuple(
        first[axis] + ratio * (second[axis] - first[axis]) for axis in range(3)
    )


def _plane_basis(normal: Point) -> tuple[Point, Point]:
    """Return two unit vectors spanning the plane with the given normal."""
    axis = 0 if abs(normal[0]) <= abs(normal[1]) else 1
    axis = axis if abs(normal[axis]) <= abs(normal[2]) else 2
    seed = tuple(1.0 if index == axis else 0.0 for index in range(3))
    first = (
        seed[1] * normal[2] - seed[2] * normal[1],
        seed[2] * normal[0] - seed[0] * normal[2],
        seed[0] * normal[1] - seed[1] * normal[0],
    )
    length = math.sqrt(sum(value * value for value in first)) or 1.0
    first = tuple(value / length for value in first)
    second = (
        normal[1] * first[2] - normal[2] * first[1],
        normal[2] * first[0] - normal[0] * first[2],
        normal[0] * first[1] - normal[1] * first[0],
    )
    return first, second


def _distance_of(point: Point, plane: Plane) -> float:
    """Return the signed distance, for points added after the initial scan."""
    return signed_distance(point, plane)


#: Every bucket a point within one tolerance of another can land in.
#:
#: `_key` floors each coordinate independently, so two points inside the
#: tolerance on all three axes can still differ by one bucket on more than one
#: axis at a time -- (0.99, 0.99, 0) and (1.01, 1.01, 0) at tolerance 1.0 key to
#: (0, 0, 0) and (1, 1, 0). Scanning only the six axial neighbours missed those
#: and kept a duplicate seam vertex, which the cap is then chained from.
#:
#: Ordered centre, then the original six, then the rest, so a lookup that
#: already succeeded returns the same candidate it always did and only the
#: previously-missed cases change.
_BUCKET_NEIGHBOURHOOD: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
) + tuple(
    (x, y, z)
    for x in (-1, 0, 1)
    for y in (-1, 0, 1)
    for z in (-1, 0, 1)
    if sum(1 for value in (x, y, z) if value) > 1
)


class _Welder:
    """Collect points, merging ones that land in the same place.

    Clipping produces the same intersection point once per face that shares the
    edge. Left unwelded the cap's boundary is a cloud of duplicates and cannot
    be chained into a loop at all.
    """

    def __init__(self, points: Sequence[Point], tolerance: float) -> None:
        """Seed the welder with a mesh's existing points.

        The spatial index is NOT built here. A plane that misses the mesh, or
        that lands exactly on existing vertices, adds no point at all, and
        bucketing every seed point for it costs as much as the cut -- measured,
        a fifth of the whole call on a 2k-face lead.
        """
        self._points: list[Point] = list(points)
        self._tolerance = tolerance
        self._index: dict[tuple[int, int, int], list[int]] | None = None

    def _ensure_index(self) -> dict[tuple[int, int, int], list[int]]:
        """Bucket every point collected so far, once.

        The keys are computed in one batched floor rather than three per point.
        The result is identical: `_key` floors a float64 quotient to an integer
        and so does this, and a mesh coordinate in millimetres against a 1e-7
        tolerance is nowhere near the range where int64 would differ.
        """
        if self._index is not None:
            return self._index
        self._index = {}
        if not self._points:
            return self._index

        for position, key in enumerate(
            quantized_keys(self._points, self._tolerance, mode="floor"),
        ):
            self._index.setdefault(key, []).append(position)
        return self._index

    def _key(self, point: Point) -> tuple[int, int, int]:
        """Return the bucket a point falls in."""
        return tuple(int(math.floor(value / self._tolerance)) for value in point)

    def add(self, point: Point) -> int:
        """Return the index of a point, reusing a coincident one."""
        index = self._ensure_index()
        key = self._key(point)
        for offset in _BUCKET_NEIGHBOURHOOD:
            bucket = (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
            for candidate in index.get(bucket, ()):
                existing = self._points[candidate]
                if all(abs(existing[axis] - point[axis]) <= self._tolerance
                       for axis in range(3)):
                    return candidate
        self._points.append(point)
        index.setdefault(key, []).append(len(self._points) - 1)
        return len(self._points) - 1

    @property
    def points(self) -> list[Point]:
        """Return every point collected so far."""
        return self._points


def _chain_loops(edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Chain directed edges into closed loops.

    Args:
        edges: Directed ``(start, end)`` pairs lying on the cut plane.

    Returns:
        One list of vertex indices per closed loop. Edges that cannot be closed
        are dropped rather than guessed at; the caller reports the shortfall.
    """
    following: dict[int, list[int]] = {}
    for start, end in edges:
        if start != end:
            following.setdefault(start, []).append(end)
    loops: list[list[int]] = []
    while following:
        start = next(iter(following))
        path = [start]
        position_of = {start: 0}
        current = start
        while True:
            successors = following.get(current)
            if not successors:
                break
            nxt = successors.pop()
            if not successors:
                del following[current]
            if nxt in position_of:
                # Coming back to a vertex already on the path closes a loop
                # HERE, and the walk carries on from it.
                #
                # Abandoning the path instead is what left K2010's joints open:
                # its rims branch at fourteen to twenty vertices, and dropping
                # the partial path threw away edges that had already been
                # consumed -- 24 of 66 rim edges ended up in a loop, the other
                # 42 in nothing at all. A rim is a set of cycles, so every edge
                # belongs to one; none of them has to be guessed at.
                at = position_of[nxt]
                cycle = path[at:]
                if len(cycle) >= 3:
                    loops.append(cycle)
                for vertex in path[at + 1:]:
                    del position_of[vertex]
                del path[at + 1:]
                current = nxt
                continue
            position_of[nxt] = len(path)
            path.append(nxt)
            current = nxt
    return [loop for loop in loops if len(loop) >= 3]


def _loop_area_and_centroid(loop: Sequence[Point]) -> tuple[float, tuple[float, float]]:
    """Return a loop's signed area and centroid in plane coordinates."""
    area = 0.0
    cx = cy = 0.0
    for index, (x0, y0) in enumerate(loop):
        x1, y1 = loop[(index + 1) % len(loop)]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-18:
        mean_x = sum(point[0] for point in loop) / len(loop)
        mean_y = sum(point[1] for point in loop) / len(loop)
        return 0.0, (mean_x, mean_y)
    return area, (cx / (6.0 * area), cy / (6.0 * area))


def _point_in_loop(point: tuple[float, float],
                   loop: Sequence[tuple[float, float]]) -> bool:
    """Return whether a point lies inside a closed polygon."""
    inside = False
    for index, (x0, y0) in enumerate(loop):
        x1, y1 = loop[(index + 1) % len(loop)]
        if (y0 > point[1]) != (y1 > point[1]):
            crossing = x0 + (point[1] - y0) * (x1 - x0) / (y1 - y0)
            if crossing > point[0]:
                inside = not inside
    return inside


def open_boundary_loops(
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
) -> list[list[int]]:
    """Return the loops of edges used by only one face.

    A closed surface uses every edge exactly twice, in opposite directions. An
    edge used once is a hole's rim, and chaining those rims gives the loops a
    cap has to close.

    Args:
        face_vertex_counts: Vertex count per face.
        face_vertex_indices: Flattened face vertex indices.

    Returns:
        One list of vertex indices per rim, wound so the missing material is
        on the same side as it is for the faces that already exist.
    """
    used: dict[tuple[int, int], int] = {}
    offset = 0
    for count in face_vertex_counts:
        face = face_vertex_indices[offset:offset + count]
        offset += count
        for position in range(count):
            edge = (face[position], face[(position + 1) % count])
            used[edge] = used.get(edge, 0) + 1
    rims = [
        edge for edge, times in used.items()
        if times == 1 and used.get((edge[1], edge[0]), 0) == 0
    ]
    # Reverse them: a rim traversed the way its own face traverses it leaves
    # the hole on the left, and the cap has to face the other way.
    return _chain_loops([(end, start) for start, end in rims])


def weld_seams(
    points: Sequence[Point],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    *,
    tolerance: float = 1.0e-6,
    passes: int = 4,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Close tears where two patches meet along a line but their vertices do not.

    Not every open edge is a hole. K2010's joints are torn along straight
    segments 0.02 to 0.10 mm long: twenty-four boundary vertices lying exactly
    on one line, enclosing no area at all -- Newell's normal comes out 0.0, and
    a cap over it would be a triangle with nothing inside it. The two sides of
    the tear cover the same segment with different vertices, so an edge on one
    side has a vertex from the other sitting in its middle and the two never
    pair off.

    The repair changes no geometry whatsoever: coincident vertices are merged,
    and an edge with another vertex lying on it is SPLIT there. Every point
    stays exactly where it was; only the indices change.

    Args:
        points: Mesh points.
        face_vertex_counts: Vertex count per face.
        face_vertex_indices: Flattened face vertex indices.
        tolerance: How close two positions must be to count as the same one.
        passes: How many times to sweep. One split can expose the next.

    Returns:
        ``(counts, indices, inserted)`` -- the topology, and how many vertices
        were spliced into an edge.
    """
    quantum = tolerance if tolerance > 0.0 else 1.0e-12
    canonical: dict[tuple[int, int, int], int] = {}
    remap: list[int] = []
    for index, key in enumerate(quantized_keys(points, quantum)):
        remap.append(canonical.setdefault(key, index))

    counts = list(face_vertex_counts)
    indices = [remap[index] for index in face_vertex_indices]
    inserted = 0

    for _ in range(passes):
        used: dict[tuple[int, int], int] = {}
        offset = 0
        for count in counts:
            face = indices[offset:offset + count]
            offset += count
            for position in range(count):
                edge = (face[position], face[(position + 1) % count])
                used[edge] = used.get(edge, 0) + 1
        rim = {edge for edge, times in used.items()
               if times == 1 and used.get((edge[1], edge[0]), 0) == 0}
        if not rim:
            break
        on_rim = {vertex for edge in rim for vertex in edge}

        grown_counts: list[int] = []
        grown_indices: list[int] = []
        added = 0
        offset = 0
        for count in counts:
            face = indices[offset:offset + count]
            offset += count
            rebuilt: list[int] = []
            for position in range(count):
                start = face[position]
                end = face[(position + 1) % count]
                rebuilt.append(start)
                if (start, end) not in rim:
                    continue
                first, second = points[start], points[end]
                along = tuple(second[axis] - first[axis] for axis in range(3))
                length2 = sum(value * value for value in along)
                if length2 <= 0.0:
                    continue
                between: list[tuple[float, int]] = []
                for vertex in on_rim:
                    if vertex in (start, end):
                        continue
                    offsets = tuple(points[vertex][axis] - first[axis]
                                    for axis in range(3))
                    t = sum(offsets[axis] * along[axis]
                            for axis in range(3)) / length2
                    if not 0.0 < t < 1.0:
                        continue
                    away = sum(
                        (offsets[axis] - t * along[axis]) ** 2
                        for axis in range(3))
                    if away <= tolerance * tolerance:
                        between.append((t, vertex))
                between.sort()
                rebuilt.extend(vertex for _, vertex in between)
                added += len(between)
            grown_counts.append(len(rebuilt))
            grown_indices.extend(rebuilt)
        if not added:
            break
        counts, indices = grown_counts, grown_indices
        inserted += added

    return tuple(counts), tuple(indices), inserted


def close_open_boundaries(
    points: Sequence[Point],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    *,
    coplanar_tolerance: float = 1.0e-3,
) -> tuple[tuple[Point, ...], tuple[int, ...], tuple[int, ...], int]:
    """Cap every hole in a surface, leaving the shape it already has.

    Adds no points and moves none: each rim is filled with triangles between
    the vertices it already has, so the silhouette is untouched.

    Rims that share a plane are capped TOGETHER, through the same nesting rule
    the cut uses. That matters for the one case this exists for: a bath clipped
    at the lead's end face leaves an OUTER rim and an INNER one, and filling
    them separately would put a disc straight over the lead's own face -- which
    is the face the dataset is about.

    Args:
        points: Mesh points.
        face_vertex_counts: Vertex count per face.
        face_vertex_indices: Flattened face vertex indices.
        coplanar_tolerance: How far two rims' planes may differ and still be
            capped as one.

    Returns:
        ``(points, counts, indices, capped_loops)``.
    """
    # Sew the tears shut before filling anything. Not every open edge is a
    # hole: a tear along a line encloses no area, so a cap over it is a
    # triangle with nothing inside, and the boundary count does not move.
    face_vertex_counts, face_vertex_indices, _ = weld_seams(
        points, face_vertex_counts, face_vertex_indices)

    loops = open_boundary_loops(face_vertex_counts, face_vertex_indices)
    if not loops:
        return (tuple(points), tuple(face_vertex_counts),
                tuple(face_vertex_indices), 0)

    def plane_of(loop: Sequence[int]) -> tuple[Point, Point] | None:
        """Return the rim's centre and unit normal by Newell's method."""
        centre = tuple(
            sum(points[index][axis] for index in loop) / len(loop)
            for axis in range(3)
        )
        normal = [0.0, 0.0, 0.0]
        for position, index in enumerate(loop):
            first = points[index]
            second = points[loop[(position + 1) % len(loop)]]
            normal[0] += (first[1] - second[1]) * (first[2] + second[2])
            normal[1] += (first[2] - second[2]) * (first[0] + second[0])
            normal[2] += (first[0] - second[0]) * (first[1] + second[1])
        length = sum(value * value for value in normal) ** 0.5
        if length < 1e-15:
            return None
        return centre, tuple(value / length for value in normal)

    described = [(loop, plane_of(loop)) for loop in loops]
    # `plane_of` returns None when the loop encloses no area -- a rim left
    # collinear even after welding. There is nothing to fill there, and fanning
    # it would add zero-area faces that read as a defect in every audit.
    described = [(loop, plane) for loop, plane in described if plane]

    groups: list[tuple[tuple[Point, Point], list[list[int]]]] = []
    for loop, plane in described:
        for existing_plane, members in groups:
            origin, normal = existing_plane
            aligned = abs(sum(normal[axis] * plane[1][axis]
                              for axis in range(3))) > 0.999
            offset = abs(sum((plane[0][axis] - origin[axis]) * normal[axis]
                             for axis in range(3)))
            if aligned and offset <= coplanar_tolerance:
                members.append(loop)
                break
        else:
            groups.append((plane, [loop]))

    existing: set[tuple[int, int]] = set()
    offset = 0
    for count in face_vertex_counts:
        face = face_vertex_indices[offset:offset + count]
        offset += count
        for position in range(count):
            existing.add((face[position], face[(position + 1) % count]))

    counts = list(face_vertex_counts)
    indices = list(face_vertex_indices)
    for plane, members in groups:
        cap_counts, cap_indices = _cap_loops(points, members, plane)
        # DECIDE WHICH WAY THE CAP FACES, BY THE MESH RATHER THAN THE PLANE.
        #
        # Ear clipping normalises a loop to counter-clockwise in whatever plane
        # it was handed, which is right for a cut -- every section shares one
        # plane and one normal. It is not right here: each rim gets its own
        # fitted plane, and half of them come out facing the other way. A cap
        # wound the SAME way as the rim seals nothing; measured on K2010, the
        # face count rose while the boundary held at 62 and the winding
        # conflicts climbed, which is a hole plus a defect where there was
        # only a hole.
        #
        # The mesh already knows: a rim edge is used once, so the cap must use
        # its REVERSE. Vote over the group and flip it whole, because the faces
        # of one cap have to agree with each other as well as with the rim.
        agree = disagree = 0
        offset = 0
        for count in cap_counts:
            face = cap_indices[offset:offset + count]
            offset += count
            for position in range(count):
                edge = (face[position], face[(position + 1) % count])
                if edge in existing:
                    disagree += 1
                elif (edge[1], edge[0]) in existing:
                    agree += 1
        if disagree > agree:
            flipped: list[int] = []
            offset = 0
            for count in cap_counts:
                face = cap_indices[offset:offset + count]
                offset += count
                flipped.extend(reversed(face))
            cap_indices = flipped
        counts.extend(cap_counts)
        indices.extend(cap_indices)
        offset = 0
        for count in cap_counts:
            face = cap_indices[offset:offset + count]
            offset += count
            for position in range(count):
                existing.add((face[position], face[(position + 1) % count]))
    return tuple(points), tuple(counts), tuple(indices), len(loops)


def cut_mesh_by_plane_reference(
    points: Sequence[Point],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    plane: Plane,
    *,
    weld_tolerance: float = 1.0e-7,
) -> tuple[tuple[Point, ...], tuple[int, ...], tuple[int, ...], int]:
    """Return the mesh with everything on the normal's side removed and capped.

    The scalar reference. `cut_mesh_by_plane` is the implementation callers
    should use; this one stays because it is what the fast path is validated
    against, result for result, in ``test_plane_cut_equivalence``.

    Args:
        points: Mesh points.
        face_vertex_counts: Vertex count per face.
        face_vertex_indices: Flattened face vertex indices.
        plane: ``(point_on_plane, unit_normal)``; the normal points at the
            material the blade takes away.
        weld_tolerance: Distance within which two points are the same point.

    Returns:
        ``(points, counts, indices, cap_faces)``. `cap_faces` is how many faces
        the new section is made of, which is zero when the plane misses the
        mesh entirely.
    """
    welder = _Welder(points, weld_tolerance)
    distances = [signed_distance(point, plane) for point in points]

    kept_counts: list[int] = []
    kept_indices: list[int] = []
    cut_edges: list[tuple[int, int]] = []

    offset = 0
    for count in face_vertex_counts:
        face = list(face_vertex_indices[offset:offset + count])
        offset += count
        if count < 3:
            continue
        clipped: list[int] = []
        # Sutherland-Hodgman against one half space.
        for position, index in enumerate(face):
            nxt = face[(position + 1) % len(face)]
            here, there = distances[index], distances[nxt]
            if here <= ON_PLANE_EPSILON:
                clipped.append(index)
            if (here > ON_PLANE_EPSILON) != (there > ON_PLANE_EPSILON):
                span = here - there
                if abs(span) > 1e-18:
                    clipped.append(welder.add(
                        _interpolate(points[index], points[nxt], here / span)))
        # Drop the duplicate a vertex sitting exactly on the plane produces.
        deduped: list[int] = []
        for index in clipped:
            if not deduped or deduped[-1] != index:
                deduped.append(index)
        if len(deduped) > 1 and deduped[0] == deduped[-1]:
            deduped.pop()
        if len(deduped) < 3:
            continue
        kept_counts.append(len(deduped))
        kept_indices.extend(deduped)
        # Read the cut edges off the CLIPPED polygon rather than recording
        # them while clipping. A plane through a vertex makes that vertex both
        # an on-plane point and the end of a crossing, so counting "the two
        # things the blade opened" misses the case entirely -- measured, a cube
        # cut at 45 degrees through its corners produced no cap at all.
        # Consecutive pairs that both lie on the plane are the cut, and they
        # already wind with the kept material on their left.
        for position, index in enumerate(deduped):
            nxt = deduped[(position + 1) % len(deduped)]
            if (abs(_distance_of(welder.points[index], plane)) <= ON_PLANE_EPSILON
                    and abs(_distance_of(welder.points[nxt], plane))
                    <= ON_PLANE_EPSILON):
                cut_edges.append((index, nxt))

    cap_faces = 0
    loops = _chain_loops(cut_edges)
    if loops:
        cap_counts, cap_indices = _cap_loops(welder.points, loops, plane)
        kept_counts.extend(cap_counts)
        kept_indices.extend(cap_indices)
        cap_faces = len(cap_counts)

    return (
        tuple(welder.points),
        tuple(kept_counts),
        tuple(kept_indices),
        cap_faces,
    )



#: A face wholly on the kept side, wholly on the blade side, or crossing.
_FACE_INSIDE, _FACE_OUTSIDE, _FACE_CROSSING = 0, 1, 2


def _classify_faces(
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    distances: Sequence[float],
) -> list[int]:
    """Say for every face whether it crosses the plane, in one pass.

    A vertex exactly on the plane counts as inside, which is the same test the
    clipper makes (`here <= ON_PLANE_EPSILON`), so a face whose vertices are all
    on-or-inside is reported INSIDE and skips clipping it does not need.

    Args:
        face_vertex_counts: Vertex count per face.
        face_vertex_indices: Flattened face vertex indices.
        distances: Signed distance per point.

    Returns:
        One of `_FACE_INSIDE`, `_FACE_OUTSIDE`, `_FACE_CROSSING` per face.

    Example:

    .. code-block:: python

        print(_classify_faces([3], [0, 1, 2], [-1.0, -1.0, -1.0]))
        # [0]

    """
    if not len(face_vertex_counts):
        return []
    import numpy as np

    counts = np.asarray(face_vertex_counts, dtype=np.int64)
    indices = np.asarray(face_vertex_indices, dtype=np.int64)
    outside = np.asarray(distances, dtype=np.float64) > ON_PLANE_EPSILON
    if indices.size == 0:
        return [_FACE_INSIDE] * len(counts)

    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    # reduceat needs every segment non-empty; a zero-count face is degenerate
    # and is skipped by the caller anyway, so give it a harmless segment.
    safe_starts = np.clip(starts, 0, max(indices.size - 1, 0))
    per_face = np.add.reduceat(outside[indices].astype(np.int64), safe_starts)
    per_face = np.where(counts > 0, per_face, 0)

    state = np.full(len(counts), _FACE_CROSSING, dtype=np.int64)
    state[per_face == 0] = _FACE_INSIDE
    state[per_face == counts] = _FACE_OUTSIDE
    return state.tolist()


def _emit_face(
    face: list[int],
    on_plane_flags: Sequence[bool],
    kept_counts: list[int],
    kept_indices: list[int],
    cut_edges: list[tuple[int, int]],
) -> None:
    """Record a face that needed no clipping, and any cut edges it carries.

    A face can be entirely on the kept side and still contribute to the cap: a
    plane through two of its vertices leaves an edge lying in the plane. That is
    the case a "no crossing means no cut edge" shortcut gets wrong.
    """
    deduped: list[int] = []
    for index in face:
        if not deduped or deduped[-1] != index:
            deduped.append(index)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    if len(deduped) < 3:
        return
    kept_counts.append(len(deduped))
    kept_indices.extend(deduped)
    for position, index in enumerate(deduped):
        following = (position + 1) % len(deduped)
        if on_plane_flags[index] and on_plane_flags[deduped[following]]:
            cut_edges.append((index, deduped[following]))

def signed_distances(
    points: Sequence[Point],
    plane: Plane,
) -> list[float]:
    """Return the signed distance from every point to a plane.

    The batched form of `signed_distance`, agreeing with it bit for bit.

    Getting that agreement is not free, and the reason is worth stating: the
    scalar version adds its three terms with the builtin `sum`, and CPython
    3.12 gives `sum` a compensated (Neumaier) accumulation for floats. Neither
    ``shifted @ normal`` nor a plain ``a + b + c`` reproduces it -- both land a
    unit in the last place away on about a fifth of inputs. That difference is
    not cosmetic here: the distances feed the interpolation ratio the clipper
    cuts with, so a last-place change MOVES an authored vertex. Measured on a
    cylinder cut obliquely, plain summation moved cut points; the compensated
    form below matched on all 20000 random points tried.

    Args:
        points: Mesh points.
        plane: ``(point_on_plane, unit_normal)``.

    Returns:
        One signed distance per point, positive on the normal's side.

    Example:

    .. code-block:: python

        print(signed_distances([(0.0, 0.0, 2.0)], ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))))
        # [2.0]

    """
    if not points:
        return []
    import numpy as np

    origin, normal = plane
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        return [signed_distance(point, plane) for point in points]
    shifted = array - np.asarray(origin, dtype=np.float64)
    total = shifted[:, 0] * normal[0]
    compensation = np.zeros_like(total)
    for axis in (1, 2):
        term = shifted[:, axis] * normal[axis]
        running = total + term
        compensation += np.where(
            np.abs(total) >= np.abs(term),
            (total - running) + term,
            (term - running) + total,
        )
        total = running
    return (total + compensation).tolist()


def cut_mesh_by_plane(
    points: Sequence[Point],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    plane: Plane,
    *,
    weld_tolerance: float = 1.0e-7,
) -> tuple[tuple[Point, ...], tuple[int, ...], tuple[int, ...], int]:
    """Return the mesh with everything on the normal's side removed and capped.

    Identical in result to `cut_mesh_by_plane_reference`, which is what it is
    tested against. It differs in three places, none of which changes an
    output value:

    * every point's distance to the plane is computed in one batched pass
      rather than one call per point;
    * a distance, once computed, is remembered. The reference recomputes it
      twice per face vertex when it reads the cut edges off the clipped
      polygon, which on a 2k-face lead was a quarter of the whole call. A
      point the clipper INTERPOLATES onto the plane is not assumed to be at
      zero -- its distance is measured like any other, because the reference
      measures it and rounding does not always land on zero;
    * the welder indexes its seed points lazily, so a plane that adds no point
      never pays for the index.

    Args:
        points: Mesh points.
        face_vertex_counts: Vertex count per face.
        face_vertex_indices: Flattened face vertex indices.
        plane: ``(point_on_plane, unit_normal)``; the normal points at the
            material the blade takes away.
        weld_tolerance: Distance within which two points are the same point.

    Returns:
        ``(points, counts, indices, cap_faces)``. `cap_faces` is how many faces
        the new section is made of, which is zero when the plane misses the
        mesh entirely.

    Example:

    .. code-block:: python

        result = cut_mesh_by_plane([], [], [], ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        print(result[3])
        # 0

    """
    welder = _Welder(points, weld_tolerance)
    distances = signed_distances(points, plane)

    kept_counts: list[int] = []
    kept_indices: list[int] = []
    cut_edges: list[tuple[int, int]] = []

    # Classify every face at once. A face entirely on one side of the plane
    # needs no clipping, and on a lead cut through the middle that is almost
    # all of them -- only one ring straddles. The scalar Sutherland-Hodgman
    # below still runs for those, IN FACE ORDER, so the welder is handed the
    # same points in the same order and the result is unchanged.
    face_state = _classify_faces(face_vertex_counts, face_vertex_indices, distances)
    on_plane_flags = [abs(value) <= ON_PLANE_EPSILON for value in distances]

    offset = 0
    for position_of_face, count in enumerate(face_vertex_counts):
        face_start = offset
        offset += count
        if count < 3:
            continue
        state = face_state[position_of_face]
        if state == _FACE_OUTSIDE:
            continue
        face = list(face_vertex_indices[face_start:face_start + count])
        if state == _FACE_INSIDE:
            # Nothing crosses, so the clipped polygon IS the face; only the
            # consecutive-duplicate collapse below can still change it.
            _emit_face(face, on_plane_flags, kept_counts, kept_indices, cut_edges)
            continue
        clipped: list[int] = []
        # Sutherland-Hodgman against one half space.
        for position, index in enumerate(face):
            nxt = face[(position + 1) % len(face)]
            here, there = distances[index], distances[nxt]
            if here <= ON_PLANE_EPSILON:
                clipped.append(index)
            if (here > ON_PLANE_EPSILON) != (there > ON_PLANE_EPSILON):
                span = here - there
                if abs(span) > 1e-18:
                    added = welder.add(
                        _interpolate(points[index], points[nxt], here / span))
                    # A welded point may be brand new or may be one already
                    # measured; either way its distance is recorded once.
                    while len(distances) <= added:
                        distances.append(
                            _distance_of(welder.points[len(distances)], plane))
                    clipped.append(added)
        # Drop the duplicate a vertex sitting exactly on the plane produces.
        deduped: list[int] = []
        for index in clipped:
            if not deduped or deduped[-1] != index:
                deduped.append(index)
        if len(deduped) > 1 and deduped[0] == deduped[-1]:
            deduped.pop()
        if len(deduped) < 3:
            continue
        kept_counts.append(len(deduped))
        kept_indices.extend(deduped)
        # Read the cut edges off the CLIPPED polygon rather than recording
        # them while clipping -- see the reference for why. Consecutive pairs
        # that both lie on the plane are the cut, and they already wind with
        # the kept material on their left.
        on_plane = [abs(distances[index]) <= ON_PLANE_EPSILON for index in deduped]
        for position, index in enumerate(deduped):
            following = (position + 1) % len(deduped)
            if on_plane[position] and on_plane[following]:
                cut_edges.append((index, deduped[following]))

    cap_faces = 0
    loops = _chain_loops(cut_edges)
    if loops:
        cap_counts, cap_indices = _cap_loops(welder.points, loops, plane)
        kept_counts.extend(cap_counts)
        kept_indices.extend(cap_indices)
        cap_faces = len(cap_counts)

    return (
        tuple(welder.points),
        tuple(kept_counts),
        tuple(kept_indices),
        cap_faces,
    )


def _triangulate_simple(
    loop: Sequence[int],
    loop_xy: Sequence[tuple[float, float]],
) -> tuple[list[int], list[int]]:
    """Ear-clip one loop with no hole into triangles.

    Emitting the section as a single n-gon looks like it works and does not.
    Hydra triangulates an n-gon as a fan from its first vertex, which is only
    correct while the polygon is convex; a joint cut at an angle is a crescent,
    and the fan then produces overlapping and inverted triangles that render as
    a hole straight through the section.

    Args:
        loop: Vertex indices, in order around the loop.
        loop_xy: The same vertices projected into the cut plane.

    Returns:
        ``(counts, indices)`` of triangles, or the loop as one face if it is
        too degenerate to clip -- a wrong cap is still better than none, and
        the caller cannot do anything more useful with it.
    """
    if len(loop) < 3:
        return [], []
    area, _ = _loop_area_and_centroid(loop_xy)
    order = list(range(len(loop)))
    if area < 0.0:
        order.reverse()

    def cross(a: int, b: int, c: int) -> float:
        """Return twice the signed area of a triangle of loop positions."""
        ax, ay = loop_xy[a]
        bx, by = loop_xy[b]
        cx, cy = loop_xy[c]
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    counts: list[int] = []
    indices: list[int] = []
    guard = 0
    while len(order) > 3 and guard < 4 * len(loop) * len(loop):
        guard += 1
        clipped = False
        for position in range(len(order)):
            previous = order[position - 1]
            current = order[position]
            following = order[(position + 1) % len(order)]
            if cross(previous, current, following) <= 1e-15:
                continue
            # An ear may not contain any other vertex of the loop. This scan
            # is the hot loop of a cut -- it runs once per candidate ear over
            # most of the loop -- so the cross products are inlined rather
            # than called. Same three operations in the same order on the same
            # values; only the call frames are gone.
            px, py = loop_xy[previous]
            cx0, cy0 = loop_xy[current]
            fx, fy = loop_xy[following]
            pc_x, pc_y = cx0 - px, cy0 - py
            cf_x, cf_y = fx - cx0, fy - cy0
            fp_x, fp_y = px - fx, py - fy
            contained = False
            for other in order:
                if other in (previous, current, following):
                    continue
                ox, oy = loop_xy[other]
                if (pc_x * (oy - py) - pc_y * (ox - px) >= 0.0
                        and cf_x * (oy - cy0) - cf_y * (ox - cx0) >= 0.0
                        and fp_x * (oy - fy) - fp_y * (ox - fx) >= 0.0):
                    contained = True
                    break
            if contained:
                continue
            counts.append(3)
            indices.extend((loop[previous], loop[current], loop[following]))
            order.pop(position)
            clipped = True
            break
        if not clipped:
            break
    if len(order) == 3:
        counts.append(3)
        indices.extend(loop[position] for position in order)
    elif len(order) > 3:
        # Could not clip it. Say so by shape rather than silently: one face,
        # which is what the old behaviour was everywhere.
        counts.append(len(order))
        indices.extend(loop[position] for position in order)
    return counts, indices


def _cap_loops(
    points: Sequence[Point],
    loops: Sequence[Sequence[int]],
    plane: Plane,
) -> tuple[list[int], list[int]]:
    """Fill the section, respecting holes.

    A lead's section is one loop and is fanned from its centroid. A joint's
    section is a loop with another loop inside it, and fanning that would paste
    solder over the lead's own face, which is coplanar with it -- so an outer
    loop with a hole is stitched to the hole instead.
    """
    _, normal = plane
    first_axis, second_axis = _plane_basis(normal)

    def project(index: int) -> tuple[float, float]:
        """Return a point's coordinates within the cut plane."""
        point = points[index]
        origin = plane[0]
        offset = tuple(point[axis] - origin[axis] for axis in range(3))
        return (
            sum(offset[axis] * first_axis[axis] for axis in range(3)),
            sum(offset[axis] * second_axis[axis] for axis in range(3)),
        )

    projected = [[project(index) for index in loop] for loop in loops]
    areas = [abs(_loop_area_and_centroid(loop)[0]) for loop in projected]

    # Nest by containment: a loop is a hole when its first vertex lies inside
    # a bigger loop. Comparing areas alone would pick the wrong parent for two
    # separate joints sharing one cut plane.
    parents: list[int | None] = [None] * len(loops)
    for child, child_loop in enumerate(projected):
        best_area = None
        for parent, parent_loop in enumerate(projected):
            if parent == child or areas[parent] <= areas[child]:
                continue
            if _point_in_loop(child_loop[0], parent_loop) and (
                    best_area is None or areas[parent] < best_area):
                parents[child], best_area = parent, areas[parent]

    counts: list[int] = []
    indices: list[int] = []
    for outer, loop in enumerate(loops):
        if parents[outer] is not None:
            continue
        holes = [index for index, parent in enumerate(parents) if parent == outer]
        if not holes:
            hole_counts, hole_indices = _triangulate_simple(
                loop, projected[outer])
            counts.extend(hole_counts)
            indices.extend(hole_indices)
            continue
        for hole in holes:
            counts_, indices_ = _stitch_ring(
                loop, projected[outer], loops[hole], projected[hole])
            counts.extend(counts_)
            indices.extend(indices_)
    return counts, indices


def _stitch_ring(
    outer: Sequence[int],
    outer_xy: Sequence[tuple[float, float]],
    inner: Sequence[int],
    inner_xy: Sequence[tuple[float, float]],
) -> tuple[list[int], list[int]]:
    """Triangulate the annulus between two loops.

    Both loops are walked by ANGLE about the inner loop's centroid rather than
    by index: they come from different faces of the mesh and share no index
    convention, so pairing them by position produces a fan that crosses itself.

    The walk is a merge over both rings with the first vertex of each repeated
    at angle + 2*pi. Closing the wrap by hand instead left exactly one triangle
    missing -- 127 where 128 were wanted -- which shows up as a hairline gap in
    the section rather than as anything obviously broken.
    """
    _, centre = _loop_area_and_centroid(inner_xy)

    def ordered(loop, loop_xy):
        """Return (index, angle) pairs anticlockwise, wrapped past 2*pi."""
        pairs = sorted(
            (
                math.atan2(point[1] - centre[1], point[0] - centre[0]),
                loop[position],
            )
            for position, point in enumerate(loop_xy)
        )
        if not pairs:
            return []
        first_angle, first_index = pairs[0]
        return [(index, angle) for angle, index in pairs] + [
            (first_index, first_angle + 2.0 * math.pi)]

    outer_ring = ordered(outer, outer_xy)
    inner_ring = ordered(inner, inner_xy)
    if len(outer_ring) < 3 or len(inner_ring) < 3:
        return [], []

    counts: list[int] = []
    indices: list[int] = []
    outer_position = inner_position = 0
    while (outer_position < len(outer_ring) - 1
           or inner_position < len(inner_ring) - 1):
        take_outer = inner_position >= len(inner_ring) - 1 or (
            outer_position < len(outer_ring) - 1
            and outer_ring[outer_position + 1][1] <= inner_ring[inner_position + 1][1]
        )
        if take_outer:
            counts.append(3)
            indices.extend((outer_ring[outer_position][0],
                            outer_ring[outer_position + 1][0],
                            inner_ring[inner_position][0]))
            outer_position += 1
        else:
            counts.append(3)
            indices.extend((outer_ring[outer_position][0],
                            inner_ring[inner_position + 1][0],
                            inner_ring[inner_position][0]))
            inner_position += 1
    return counts, indices
