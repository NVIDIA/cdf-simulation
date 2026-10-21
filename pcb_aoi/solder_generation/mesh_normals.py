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

"""Crease-aware vertex normals for solver triangle meshes.

Solver meshes (Surface Evolver output in particular) carry raw triangles with
no shading information; authored as-is they flat-shade per facet and the
curved free surface reads as faceted. Plain all-neighbour averaging is no
better for this geometry: after planar-patch decimation the wet patches are a
handful of LARGE triangles, and a normal bent at their rim by the free
surface interpolates a shading gradient across the whole flat patch.

So: area-weighted averaging WITHIN smoothing groups, with the mesh split at
creases. Incident faces of each vertex are greedily clustered by normal
similarity (``crease_deg`` threshold); one cluster keeps the vertex, every
additional cluster gets a positional duplicate. Curved regions smooth, the
free-surface/pad-plane crease stays sharp.
"""

from __future__ import annotations

import math

import numpy as np


def crease_aware_vertex_normals(
    points: np.ndarray,
    triangles: np.ndarray,
    crease_deg: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split vertices at creases and compute area-weighted vertex normals.

    Args:
        points: ``(n, 3)`` float vertex positions.
        triangles: ``(m, 3)`` int vertex indices.
        crease_deg: Dihedral threshold in degrees. Incident faces whose
            normals differ from a smoothing group's average by more than
            this start a new group (and a duplicated vertex).

    Returns:
        ``(points, triangles, normals)`` -- positions with duplicates
        appended, re-indexed triangles, and one unit normal per point.
        Geometry is unchanged; only indexing and shading data differ.

    """
    P = np.asarray(points, float).reshape(-1, 3)
    T = np.asarray(triangles, int).reshape(-1, 3)
    if not len(T):
        return P, T, np.zeros_like(P)

    # Unnormalised triangle cross products: direction is the face normal,
    # magnitude is twice the area -- exactly the area weighting we want.
    cross = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    area2 = np.linalg.norm(cross, axis=1)
    ok = area2 > 0.0
    unit = np.zeros_like(cross)
    unit[ok] = cross[ok] / area2[ok, None]

    incident: list[list[int]] = [[] for _ in range(len(P))]
    for f, tri in enumerate(T):
        for v in tri:
            incident[v].append(f)

    cos_th = math.cos(math.radians(float(crease_deg)))
    out_T = T.copy()
    extra_points: list[np.ndarray] = []
    extra_normals: list[np.ndarray] = []
    normals = np.zeros_like(P)

    def _unitized(vec: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(vec))
        return vec / n if n > 0.0 else np.array([0.0, 0.0, 1.0])

    for v, faces in enumerate(incident):
        if not faces:
            normals[v] = (0.0, 0.0, 1.0)
            continue
        # Greedy clustering, largest faces first: big patches seed the
        # groups, slivers join whichever side of the crease they lie on.
        clusters: list[tuple[np.ndarray, list[int]]] = []
        for f in sorted(faces, key=lambda f: -area2[f]):
            if not ok[f] and clusters:
                clusters[0][1].append(f)  # degenerate: no vote, no split
                continue
            for total, members in clusters:
                if float(unit[f] @ _unitized(total)) >= cos_th:
                    total += cross[f]
                    members.append(f)
                    break
            else:
                clusters.append((cross[f].copy(), [f]))
        if not clusters:
            normals[v] = (0.0, 0.0, 1.0)
            continue
        normals[v] = _unitized(clusters[0][0])
        for total, members in clusters[1:]:
            new_index = len(P) + len(extra_points)
            extra_points.append(P[v])
            extra_normals.append(_unitized(total))
            for f in members:
                out_T[f][T[f] == v] = new_index

    if extra_points:
        P = np.vstack([P, np.asarray(extra_points)])
        normals = np.vstack([normals, np.asarray(extra_normals)])
    return P, out_T, normals


def solder_mesh_shading(
    points: np.ndarray,
    triangles: np.ndarray,
    crease_deg: float = 50.0,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...], tuple]:
    """``crease_aware_vertex_normals`` in ``SolderMesh`` field shapes.

    Returns:
        ``(points, face_vertex_counts, face_vertex_indices, normals)`` where
        the last three are tuples ready for ``SolderMesh``; ``points`` stays
        an array so callers can still translate twins cheaply.

    """
    P, T, N = crease_aware_vertex_normals(points, triangles, crease_deg)
    return (
        P,
        tuple(3 for _ in range(len(T))),
        tuple(int(i) for tri in T for i in tri),
        tuple((float(n[0]), float(n[1]), float(n[2])) for n in N),
    )
