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

"""Verify that a solved solder surface is a faithful equilibrium.

An equilibrium capillary surface has physical signatures we can measure, which
double as correctness tests -- so we trust the shape only when it passes them:

  * contact angle at a free contact line == the prescribed Young angle,
  * body volume == the target,
  * constant mean curvature (a fixed-pressure fluid surface),
  * refinement-stability (the measured quantities stop changing).

This module starts with the linchpin: the contact-angle measurement. Because the
free-wetting solves PRESCRIBE the angle (via APAD), measuring it back and getting
the same number is the self-consistency test for the whole solver.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _boundary_edges(F):
    """Edges belonging to exactly one triangle (the open mesh boundary) and, for
    each, its single owning face index."""
    owner = defaultdict(list)
    for fi, tri in enumerate(F):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            owner[(min(u, v), max(u, v))].append(fi)
    return [(e, fs[0]) for e, fs in owner.items() if len(fs) == 1]


def measure_contact_angle(V, F, plane_z=0.0, tol=1e-4, up=(0.0, 0.0, 1.0)):
    """Contact angle (deg) where a solder surface meets a flat substrate at z=plane_z.

    The contact line is the mesh boundary lying on the plane. At each contact-line
    edge the adjacent facet's OUTWARD (vapor-side) unit normal N gives the angle by
    cos(theta) = N . up  (theta=0 -> film lies flat; 90 -> vertical wall; >90 ->
    non-wetting overhang). Outward is resolved by pointing N away from the solder
    centroid. Returns per-contact-line stats -- a small `std` means a clean,
    uniform contact line; a large one means the solve/mesh is not trustworthy.
    """
    V = np.asarray(V, float)
    F = np.asarray(F, int).reshape(-1, 3)
    up = np.asarray(up, float)
    csol = V.mean(0)
    angs = []
    for (u, v), fi in _boundary_edges(F):
        if abs(V[u, 2] - plane_z) > tol or abs(V[v, 2] - plane_z) > tol:
            continue                                     # edge not on the substrate
        a, b, c = F[fi]
        n = np.cross(V[b] - V[a], V[c] - V[a])
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n /= ln
        fc = V[F[fi]].mean(0)
        if np.dot(n, csol - fc) > 0:                     # make N point away from the liquid
            n = -n
        angs.append(np.degrees(np.arccos(np.clip(np.dot(n, up), -1.0, 1.0))))
    angs = np.asarray(angs)
    if not len(angs):
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"n": int(len(angs)), "mean": float(angs.mean()), "std": float(angs.std()),
            "min": float(angs.min()), "max": float(angs.max())}


# ------------------------------------------------- sessile-drop verification
# (folded in from the old validation/contact_angle.py: given the free surface
# of a drop resting on z = 0, fit a sphere and recover the angle the liquid
# makes with the substrate. For an equilibrium sessile drop this must match
# the prescribed angle -- the physics gate for the Evolver backend.)

def fit_sphere(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic least-squares sphere fit. Returns (center[3], radius)."""
    pts = np.asarray(pts, dtype=float)
    A = np.column_stack([2.0 * pts, np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:3]
    radius = float(np.sqrt(max(sol[3] + center @ center, 0.0)))
    return center, radius


def sessile_contact_angle_deg(vertices: np.ndarray) -> float:
    """Contact angle (deg) of a spherical-cap drop on z = 0.

    The drop surface is part of a sphere of radius R centered at height `cz`.
    The angle through the liquid at the contact line satisfies cos(theta) = -cz/R
    (cz < 0 -> acute/wetting; cz = 0 -> hemisphere 90 deg; cz > 0 -> obtuse/beading).
    """
    center, radius = fit_sphere(vertices)
    if radius < 1e-30:
        return float("nan")
    cos_t = np.clip(-center[2] / radius, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))
