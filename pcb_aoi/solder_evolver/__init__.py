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

"""Equilibrium solder-joint geometry on Ken Brakke's Surface Evolver.

The method, end to end: seed an initial blob of solder; intersect it against
the mesh geometry of the board and component (manifold3d booleans, with
per-triangle provenance so every surviving facet knows which solid it came
from); PIN the intersection-outline boundary curves; then update the free
surface to minimise surface-area energy under a volume constraint.

There is experimental support for an SDF-based "collision" energy -- a penalty
keeping the free surface out of the component (and out of the board, via
``floor_z``). The same machinery could in principle drive Young's-equation
spreading of the boundary curve instead of pinning it; that work is incomplete
(the free-boundary attempt diverges below ~70 deg -- see branch
denomme/young-pad and ``datafiles/sessile.fe`` for the constraint-integrand
formulation that is the likely path back in).

Layout: ``boolean``/``tht`` build and cut the seed, ``emit`` writes the ``.fe``
datafile, ``solve`` drives Evolver (refine ladder, staged Newton, mesh
maintenance), ``backend`` owns the process and the dump parser, ``sdf`` bakes
keep-out grids, ``volume`` is the one shared volume rule, ``defects`` holds the
per-defect strategies, ``tht_spec`` shapes through-hole inputs like the
PCBA->USD contract, ``verify`` measures equilibrium signatures.
"""

from __future__ import annotations

import numpy as _np

from .boolean import check_body, cluster_rects, cut_joint as cut_smt_joint, make_blob
from .defects import resolve as resolve_defect
from .defects.base import DefectStrategy
from .emit import emit as emit_datafile
from .sdf import bake_keepout_sdf, mesh_sdf_grid, write_grid
from .solve import solve_stages
from .tht import cut_tht_joint
from .verify import measure_contact_angle
from .volume import volume_plan

__all__ = [
    # geometry: seed a blob, intersect against mesh geometry, label provenance
    "make_blob",
    "cut_smt_joint",
    "cut_tht_joint",
    "check_body",
    "cluster_rects",
    # energy: pin the intersection outlines, minimise area at fixed volume
    "emit_datafile",
    "solve_stages",
    "solve_joint",
    # volume policy and defect strategies
    "volume_plan",
    "resolve_defect",
    "DefectStrategy",
    # experimental SDF collision energy (Young spreading incomplete)
    "bake_keepout_sdf",
    "mesh_sdf_grid",
    "write_grid",
    # verification
    "measure_contact_angle",
]

#: Default Young angles per wetted family. INERT while the wetted patches are
#: pinned (a fixed facet's tension is a constant in the energy) -- kept because
#: they become live the moment a contact line is freed.
DEFAULT_THETA = {"wet_board": 35.0, "wet_metal": 25.0, "wet_body": 25.0}


def solve_joint(d, joint, *, kind="smt", theta=None, target_volume=None,
                keepout=None, cut_kw=None, solve_kw=None):
    """The whole formulation in one call: cut -> pin -> minimise.

    ``d`` is a joint dict (frozen npz mapping, or the equivalents built by
    ``cdf`` stage_read / ``tht.dict_from_joints``); ``joint`` selects one
    joint. ``kind`` picks the SMT or THT cut. ``target_volume`` defaults to the
    cut's own volume -- pass the real solder volume (e.g. from
    :func:`volume_plan`) to decouple footprint from fill, which is the point of
    the blob/target split.

    Returns ``(meshes, stats, cut)``: every solve checkpoint (the last is the
    equilibrium), the Evolver stats dict, and the cut result with its labels
    and provenance.
    """
    cut_kw = dict(cut_kw or {})
    cut_kw.setdefault("verbose", False)
    if kind == "smt":
        cut = cut_smt_joint(d, joint, **cut_kw)
    elif kind == "tht":
        cut = cut_tht_joint(d, joint, **cut_kw)
    else:
        raise ValueError(f"kind must be 'smt' or 'tht', got {kind!r}")

    lab = _np.asarray(cut["lab"]).astype(str)
    fe_text, _ = emit_datafile(cut["V"], cut["F"], lab,
                               dict(theta or DEFAULT_THETA), keepout=keepout)
    if target_volume is None:
        # the two cuts name their volume differently (SMT: the inflated blob;
        # THT: the cut body)
        target_volume = float(cut.get("target_volume", cut.get("volume")))
    meshes, stats, _tags = solve_stages(fe_text, float(target_volume),
                                        **dict(solve_kw or {}))
    return meshes, stats, cut
