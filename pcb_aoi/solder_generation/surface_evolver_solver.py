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

"""Surface Evolver as a cdf solder solver.

Implements the ``SolderSolverBase`` contract from
``pcb_aoi/solder_generation/solver_base.py`` so the six-area workflow can drive
an equilibrium solve instead of a procedural fillet.

Scope of this first cut, deliberately narrow:

  * DEFECTS are handled by strategy, one per defect type
    (``pcb_aoi.solder_evolver.defects``). A work item whose ``defect_type`` has no
    registered strategy is reported INELIGIBLE rather than quietly solving as
    nominal -- a defect that silently produced a good joint would be
    mislabelled training data.
  * A BUDGET. A full board is thousands of joints and a collider-on solve is
    minutes each, so ``solver_options`` takes an allowlist and an n-cap.
    Everything outside the budget is reported INELIGIBLE **with a reason** and
    simply gets no solder -- visible in the manifest rather than silently
    missing, which is the difference between "not attempted" and "failed".

Reuse is keyed on the component-local paste footprint, not on ``prototype_id``.
See ``surface_evolver_stage_read.footprint_key`` for why that distinction is load-bearing.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import tempfile
import threading
import traceback
from collections.abc import Sequence
from contextlib import contextmanager

import numpy as np

from pcb_aoi.solder_generation.data_types import (
    IDENTITY_MATRIX_4X4,
    SolderCoordinateFrame,
    SolderCoverageStatus,
    SolderGenerationRequest,
    SolderGeometryArtifact,
    SolderMesh,
    SolderReuseScope,
    SolderSolverRequest,
    SolderSolverResult,
    SolderSolverType,
    SolderWorkItem,
    SolderWorkItemCoverage,
)
from pcb_aoi.solder_generation.mesh_normals import solder_mesh_shading
from pcb_aoi.solder_generation.solver_base import SolderSolverBase, _component_allowlist

from pcb_aoi.solder_evolver import defects as defect_strategies

from . import surface_evolver_stage_read as stage_read

SURFACE_EVOLVER_SOLVER_ID = "surface_evolver"

#: Identity, row-major, matching the contract's convention.
_IDENTITY = (1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0)

#: Serialises the GPU (Warp) collider bake across joint-fan-out threads. The
#: Evolver solve is a subprocess and stays unlocked, so only the short bake is
#: single-file; the long solves still overlap.
_bake_lock = threading.Lock()

#: The core budget, enforced as "live Evolver subprocesses at once". Installed
#: by ``_run_solves`` BEFORE the footprint pool forks, so the pool children
#: inherit the same kernel semaphore and the budget holds across the process
#: fan-out and every within-footprint thread fan combined. None (the default,
#: and what a direct ``solve_footprint`` caller sees) means
#: ungated. The semaphore -- not the pool/thread sizes -- is the scheduler:
#: it is work-conserving, so when one footprint finishes its slots flow to
#: whoever is still solving. (A static budget/groups split was tried instead
#: and measured SLOWER, 8m32s vs 5m40s on the reference THT bake: it starves
#: the straggler while finished groups' cores sit idle.)
_solve_gate = None
_core_budget = None


def _install_solve_gate(budget):
    """Arm the budget for this request. Must run before the pool forks."""
    global _solve_gate, _core_budget
    _core_budget = int(budget)
    _solve_gate = multiprocessing.BoundedSemaphore(_core_budget)


@contextmanager
def _solve_slot():
    """Hold one core's worth of budget for the duration of an Evolver solve."""
    gate = _solve_gate
    if gate is None:
        yield
        return
    with gate:
        yield


def _evolver_available():
    """(available, reason). Unavailable is normal, not an error: `vendor/` is
    gitignored, so a fresh checkout has the build script but no binary."""
    try:
        from pcb_aoi.solder_evolver import backend as ev

        path = ev.find_evolver()
    except Exception as exc:                                    # pragma: no cover
        return False, f"Surface Evolver backend unimportable: {exc}"
    if not path:
        return False, (
            "no Surface Evolver binary; build it with "
            "pcb_aoi/solder_evolver/third_party/evolver/build.sh "
            "(the packaged evolver lacks -DENABLE_DLL and cannot run our "
            "grid-SDF collider)"
        )
    return True, None


class SurfaceEvolverSolver(SolderSolverBase):
    """Equilibrium solder geometry from a Surface Evolver minimisation."""

    solver_id = SURFACE_EVOLVER_SOLVER_ID
    solver_type = SolderSolverType.SIMULATED

    #: Paste is NOT a discovery gate: SMT joints are defined by their stencil
    #: apertures, but a paste-less component may still be a THROUGH-HOLE part
    #: whose joints come from the board's plated holes (tht_spec.discover:
    #: catalog-driven on a schema board, raw-layers shim on a pre-schema one).
    #: prepare() classifies per instance -- paste -> SMT, no paste -> try THT,
    #: neither -> INELIGIBLE.
    requires_source_paste = False

    def __init__(self):
        self.available, self.unavailable_reason = _evolver_available()

    # ------------------------------------------------------------------ prepare

    def prepare(
        self,
        request: SolderGenerationRequest,
        work_items: Sequence[SolderWorkItem],
    ) -> SolderSolverRequest:
        """Distil the live stage into a stage-free payload.

        Reads geometry in the component-local frame and records, per work item,
        the paste footprint and the component meshes the collider needs. Also
        decides here -- while we still have identity information -- which items
        fit the budget, so ``generate`` never has to guess.
        """
        opts = dict(request.solver_options or {})
        allow = _component_allowlist(opts)
        # No cap by default: the production path solves the whole board. A dev
        # harness that wants a cheap partial run passes max_components explicitly
        # (cdf_board.py --max). The old default of 4 silently dropped most of a
        # real board -- and a dropped item that never reached `generate` still
        # owes the extension a coverage row, or its result is rejected wholesale.
        max_items = opts.get("max_components")

        entries, skipped = [], []
        # Shared across this prepare(): tht_spec re-reads the soldermask sheet
        # and thousands of plating prims per component without it.
        tht_cache: dict = {}

        def _skip(work_item_id, paths, why):
            # Every skip carries its work_item_id -- the extension validates that
            # coverage maps 1:1 onto the work items, so a None id ("unassigned")
            # fails the whole result rather than marking one item INELIGIBLE.
            skipped.append({"work_item_id": work_item_id,
                            "component_paths": [str(p) for p in paths],
                            "reason": why})

        for item in work_items:
            # A defect is handled when a strategy is registered for it.
            # Unregistered ones are reported INELIGIBLE rather than silently
            # solving as nominal -- a defect that quietly produced a good joint
            # would be mislabelled training data.
            want = item.defect.defect_type if item.defect is not None else None
            if defect_strategies.resolve(want) is None:
                _skip(item.work_item_id, item.component_paths,
                      f"no strategy for defect '{want}'; "
                      f"known: {', '.join(defect_strategies.known())}")
                continue
            for comp in item.components:
                path = comp.instance_path
                selected = (not allow) or (comp.refdes in allow) or (path in allow)
                if not selected:
                    _skip(item.work_item_id, item.component_paths,
                          f"{comp.refdes} not in the configured component allowlist")
                    continue
                try:
                    entry = self._read_component(
                        request.stage, item, comp,
                        drop_occluded=bool(opts.get("drop_occluded", True)),
                        catalog=request.board_metadata)
                except Exception as exc:                        # pragma: no cover
                    _skip(item.work_item_id, item.component_paths,
                          f"could not read {comp.refdes}: {exc}")
                    continue
                if entry is None:
                    # No paste footprint -- but a paste-less part may be a
                    # THROUGH-HOLE component whose joints are the board's
                    # occupied plated holes. Use the shared discover route:
                    # catalog-driven on a schema board,
                    # raw-layers shim on a pre-schema one.
                    try:
                        entry = self._read_tht_component(
                            request.stage, request.root_path, item, comp,
                            tht_cache, catalog=request.board_metadata)
                    except Exception as exc:                    # pragma: no cover
                        _skip(item.work_item_id, item.component_paths,
                              f"could not read {comp.refdes} as THT: {exc}")
                        continue
                if entry is None:
                    _skip(item.work_item_id, item.component_paths,
                          f"{comp.refdes} has no usable paste footprint and "
                          f"no occupied plated holes")
                    continue
                # A strategy may only exist on one side (dry contact is a THT
                # cut). The wrong kind is INELIGIBLE, not silently nominal --
                # a nominal joint labelled defective is mislabelled data.
                if want is not None:
                    kind = "tht" if entry.get("kind") == "tht" else "smt"
                    if kind not in defect_strategies.resolve(want).kinds:
                        _skip(item.work_item_id, item.component_paths,
                              f"defect '{want}' applies to "
                              f"{'/'.join(defect_strategies.resolve(want).kinds)}"
                              f" joints only; {comp.refdes} is {kind}")
                        continue
                entries.append(entry)

        # Apply the n-cap over DISTINCT footprints, not over instances: the
        # expensive thing is a solve, and instances sharing a footprint are free.
        if max_items is not None:
            seen, kept = {}, []
            for e in entries:
                seen.setdefault(e["footprint_key"], []).append(e)
            for k, group in list(seen.items())[: int(max_items)]:
                kept.extend(group)
            for e in (e for e in entries if e not in kept):
                _skip(e["work_item_id"], [e["instance_path"]],
                      f"{e['refdes']}: beyond max_components={max_items}")
            entries = kept

        # A shim answer on a board whose catalog DOES carry hole records means
        # the association chain broke (exporter gap or resolver miss) -- worth
        # a warning, not silence: the two ingests can disagree on geometry.
        catalog_has_holes = bool(getattr(request.board_metadata, "holes", ()))
        prep_warnings = [
            f"{e['refdes']}: THT ingest fell back to the raw-layers shim "
            f"although the catalog carries hole records"
            for e in entries
            if e.get("kind") == "tht"
            and e.get("tht_source") == "raw_layers"
            and catalog_has_holes
        ]
        payload = {"entries": entries, "skipped": skipped,
                   "preparation_warnings": prep_warnings}
        return SolderSolverRequest(
            request_id=request.request_id,
            solver_id=self.solver_id,
            source_usd_path=request.source_usd_path,
            root_path=request.root_path,
            work_items=tuple(work_items),
            solver_options=opts,
            payload=payload,
        )

    def _read_component(self, stage, item, comp, drop_occluded=True,
                        catalog=None):
        """One component -> a transport-safe entry, or None if unusable."""
        path = comp.instance_path
        l2w, w2l = stage_read.component_local_transform(stage, path)
        paste_paths = tuple(item.source_paste_prims.get(path, ()))
        if not paste_paths:
            return None
        paste_pts = stage_read.paste_points_local(stage, paste_paths, w2l)
        if not paste_pts:
            return None
        V, F = stage_read.component_meshes_local(stage, path, w2l)
        if V is None:
            return None

        # Rotate into the frame the solver assumes (board = z=0, part above).
        # Which component-local axis is the board normal varies by asset, so it
        # is measured from the paste rather than assumed -- see solver_frame.
        R, Rinv = stage_read.solver_frame(paste_pts, V)
        paste_pts = [stage_read.apply4(P, R) for P in paste_pts]
        V = stage_read.apply4(V, R)

        # Exposed pad outlines (solver frame): the pad-plane clip's input.
        # Solder wets copper only -- without these the footprint is paste
        # plus blind offsets, which overruns 1:1-aperture pads and bridges
        # fine-pitch neighbours. Empty on pad-less/sidecar boards: the clip
        # simply does not run and legacy behaviour is untouched.
        pad_polys = []
        if catalog is not None:
            pad_paths = tuple(p.prim_path for p in catalog.pads
                              if p.component_path == path)
            if pad_paths:
                pad_meshes = stage_read.pad_meshes_local(stage, pad_paths, w2l)
                pad_polys = stage_read.pad_outline_polys(
                    [(stage_read.apply4(P, R), T) for P, T in pad_meshes])

        # Drop deposits hidden under the part itself. Invisible to a top-down
        # camera, and the blob squeezing out from under a package is a source of
        # solder that protrudes where none should exist. The silhouette comes
        # from the shared route (body-role silhouette on schema assets,
        # whole-part bbox otherwise), so a
        # connector whose pins tuck under protruding leads keeps its pin
        # joints in board bakes exactly as it does in the lab.
        if drop_occluded:
            vis = stage_read.occluded_deposits(
                paste_pts, V,
                silhouette_pts=stage_read.occlusion_silhouette(
                    stage, path, w2l, frame=R))
            keep = [P for P, hid in zip(paste_pts, vis) if not hid]
            n_hidden = sum(vis)
            if not keep:
                return None                 # nothing visible on this component
            paste_pts = keep
        else:
            n_hidden = 0

        rects = stage_read.rects_from_points(paste_pts)
        if not rects:
            return None
        # The pad plane is the UNDERSIDE of the paste deposits -- the copper the
        # paste was printed onto. Paste z is a far more reliable datum than the
        # component's own min-z, which picks up whatever the lowest bit of
        # geometry happens to be. Taking the deposit TOP instead, as this did,
        # sat every joint a full stencil thickness (0.076 mm here) above its pad.
        pad_z = float(np.mean([r[4] for r in rects]))
        # True aperture outlines. Bounding boxes square off the 45% of this
        # board's deposits that are rounded, which changes both the flooded
        # shape and -- because volume is area x thickness -- the volume itself.
        polys = stage_read.polys_from_points(paste_pts)
        # The frame is part of the footprint identity: two instances may only
        # share a solve if their geometry AND its orientation agree.
        key = stage_read.footprint_key(
            [r[:4] for r in rects], pad_z,
            extra=(comp.component_definition_path,
                   tuple(np.round(R[:3, :3].ravel(), 6)),
                   # The pad outlines gate the pad-plane clip, so they are
                   # part of the footprint identity too.
                   tuple(tuple(np.round(np.asarray(P, float).ravel(), 4))
                         for P in pad_polys),
                   # Two deposits can share a bbox and differ in shape (a
                   # stadium and its enclosing rectangle), so the outline has to
                   # be part of the identity or they would wrongly share a solve.
                   tuple(tuple(np.round(np.asarray(P, float).ravel(), 4))
                         if P is not None else () for P in polys)))
        if item.defect is not None:
            # The DEFECT is part of the footprint identity too. Geometry alone
            # would group a defected instance with its nominal footprint-mates,
            # and generate() solves ONE representative per group -- the defect
            # strategy (volume scale, replaced steps) would silently never run
            # anywhere the footprint recurs. A lone defected coupon hid this:
            # sole instance, sole entry, its strategy drove the solve.
            key = f"{key}__{item.defect.defect_type}"
            # ...and so are its sampled observables: two instances of one
            # footprint at different volume_ratio must not share a solve.
            obs = _observables(item.defect.metadata)
            if obs:
                key += "__" + hashlib.sha1(
                    repr(sorted(obs.items())).encode()).hexdigest()[:8]
        return {
            "work_item_id": item.work_item_id,
            "defect_type": (item.defect.defect_type
                            if item.defect is not None else None),
            "defect_params": (_observables(item.defect.metadata)
                              if item.defect is not None else None),
            "instance_path": path,
            "refdes": comp.refdes,
            "prototype_path": comp.prototype_path,
            "footprint_key": key,
            "rects": [list(r[:4]) for r in rects],
            "polys": [np.asarray(P, float).tolist() if P is not None else None
                      for P in polys],
            "pad_polys": [np.asarray(P, float).tolist() for P in pad_polys],
            "pad_z": pad_z,
            # Measured, not assumed -- see stage_read.paste_thickness.
            "paste_thickness": stage_read.paste_thickness(paste_pts),
            "occluded_deposits": int(n_hidden),
            "mesh_points": V.astype(float).tolist(),
            "mesh_faces": F.astype(int).tolist(),
            "local_to_world": tuple(float(v) for row in np.array(l2w, float)
                                    .reshape(4, 4) for v in row),
            # Solved geometry is in the solver frame, so placing it needs this
            # first: p_world = p_solver @ solver_to_local @ local_to_world.
            "solver_to_local": tuple(float(v) for v in Rinv.ravel()),
        }

    def _read_tht_component(self, stage, root_path, item, comp, cache,
                            catalog=None):
        """One paste-less component -> a THT entry, or None if it has no joints.

        ``tht_spec.discover`` reads the catalog-driven conformant path on a
        schema-carrying board and falls back to the raw-layers shim on a
        pre-schema one; the entry records which source answered.
        ``tht.dict_from_joints`` packs the joints into the plain-array mapping
        ``cut_tht_joint`` consumes. The dict is numpy-only, so the entry stays
        picklable for the process pool.

        Unlike SMT there is no footprint reuse: a plated hole is a BOARD
        feature at an absolute position, so every instance is its own solve
        and the meshes come back in WORLD coordinates.
        """
        from pcb_aoi.solder_evolver import tht, tht_spec

        path = comp.instance_path
        l2w, w2l = stage_read.component_local_transform(stage, path)
        V, F = stage_read.component_meshes_local(stage, path, w2l)
        if V is None:
            return None
        V_world = stage_read.apply4(np.asarray(V, float), np.asarray(l2w))
        joints, tht_source = tht_spec.discover(
            stage, root_path, path, np.asarray(V_world, float),
            catalog=catalog, cache=cache)
        if not joints:
            return None
        d = tht.dict_from_joints(joints, V_world,
                                 np.asarray(F, np.int64))
        return {
            "kind": "tht",
            "tht_source": tht_source,
            "work_item_id": item.work_item_id,
            "defect_type": (item.defect.defect_type
                            if item.defect is not None else None),
            "defect_params": (_observables(item.defect.metadata)
                              if item.defect is not None else None),
            "instance_path": path,
            "refdes": comp.refdes,
            "prototype_path": comp.prototype_path,
            # A board-absolute feature cannot share a solve across instances;
            # the instance path keys a group of exactly one.
            "footprint_key": "tht_" + hashlib.sha1(
                str(path).encode()).hexdigest()[:16],
            "tht": d,
            "joint_count": len(joints),
        }

    # ----------------------------------------------------------------- generate

    def generate(self, request: SolderSolverRequest) -> SolderSolverResult:
        """Solve one joint per DISTINCT footprint and share the result.

        Stage-free by contract, so everything here works off ``request.payload``.
        A failed solve becomes SOLVER_FAILED coverage rather than an exception --
        the extension is responsible for falling back to the procedural solver,
        and it can only do that if we return.
        """
        entries = list(request.payload.get("entries", ()))
        skipped = list(request.payload.get("skipped", ()))
        opts = dict(request.solver_options or {})

        groups: dict[str, list[dict]] = {}
        for e in entries:
            groups.setdefault(e["footprint_key"], []).append(e)

        # Solve one representative per DISTINCT footprint. Each solve is an
        # independent Evolver subprocess ("minutes each"), so they fan out
        # across cores -- see _run_solves. The build loop below stays a single
        # deterministic pass over `groups` in insertion order, so the artifact
        # ordering is identical whether the solves ran serially or pooled.
        # solver_options["workers"] is the TOTAL core budget (0/absent = the
        # machine minus two): _run_solves arms a semaphore sized to it, and
        # every Evolver solve -- footprint-level process or within-footprint
        # thread -- holds one slot while it runs. The semaphore, not the pool
        # widths, is the scheduler: work-conserving, so a joint-heavy
        # straggler (a 168-pin connector) expands into whatever the finished
        # groups release, and the budget is never exceeded.
        solved, failed = self._run_solves(groups, opts)

        artifacts, coverage = [], []
        warnings = [str(w) for w in
                    request.payload.get("preparation_warnings", ()) if w]
        for key, group in groups.items():
            rep = group[0]
            if key in failed:
                exc, tb = failed[key]
                warnings.append(f"{rep['refdes']}: solve failed: {exc}")
                warnings.append(tb)
                for e in group:
                    coverage.append(SolderWorkItemCoverage(
                        work_item_id=e["work_item_id"],
                        component_paths=(e["instance_path"],),
                        status=SolderCoverageStatus.SOLVER_FAILED,
                        message=str(exc),
                    ))
                continue
            meshes, solve_warnings = solved[key]
            warnings.extend(solve_warnings)

            # One artifact per footprint, shared by every instance that matched.
            # PROTOTYPE scope is only claimed when the footprint genuinely
            # recurs; a lone instance gets INSTANCE scope so nothing downstream
            # is tempted to reuse it more widely than we verified.
            scope = (SolderReuseScope.PROTOTYPE if len(group) > 1
                     else SolderReuseScope.INSTANCE)
            artifact_id = f"surface_evolver_{key}"
            if rep.get("kind") == "tht":
                # THT joints are plated-hole features solved in WORLD
                # coordinates -- board-absolute, one artifact per instance,
                # nothing to place or reuse.
                frame = SolderCoordinateFrame.WORLD
                scope = SolderReuseScope.INSTANCE
                to_output = IDENTITY_MATRIX_4X4
            else:
                # Meshes come back in the SOLVER frame (board at z=0). This is
                # the rotation back to component-local, which the contract says
                # placement should apply before the instance transform.
                frame = SolderCoordinateFrame.COMPONENT_LOCAL
                to_output = tuple(rep["solver_to_local"])
            artifacts.append(SolderGeometryArtifact(
                artifact_id=artifact_id,
                work_item_ids=tuple(e["work_item_id"] for e in group),
                component_paths=tuple(e["instance_path"] for e in group),
                coordinate_frame=frame,
                reuse_scope=scope,
                meshes=tuple(meshes),
                solver_to_output_transform=to_output,
                metadata={
                    "footprint_key": key,
                    "instances": len(group),
                    "joints": len(meshes),
                    "refdes": [e["refdes"] for e in group],
                },
            ))
            for e in group:
                coverage.append(SolderWorkItemCoverage(
                    work_item_id=e["work_item_id"],
                    component_paths=(e["instance_path"],),
                    status=SolderCoverageStatus.GENERATED,
                    artifact_ids=(artifact_id,),
                ))

        for s in skipped:
            coverage.append(SolderWorkItemCoverage(
                work_item_id=s.get("work_item_id") or "unassigned",
                component_paths=tuple(s.get("component_paths", ())),
                status=SolderCoverageStatus.INELIGIBLE,
                message=s.get("reason"),
            ))

        return SolderSolverResult(
            request_id=request.request_id,
            solver_id=self.solver_id,
            artifacts=tuple(artifacts),
            coverage=tuple(coverage),
            warnings=tuple(warnings),
            metadata={
                "distinct_footprints": len(groups),
                "instances_covered": sum(len(g) for g in groups.values()),
                "skipped": len(skipped),
            },
        )

    def _run_solves(self, groups, opts):
        """Solve one representative per footprint, serially or fanned out.

        Returns ``(solved, failed)`` keyed by footprint key: ``solved[key]`` is
        the SolderMesh list, ``failed[key]`` is ``(exc, traceback_str)``. The
        caller turns these into artifacts/coverage in a deterministic order, so
        the ONLY thing the worker count changes is wall-clock.

        A failed solve is captured here (never raised) because the contract
        makes a bad footprint SOLVER_FAILED coverage, not a dead request.
        """
        reps = [(key, group[0]) for key, group in groups.items()]
        budget = _resolve_core_budget(opts)
        # Armed before ANY solve (and before the pool forks below): the serial
        # path still fans joints across threads inside one footprint, and the
        # pool children must inherit the same kernel semaphore for the budget
        # to hold across processes.
        _install_solve_gate(budget)
        workers = min(budget, len(reps))
        solved: dict[str, object] = {}
        failed: dict[str, tuple] = {}

        # One footprint (for example, a single-family board) or a budget of 1:
        # stay serial. No pool overhead, and the stack traces stay local for
        # debug. The joint fan inside the footprint still uses the budget.
        if workers <= 1 or len(reps) <= 1:
            for key, rep in reps:
                try:
                    solved[key] = _solve_footprint_entry(rep, opts)
                except Exception as exc:                       # noqa: BLE001
                    failed[key] = (exc, traceback.format_exc(limit=3))
            return solved, failed

        # Fan out: each solve is an isolated Evolver subprocess, so the unique
        # solves are embarrassingly parallel (solder_evolver/docs/scaling.md).
        # We fork before any solve runs, while this process is still stage-free
        # and has touched no CUDA context, so the pool start is clean -- and
        # the fork is also what carries _solve_gate/_core_budget into the
        # children (a spawn context would reset them to None = ungated).
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_solve_footprint_entry, rep, opts): key
                       for key, rep in reps}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    solved[key] = fut.result()
                except Exception as exc:                       # noqa: BLE001
                    failed[key] = (exc, traceback.format_exc(limit=3))
        return solved, failed

    def _solve_entry(self, entry, opts):
        """Cut + Evolver for one footprint -> (SolderMesh list, warnings)."""
        return _solve_footprint_entry(entry, opts)


def _resolve_core_budget(opts):
    """The total cores the solve may occupy at once.

    ``solver_options["workers"]``: a positive count is a HARD budget -- at
    most that many Evolver subprocesses are alive at any moment, across the
    footprint process pool and every within-footprint thread fan combined
    ("just use X cores, I need the rest"). 0/absent is auto: the whole
    machine minus two, headroom for the OS and the parent (which is doing
    the boolean cuts, mesh readback and USD assembly between solves).
    """
    try:
        want = int(opts.get("workers", 0) or 0)
    except (TypeError, ValueError):
        want = 0
    if want <= 0:
        want = max(1, (os.cpu_count() or 2) - 2)
    return want


def _joint_fan(o, n_distinct):
    """Thread count for the within-footprint fan (distinct solves only).

    Sized to the core budget and clamped to the distinct-solve count; the
    actual scheduling is the solve gate -- threads beyond free budget just
    block cheaply on the semaphore until slots drain back. ``joint_workers``
    (internal debugging) forces an explicit width instead.
    """
    try:
        want = int(o.get("joint_workers", 0) or 0)
    except (TypeError, ValueError):
        want = 0
    if want <= 0:
        want = _core_budget or max(1, (os.cpu_count() or 2) - 2)
    return max(1, min(want, max(1, n_distinct)))


#: The defect-record keys that parameterize a strategy: the solder-defect
#: post-solve observables, the shape parameters, plus the sampling seed the
#: per-joint draws (dry-contact azimuth/gap, shape ranges, exposure
#: azimuth) key off. Everything else in a record (pose transforms,
#: selection bookkeeping) is not strategy input and must NOT enter the
#: footprint identity -- hashing per-instance pose transforms would
#: silently break solve sharing for pose-defected instances. EVERY new
#: strategy parameter must be added here, or JobConfig runs silently drop
#: it while direct solve_footprint callers (the lab, tests) still see it.
_OBSERVABLE_KEYS = ("volume_ratio", "gap_fraction", "seed",
                    "lead_climb", "fullness", "pad_exposure")


def _observables(metadata):
    """The strategy-facing slice of one defect record."""
    md = metadata or {}
    return {k: md[k] for k in _OBSERVABLE_KEYS if md.get(k) is not None}


def _mesh_volume(points, face_indices):
    """Enclosed volume of a closed triangle mesh (divergence theorem)."""
    P = np.asarray(points, float)
    T = np.asarray(face_indices, int).reshape(-1, 3)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def _verify_volumes(strategy, solved, jobs, members, order, report):
    """The solder-defect verification rule, volume half.

    Measure each solved representative against its REQUESTED target, warn
    beyond the strategy's tolerance, then discard the measurement -- the
    numbers are never exported (coverage, USD, results), so generation cannot
    quietly become solder metrology of real boards.
    """
    if report is None or not getattr(strategy, "verify_volume", False):
        return
    tol = float(strategy.volume_tolerance)
    for key in order:
        # (P, fc, fi, N) since the vertex-normals fix; the 3-tuple unpack
        # crashed every verify_volume run after it (latent until a defect
        # actually exercised verification end to end).
        P, _fc, fi = solved[key][:3]
        target = float(jobs[members[key][0]][2])
        if target <= 0.0:
            continue
        got = _mesh_volume(P, fi)
        if abs(got - target) > tol * target:
            report.append(
                f"{strategy.defect_type} joint {members[key][0]}: solved "
                f"volume {got:.5f} deviates >{tol:.0%} from the requested "
                f"{target:.5f}")


def _solve_footprint_entry(entry, opts):
    """Boolean cut + Evolver for one footprint -> (SolderMesh list, warnings).

    Module-level (not a bound method) so it is picklable for the process pool:
    it closes over nothing and takes only the plain-data entry and options.
    The strategy is resolved from a plain string HERE -- the strategy object
    itself must not cross the stage-free / process boundary -- then configured
    from the record's observables (volume_ratio, gap_fraction), which DO
    cross it as plain data in the entry.

    The warnings are the post-solve observable checks (verified, then
    discarded); generate() forwards them into the result's warning stream.
    """
    strategy = defect_strategies.resolve(
        entry.get("defect_type") or opts.get("defect"))
    if strategy is not None and entry.get("defect_params"):
        strategy = strategy.configure(entry["defect_params"])
    report: list[str] = []
    if entry.get("kind") == "tht":
        meshes = _solve_tht_entry(entry, strategy, opts, report=report)
    else:
        meshes = solve_footprint(
            strategy=strategy,
            rects=entry["rects"],
            polys=entry.get("polys"),
            pad_polys=entry.get("pad_polys") or None,
            pad_z=entry["pad_z"],
            comp_points=np.asarray(entry["mesh_points"], float),
            comp_faces=np.asarray(entry["mesh_faces"], int),
            paste_thickness=entry.get("paste_thickness") or None,
            name_prefix=f"solder_{entry['footprint_key']}",
            options=opts,
            report=report,
        )
    return meshes, [f"{entry['refdes']}: {w}" for w in report]


def _center_row(barrels, k):
    """The barrel row (cx, cy, r, id) for joint ``k``."""
    return barrels[barrels[:, 3].astype(int) == int(k)][0]


def _carved_blob_key(V):
    """Identity of a CARVED blob, up to translation.

    Recentre the cut geometry to its centroid, round and sort -- so a chip's
    two ends key APART (the component body is subtracted from opposite sides,
    so the blobs are mirror images, not translates), while a BGA's identical
    balls -- or a connector's rows of identical plated holes -- key together.
    Keying the paste footprint or the hole radius instead misses this: those
    are symmetric between joints whose body/lead subtraction is not.

    Shared by the SMT and THT paths ON PURPOSE: the dedup identity must not
    drift between them.
    """
    V = np.asarray(V, float)
    Vc = np.round(V - V.mean(axis=0), 3)
    Vc = Vc[np.lexsort((Vc[:, 2], Vc[:, 1], Vc[:, 0]))]
    return hashlib.sha1(np.ascontiguousarray(Vc).tobytes()).hexdigest()


def _solve_tht_entry(entry, strategy, options, report=None):
    """Cut + Evolver for every plated-hole joint of one THT component.

    ``cut_tht_joint`` carves the barrel plug + crown
    from the packed hole/pad/annulus data, ``emit`` writes the datafile with
    the same wetting angles the SMT path uses, and the volume target is
    ``tht_fill`` of the cut volume. Meshes come back in WORLD coordinates
    (plated holes are board features), so the artifact needs no placement.

    The defect strategy applies exactly as on SMT: it scales (or zeroes) each
    joint's target volume, so insufficient/excess/missing_solder work on THT
    joints with no extra wiring. Pose defects resolve to Nominal here too.
    """
    from pcb_aoi.solder_evolver import boolean as wb
    from pcb_aoi.solder_evolver import tht
    from pcb_aoi.solder_evolver.emit import emit
    from pcb_aoi.solder_evolver.solve import solve_stages

    user = {k: v for k, v in (options or {}).items() if k in DEFAULTS}
    o = strategy.options(dict(DEFAULTS))
    o.update(user)
    theta = {"wet_board": o["theta_board"],
             "wet_metal": o["theta_comp"],
             "wet_body": o["theta_comp"]}
    d = entry["tht"]
    refdes = entry["refdes"]

    barrels = np.asarray(d["barrels"], float)

    def _center(k):
        """XY of joint k's barrel -- the translation datum between twins."""
        row = barrels[barrels[:, 3].astype(int) == int(k)][0]
        return np.array([float(row[0]), float(row[1])], float)

    # Per-joint THT shape: the strategy's spec (a configured defect) wins
    # over the options' (nominal); a [min, max] spec samples per joint,
    # deterministic in (seed, joint index) so the same request re-solves to
    # the same board. The strategy seed is the record's; nominal variation
    # keys off the refdes.
    import zlib

    from pcb_aoi.solder_evolver.defects.base import sample_scalar

    shape_seed = getattr(strategy, "seed", None)
    if shape_seed is None:
        shape_seed = zlib.crc32(str(refdes).encode()) & 0x7FFFFFFF
    #: A strategy carrying its own shape DRIVES the target: target_volume()
    #: is then skipped on THT, so a ratio default cannot double-scale a
    #: shaped defect (and missing_solder, which shapes nothing, still zeroes).
    shape_drives = (getattr(strategy, "lead_climb", None) is not None
                    or getattr(strategy, "fullness", None) is not None)

    def _shape(k):
        rng = np.random.default_rng(int(shape_seed) + int(k))
        climb = getattr(strategy, "lead_climb", None)
        if climb is None:
            climb = o.get("lead_climb")
        full = getattr(strategy, "fullness", None)
        if full is None:
            full = o.get("fullness")
        return (None if climb is None else float(sample_scalar(climb, rng)),
                None if full is None else float(sample_scalar(full, rng)))

    # Phase 1 -- cut every joint (manifold booleans; cheap next to a solve)
    # and key it on the carved blob. A connector's rows of identical plated
    # holes collapse to ONE solve; a hole with a different lead or annulus
    # keys apart. Same two-phase shape as solve_footprint on the SMT side.
    dedup = bool(o.get("joint_dedup", True))
    jobs = {}
    for k in range(int(entry["joint_count"])):
        climb_k, full_k = _shape(k)
        # The strategy may replace the cut step (dry contact translates its
        # gap_fraction observable into the interrupt-cut construction; the
        # barrel radius is what its circumference math needs).
        res = tht.cut_tht_joint(
            d, k, verbose=False,
            barrel_fill=float(o["tht_barrel_fill"]),
            lead_clear=float(o["tht_lead_clear"]),
            crown=float(o["tht_crown"]),
            lead_climb=climb_k,
            **strategy.tht_cut_kwargs(k, float(_center_row(barrels, k)[2])))
        # The class check that cannot lie is topological: a dry-contact CUT
        # must not ring the pin (genus 0). Checked on the cut -- the cut
        # decides the topology; the pinned solve preserves it.
        if report is not None and getattr(strategy, "expect_genus0", False):
            eu = wb.check_body(res["V"], np.asarray(res["F"], int),
                               np.asarray(res["lab"]).astype(str)).get("euler")
            if eu is not None and (2 - eu) // 2 != 0:
                report.append(
                    f"dry-contact joint {k}: cut still rings the barrel "
                    f"(genus {(2 - eu) // 2}); the gap front never crossed "
                    f"the pin")
        if report is not None and getattr(strategy, "expect_genus1", False):
            eu = wb.check_body(res["V"], np.asarray(res["F"], int),
                               np.asarray(res["lab"]).astype(str)).get("euler")
            if eu is not None and (2 - eu) // 2 != 1:
                report.append(
                    f"interrupted joint {k}: the pad-exposure bite BROKE "
                    f"the ring (genus {(2 - eu) // 2}); that is dry "
                    "contact, not interrupted -- lower pad_exposure")
        if full_k is not None:
            # The fullness target: this joint's own plug + straight-cone
            # fillet references, from the cut. volume_ratio stays a MEASURED
            # observable on THT joints (it is only a control for SMT).
            nominal = (float(res["plug_volume"])
                       + full_k * float(res["fillet_cone_volume"]))
            target = (nominal if shape_drives
                      else strategy.target_volume(nominal, k))
            # The seed is an initial condition, NOT a bound: the volume
            # constraint inflates the free surface past it (measured on a
            # through-hole header: 138% of the seed, exact volume, zero overhang past the
            # pinned contact lines). Only an extreme ask -- likely a
            # construction failure -- is worth a warning.
            if target and target > 2.0 * float(res["volume"]) \
                    and report is not None:
                report.append(
                    f"joint {k}: fullness target {target:.4f} mm^3 is over "
                    f"2x the seed volume {res['volume']:.4f} -- expect a "
                    "distorted inflate; check the joint's references")
        else:
            # Legacy: fullness explicitly null -> tht_fill's fraction of the
            # cut volume (deprecated; construction-dependent).
            nominal = float(o["tht_fill"]) * float(res["volume"])
            target = strategy.target_volume(nominal, k)
        if not target or target <= 0.0:
            jobs[k] = None                  # the strategy leaves this joint bare
            continue
        # Target joins the key so a per-joint fill defect on an otherwise
        # identical hole still gets its own solve.
        key = ((_carved_blob_key(res["V"]), round(float(target), 9))
               if dedup else k)
        jobs[k] = (res, key, target, _center(k))

    order, members = [], {}
    for k in sorted(jobs):
        if jobs[k] is None:
            continue
        key = jobs[k][1]
        if key not in members:
            order.append(key)
            members[key] = []
        members[key].append(k)

    # Phase 2 -- one solve per distinct carve, fanned across threads when
    # joint_workers allows (the Evolver is a subprocess; blocking I/O drops
    # the GIL, so the solves overlap under one interpreter).
    def _solve_key(key):
        res, _k, target, _c = jobs[members[key][0]]
        wd = tempfile.mkdtemp(prefix="cdf_tht_")
        keepout = None
        if o["sdf"]:
            # GPU (Warp) bake -- single-file across joint threads, exactly
            # like the SMT path; the Evolver solve stays unlocked.
            with _bake_lock:
                keepout = _bake_tht_collider(d, res, o, wd)
        fe_text, _ = emit(res["V"], res["F"],
                          np.asarray(res["lab"]).astype(str), theta,
                          keepout=keepout)
        with _solve_slot():
            meshes, _s, _t = solve_stages(
                fe_text, target, nrefine=o["nrefine"], wd=wd,
                converge=bool(o["converge"]), nconv=o["nconv"],
                convblock=o["convblock"], hess_stage=int(o["hess_stage"]))
        m = meshes[-1]
        # Split creases + area-weighted vertex normals once per solved
        # representative; twins translate in XY, which leaves normals intact.
        return solder_mesh_shading(np.asarray(m.vertices, float),
                                   np.asarray(m.triangles, int))

    jworkers = _joint_fan(o, len(order))
    if jworkers > 1 and len(order) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jworkers) as pool:
            solved = dict(zip(order, pool.map(_solve_key, order)))
    else:
        solved = {key: _solve_key(key) for key in order}

    _verify_volumes(strategy, solved, jobs, members, order, report)

    # Assemble in joint order: the representative as solved, every twin the
    # same mesh translated to its own barrel centre (identical carves differ
    # by exactly that XY offset; the pad plane is shared, so z is untouched).
    out = []
    for k in sorted(jobs):
        if jobs[k] is None:
            continue
        key = jobs[k][1]
        P, fc, fi, N = solved[key]
        off = jobs[k][3] - jobs[members[key][0]][3]
        out.append(SolderMesh(
            name=f"solder_{refdes}_j{k}",
            points=tuple((float(p[0] + off[0]), float(p[1] + off[1]),
                          float(p[2])) for p in P),
            face_vertex_counts=fc,
            face_vertex_indices=fi,
            normals=N))
    return strategy.postprocess(out)


def register(registry_register=None):
    """Register with the pcb_aoi solder-solver registry. IDEMPOTENT.

    Both ``pcb_aoi.solder_generation`` and this package's ``__init__`` may
    attempt registration, depending on which is imported first -- and the
    registry raises on a duplicate id. Guarding here keeps both call sites safe
    rather than making each of them remember.
    """
    from pcb_aoi.solder_generation.solver_registry import registered_solver_ids

    if SURFACE_EVOLVER_SOLVER_ID in registered_solver_ids():
        return
    if registry_register is None:
        from pcb_aoi.solder_generation.solver_registry import register_solver

        registry_register = register_solver
    registry_register(SurfaceEvolverSolver())


# =========================================================================
# The bridge onto the boolean-cut + Evolver pipeline (formerly runner.py --
# dissolved here now that the pipeline is the pcb_aoi.solder_evolver API).
# =========================================================================

# Defaults chosen from the measurements in docs/ and the volume_sweep help text:
# icosphere avoids the octahedral sphere's isolated valence-4 vertex, and 30
# convergence blocks is well past the visible part of the tail while costing a
# fraction of the 80 that a publication-grade solve wants.
DEFAULTS = {
    # aperture_offset -- the "slumped print" blob, dilated from the actual
    # stencil apertures. Measured wetted fraction of the termination height:
    # sphere/icosphere 96%, hull_offset 76%, aperture_offset 35%,
    # aperture_prism 19%, ellipsoid 17%. A volume-matched sphere engulfs the
    # part -- and since the wetted patch is frozen, that overshoot is then
    # imposed on every volume solved afterwards.
    "blob": "aperture_offset",

    # VOLUME. `fill` scales the TARGET volume only: 1.0 = nominal, small values
    # = the insufficient-solder defect. It does NOT resize the blob, so the
    # footprint is identical across fills and "less solder" cannot be confused
    # with "wets less far".
    "fill": 1.0,
    # Printed paste is roughly half flux by volume, and the flux boils off, so
    # the solder left after reflow is about half the printed deposit. Not a
    # measurement -- it depends on paste type, and neither the sidecar nor the
    # USD spec carries it. Stated here so it is one number to argue with rather
    # than an assumption buried in a volume calculation.
    "metal_fraction": 0.5,
    # Cut a slightly LARGER blob than the target volume. The footprint comes
    # from the cut, so the solve then shrinks into a well-defined contact line
    # instead of `match_volume` inflating the blob to hit the target and
    # dragging the wetted patch up the part with it.
    "blob_scale": 1.25,
    # Flange added at the pad plane before the cut, so the PINNED contact line
    # lands at the pad edge rather than the paste aperture. Distinct from
    # blob_scale, not a duplicate of it: a scaled blob is rounded, so it PINCHES
    # at the pad plane -- on an 0805 at blob_scale 1.6 the cut footprint is
    # 1.045 x 1.235, smaller than the 1.15 x 1.45 aperture it came from. Scaling
    # bulges the middle; the skirt widens the base. Off by default.
    "skirt": 0.0,
    # Fallback only. Every deposit is measured (stage_read.paste_thickness);
    # this is used only when a caller supplies no geometry to measure.
    "stencil": 0.076,

    "nrefine": 2,
    # Newton steps interleaved into the refine ladder (volume_sweep.hess_stage).
    # Not the convergence tail -- see "converge" below, which stays off. The
    # fixed recipe alone leaves a faceted golfball shape; a short hessian_seek
    # burst at each level snaps it into the right one, and it is nearly free
    # because factorisation cost goes with vertex count and the coarse levels
    # are where the faceting is.
    #
    # OFF here, ON in the nominal strategy. Newton is documented to lose against
    # the keep-out collider on STARVED joints -- it stalls 2.8e-2 short of
    # gradient, a wrong shape rather than a tolerance -- and starved-with-
    # collider is exactly what the insufficient/missing strategies are. Nominal
    # runs without a collider, which is the case Newton is good at.
    "hess_stage": 0,
    # Nominal joints do not need the convergence TAIL: with hess_stage above
    # doing the real work in the ladder, the tail moves the energy by ~3e-5 for
    # 4x the wall clock. Turn it on for anything being measured rather than
    # rendered.
    "converge": False,
    "nconv": 30,
    "convblock": 100,
    "theta_board": 35.0,
    "theta_comp": 25.0,
    # --- THROUGH-HOLE. Values were validated against interactive reference
    # renders. tht_fill scales the cut volume to the solve target.
    "tht_barrel_fill": 0.75,
    "tht_lead_clear": 0.7,
    "tht_crown": 0.30,
    # THE TWO DOCUMENTED THT SHAPE PARAMETERS, both dimensionless and
    # component-invariant; scalar or [min, max] sampled per joint.
    #   lead_climb  wetted lead height / measured protrusion. > 1 seeds the
    #               blob over the pin tip (hidden/excess). Explicit None
    #               falls back to the legacy crown/lead_clear construction.
    #   fullness    fillet volume / the joint's own straight-cone fillet
    #               (pad outline lofted to the wetted lead ring). 1.0 is a
    #               flat silhouette, < 1 concave (bright/starved), > 1
    #               convex. Explicit None falls back to tht_fill, which is a
    #               fraction of the CUT volume -- a construction artifact
    #               that reads differently on every footprint (deprecated).
    #               THIS DEFAULT IS THE THT NOMINAL ONLY: the SMT nominal
    #               stays paste-driven (the printed deposit is mass
    #               conservation); SMT honours fullness when a defect
    #               strategy carries it or a caller passes it explicitly.
    "lead_climb": 0.8,
    "fullness": 0.85,
    "tht_fill": 0.9,
    "sdf": True,
    "kpen": 1000.0,
    "sdf_res": 64,
    "sdf_margin": 0.55,
    "sdf_inflate": 0.01,
    # Aperture-clustering distance: deposits whose edges are within this are
    # fused into ONE joint. DEFAULT 0.0 -> one paste aperture is one joint, the
    # spec's own contract ("paste apertures define joints") -- only apertures
    # that actually overlap (a stencil can't print two separate deposits on top
    # of each other) fuse. A positive gap merges *separate* deposits, which on
    # every board seen so far only mis-fuses fine-pitch pins (a TSSOP's 0.2 mm
    # spacing) into one blob. A real windowpane thermal pad (several apertures,
    # one physical pad) is the only case that wants merging, and the sidecar
    # gives no pad grouping to key it off -- so that must be driven by a spec
    # signal, not this distance guess. Left as a knob for that eventual case.
    "gap": 0.0,
    # Solve one representative per identical joint footprint and translate the
    # result to its twins -- an array of identical joints (a BGA's balls, a
    # QFN's pads) becomes ONE solve. Sound because the solder is local to its
    # joint; only pure XY translates are reused (a rotated pair keys apart).
    # Off falls back to a solve per joint.
    "joint_dedup": True,
    # INTERNAL (debugging) -- not a user surface. Forces the
    # within-footprint thread-fan width; 0 (the default) sizes the fan to the
    # core budget and lets the solve gate do the actual scheduling. The user
    # knob is solver_options["workers"]: the TOTAL core budget across the
    # footprint pool and every joint fan (see _resolve_core_budget).
    "joint_workers": 0,
}


def _as_joint_dict(rects, pad_z, comp_points, comp_faces, gap=1.0, strategy=None,
                   polys=None, pad_polys=None):
    """Shape our inputs like the frozen-npz dict ``cut_joint`` expects.

    ``cut_joint`` was written against ``dump_joint.py`` output. Feeding it an
    equivalent mapping keeps ONE implementation of the geometry pipeline rather
    than a second copy that drifts -- which is exactly the bug ``cut_joint``
    exists to prevent (see emit_fe.py's note about the lift).

    Apertures are clustered into joints by proximity, the same way
    ``ingest.board_component`` does: deposits closer than `gap` belong to one
    joint (a QFN thermal pad is several apertures), anything further apart is a
    separate joint (the two ends of an 0805).
    """
    # Clustering is a GEOMETRY STEP a defect strategy may replace: a bridge
    # fuses adjacent pads into one joint, which is the inverse of what a nominal
    # joint wants. The strategy's default is the same union-find rule the
    # file-based ingest uses, so "joint" means the same thing on both paths.
    if strategy is None:
        from pcb_aoi.solder_evolver.defects import resolve as _resolve

        strategy = _resolve("nominal")

    ap = np.asarray([[r[0], r[1], r[2], r[3], pad_z] for r in rects], float)
    jid = np.zeros(len(ap), int)
    for k, idx in enumerate(strategy.cluster(ap, gap)):
        for i in idx:
            jid[i] = k
    return {
        "pad_z": np.asarray(pad_z, float),
        "apertures": ap,
        # Real outlines, parallel to `apertures`. Kept as an object list rather
        # than an array because deposits have different vertex counts.
        "polys": list(polys) if polys is not None else None,
        # Exposed pad outlines (solver frame); cut_joint reads these for the
        # pad-plane clip. None/empty = no clip (pad-less boards).
        "pad_polys": list(pad_polys) if pad_polys else None,
        "joint_id": jid,
        "metal_V": np.asarray(comp_points, float),
        "metal_F": np.asarray(comp_faces, int).reshape(-1, 3),
        # Everything is handed in as one mesh; the metal/body split only ever
        # existed because the asset tags are unreliable and we search both.
        "body_V": np.zeros((0, 3), float),
        "body_F": np.zeros((0, 3), int),
    }


def solve_footprint(rects, pad_z, comp_points, comp_faces, name_prefix,
                    paste_thickness=None, options=None, strategy=None,
                    polys=None, pad_polys=None, report=None):
    """Cut and minimise one component's joints. Returns a list of SolderMesh.

    The paste deposits of ONE component are several separate joints, not one --
    an 0805 has a deposit at each end, a QFN one per pad. They must be clustered
    and solved individually; fusing them would grow a single blob bridging the
    whole part, which is a solder BRIDGE defect, not a nominal joint.

    `report`, when given, receives the post-solve observable-check warnings
    (defect strategies only; see _verify_volumes). Checked, then discarded.
    """
    from pcb_aoi.solder_evolver import boolean as wb
    from pcb_aoi.solder_evolver.emit import emit
    from pcb_aoi.solder_evolver.solve import solve_stages

    from pcb_aoi.solder_evolver.defects import resolve as _resolve

    if strategy is None:
        strategy = _resolve("nominal")

    # Precedence: package defaults -> strategy -> caller. The strategy supplies
    # the defaults FOR THAT DEFECT, but an explicit option must still win, or
    # tuning a run per part family silently does nothing.
    user = {k: v for k, v in (options or {}).items() if k in DEFAULTS}
    o = strategy.options(dict(DEFAULTS))
    o.update(user)

    d = _as_joint_dict(rects, pad_z, comp_points, comp_faces, gap=o["gap"],
                       pad_polys=pad_polys,
                       strategy=strategy, polys=polys)
    theta = {"wet_board": o["theta_board"],
             "wet_metal": o["theta_comp"],
             "wet_body": o["theta_comp"]}
    joint_ids = strategy.joint_ids(sorted(set(int(j) for j in d["joint_id"])))

    # Solder volume, from the printed deposit rather than a guessed stencil.
    #   solder = paste_area x measured_thickness x metal_fraction
    # The BLOB is cut to blob_scale x that, and the SOLVE targets fill x that,
    # so the footprint is fixed by the blob and the volume is set independently.
    from pcb_aoi.solder_evolver.volume import volume_plan

    blob_stencil, target_fraction = volume_plan(
        paste_thickness, o["metal_fraction"], o["blob_scale"])

    # Solidify the component ONCE for all its joints. cut_joint would otherwise
    # re-run trimesh repair over the whole mesh per joint, which is the entire
    # cost on parts with many pads.
    lift = wb.component_lift(d)
    comp_solids = wb.build_component_solids(d, lift=lift)

    apert = d["apertures"]
    jid_of = d["joint_id"]
    dedup = bool(o.get("joint_dedup", True))

    def _center(jid):
        """XY centroid of a joint's paste rects -- the translation datum."""
        r = apert[jid_of == jid]
        return np.array([float(np.mean((r[:, 0] + r[:, 1]) * 0.5)),
                         float(np.mean((r[:, 2] + r[:, 3]) * 0.5))], float)

    def _cut(jid):
        """The blob cut for one joint -- the cheap half; the solve is the pole.
        fill stays 1: the inflation rides in through stencil, so fill never
        touches the footprint."""
        jpolys = None
        if d["polys"] is not None:
            jpolys = [P for P, k in zip(d["polys"], jid_of)
                      if int(k) == jid and P is not None]
        return wb.cut_joint(d, jid, o["blob"], stencil=blob_stencil,
                            fill=1.0, verbose=False, skirt=o["skirt"],
                            polys=jpolys or None, lift=lift,
                            component_solids=comp_solids,
                            **strategy.smt_cut_kwargs(jid))

    def _blob_key(res):
        """A joint reuses another's SOLVE only when its CARVED blob is identical
        up to translation -- see _carved_blob_key for why the paste footprint
        alone is not a safe key.
        """
        return _carved_blob_key(res["V"])

    def _emit_cut(res, jid, wd):
        lab = np.asarray(res["lab"]).astype(str)
        # The bake touches the GPU (Warp); serialise it across joint threads.
        # The Evolver solve is the long pole and stays unlocked, so the threads
        # still overlap there -- only the short bake is single-file.
        if o["sdf"]:
            with _bake_lock:
                keepout = _bake_collider(d, res, o, wd, jid)
        else:
            keepout = None
        return emit(res["V"], res["F"], lab, theta, keepout=keepout)[0]

    def _mesh(jid, P, fc, fi, N, off):
        pts = tuple((float(p[0] + off[0]), float(p[1] + off[1]), float(p[2]))
                    for p in P)
        return SolderMesh(name=f"{name_prefix}_j{jid}", points=pts,
                          face_vertex_counts=fc, face_vertex_indices=fi,
                          normals=N)

    # Per-joint SMT fullness. The SMT nominal is PASTE-DRIVEN (the printed
    # deposit is mass conservation, real physics), so fullness engages only
    # when a defect strategy carries it or the caller passes it EXPLICITLY --
    # the DEFAULTS entry is the THT nominal and deliberately does not apply
    # here. lead_climb is THT-only. Ranges sample per joint, deterministic
    # in (record seed | refdes, joint id), same as the THT path.
    import zlib

    from pcb_aoi.solder_evolver.defects.base import sample_scalar

    _strategy_full = getattr(strategy, "fullness", None)
    _user_full = (options or {}).get("fullness") if "fullness" in (
        options or {}) else None
    _full_spec = _strategy_full if _strategy_full is not None else _user_full
    _full_drives = _strategy_full is not None
    _shape_seed = getattr(strategy, "seed", None)
    if _shape_seed is None:
        _shape_seed = zlib.crc32(str(name_prefix).encode()) & 0x7FFFFFFF

    def _fullness(jid):
        if _full_spec is None:
            return None
        rng = np.random.default_rng(int(_shape_seed) + int(jid))
        return float(sample_scalar(_full_spec, rng))

    # Phase 1 -- cut every joint (cheap vs the solve) and record what solving it
    # needs: the carved-blob key, the target volume and the joint centre. None
    # marks a joint the strategy leaves bare. res["target_volume"] is the
    # INFLATED blob volume; divide it back out to the printed nominal.
    jobs = {}
    for jid in joint_ids:
        res = _cut(jid)
        full_j = _fullness(jid)
        if full_j is not None and res.get("fillet_cone_volume", 0.0) > 1e-6:
            # The fullness target: this joint's own straight-cone fillet
            # (under-lead standoff included). The seed is an initial
            # condition, not a bound -- the volume constraint inflates past
            # it when needed.
            nominal = full_j * float(res["fillet_cone_volume"])
            target = (nominal if _full_drives
                      else strategy.target_volume(nominal, jid))
        else:
            if full_j is not None and report is not None:
                report.append(
                    f"joint {jid}: fullness requested but the fillet "
                    "reference is degenerate (no wetted termination?); "
                    "falling back to the paste-driven target")
            nominal = (float(o["fill"]) * res["target_volume"]
                       * target_fraction)
            target = strategy.target_volume(nominal, jid)
        if not target or target <= 0.0:
            jobs[jid] = None
            continue
        # Target joins the key so a per-joint fill defect on an otherwise
        # identical footprint still gets its own solve.
        key = (_blob_key(res), round(float(target), 9)) if dedup else jid
        jobs[jid] = (res, key, target, _center(jid))

    # Phase 2 -- one solve per distinct key, fanned across threads when
    # joint_workers > 1 (the Evolver solve is a subprocess, so its blocking I/O
    # drops the GIL and the solves overlap under one interpreter).
    order, members = [], {}
    for jid in joint_ids:
        if jobs[jid] is None:
            continue
        key = jobs[jid][1]
        if key not in members:
            order.append(key)
            members[key] = []
        members[key].append(jid)

    def _solve_key(key):
        res, _k, target, _c = jobs[members[key][0]]
        wd = tempfile.mkdtemp(prefix="cdf_solve_")
        fe_text = _emit_cut(res, members[key][0], wd)
        with _solve_slot():
            meshes, _s, _t = solve_stages(
                fe_text, target, nrefine=o["nrefine"], wd=wd,
                converge=bool(o["converge"]), nconv=o["nconv"],
                convblock=o["convblock"], hess_stage=int(o["hess_stage"]))
        m = meshes[-1]
        # Split creases + area-weighted vertex normals once per solved
        # representative; twins translate in XY, which leaves normals intact.
        return solder_mesh_shading(np.asarray(m.vertices, float),
                                   np.asarray(m.triangles, int))

    jworkers = _joint_fan(o, len(order))
    if jworkers > 1 and len(order) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jworkers) as pool:
            solved = dict(zip(order, pool.map(_solve_key, order)))
    else:
        solved = {key: _solve_key(key) for key in order}

    _verify_volumes(strategy, solved, jobs, members, order, report)

    # Assemble in joint order: the representative as solved, every twin the same
    # mesh translated to its own centre (the carved blobs are identical up to
    # exactly that translation).
    out = []
    for jid in joint_ids:
        if jobs[jid] is None:
            continue
        key = jobs[jid][1]
        P, fc, fi, N = solved[key]
        off = jobs[jid][3] - jobs[members[key][0]][3]
        out.append(_mesh(jid, P, fc, fi, N, off))
    return strategy.postprocess(out)


def _bake_tht_collider(d, res, o, wd):
    """Bake the PIN keep-out SDF for one THT joint.

    Same grid mechanism as the SMT bake, THT-shaped: bounds come from the cut
    body itself (which already spans crown + plug + fillet at the largest
    scale the solve will see), the keep-out mesh is the component metal
    cropped to that box, and floor_z stays None -- the SMT floor folds "no
    solder below the pad" into the field, which for a BOTTOM-side fillet
    (solder legitimately below the pad plane) would poison the whole region.
    The board contact needs no floor: the wetted patches are pinned.

    Without this the free meniscus has no knowledge of the pin at all:
    an over-the-top dome (lead_climb > 1) simply descends THROUGH the lead
    to a sphere cap, and even sub-tip fillets can graze the pin flanks.
    """
    from pcb_aoi.solder_evolver import bake_keepout_sdf

    import manifold3d as m3d

    from pcb_aoi.solder_evolver import boolean as wb

    P = np.asarray(res["V"], float)
    m = float(o["sdf_margin"])
    lo = P.min(0) - m
    hi = P.max(0) + m
    # Bake from a CLOSED solid, never the raw triangle soup: the exports'
    # lead/body meshes are open sheets, and Warp's sign determination needs
    # an interior -- an open-soup bake has NO negative region, so the
    # keep-out penalty is silently INERT and marching-cubes shows only
    # slivers hugging the authored faces. The cut already repaired the
    # whole component once (the J6 hoist caches it on the joint dict);
    # crop that solid to the bake box by intersection.
    solid = d.get("_metal_solid") if isinstance(d, dict) else None
    if solid is None:
        Vc = np.asarray(d["metal_V"], float)
        Fc = np.asarray(d["metal_F"], int).reshape(-1, 3)
        solid = wb.to_manifold(Vc, Fc, repair=True, verbose=False)
        if isinstance(d, dict):
            d["_metal_solid"] = solid
    box = m3d.Manifold.cube(list(hi - lo + 1.0), False).translate(
        list(lo - 0.5))
    crop = solid ^ box
    if crop.is_empty() or crop.volume() <= 0:
        return None
    sV, sF = wb.from_manifold(crop)
    # TRUE surface, no inflate: the SMT default (+0.01) GROWS the keep-out,
    # which puts the THT seed's pin-hugging free vertices inside the field
    # at iteration zero. The initial-penetration blowup that once motivated
    # a half-cell deflate is instead handled by the solve ladder's kpen
    # RAMP (penalty continuation): soft start, full stiffness by the first
    # refine -- so the collider keeps its exact geometry.
    inflate = min(0.0, float(o["sdf_inflate"] or 0.0))
    bake_keepout_sdf(os.path.join(wd, "collider.bin"),
                     np.asarray(sV, float),
                     np.asarray(sF, int).reshape(-1, 3),
                     lo, hi, res=(int(o["sdf_res"]),) * 3, floor_z=None,
                     inflate=inflate)
    return {"grid": "collider.bin", "kpen": float(o["kpen"])}


def _bake_collider(d, res, o, wd, joint_id=0):
    """Bake the component keep-out SDF and drop it beside the datafile.

    The grid path in the datafile resolves against the process cwd, so the .bin
    has to sit in the same directory Evolver runs in. The actual bake is the
    API's bake_keepout_sdf; this only chooses the bounds and the file name.
    """
    from pcb_aoi.solder_evolver import bake_keepout_sdf

    from pcb_aoi.solder_evolver import boolean as wb

    lift = float(res.get("lift", 0.0))
    # CLOSED solid, never raw soup -- an open-sheet bake has no interior and
    # the keep-out penalty is silently inert (see _bake_tht_collider). The
    # repair is the expensive step, so it runs ONCE per component and is
    # cached on `d` (same hoist as the THT path's _metal_solid); the lift is
    # a rigid translate applied to the cached solid per bake.
    # Raw mesh FIRST (lifted), so the fallback below is always bound and
    # already in the lifted frame -- the cached-solid warm path previously
    # left Vc/Fc unassigned when the repair produced an empty solid.
    Vc = np.asarray(d["metal_V"], float) + [0.0, 0.0, lift]
    Fc = np.asarray(d["metal_F"], int).reshape(-1, 3)
    base = d.get("_metal_solid_smt") if isinstance(d, dict) else None
    if base is None:
        base = wb.to_manifold(np.asarray(d["metal_V"], float), Fc,
                              repair=True, verbose=False)
        if isinstance(d, dict) and not base.is_empty() and base.volume() > 0:
            d["_metal_solid_smt"] = base    # cache only a USABLE solid
    solid = base.translate([0.0, 0.0, lift]) if lift else base
    if not solid.is_empty() and solid.volume() > 0:
        sV, sF = wb.from_manifold(solid)
        Vc = np.asarray(sV, float)
        Fc = np.asarray(sF, int).reshape(-1, 3)
    pad_z = float(d["pad_z"])
    rects = np.asarray(d["apertures"], float)[np.asarray(d["joint_id"]) == joint_id]

    m = float(o["sdf_margin"])
    ctop = res.get("comp_top", float("nan"))
    ctop = 1.0 if ctop != ctop else float(ctop)
    lo = np.array([rects[:, 0].min() - m, rects[:, 2].min() - m, pad_z - 0.15])
    hi = np.array([rects[:, 1].max() + m, rects[:, 3].max() + m, pad_z + ctop + m])
    bake_keepout_sdf(os.path.join(wd, "collider.bin"), Vc, Fc, lo, hi,
                     res=(int(o["sdf_res"]),) * 3, floor_z=pad_z,
                     inflate=float(o["sdf_inflate"] or 0.0))
    return {"grid": "collider.bin", "kpen": float(o["kpen"])}
