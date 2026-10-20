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

"""SCRATCH / EXPERIMENTAL -- wet-triangles investigation.

Volume sweep on ONE pinned footprint.

The point of the boolean is that the contact line is derived, not authored. So
cut the joint ONCE, freeze the wetted patches, and then ask Evolver for a range
of solder volumes on that same footprint. The footprint stays put (every wet
vertex is `fixed`); only the free meniscus moves. That is the insufficient ->
good -> excess defect axis, with the wetted region held physically fixed.

Note what is NOT being varied: the blob. Scaling the blob would move the contact
line too, which confounds "more solder" with "wets further" -- the two knobs the
defect taxonomy wants to keep apart.

    .venv/bin/python -m pcb_aoi.solder_evolver.solve --family PASSIVE_0805_g02 \
        --joint 0 --fills 0.4,0.7,1.0,1.5,2.2 --capture-dir /tmp/sweep

--capture-dir writes one USD per fill (pure pxr; open in usdview).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from . import backend as ev
from . import boolean as wb
from .emit import emit


def solve_stages_stream(fe_text, target, nrefine=2, ngrad=60, wd=None, converge=True,
                 nconv=80, convblock=100, tiny=0.25, timeout=1800,
                 hessian="off", nhess=50, hessblock=20, hess_stage=0,
                 cancel=None):
    """The solve as a GENERATOR: yields each checkpoint mesh AS EVOLVER WRITES
    IT, which is what lets a viewer show the minimisation live while the
    process runs. Yields ("stage", tag, mesh) per checkpoint, then one
    ("done", stats, tags). `cancel` is polled between events; returning True
    kills the Evolver process (stats gain CANCELLED=True).

    Completion detection is a stdout marker, not file polling: every `dump` is
    followed by `printf "DUMPED n"`, so a stage file is only ever parsed after
    Evolver has finished writing it.

    (solve_stages below is the original blocking form, now a thin wrapper.)

    Returns (meshes, stats, tags) so callers can inspect or animate the
    minimisation. The convergence pass is broken into `nconv` calls of at most
    `convblock` iterations each, purely so there is something to watch during
    the long tail -- the stopping rule is unchanged, and once the energy stops
    falling the remaining calls return immediately.

    `hessian` picks the minimiser for the convergence pass. GRADIENT DESCENT IS
    STILL THE DEFAULT, and the "off" path below is byte-for-byte what it was:

      "off"     plain `g`, the reference.
      "seek"    Newton instead -- `hconverge` (a block of `hessian_seek`) in
                place of `converge`. Same maintenance cadence, same energy
                stopping rule, so the two are directly comparable.
      "polish"  the gradient pass in full, THEN Newton blocks on top. The
                textbook hybrid: first order to get into the basin, second
                order to finish. The safe choice if "seek" ever wanders.

    `hess_stage` is a DIFFERENT knob: Newton steps interleaved into the refine
    ladder itself, rather than as a tail. The fixed recipe leaves the surface
    faceted -- it holds a golfball shape through every gradient block and only
    snaps into the right one when Newton first touches it. Gradient descent is a
    local mover (the area-gradient force scales with edge length), so a coarse
    faceted mesh relaxes slowly and unevenly; `hessian_seek` solves globally and
    fixes the shape in a handful of steps. Doing that EARLY is cheap -- the
    factorisation cost goes with vertex count, so the same step costs orders of
    magnitude less at the coarse levels than after two refinements. 0 = off, and
    the ladder is byte-for-byte unchanged.

    `hessian_seek` and not raw `hessian`: see the long note in emit_fe.emit().
    The short version is that our Hessian is a bordered KKT system (a fixed body
    volume contributes a Lagrange row, so the matrix has negative eigenvalues BY
    CONSTRUCTION) and our keep-out penalty is C1 but not C2, so the undamped
    Newton step is not trustworthy -- which is how the previous attempt at this
    (commit e6db6ec) inverted triangles. Line-searching along the Newton
    direction and only committing to a step that lowers the energy makes both
    problems non-fatal.

    `stats["WALL"]` is the Evolver subprocess wall clock. ITERATION COUNTS ARE
    NOT COMPARABLE between the two minimisers -- one `hessian_seek` factors a
    sparse matrix and costs far more than one `g` -- so wall clock is the only
    honest axis for "which is faster".
    """
    if hessian not in ("off", "seek", "polish"):
        raise ValueError(f"hessian= must be off/seek/polish, got {hessian!r}")
    wd = wd or tempfile.mkdtemp(prefix="vsweep_")
    open(os.path.join(wd, "j.fe"), "w").write(fe_text)
    cmds, dumps = [], []

    def snap(tag):
        p = os.path.join(wd, f"st{len(dumps):03d}.fe")
        cmds.append(f'dump "{p}";')
        cmds.append(f'printf "DUMPED {len(dumps)}\\n";')
        dumps.append((tag, p))

    # Mesh maintenance. `t` takes an EXPRESSION (verified: `t 1.5` and
    # `t 9*total_length/edge_count` delete identically), so scaling the threshold
    # by the current mean edge length makes one setting work at every refinement
    # level. Keep the fraction well under 1 -- at mean edge 0.167, `t 1.5` deleted
    # every edge in the mesh.
    keep = (f"t {float(tiny):g}*total_length/edge_count;") if tiny else None

    def maintain():
        """Even out the mesh: equiangulate, average vertex positions, drop tiny
        edges. This is the whole point of the pass -- Evolver takes one global
        step scale and the area-gradient force scales with local edge length, so
        a patch of small triangles descends slower than everything around it and
        lags as a visible bump. Evening the sizes attacks that directly."""
        cmds.extend(["u;", "V;"] + ([keep] if keep else []))

    def stage_newton(where):
        """A short Newton burst at the CURRENT refinement level.

        Maintain first, for the same reason the tail does: `hessian_seek` moves
        every free vertex along one globally-solved direction, so a patch of
        degenerate triangles poisons the whole factorisation instead of merely
        lagging the way it does under `g`.
        """
        maintain()
        # extend, not `+=`: `cmds += ...` inside the closure would rebind it as
        # a local and blow up on the read.
        cmds.extend([f"hmax := {int(hess_stage)};", "hconverge;",
                     'printf "SHSTEPS %g\\n", hsteps;',
                     'printf "SHSCALE %.6g\\n", hscale;',
                     'printf "SHENERGY %.12g\\n", total_energy;'])
        snap(f"newton @ {where}")

    cmds += [f"set body[1] target {target:.9g};", "recalc;"]
    # Fixed facets take no part in the minimisation -- their tension term is
    # constant -- so refining them only multiplies whatever triangle quality
    # the cut delivered (a curved lead's wetted band arrives fragmented, and
    # `u`/`V` cannot touch fixed geometry). no_refine keeps them at cut
    # resolution; facets bordering refined neighbours still split as needed
    # for conformity. Measured: identical volume and energy, ~2x fewer fixed
    # facets in the output, wall time down ~10%.
    # BOTH facet and edge: `r` bisects EDGES and facet subdivision follows,
    # so the facet attribute alone is inert -- measured on a THT seed, the
    # 2,080-facet fixed wetted band still quadrupled to 8,320 under one `r`.
    # With the edge attribute it stays at 2,080 and the refined mesh halves.
    # (Fixed geometry cannot move, so refining it buys nothing but wall time;
    # conformity still splits free facets that border the band.)
    cmds.append("set facet no_refine where fixed;")
    cmds.append("set edge no_refine where fixed;")
    # PENALTY CONTINUATION for the keep-out collider, when one is loaded.
    # The seed's wetted facets sit exactly ON the collider surface, so free
    # neighbours can start marginally inside the field (within grid noise) --
    # at full stiffness the penalty gradient launches them on the first
    # iteration ("flying vertices"). Start soft so they relax OUT of the
    # field, then restore full stiffness before the first refine. The
    # collider geometry itself stays exact; only the spring constant ramps.
    # (Gated python-side: Evolver parses an `if` body even when the guard is
    # false, so referencing kpen in a datafile without one is a syntax error.)
    ramp = "parameter kpen" in fe_text
    if ramp:
        cmds.append("kpen0 := kpen; kpen := kpen0/50;")
    # Condition the RAW CUT before anything else touches it -- INCLUDING the
    # first checkpoint. The boolean arrives with sliver bands (the Minkowski
    # blob against dense exporter outlines measured ~28% of facets under
    # 5 deg min-angle), and the first maintenance used to run only AFTER the
    # first `r` -- by which point every sliver has been quartered into four
    # smaller slivers that equiangulation can no longer swap away. THREE
    # rounds, not one: a single u/V pass is one sweep of edge flips and one
    # of averaging, and equiangulation needs the alternation to converge
    # (measured on a THT cut, whole-mesh median min-angle 18 -> 29 -> 34 ->
    # 36 deg over rounds, saturating near 38; the surviving sub-5 residue is
    # fixed wetted facets that maintenance is not allowed to touch). Running
    # them BEFORE the snapshot means the viewer's stage 0 and the first
    # descent both see the conditioned mesh. Same pass again after the first
    # descent, before the ladder starts multiplying facets.
    for _ in range(3):
        maintain()
    snap("initial (the cut)")
    cmds.append("g 30;")
    if ramp:
        cmds += ["kpen := kpen0/7;", "g 15;", "kpen := kpen0;", "g 15;"]
    maintain()
    snap("g 30")
    if hess_stage:
        # The cheapest and most valuable one: coarsest mesh, so the factorisation
        # is tiny and the faceting it removes is the largest.
        stage_newton("coarse")
    for k in range(nrefine):
        cmds += ["r;", f"g {ngrad};"]
        maintain()
        snap(f"refine {k + 1}")
        if hess_stage:
            stage_newton(f"refine {k + 1}")
    cmds += [f"g {ngrad};", "u;", f"g {ngrad};"]
    snap("end of fixed recipe")
    if converge and hessian != "seek":
        # Second pass: the recipe above stops on a COUNT, which is not a
        # stopping criterion. Keep going until the energy stops falling (or the
        # cap trips). Plain `g` only -- no conjugate gradient, no Hessian.
        cmds.append(f"gmax := {int(convblock)};")
        for k in range(nconv):
            # Maintain BEFORE descending, so every stage ends on a minimisation
            # rather than on a perturbation that has not been relaxed yet.
            maintain()
            cmds += ["converge;", 'printf "GSTEPS %g\\n", gsteps;',
                     'printf "GENERGY %.12g\\n", total_energy;']
            snap(f"converge +{(k + 1) * convblock}")
    if hessian != "off":
        # The Newton pass. Structurally identical to the gradient pass above --
        # same maintain-then-descend cadence, same per-block energy trace -- so
        # that a comparison between them is a comparison of the MINIMISER and
        # not of two different schedules.
        #
        # Maintenance still matters, and arguably more: `hessian_seek` moves
        # every free vertex at once along a globally-solved direction, so a
        # patch of degenerate triangles poisons the whole factorisation rather
        # than just lagging locally the way it does under `g`.
        cmds.append(f"hmax := {int(hessblock)};")
        for k in range(nhess):
            maintain()
            cmds += ["hconverge;", 'printf "HSTEPS %g\\n", hsteps;',
                     'printf "HSCALE %.6g\\n", hscale;',
                     'printf "HENERGY %.12g\\n", total_energy;']
            snap(f"hessian +{(k + 1) * hessblock}")
    cmds += ['printf "VOL %.12g\\n", body[1].volume;',
             'printf "AREA %.12g\\n", total_area;',
             'printf "ENERGY %.12g\\n", total_energy;', "q"]
    # Stream the SUBPROCESS: a reader thread turns stdout into DUMPED markers
    # and keeps every line for the stats parse at the end.
    import queue as _queue
    import threading as _threading

    t0 = time.perf_counter()
    proc = subprocess.Popen([ev.find_evolver(), "j.fe"], cwd=wd,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            # The timeout/cancel kills below only run while WE
                            # are alive; this makes the kernel reap the evolver
                            # if the parent dies mid-solve (the orphaned
                            # vsweep_* spinner bug).
                            **ev.die_with_parent_kwargs())
    proc.stdin.write("\n".join(cmds) + "\n")
    proc.stdin.flush()
    proc.stdin.close()

    lines, marks = [], _queue.Queue()

    def _reader():
        import re as _re

        # Evolver's "Enter command:" prompt has no trailing newline, so marker
        # lines arrive GLUED to it ("Enter command: DUMPED 3") -- the same
        # reason the stats parse strips prompts before matching. Search, don't
        # startswith.
        for line in proc.stdout:
            lines.append(line)
            m_ = _re.search(r"DUMPED (\d+)", line)
            if m_:
                marks.put(int(m_.group(1)))
        marks.put(None)

    rt = _threading.Thread(target=_reader, daemon=True)
    rt.start()

    cancelled = False
    emitted = []
    while True:
        if time.perf_counter() - t0 > timeout and not cancelled:
            cancelled = True
            proc.kill()
        if cancel is not None and not cancelled and cancel():
            cancelled = True
            proc.kill()
        try:
            m = marks.get(timeout=0.1)
        except _queue.Empty:
            continue
        if m is None:
            break
        tag, path = dumps[m]
        try:
            mesh = ev.parse_dump(path)
        except Exception:                                     # noqa: BLE001
            continue                    # a kill can truncate the last dump
        emitted.append(tag)
        yield ("stage", tag, mesh)
    rt.join(timeout=5.0)
    proc.wait()
    wall = time.perf_counter() - t0

    out = "".join(lines).replace("Enter command:", "\n")
    got, steps, energies = {}, [], []
    hsteps, henergies, hscales = [], [], []
    shsteps, shenergies, shscales = [], [], []
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("GSTEPS "):
            steps.append(float(s.split()[1])); continue
        if s.startswith("GENERGY "):
            energies.append(float(s.split()[1])); continue
        if s.startswith("SHSTEPS "):
            shsteps.append(float(s.split()[1])); continue
        if s.startswith("SHSCALE "):
            shscales.append(float(s.split()[1])); continue
        if s.startswith("SHENERGY "):
            shenergies.append(float(s.split()[1])); continue
        if s.startswith("HSTEPS "):
            hsteps.append(float(s.split()[1])); continue
        if s.startswith("HSCALE "):
            hscales.append(float(s.split()[1])); continue
        if s.startswith("HENERGY "):
            henergies.append(float(s.split()[1])); continue
        for k in ("VOL", "AREA", "ENERGY"):
            if s.startswith(k + " "):
                try:
                    got[k] = float(s.split()[1])
                except ValueError:
                    pass
    got["WALL"] = wall
    got["CANCELLED"] = cancelled
    got["GSTEPS"] = float(sum(steps))
    got["GBLOCKS"] = steps
    got["GENERGY"] = energies
    got["HSTEPS"] = float(sum(hsteps))
    got["SHBLOCKS"] = shsteps
    got["SHENERGY"] = shenergies
    got["SHSCALE"] = shscales
    got["HBLOCKS"] = hsteps
    got["HENERGY"] = henergies
    got["HSCALE"] = hscales
    got["INDEF"] = sum(1 for l in out.splitlines()
                       if "not positive definite" in l)
    if not emitted and not cancelled:
        print(out[-1500:])
        raise RuntimeError("evolver produced no dump")
    # Judge convergence on ENERGY, not on the step count. With maintenance
    # interleaved, every `V` perturbs the mesh slightly and the next block
    # re-descends, so blocks keep using their full cap long after the energy has
    # settled -- a step-count test would cry wolf on every single run.
    #
    # The threshold is 1e-6 and not smaller because that same perturb/re-descend
    # cycle never fully stops: measured over 200 blocks on a starved 0805 it
    # settles into a steady ~6e-8-per-block crawl that continues indefinitely.
    # Anything tighter than ~1e-6 can therefore never go quiet, however long it
    # runs, and would be a permanent false alarm rather than a signal.
    #
    # Whichever pass ran LAST is the one whose tail matters, so check that one.
    tail, nsteps, knob = ((henergies, sum(hsteps), "--nhess/--hessblock")
                          if hessian != "off" and henergies
                          else (energies, sum(steps), "--nconv/--convblock"))
    if len(tail) >= 2:
        de = tail[-2] - tail[-1]
        rel = de / (1.0 + abs(tail[-1]))
        if rel > 1e-6:
            print(f"    !! still descending: last block gained {de:.2e} "
                  f"({rel:.1e} relative) after {nsteps:.0f} extra "
                  f"iterations -- raise {knob}")
    yield ("done", got, emitted)


def solve_stages(fe_text, target, **kw):
    """Blocking form of :func:`solve_stages_stream` -- identical schedule and
    stats; collects every checkpoint and returns (meshes, stats, tags)."""
    meshes, tags, stats = [], [], {}
    for kind, a, b in solve_stages_stream(fe_text, target, **kw):
        if kind == "stage":
            tags.append(a)
            meshes.append(b)
        else:
            stats = a
    if not meshes:
        raise RuntimeError("evolver produced no dump (cancelled?)")
    return meshes, stats, tags


def solve(fe_text, target, nrefine=2, ngrad=60, wd=None, converge=True, **kw):
    """Run Evolver on `fe_text` at `target` volume; return (Mesh, stats).

    `wd` runs in an existing directory instead of a fresh temp one -- needed
    when the datafile names an SDF grid file, which is resolved relative to the
    process's cwd and so has to be sitting there already.

    Extra keywords go straight to :func:`solve_stages` -- `hessian=`, `nhess=`
    and friends.
    """
    meshes, got, _ = solve_stages(fe_text, target, nrefine=nrefine, ngrad=ngrad,
                                  wd=wd, converge=converge, **kw)
    return meshes[-1], got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="PASSIVE_0805_g02")
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--blob", default="sphere")
    ap.add_argument("--fills", default="0.4,0.7,1.0,1.5,2.2",
                    help="multiples of the printed paste volume")
    ap.add_argument("--nrefine", type=int, default=2)
    ap.add_argument("--capture-dir", default="")
    ap.add_argument("--hessian", default="off", choices=("off", "seek", "polish"),
                    help="off = gradient descent (default). seek = Newton via "
                         "`hessian_seek` in place of `g`. polish = both. "
                         "See scratch/wet_triangles/bench_hessian.py.")
    ap.add_argument("--theta-board", type=float, default=35.0)
    ap.add_argument("--theta-comp", type=float, default=25.0)
    a = ap.parse_args()

    d = np.load(os.path.join(wb.OUT, a.family + ".npz"))

    # ---- cut ONCE. This footprint is reused for every volume. --------------
    res = wb.cut_joint(d, a.joint, a.blob, verbose=False)
    V, F, lab = res["V"], res["F"], np.asarray(res["lab"]).astype(str)
    paste = res["target_volume"]
    theta = {"wet_board": a.theta_board, "wet_metal": a.theta_comp,
             "wet_body": a.theta_comp}
    fe_text, mvol = emit(V, F, lab, theta)

    uniq, cnt = np.unique(lab, return_counts=True)
    print(f"{a.family} joint {a.joint} -- footprint cut once, then pinned")
    print(f"  lift {res['lift']:+.4f}  clearance {res['clearance']:+.4f}")
    print(f"  facets: " + ", ".join(f"{u}={c}" for u, c in zip(uniq, cnt)))
    print(f"  printed paste volume = {paste:.6f} mm^3\n")

    wetv = np.unique(F.reshape(-1, 3)[lab != "meniscus"].ravel())
    zpad = float(V[wetv][:, 2].min())
    # The tallest vertex of the whole body is usually a PINNED one on the wetted
    # patch, so reporting it gives the same number at every fill and hides the
    # very thing the sweep is for. Measure the free surface: everything strictly
    # above the top of the frozen footprint.
    wet_top = float(V[wetv][:, 2].max())

    print(f"  footprint frozen up to {wet_top - zpad:.4f} mm above the pad; "
          f"'free h' below measures only what is above that\n")
    print(f"{'fill':>6} {'target':>10} {'evolver vol':>12} {'area':>10} "
          f"{'energy':>11} {'apex h':>8} {'free h':>8} {'verts':>7}")
    rows = []
    for s in [float(x) for x in a.fills.split(",")]:
        target = s * paste
        try:
            mesh, st = solve(fe_text, target, nrefine=a.nrefine,
                             hessian=a.hessian)
        except Exception as e:
            print(f"{s:6.2f} {target:10.5f}   FAILED: {type(e).__name__}: {e}")
            continue
        P = np.asarray(mesh.vertices)
        h = float(P[:, 2].max() - zpad)
        free = P[P[:, 2] > wet_top + 1e-9]
        fh = float(free[:, 2].max() - zpad) if len(free) else float("nan")
        print(f"{s:6.2f} {target:10.5f} {st.get('VOL', float('nan')):12.6f} "
              f"{st.get('AREA', float('nan')):10.5f} "
              f"{st.get('ENERGY', float('nan')):11.5f} {h:8.4f} {fh:8.4f} "
              f"{len(P):7d}")
        rows.append((s, mesh))

    if a.capture_dir:
        os.makedirs(a.capture_dir, exist_ok=True)
        # Engine debug artifact: one orbitable USD per fill (pure pxr; view in
        # usdview).
        from .viz.usd_export import export_scene
        for s, mesh in rows:
            out = os.path.join(a.capture_dir, f"fill_{s:.2f}.usda")
            export_scene(out, [[mesh]], {})
            print("  wrote", out)


if __name__ == "__main__":      # `solve` is imported by the viewers/adapter.py --solve
    main()
