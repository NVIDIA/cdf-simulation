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

"""Surface Evolver path: equilibrium solder-joint shapes for optical-inspection frames.

This is an ADDITIVE path, independent of the MPM solver. Surface Evolver (Ken Brakke,
https://kenbrakke.com/evolver/) is the 30-year industry-standard tool for solder-fillet
equilibrium shapes: it minimises surface energy subject to a VOLUME constraint and
per-surface CONTACT-ANGLE boundary conditions, and it is what NIST's Solder Interconnect
Design Team and Racz & Szekely used for solder menisci.

We do NOT reimplement an energy minimiser. This module only:
  * locates a Surface Evolver binary (see :func:`find_evolver`),
  * fills a datafile's run-time PARAMETERs (contact angle, volume, width),
  * drives Evolver to equilibrium over stdin (refine / iterate / equiangulate / hessian),
  * parses the dumped datafile into a triangle mesh, and
  * writes STL / OFF for rendering.

One validated datafile ships in ``datafiles/``:
  * ``sessile.fe`` -- a sessile drop at a prescribed contact angle (recovers it to
    <0.5 deg). Kept deliberately: it is the reference for the CONSTRAINT-INTEGRAND
    Young formulation (wetting energy as a vector integrand on the constraint, no
    bottom facets), which is the path back into a sliding contact line after the
    free-boundary attempt on denomme/young-pad diverged below ~70 deg.

Contact angle is imposed by a constraint ``energy`` integral ( = -cos(theta) * wetted area );
this is the numerically stable equivalent of "wall facet tension = -cos theta" (a literal
negative-tension facet runs away). Volume of the open-based body is closed by a constraint
``content`` integral.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

DATADIR = Path(__file__).parent / "datafiles"


def die_with_parent_kwargs():
    """Popen/run kwargs that tie an Evolver subprocess's lifetime to its parent.

    Evolver runs a pre-written command script with stdin already closed, so if
    the PARENT dies (app quit, worker killed) nothing stops the child: no one
    is left to enforce the timeout, and a non-converging script step crunches
    at 100% CPU forever. Observed at scale: 23 orphaned `vsweep_*` evolvers,
    two of them seven days old, quietly stealing most of a workstation.

    ``PR_SET_PDEATHSIG`` makes the KERNEL deliver SIGKILL to the child when
    its spawner dies -- no cooperation from either side required. Linux-only
    (elsewhere this is a no-op and the timeout paths remain the only guard).

    Caveat, deliberate: the death signal fires when the spawning THREAD exits,
    not just the process. Every solve here is synchronous -- the calling
    thread blocks until the evolver exits -- so a thread cannot outlive its
    child, and joint-level ThreadPool fan-out stays safe.
    """
    import sys

    if sys.platform != "linux":                          # pragma: no cover
        return {}
    import ctypes
    import signal

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    PR_SET_PDEATHSIG = 1

    def _preexec():                                      # runs in the child
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)

    return {"preexec_fn": _preexec}

# Binary names shipped by the Debian/Ubuntu `evolver` package (evolver-nox) and upstream.
_EVOLVER_NAMES = ("evolver", "evolver-nox-d", "evolver-nox-q", "evolver-nox-ld", "evolver-ogl")


class EvolverNotFound(RuntimeError):
    pass


def vendored_evolver_dir() -> Path:
    """``pcb_aoi/solder_evolver/vendor/surface-evolver`` -- where this
    package's own ``third_party/evolver/build.sh`` installs the from-source
    build. Package-relative on purpose: the engine carries its source, its
    build, and its binary under one roof."""
    return Path(__file__).resolve().parent / "vendor" / "surface-evolver"


def find_evolver() -> str:
    """Return a path to a runnable Surface Evolver binary, or raise EvolverNotFound.

    Search order:
      1. ``$SOLDER_EVOLVER_BIN`` (explicit override),
      2. our from-source build under ``vendor/surface-evolver/``,
      3. any of ``evolver`` / ``evolver-nox-d`` / ... on ``$PATH``.

    The vendored build is preferred over ``$PATH`` **on purpose**: the
    Debian/Ubuntu `evolver` package is compiled WITHOUT ``-DENABLE_DLL``, so it
    cannot ``load_library`` our grid-SDF constraint at all (it rejects the
    datafile outright). A system binary is only a fallback for the legacy
    formula-string datafiles. Build the good one with
    ``third_party/evolver/build.sh``.
    """
    # Which binary solved the surface is a geometry input, and an override of
    # it would otherwise appear in no config, no job summary and no commit. It
    # goes into the provenance registry AFTER resolution: the record carries
    # the binary that actually ran, with the raw override beside it, so an
    # override pointing at a missing file cannot masquerade as the solver.
    from cdf.provenance import record_override

    raw = os.environ.get("SOLDER_EVOLVER_BIN")
    fallback = _search_evolver()
    if raw and Path(raw).exists():
        resolved: str | None = raw
    else:
        if raw:
            warnings.warn(
                f"SOLDER_EVOLVER_BIN points at {raw!r}, which does not exist; "
                "falling back to the vendored binary or $PATH.",
                RuntimeWarning,
                stacklevel=2,
            )
        resolved = fallback
    record_override("SOLDER_EVOLVER_BIN", fallback, raw, resolved)
    if resolved is None:
        raise EvolverNotFound(
            "Surface Evolver binary not found. Build it with "
            "`pcb_aoi/solder_evolver/third_party/evolver/build.sh`, or set "
            "$SOLDER_EVOLVER_BIN, or install the (DLL-less) system package with "
            "`sudo apt-get install evolver`."
        )
    return resolved


def _search_evolver() -> str | None:
    """Return the binary used when ``$SOLDER_EVOLVER_BIN`` is unset or unusable.

    The vendored build first, then ``$PATH``; ``None`` when neither has one.
    """
    vend = vendored_evolver_dir()
    for name in _EVOLVER_NAMES:
        cand = vend / name
        if cand.exists():
            return str(cand)
    for name in _EVOLVER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def evolver_available() -> bool:
    try:
        find_evolver()
        return True
    except EvolverNotFound:
        return False


def evolver_has_dll(binary: str | None = None) -> bool:
    """Whether `binary` was compiled with ``-DENABLE_DLL`` (dynamic load).

    Required for grid-SDF constraints (``load_library "x.so"``). Checked by
    probing for the compiled-out error string rather than by running the binary,
    which is cheap and avoids Evolver's interactive prompt on a bad datafile.
    """
    try:
        path = binary or find_evolver()
    except EvolverNotFound:
        return False
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
    except OSError:
        return False
    # Present only in the #else branch of load_library() -- i.e. in builds
    # WITHOUT ENABLE_DLL. Its absence is the positive signal.
    return b"not compiled for dynamic load libraries" not in blob


# --------------------------------------------------------------------------- mesh
@dataclass
class Mesh:
    """A triangle mesh recovered from an Evolver dump."""

    vertices: list  # list[(x,y,z)]
    triangles: list  # list[(i,j,k)] into vertices
    face_on_constraint: list = field(default_factory=list)  # per-triangle bool
    # Per-vertex `fixed` flag, straight from the dump. Needed to tell the FREE
    # surface from the pinned wetted footprint after the fact: a mesh-quality
    # measurement that includes the frozen patches mostly measures the boolean
    # cut, which is identical in every run and drowns out what the solve did.
    # Empty when the dump had no vertices section (nothing else depends on it).
    vertex_fixed: list = field(default_factory=list)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    def free_surface(self) -> "Mesh":
        """Sub-mesh of only the liquid-air (free, non-constrained) facets."""
        keep = [t for t, c in zip(self.triangles, self.face_on_constraint) if not c]
        used = sorted({i for t in keep for i in t})
        remap = {old: new for new, old in enumerate(used)}
        return Mesh(
            vertices=[self.vertices[i] for i in used],
            triangles=[(remap[a], remap[b], remap[c]) for a, b, c in keep],
            face_on_constraint=[False] * len(keep),
        )

    def arrays(self):
        """Return (vertices[N,3] float32, triangles[M,3] int32) as numpy arrays."""
        import numpy as np

        return (np.asarray(self.vertices, np.float32),
                np.asarray(self.triangles, np.int32).reshape(-1, 3))

    def vertex_normals(self):
        """Area-weighted per-vertex normals (float32 [N,3]) for smooth shading."""
        import numpy as np

        V, F = self.arrays()
        n = np.zeros_like(V)
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        for k in range(3):
            np.add.at(n, F[:, k], fn)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        return (n / np.maximum(ln, 1e-20)).astype(np.float32)

    def transformed(self, scale=(1.0, 1.0, 1.0), translate=(0.0, 0.0, 0.0), mirror_x=False) -> "Mesh":
        """Return a copy with per-axis ``scale`` then ``translate`` applied (optionally x-mirrored).

        Used to place a normalised Evolver fillet into the physical capacitor scene.
        Triangle winding is flipped on mirror so outward normals stay outward.
        """
        sx, sy, sz = scale
        tx, ty, tz = translate
        mx = -1.0 if mirror_x else 1.0
        verts = [(mx * (sx * x) + tx, sy * y + ty, sz * z + tz) for x, y, z in self.vertices]
        tris = ([(a, c, b) for a, b, c in self.triangles] if mirror_x else list(self.triangles))
        con = list(self.face_on_constraint)
        return Mesh(vertices=verts, triangles=tris, face_on_constraint=con,
                    vertex_fixed=list(self.vertex_fixed))

    def write_off(self, path) -> None:
        path = Path(path)
        with path.open("w") as fh:
            fh.write("OFF\n")
            fh.write(f"{self.n_vertices} {self.n_triangles} 0\n")
            for x, y, z in self.vertices:
                fh.write(f"{x:.9g} {y:.9g} {z:.9g}\n")
            for a, b, c in self.triangles:
                fh.write(f"3 {a} {b} {c}\n")

    def write_stl(self, path, name="solder") -> None:
        import math

        path = Path(path)
        with path.open("w") as fh:
            fh.write(f"solid {name}\n")
            for a, b, c in self.triangles:
                p, q, r = (self.vertices[a], self.vertices[b], self.vertices[c])
                ux, uy, uz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
                vx, vy, vz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
                nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
                nn = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
                fh.write(f"  facet normal {nx / nn:.6e} {ny / nn:.6e} {nz / nn:.6e}\n")
                fh.write("    outer loop\n")
                for vx3 in (p, q, r):
                    fh.write(f"      vertex {vx3[0]:.6e} {vx3[1]:.6e} {vx3[2]:.6e}\n")
                fh.write("    endloop\n  endfacet\n")
            fh.write(f"endsolid {name}\n")


def _join_continuations(text: str) -> list:
    """Evolver dumps wrap long lists with a trailing backslash; rejoin them."""
    out, buf = [], ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
        else:
            out.append(buf + line)
            buf = ""
    if buf:
        out.append(buf)
    return out


def parse_dump(path) -> Mesh:
    """Parse a Surface Evolver dumped datafile into a triangle :class:`Mesh`.

    Reads the ``vertices`` (id -> xyz), ``edges`` (id -> v1,v2) and ``faces``
    (id -> signed edge loop) sections, resolves each face loop to ordered vertices
    and fan-triangulates. Faces carrying a ``constraint`` attribute are flagged so
    the free liquid-air surface can be isolated (see :meth:`Mesh.free_surface`).
    """
    text = Path(path).read_text()
    section = None
    verts, edges, tris, on_con = {}, {}, [], []
    vfixed = {}

    def is_header(tok):
        return tok in ("vertices", "edges", "faces", "facets", "bodies")

    for raw in _join_continuations(text):
        s = raw.strip()
        if not s or s.startswith("//") or s.startswith("/*"):
            continue
        tok0 = s.split()[0].lower()
        if is_header(tok0) and "predicted" not in s:
            section = "faces" if tok0 == "facets" else tok0
            continue
        if not s[0].isdigit():
            if section == "vertices":  # ended by e.g. "read" or a keyword
                pass
            continue
        # strip trailing /* ... */ comment
        c = s.find("/*")
        body = (s[:c] if c >= 0 else s).split()
        try:
            idx = int(body[0])
        except ValueError:
            continue
        if section == "vertices":
            verts[idx] = (float(body[1]), float(body[2]), float(body[3]))
            # Attributes follow the coordinates; `fixed` is the only one we
            # need. Match on the raw line rather than the tokenised `body`,
            # which stops at the first non-numeric token.
            vfixed[idx] = " fixed" in s.lower()
        elif section == "edges":
            edges[idx] = (int(body[1]), int(body[2]))
        elif section == "faces":
            loop = []
            for t in body[1:]:
                try:
                    loop.append(int(t))
                except ValueError:
                    break  # first non-integer -> start of attributes
            constrained = "constraint" in s.lower()
            # ordered vertices around the loop = start vertex of each oriented edge
            ring = []
            for se in loop:
                v1, v2 = edges[abs(se)]
                ring.append(v1 if se > 0 else v2)
            for k in range(1, len(ring) - 1):  # fan-triangulate
                tris.append((ring[0], ring[k], ring[k + 1]))
                on_con.append(constrained)

    order = sorted(verts)
    remap = {vid: i for i, vid in enumerate(order)}
    vlist = [verts[vid] for vid in order]
    tlist = [(remap[a], remap[b], remap[c]) for a, b, c in tris]
    return Mesh(vertices=vlist, triangles=tlist, face_on_constraint=on_con,
                vertex_fixed=[vfixed.get(vid, False) for vid in order])


# --------------------------------------------------------------------------- run
def _evolve_script(assigns: dict, target_volume: float, nrefine: int, dump_path: Path,
                   pre_commands=None, use_hessian: bool = False, refine_commands=None) -> str:
    """Build the stdin command stream that drives Evolver to equilibrium then dumps.

    Gradient descent is the DEFAULT (`use_hessian=False`) and stays that way
    until the second-order path has been measured on the shapes THIS function is
    used for. `use_hessian=True` now issues `hessian_seek` rather than raw
    `hessian`; on the wet-triangle joints that swap turned Hessian from
    "overshoots and inverts triangles" into a 1.8x wall-clock win at a lower
    energy (measured in the retired bench_hessian study; see git history of
    scripts/surface_evolver/scratch), so the option is worth trying again --
    but the pad puddles here have not been re-measured.

    The old warning that an SDF constraint forbids a Hessian no longer applies to
    the GRID collider: its cubic B-spline field is C2 and third_party/evolver
    supplies exact analytic seconds. It still applies to constraints written as
    abs/min/max formula strings, which are only C0.

    `refine_commands` run after each refine (and before the final relax) — e.g. to
    re-apply a contact-angle tension to facets that have newly landed on a wetting
    constraint (`set facet tension ... where on_constraint N`).
    """
    lines = []
    for k, v in assigns.items():
        lines.append(f"{k} := {v};")
    lines.extend(pre_commands or [])                    # e.g. move a fixed contact edge
    lines.append(f"set body[1] target {target_volume};")
    lines.append("recalc;")
    # coarse relax, then progressively refine with equiangulation + vertex averaging
    lines.append("g 20;")
    for _ in range(nrefine):
        lines.append("r;")
        lines.extend(refine_commands or [])            # re-tag newly-created contact facets
        lines += ["g 30;", "u;", "V;"]
    lines.extend(refine_commands or [])
    if use_hessian:
        # `hessian_seek`, NOT raw `hessian`. Raw `hessian` takes the Newton step
        # at stepsize exactly 1 and only WARNS when the factored matrix is
        # indefinite -- and it is indefinite here by construction, because a
        # fixed body volume enters as a bordered Lagrange row. That is how this
        # option came to be reverted as unstable in the first place (e6db6ec:
        # "overshoot ... they invert triangles and blow up"). hessian_seek line
        # searches along the same direction and only commits to a step that
        # lowers the energy, which is measurably both faster and more accurate
        # than the gradient path (scratch/wet_triangles/bench_hessian.py).
        lines += ["hessian_seek;", "g 20;", "u;", "hessian_seek;", "g 20;"]
    else:
        lines += ["g 60;", "u;", "V;", "g 60;", "u;", "g 40;"]
    lines.append(f'dump "{dump_path}";')
    lines.append("q")
    return "\n".join(lines) + "\n"


def run_evolver(
    datafile,
    assigns: dict | None = None,
    target_volume: float = 1.0,
    nrefine: int = 3,
    workdir=None,
    keep: bool = False,
    timeout: float = 300.0,
    pre_commands=None,
    use_hessian: bool = False,
    refine_commands=None,
) -> Mesh:
    """Run Surface Evolver on ``datafile`` to equilibrium and return the mesh.

    Parameters
    ----------
    datafile : path
        A base ``.fe`` (geometry + physics only, no command section).
    assigns : dict
        Run-time PARAMETER overrides, e.g. ``{"ANG": 40}`` or ``{"APAD": 30, "ALEAD": 20}``.
    target_volume : float
        Body target volume (the volume constraint).
    nrefine : int
        Number of refine/iterate rounds (mesh resolution vs. speed).
    pre_commands : list[str] | None
        Extra Evolver commands run after the assigns, before evolving (e.g. moving a fixed
        contact edge to set the fillet climb height).
    """
    binary = find_evolver()
    assigns = assigns or {}
    import tempfile

    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="evolver_"))
    wd.mkdir(parents=True, exist_ok=True)
    dump_path = wd / "out.fe"
    script = _evolve_script(assigns, target_volume, nrefine, dump_path, pre_commands,
                            use_hessian=use_hessian, refine_commands=refine_commands)
    proc = subprocess.run(
        [binary, str(datafile)],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(wd),
        **die_with_parent_kwargs(),
    )
    if not dump_path.exists():
        raise RuntimeError(
            f"Evolver produced no dump.\n--- stdout tail ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-2000:]}"
        )
    mesh = parse_dump(dump_path)
    if not keep and workdir is None:
        shutil.rmtree(wd, ignore_errors=True)
    return mesh


def _stage_script(assigns, target_volume, nrefine, wd, pre_commands=None,
                  use_hessian=False, refine_commands=None):
    """Command stream that dumps a snapshot after each convergence checkpoint.

    Surface Evolver is a quasi-static energy *minimiser* (gradient descent on surface
    energy), not a physical dynamic solver -- there is no time. But the successive
    iterates on the way to equilibrium can be replayed as an animation of "watching it
    solve". Returns (script, [stage_paths]) where the last stage is the equilibrium.

    Mirrors `_evolve_script`: `use_hessian=False` (gradient only) for kinky
    constraints; `refine_commands` re-applied after each refine.
    """
    lines, stages = [], []
    for k, v in assigns.items():
        lines.append(f"{k} := {v};")
    lines.extend(pre_commands or [])
    lines.append(f"set body[1] target {target_volume};")
    lines.append("recalc;")

    def snap():
        p = wd / f"stage_{len(stages):03d}.fe"
        stages.append(p)
        lines.append(f'dump "{p}";')

    snap()                     # initial geometry
    lines.append("g 20;")
    snap()
    for _ in range(nrefine):
        lines.append("r;")
        lines.extend(refine_commands or [])
        lines += ["g 30;", "u;", "V;"]
        snap()
    lines.extend(refine_commands or [])
    if use_hessian:
        # hessian_seek and not raw hessian -- see _evolve_script for why.
        lines += ["hessian_seek;", "g 20;", "u;", "hessian_seek;", "g 20;"]
    else:
        lines += ["g 60;", "u;", "V;", "g 60;", "u;", "g 40;"]
    snap()                     # equilibrium
    lines.append("q")
    return "\n".join(lines) + "\n", stages


def run_evolver_stages(datafile, assigns=None, target_volume=1.0, nrefine=3, timeout=300.0,
                       pre_commands=None, use_hessian=False, refine_commands=None):
    """Solve and return a LIST of meshes, one per convergence checkpoint (last = equilibrium).

    For visualising the minimisation as an animation. See :func:`run_evolver` for params.
    """
    import tempfile

    binary = find_evolver()
    wd = Path(tempfile.mkdtemp(prefix="evolver_stg_"))
    script, stages = _stage_script(assigns or {}, target_volume, nrefine, wd, pre_commands,
                                   use_hessian=use_hessian, refine_commands=refine_commands)
    subprocess.run([binary, str(datafile)], input=script, capture_output=True,
                   text=True, timeout=timeout, cwd=str(wd),
                   **die_with_parent_kwargs())
    meshes = [parse_dump(p) for p in stages if p.exists()]
    shutil.rmtree(wd, ignore_errors=True)
    if not meshes:
        raise RuntimeError("Evolver produced no stage dumps")
    return meshes


# ------------------------------------------------------------------- convenience
def solve_sessile(contact_angle=40.0, volume=1.0, nrefine=3, **kw) -> Mesh:
    """Equilibrium sessile drop at ``contact_angle`` (deg) with the given volume."""
    return run_evolver(
        DATADIR / "sessile.fe",
        assigns={"ANG": float(contact_angle)},
        target_volume=float(volume),
        nrefine=nrefine,
        **kw,
    )

