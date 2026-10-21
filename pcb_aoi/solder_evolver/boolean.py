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

Stage 2: take the frozen joint geometry from ``dump_joint.py``, build a candidate
solder BLOB, and cut it against the board plane + the component metallization with
a real mesh boolean (manifold3d).  Report whether the result is a single closed,
consistently-oriented body whose triangles partition cleanly into

    * meniscus  (came from the blob surface)      -> tension 1
    * wet-board (came from the board half-space)  -> tension -cos(theta_board)
    * wet-comp  (came from the component solid)   -> tension -cos(theta_comp)

and whether the contact curve between them is a conforming edge chain (every
meniscus/wet boundary edge is shared by exactly one triangle of each label).

    .venv/bin/python -m pcb_aoi.solder_evolver.boolean \
        --family D2PAK --joint 1 --blob aperture_offset

Blob candidates (``--blob``):
    sphere            analytic sphere of the paste volume, seated on the pad
    icosphere         the SAME sphere, triangulated from an icosahedron instead
                      of an octahedron. manifold3d's sphere leaves six valence-4
                      poles; whichever one survives the cut becomes an isolated
                      irregular vertex on the free surface and sticks (it drops
                      to the pad on the first descent, then is frozen
                      bit-for-bit through every later refinement). An icosphere
                      has only valence 5 and 6, so no such vertex exists.
    ellipsoid         axis-aligned ellipsoid matched to the aperture bbox
    aperture_prism    the printed paste brick itself (aperture x stencil height)
    aperture_offset   paste brick, isotropically dilated (Minkowski / level-set)
                      then volume-rescaled -- the "slumped print" blob
    hull_offset       convex hull of (paste brick + wettable metal band), dilated
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    import manifold3d as m3d
except ImportError:                                     # pragma: no cover
    raise SystemExit("pip install manifold3d")

HERE = os.path.dirname(os.path.abspath(__file__))
# Dev npz dumps: frozen joints for the module CLIs and the two dump-driven
# tests (which SKIP when none are present -- a fresh checkout has none). The
# location is engine-local and gitignored, for hand-frozen dumps and
# $SOLDER_EVOLVER_OUT overrides.
OUT = os.environ.get("SOLDER_EVOLVER_OUT", os.path.join(HERE, ".dumps"))


# ------------------------------------------------------------------ mesh health

def mesh_health(V, F, name):
    """Edge-manifoldness / closure / orientation report for a raw asset mesh."""
    F = np.asarray(F, int).reshape(-1, 3)
    he = {}
    for t in F:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            he[(a, b)] = he.get((a, b), 0) + 1
    und = {}
    for (a, b), c in he.items():
        k = (min(a, b), max(a, b))
        und[k] = und.get(k, 0) + c
    boundary = sum(1 for c in und.values() if c == 1)
    nonman = sum(1 for c in und.values() if c > 2)
    flipped = sum(1 for (a, b), c in he.items() if c > 1)   # same-direction dupes
    vol = signed_volume(V, F)
    print(f"  [{name}] V={len(V)} F={len(F)} edges={len(und)} "
          f"boundary_edges={boundary} nonmanifold_edges={nonman} "
          f"same_dir_halfedges={flipped} signed_vol={vol:+.5f}")
    return dict(boundary=boundary, nonmanifold=nonman, flipped=flipped, vol=vol)


def signed_volume(V, F):
    V = np.asarray(V, float); F = np.asarray(F, int).reshape(-1, 3)
    if not len(F):
        return 0.0
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


# ---------------------------------------------------------------- manifold glue

def weld(V, F, tol=1e-6):
    """Merge coincident vertices. USD assets arrive per-face-unwelded, which reads
    as a wall of boundary edges and makes every mesh 'non-manifold'."""
    V = np.asarray(V, float); F = np.asarray(F, int).reshape(-1, 3)
    key = np.round(V / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    Vw = V[first]
    Fw = inv[F]
    Fw = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])]
    return Vw, Fw


def cap_boundaries(V, F):
    """Fan-cap every open boundary loop (the trick `sphere_contact_loop` already
    uses for `pkg_lead`'s missing inner face). Returns (V, F, n_loops)."""
    V = np.asarray(V, float).copy(); F = np.asarray(F, int).reshape(-1, 3)
    cnt = {}
    for t in F:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (min(a, b), max(a, b))
            cnt[k] = cnt.get(k, 0) + 1
    bnd = [e for e, c in cnt.items() if c == 1]
    if not bnd:
        return V, F, 0
    nb = {}
    for a, b in bnd:
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    seen, extra, nloop = set(), [], 0
    for s in list(nb):
        if s in seen:
            continue
        loop, cur, prev = [s], s, None
        seen.add(s)
        while True:
            nxts = [x for x in nb[cur] if x != prev and x not in seen]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
            seen.add(cur); loop.append(cur)
        if len(loop) < 3:
            continue
        nloop += 1
        ci = len(V); V = np.vstack([V, V[loop].mean(0)])
        for k in range(len(loop)):
            extra.append([loop[k], loop[(k + 1) % len(loop)], ci])
    if extra:
        F = np.vstack([F, np.array(extra, int)])
    return V, F, nloop


def solidify(V, F, verbose=True):
    """Asset mesh -> valid solid, the four repairs real USD part meshes need:
       1. weld       (assets arrive per-face unwelded)
       2. split      (one prim is often several disjoint shells: tab + 2 leads)
       3. cap        (open shells, e.g. pkg_lead's missing inner face)
       4. re-wind    (winding is not consistent; some tris are flipped)
    Returns (V, F, report)."""
    import trimesh
    V, F = weld(V, F)
    t = trimesh.Trimesh(V, F, process=False)
    shells = t.split(only_watertight=False)
    if not len(shells):
        shells = [t]
    outV, outF, base, rep = [], [], 0, []
    for s in shells:
        sv, sf, ncap = cap_boundaries(np.asarray(s.vertices), np.asarray(s.faces))
        st = trimesh.Trimesh(sv, sf, process=False)
        trimesh.repair.fix_winding(st)
        trimesh.repair.fix_inversion(st)
        rep.append(dict(V=len(st.vertices), F=len(st.faces), caps=ncap,
                        watertight=bool(st.is_watertight), vol=float(st.volume)))
        outV.append(np.asarray(st.vertices)); outF.append(np.asarray(st.faces) + base)
        base += len(st.vertices)
    if verbose:
        for i, r in enumerate(rep):
            print(f"    shell {i}: V={r['V']} F={r['F']} caps={r['caps']} "
                  f"watertight={r['watertight']} vol={r['vol']:+.4f}")
    return np.vstack(outV), np.vstack(outF), rep


def to_manifold(V, F, repair=True, verbose=True):
    if repair:
        V, F, _ = solidify(V, F, verbose=verbose)
    mesh = m3d.Mesh(np.ascontiguousarray(np.asarray(V, np.float32)),
                    np.ascontiguousarray(np.asarray(F, np.uint32).reshape(-1, 3)))
    mesh.merge()
    return m3d.Manifold(mesh)


def from_manifold(man):
    mesh = man.to_mesh()
    return (np.asarray(mesh.vert_properties)[:, :3].astype(float),
            np.asarray(mesh.tri_verts).astype(int))


def tri_labels(man, id_names):
    """Per-triangle provenance label from manifold's run_original_id / run_index."""
    mesh = man.to_mesh()
    ntri = len(mesh.tri_verts)
    lab = np.full(ntri, "?", dtype=object)
    ridx = np.asarray(mesh.run_index)
    rid = np.asarray(mesh.run_original_id)
    for k in range(len(rid)):
        t0, t1 = ridx[k] // 3, ridx[k + 1] // 3
        lab[t0:t1] = id_names.get(int(rid[k]), f"id{int(rid[k])}")
    return lab


# ---------------------------------------------------------------------- solids

def board_solid(pad_z, bbox, pad=5.0, depth=2.0):
    x0, x1, y0, y1 = bbox
    c = m3d.Manifold.cube([x1 - x0 + 2 * pad, y1 - y0 + 2 * pad, depth], center=True)
    return c.translate([0.5 * (x0 + x1), 0.5 * (y0 + y1), pad_z - depth / 2.0])


def joint_polys(d, joint):
    """Unpack the ragged aperture outlines of one joint from a dumped npz.

    Returns None for an npz written before outlines were stored, so old dumps
    keep working and silently fall back to bounding boxes.
    """
    if "poly_xy" not in getattr(d, "files", ()) or "poly_ptr" not in d.files:
        return None
    xy = np.asarray(d["poly_xy"], float)
    ptr = np.asarray(d["poly_ptr"], int)
    jid = np.asarray(d["joint_id"], int)
    out = []
    for i in np.nonzero(jid == joint)[0]:
        seg = xy[ptr[i]:ptr[i + 1]]
        if len(seg) >= 3:
            out.append(seg)
    return out or None


def poly_cross_section(polys):
    """manifold3d CrossSection over the real aperture outlines."""
    return m3d.CrossSection([[(float(q[0]), float(q[1])) for q in P] for P in polys])


def poly_area(polys):
    """Total aperture area from the true outlines, not their bounding boxes.

    Matters for VOLUME, not just looks: solder volume is paste area x thickness,
    and 45% of this board's deposits are rounded. A bbox overstates a stadium
    pad by ~16% and a circular BGA pad by 21% (4/pi).
    """
    return float(poly_cross_section(polys).area())


#: Geometric simplification tolerance for BLOB CONSTRUCTION, in mm. Some
#: exporters tessellate rounded pads into ~68-vertex arc
#: polylines; extruding those and Minkowski-dilating multiplies every outline
#: edge into a band of sliver triangles across the whole blob dome (measured:
#: 2172 pre-cut triangles at 34-41% under 5 deg min-angle, vs 428 from plain
#: rectangular pads). 5 um is far below paste registration tolerance and below
#: anything the solve can resolve (10 um measured as the knee: the sphere
#: bands collapse and the median min-angle jumps 14->36 deg). GEOMETRY ONLY: volume bookkeeping
#: (poly_area) keeps the exact outlines, and rescale_volume normalises the
#: blob to that exact target regardless, so simplification cannot shift
#: solder volume. footprint_key also hashes the exact outlines, so solve
#: dedup identity is untouched.
_BLOB_SIMPLIFY_EPS = 0.010


def paste_prism(rects, pad_z, h, polys=None):
    """Union of the printed paste bricks (aperture outline x stencil thickness).

    `polys` are the real 2D outlines; `rects` their bounding boxes. Prefer the
    polygons -- squaring off a rounded pad changes both the flooded shape and
    the printed volume.
    """
    if polys is not None and len(polys):
        cs = poly_cross_section(polys).simplify(_BLOB_SIMPLIFY_EPS)
        return cs.extrude(h).translate([0.0, 0.0, pad_z])
    parts = []
    for x0, x1, y0, y1, _z in rects:
        parts.append(m3d.Manifold.cube([x1 - x0, y1 - y0, h], center=True)
                     .translate([0.5 * (x0 + x1), 0.5 * (y0 + y1), pad_z + h / 2.0]))
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


def dilate(man, r, seg=16):
    """Isotropic dilation via Minkowski sum with a sphere (rounded offset).

    Minkowski cost is multiplicative -- every input edge sprouts a band of
    sphere patches -- so the result is simplified at the blob tolerance:
    the sliver bands collapse while the rounded shape stays within 5 um.
    """
    return man.minkowski_sum(m3d.Manifold.sphere(r, seg)).simplify(
        _BLOB_SIMPLIFY_EPS)


def rescale_volume(man, target, ctr):
    """Uniform scale about ``ctr`` so the manifold hits ``target`` volume."""
    v = man.volume()
    if v <= 0:
        return man
    s = (target / v) ** (1.0 / 3.0)
    return man.translate([-ctr[0], -ctr[1], -ctr[2]]).scale([s, s, s]).translate(list(ctr))


# ----------------------------------------------------------------------- blobs

def make_blob(kind, rects, pad_z, vol, metalV, stencil=0.13, dil=0.25,
              polys=None):
    cx = float(np.mean([0.5 * (r[0] + r[1]) for r in rects]))
    cy = float(np.mean([0.5 * (r[2] + r[3]) for r in rects]))
    x0 = min(r[0] for r in rects); x1 = max(r[1] for r in rects)
    y0 = min(r[2] for r in rects); y1 = max(r[3] for r in rects)

    if kind == "sphere":
        R = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        # seat so the sphere centre sits a little above the pad -> it cuts the board
        return m3d.Manifold.sphere(R, 64).translate([cx, cy, pad_z + 0.35 * R])
    if kind == "icosphere":
        # Same sphere, different triangulation. manifold3d's sphere is built by
        # subdividing an OCTAHEDRON, which leaves six valence-4 poles; an
        # icosphere has only valence 5 and 6. Whichever pole survives the cut
        # ends up as an isolated irregular vertex on the free surface, and it
        # sticks: see docs -- it drops to the pad on the first descent and is
        # then frozen bit-for-bit through every later refinement.
        import trimesh
        R = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        ico = trimesh.creation.icosphere(subdivisions=3, radius=R)
        man = to_manifold(np.asarray(ico.vertices, float),
                          np.asarray(ico.faces, int), repair=False, verbose=False)
        return man.translate([cx, cy, pad_z + 0.35 * R])
    if kind == "ellipsoid":
        ax, ay = 0.5 * (x1 - x0) * 1.15, 0.5 * (y1 - y0) * 1.15
        az = 3.0 * vol / (4.0 * np.pi * ax * ay)
        s = m3d.Manifold.sphere(1.0, 64).scale([ax, ay, az])
        return s.translate([cx, cy, pad_z])
    if kind == "aperture_prism":
        a = poly_area(polys) if polys is not None and len(polys) else \
            sum((r[1] - r[0]) * (r[3] - r[2]) for r in rects)
        return paste_prism(rects, pad_z, vol / a, polys)
    if kind == "aperture_offset":
        p = paste_prism(rects, pad_z, stencil, polys)
        p = dilate(p, dil)
        return rescale_volume(p, vol, (cx, cy, pad_z))
    if kind == "hull_offset":
        p = paste_prism(rects, pad_z, stencil, polys)
        pts = np.asarray(from_manifold(p)[0])
        if metalV is not None and len(metalV):
            band = metalV[(metalV[:, 2] > pad_z - 0.05) & (metalV[:, 2] < pad_z + 0.8)]
            if len(band):
                pts = np.vstack([pts, band])
        h = m3d.Manifold.hull_points([tuple(map(float, q)) for q in pts])
        h = dilate(h, dil)
        return rescale_volume(h, vol, (cx, cy, pad_z))
    raise ValueError(kind)


# ------------------------------------------------- wet-patch decimation

def exposure_keep(outline_cs, cx, cy, core_r, exposure, angle, rng,
                  amp=0.08, lam=0.35):
    """KEEP region for an INTERRUPTED (pad-exposure) joint, either kind.

    The foot may occupy everything behind a NOISY front pushed in from
    `angle`, union a core disc (radius `core_r` about the joint centre) that
    keeps the joint connected -- the ring around a THT pin, the lead fillet
    on SMT. The front is a band-limited scallop (three sinusoid octaves,
    random phases from `rng`), because a straight dewetting boundary reads
    synthetic; the front DEPTH is bisected so the kept fraction of
    `outline_cs`'s area lands on 1 - exposure exactly -- the noise shapes
    the boundary, never the observable.

    Returns (keep_cs, achieved_exposure).
    """
    area0 = float(outline_cs.area())
    if area0 <= 0.0 or exposure <= 0.0:
        return outline_cs, 0.0
    b = outline_cs.bounds()
    R = 0.5 * float(np.hypot(b[2] - b[0], b[3] - b[1])) + 0.1
    ux, uy = float(np.cos(angle)), float(np.sin(angle))
    vx, vy = -uy, ux
    phases = rng.uniform(0.0, 2.0 * np.pi, 3)
    combs = ((0.55 * amp, lam), (0.30 * amp, lam / 2.3),
             (0.15 * amp, lam / 5.1))

    def noise(tv):
        return sum(a * np.sin(2.0 * np.pi * tv / l + ph)
                   for (a, l), ph in zip(combs, phases))

    core = m3d.CrossSection.circle(float(core_r), 64).translate((cx, cy))
    ts = np.linspace(-R, R, 96)

    def keep_at(depth):
        # front at signed distance `depth` along u from the centre; keep is
        # BEHIND it (dot <= depth + noise), built as a closed polygon in
        # world coordinates, then union the core.
        front = [(cx + (depth + float(noise(tv))) * ux + tv * vx,
                  cy + (depth + float(noise(tv))) * uy + tv * vy)
                 for tv in ts]
        back = [(cx - 2.0 * R * ux + R * vx, cy - 2.0 * R * uy + R * vy),
                (cx - 2.0 * R * ux - R * vx, cy - 2.0 * R * uy - R * vy)]
        poly = front + back
        return m3d.CrossSection([poly]) + core

    lo, hi = -R, R                       # lo = max bite, hi = no bite
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        kept = float((outline_cs ^ keep_at(mid)).area()) / area0
        # kept too small -> retreat the front (raise depth)
        lo, hi = (mid, hi) if kept < 1.0 - exposure else (lo, mid)
    keep = keep_at(0.5 * (lo + hi))
    achieved = 1.0 - float((outline_cs ^ keep).area()) / area0
    return keep, achieved


def decimate_planar_patches(V, F, lab, tol=1e-6, verbose=False):
    """Retriangulate exactly-planar WETTED patches from their boundary loops.

    The boolean leaves the wetted patches as sliver confetti (typically over
    90% of a wet_board's facets under 5 deg min-angle): the cut fragments
    the blob's faces along every intersection curve. The patches are FIXED in
    the solve, so Evolver's mesh maintenance cannot touch them, and each
    refinement quarters every sliver -- left alone they make up ~35% of a
    solved mesh's facets and nearly all of its degenerate ones.

    A fixed patch's interior triangulation is FREE CHOICE: only its boundary
    is shared (with the meniscus at the pinned contact chain, or with a
    neighbouring patch), and only its spanned surface enters the energy and
    volume integrals. Patches are split into maximal CONNECTED COPLANAR
    clusters -- a wet_metal band across a lead's box faces is piecewise
    planar, one cluster per face -- and each cluster is re-triangulated from
    its boundary loops alone via manifold3d's hole-aware triangulator, which
    introduces NO new vertices: every boundary vertex survives bit-for-bit,
    so conformity with the meniscus and with neighbouring clusters cannot
    break, and on an exactly planar cluster the spanned surface (hence
    volume) is unchanged exactly. Genuinely curved clusters degenerate to
    single facets and are left untouched by the size guard.
    """
    V = np.asarray(V, float)
    F = np.asarray(F, int).reshape(-1, 3)
    lab = np.asarray(lab).astype(str)
    keep = np.ones(len(F), bool)
    newF, newL = [], []

    # per-facet unit normals + plane offsets, for the coplanarity test
    _a, _b, _c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    _n = np.cross(_b - _a, _c - _a)
    _areas = np.linalg.norm(_n, axis=1)
    _nu = _n / np.maximum(_areas, 1e-30)[:, None]
    _off = np.einsum("ij,ij->i", _nu, _a)

    def _coplanar(fa, fb):
        return (_nu[fa] @ _nu[fb] > 0.999999
                and abs(_off[fa] - _off[fb]) < tol)

    #: Vertices used by any wetted facet: the pinned set. A free cluster may
    #: be rewritten only when its whole boundary lies on these (see below).
    _wet_verts = set(F[lab != "meniscus"].ravel().tolist())

    for label in sorted(set(lab)):
        sel = np.nonzero(lab == label)[0]
        if not len(sel):
            continue
        # connected COPLANAR clusters: union facets across a shared edge only
        # when they lie in the same plane, so a box-faced lead splits into one
        # cluster per face and each face decimates independently.
        parent = {}
        def find(a):
            root = a
            while parent.get(root, root) != root:
                root = parent[root]
            while parent.get(a, a) != a:
                parent[a], a = root, parent[a]
            return root
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        owner = {}
        for fi in sel:
            if _areas[fi] <= 1e-12:
                continue
            t = F[fi]
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                k = (min(e), max(e))
                if k in owner:
                    if _coplanar(fi, owner[k]):
                        union(fi, owner[k])
                else:
                    owner[k] = fi
        comps = {}
        for fi in sel:
            if _areas[fi] > 1e-12:
                comps.setdefault(find(fi), []).append(fi)

        for fis in comps.values():
            T = F[fis]
            if len(T) < 8:
                continue                       # nothing worth rewriting
            nref = _nu[fis[0]]
            verts = np.unique(T)
            d = (V[verts] - V[verts[0]]) @ nref
            if np.abs(d).max() > tol:
                continue                       # paranoia; clusters are planar
            # directed boundary half-edges (undirected count == 1)
            cnt, direct = {}, {}
            for t in T:
                for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    k = (min(e), max(e))
                    cnt[k] = cnt.get(k, 0) + 1
                    direct[k] = e
            nxt = {}
            bad = False
            for k, ccount in cnt.items():
                if ccount == 1:
                    e = direct[k]
                    if e[0] in nxt:            # non-manifold boundary vertex
                        bad = True
                        break
                    nxt[e[0]] = e[1]
            if bad or not nxt:
                continue
            loops, seen = [], set()
            for start in list(nxt):
                if start in seen:
                    continue
                loop, cur = [], start
                while True:
                    loop.append(cur)
                    seen.add(cur)
                    cur = nxt.get(cur)
                    if cur is None or cur == start:
                        break
                if cur == start and len(loop) >= 3:
                    loops.append(loop)
            if not loops:
                continue
            # MENISCUS clusters: rewrite only when every boundary vertex is
            # pinned (shared with a wetted facet). Such a face -- the plug's
            # under-fill disc is the one that matters, ringed by the barrel
            # and pin contact chains -- is re-triangulated by the boolean
            # into wedge confetti (measured 3.8 deg median min-angle, 73%
            # under 5 deg), and Evolver's maintenance barely improves it
            # because every wedge touches the pinned chain. A free cluster
            # with any FREE boundary (the skirt wall strips, the dome) keeps
            # its deliberate construction resolution.
            if label == "meniscus" and any(
                    v not in _wet_verts for loop in loops for v in loop):
                continue
            # 2D basis with u x v = nref, so the surface winding carries over:
            # outer loops arrive CCW and holes CW, which is what the
            # triangulator expects.
            u = np.cross(nref, [0.0, 0.0, 1.0])
            if np.linalg.norm(u) < 1e-6:
                u = np.cross(nref, [0.0, 1.0, 0.0])
            u /= np.linalg.norm(u)
            w = np.cross(nref, u)
            rings = [[(float(V[i] @ u), float(V[i] @ w)) for i in loop]
                     for loop in loops]
            flat = [i for loop in loops for i in loop]
            try:
                tris = np.asarray(m3d.triangulate(rings), int)
            except Exception:                  # noqa: BLE001 -- keep original
                continue
            # Wetted clusters must strictly SHRINK (that is the point). A
            # pinned-boundary meniscus cluster with no interior vertices --
            # the under-fill annulus -- cannot shrink (its count is fixed by
            # its boundary), but a rewrite still replaces the boolean's wedge
            # confetti with the triangulator's ring pattern, so allow the
            # equal-count rewrite there.
            limit = len(T) + (1 if label == "meniscus" else 0)
            if not len(tris) or len(tris) >= limit:
                continue
            remapped = np.asarray(flat, int)[tris]
            # orient with the patch normal
            aa, bb, cc = V[remapped[:, 0]], V[remapped[:, 1]], V[remapped[:, 2]]
            if np.einsum("ij,j->", np.cross(bb - aa, cc - aa), nref) < 0:
                remapped = remapped[:, ::-1]
            keep[np.asarray(fis, int)] = False
            newF.append(remapped)
            newL.append(np.full(len(remapped), label))
            if verbose:
                print(f"  decimated {label}: {len(T)} -> {len(remapped)} tris")

    if not newF:
        return V, F, lab
    F2 = np.vstack([F[keep]] + newF)
    L2 = np.concatenate([lab[keep]] + newL)
    # conformity is the contract; a decimation that breaks it never ships
    chk = check_body(V, F2, L2)
    if not (chk.get("closed") and chk.get("oriented")):
        return V, F, lab
    return V, F2, L2


# ------------------------------------------------------- conformity / validation

def check_body(V, F, lab):
    """Watertight + consistent orientation + conforming contact chain."""
    F = np.asarray(F, int).reshape(-1, 3)
    he = {}
    for i, t in enumerate(F):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            he.setdefault((a, b), []).append(i)
    bad_dir = [e for e, v in he.items() if len(v) > 1]
    unmatched = [e for e in he if (e[1], e[0]) not in he]
    euler = len(V) - len(he) // 2 + len(F)

    # contact chain: undirected edges whose two incident tris carry different labels
    und = {}
    for i, t in enumerate(F):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            und.setdefault((min(a, b), max(a, b)), []).append(i)
    chain = [(e, v) for e, v in und.items() if len(v) == 2 and lab[v[0]] != lab[v[1]]]
    # Per label-PAIR the contact chain must be a union of closed loops (every
    # vertex valence 2), except at genuine triple points where 3 labels meet.
    pairs = {}
    for e, v in chain:
        k = tuple(sorted((lab[v[0]], lab[v[1]])))
        pairs.setdefault(k, []).append(e)
    # a vertex incident to >=3 distinct labels is a legitimate TRIPLE POINT
    vlab = {}
    for i, t in enumerate(F):
        for v in t:
            vlab.setdefault(v, set()).add(lab[i])
    triple = {v for v, s in vlab.items() if len(s) >= 3}

    tot_odd = 0
    for k, es in sorted(pairs.items()):
        val = {}
        for a, b in es:
            val[a] = val.get(a, 0) + 1
            val[b] = val.get(b, 0) + 1
        odd = [v for v, c in val.items() if c != 2]
        bad = [v for v in odd if v not in triple]
        tot_odd += len(bad)
        print(f"    contact chain {k[0]}|{k[1]}: {len(es)} edges, "
              f"{len(odd)} non-degree-2 vert(s) ({len(odd) - len(bad)} triple points, "
              f"{len(bad)} UNEXPLAINED)")

    print(f"  closed={not unmatched} (unmatched_halfedges={len(unmatched)})  "
          f"orientation_ok={not bad_dir} (dup_dir={len(bad_dir)})  "
          f"euler={euler}  vol={signed_volume(V, F):+.6f}")
    return dict(closed=not unmatched, oriented=not bad_dir, euler=euler,
                chain=len(chain), odd=tot_odd)


def label_edges(F, lab, pred_a, pred_b=None):
    """Undirected edges of a CLOSED body whose two incident triangles straddle a
    label boundary.

    ``pred_a(label) -> bool`` selects a triangle set A. With ``pred_b`` given,
    return the edges with one triangle in A and the other in B (the *contact
    curve* between two provenances). Without it, return the boundary of A: edges
    with exactly one incident triangle in A (the *outline* of the patch).
    """
    F = np.asarray(F, int).reshape(-1, 3)
    und = {}
    for i, t in enumerate(F):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            und.setdefault((min(a, b), max(a, b)), []).append(i)
    out = []
    for e, tris in und.items():
        if len(tris) != 2:                       # non-manifold / boundary: skip
            continue
        i, j = tris
        ai, aj = pred_a(lab[i]), pred_a(lab[j])
        if pred_b is None:
            if ai != aj:
                out.append(e)
        elif (ai and pred_b(lab[j])) or (aj and pred_b(lab[i])):
            out.append(e)
    return out


def inset_vertices(V, F, mask, dist):
    """Vertices pulled ``dist`` toward the interior of the patch ``mask``.

    Used to draw a wet-patch outline just INSIDE its patch, so it does not sit
    exactly on top of the contact curve (they share the same edges) and all the
    overlays stay separately readable.
    """
    V = np.asarray(V, float); F = np.asarray(F, int).reshape(-1, 3)
    Fk = F[np.asarray(mask, bool)]
    acc = np.zeros_like(V); cnt = np.zeros(len(V))
    cen = V[Fk].mean(1)
    for k in range(3):
        np.add.at(acc, Fk[:, k], cen)
        np.add.at(cnt, Fk[:, k], 1.0)
    ok = cnt > 0
    d = np.zeros_like(V)
    d[ok] = acc[ok] / cnt[ok, None] - V[ok]
    n = np.linalg.norm(d, axis=1, keepdims=True)
    out = V.copy()
    out[ok] += (d[ok] / np.maximum(n[ok], 1e-12)) * dist
    return out


def chain_loops(edges):
    """Chain undirected edges into ordered vertex-index paths.

    On a conforming boolean output these come out as CLOSED loops. Triple points
    (valence >= 4) are walked greedily, which splits the pencil there into
    several closed loops -- fine for drawing. Any chain that does not close is
    returned as an out-and-back path, so a bead renderer that auto-closes loops
    does not draw a bogus chord across the gap.

    Returns (paths, n_closed, n_open); index each path into a vertex array to
    draw it.
    """
    adj = {}
    for k, (a, b) in enumerate(edges):
        adj.setdefault(a, []).append((b, k))
        adj.setdefault(b, []).append((a, k))
    used = set()
    loops, n_closed, n_open = [], 0, 0
    for start in list(adj):
        while any(k not in used for _n, k in adj[start]):
            path, cur, prev = [start], start, None
            while True:
                nxt = [(n, k) for n, k in adj[cur] if k not in used]
                if not nxt:
                    break
                # prefer to keep going forward rather than immediately back
                n, k = next(((n, k) for n, k in nxt if n != prev), nxt[0])
                used.add(k)
                prev, cur = cur, n
                if cur == start:
                    break
                path.append(cur)
            if len(path) < 2:
                continue
            if cur == start:
                n_closed += 1
                loops.append(np.asarray(path, int))
            else:
                n_open += 1
                loops.append(np.asarray(path + path[-2:0:-1], int))   # out and back
    return loops, n_closed, n_open


def components(F, mask):
    """Number of connected components among triangles selected by ``mask``."""
    F = np.asarray(F, int).reshape(-1, 3)
    idx = np.flatnonzero(mask)
    if not len(idx):
        return 0
    adj = {}
    for i in idx:
        for a, b in ((F[i, 0], F[i, 1]), (F[i, 1], F[i, 2]), (F[i, 2], F[i, 0])):
            adj.setdefault((min(a, b), max(a, b)), []).append(i)
    nb = {i: set() for i in idx}
    for tris in adj.values():
        for u in tris:
            for v in tris:
                if u != v:
                    nb[u].add(v)
    seen, n = set(), 0
    for i in idx:
        if i in seen:
            continue
        n += 1
        stack = [i]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(nb[u] - seen)
    return n


# ------------------------------------------------------------------ pipeline API

def component_lift(d, clearance=0.005):
    """How far to raise the component so it stops interpenetrating the board.

    Assets are not seated perfectly. On the D2PAK the three pins dip 0.071 mm
    BELOW the pad plane while the thermal tab floats 0.177 mm above it. A part
    buried in the board is not something we want to solder to, so lift the whole
    component until its lowest point clears the pad.

    The lift is RIGID -- one number for the part, because the part is rigid --
    so it CANNOT fix every joint at once. It removes all penetration and widens
    any pre-existing gap by the same amount. `cut_joint` reports both.

    `clearance` is deliberately non-zero: seating the part EXACTLY on pad_z puts
    the lead underside coplanar with the pad top, and coplanar inputs are the one
    degeneracy that reliably breaks these booleans (docs/wet_triangles.md -- it's
    why the blob itself is seated 3 um low). 5 um clears it and is invisible.

    Returns 0.0 when the part already clears the board.
    """
    pad_z = float(d["pad_z"])
    zs = [float(np.asarray(d[nm + "_V"], float)[:, 2].min())
          for nm in ("metal", "body") if len(d[nm + "_F"])]
    return max(0.0, pad_z - min(zs) + clearance) if zs else 0.0


def build_component_solids(d, lift=None, verbose=False):
    """Solidify the component meshes into manifolds ONCE.

    This is the expensive half of `cut_joint`'s setup: `to_manifold` runs
    trimesh repair (weld -> split -> cap -> re-wind) over the whole component
    mesh. It depends only on the component and its lift -- NOT on which joint is
    being cut -- so repairing per joint would pay for N identical repairs on a
    part with N joints.

    Cost scales with component mesh size: joints on small 2-pad passives are
    cheap, while joints on heavy parts can take seconds each if the mesh is
    re-repaired per joint.

    Pass the result to `cut_joint(component_solids=...)`. Safe to share across
    joints: `cut_joint` calls `as_original()` on each solid per joint to get
    fresh provenance ids, and never mutates what it is given.
    """
    lift = component_lift(d) if lift is None else float(lift)
    out = {}
    for nm in ("metal", "body"):
        if not len(d[nm + "_F"]):
            continue
        Vm = np.asarray(d[nm + "_V"], float)
        if lift:
            Vm = Vm + np.array([0.0, 0.0, lift])
        man = to_manifold(Vm, d[nm + "_F"], verbose=verbose)
        if man.is_empty():
            print(f"  !! {nm}: manifold REJECTED the asset mesh (empty)")
            continue
        out[nm] = man
    return out


def cut_joint(d, joint, blob="sphere", *, stencil=0.13, dil=0.25, seat=0.003,
              fill=1.0, match_volume=True, verbose=True, lift=None,
              component_solids=None, crop=0.05, nbisect=12,
              skirt=0.0, skirt_h=0.02, polys=None, pad_polys=None,
              pad_clip_h=0.06, expose=0.0, expose_angle=0.0,
              expose_seed=None, meet_reach=0.0):
    """One joint of a frozen ``dump_joint.py`` npz -> labelled closed solder body.

    The whole §8 per-joint pipeline in one call, so the viewer and the .fe
    emitter drive the identical geometry. Returns a dict with

        V, F      the closed body (shared vertex array -- conforming by
                  construction, which is the point of using manifold3d)
        lab       per-triangle provenance: meniscus / wet_board / wet_metal /
                  wet_body
        rects, pad_z, target_volume, body (the manifold)
        blob      the PRE-CUT blob manifold, and blob_mesh its (V, F) -- the
                  input to the boolean, so a viewer can show input vs output
        lift      how far the component was raised off the board (see
                  `component_lift`); `clearance` its resulting gap at THIS joint

    `lift`: None (default) auto-seats the part off the board, 0.0 leaves the
    asset exactly as authored, a float lifts by that much.
    """
    pad_z = float(d["pad_z"])
    rects = d["apertures"][d["joint_id"] == joint]
    if not len(rects):
        raise SystemExit(f"joint {joint} has no apertures; "
                         f"available: {sorted(set(map(int, d['joint_id'])))}")
    if polys is not None:
        polys = [np.asarray(P, float) for P in polys]
    area = poly_area(polys) if polys else \
        float(sum((r[1] - r[0]) * (r[3] - r[2]) for r in rects))
    vol = fill * area * stencil

    x0 = rects[:, 0].min() - 1.0; x1 = rects[:, 1].max() + 1.0
    y0 = rects[:, 2].min() - 1.0; y1 = rects[:, 3].max() + 1.0
    solids = {"board": board_solid(pad_z, (x0, x1, y0, y1))}
    lift = component_lift(d) if lift is None else float(lift)
    zloc = []                      # component z near THIS joint, to report the gap
    ztop = []                      # ...and its TOP, to report how far solder wets up
    # Per-JOINT: which part of the component sits over THIS joint. Cheap numpy,
    # and measured on the UNLIFTED mesh (the lift is added back below).
    for nm in ("metal", "body"):
        if not len(d[nm + "_F"]):
            continue
        Vm = np.asarray(d[nm + "_V"], float)
        m = ((Vm[:, 0] > x0) & (Vm[:, 0] < x1) & (Vm[:, 1] > y0) & (Vm[:, 1] < y1))
        if m.any():
            zloc.append(float(Vm[m][:, 2].min()))
            ztop.append(float(Vm[m][:, 2].max()))

    # Per-COMPONENT: the expensive solidify. Hoisted out of the joint loop by
    # `build_component_solids`; built here only when a caller did not supply it,
    # so single-joint callers (the viewers) are unchanged.
    if component_solids is None:
        component_solids = build_component_solids(d, lift=lift, verbose=verbose)
    solids.update(component_solids)

    # Gap at this joint AFTER lifting. Negative would mean the part still digs
    # into the board here, which the rigid lift is supposed to have removed.
    clearance = (min(zloc) + lift - pad_z) if zloc else float("nan")
    if verbose and lift:
        print(f"  lifted component +{lift:.4f} mm off the board "
              f"(gap at this joint now {clearance:+.4f} mm)")

    b = make_blob(blob, rects, pad_z, vol, d["metal_V"], stencil=stencil, dil=dil,
                  polys=polys)
    if seat:
        b = b.translate([0.0, 0.0, -seat])
    ctr = (0.5 * (rects[:, 0].min() + rects[:, 1].max()),
           0.5 * (rects[:, 2].min() + rects[:, 3].max()), pad_z)

    # CROP the component to the blob's neighbourhood before subtracting. This is
    # EXACT, not an approximation: outside a box that contains the blob, the
    # component cannot affect `blob - component`, so intersecting it away changes
    # nothing. It turns N big booleans into ONE big boolean plus N small ones,
    # which matters because `match_volume` below re-cuts 24 times per joint
    # against what is otherwise the entire component mesh.
    #
    # The box must contain the blob at its LARGEST candidate scale, not its
    # current one -- the volume search scales up to `hi` about `ctr`.
    _LO, _HI = 0.2, 6.0
    if crop:
        bb = b.bounding_box()
        lo_c, hi_c = np.array(bb[:3], float), np.array(bb[3:], float)
        half = 0.5 * (hi_c - lo_c) * _HI + float(crop)
        mid = 0.5 * (lo_c + hi_c)
        # scaling happens about `ctr`, so the box has to follow that centre too
        mid = np.asarray(ctr, float) + (mid - np.asarray(ctr, float)) * _HI
        box = m3d.Manifold.cube(list(2.0 * half), center=True).translate(list(mid))
        solids = {k: (v ^ box) if k != "board" else v for k, v in solids.items()}
        solids = {k: v for k, v in solids.items() if not v.is_empty()}

    def cut(x):
        for s in solids.values():
            x = x - s
        return x

    if match_volume:
        # Bisection on the blob scale. Each step costs a boolean, so the count
        # is a direct multiplier on solve time -- and 24 steps is far more
        # precision than this needs. It only sets the FOOTPRINT: Evolver is
        # given a volume target afterwards and hits it exactly regardless, so
        # bisection error moves the contact line a hair, not the solder volume.
        # 24 -> 12 leaves ~1e-3 relative on the scale, which is below the 1e-4 mm
        # quantisation the footprint key already rounds to.
        lo, hi = _LO, _HI
        for _ in range(nbisect):
            mid = 0.5 * (lo + hi)
            v = cut(b.translate([-ctr[0], -ctr[1], -ctr[2]]).scale([mid] * 3)
                    .translate(list(ctr))).volume()
            lo, hi = (mid, hi) if v < vol else (lo, mid)
        b = (b.translate([-ctr[0], -ctr[1], -ctr[2]]).scale([0.5 * (lo + hi)] * 3)
             .translate(list(ctr)))

    # SKIRT: a thin flange at the pad plane, out to the pad boundary. The
    # contact line stays PINNED -- this just makes the boolean pin it at the pad
    # edge instead of at the paste aperture, which is what "flood the pad" means
    # when the footprint is prescribed rather than solved for.
    #
    # Added AFTER volume matching on purpose. match_volume scales the blob
    # uniformly, so a skirt built in beforehand would be resized along with it
    # and the footprint would depend on the volume -- exactly the coupling the
    # blob/target split exists to avoid. The skirt therefore contributes a little
    # extra volume to the CUT, which does not matter: Evolver is given its target
    # separately and the skirt's job is only to place the contact line.
    if skirt > 0.0:
        if polys:
            # ROUND joins, so a rounded pad floods as a rounded pad. A Miter or
            # Square join would put corners back on a stadium outline, which is
            # the bbox bug in a different place. Then SIMPLIFY: the offset's
            # round joins add arc segments on top of an already-dense exporter
            # outline, and this flange's rim IS the pad-plane contact line --
            # unsimplified it pinned a 136-vertex outline whose fragmentation
            # dominated the wetted-patch sliver count.
            cs = poly_cross_section(polys).offset(
                skirt, m3d.JoinType.Round, 2.0, 32).simplify(
                _BLOB_SIMPLIFY_EPS)
            flange = cs.extrude(skirt_h + 0.10).translate(
                [0.0, 0.0, pad_z - 0.05])
        else:
            flange = None
            for x0r, x1r, y0r, y1r, _z in rects:
                c = m3d.Manifold.cube(
                    [x1r - x0r + 2.0 * skirt, y1r - y0r + 2.0 * skirt,
                     skirt_h + 0.10], center=True).translate(
                    [0.5 * (x0r + x1r), 0.5 * (y0r + y1r),
                     pad_z + 0.5 * skirt_h - 0.05])
                flange = c if flange is None else flange + c
        if flange is not None:
            b = b + flange

    if meet_reach and meet_reach > 0.0:
        # MEET THE METAL (tombstoned / lifted parts): a posed lead can leave
        # the pad-hugging blob entirely -- the cut then wets no metal and
        # the part hovers over a bare paste dome. When the blob misses the
        # metal but metal exists within `meet_reach`, hull the blob together
        # with that nearby metal band: the hull is the stretched fillet from
        # pad to the posed lead face, the boolean below re-carves the lead
        # out of it, and the pad clip re-trims the pad plane. Self-
        # selecting: a tombstone's LIFTED end has no metal within reach and
        # keeps its paste dome; the ANCHORED end connects. No-op whenever
        # the blob already touches metal (flat parts).
        metal_ref = solids.get("metal")
        if metal_ref is not None and (b ^ metal_ref).volume() < 1e-6:
            bb = b.bounding_box()
            zone = m3d.Manifold.cube(
                (bb[3] - bb[0] + 2.0 * meet_reach,
                 bb[4] - bb[1] + 2.0 * meet_reach,
                 bb[5] - bb[2] + 2.0 * meet_reach), True).translate(
                (0.5 * (bb[0] + bb[3]), 0.5 * (bb[1] + bb[4]),
                 0.5 * (bb[2] + bb[5])))
            band = metal_ref ^ zone
            if band.volume() > 1e-9:
                b = (b + band).hull()
                if verbose:
                    print(f"  meet: blob hulled to metal within "
                          f"{meet_reach:g} mm (the stretched fillet)")
            elif verbose:
                print(f"  meet: NO metal within {meet_reach:g} mm -- "
                      "paste dome (the lifted end)")

    # PAD CLIP: solder wets copper, not mask. When the exposed pad outline is
    # known, everything in the pad-plane band OUTSIDE it is bare mask -- carve
    # it away, so the pinned contact line cannot overrun the pad (spec boards
    # author paste 1:1 with the pad at fine pitch: any overshoot here is what
    # bridges neighbours). Applied AFTER volume matching and the skirt, for
    # the same reason the skirt is: the footprint must not couple into the
    # volume bisection. Only the BAND is clipped -- a real fillet may bulge
    # over the mask above its pad; it just cannot WET past it.
    if pad_polys is None and hasattr(d, "get"):
        pad_polys = d.get("pad_polys")
    if pad_polys is not None and len(pad_polys):
        pad_cs = poly_cross_section(
            [np.asarray(P, float) for P in pad_polys]).simplify(
            _BLOB_SIMPLIFY_EPS)
        bb = b.bounding_box()
        frame_cs = m3d.CrossSection([[
            (bb[0] - 1.0, bb[1] - 1.0), (bb[3] + 1.0, bb[1] - 1.0),
            (bb[3] + 1.0, bb[4] + 1.0), (bb[0] - 1.0, bb[4] + 1.0)]])
        mask_ring = frame_cs - pad_cs
        if not mask_ring.is_empty():
            b = b - mask_ring.extrude(pad_clip_h + 0.05).translate(
                [0.0, 0.0, pad_z - 0.05])

    if expose and expose > 0.0:
        # INTERRUPTED (pad exposure): the foot stops short of the pad edge.
        # The outline the exposed fraction is measured against is the foot
        # the joint would otherwise flood (the blob's own pad-plane
        # footprint); the core disc keeps the joint's middle -- and with it
        # the lead fillet -- connected. Noisy front, exact area fraction
        # (see exposure_keep). Clips the WHOLE blob, so the exposed copper
        # is real geometry, not a rendering trick.
        rng = np.random.default_rng(expose_seed)
        foot = b.project().simplify(0.005)
        fb = foot.bounds()
        core_r = 0.30 * min(fb[2] - fb[0], fb[3] - fb[1])
        fcx, fcy = 0.5 * (fb[0] + fb[2]), 0.5 * (fb[1] + fb[3])
        # LEAD-AWARE core + TOE BIAS. The connectivity core belongs on the
        # wetted-metal footprint, not the blob centroid -- a gullwing's lead
        # sits at one end of a long pad, and a centroid core would let the
        # bite eat the lead fillet. And the bite azimuth is biased toward
        # the TOE (from the lead centroid through the foot centroid, the
        # pad's free end): real interrupted joints photograph as toe
        # exposure, because that is the far end of the wetting path. The
        # strategy's randomized expose_angle becomes +/-60 deg of jitter
        # around the toe; with no lead over the pad (pad-only joint) both
        # fall back to the centroid core and the raw azimuth.
        ccx, ccy, ang = fcx, fcy, float(expose_angle)
        metal_ref = solids.get("metal")
        if metal_ref is not None:
            eps = 0.01
            gap0 = clearance if clearance == clearance else 0.0
            zs = pad_z + max(float(gap0), 0.0) + eps
            mrx0 = float(rects[:, 0].min()) - (skirt + 0.05)
            mrx1 = float(rects[:, 1].max()) + (skirt + 0.05)
            mry0 = float(rects[:, 2].min()) - (skirt + 0.05)
            mry1 = float(rects[:, 3].max()) + (skirt + 0.05)
            sec = metal_ref ^ m3d.Manifold.cube(
                (mrx1 - mrx0, mry1 - mry0, 2 * eps), True).translate(
                (0.5 * (mrx0 + mrx1), 0.5 * (mry0 + mry1), zs))
            if sec.volume() > 1e-9:
                mv = np.asarray(sec.to_mesh().vert_properties, float)[:, :3]
                mcx, mcy = float(mv[:, 0].mean()), float(mv[:, 1].mean())
                tx, ty = fcx - mcx, fcy - mcy
                if np.hypot(tx, ty) > 0.02:
                    toe = float(np.arctan2(ty, tx))
                    tnorm = (float(expose_angle) % (2.0 * np.pi)) / (
                        2.0 * np.pi)
                    ang = toe + (tnorm - 0.5) * (2.0 * np.pi / 3.0)
                ccx, ccy = mcx, mcy
        keep, achieved = exposure_keep(
            foot, ccx, ccy, core_r, float(expose), ang, rng)
        bb = b.bounding_box()
        b = b ^ keep.extrude((bb[5] - bb[2]) + 0.4).translate(
            [0.0, 0.0, bb[2] - 0.2])
        parts = b.decompose()
        if len(parts) > 1:
            # Same island pruning as the THT path: a stranded arc is a lie
            # to the genus check and floating solder in the render.
            parts = sorted(parts, key=lambda m: m.volume(), reverse=True)
            b = parts[0]
        if verbose:
            print(f"  interrupted: pad exposure {achieved:.0%} "
                  f"(requested {float(expose):.0%})")

    b = b.as_original()
    solids = {k: v.as_original() for k, v in solids.items()}
    names = {b.original_id(): "meniscus"}
    for nm, s in solids.items():
        names[s.original_id()] = "wet_" + nm

    body = cut(b)
    V, F = from_manifold(body)
    lab = tri_labels(body, names).astype(str)
    V, F, lab = decimate_planar_patches(V, F, lab, verbose=verbose)

    # How far UP the part does the solder wet? The wetted patch is frozen, so
    # this is not a detail of the initial guess -- it is the footprint, and it
    # is imposed at every volume the sweep asks for afterwards.
    #
    # `match_volume` has a feedback loop that makes this easy to get wrong: it
    # scales the blob until the CUT volume matches the paste, and a bigger blob
    # loses more to the cut, so it has to grow further still. A sphere seated
    # near the pad ends up engulfing the whole termination -- 96% of its height
    # on an 0805, versus 17-35% for every other blob shape.
    Ft = np.asarray(F, int).reshape(-1, 3)
    isw = np.asarray(lab) != "meniscus"
    comp_top = (max(ztop) + lift - pad_z) if ztop else float("nan")
    wet_z = (float(V[Ft[isw]].reshape(-1, 3)[:, 2].max()) - pad_z) if isw.any() else 0.0
    wet_frac = wet_z / comp_top if comp_top and comp_top == comp_top else float("nan")
    if verbose:
        print(f"  solder wets to {wet_z:.4f} mm above the pad "
              f"({wet_frac:.0%} of the {comp_top:.4f} mm termination)")
    if wet_frac == wet_frac and wet_frac > 0.9:
        print(f"  !! the '{blob}' blob wraps OVER the top of the part "
              f"({wet_frac:.0%} of its height wetted). That footprint is frozen, "
              f"so every volume you solve afterwards is forced to keep wetting "
              f"the top -- and the topmost wet vertex is pinned, which shows up "
              f"as a stuck apex that never moves. Try --blob ellipsoid or "
              f"aperture_offset.")

    # REFERENCE VOLUME for the `fullness` parameter, from THIS joint's own
    # geometry (the SMT counterpart of tht.py's plug + fillet cone). An SMT
    # termination is a block FACE, not a pin ring, so the reference is a
    # WEDGE, not an annular frustum: the wetted pad area OUTSIDE the
    # termination footprint lofted to the wetted band's top edge (a line,
    # so the loft tapers to ~zero area: V = h * A_out / 3), plus the thin
    # under-lead slab (standoff clearance x the pad area the termination
    # covers). fullness 1.0 IS this straight fillet; < 1 concave, > 1
    # convex -- the same number reads the same on every footprint, unlike
    # a fraction of the printed paste, whose fillet steepness depends on
    # pad and termination geometry.
    fillet_cone = 0.0
    metal_s = solids.get("metal")
    if metal_s is not None and wet_z > 0.02 and isw.any():
        # Wetted pad area: the planar wet_board patch, exact from the cut.
        wb_m = np.asarray(lab) == "wet_board"
        if wb_m.any():
            Tb = Ft[wb_m]
            aa, bb2, cc = V[Tb[:, 0]], V[Tb[:, 1]], V[Tb[:, 2]]
            nrm = np.cross(bb2 - aa, cc - aa)
            a0 = 0.5 * float(np.abs(nrm[:, 2]).sum())
        else:
            a0 = 0.0

        # RAW paste extent -- x0/x1 above already carry a 1 mm working
        # margin, which must not leak in here: the under-lead area is by
        # definition metal OVER THE PAD, so the slab is the aperture extent
        # plus only the skirt's flood.
        rx0 = float(rects[:, 0].min()) - (skirt + 0.05)
        rx1 = float(rects[:, 1].max()) + (skirt + 0.05)
        ry0 = float(rects[:, 2].min()) - (skirt + 0.05)
        ry1 = float(rects[:, 3].max()) + (skirt + 0.05)

        def _metal_area(z):
            eps = 0.01
            slab = m3d.Manifold.cube((rx1 - rx0, ry1 - ry0, 2 * eps),
                                     True).translate(
                (0.5 * (rx0 + rx1), 0.5 * (ry0 + ry1), z))
            return max(0.0, float((metal_s ^ slab).volume()) / (2 * eps))

        if a0 > 0.0:
            gap = clearance if clearance == clearance else 0.0
            gap = min(max(float(gap), 0.0), wet_z)
            # Termination footprint just above its underside.
            a_met = _metal_area(pad_z + gap + 0.01)
            a_under = min(a_met, a0)
            a_out = max(a0 - a_under, 0.0)
            fillet_cone = float(wet_z * a_out / 3.0 + gap * a_under)

    return dict(V=V, F=F, lab=lab, rects=rects, pad_z=pad_z, target_volume=vol,
                body=body, solids=solids, lift=lift, clearance=clearance,
                wet_z=wet_z, comp_top=comp_top, wet_frac=wet_frac,
                fillet_cone_volume=float(fillet_cone),
                blob=b, blob_mesh=from_manifold(b))


# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="D2PAK")
    ap.add_argument("--joint", type=int, default=1)
    ap.add_argument("--blob", default="aperture_offset")
    ap.add_argument("--fill", type=float, default=1.0, help="volume multiplier")
    ap.add_argument("--dil", type=float, default=0.25)
    ap.add_argument("--stencil", type=float, default=0.13)
    ap.add_argument("--obj", default="", help="write labelled OBJ here")
    ap.add_argument("--no-match-volume", dest="match_volume", action="store_false")
    ap.add_argument("--union-first", action="store_true")
    ap.add_argument("--board-dz", type=float, default=0.0,
                    help="raise the board top plane by this much, to swallow\n                          package faces that sit exactly coplanar on it")
    ap.add_argument("--seat", type=float, default=0.003,
                    help="sink the blob this far below pad_z, so the board cut is "
                         "transversal rather than exactly coplanar")
    a = ap.parse_args()

    d = np.load(os.path.join(OUT, f"{a.family}.npz"))
    pad_z = float(d["pad_z"])
    aps, jid = d["apertures"], d["joint_id"]
    rects = aps[jid == a.joint]
    area = float(sum((r[1] - r[0]) * (r[3] - r[2]) for r in rects))
    vol = a.fill * area * a.stencil

    print(f"{a.family} joint {a.joint}: {len(rects)} apertures  area {area:.3f} mm^2  "
          f"paste volume {vol:.5f} mm^3  pad_z {pad_z:.4f}")

    print("asset mesh health (raw vs welded):")
    for nm in ("metal", "body"):
        VV, FF = d[nm + "_V"], d[nm + "_F"]
        if not len(FF):
            continue
        mesh_health(VV, FF, nm + " raw")
        mesh_health(*weld(VV, FF), nm + " welded")

    # --- localise the component geometry to this joint (a margin box around it)
    x0 = rects[:, 0].min() - 1.0; x1 = rects[:, 1].max() + 1.0
    y0 = rects[:, 2].min() - 1.0; y1 = rects[:, 3].max() + 1.0

    board = board_solid(pad_z + a.board_dz, (x0, x1, y0, y1))

    print("solidify:")
    solids = {"board": board}
    for nm, (VV, FF) in (("metal", (d["metal_V"], d["metal_F"])),
                         ("body", (d["body_V"], d["body_F"]))):
        if not len(FF):
            continue
        try:
            man = to_manifold(VV, FF)
        except Exception as e:
            print(f"  !! {nm}: manifold construction raised {e}")
            continue
        print(f"  {nm} -> manifold status={man.status()} vol={man.volume():+.5f} "
              f"tris={man.num_tri()} genus={man.genus()} empty={man.is_empty()}")
        if man.is_empty():
            print(f"  !! {nm}: manifold REJECTED the asset mesh (empty)")
            continue
        solids[nm] = man

    ctr = (0.5 * (rects[:, 0].min() + rects[:, 1].max()),
           0.5 * (rects[:, 2].min() + rects[:, 3].max()), pad_z)

    if a.union_first:
        # Union the obstacles once, then a single difference. Resolves
        # solid-vs-solid coplanarity (a seated package bottom lying exactly on the
        # board plane) inside the union instead of leaking into the solder body.
        # original_id provenance survives the union, so labels are preserved.
        obstacle = m3d.Manifold.batch_boolean(list(solids.values()), m3d.OpType.Add)

        def cut(b):
            return b - obstacle
    else:
        def cut(b):
            for s in solids.values():
                b = b - s
            return b

    blob = make_blob(a.blob, rects, pad_z, vol, d["metal_V"],
                     stencil=a.stencil, dil=a.dil)
    if a.seat:
        blob = blob.translate([0.0, 0.0, -a.seat])
    print(f"blob '{a.blob}':vol={blob.volume():.5f} tris={blob.num_tri()} "
          f"genus={blob.genus()} status={blob.status()}")
    # Bisect a uniform scale about the joint centre so the CUT volume (what the
    # solder actually occupies) equals the paste volume, not the raw blob volume.
    if a.match_volume:
        lo, hi = 0.2, 6.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            v = cut(blob.translate([-ctr[0], -ctr[1], -ctr[2]]).scale([mid] * 3)
                    .translate(list(ctr))).volume()
            if v < vol:
                lo = mid
            else:
                hi = mid
        blob = (blob.translate([-ctr[0], -ctr[1], -ctr[2]]).scale([0.5 * (lo + hi)] * 3)
                .translate(list(ctr)))
        print(f"  volume-matched: scale={0.5*(lo+hi):.4f} raw_vol={blob.volume():.5f}")

    # as_original() stamps a fresh provenance id on a composite manifold, so the
    # boolean output can tell blob-derived triangles from solid-derived ones.
    blob = blob.as_original()
    solids = {k: v.as_original() for k, v in solids.items()}
    names = {blob.original_id(): "meniscus"}
    for nm, s in solids.items():
        names[s.original_id()] = "wet_" + nm

    body = cut(blob)
    print(f"result: vol={body.volume():.5f} (target {vol:.5f}) tris={body.num_tri()} "
          f"genus={body.genus()} status={body.status()} "
          f"n_components={len(body.decompose())}")

    V, F = from_manifold(body)
    lab = tri_labels(body, names)
    uniq, cnt = np.unique(lab.astype(str), return_counts=True)
    print("  triangle provenance: " + ", ".join(f"{u}={c}" for u, c in zip(uniq, cnt)))
    labs = lab.astype(str)
    for u in uniq:
        print(f"    patch '{u}': {components(F, labs == u)} connected component(s)")
    check_body(V, F, labs)

    if a.obj:
        with open(a.obj, "w") as f:
            for v in V:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for u in uniq:
                f.write(f"g {u}\n")
                for t in F[labs == u]:
                    f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
        print(f"  wrote {a.obj}")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------- joint clustering
def cluster_rects(rects, gap=1.0):
    """Union-find clustering of aperture rects by bbox proximity (< gap apart)."""
    n = len(rects); par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a

    for i in range(n):
        ax0, ax1, ay0, ay1 = rects[i][:4]
        for j in range(i + 1, n):
            bx0, bx1, by0, by1 = rects[j][:4]
            if (ax0 - gap <= bx1 and bx0 - gap <= ax1) and (ay0 - gap <= by1 and by0 - gap <= ay1):
                par[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
