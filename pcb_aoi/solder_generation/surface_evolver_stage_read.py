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

"""Read a live USD stage into transport-safe solder inputs.

This is the ONLY place in the cdf integration allowed to touch a stage, and it
is read-only. ``SolderSolverBase.prepare`` may inspect the live stage; ``generate``
must run without one, so everything the solver needs has to be distilled here
into plain Python (see ``pcb_aoi/solder_generation/solver_base.py``).

Everything is expressed in the COMPONENT-LOCAL frame, not world. That is not
cosmetic: two instances of the same prototype only share a solved joint if their
inputs are identical, and in world coordinates they never are. Local frame plus
the instance's own transform is what makes the reuse in ``footprint_key`` sound.

Contrast with ``ingest.board_component.load_board_component``, which opens a
board file itself and hunts paste out of a separate pastemask layer via a
manifest. Here the workflow hands us the paste prim paths directly on the work
item, so no discovery is needed -- and the stage may already carry applied
structural defects, which a re-open would lose.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _fan_triangulate(counts, indices):
    """Polygon soup -> triangles. USD meshes are n-gons; Evolver wants triangles."""
    tris, off = [], 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((indices[off], indices[off + k], indices[off + k + 1]))
        off += c
    return np.asarray(tris, np.int64).reshape(-1, 3)


def component_local_transform(stage, instance_path, time=None):
    """Row-major 4x4 local->world for a component instance, and its inverse.

    Returned row-major to match the cdf contract, which specifies row-major
    16-tuples throughout. Gf matrices are row-vector convention (point * matrix),
    so ``GetInverse()`` composes the way we need without a transpose.
    """
    from pxr import Usd, UsdGeom

    time = Usd.TimeCode.Default() if time is None else time
    prim = stage.GetPrimAtPath(instance_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"no prim at {instance_path}")
    xf = UsdGeom.Xformable(prim)
    l2w = xf.ComputeLocalToWorldTransform(time)
    return l2w, l2w.GetInverse()


def _mesh_points_local(stage, prim_path, world_to_local, time=None):
    """Triangulated points of one mesh prim, expressed in the component frame."""
    from pxr import Usd, UsdGeom

    time = Usd.TimeCode.Default() if time is None else time
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        return None
    pts = mesh.GetPointsAttr().Get(time)
    counts = mesh.GetFaceVertexCountsAttr().Get(time)
    idx = mesh.GetFaceVertexIndicesAttr().Get(time)
    if not pts or not counts or not idx:
        return None
    l2w = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
    to_comp = l2w * world_to_local          # prim-local -> world -> component-local
    P = np.array([[p[0], p[1], p[2]] for p in pts], float)
    ones = np.ones((len(P), 1))
    M = np.array(to_comp, float).reshape(4, 4)
    P = (np.hstack([P, ones]) @ M)[:, :3]   # row-vector convention
    return P, _fan_triangulate(list(counts), list(idx))


def _iter_mesh_prims(stage, root_path):
    """Every Mesh at or under a prim. Component geometry is usually a subtree, and
    on an instanceable prim the children live behind a prototype, so plain
    ``GetChildren`` on the instance would come back empty -- traverse instance
    proxies explicitly."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    if UsdGeom.Mesh(root):
        yield root
    pred = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for prim in Usd.PrimRange(root, pred):
        if prim != root and UsdGeom.Mesh(prim):
            yield prim


def paste_points_local(stage, paste_prim_paths, world_to_local, time=None):
    """Points of each paste deposit, in component-local frame (one array each).

    The joint footprint is defined by the PASTE, not by the metal -- the stencil
    apertures are what actually deposits solder, and the contract keeps
    ``board_paste`` addressable per deposit precisely so this is possible. Note
    the spec is explicit that paste-mask openings differ from solder-mask
    openings by design, so deriving one from the other would be wrong.

    Returns raw points rather than rects because the board-normal axis is not
    known until they have been measured -- see ``solver_frame``.
    """
    from pxr import Usd

    time = Usd.TimeCode.Default() if time is None else time
    out = []
    for path in paste_prim_paths:
        got = None
        for prim in _iter_mesh_prims(stage, path):
            m = _mesh_points_local(stage, prim.GetPath().pathString,
                                   world_to_local, time)
            if m is None:
                continue
            got = m[0] if got is None else np.vstack([got, m[0]])
        if got is not None and len(got):
            out.append(got)
    return out


def pad_meshes_local(stage, pad_prim_paths, world_to_local, time=None):
    """(points, triangles) of each board_pad mesh, in COMPONENT-LOCAL frame.

    The exposed pad geometry for the pad-plane clip: solder wets copper, not
    mask, so the blob's contact footprint must stay inside the pad outline.
    Points only would not do -- the outline comes from the exposed FACE, which
    needs the triangles (see ``pad_outline_polys``).
    """
    out = []
    for path in pad_prim_paths:
        for prim in _iter_mesh_prims(stage, path):
            m = _mesh_points_local(stage, prim.GetPath().pathString,
                                   world_to_local, time)
            if m is not None and len(m[0]) and len(m[1]):
                out.append(m)
    return out


def pad_outline_polys(pad_meshes_solver):
    """Exposed-face outlines of pad meshes ALREADY in the solver frame.

    Per pad: the TOP-face triangles (the solver frame guarantees +z is away
    from the board, so the exposed copper is the max-z face) unioned into
    contours -- the same face-union trick the THT annulus uses, because pad
    sheets are not watertight and slicing a Manifold silently yields nothing.

    Returns:
        2D outline rings (each (n, 2) float array), possibly several per pad.

    """
    import manifold3d as m3d

    polys = []
    for P, T in pad_meshes_solver:
        P = np.asarray(P, float)
        T = np.asarray(T, int).reshape(-1, 3)
        if not len(T):
            continue
        zf = float(P[:, 2].max())
        face = T[P[T][:, :, 2].min(1) > zf - 1e-4]
        parts = []
        for tri in face:
            q = [(float(P[i, 0]), float(P[i, 1])) for i in tri]
            a2 = ((q[1][0] - q[0][0]) * (q[2][1] - q[0][1])
                  - (q[1][1] - q[0][1]) * (q[2][0] - q[0][0]))
            parts.append(m3d.CrossSection([q[::-1] if a2 < 0 else q]))
        if not parts:
            continue
        cs = m3d.CrossSection.batch_boolean(parts, m3d.OpType.Add)
        if cs.is_empty():
            continue
        polys.extend(np.asarray(x, float) for x in cs.to_polygons())
    return polys


def solver_frame(paste_pts, comp_pts):
    """Rotation taking component-local coords into the frame the solver assumes.

    The whole geometry pipeline assumes the board is the z=0 plane with the part
    above it. That is NOT a property of component-local space: which local axis
    is the board normal depends on how the part was authored and placed. On the
    reference board, an 0805's component-local paste extent is (2.95, 0.076, 1.45)
    -- the normal is local **Y**, and assuming Z produced a footprint 0.076 mm
    wide, i.e. the paste's THICKNESS mistaken for its width.

    Derive it instead: a paste deposit is a thin slab, so its thinnest local axis
    IS the board normal. Orient that axis to +z, pointing away from the board
    (the component body is on the far side from the pad plane), and complete a
    right-handed frame from the remaining two axes so the rotation has det +1 and
    does not mirror the geometry.

    Returns a 4x4 row-major rotation for row-vector convention (p_solver =
    p_local @ R), plus its inverse.
    """
    paste = np.vstack(paste_pts)
    ext = paste.max(0) - paste.min(0)
    n = int(np.argmin(ext))                      # thin axis == board normal

    e = np.eye(3)
    up = e[n].copy()
    # A POSE-DEFECTED part's paste is board-fixed, so in the part's local frame
    # the slab is tilted -- its thin direction lies BETWEEN coordinate axes and
    # the axis snap above leaves the residual tilt in the solve (a ~80 deg
    # tombstone put the frame 10.5 deg off; the solder visibly leaned). Measure
    # the true thin direction instead, but keep the snapped axis whenever they
    # agree: nominal deposits are axis-aligned by construction, and the exact
    # axis keeps the rotation -- and with it the footprint dedup key -- bitwise
    # stable across instances of the same footprint.
    centered = paste - paste.mean(0)
    normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
    norm = float(np.linalg.norm(normal))
    if norm > 1e-12:
        normal = normal / norm
        cos_to_axis = float(np.dot(normal, up))
        if abs(cos_to_axis) < np.cos(np.radians(1.0)):
            up = normal if cos_to_axis >= 0.0 else -normal
    # Point it away from the board: the component body sits on the far side of
    # the pad plane. Without this a bottom-side part would solve upside down.
    # Decide by EXTENT, not by vertex mean: the body reaches farther on the up
    # side (a QFN reaches +0.75 mm above its pads and ~0 below), but its
    # vertex MEAN can sit below the paste mean when the mesh is dense at the
    # leads/thermal pad and sparse above -- exactly U1 (QFN-32) on the
    # environment-sensor board, whose frame flipped and solved the footprint
    # upside down AND y-mirrored. A mirrored solve is invisible on symmetric
    # two-pad passives, which is why this survived every 0402/0805 test.
    if comp_pts is not None and len(comp_pts):
        proj = (np.asarray(comp_pts, float) - paste.mean(0)) @ up
        if -float(proj.min()) > float(proj.max()):
            up = -up
    a, b = [i for i in range(3) if i != n]
    x = e[a]
    y = np.cross(up, x)                          # right-handed by construction
    if np.linalg.norm(y) < 1e-12:                # up parallel to e[a] (a 90 deg
        y = np.cross(up, e[b])                   # pose); e[b] cannot also be
    y = y / np.linalg.norm(y)                    # parallel, so this is nonzero
    x = np.cross(y, up)

    R3 = np.stack([x, y, up], axis=1)            # columns: local -> solver axes
    R = np.eye(4)
    R[:3, :3] = R3
    Rinv = np.eye(4)
    Rinv[:3, :3] = R3.T                          # rotation: inverse == transpose
    return R, Rinv


def rects_from_points(paste_pts_solver):
    """(x0,x1,y0,y1,z_pad) per deposit, once already in the solver frame.

    z is the deposit's UNDERSIDE, which is the copper it was printed onto -- not
    its top. The joint plane is the pad, and taking the top instead floated
    every joint one full stencil thickness (0.076 mm on the reference board)
    above the metal it is supposed to be wetting.

    Bounding boxes. Fine for extent and clustering; NOT fine for shape or area,
    since squaring off a rounded aperture inflates it -- use `polys_from_points`
    for anything that cares.
    """
    out = []
    for P in paste_pts_solver:
        out.append((float(P[:, 0].min()), float(P[:, 0].max()),
                    float(P[:, 1].min()), float(P[:, 1].max()),
                    float(P[:, 2].min())))
    return out


def polys_from_points(paste_pts_solver, quant=1e-5):
    """The real 2D outline of each deposit, in the solver frame.

    A paste deposit is a flat extruded slab, so its outline is just the distinct
    XY points of its mesh. 45% of the reference board's deposits are rounded
    (30-, 34- and 42-vertex outlines); a bounding box turns a stadium pad into a
    rectangle and a BGA's circular pad into a square, which is wrong twice over
    -- the flooded shape AND the printed volume, the latter by 4/pi = 21% on a
    circle.

    Returns the CONVEX HULL of those points, so the winding is known-good for
    manifold3d without depending on the mesh's face order. Exact for the pad
    shapes that actually occur (rect, stadium, circle, chamfered rect); a
    genuinely concave aperture -- a C or U opening -- would be filled in, and
    would need the real boundary walked instead.
    """
    import manifold3d as m3d

    out = []
    for P in paste_pts_solver:
        Q = np.unique(np.round(np.asarray(P, float)[:, :2] / quant), axis=0) * quant
        if len(Q) < 3:
            out.append(None)
            continue
        try:
            cs = m3d.CrossSection.hull_points([(float(a), float(b)) for a, b in Q])
            rings = cs.to_polygons()
            out.append(np.asarray(rings[0], float) if rings else None)
        except Exception:                       # noqa: BLE001 -- degenerate hull
            out.append(None)
    return out


def paste_thickness(paste_pts_solver):
    """Area-weighted mean stencil thickness (mm), MEASURED not assumed.

    Paste deposits are authored as real 3D geometry, so the stencil thickness is
    sitting in the z extent and there is no reason to guess it. It was previously
    hardcoded at 0.13 mm; every deposit on the reference board measures 0.076, so
    the assumption inflated the printed volume by 71% on every joint.

    Area-weighted because a joint may cluster deposits of different sizes and it
    is the total volume that matters, not the mean of thicknesses.
    """
    num = den = 0.0
    for P in paste_pts_solver:
        a = float((P[:, 0].max() - P[:, 0].min())
                  * (P[:, 1].max() - P[:, 1].min()))
        t = float(P[:, 2].max() - P[:, 2].min())
        num += a * t
        den += a
    return (num / den) if den > 0 else 0.0


def apply4(P, M):
    """Row-vector transform of an (n,3) point array by a 4x4."""
    P = np.asarray(P, float)
    return (np.hstack([P, np.ones((len(P), 1))]) @ np.asarray(M, float))[:, :3]


def component_meshes_local(stage, instance_path, world_to_local,
                           name_filters=None, time=None):
    """Concatenated component geometry in the component frame.

    Takes ALL meshes under the instance by default rather than only those tagged
    as solder contacts. The tag is not reliable -- on a D2PAK the gull leads are
    authored under the package body -- and for a keep-out field we want every
    solid the solder could run into anyway.
    """
    Vs, Fs, off = [], [], 0
    for prim in _iter_mesh_prims(stage, instance_path):
        name = prim.GetName().lower()
        if name_filters and not any(f.lower() in name for f in name_filters):
            continue
        m = _mesh_points_local(stage, prim.GetPath().pathString,
                               world_to_local, time)
        if m is None or not len(m[1]):
            continue
        Vs.append(m[0])
        Fs.append(m[1] + off)
        off += len(m[0])
    if not Vs:
        return None, None
    return np.vstack(Vs), np.vstack(Fs)


def component_body_points_local(stage, instance_path, world_to_local,
                                time=None):
    """The component's BODY vertices only, in the component frame.

    Reads the spec's authoritative ``pcba:mesh:role`` tags (pkg_body) -- the
    body/lead split the occlusion silhouette wants. Returns None when the
    asset carries no role tags (a pre-schema export), so callers fall back to
    the whole-part silhouette unchanged.
    """
    Vs = []
    saw_roles = False
    for prim in _iter_mesh_prims(stage, instance_path):
        from pcb_aoi.board_metadata import MESH_ROLE_ATTR

        attr = prim.GetAttribute(MESH_ROLE_ATTR)
        role = str(attr.Get()) if attr and attr.HasValue() else ""
        if role:
            saw_roles = True
        if role != "pkg_body":
            continue
        m = _mesh_points_local(stage, prim.GetPath().pathString,
                               world_to_local, time)
        if m is not None and len(m[0]):
            Vs.append(m[0])
    if not saw_roles or not Vs:
        return None
    return np.vstack(Vs)


def occlusion_silhouette(stage, instance_path, world_to_local,
                         frame=None, pose=None, time=None):
    """THE silhouette ``occluded_deposits`` tests against, for any caller.

    One implementation on purpose: the lab and the production reader briefly
    diverged here (the lab gained the body-only silhouette first, and a JST
    connector kept losing its pin joints in board bakes) -- this helper is
    the single route both now take. Returns the BODY vertices where the
    asset's ``pcba:mesh:role`` tags give the body/lead split, transformed by
    ``pose`` (a component-local defect transform, if any) then ``frame`` (the
    solver frame) -- the same trip the component geometry makes. ``None`` on
    pre-schema assets, which keeps the whole-part-bbox behaviour.
    """
    body = component_body_points_local(stage, instance_path, world_to_local,
                                       time=time)
    if body is None:
        return None
    if pose is not None:
        body = apply4(body, pose)
    if frame is not None:
        body = apply4(body, frame)
    return body


def footprint_key(rects, pad_z, extra=()):
    """Stable hash of a joint footprint, for deciding what may share a solve.

    This is the load-bearing part of prototype reuse. ``prototype_id`` alone is
    NOT a safe key: component geometry is shared through the USD prototype, but
    paste deposits are separate board prims per instance, and the USD contract
    states outright that paste-mask openings are "deliberately different sizes,
    driven by DFM/DFA rules". Two instances of the same part with different
    stencil apertures must not share a solved joint -- and if they wrongly did,
    the result would look entirely plausible.

    Quantised to 1e-4 mm (0.1 um) so float noise in the transform chain does not
    split otherwise-identical footprints into separate solves.
    """
    q = 1e-4
    parts = [tuple(round(v / q) for v in r) for r in sorted(rects)]
    blob = repr((parts, round(float(pad_z) / q), tuple(extra))).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def occluded_deposits(paste_pts_solver, comp_pts_solver, margin=0.0,
                      silhouette_pts=None):
    """Which paste deposits lie under the component's own silhouette.

    A deposit fully inside the part's XY footprint -- a thermal pad, a QFN
    centre pad, anything beneath the body -- is invisible to a top-down AOI
    camera. Solder there contributes nothing to the image, and worse: with the
    package sitting at or below the pad plane the blob has nowhere to go, so it
    squeezes sideways and rears up into a shell that DOES protrude past the part.
    Measured on a TO460 thermal tab, ~33% of the solved vertices ended up outside
    the component footprint, ~1.5 mm clear of it -- visible solder that should
    not exist.

    Silhouette is the XY bounding box of ``silhouette_pts`` when given, else of
    the whole part. Pass the BODY vertices (component_body_points_local) on a
    schema-carrying asset: the whole-part bbox includes protruding leads and
    tabs, so a connector whose pins tuck under the housing rear (a JST SH:
    pin paste at the bbox edge, under the leads but past the BODY) had every
    pin joint wrongly dropped. The body/lead split is the spec's own
    ``pcba:mesh:role`` tags; pre-schema assets don't carry them and keep the
    crude whole-part bbox.

    NOTE this is a TOP-DOWN assumption. For X-ray or side views these joints
    matter, so the caller decides -- it is a filter, not a rule.

    Returns a list of bools, one per deposit.
    """
    if silhouette_pts is not None and len(silhouette_pts):
        comp_pts_solver = silhouette_pts
    if comp_pts_solver is None or not len(comp_pts_solver):
        return [False] * len(paste_pts_solver)
    C = np.asarray(comp_pts_solver, float)
    x0, y0 = C[:, 0].min() + margin, C[:, 1].min() + margin
    x1, y1 = C[:, 0].max() - margin, C[:, 1].max() - margin
    out = []
    for P in paste_pts_solver:
        out.append(bool(P[:, 0].min() >= x0 and P[:, 0].max() <= x1
                        and P[:, 1].min() >= y0 and P[:, 1].max() <= y1))
    return out
