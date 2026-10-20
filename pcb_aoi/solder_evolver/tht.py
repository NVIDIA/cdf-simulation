#!/usr/bin/env python
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

"""SCRATCH -- boolean cut for a THROUGH-HOLE joint.

Same shape as ``wet_boolean.cut_joint``: build a seed blob, subtract the solids,
and label the surviving triangles by which solid they came from. What differs is
the board and the topology.

THE BOARD HAS A HOLE IN IT. The board solid is the substrate slab MINUS the
barrel cylinder, so the hole is a real void the solder can enter -- the contract
is explicit that holes are voids, not solid cylinders. Everything the solder
touches on the board side is one label: the top annulus and the barrel wall are
both board copper, so both come back ``wet_board`` and both get pinned. That is
physically right -- the plating is continuous with the pad.

THE JOINT IS A RING, NOT A BLOB. Subtracting the pin from a seed that spans the
hole leaves a body that WRAPS the pin, so the free surface is an annulus and the
body has genus 1. That is the whole reason this case is interesting, and it is
also why the seed has to span the hole in the first place: Evolver cannot open a
hole in a surface, so any topology the joint needs must exist in the cut. A seed
that merely sits on top of the pad would minimise to a dome with the pin poking
through it, which is a different joint.

TO92's pin is SQUARE (0.5 x 0.5 mm) in a round 0.850 hole, so the annular gap is
not constant -- it pinches at the flats and opens at the diagonals. Nothing here
assumes axisymmetry.

    .venv/bin/python -m pcb_aoi.solder_evolver.tht \\
        --family TO92 --joint 0

Volume has NO datum for THT (see dump_tht.py): there is no paste to measure, and
nothing on the board says how full a barrel is. `barrel_fill` is therefore a
stated parameter, not a measurement. IPC-A-610 class 2 wants >= 75% barrel fill,
which is the number to argue with.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


import manifold3d as m3d                                    # noqa: E402

from . import boolean as wb

from .boolean import OUT  # shared dev-dump location


def load_tht(family, tag="_tht"):
    return np.load(os.path.join(OUT, f"{family}{tag}.npz"))


def dict_from_joints(joints, metal_V, metal_F):
    """Pack tht_spec.ThtJoint objects into the mapping cut_tht_joint expects.

    Same trick as runner._as_joint_dict on the SMD side: shape live data like
    the frozen npz so there is ONE implementation of the geometry, rather than a
    board path and a viewer path that drift.
    """
    pad_xy, pad_ptr, pad_ring, barrels, pad_zs = [], [0], [], [], []
    for k, jt in enumerate(joints):
        cx, cy = jt.hole.center_mm
        barrels.append((cx, cy, 0.5 * jt.hole.dims_mm[0], k))
        pd_z = jt.pad_top or jt.pad_bottom
        pad_zs.append(float(pd_z.z) if pd_z is not None
                      else float(jt.hole.z1))
        # pad_top OR pad_bottom -- a bottom-soldered joint has only the latter,
        # and taking pad_top alone silently packed NO contours, which collapsed
        # the seed to an empty body for every top-inserted part.
        pd = jt.pad_top or jt.pad_bottom
        for ring in (pd.contours if pd else ()):
            pad_xy.extend([(float(x), float(y)) for x, y in ring])
            pad_ptr.append(len(pad_xy))
            pad_ring.append(k)
    return {
        "pad_z": np.asarray(max(j.hole.z1 for j in joints), float),
        # Per-joint FOOT plane: the pad's exposed-copper face, which is NOT
        # the substrate face -- copper has thickness, and pinning the fillet
        # at the substrate plane buries the contact line inside the ring.
        "pad_zs": np.asarray(pad_zs, float),
        "board_z0": np.asarray(min(j.hole.z0 for j in joints), float),
        "barrels": np.asarray(barrels, float).reshape(-1, 4),
        "pad_xy": np.asarray(pad_xy, float).reshape(-1, 2),
        "pad_ptr": np.asarray(pad_ptr, int),
        "pad_ring": np.asarray(pad_ring, int),
        "side": (joints[0].pad_top.side if joints[0].pad_top
                 else joints[0].pad_bottom.side),
        "metal_V": np.asarray(metal_V, float),
        "metal_F": np.asarray(metal_F, int).reshape(-1, 3),
    }


def joint_pad(d, joint):
    """The annulus contours for one joint, as a CrossSection."""
    xy = np.asarray(d["pad_xy"], float)
    ptr = np.asarray(d["pad_ptr"], int)
    ring = np.asarray(d["pad_ring"], int)
    rings = [xy[ptr[i]:ptr[i + 1]] for i in np.nonzero(ring == joint)[0]]
    cs = m3d.CrossSection([[(float(x), float(y)) for x, y in r]
                           for r in rings if len(r) >= 3])
    # SIMPLIFY. The annulus is built by unioning hundreds of mask and copper
    # triangles, which leaves sub-micron slivers along every seam. Extruding
    # that verbatim and intersecting turned the crown into lace -- genus 34
    # instead of 1. 5 um is far below anything the solve resolves.
    return cs.simplify(0.005)


def densify_cs(cs, step=0.15):
    """Insert points so no polygon edge exceeds ``step``.

    The tapered flange wall is only 0.04 mm tall: extruding a simplified
    outline whose edges run millimetres (a square pad's whole side) makes
    wall quads with 20:1 aspect. Subdividing the outline first bounds the
    aspect without disturbing the shape.
    """
    out = []
    for poly in cs.to_polygons():
        P = np.asarray(poly, float)
        ring = []
        for a, b in zip(P, np.roll(P, -1, axis=0)):
            n = int(max(1, np.ceil(np.linalg.norm(b - a) / step)))
            for k in range(n):
                q = a + (b - a) * (k / n)
                ring.append((float(q[0]), float(q[1])))
        out.append(ring)
    return m3d.CrossSection(out)


def pad_outer(pad_cs):
    """The annulus OUTER ring alone, as a filled region.

    Used to clip the crown. The inner contour is the bore, which the plug fills
    anyway, so carrying it into the clip only invites more topology.
    """
    rings = pad_cs.to_polygons()
    if not rings:
        return pad_cs

    def _area(rg):
        P = np.asarray(rg, float)
        x, y = P[:, 0], P[:, 1]
        return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    outer = max(rings, key=_area)
    return m3d.CrossSection([[(float(x), float(y)) for x, y in outer]])


def board_with_hole(cx, cy, r, z0, z1, half=3.0):
    """Substrate slab minus the barrel: the hole is a VOID.

    Cut generously deep top and bottom -- the cylinder spans well past the slab
    so the boolean leaves no sliver of substrate capping the hole.
    """
    slab = m3d.Manifold.cube([2 * half, 2 * half, z1 - z0], center=True).translate(
        [cx, cy, 0.5 * (z0 + z1)])
    # SUBDIVIDE THE BORE VERTICALLY. The solder's barrel-wall facets are the
    # BOARD's bore surface, not the plug's -- the body is blob - board - pin, so
    # the wall comes from this subtraction. Manifold.cylinder segments the
    # circumference only, so the wall arrived as full-height quads: 1.362 mm
    # tall by 0.028 wide, and the patch ran 25000:1 between its shortest and
    # longest edge. Those are the triangles that look missing -- they are there,
    # just edge-on and too degenerate to shade.
    #
    # Vertical divisions target a fixed 0.3 mm ring height, NOT square cells
    # against the circumferential arc. Square cells put ~25 rings down a 1.6 mm
    # board and the barrel wall alone was 1,920 triangles -- 61% of the whole
    # seed, all of it fixed, buried in the hole, and invisible; it then
    # multiplied under refinement into ~22% of the final mesh doing nothing
    # (measured, ict J4 j0). The wall only has to SHADE, which needs quads
    # somewhere under ~10:1, not 48:1 full-height slivers and not square cells:
    # 0.3 mm rings against a 0.068 mm arc is ~4:1. The circumference stays at
    # 48 segments so the hole mouth (which CAN show, e.g. dry_contact) stays
    # round. 96 segments once gave a contact chain with 130 branch vertices;
    # the overshoot beyond the board is 0.05 rather than 1.0 -- subdividing a
    # metre of cylinder that the cut throws away only buys vertices on the seam.
    nseg = 48
    over = 0.05
    h = (z1 - z0) + 2.0 * over
    ndiv = int(max(2, min(8, round(h / 0.30))))
    bore = m3d.CrossSection.circle(r, nseg).extrude(h, ndiv).translate(
        [cx, cy, z0 - over])
    return slab - bore


def pad_outer_radius(pad_cs, cx, cy, r_hole):
    """Equivalent-circle radius of the annulus, from its AREA.

    Not the bbox: the exposed pad is copper ^ mask and is not a clean disc, so
    its bounding box reads 1.000 mm where the annulus that actually holds solder
    is 0.752. Sizing a seed off the bbox makes it a third too big, which is
    enough to stop a dry-contact seed from breaking the ring at all.
    """
    return float(np.sqrt(pad_cs.area() / np.pi + r_hole * r_hole))


def interrupt_region(cx, cy, r, pad_cs, *, front=0.0, bite=2.5, angle=0.0,
                     pin_half=0.25, radius=None, offset=None):
    """Where an INTERRUPTED joint's solder is allowed to sit.

    Built by taking a BITE out of the pad rather than intersecting a small disc.
    That distinction is the whole point of the shape: intersecting a disc makes
    the outer boundary a circular arc of the disc's radius, which reads as a
    sphere stuck on the board and fits the pad nowhere. Subtracting a LARGE
    circle instead leaves the pad's own outline as the outer boundary -- so the
    covered half fits the pad exactly -- and the receding front is a gentle
    concave curve, which is what solder pulling back actually looks like.

    `front` is where the front crosses, as a signed distance from the hole
    centre along the gap direction: 0.0 cuts through the middle, negative
    leaves more solder, positive leaves less. `bite` scales the biting circle
    against the pad radius -- large is nearly a straight chord, small curls the
    front around.

    `radius`/`offset` are the older disc construction, kept so a caller that
    asks for them explicitly still gets one.
    """
    R = pad_outer_radius(pad_cs, cx, cy, r)
    ux, uy = np.cos(angle), np.sin(angle)
    if radius is not None or offset is not None:
        off = pin_half if offset is None else float(offset)
        rad = (0.5 * R) if radius is None else float(radius)
        disc = m3d.CrossSection.circle(rad, 96).translate(
            (cx + off * ux, cy + off * uy))
        return disc, R, off, rad, (rad < off + pin_half)

    Rb = max(bite * R, R + abs(front) + 1e-3)
    d = Rb + float(front)
    cut = m3d.CrossSection.circle(Rb, 192).translate((cx + d * ux, cy + d * uy))
    region = pad_outer(pad_cs) - cut
    # The ring only breaks if the front falls on the near side of the pin's far
    # edge. Silent otherwise: a perfectly valid joint that is not dry contact.
    return region, R, d, Rb, (front < pin_half)


def seed_blob(cx, cy, r, z0, z1, pad_cs, *, barrel_fill=0.75, fillet=0.35,
              crown=0.30, seed_cover=1.0, seat=0.003, interrupt=None,
              side="top"):
    """The initial solder: a crown over the pad plus a plug down the barrel.

    `seed_cover` < 1 shrinks the plug's RADIUS, which is what "start partially
    covering the hole" means geometrically -- the seed does not fill the bore, so
    the cut leaves a gap between solder and barrel wall that the solve then has
    to close (or not). At 1.0 the plug fills the bore and the barrel wall is
    wetted from the start.

    The crown is a truncated cone rather than a cylinder so the free surface
    starts somewhere near a fillet instead of a step, which saves the minimiser
    from unwinding a right angle.
    """
    # The fillet sits on the SOLDER SIDE, which is the face opposite the
    # component body -- a top-inserted part is soldered from below. `zf` is that
    # face and `sgn` points out of the board from it, so the same construction
    # serves both: everything below just mirrors.
    zf, sgn = (z1, 1.0) if side == "top" else (z0, -1.0)
    thick = z1 - z0
    # Seat the BOTTOM too, but only on a full-depth plug: at barrel_fill 1.0 it
    # ends exactly on the board underside, which is the same coincident-face
    # problem as the crown. A partial plug ends mid-barrel, where its bottom is
    # a free surface and must not be nudged -- that face IS the under-fill.
    bot = seat if barrel_fill >= 1.0 else 0.0
    ph = barrel_fill * thick + 0.02 + bot
    # SUBDIVIDE THE WALL VERTICALLY. Manifold.cylinder segments the
    # CIRCUMFERENCE only, so the bore wall came out as full-height quads: 1.362
    # mm tall by 0.028 wide, a 48:1 sliver, and the patch as a whole ran
    # 25000:1 between its shortest and longest edge. Those are the "missing"
    # triangles -- they are present but edge-on and degenerate to shade.
    #
    # Divisions are chosen to make the wall cells roughly SQUARE against the
    # circumferential step, rather than being a fixed number that is wrong for
    # every other board thickness.
    # OVERSIZE the plug slightly when it is meant to fill the bore. At exactly
    # the bore radius the plug wall and the barrel wall are COINCIDENT, and the
    # boolean resolves that degeneracy by collapsing the wall back to
    # full-height quads -- which is why subdividing the bore looked like it did
    # nothing (it survives `slab - bore` at 28 z levels and is destroyed by
    # `blob - board`). Overlapping makes the cut transversal and the wall comes
    # through with the bore's own triangulation. Same lesson as `seat`.
    #
    # Only when seed_cover >= 1: below that the plug is deliberately narrower
    # than the bore and must NOT touch the wall -- that gap is the point.
    rp = r * seed_cover + (0.01 if seed_cover >= 1.0 else 0.0)
    nseg = 48
    step = 2.0 * np.pi * rp / nseg
    ndiv = int(max(4, min(64, round(ph / max(step, 1e-6)))))
    # Plug runs INTO the board from the solder face.
    pz = (zf - barrel_fill * thick - bot) if sgn > 0 else (zf - 0.02)
    plug = m3d.CrossSection.circle(rp, nseg).extrude(
        ph, ndiv).translate([cx, cy, pz])
    # SIMPLIFY THE PLUG when it fills the bore, same cure as the dome pole:
    # the extrude caps are 48-wedge triangle FANS (measured 3.8 deg median
    # min-angle on the under-fill face at barrel_fill < 1, the worst patch of
    # the whole seed), and a fan on a flat face is exactly what a tolerance
    # decimation collapses into a few large well-shaped triangles. 0.005 and
    # not 0.01: the plug oversizes the bore by 0.01, and the decimated
    # 48-gon's chords may dip ~(tol + 0.001) inward -- 0.005 keeps the wall
    # strictly outside the bore so the cut stays transversal. Gated on a
    # full-radius plug: below seed_cover 1.0 the wall is a FREE surface (the
    # gap is the point) and must keep its deliberate z-subdivision.
    if seed_cover >= 1.0:
        plug = plug.simplify(0.005)

    # CROWN, CLIPPED TO THE PAD. Intersecting the cone with the pad annulus
    # extruded upward is what stops solder wetting bare laminate: without it the
    # board solid is just a slab, the crown spreads over whatever it touches,
    # and the annulus computed so carefully in the ingest does nothing. It also
    # cleans up the contact chain -- an unclipped cone meets the slab tangentially
    # around its base and leaves branch vertices where the chain should be
    # simple loops (21 of them, measured).
    outer = pad_cs.bounds()
    rr = 0.5 * max(outer[2] - outer[0], outer[3] - outer[1])
    rr = min(rr, r + fillet)
    # SEAT the crown slightly INTO the board. Standing it exactly on z1 makes
    # its base coplanar with the board top, and a coincident face is what leaves
    # branch vertices in the contact chain (~20 of them, measured) instead of
    # clean closed loops. Overlapping makes the cut transversal. Same trick, and
    # the same reason, as `seat` in the SMD path.
    # DOME, not cone-plus-cap. The truncated cone's flat top was a 96-slice
    # triangle FAN (free surface, so decimation rightly leaves it alone) and
    # its silhouette was piecewise linear -- a hard rim at the top and a hard
    # angle where the taper met the skirt, both of which the minimiser then
    # spends iterations unwinding. Revolve a smooth fillet profile instead:
    # a cosine-eased curve from the foot radius to the axis, so the top
    # closes with no cap at all and the silhouette has no crease. Profile
    # samples sized for roughly square cells, same rule as bore and plug.
    # BODY then CAP. The body eases from the foot radius to the WRAPPING
    # radius (r * 0.9, the old cone tip: it must stay wider than the pin all
    # the way up or the seed stops ringing the pin below the requested
    # climb); the cap then closes to the axis just above, so an over-the-top
    # crown still buries the tip and a sub-tip crown still wets to height.
    # The whole dome spans exactly the CROWN height: body eases from the
    # foot radius to the wrapping radius, the cap closes to the axis within
    # the top of that span. With a sub-tip crown the closing cap sits
    # entirely inside the pin and is cut away (the seed opens around the
    # pin: genus 1); with an over-the-top crown the cap clears the tip and
    # seals it (genus 0). Extending the cap BEYOND the crown height buried
    # the tip even at climb 0.7.
    ch = crown + seat
    r_tip = r * 0.9
    cap_h = min(0.6 * r_tip, 0.4 * ch)
    bh = ch - cap_h
    cstep = 2.0 * np.pi * rr / 96
    nprof = int(max(6, min(48, round(bh / max(cstep, 1e-6)))))
    ts = np.linspace(0.0, 1.0, nprof + 1)
    body_r = r_tip + (rr - r_tip) * 0.5 * (1.0 + np.cos(np.pi * ts))
    body_z = bh * ts
    cs_t = np.linspace(0.0, np.pi / 2.0, 8)[1:]
    cap_r = r_tip * np.cos(cs_t)
    cap_z = bh + cap_h * np.sin(cs_t)
    # (x=radius, y=height) polygon, closed along the axis; revolve is about
    # +Y and re-axes the result to +Z.
    poly = [(0.0, 0.0)]
    poly += [(float(r_), float(z_)) for r_, z_ in zip(body_r, body_z)]
    poly += [(float(r_), float(z_)) for r_, z_ in zip(cap_r, cap_z)
             if r_ > 1e-6]
    poly += [(0.0, float(ch))]
    cone = m3d.Manifold.revolve(m3d.CrossSection([poly]), 96)
    # SIMPLIFY THE DOME SOLID, the SMD blob's own cure (10 um tolerance,
    # python-side, pre-solve): a revolve keeps 96 segments at every radius,
    # so the cap rings needle toward the pole -- but the near-pole fan is
    # nearly coplanar, and a tolerance decimation collapses it into a few
    # large well-shaped triangles instead of adding resolution to chase
    # square cells. Applied to the dome ALONE: on the flange it would strip
    # the densified (collinear) outline points and resurrect the skirt
    # slivers.
    cone = cone.simplify(0.01)
    if sgn < 0:
        cone = cone.mirror([0.0, 0.0, 1.0])
    cone = cone.translate([cx, cy, zf - sgn * seat])
    # Extrude POSITIVE and mirror; a negative height returns an empty manifold,
    # which silently produced an empty body for every bottom-soldered joint.
    prism = pad_outer(pad_cs).simplify(0.01).extrude(crown + 1.0)
    if sgn < 0:
        prism = prism.mirror([0.0, 0.0, 1.0])
    cone = cone ^ prism.translate([0.0, 0.0, zf - sgn * seat])

    # FLOOD THE PAD. The cone's base is a CIRCLE of at most r + fillet, so on
    # a non-circular (or simply larger) pad the contact line pinned well
    # inside the copper -- a square pad got a round foot a third smaller than
    # the annulus. A thin flange at the foot plane, shaped by the pad annulus
    # itself, pins the line at the TRUE pad edge (nominal wave soldering wets
    # the whole exposed annulus); the cone above stays the volume-carrying
    # body. Same idea as the SMD skirt, with the pad contour as the boundary.
    # Taper the flange so the skirt rim is a shallow wedge rather than a
    # vertical 0.04 mm step -- the foot outline (the FLOOD, at the pad edge)
    # is unchanged; only the wall leans inward toward the dome base.
    # scale_top scales about the origin, so recentre on the hole first.
    # Wall cells sized against the SLANT, not the height: the taper drags
    # the top ring (1-s)*r_pad inward, so on a wide pad the wall is a
    # near-horizontal shelf whose cells are long RADIALLY -- the fixed
    # two-division wall measured 7.6 deg median min-angle (0.02 mm tall
    # cells against 0.15 mm outline edges), and a single division just
    # turned the same slivers sideways. Matching divisions to the slant at
    # the 0.10 mm outline step bounds the aspect from both directions.
    fl_cs = densify_cs(pad_outer(pad_cs).simplify(0.01), step=0.10).translate(
        (-cx, -cy))
    fb = fl_cs.bounds()
    r_pad = 0.5 * max(fb[2] - fb[0], fb[3] - fb[1])
    s = float(min(0.95, max(0.55, rr / max(r_pad, 1e-9))))
    slant = float(np.hypot(0.04 + seat, (1.0 - s) * r_pad))
    fdiv = int(max(1, min(6, round(slant / 0.10))))
    flange = fl_cs.extrude(0.04 + seat, fdiv, 0.0, (s, s)).translate(
        (cx, cy, 0.0))
    if sgn < 0:
        flange = flange.mirror([0.0, 0.0, 1.0])
    flange = flange.translate([0.0, 0.0, zf - sgn * seat])
    body = plug + cone + flange

    if interrupt is not None:
        # Confine crown, plug AND flange to the offset disc. Clipping only the
        # crown would leave a full-depth plug sitting in the bore, which still
        # rings the pin below the surface -- the joint would read as dry contact
        # from above and still be genus 1.
        keep = interrupt.extrude(z1 - z0 + crown + 2.0).translate(
            [0.0, 0.0, z0 - 1.0])
        return body ^ keep
    return body


def cut_tht_joint(d, joint, *, barrel_fill=0.75, fillet=0.35, crown=0.30,
                  seed_cover=1.0, seat=0.003, verbose=True, side=None,
                  lead_clear=0.7, lead_climb=None,
                  interrupt=False, interrupt_offset=None, interrupt_radius=None,
                  interrupt_angle=0.0, interrupt_front=0.0, interrupt_bite=2.5,
                  expose=0.0, expose_angle=0.0, expose_seed=None):
    """-> dict(V, F, lab, ...) -- a closed, labelled THT solder body."""
    barrels = np.asarray(d["barrels"], float)
    row = barrels[int(barrels[:, 3].astype(int).tolist().index(joint))] \
        if joint in barrels[:, 3].astype(int) else None
    if row is None:
        raise SystemExit(f"joint {joint} not in {sorted(barrels[:,3].astype(int))}")
    cx, cy, r = float(row[0]), float(row[1]), float(row[2])
    z0, z1 = float(d["board_z0"]), float(d["pad_z"])

    # Solder side: from the caller, else the npz, else read off the component.
    if side is None:
        side = str(d["side"]) if "side" in getattr(d, "files", ()) or (
            isinstance(d, dict) and "side" in d) else None
    if side is None:
        from .tht_spec import solder_side
        side = solder_side(np.asarray(d["metal_V"], float), z0, z1)

    # The FOOT plane is the pad's exposed-copper face when the joint carries
    # one (pad_zs, per joint) -- the substrate face otherwise. Copper has
    # thickness: pinning at the substrate plane sinks the contact line a full
    # copper depth into the ring. Extending the solder-side face to the pad
    # plane also lets the plug wet through the ring's bore, which is real.
    has_pz = (isinstance(d, dict) and d.get("pad_zs") is not None) or (
        not isinstance(d, dict) and "pad_zs" in getattr(d, "files", ()))
    if has_pz:
        pz = float(np.asarray(d["pad_zs"], float)[
            int(np.nonzero(barrels[:, 3].astype(int) == joint)[0][0])])
        if side == "bottom":
            z0 = min(z0, pz)
        else:
            z1 = max(z1, pz)

    pad_cs = joint_pad(d, joint)
    solids = {"board": board_with_hole(cx, cy, r, z0, z1)}

    mV = np.asarray(d["metal_V"], float)
    mF = np.asarray(d["metal_F"], int).reshape(-1, 3)
    # Only the geometry near THIS hole; the other pins and the package body are
    # someone else's joint and would just slow the boolean down. Solidify the
    # WHOLE component once and CROP with an intersection -- clipping triangles
    # first leaves an open soup at the box boundary that repair cannot always
    # close (J6 on artix-dc-scm: 11 of 14 crops repaired to an INVALID
    # manifold, and manifold3d propagates that through the boolean, silently
    # collapsing the joint to volume 0). Same hoist the SMT path already got.
    metal_solid = d.get("_metal_solid") if isinstance(d, dict) else None
    if metal_solid is None:
        metal_solid = wb.to_manifold(mV, mF, repair=True, verbose=False)
        if isinstance(d, dict):
            d["_metal_solid"] = metal_solid   # once per component, not per joint
    crop = metal_solid ^ m3d.Manifold.cube((6.0, 6.0, 1e4), True).translate(
        (cx, cy, 0.0))
    if crop.volume() > 0:
        solids["metal"] = crop
    elif metal_solid.volume() <= 0:
        # The whole-component repair itself failed: fall back to the old
        # per-joint crop rather than solder straight through the lead.
        keep = np.abs(mV[mF][:, :, 0] - cx).max(1) < 3.0
        keep &= np.abs(mV[mF][:, :, 1] - cy).max(1) < 3.0
        if keep.any():
            solids["metal"] = wb.to_manifold(mV, mF[keep], repair=True,
                                             verbose=False)

    # CLIP THE CROWN TO THE LEAD PROTRUSION. The crown height was a fixed
    # 0.30 mm regardless of how far the lead actually sticks out, so on a short
    # poke-through the solder domes straight over the lead tip: TO92's pin
    # clears the board by 0.138 mm and the crown buried it by 0.162 mm. A real
    # joint leaves the tip proud, and a buried lead is its own AOI defect --
    # you cannot verify a lead you cannot see.
    #
    # Measured per joint, because protrusion varies by part and by how well the
    # part was seated. lead_clear is the fraction of the protrusion the solder
    # may reach.
    mv = np.asarray(d["metal_V"], float)
    nr = mv[np.hypot(mv[:, 0] - cx, mv[:, 1] - cy) <= r + 0.05]
    if len(nr):
        prot = float((z0 - nr[:, 2].min()) if side == "bottom"
                     else (nr[:, 2].max() - z1))
        if lead_climb is not None:
            # LEAD CLIMB: the requested wetted height as a fraction of the
            # measured protrusion -- the user-facing observable. climb <= 1
            # leaves the tip proud; climb > 1 seeds the blob OVER the tip
            # (crown passes the pin end, so the cut keeps a solder cap on it)
            # and the volume target shrink-wraps the dome from there. The
            # crown/lead_clear pair become derived internals under this knob.
            if prot < 0.02:
                # protrusion not modeled (flush-cut export): the fraction has
                # no datum. Keep the requested absolute crown and say so.
                import warnings
                warnings.warn(
                    f"THT joint {joint}: lead protrusion measures "
                    f"{prot:.4f} mm -- lead_climb={lead_climb:g} has no "
                    f"datum; keeping the absolute crown {crown:.3f} mm")
            else:
                crown = float(lead_climb) * prot
                if verbose:
                    over = "" if lead_climb <= 1.0 else (
                        "  (OVER THE TOP: tip buried by "
                        f"{crown - prot:.3f} mm)")
                    print(f"    lead protrudes {prot:.3f} mm; lead_climb "
                          f"{lead_climb:g} -> crown {crown:.3f}{over}")
        else:
            cap = max(0.02, lead_clear * prot)
            if crown > cap:
                if verbose:
                    print(f"    lead protrudes {prot:.3f} mm; crown "
                          f"{crown:.3f} -> {cap:.3f} so the tip stays proud")
                crown = cap

    cut_disc = None
    if expose and expose > 0.0:
        # INTERRUPTED (pad exposure): the foot stops short of the pad edge,
        # leaving copper visible, but the ring around the pin stays CLOSED
        # (the core disc) -- the genus-breaking gap is dry_contact, a
        # different defect. The keep front is noisy (see exposure_keep);
        # the exposed-area fraction is exact by construction.
        rng = np.random.default_rng(expose_seed)
        cut_disc, achieved = wb.exposure_keep(
            pad_outer(pad_cs), cx, cy, r + 0.15, float(expose),
            float(expose_angle), rng)
        if verbose:
            print(f"    interrupted: pad exposure {achieved:.0%} "
                  f"(requested {float(expose):.0%}) from "
                  f"{np.degrees(expose_angle):.0f} deg")
    if interrupt:
        idisc, R, off, rad, breaks = interrupt_region(
            cx, cy, r, pad_cs, offset=interrupt_offset, radius=interrupt_radius,
            angle=interrupt_angle, front=interrupt_front, bite=interrupt_bite)
        cut_disc = idisc if cut_disc is None else (cut_disc ^ idisc)
        if verbose:
            print(f"    dry contact: pad outer R={R:.3f} (from area), "
                  f"front {interrupt_front:+.3f} bite x{interrupt_bite:g} at "
                  f"{np.degrees(interrupt_angle):.0f} deg"
                  + ("" if breaks else "   !! front is past the pin -- "
                                       "this will NOT break the ring"))
    b = seed_blob(cx, cy, r, z0, z1, pad_cs, barrel_fill=barrel_fill,
                  fillet=fillet, crown=crown, seed_cover=seed_cover,
                  seat=seat, interrupt=cut_disc, side=side)

    b = b.as_original()
    solids = {k: v.as_original() for k, v in solids.items()}
    names = {b.original_id(): "meniscus"}
    for nm, s in solids.items():
        names[s.original_id()] = "wet_" + nm

    body = b
    for s in solids.values():
        body = body - s
    if expose and expose > 0.0 and body.volume() > 0.0:
        # The bite can STRAND solder: a far-side flange arc cut off from the
        # dome survives the keep region as its own closed component, and a
        # disconnected island both lies (euler is additive, so the genus
        # check misreads) and renders as floating solder. Keep the main
        # body; islands under 5% of the volume are construction artifacts,
        # bigger ones are reported and still dropped (a real detached
        # splash is a DIFFERENT defect class).
        parts = body.decompose()
        if len(parts) > 1:
            parts = sorted(parts, key=lambda m: m.volume(), reverse=True)
            dropped = sum(m.volume() for m in parts[1:])
            if verbose or dropped > 0.05 * body.volume():
                import warnings
                warnings.warn(
                    f"THT joint {joint}: pad exposure stranded "
                    f"{len(parts) - 1} solder island(s) "
                    f"({dropped:.4f} mm^3) -- dropped")
            body = parts[0]
    if body.volume() <= 0.0 and b.volume() > 0.0:
        # A non-empty seed that cuts to NOTHING is a boolean failure, not a
        # bare joint -- downstream drops zero-volume joints, so say so loudly.
        import warnings
        warnings.warn(
            f"THT joint {joint} at ({cx:.3f}, {cy:.3f}): seed blob "
            f"{b.volume():.4f} mm^3 cut to volume 0 -- boolean failure, "
            f"joint will be dropped")
    V, F = wb.from_manifold(body)
    lab = wb.tri_labels(body, names).astype(str)
    # Retriangulate exactly-planar WETTED patches from their boundary loops,
    # same as the SMD cut: the boolean fragments the pad annulus, pin walls
    # and barrel seams into sliver confetti, and those patches are FIXED --
    # Evolver's mesh maintenance cannot touch them, and every refinement
    # quarters every sliver. Guarded internally (check_body) so a patch it
    # cannot retriangulate safely is left alone.
    V, F, lab = wb.decimate_planar_patches(V, F, lab, verbose=verbose)

    if verbose:
        import collections
        n = collections.Counter(lab)
        chk = wb.check_body(V, F, lab)
        eu = chk.get("euler")
        print(f"  joint {joint}: hole d={2*r:.3f} at ({cx:.3f},{cy:.3f})  "
              f"solder side {side.upper()}  vol {body.volume():.5f} mm^3")
        print(f"    labels: {dict(n)}")
        print(f"    euler={eu}  genus={(2 - eu)//2 if eu is not None else '?'}  "
              f"(genus 1 = wraps the pin; genus 0 = dry contact, open on one side)")
    # Keep the PRE-CUT blob too, so a viewer can show the boolean's input
    # against its output. For THT that pairing is worth more than for SMD: the
    # seed's plug is what decides whether the joint ends up wrapping the pin at
    # all, and that is invisible once the cut has happened.
    bV, bF = wb.from_manifold(b)

    # REFERENCE VOLUMES for the `fullness` parameter, both from THIS joint's
    # own geometry so the number means the same thing on every component:
    #   plug_volume         solder filling the bore beside the pin
    #   fillet_cone_volume  the straight-sided fillet between the pad outline
    #                       and the wetted lead ring at the climb height
    # fullness 1.0 IS the straight cone; < 1 is concave (bright/starved),
    # > 1 convex (hidden/excess). The solve target is
    #   plug_volume + fullness * fillet_cone_volume,
    # replacing tht_fill's fraction-of-the-cut-volume, which was a fraction
    # of a CONSTRUCTION artifact and read differently on every footprint.
    # Pin cross-sections are measured from the joint's metal crop by
    # thin-slab volumes -- exact for prismatic leads, and blade pins (SW1)
    # are as prismatic as round ones.
    zf_v, sgn_v = (z0, -1.0) if side == "bottom" else (z1, 1.0)
    _metal = solids.get("metal")

    def _metal_area(z):
        # Confined to THIS hole's neighbourhood: the crop box is 6 mm wide
        # and on a fine-pitch connector it holds the NEIGHBOUR pins too --
        # measured on ict J4 (2.54 mm pitch) the unconfined slab counted
        # more pin area than the whole bore and zeroed the plug reference.
        if _metal is None:
            return 0.0
        eps = 0.01
        w = 2.0 * (r + 0.3)
        slab = m3d.Manifold.cube((w, w, 2 * eps), True).translate(
            (cx, cy, z))
        return max(0.0, float((_metal ^ slab).volume()) / (2 * eps))

    h_eff = float(crown)
    if len(nr) and prot >= 0.02:
        h_eff = float(min(crown, prot))
    a_pad = float(pad_outer(pad_cs).area())
    a_pin_foot = _metal_area(zf_v + sgn_v * min(0.02, 0.5 * h_eff))
    a_pin_tip = _metal_area(zf_v + sgn_v * max(h_eff - 0.02, 0.5 * h_eff))
    v_outer = h_eff / 3.0 * (a_pad + a_pin_tip + np.sqrt(a_pad * a_pin_tip))
    v_pin = h_eff * 0.5 * (a_pin_foot + a_pin_tip)
    fillet_cone = max(0.0, float(v_outer - v_pin))
    span = float(barrel_fill) * (z1 - z0)
    a_pin_bore = _metal_area(zf_v - sgn_v * 0.5 * span)
    plug = max(0.0, float(np.pi * r * r - a_pin_bore)) * span

    return dict(V=V, F=F, lab=lab, body=body, cx=cx, cy=cy, r=r, side=side,
                pad_z=z1, board_z0=z0, volume=float(body.volume()),
                plug_volume=float(plug), fillet_cone_volume=float(fillet_cone),
                climb_h=float(h_eff),
                blob=b, blob_mesh=(bV, bF))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="TO92")
    ap.add_argument("--tag", default="_tht")
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--barrel-fill", type=float, default=0.75,
                    help="fraction of the board thickness the solder plug "
                         "spans, measured DOWN FROM THE TOP. Sets the wetted "
                         "barrel wall exactly 1:1 -- 0.75 wets z 0.340..1.362 "
                         "on a 1.362 mm board, i.e. 75%%. Default is 0.75 "
                         "because that is IPC-A-610 class 2's minimum barrel "
                         "fill; 1.0 wets the entire wall, which is more solder "
                         "than a real joint has. NOT a measurement either way "
                         "-- THT has no volume datum.")
    ap.add_argument("--seed-cover", type=float, default=1.0,
                    help="plug radius as a fraction of the bore. <1 starts the "
                         "seed only partially covering the hole.")
    ap.add_argument("--interrupt", action="store_true",
                    help="INTERRUPTED joint: confine the solder to a disc of "
                         "about half the normal radius, offset so its centre is "
                         "on the pin boundary and its far edge reaches the pad "
                         "edge. The far side is left bare.")
    ap.add_argument("--interrupt-offset", type=float, default=None)
    ap.add_argument("--interrupt-radius", type=float, default=None)
    ap.add_argument("--interrupt-angle", type=float, default=0.0,
                    help="degrees; which way the gap faces")
    ap.add_argument("--interrupt-front", type=float, default=0.0,
                    help="where the receding front crosses, signed mm from the "
                         "hole centre. 0 = through the middle; negative leaves "
                         "more solder, positive less.")
    ap.add_argument("--interrupt-bite", type=float, default=2.5,
                    help="radius of the biting circle, x the pad radius. Large "
                         "= nearly a straight chord, small curls the front.")
    ap.add_argument("--fillet", type=float, default=0.35)
    ap.add_argument("--crown", type=float, default=0.30,
                    help="fillet height above the solder face, mm. CLIPPED to "
                         "--lead-clear x the measured lead protrusion so the "
                         "solder never domes over the lead tip.")
    ap.add_argument("--lead-clear", type=float, default=0.7,
                    help="fraction of the lead protrusion the solder may reach. "
                         "1.0 buries the tip exactly; above 1.0 covers it.")
    a = ap.parse_args()

    d = load_tht(a.family, a.tag)
    js = sorted(np.asarray(d["barrels"], float)[:, 3].astype(int)) if a.all \
        else [a.joint]
    print(f"{a.family}{a.tag}: barrel_fill={a.barrel_fill} seed_cover={a.seed_cover}")
    for j in js:
        cut_tht_joint(d, j, barrel_fill=a.barrel_fill, fillet=a.fillet,
                      crown=a.crown, seed_cover=a.seed_cover,
                      lead_clear=a.lead_clear,
                      interrupt=a.interrupt, interrupt_offset=a.interrupt_offset,
                      interrupt_radius=a.interrupt_radius,
                      interrupt_angle=np.radians(a.interrupt_angle),
                      interrupt_front=a.interrupt_front,
                      interrupt_bite=a.interrupt_bite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
