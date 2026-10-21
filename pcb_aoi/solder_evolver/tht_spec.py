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

"""Build THT joint inputs in the shape of the PCBA-to-USD contract.

A compatibility shim derives the fields from non-conformant boards.

The PCBA-to-USD contract already specifies everything a through-hole joint
needs:

    board_hole    holeCenterMm, holeShape, holeDimsMm, holeSide,
                  plated, occupied, component, componentPinNumber
    board_barrel  the copper through the board thickness
    board_pad     padContour, padNumber, occupied
    pin           number, solderable, crossSection, terminalType
    component     processClass = through_hole

Two ingest paths feed the same :class:`ThtJoint` shape, so the solver only ever
sees contract fields: :func:`from_conformant` reads a schema-carrying board via
the board_metadata catalog, and :func:`from_raw_layers` SYNTHESISES the same
fields from the raw layers of a pre-schema board. :func:`discover` routes --
conformant when the catalog yields joints, shim otherwise -- and reports which
path answered.

Three design points follow from the contract:

* ``padContour`` is the **solder-mask opening, not the copper contour**,
  because the copper under a pad is often one continuous pour. Reading copper
  can give one joint a 12.4 mm^2 pour and another a 5.8 mm^2 trace, where the
  mask openings give 1.21 and 1.51 mm^2. Where per-pad copper cannot be
  separated, the mask opening is the per-pad geometry.

* Holes are **real voids**. Some exports ship solid cylinders instead (for
  example solid ``Platings`` barrels); the shim carves the void itself.

* ``plated`` and the pin<->hole association are **stated by the exporter, not
  inferred** -- association is authored board-side rather than derived from XY
  coincidence. The shim infers both for pre-schema boards: it calls every
  plating plated, and it decides occupancy by testing whether a lead sits
  within the hole in XY. That is the least reliable part of this module, and it
  is why every synthesised field is tagged.

Provenance mirrors the contract's ``pcba:prov:<attr> = "not_yet_available"``
escape hatch, so a run can report what was read versus what was invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Field came from a conformant asset.
SPEC = "spec"
#: Field was synthesised by the shim; the asset does not carry it yet.
SYNTH = "not_yet_available"


@dataclass
class BoardHole:
    """`board_hole` -- a real void through the substrate."""

    center_mm: tuple[float, float]
    shape: str  # "circular" | other
    dims_mm: tuple[float, float]  # diameter, or extents for non-circular
    side: str  # "top" | "bottom"
    plated: bool
    occupied: bool
    component: str | None = None  # rel -> component instance
    component_pin_number: str | None = None
    z0: float = 0.0  # substrate bottom / top; not a contract
    z1: float = 0.0  # field, but the barrel implies it


@dataclass
class BoardPad:
    """`board_pad` -- one prim per pad / annular ring, per side.

    ``contour`` is the EXPOSED pad (the solder-mask opening), per the contract,
    and for a through-hole pad it is an ANNULUS: outer ring plus the hole. It is
    therefore a list of contours, not one.
    """

    pad_number: str
    contours: list[np.ndarray]
    side: str
    occupied: bool
    z: float = 0.0

    def area(self):
        """Return the exposed pad area after subtracting contour holes.

        Returns:
            Cross-sectional area in squared board-coordinate units.

        Example:

        .. code-block:: python

            print(pad.area())

        """
        import manifold3d as m3d

        return float(
            m3d.CrossSection(
                [[(float(x), float(y)) for x, y in c] for c in self.contours]
            ).area()
        )


@dataclass
class Pin:
    """Component-side pin.

    `crossSection` matters because a rectangular pin leaves a different
    annular gap from a circular pin.
    """

    number: str
    solderable: bool = True
    cross_section: str = "other"  # circular | rectangular | other
    terminal_type: str = "throughHolePin"


@dataclass
class ThtJoint:
    """One through-hole joint: a hole, its barrel, its pad(s) and the pin."""

    hole: BoardHole
    pad_top: BoardPad | None
    pad_bottom: BoardPad | None
    pin: Pin
    barrel_z: tuple[float, float]
    prov: dict = field(default_factory=dict)

    def synthesised(self):
        """Which contract fields were invented rather than read."""
        return sorted(k for k, v in self.prov.items() if v == SYNTH)


def describe(joints):
    """Summarize which joint fields were read and which were synthesized.

    Do not imply that a synthesized field was read from the asset.
    """
    if not joints:
        return "no joints"
    synth = sorted({k for j in joints for k in j.synthesised()})
    spec = sorted({k for j in joints for k, v in j.prov.items() if v == SPEC})
    return (
        f"{len(joints)} joint(s)\n"
        f"  from the asset : {', '.join(spec) or '(nothing)'}\n"
        f"  SYNTHESISED    : {', '.join(synth) or '(nothing)'}"
    )


# --------------------------------------------------------------- conformant


def from_conformant(
    stage,
    catalog,
    component_path,
    comp_V,
    xform_cache=None,
    cache=None,
    lead_margin=0.05,
    pad_ratio=1.8,
    pour_clamp=3.0,
    board_root=None,
):
    """Read `pcba:*` attributes off a schema-carrying stage, via the catalog.

    `catalog` is a board_metadata BoardMetadataCatalog (the UsdSchema provider's
    output); its hole->component association is the candidate set. Discovery is
    by ROLE and ATTRIBUTE, never by prim name -- the contract's stage layout is
    illustrative.

    ATTRIBUTE values come from the CATALOG RECORDS, never re-read off prims:
    the usd_schema provider is the single owner of the ``pcba:*`` attribute
    names, so a schema rename is a one-place change. The stage is touched
    only for GEOMETRY (hole/pad/substrate meshes) -- which is not catalog
    material. Authoredness survives the catalog: these are custom attributes,
    so a record field is non-None exactly when the exporter authored it.

    What is READ (SPEC) versus still INVENTED (SYNTH) on current exports:

    * hole center / dims / shape / plated come from the HoleRecord (the
      provider read them off ``pcba:board:*``) -- SPEC at last.
    * the hole->component association: SPEC when resolved from either
      authored form -- the refdes + componentPinNumber PAIR (the contract's
      primary association) or the secondary
      ``pcba:board:component`` rel; ``HOLE_ASSOCIATION_ORIGIN_KEY`` in the
      record extensions says which. Only the geometric fallback for holes
      with no authored association at all stays SYNTH.
    * ``occupied``: the exporter authors occupied=False +
      intentionallyUnpopulated=True on EVERY hole while leads demonstrably
      pass through them, so an authored False (or absent) value is never
      trusted -- the same geometric guess the shim uses decides, tagged SYNTH.
      Only an authored True is taken as SPEC. A hole the guess also calls
      empty yields no joint.
    * the FILLET-side pad: when a pad prim exists on that side its outer-face
      triangles are unioned into the annulus contour (SPEC); these exports
      author top copper only, so a bottom fillet falls back to a drill-derived
      annulus at ``pad_ratio * r`` (SYNTH), the shim's own stand-in.
    * z0/z1 come from the ``board_substrate`` mesh (by ``pcba:mesh:role``),
      cross-checked against ``pcba:board:thicknessMm``; the barrel is still
      derived, not read from a board_barrel prim, so it stays SYNTH.

    `comp_V` is the component's world-space vertices (occupancy guess and
    fillet side). Pass a dict as `cache` when scanning a whole board.
    """
    import manifold3d as m3d
    from pxr import Usd, UsdGeom

    from pcb_aoi.board_metadata import (
        HOLE_ASSOCIATION_ORIGIN_KEY,
        MESH_ROLE_ATTR,
    )

    holes = [h for h in catalog.holes if h.component_path == component_path]
    comp_V = np.asarray(comp_V, float)
    if not holes and not any(
        c.prim_path == component_path and c.process_class == "through_hole"
        for c in catalog.components
    ):
        # No authored association and not even declared through-hole:
        # nothing to guess from. (Through-hole parts fall through to the
        # geometric fallback below, AFTER the board faces are known.)
        return []
    xc = xform_cache or UsdGeom.XformCache()
    cache = {} if cache is None else cache
    pred = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

    def _mesh_world(prim):
        """(world points, fan triangles) of one Mesh prim, or (None, None)."""
        m = UsdGeom.Mesh(prim)
        if not m:
            return None, None
        pts = m.GetPointsAttr().Get()
        counts = m.GetFaceVertexCountsAttr().Get()
        idx = m.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            return None, None
        M = xc.GetLocalToWorldTransform(prim)
        P = np.array([list(M.Transform(q)) for q in pts], float)
        out, off = [], 0
        for c in counts:
            for k in range(1, c - 1):
                out.append((idx[off], idx[off + k], idx[off + k + 1]))
            off += c
        return P, np.asarray(out, np.int64).reshape(-1, 3)

    # --- board faces: the catalog's measured substrate extent when the
    # provider recorded one; a role-scan of the stage only as the fallback
    # (e.g. a catalog whose source could not measure geometry).
    # The caller's board root wins; a hole's recorded root next. Guessing
    # boards[0] is only safe on a single-board catalog -- on a PANEL it can
    # silently read another board's substrate extent.
    root = board_root or (holes[0].board_root if holes else None)
    if root is None:
        if len(catalog.boards) > 1:
            raise ValueError(
                f"{component_path}: no board root given and the catalog "
                f"holds {len(catalog.boards)} boards -- refusing to guess"
            )
        root = catalog.boards[0].root_path if catalog.boards else None
    if root is None:
        return []
    skey = ("conformant_substrate", root)
    if skey not in cache:
        board = next((b for b in catalog.boards if b.root_path == root), None)
        zlo = zhi = None
        if board is not None and board.substrate_z_range is not None:
            zlo, zhi = board.substrate_z_range
        else:
            rootp = stage.GetPrimAtPath(root)
            if rootp and rootp.IsValid():
                for pr in Usd.PrimRange(rootp, pred):
                    a = pr.GetAttribute(MESH_ROLE_ATTR)
                    role = a.Get() if a and a.HasAuthoredValue() else None
                    if str(role or "") != "board_substrate":
                        continue
                    P, _T = _mesh_world(pr)
                    if P is None:
                        continue
                    z = P[:, 2]
                    zlo = float(z.min()) if zlo is None else min(zlo, float(z.min()))
                    zhi = float(z.max()) if zhi is None else max(zhi, float(z.max()))
        thick = board.thickness_mm if board is not None else None
        if zlo is None:
            if thick is None:
                raise ValueError(
                    f"{root}: no board_substrate mesh (pcba:mesh:role) and no "
                    f"pcba:board:thicknessMm -- the board faces are unknowable"
                )
            zlo, zhi = 0.0, float(thick)
        elif thick is not None and abs((zhi - zlo) - float(thick)) > max(
            0.1, 0.1 * float(thick)
        ):
            # substrate mesh and stated thickness disagree beyond tolerance;
            # the MESH wins (the solve happens in mesh space), but flag it.
            import warnings

            warnings.warn(
                f"{root}: board_substrate spans {zhi - zlo:.3f} mm but "
                f"pcba:board:thicknessMm says {float(thick):.3f}"
            )
        cache[skey] = (zlo, zhi)
    z0, z1 = cache[skey]

    if not holes:
        # ASSOCIATION FALLBACK: the exporter authored no hole->component rel
        # for this through-hole part (SIEMANS: 49 of 1590 holes carry one).
        # Infer it geometrically with a lead-in-hole test against UNASSOCIATED
        # holes only, so an authored association can never be stolen. Every
        # field of a guessed joint is already SYNTH-tagged downstream (no
        # HOLE_ASSOCIATION_ORIGIN_KEY on the record => assoc prov SYNTH).
        gkey = ("hole_geoms", root)
        if gkey not in cache:
            geoms = []
            for hh in catalog.holes:
                if hh.component_path is not None or hh.plated is False:
                    continue
                if hh.center_mm is not None and hh.dimensions_mm is not None:
                    gcx, gcy = float(hh.center_mm[0]), float(hh.center_mm[1])
                    gr = 0.5 * float(max(hh.dimensions_mm[0], hh.dimensions_mm[1]))
                else:
                    HP, _T = _mesh_world(stage.GetPrimAtPath(hh.prim_path))
                    if HP is None:
                        continue
                    gcx, gcy = float(HP[:, 0].mean()), float(HP[:, 1].mean())
                    e = HP.max(0) - HP.min(0)
                    gr = 0.5 * float(max(e[0], e[1]))
                geoms.append((hh, gcx, gcy, gr))
            cache[gkey] = geoms
        guessed = []
        for hh, gcx, gcy, gr in cache[gkey]:
            near = comp_V[
                np.hypot(comp_V[:, 0] - gcx, comp_V[:, 1] - gcy) <= gr + lead_margin
            ]
            if not len(near):
                continue
            if not (near[:, 2].min() < z0 + 1e-6 and near[:, 2].max() > z1 - 1e-6):
                continue
            import dataclasses

            guessed.append(dataclasses.replace(hh, component_path=component_path))
        if guessed:
            import warnings

            warnings.warn(
                f"{component_path}: {len(guessed)} hole(s) associated "
                f"GEOMETRICALLY -- the exporter authored no "
                f"pcba:board:component rel for this through-hole part"
            )
        holes = guessed
    if not holes:
        return []

    comp_rec = next(
        (c for c in catalog.components if c.prim_path == component_path), None
    )

    # FILLET SIDE: the authored placement side wins -- a top-placed part is
    # soldered from the bottom, and vice versa (the same rule the procedural
    # solver applies). The geometry heuristic is only the fallback for
    # unauthored exports, because it reads "which side has the body" off
    # vertex mass -- and an export that models ONLY pkg_lead meshes (SIEMANS
    # one such export: no pkg_body prims at all) puts all its mass on the
    # PROTRUSION side, flipping the answer and soldering inside the barrel.
    placement = (comp_rec.side or "").lower() if comp_rec is not None else ""
    if placement.startswith(("t", "b")):
        sd = "bottom" if placement.startswith("t") else "top"
        side_prov = SPEC
    else:
        sd = solder_side(comp_V, z0, z1)
        side_prov = SYNTH

    pads_by_term = {}
    for p in catalog.pads:
        if p.component_path == component_path and p.terminal_number:
            pads_by_term.setdefault(p.terminal_number, []).append(p)
    terms = {
        t.number: t
        for t in catalog.terminals
        if t.component_path == component_path and t.number
    }
    pc_prov = SPEC if (comp_rec is not None and comp_rec.process_class) else SYNTH

    joints = []
    for h in holes:
        prim = stage.GetPrimAtPath(h.prim_path)
        if not prim or not prim.IsValid():
            continue
        HP, _HT = _mesh_world(prim)  # some exports mesh the hole prim

        if h.center_mm is not None:
            cx, cy = float(h.center_mm[0]), float(h.center_mm[1])
            center_prov = SPEC
        elif HP is not None:
            cx, cy = float(HP[:, 0].mean()), float(HP[:, 1].mean())
            center_prov = SYNTH
        else:
            continue  # nothing locates this hole

        if h.dimensions_mm is not None:
            # dims_mm[0] must be the GOVERNING diameter (downstream takes
            # dims_mm[0] / 2 as the radius), so keep it max-first.
            dx, dy = sorted(
                (float(h.dimensions_mm[0]), float(h.dimensions_mm[1])), reverse=True
            )
            dims_prov = SPEC
        elif HP is not None:
            e = HP.max(0) - HP.min(0)
            dx = dy = float(max(e[0], e[1]))
            dims_prov = SYNTH
        else:
            continue
        r = 0.5 * dx

        shape = h.shape
        shape_prov = SPEC if shape else SYNTH
        shape = str(shape) if shape else "circular"

        plated = h.plated
        if plated is False:
            continue  # unplated: not a solder site
        plated_prov = SPEC if plated is not None else SYNTH

        # ASSOCIATION: SPEC when the provider resolved it from EITHER
        # authored form -- the refdes + componentPinNumber pair (the
        # contract's PRIMARY association) or
        # the secondary pcba:board:component rel. SYNTH only for holes our
        # geometric fallback matched (no authored association at all).
        assoc = (
            SPEC
            if h.extensions.get(HOLE_ASSOCIATION_ORIGIN_KEY)
            in ("relationship", "refdes")
            else SYNTH
        )

        # OCCUPANCY: an authored True is read; False/absent is the exporter
        # gap and falls to the shim's geometric guess (a lead inside the hole
        # radius in XY that spans the substrate).
        if h.occupied is True:
            occ_prov = SPEC
        else:
            near = comp_V[
                np.hypot(comp_V[:, 0] - cx, comp_V[:, 1] - cy) <= r + lead_margin
            ]
            if not len(near):
                continue
            if not (near[:, 2].min() < z0 + 1e-6 and near[:, 2].max() > z1 - 1e-6):
                continue
            occ_prov = SYNTH

        bz0, bz1 = z0, z1
        if HP is not None:
            hz0, hz1 = float(HP[:, 2].min()), float(HP[:, 2].max())
            if hz1 - hz0 > bz1 - bz0:
                bz0, bz1 = hz0, hz1

        # --- the FILLET-side pad. Only that side is packed: downstream takes
        # pad_top OR pad_bottom as "the" pad and reads the solder side off it.
        # --- the FILLET-side pad foot, best evidence first:
        #   1. authored padContour on a fillet-side pad record (the contract's
        #      answer: the mask opening, immune to pour bleed)      -> SPEC
        #   2. fillet-side pad MESH exposed-face union, at its MEASURED
        #      exposed-face z                                        -> SPEC
        #      (pour-clamped: copper is frequently pour-connected, so anything
        #       past pour_clamp x drill radius is cut away; a clamp that
        #       actually bites means we invented the boundary        -> SYNTH)
        #   3. the MASK-OPENING derivation the contract itself prescribes
        #      (the pad-ring ROI): the solder-side board_mask opening around
        #      the hole, limited to the copper sheet, at the copper's
        #      exposed z. The normal case on exports that author board_pad
        #      prims for one side only                               -> SYNTH
        #   4. drill-derived annulus at pad_ratio x r, the last-resort
        #      stand-in that used to be the ONLY fallback            -> SYNTH
        fillet_z = z1 if sd == "top" else z0
        drill = m3d.CrossSection.circle(r, 64).translate((cx, cy))
        clamp_cs = m3d.CrossSection.circle(pour_clamp * r, 96).translate((cx, cy))

        def _pad_from_contour(p):
            """Convert the authored pad contour to an annulus, or return None."""
            if not p.contour or len(p.contour) < 3:
                return None
            cs = m3d.CrossSection([[(float(q[0]), float(q[1])) for q in p.contour]])
            if cs.is_empty():
                cs = m3d.CrossSection(
                    [[(float(q[0]), float(q[1])) for q in p.contour][::-1]]
                )
            cs = cs - drill
            if cs.is_empty():
                return None
            return [np.asarray(x, float) for x in cs.to_polygons()]

        def _pad_from_mesh(p):
            """Exposed-face union of a pad MESH, pour-clamped.

            Returns (contours, clamped) or None. The exposed face is picked by
            the PAD's own side (a top pad's exposed face is its max-z face)
            so an opposite-side pad unions correctly before re-basing.
            """
            pp = stage.GetPrimAtPath(p.prim_path)
            P, T = _mesh_world(pp)
            if P is None or not len(T):
                return None
            if (p.side or sd) == "top":
                zf = float(P[:, 2].max())
                face = T[P[T][:, :, 2].min(1) > zf - 1e-4]
            else:
                zf = float(P[:, 2].min())
                face = T[P[T][:, :, 2].max(1) < zf + 1e-4]
            parts = []
            for tri in face:
                q = [(float(P[i, 0]), float(P[i, 1])) for i in tri]
                # a bottom face projects clockwise; wind CCW so the union fills
                a2 = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[1][1] - q[0][1]) * (
                    q[2][0] - q[0][0]
                )
                parts.append(m3d.CrossSection([q[::-1] if a2 < 0 else q]))
            if not parts:
                return None
            cs = m3d.CrossSection.batch_boolean(parts, m3d.OpType.Add)
            if cs.is_empty():
                return None
            area = cs.area()
            cs = cs ^ clamp_cs
            if cs.is_empty():
                return None
            clamped = cs.area() < 0.95 * area
            return ([np.asarray(x, float) for x in cs.to_polygons()], clamped, zf)

        def _mask_opening_foot():
            """Derive the fillet-side mask opening around the hole.

            Limit the opening to the copper sheet, at the copper's
            exposed z -- the contract's own derivation for when no
            fillet-side board_pad prim exists (exports that author pads for
            one side only). Returns (contours, z) or None.
            """
            lkey = ("layer_faces", root, sd)
            if lkey not in cache:
                layers = {"board_mask": [], "board_copper": []}
                zmid = 0.5 * (z0 + z1)
                rootp2 = stage.GetPrimAtPath(root)
                if rootp2 and rootp2.IsValid():
                    for pr2 in Usd.PrimRange(rootp2, pred):
                        a2v = pr2.GetAttribute(MESH_ROLE_ATTR)
                        role = str(a2v.Get()) if a2v and a2v.HasAuthoredValue() else ""
                        if role not in layers:
                            continue
                        P2, T2 = _mesh_world(pr2)
                        if P2 is None or not len(T2):
                            continue
                        # this side's sheet only (by z, the frame-proof test)
                        if (float(P2[:, 2].mean()) < zmid) != (sd == "bottom"):
                            continue
                        if sd == "bottom":
                            zf2 = float(P2[:, 2].min())
                            face = T2[P2[T2][:, :, 2].max(1) < zf2 + 1e-4]
                        else:
                            zf2 = float(P2[:, 2].max())
                            face = T2[P2[T2][:, :, 2].min(1) > zf2 - 1e-4]
                        if len(face):
                            layers[role].append((P2, face, zf2))
                cache[lkey] = layers
            layers = cache[lkey]
            if not layers["board_mask"] or not layers["board_copper"]:
                return None
            win_r = max(pour_clamp * r, pad_ratio * r + 0.5)

            def _union_near(entries):
                parts = []
                for P2, F2, _z in entries:
                    tri = P2[F2]
                    near = (np.abs(tri[:, :, 0] - cx).min(1) < win_r + 0.5) & (
                        np.abs(tri[:, :, 1] - cy).min(1) < win_r + 0.5
                    )
                    for q3 in tri[near]:
                        q = [(float(x), float(y)) for x, y, _zq in q3]
                        w2 = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (
                            q[1][1] - q[0][1]
                        ) * (q[2][0] - q[0][0])
                        parts.append(m3d.CrossSection([q[::-1] if w2 < 0 else q]))
                if not parts:
                    return None
                return m3d.CrossSection.batch_boolean(parts, m3d.OpType.Add)

            mask_cs = _union_near(layers["board_mask"])
            copper_cs = _union_near(layers["board_copper"])
            if mask_cs is None or copper_cs is None:
                return None
            window = m3d.CrossSection.circle(win_r, 96).translate((cx, cy))
            open_cs = (((window - mask_cs) ^ copper_cs) - drill) ^ clamp_cs
            if open_cs.is_empty():
                return None
            # keep only the piece that RINGS this hole -- another pad's
            # opening inside the window is someone else's joint
            probe = m3d.CrossSection.circle(0.05 * r, 16).translate((cx + 1.02 * r, cy))
            keep = None
            for piece in open_cs.decompose():
                if not (piece ^ probe).is_empty():
                    keep = piece
                    break
            if keep is None or keep.is_empty():
                return None
            zs = [e[2] for e in layers["board_copper"]]
            zf2 = min(zs) if sd == "bottom" else max(zs)
            return ([np.asarray(x, float) for x in keep.to_polygons()], zf2)

        pad = None
        pad_prov = padnum_prov = SYNTH
        cand = [
            p
            for p in pads_by_term.get(h.terminal_number or "", ())
            if (p.side or "") == sd
        ]
        cand.sort(key=lambda p: p.prim_path != (h.pad_prim_path or ""))

        def _pad_mesh_z(p):
            """Return the pad mesh's exposed-face height, or None.

            The authored contour
            is 2D -- its height must still come from the copper geometry, or
            the foot pins at the substrate face, a full copper thickness
            inside the ring (the rung-2 measured-z fix, applied to rung 1).
            """
            pp = stage.GetPrimAtPath(p.prim_path)
            P, _T = _mesh_world(pp)
            if P is None or not len(P):
                return None
            return (
                float(P[:, 2].max())
                if (p.side or sd) == "top"
                else float(P[:, 2].min())
            )

        for p in cand:
            contours = _pad_from_contour(p)
            foot_z = fillet_z
            if contours is not None:
                pad_prov = SPEC
                mz = _pad_mesh_z(p)
                if mz is not None:
                    foot_z = mz
            else:
                got = _pad_from_mesh(p)
                if got is None:
                    continue
                contours, clamped, foot_z = got
                pad_prov = SYNTH if clamped else SPEC
            pad = BoardPad(
                pad_number=h.terminal_number or "?",
                contours=contours,
                side=sd,
                occupied=True,
                z=foot_z,
            )
            # PadRecord.terminal_number is non-None exactly when the provider
            # read an authored componentPinNumber/padNumber off the pad.
            padnum_prov = SPEC if p.terminal_number else SYNTH
            break
        if pad is None:
            got = _mask_opening_foot()
            if got is not None:
                mask_contours, mask_z = got
                pad = BoardPad(
                    pad_number=h.terminal_number or "?",
                    contours=mask_contours,
                    side=sd,
                    occupied=True,
                    z=mask_z,
                )
        if pad is None:
            # nothing pad-shaped anywhere: drill-derived annulus, the
            # shim's own pad_ratio stand-in.
            ann = (
                m3d.CrossSection.circle(pad_ratio * r, 96)
                - m3d.CrossSection.circle(r, 64)
            ).translate((cx, cy))
            pad = BoardPad(
                pad_number=h.terminal_number or "?",
                contours=[np.asarray(x, float) for x in ann.to_polygons()],
                side=sd,
                occupied=True,
                z=fillet_z,
            )

        # --- the pin, off the associated TerminalRecord when one exists
        t = terms.get(h.terminal_number or "")
        solderable = t.solderable if t is not None else None
        cross = t.cross_section if t is not None else None
        pin = Pin(
            number=h.terminal_number or "?",
            solderable=bool(solderable) if solderable is not None else True,
            cross_section=str(cross) if cross else "other",
        )

        hole = BoardHole(
            center_mm=(cx, cy),
            shape=shape,
            dims_mm=(dx, dy),
            side=h.side or "top",
            plated=bool(plated) if plated is not None else True,
            occupied=True,
            component=component_path,
            component_pin_number=h.terminal_number,
            z0=z0,
            z1=z1,
        )
        joints.append(
            ThtJoint(
                hole=hole,
                pad_top=(pad if sd == "top" else None),
                pad_bottom=(None if sd == "top" else pad),
                pin=pin,
                barrel_z=(bz0, bz1),
                prov={
                    "board_hole.holeCenterMm": center_prov,
                    "board_hole.holeShape": shape_prov,
                    "board_hole.holeDimsMm": dims_prov,
                    "board_hole.plated": plated_prov,
                    "board_hole.occupied": occ_prov,
                    "board_hole.component": assoc,
                    "board_hole.componentPinNumber": assoc,
                    "board_barrel": SYNTH,
                    "board_pad.padContour": pad_prov,
                    "board_pad.padNumber": padnum_prov,
                    # the pin NUMBER is only as good as the pin<->hole match, and
                    # that match is the association above
                    "pin.number": assoc,
                    "pin.solderable": SPEC if solderable is not None else SYNTH,
                    "pin.crossSection": SPEC if cross else SYNTH,
                    "component.processClass": pc_prov,
                    "component.side": side_prov,
                },
            )
        )

    def _num(j):
        try:
            return (0, int(j.pin.number))
        except (TypeError, ValueError):
            return (1, 0)

    joints.sort(key=lambda j: (_num(j), j.hole.center_mm[1], j.hole.center_mm[0]))
    return joints


def discover(
    stage,
    board_root,
    component_path,
    comp_V,
    catalog=None,
    cache=None,
    xform_cache=None,
):
    """Route one component's THT ingest through the catalog or raw-layers shim.

    Use the catalog when the board carries the schema.

    Returns ``(joints, source)`` with source "conformant" when a catalog was
    given and yielded joints for this component, else "raw_layers" (which may
    itself yield no joints -- a genuinely SMT part). `from_raw_layers` is
    untouched either way.
    """
    if catalog is not None and getattr(catalog, "holes", ()):
        joints = from_conformant(
            stage,
            catalog,
            component_path,
            comp_V,
            xform_cache=xform_cache,
            cache=cache,
            board_root=board_root,
        )
        if joints:
            return joints, "conformant"
    return from_raw_layers(
        stage, board_root, component_path, comp_V, xform_cache=xform_cache, cache=cache
    ), "raw_layers"


# --------------------------------------------------------------------- shim


def solder_side(comp_V, z0, z1):
    """Which face the joint's fillet is on: OPPOSITE the component body.

    A part inserted from the top has its body above the board and is soldered
    from below, so its fillet is on the BOTTOM face -- and vice versa. Getting
    this from `pcba:component:processClass` + placement side is the contract's
    job; here it is read off the geometry, which is unambiguous because a
    through-hole part's body is overwhelmingly on one side.
    """
    V = np.asarray(comp_V, float)
    return "bottom" if (V[:, 2] > z1).mean() > (V[:, 2] < z0).mean() else "top"


def from_raw_layers(
    stage,
    board_root,
    component_path,
    comp_V,
    window=1.6,
    lead_margin=0.05,
    xform_cache=None,
    cache=None,
    side=None,
    pad_ratio=1.8,
):
    """Synthesise contract fields from a pre-schema board's raw layers.

    Sources, none of which the contract asks a consumer to touch:

        board_hole / board_barrel   <- Board/Platings   (solid, so the void is
                                       ours to carve)
        board_pad.padContour        <- Board/Soldermask{Top,Bottom} opening,
                                       which IS the contract's definition of
                                       the pad. WHICH SIDE matters: the fillet
                                       sits on the face opposite the component
                                       body, so a top-inserted part is soldered
                                       from below and its pad is bottom-side.
        occupied / plated           <- GUESSED (see module docstring)

    `comp_V` is the component's world-space vertices, used only for the
    occupancy guess.

    Pass a dict as `cache` when scanning a whole board. The soldermask is ONE
    267k-triangle sheet and the platings and copper are thousands of prims;
    re-reading and re-transforming them per component makes a board scan
    quadratic in the board, not linear in the parts.
    """
    cache = {} if cache is None else cache
    import manifold3d as m3d
    from pxr import Usd, UsdGeom

    xc = xform_cache or UsdGeom.XformCache()
    pred = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

    def _tris(mesh):
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or ())
        idx = list(mesh.GetFaceVertexIndicesAttr().Get() or ())
        out, off = [], 0
        for c in counts:
            for k in range(1, c - 1):
                out.append((idx[off], idx[off + k], idx[off + k + 1]))
            off += c
        return np.asarray(out, np.int64).reshape(-1, 3)

    def _meshes(path):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return
        for pr in Usd.PrimRange(prim, pred):
            m = UsdGeom.Mesh(pr)
            if not m:
                continue
            pts = m.GetPointsAttr().Get()
            if not pts:
                continue
            M = xc.GetLocalToWorldTransform(pr)
            yield np.array([list(M.Transform(q)) for q in pts], float), _tris(m)

    lo, hi = comp_V.min(0), comp_V.max(0)

    def _barrels():
        """(cx, cy, r, z0, z1, |dx-dy|) for every THROUGH-hole, built once."""
        if "bar" in cache:
            return cache["bar"]
        rows = []
        for P, _T in cache.setdefault(
            "plat", [(p, t) for p, t in _meshes(f"{board_root}/Board/Platings")]
        ):
            c = P[:, :2].mean(0)
            e = P.max(0) - P.min(0)
            rows.append(
                (
                    c[0],
                    c[1],
                    0.5 * float(max(e[0], e[1])),
                    float(P[:, 2].min()),
                    float(P[:, 2].max()),
                    abs(float(e[0] - e[1])),
                )
            )
        B = np.asarray(rows, float).reshape(-1, 6)
        # THROUGH-holes only. Board/Platings also carries INNER-layer segments
        # (INNER1__INNER2, 0.32 mm tall against a 1.362 mm board). Those are not
        # through-holes, and matching one makes the joint WRONG rather than
        # absent: the board plane is then taken from that segment, so LCD2's pad
        # landed at z 0.841 -- mid-laminate, fillet buried inside the board.
        if len(B):
            span = B[:, 4] - B[:, 3]
            B = B[span > 0.9 * span.max()]
        cache["bar"] = B
        return B

    # BOARD EXTENT FIRST, then the side. Reading it from a cache the function
    # has not populated yet falls back to the COMPONENT's own z range, which
    # makes the above/below test degenerate to 0 > 0 and answer "top" for
    # everything -- so every top-inserted part was soldered on the wrong face.
    _B = _barrels()
    _z0, _z1 = (
        (float(_B[:, 3].min()), float(_B[:, 4].max()))
        if len(_B)
        else (float(lo[2]), float(hi[2]))
    )
    sd = side or solder_side(comp_V, _z0, _z1)
    layer = "Top" if sd == "top" else "Bottom"
    mask_layer = "SoldermaskTop" if sd == "top" else "SoldermaskBottom"

    def _top_index():
        """Copper prims as (bbox array, prepared top-face triangles).

        Built ONCE. The naive form re-scanned all 1659 top-copper prims for
        every joint and re-selected their top faces each time -- that is the
        quadratic term that made a board scan take ~20 minutes. Here the bboxes
        become one array so candidate selection is a single vectorised test, and
        each prim's top-face triangles are prepared once.
        """
        ikey = "idx_" + sd
        if ikey in cache:
            return cache[ikey]
        bbs, prepped = [], []
        for P, T in cache.setdefault(
            "cu_" + sd, list(_meshes(f"{board_root}/Board/{layer}"))
        ):
            # OUTER face of the copper: the top of top-side copper, the BOTTOM
            # of bottom-side copper. Taking the top face of bottom copper picks
            # the side buried against the laminate.
            if sd == "top":
                zt = float(P[:, 2].max())
                top = T[P[T][:, :, 2].min(1) > zt - 1e-4]
            else:
                zb = float(P[:, 2].min())
                top = T[P[T][:, :, 2].max(1) < zb + 1e-4]
            if not len(top):
                continue
            bbs.append((P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()))
            prepped.append((P, top))
        cache[ikey] = (np.asarray(bbs, float).reshape(-1, 4), prepped)
        return cache[ikey]

    def copper(cx, cy):
        """Top copper covering this hole, as a 2D region.

        WETTABLE AREA IS COPPER ^ MASK OPENING, not the opening alone. The
        contract defines padContour as the mask opening, which is right for SMT
        where the opening is pad-sized -- but a THT hole on this board sits in a
        6.7 mm^2 mask relief around a 1.2 mm^2 annular ring. Taking the opening
        alone makes the pad 5x too big and lets solder wet bare laminate. So we
        intersect, and the divergence from the literal spec wording is
        deliberate: the physical rule is that solder wets exposed COPPER.
        """
        bb, prepped = _top_index()
        if not len(bb):
            return None
        hit = np.nonzero(
            (bb[:, 0] - 0.1 <= cx)
            & (cx <= bb[:, 1] + 0.1)
            & (bb[:, 2] - 0.1 <= cy)
            & (cy <= bb[:, 3] + 0.1)
        )[0]
        parts = []
        for i in hit:
            P, top = prepped[i]
            # Union the TOP-FACE triangles directly, the same way the mask is
            # handled. Building a Manifold and slicing it looks cleaner and
            # silently yields nothing: these copper pours are not watertight, so
            # the Manifold comes back empty and the intersection becomes a no-op
            # that leaves the pad 5x too large.
            parts.extend(
                m3d.CrossSection([[(float(P[i0, 0]), float(P[i0, 1])) for i0 in t]])
                for t in top
            )
        if not parts:
            return None
        return m3d.CrossSection.batch_boolean(parts, m3d.OpType.Add)

    # --- the mask, cropped per joint. One board-wide sheet, so only its top
    # face is unioned locally; manifolding the whole thing would be absurd.
    mkey = "mask_" + sd
    if mkey in cache:
        mask_P, mask_T = cache[mkey]
    else:
        mask_P = mask_T = None
        for P, T in _meshes(f"{board_root}/Board/{mask_layer}"):
            if mask_P is None or len(P) > len(mask_P):
                mask_P, mask_T = P, T
        cache[mkey] = (mask_P, mask_T)
    mask_c = mask_P[mask_T].mean(1) if mask_P is not None else None
    # OUTER face of the mask sheet, matching the side. Using the top face of the
    # BOTTOM mask samples the surface buried against the laminate and returns
    # the wrong opening -- Q3's bottom pad came out 8.63 mm^2 against 1.2 on top.
    mask_zref = (
        (float(mask_P[:, 2].max()) if sd == "top" else float(mask_P[:, 2].min()))
        if mask_P is not None
        else 0.0
    )

    def opening(cx, cy, w):
        win = m3d.CrossSection.square((2 * w, 2 * w)).translate((cx - w, cy - w))
        if mask_P is None:
            return win
        near = (np.abs(mask_c[:, 0] - cx) < w) & (np.abs(mask_c[:, 1] - cy) < w)
        face = (
            mask_P[mask_T][:, :, 2].min(1) > mask_zref - 1e-4
            if sd == "top"
            else mask_P[mask_T][:, :, 2].max(1) < mask_zref + 1e-4
        )
        sel = near & face
        parts = [
            m3d.CrossSection([[(float(mask_P[i, 0]), float(mask_P[i, 1])) for i in t]])
            for t in mask_T[sel]
        ]
        if not parts:
            return win
        return win - m3d.CrossSection.batch_boolean(parts, m3d.OpType.Add)

    def own(cs, cx, cy):
        """Keep only the opening around this hole.

        A local window can clip a neighboring pad's opening. The contract
        avoids this entirely by making pads individually addressable prims.
        """
        pieces = cs.decompose()
        if len(pieces) <= 1:
            return cs
        for pc in pieces:
            x0, y0, x1, y1 = pc.bounds()
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return pc
        return cs

    B = _barrels()
    under = np.nonzero(
        (B[:, 0] >= lo[0])
        & (B[:, 0] <= hi[0])
        & (B[:, 1] >= lo[1])
        & (B[:, 1] <= hi[1])
    )[0]

    joints = []
    for bi in under:
        c = B[bi, :2]
        r = float(B[bi, 2])
        z0, z1 = float(B[bi, 3]), float(B[bi, 4])
        ediff = float(B[bi, 5])

        # OCCUPANCY INFERENCE. On a conformant board, association is authored
        # board-side via pcba:board:component / componentPinNumber rather than
        # derived from XY coincidence. Here it is inferred because the asset
        # carries nothing to read. Note the test must not look for
        # vertices INSIDE the barrel's z range: a pin is a box, so its only
        # vertices are its eight corners, none of which lie within the board.
        # The lead must SPAN THE BOARD, not merely sit near the hole. XY
        # proximity alone makes SMD parts look through-hole: an 0805's
        # termination can land within lead_margin of a neighbouring via and the
        # part would come back as a THT joint.
        #
        # "Penetrates below the top face" is not enough either, because a
        # BOTTOM-side part is entirely below the board and passes trivially.
        # A through-hole lead reaches above the top
        # face AND below the bottom one, which no surface-mount part on either
        # side does.
        near = comp_V[
            np.hypot(comp_V[:, 0] - c[0], comp_V[:, 1] - c[1]) <= r + lead_margin
        ]
        if not len(near):
            continue  # unoccupied -> a via, not a joint
        if not (near[:, 2].min() < z0 + 1e-6 and near[:, 2].max() > z1 - 1e-6):
            continue  # sits on one face; not through the hole

        cx, cy = float(c[0]), float(c[1])
        exposed = opening(cx, cy, window)
        cop = copper(cx, cy)
        if cop is not None and not cop.is_empty():
            exposed = exposed ^ cop
        # CAP THE ANNULAR RING. copper ^ mask localises the pad only where the
        # copper is an island. On a POUR it does not: when one side is ground
        # plane under a wide mask relief, the same hole can yield 1.207 mm^2 on
        # top and 9.673 mm^2 on the bottom -- an 8x footprint from identical
        # inputs. Per-pad addressable copper is hard to export for exactly this
        # reason (a pad's copper extent is often part of a plane or pour), so
        # there is nothing to read.
        #
        # pad_ratio is therefore a STATED stand-in for the annular ring, the way
        # blob_scale and skirt stand in for the pad on the SMT side. 1.8 is not
        # arbitrary: the top-side pads, where copper IS an island, measure an
        # equivalent radius of 0.752 mm on a 0.425 mm hole -- 1.77x. It caps
        # only; a pad already smaller than the cap is untouched.
        # Localise to THIS barrel first: own() is what discards a neighbouring
        # pad's opening caught by the window, and running the cap before it
        # compares against an area that includes those neighbours -- which
        # tripped the threshold on top-side pads that needed no capping.
        exposed = own(exposed, cx, cy)
        if pad_ratio:
            capped = exposed ^ m3d.CrossSection.circle(pad_ratio * r, 96).translate(
                (cx, cy)
            )
            # Only engage when the pad is IMPLAUSIBLY large. Applying the cap
            # unconditionally trims the corners off the top-side island pads,
            # which were already right: Q3's top pad went 1.207 -> 0.841 because
            # it is not a circle. A pad that already fits inside the cap is left
            # exactly as read.
            if capped.area() > 1e-9 and exposed.area() > 1.5 * capped.area():
                exposed = capped
        ann = exposed - m3d.CrossSection.circle(r, 64).translate((cx, cy))
        pad = BoardPad(
            pad_number="?",
            side=sd,
            occupied=True,
            z=(z1 if sd == "top" else z0),
            contours=[np.asarray(x, float) for x in ann.to_polygons()],
        )
        hole = BoardHole(
            center_mm=(cx, cy),
            # 5e-3 mm, not 1e-3: a 64-gon approximating a 0.85 dia circle has
            # ~1e-3 mm of facet slop between its x and y extents, so a tighter
            # test calls every round hole "other".
            shape="circular" if ediff < 5e-3 else "other",
            dims_mm=(2.0 * r, 2.0 * r),
            side="top",
            plated=True,
            occupied=True,
            component=component_path,
            component_pin_number=None,
            z0=z0,
            z1=z1,
        )
        joints.append(
            ThtJoint(
                hole=hole,
                pad_top=(pad if sd == "top" else None),
                pad_bottom=(None if sd == "top" else pad),
                pin=Pin(number="?", cross_section="other"),
                barrel_z=(z0, z1),
                prov={
                    "board_hole.holeCenterMm": SYNTH,
                    "board_hole.holeShape": SYNTH,
                    "board_hole.holeDimsMm": SYNTH,
                    "board_hole.plated": SYNTH,
                    "board_hole.occupied": SYNTH,
                    "board_hole.componentPinNumber": SYNTH,
                    "board_barrel": SYNTH,
                    "board_pad.padContour": SYNTH,
                    "board_pad.padNumber": SYNTH,
                    "pin.number": SYNTH,
                    "pin.crossSection": SYNTH,
                    "component.processClass": SYNTH,
                },
            )
        )
    joints.sort(key=lambda j: (j.hole.center_mm[1], j.hole.center_mm[0]))
    for i, j in enumerate(joints):
        pd = j.pad_top or j.pad_bottom
        j.pin.number = pd.pad_number = str(i + 1)
        j.hole.component_pin_number = str(i + 1)
    return joints
