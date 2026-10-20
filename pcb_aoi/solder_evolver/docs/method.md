# Wet triangles: closing the solder body for a native Evolver volume constraint

> Part of the `pcb_aoi.solder_evolver` module docs. ARCHIVAL: this text
> predates the graduation out of `scripts/surface_evolver/scratch/`, a dev
> tree that has since been deleted outright (2026-08-12; git history retains
> it). Module names were updated (boolean / emit / solve); paths in examples
> refer to the deleted layout and are kept as history, not instructions.

How to take an initial solder **blob**, cut it against the board and the component
metallization, and hand Surface Evolver a **single closed, consistently-oriented
surface** so that native `body ... volume` works exactly — replacing the
`content: c1: 0 c2: 0 c3: -x*y` divergence-theorem trick in
`backends/evolver.py`.

The output is a triangle soup partitioned by provenance:

| patch | comes from | Evolver tension |
|---|---|---|
| **meniscus** | the blob surface, outside all solids | `1` |
| **wet_board** | the board plane, inside the blob | `-cos(theta_board)` |
| **wet_metal / wet_body** | the component surface, inside the blob | `-cos(theta_comp)` |

**Scope.** The contact line is **PINNED**. No sliding contact lines, no per-patch
smooth-constraint fitting. This document is only about producing a correct closed
body plus its wetted patches.

Status: design study; the working prototype lived under
`scripts/surface_evolver/scratch/wet_triangles/` (deleted; see git history).
The graduated implementation is this package.

---

## 0. TL;DR recommendation

1. **Blob = dilated paste-aperture volume** (`aperture_offset`): union the joint's
   paste aperture bricks at stencil thickness, Minkowski-dilate by a radius large
   enough to merge the bricks, and let Evolver's `body volume` (set from
   `area x stencil_thickness`) do the volume enforcement. Physically motivated,
   generalizes to every footprint we have, no per-family shape library.
2. **Boolean = manifold3d** (`blob - board - metal - body`), not a hand-rolled
   marching intersection. It is the only option that *guarantees* the conforming
   shared edge chain, and it hands back per-triangle provenance for free via
   `run_original_id`.
3. **Seat the blob 3 um below the pad plane.** Exact coplanarity is the one
   degeneracy that reliably corrupts the contact chain, and a sub-micron-scale
   sink removes it entirely (measured, §5.1).
4. **Repair the asset meshes first** — weld, split into shells, fan-cap, re-wind.
   Every real part mesh in `pcb_usd_06` fails without this (§4).

**Validated end to end:** the emitted `.fe` was run through `/usr/bin/evolver`,
which reports `body[1].volume` matching the mesh volume to 8-9 significant
figures with **no `content:` clause**, and conserves it under `g 5` (§6).

---

## 1. The requirement, stated precisely

Evolver computes a body's volume as a surface integral over its facets. For that
to be the true enclosed volume the facet set must be a **closed oriented
2-manifold**: every directed half-edge appears exactly once, and its reverse
appears exactly once, in the opposite facet.

That means the meniscus and the wet patches cannot merely *look* like they meet
along the contact curve — they must **share the identical vertices and edges**.
One duplicated-but-slightly-different vertex on the contact line and the body
leaks: Evolver's volume becomes garbage and the constraint drives the surface
somewhere meaningless. This is the single hard requirement, and it is the axis on
which the boolean machinery must be judged.

Everything else (contact angle via `-cos(theta)` tension, pinning, refinement) is
downstream and easy once the body closes.

---

## 2. Blob representation — the options

The blob is only a **seed**: it sets the initial contact footprint and the
topology. Evolver's `body ... volume` enforces the volume afterwards, so the
blob's own volume does not have to be exact — only its *shape and reach* matter.
That observation kills the main argument for iterative volume matching (§7).

### 2a. Analytic sphere — the status quo (`sphere_contact_loop`)

`geometry/convex.py::sphere_contact_loop()` already intersects a sphere with a
mesh and chains the edge-keyed crossings into an ordered loop. It is the right
*idea* and the wrong *primitive*.

* **Does not generalize.** A sphere's footprint is a disc. Real joints are
  rectangular paste apertures, windowpaned tabs, gull-wing toes and BGA balls.
  Getting a sphere to cover a 9.7 x 6.0 mm D2PAK tab means a radius that also
  swallows the neighbouring leads.
* **Measured symptom.** On the D2PAK gull leads the sphere blob produces a fine
  body, but on the QFN perimeter leads it fragments into **2–4 disconnected
  components** because a sphere sized to the paste volume simply does not reach
  along the lead (§5, table).
* It only returns *one* loop and returns `None` for anything else — the multi-loop,
  multi-patch reality of a real joint is out of reach.

Verdict: **keep it as a special case for BGA balls if you like; reject as the
general path.**

### 2b. Ellipsoid fitted to the aperture bbox

Cheap upgrade over the sphere, matches the aperture aspect ratio. Measured: better
than sphere on the D2PAK (always 1 component), but on the BGA/QFN it over-reaches
laterally and produced genus 7–10 bodies (it swallows adjacent geometry). It is
still a shape prior with no physical basis. **Reject.**

### 2c. Implicit / SDF blob evaluated on a grid

Build an SDF for the blob, marching-cubes it, boolean the result. Attractive
because it composes with the existing `min-of-max` hull SDFs and because
manifold3d has `Manifold.level_set`. But:

* it converts a clean analytic boundary into a *sampled* one, so the wet patch
  resolution is set by the grid, not by the geometry;
* it does nothing for the conforming-edge problem — you still need a boolean
  afterwards;
* on 0.4 mm BGA balls a grid fine enough not to alias the ball is expensive per
  joint, and we have thousands of joints.

Verdict: **reject as the blob representation.** (Useful separately if we ever want
a smooth *keep-out* field — see `analytic_representation_alternatives.md` §1.)

### 2d. Paste-aperture prism — the physical volume, unslumped

The printed deposit: each aperture rect extruded to the stencil thickness. This is
literally the solder that exists before reflow, and it gives the joint volume for
free (`Joint.paste_volume()` already computes it).

Measured problem: **a windowpaned tab prints as N separate bricks.** D2PAK joint 0
is 15 apertures on a 0.84–0.92 mm grid, and the raw prism blob yields **15
disconnected solder bodies** (genus −14). Real reflow merges them. So the raw
prism is physically right at t=0 and topologically wrong at t=reflow.

### 2e. Paste-aperture prism, dilated — **recommended**

Take 2d and Minkowski-dilate by a radius `r`. This models slump/coalescence, and
the dilation radius is the one knob:

| `r` on D2PAK joint 0 (15 apertures, 0.84–0.92 mm gaps) | result |
|---|---|
| 0.25 mm | 15 components, genus −14 |
| 0.50 mm | 1 component but **genus 8** (holes between panes) |
| 0.90 mm | **1 component, genus 0, 586 tris, chains all clean** |

Rule that falls out: **`r` >= the largest inter-aperture gap inside a joint** (not
half of it — half merges but leaves holes). That number is computable from the
aperture rects, so it is not a magic constant.

Why this is the right default:

* **Physically motivated.** The blob *is* the printed paste, dilated by slump.
  Volume comes from the same source, so `body volume` and the blob agree by
  construction.
* **Generalizes with no shape library.** It worked unchanged on D2PAK power tab,
  D2PAK gull leads, BGA balls and QFN perimeter leads.
* **Robust wet footprints.** It is the only candidate that gave 1 component /
  genus 0 (or the correct genus 1 ring) with zero contact-chain defects across
  every clean joint we tested.
* **Cheap.** ~46 ms per joint including both booleans, against a 144k-triangle
  component solid (§7).

### 2f. Convex hull of (paste brick + wettable metal band), dilated

`hull_offset` in the prototype. Merges windowpanes automatically and reaches the
lead. But the hull is convex, so where a lead pokes through it the solder gets
**split in two** — measured on D2PAK joints 1 and 3: 2 components, genus −1. Also
it depends on the unreliable `solder_contact_meshes` tag. **Reject as default;
keep as a fallback when a joint's apertures are too sparse to merge.**

### 2g. Swept / offset of the pad copper

Not evaluated: we do not currently ingest the copper layer, only the paste mask.
Worth revisiting only if paste apertures turn out to be missing for some part
families. Aperture-driven onboarding already established paste as the reliable
signal.

---

## 3. Boolean machinery — hand-rolled vs. library

### 3a. Extending `sphere_contact_loop` (hand-rolled marching intersection)

The existing code is careful in exactly the right way: crossing points are keyed
by the mesh **edge** `(min(a,b), max(a,b))`, so adjacent triangles reuse the same
point and the segments chain. That gets you a conforming *curve*.

What it does **not** get you, and what wet triangles need:

1. **Retriangulating the cut triangles.** A triangle straddling the blob boundary
   must be split into an inside part and an outside part, both using the same
   crossing vertices. That is a constrained triangulation, per triangle, with all
   the usual degeneracies (crossing through a vertex, crossing along an edge,
   two crossings on the same edge).
2. **The blob side of the cut**, retriangulated against the *component's* edges
   this time — the reverse problem, and the two retriangulations must agree.
3. **Three-way junctions.** Board, component and meniscus meet at triple points.
   Every real joint we measured has them (D2PAK joint 1: 2 triple points). A
   pairwise curve-chaser has no concept of these.
4. **Multiple solids at once** (board + metal + body, several shells each).
5. **Failure is silent.** `sphere_contact_loop` returns `None` when the result
   isn't a single clean loop — for real joints the answer *isn't* a single loop,
   so the honest generalization has to succeed at all of the above, not bail.

Getting all of this right is writing a mesh boolean kernel. It is weeks of work
whose failure mode is exactly the leak we cannot tolerate.

### 3b. manifold3d — **recommended**

`manifold3d` (Apache-2.0, pip, 5 ms-scale on our sizes) computes
`blob - board - metal - body` and by construction returns a **single indexed
mesh**: the meniscus and wet triangles literally index the same vertex array
along the contact curve. Conformity is not something you check afterwards; it is
structural. Manifold's whole design invariant is that its output is a valid
oriented 2-manifold.

Two properties that make it more than "just a boolean":

* **`run_original_id` / `run_index`** give per-triangle provenance runs on the
  output — so the meniscus/wet_board/wet_metal labelling is *exact*, not a
  nearest-surface heuristic. (It survives `batch_boolean` unions too, which we
  checked.)
* **`Manifold.status()` / `genus()` / `decompose()`** are a ready-made validation
  surface (§5).

trimesh 4.12 booleans dispatch to manifold3d anyway, so use manifold3d directly
and keep the provenance.

**Cost of the recommendation:** one new dependency, `manifold3d==3.5.2`
(installed into the repo venv during this investigation). Apache-2.0, no
system libs, imports cleanly alongside `pxr` and `coacd`.

Rejected alternatives: **CGAL/pymesh** (heavy, GPL-ish, build pain);
**blender-as-a-service** (already how trimesh's other backend works — process
launch per joint, unacceptable at thousands of joints); **igl** (boolean is a
CGAL wrapper, same problem).

---

## 4. Asset mesh reality — what the real parts do to you

Measured on the reference board export ($SOLDER_BOARD_DIR). **Every** part mesh
needs repair before any boolean will touch it. Four distinct problems, in order:

| # | Problem | Evidence (D2PAK) | Fix |
|---|---|---|---|
| 1 | **Unwelded** — USD arrives per-face, vertices duplicated | metal: 70 V / 52 F, **70 boundary edges**; body: 380 V / 292 F, **380 boundary edges** | weld on a 1 um grid → metal 38 V, body 164 V |
| 2 | **Multi-shell** — one prim is several disjoint solids | `pkg_body` splits into **4 shells**: the tab and 3 leads | `trimesh.split(only_watertight=False)`, repair each |
| 3 | **Open** — missing interior faces (the `pkg_lead` hole) | still **28 boundary edges** after welding | fan-cap each boundary loop (the trick `sphere_contact_loop` already uses) |
| 4 | **Inconsistent winding** | metal has **12 same-direction half-edges** after capping | `trimesh.repair.fix_winding` + `fix_inversion` **per shell** |

Only after all four does `manifold3d.Manifold(...)` accept the mesh. Skipping any
one of them yields `Error.NotManifold` and an *empty* manifold — which silently
turns into "the component isn't there" and a solder body that ignores the lead.
**Fail loudly on an empty manifold.** The prototype prints
`!! metal: manifold REJECTED the asset mesh (empty)`.

Two caveats on the repair:

* **Fan-capping an open shell can invent volume.** D2PAK's `pkg_lead` capped into
  a 408 mm^3 solid spanning nearly the whole part bounding box. Harmless here (it
  doesn't overlap the joints we solved), but it is a real hazard: if a cap-created
  solid overlaps a joint, it eats solder. Mitigation: sanity-check
  `capped_volume / convex_hull_volume` and warn.
* Repairing the 90k-vertex BGA body costs **4.6 s**. Do it **once per prototype**
  and cache — it is instance-invariant, exactly like the CoACD hull cache in
  `ingest/convex_decomp.py`.

---

## 5. Validation and degenerate cases

### 5.1 The validation battery

Implemented in `boolean.py::check_body()`:

1. **Closed** — every directed half-edge has its reverse present.
   (`unmatched_halfedges == 0`)
2. **Consistently oriented** — no directed half-edge appears twice.
   (`dup_dir == 0`)
3. **Volume** — numpy divergence-theorem signed volume > 0 and equal to
   `Manifold.volume()`.
4. **Contact-chain conformity** — for each label *pair*, the edges whose two
   incident triangles carry different labels must form **closed loops** (every
   vertex degree exactly 2), except at genuine **triple points** (vertices
   incident to >= 3 distinct labels), which are legitimate. Anything else is
   reported as `UNEXPLAINED` and is a real defect.
5. **Patch connectivity** — number of connected components per label, and
   `Manifold.decompose()` for the body as a whole.
6. **Genus** — `genus 0` is a plain fillet; `genus 1` is a **ring around a lead or
   ball**, which is correct and desirable (it is exactly the topology commit
   `933a39f` had to seed by hand); `genus < 0` means multiple shells.

Criterion 4 is the one that actually earns its keep — it is what catches a
coplanarity defect that criteria 1–3 sail straight past.

### 5.2 Measured results

D2PAK, all 4 joints x 5 blob candidates, 3 um seat
(`sweep.py --family D2PAK`):

```
joint             blob   tris genus closed orient ncomp  volerr% badchain  patches
    0           sphere    516     0   True   True     1     0.00        0  meniscus=312, wet_board=98, wet_body=106
    0        ellipsoid   1206     0   True   True     1     0.00        0  meniscus=1084, wet_board=122
    0   aperture_prism    180   -14   True   True    15     0.00        0  meniscus=150, wet_board=30
    0  aperture_offset   2416   -14   True   True    15     0.00        0  meniscus=1806, ... wet_metal=6
    0      hull_offset    252     0   True   True     1     0.00        0  meniscus=204, wet_board=34, wet_body=14
    1           sphere   1520     1   True   True     1     0.00        0  meniscus=1306, wet_board=81, wet_body=133
    1        ellipsoid   1208     1   True   True     1     0.00        0  meniscus=1000, wet_board=128, wet_body=80
    1   aperture_prism     40     1   True   True     1     0.00        0  meniscus=16, wet_board=8, wet_body=16
    1  aperture_offset    204     1   True   True     1     0.00        0  meniscus=154, wet_board=24, wet_body=26
    1      hull_offset    244    -1   True   True     2     0.00        0  meniscus=182, wet_board=31, wet_body=31
    2           sphere   1622     0   True   True     1     0.00        0  meniscus=1524, wet_board=98      <- NO component contact
    2  aperture_offset    156     0   True   True     1     0.00        0  meniscus=138, wet_board=18       <- NO component contact
    3  aperture_offset    204     1   True   True     1     0.00        0  meniscus=154, wet_board=24, wet_body=26
```

BGA_0084, per-ball joints (clustering gap 0.2 mm), 4 joints x 5 blobs: **every
single case** closed, oriented, 1 component, genus 1 (the ring around the ball),
exact volume, zero chain defects.

Headline: **closure and orientation were never the failure mode** once the asset
repair (§4) and the seat epsilon (§5.3) are in place — manifold3d simply does not
emit a non-closed result. What *does* vary between blob candidates is **topology
sanity** (component count, genus), and `aperture_offset` with a gap-sized dilation
is the only candidate that is sane everywhere.

### 5.3 Degenerate case: coplanar blob / board face

**The one that bites.** `aperture_prism` places the blob's bottom face exactly on
the pad plane. Measured, D2PAK joint 1:

```
seat = 0      wet_board: 3 connected components; chain meniscus|wet_board: 1 UNEXPLAINED vertex
seat = 3 um   wet_board: 1 component; all three chains clean closed loops
```

The body was *closed and oriented in both cases* — only criterion 4 caught it.

**Rule: sink the blob a few microns below the pad plane** so every cut is
transversal. 3 um is ~0.002% of a 0.13 mm stencil thickness — geometrically
irrelevant, numerically decisive. The same applies to any blob face that lands on
an axis-aligned component face.

Coplanarity between two *solids* (a package bottom seated exactly on the board
plane) is a separate case; we tried both raising the board plane and unioning the
obstacles before the difference (`--union-first`) and **neither helped**. That one
resolves itself once joint clustering is right (§5.5) — the residual defects we
saw all came from an over-merged joint, not from the coplanarity.

### 5.4 Degenerate case: blob doesn't reach the component

**D2PAK joint 2 is a live example.** Its lead geometry has bbox z ∈ [3.96, 4.50]
while `pad_z = 1.40` — the middle lead never comes down to the board. Every blob
candidate produced `meniscus + wet_board` and **no component patch at all**.

The result is still a perfectly valid closed body — it is just a puddle of solder
on the pad attached to nothing. Silently solving it would produce a joint that
looks fine and means nothing.

**Rule: require a non-empty component wet patch; otherwise flag the joint.**
Either the part is genuinely bottom-terminated with no climbable metal (fall back
to the flat squeeze-film path `prong_mesh()` already detects) or the asset is
wrong. Do not solve it silently.

### 5.5 Degenerate case: blob touching multiple leads / wet patch splitting

Both are real and both trace back to **joint clustering, not the boolean**.

`load_board_component(..., gap=1.0)` merged all 84 BGA apertures into **2
joints of 42 balls each**, and all 33 QFN apertures into **1 joint**. Solving
those over-merged "joints" gives exactly the pathology you would expect:

```
QFN_32, all 33 apertures as one joint, aperture_offset:
  n_components=3, meniscus 11 components, wet_board 3, wet_body 36,
  6 UNEXPLAINED chain vertices
```

Drop the clustering gap below the joint pitch (`gap=0.2`) and the same parts
resolve into 84 / 15 clean single joints that solve perfectly (§5.2).

**Rule: the aperture clustering gap must be smaller than the joint pitch.** The
current default of 1.0 mm is wrong for anything finer than 1 mm pitch, which is
most fine-pitch parts. Derive it from the aperture spacing histogram per part, or
expose it per family.

Genuine bridging (one blob deliberately spanning two leads, a *defect* we want to
synthesize) is then the same machinery with the clustering deliberately merged —
and the prototype shows it produces a valid closed body with `wet_body` in several
components, which Evolver can still take. The multi-component *solder body* case
(`n_components > 1`) is the one to reject: emit one Evolver body per solder
component, or reject the blob.

### 5.6 Note: wet-patch triangle counts

The BGA ball's wet patch came out at **1556 triangles** against **133** for the
meniscus. The wet patch inherits the component mesh's tessellation, which was
authored for rendering, not for a solver. The wet patches are `fixed` so they do
not cost solver iterations, but they do cost file size and `total_area`
evaluation. Decimate the component solid (or the wet patch after the cut) before
emitting. Not investigated further.

---

## 6. End-to-end proof: Evolver's native volume

`emit.py` writes the labelled body as a datafile with a plain

```
bodies
1 1 2 3 ... N  volume V
```

and **no `content:` clause**, then runs `/usr/bin/evolver -x`. Results:

| case | mesh volume (numpy) | Evolver `body[1].volume` | after `g 5` | energy |
|---|---|---|---|---|
| D2PAK joint 1, gull lead | 0.409500154 | 0.409500122872 | 0.409500153663 | −0.7286 → −0.7432 |
| D2PAK joint 0, power tab (dil 0.9) | 3.39065948 | 3.39065938018 | 3.39065948001 | −29.766 → −29.781 |
| BGA_0084 ball joint | 0.0262513287 | 0.0262513258426 | 0.0262513284563 | 0.02629 → 0.02193 |

Agreement to 8–9 significant figures, volume conserved under gradient descent,
energy decreasing. **The divergence trick is not needed.** That is the deliverable
this whole design exists to enable.

(The emitter pins the contact line by marking every wet-patch vertex and edge
`fixed`, and gives wet facets `tension -cos(theta)`; the meniscus keeps tension 1.
That is the whole contact-angle model under the pinned-contact-line scope.)

---

## 7. Performance / scaling

Measured on the BGA_0084 (the worst asset: 90,792-vertex body mesh):

| step | cost | frequency |
|---|---|---|
| solidify + `Manifold(...)` of the component | **4.6 s** | **once per prototype**, cache it |
| blob construction + both booleans + evaluation | **46 ms** | per joint |
| iterative volume matching (24-step bisection) | ~1.1 s | **skip it** |

The bisection exists only in the prototype so the tables could report an exact
volume. In production the blob's volume does not need to be right — Evolver's
`body volume` target, computed from `aperture_area x stencil_thickness`, enforces
it. **Drop the bisection and the per-joint cost is 46 ms**, which is fine for the
thousands-of-components target in `docs/scaling.md`.

---

## 8. Recommended pipeline

```
per prototype (cached, like the CoACD hull cache):
    USD meshes -> weld -> split shells -> fan-cap -> fix winding/inversion
               -> manifold3d.Manifold           [FAIL LOUDLY if empty]
               -> decimate                       [not yet built]

per joint:
    paste apertures (gap < pitch!) -> union of bricks at stencil thickness
        -> Minkowski dilate by r >= max intra-joint aperture gap
        -> sink 3 um below pad_z
    body = blob - board_halfspace - component_solid        [manifold3d]
    labels = run_original_id                                [exact provenance]

validate:
    closed / oriented / volume > 0
    contact chains are closed loops modulo triple points
    exactly one solder component
    component wet patch non-empty                          [else flag the joint]

emit:
    vertices / edges / facets, wet facets tension -cos(theta) + fixed,
    meniscus tension 1, one `body ... volume V`  -- no content: clause
```

---

## 9. Rejected alternatives, in one place

* **Sphere blob (status quo).** Footprint is a disc; fragments on real leads;
  `sphere_contact_loop` returns `None` for everything that isn't a single loop.
* **Ellipsoid blob.** Over-reaches laterally; genus 7–10 on BGA/QFN; no physical
  basis.
* **Implicit/SDF blob + marching cubes.** Resolution-limited wet patches, still
  needs a boolean afterwards, too expensive per joint at BGA-ball scale.
* **Raw paste prism (undilated).** 15 disconnected bodies on a windowpaned tab.
* **Convex hull of paste + metal.** Splits in two where a lead pokes through;
  depends on the unreliable `solder_contact_meshes` tag.
* **Hand-rolled marching boolean (extending `sphere_contact_loop`).** Would have
  to solve constrained retriangulation on both sides plus triple points plus
  multi-solid; that is a boolean kernel, and its failure mode is a silent leak.
* **CGAL / pymesh / igl / blender-subprocess booleans.** Weight, licensing, or a
  process launch per joint.
* **Union obstacles before subtracting (`--union-first`).** Tried; no measurable
  improvement over the sequential difference. Kept as a flag, not the default.

---

## 10. Open questions for the user

1. **Is `manifold3d` an acceptable dependency?** It is Apache-2.0, pure pip, and
   I installed 3.5.2 into the shared repo venv to run these experiments. If not,
   the recommendation changes materially — there is no good hand-rolled fallback.
2. **Where does the dilation radius come from?** I recommend
   `r = max intra-joint aperture gap`, derived per joint. Do you want it derived,
   or a per-family constant, or a defect knob (small `r` = insufficient/uncoalesced
   paste, large `r` = excess)?
3. **Aperture clustering gap.** The current 1.0 mm default over-merges every
   fine-pitch part (BGA 84 apertures -> 2 joints). Should I derive it from the
   aperture spacing histogram, or is per-family config acceptable? This is in
   `ingest/board_component.py` and outside the scope I was given, but it is the
   dominant source of degenerate results.
4. **Decimation of wet patches.** BGA wet patch = 1556 triangles vs 133 meniscus.
   Decimate the component solid once per prototype (changes the geometry slightly)
   or decimate the wet patch after the cut (must preserve the contact chain
   exactly)? I'd take the former; it is far simpler and the error is ~mesh
   tolerance.
5. **`CAP_SMD_0603` loaded 35 joints from 118 apertures** for what should be a
   two-terminal part. That looks like an instance/footprint association bug in
   `load_board_component` (it associates apertures spatially under a footprint
   that appears too large). Want me to chase it?
6. **Multi-component solder bodies** — reject the blob, or emit one Evolver body
   per component? Relevant to the bridging defect.

---

## 11. What was validated vs. what is assumption

**Validated by running code on real `pcb_usd_06` geometry:**

* Asset repair pipeline (§4) — all four failures observed and fixed, D2PAK / QFN /
  BGA.
* manifold3d produces closed, consistently-oriented output with exact
  per-triangle provenance, on 4 D2PAK joints x 5 blobs and 4 BGA joints x 5 blobs
  and the QFN.
* Contact chains are closed loops modulo triple points, on every clean joint.
* The coplanarity defect and its 3 um fix (§5.3), measured both ways.
* "Blob doesn't reach the component" occurs on a real part (D2PAK joint 2).
* Over-merged joints produce multi-component / pinched bodies; tight clustering
  fixes it (§5.5).
* Dilation radius vs. windowpane merge (§2e), three radii measured.
* **Evolver accepts the body and reports the right native volume** (§6), three
  cases.
* Timings (§7).

**Still assumption / not built:**

* Wet-patch decimation — identified as needed, not implemented or measured.
* Behaviour under Evolver **refinement** (`r`) of the wet patches: they are
  `fixed`, so they should not move, but I only ran `g 5`, never `r`.
* Behaviour on **posed/defective** components (tombstone, shift, sideflip).
  The boolean should not care — the component solid is just transformed — but
  untested.
* Whether the fan-cap "invented volume" hazard (§4) ever actually eats a joint on
  a real board. It did not in these tests.
* Anything beyond D2PAK / BGA_0084 / QFN_32 / a peek at CAP_SMD_0603 — 4 of 45
  families.
* Contact-angle *correctness* (does the pinned wet-patch tension reproduce the
  intended theta). `validation/contact_angle.py` and `verify.py` exist for this;
  not exercised here.

---

## 12. Prototype code

`scripts/surface_evolver/scratch/wet_triangles/` — throwaway, not importable from
`src/`. `out/` is gitignored.

```bash
V=.venv/bin/python
S=scripts/surface_evolver/scratch/wet_triangles

# 1. freeze a real component + its paste joints (needs pxr)
$V $S/dump_joint.py --family D2PAK
$V $S/dump_joint.py --family BGA_0084 --gap 0.2 --tag _g02

# 2. one blob candidate, one joint, full report (+ labelled OBJ)
$V $S/boolean.py --family D2PAK --joint 1 --blob aperture_offset --obj /tmp/j1.obj

# 3. sweep every blob x every joint
$V $S/sweep.py --family D2PAK

# 4. emit a .fe with a NATIVE body volume and run the real evolver
cd $S && $V emit.py --family D2PAK --joint 1 --blob aperture_offset --run
```

## 13. Looking at it

Use the fill-sweep CLI to export one orbitable USD per fill and inspect the
results in usdview or another USD tool:

```bash
python -m pcb_aoi.solder_evolver.solve \
  --capture-dir /tmp/solder-evolver-captures \
  <other solve arguments>
```

The exported stages preserve the geometry needed to inspect the boolean
output and convergence sequence. Automated topology and solver checks live in
`tests/unittests/solver/solder_evolver/test_*.py`.
