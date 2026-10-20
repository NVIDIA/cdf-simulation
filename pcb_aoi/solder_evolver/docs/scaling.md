# Scaling to a full board (thousands of components)

> Part of the `pcb_aoi.solder_evolver` module docs. ARCHIVAL: this text
> predates the graduation out of `scripts/surface_evolver/scratch/`, a dev
> tree that has since been deleted outright (2026-08-12; git history retains
> it). Module names were updated (boolean / emit / solve); paths in examples
> refer to the deleted layout and are kept as history, not instructions.

## The problem

The reference board export (355 placed components) has **355 component
instances** and would be **thousands** on a large design. A naive pipeline that
solves the solder for every joint is O(instances) Surface Evolver runs — hundreds
to tens of thousands of solves per board. Intractable, and wasteful.

## The one insight that makes it tractable

**Solder geometry is a property of the FOOTPRINT, not the instance.** The
equilibrium solder shape at a joint depends only on the local pads/leads/paste
geometry, the wetting parameters, the volume, and gravity direction — all
identical across every instance of the same footprint on the same board side.

So the number of *unique solves* is bounded by

    distinct (footprint × side × defect-class × defect-params)

which grows **sublinearly** with board size and is effectively constant: a board
of 5000 passives still has ~40–100 distinct footprints. Everything else is a
rigid transform + a USD write.

Measured on the real board manifest:

| | count |
|---|---|
| component instances | 355 |
| distinct (family, side) groups | **45** |
| unique geometry prototypes | 40 |
| reuse factor (avg) | ~8× (top footprint: **52×**) |

The solve cost is therefore **decoupled from board size**. A 5000-component board
costs about the same to solve as a 300-component one.

## What the board asset already gives us

`<board>.components.json` (the board asset's component sidecar) is built for exactly this:

- `components[]` — one entry per **prototype** (footprint geometry): `family`,
  `source` (payload `.usdc`), `prototype_path` (USD `_Prototypes/...`),
  `instance_paths`, `body_meshes`, `solder_contact_meshes`.
- `component_instances[]` — 355 placed instances: `refdes`, `family`, `type`,
  `side` (Top/Bottom), `solder_contact_prims` (the leads/pads = joints),
  `solder_paste_prims`, `solder_paste_usdc_paths`.
- `solder_paste_layers` / `solder_paste_sources` — the printed-paste geometry
  that sets the nominal solder **volume** per footprint.

USD itself is already instanced (`_Prototypes` + instanceable references), so
prototypes are first-class.

## Pipeline (five stages, only stage 3 is expensive)

```
1. INGEST      manifest + prototype payloads
                 -> group instances by (family, side); read each prototype's
                    pads/leads/paste geometry ONCE.

2. FIT         per prototype: fit analytic SDF primitives (RoundBox/Cylinder)
                 to the pads/leads/body. Cache per prototype. O(footprints).

3. SOLVE       per (footprint, side, GOOD): one Surface Evolver solve -> local
   (cache)      solder mesh in footprint-local frame. Cache it.
                 per (footprint, side, DEFECT-class): solve on demand, only for
                 footprints that actually receive that defect. O(footprints x few).

4. ASSIGN      defect-injection pass: pick joints to defect (seeded RNG, target
                 rate + class distribution). Per-joint granularity.

5. PLACE       per instance/joint: look up the cached local solder mesh (good or
                 assigned defect), apply the instance world transform, emit into
                 an instanced USD result layer. O(instances), pure geometry.
```

Stages 1,2,4,5 are seconds even at thousands of instances. Stage 3 is the only
solver cost, and it is bounded by footprints, cached, and embarrassingly
parallel.

## The three reuse layers (the tractability levers)

### A. Component collider models — reuse per prototype
Fit the analytic SDF colliders (our `RoundBox`/`Cylinder`) on the **prototype**
payload, once per family. Every instance references the same fitted primitives
(transformed). Never re-fit per instance.

### B. "Good" solder meshes — reuse across all non-defect joints
Realistic defect rates are low (≪5%). Solve the **good** joint once per
(footprint, side); place a rigid-transformed copy at every non-defect instance.
This is the overwhelming majority of joints, at ~zero marginal cost.

### C. Defect solder meshes — reuse per (footprint, defect-class)
A defect solve is also footprint-local, so it caches the same way: solve
`insufficient` for PASSIVE_0805 once, reuse at every 0805 chosen to be
insufficient. With ~6 defect classes and a small defected subset, this is a
small, bounded number of extra solves. Exceptions that are genuinely
per-*site* (bridge, which depends on the neighbor spacing) are solved per
occurrence — but they are rare.

## Content-addressed solve cache

    key   = hash(prototype_id, side, wetting_params, gravity_dir,
                 defect_class, defect_params, solver_version)
    value = local solder mesh (footprint frame) + metrics

- On-disk, shared **across boards**: the same footprints recur board-to-board, so
  amortized solve cost → ~0 for common footprints after the first board.
- Deterministic keys → reproducible datasets; cache invalidates cleanly on a
  solver-version or parameter bump.

## Instanced output (keep the result USD small)

Do **not** write N full solder meshes. Author the good solder for each footprint
once under a `/SolderPrototypes` scope and **instance** it at every good joint via
USD instanceable references — output size O(footprints + defects), not
O(instances). Defect joints get their own (small count of) unique meshes. This
keeps the result layer light and fast for the SDG renderer to consume.

## Defect injection (random, reproducible)

A seeded assignment pass over joints:
- target defect **rate** (e.g. 1–3%) and a **class distribution**
  (insufficient/excess/bridge/missing/…), possibly per family (BGAs bridge,
  passives tombstone, etc.).
- per-**joint** granularity (one terminal of a chip can be insufficient while the
  other is good).
- records ground-truth label per joint → the training annotation.
- pose defects (tombstone/offset/missing-component) stay in the SDG
  component-transform pipeline; the solver owns the solder-morphology defects.

## Cost model

Let N = instances, K = distinct footprints (~const), D = defected joints (≪ N),
C = defect classes.

- solves ≈ K (good) + min(D, K·C) (defects)  →  **O(K)**, ~tens–hundreds, cached.
- fit ≈ O(K); assign ≈ O(N); place + author ≈ O(N) transforms/writes.

A full board's solver work is a few minutes cold, seconds warm (cache hit),
independent of N. Unique solves are independent → parallelize across cores.

## Edge cases / honesty

- **Locality assumption**: reuse holds because the solder is local to its joint.
  It breaks only for defects that couple neighbors — **bridge** (needs the real
  neighbor spacing) — handled as per-site solves (rare).
- **Board warp / non-flat**: assume rigid board; gravity per side. Local joint
  geometry is unaffected.
- **BGA/array parts**: an array of identical balls is ONE ball solve, instanced
  across the array (+ a few defect balls). High-pin parts are not expensive.
- **Paste variation**: volume comes from the footprint's paste aperture (manifest)
  — a per-footprint constant, so it does not break reuse.
- **Odd/unfittable footprints**: fall back to a mesh-obstacle or a generic
  primitive; flag low-confidence fits for review.

## Milestones

1. **Manifest ingest + grouping** — read `components.json`, group instances by
   (family, side), resolve prototype payloads. (No solver.)
2. **Prototype fitter** — fit SDF colliders per prototype; cache. Validate on the
   40 real prototypes.
3. **Solve cache** — content-addressed good-solder solve per (footprint, side),
   on-disk, reused across boards.
4. **Placement + instanced USD authoring** — transform cached meshes to every
   instance; emit an instanced result layer; render one full board (all good).
5. **Defect injection** — seeded per-joint assignment + per-(footprint,class)
   defect solves + labels.
6. **Scale + parallelize** — batch unique solves across cores; demonstrate a
   thousands-component board with a defect budget in minutes.
   *Done (fan-out):* `SurfaceEvolverSolver.generate()` solves one representative
   per distinct footprint across a `ProcessPoolExecutor` — each solve is its own
   Evolver subprocess, so the unique solves run in parallel.
   `solver_options["workers"]` is the TOTAL core budget (0 = auto, the machine
   minus two): a fork-inherited semaphore holds every Evolver solve — footprint
   process or joint thread — to one slot, so a positive value is a hard "use at
   most X cores" cap and the budget is never exceeded by the two fan-out levels
   combining. The semaphore is work-conserving (slots released by finished
   footprints flow to whoever still solves), unlike the static budget/groups
   split that was tried and measured slower. Artifact ordering is a
   deterministic pass over the footprint groups, independent of completion
   order.
   *Done (intra-footprint dedup):* within a footprint, `solve_footprint` keys
   joints by their translation-invariant local footprint and solves one
   representative per key, translating the result to its twins — the "array of
   identical balls is ONE solve" case above, now real. `solver_options
   ["joint_dedup"]` (default on). The key is the CARVED BLOB, not the paste
   footprint: cut every joint (cheap next to the solve), recentre and hash the
   cut geometry, and reuse a solve only when the carved blobs match. Keying the
   paste alone is WRONG — a chip's two ends share a paste rect but the component
   body is subtracted from opposite sides, so the blobs are mirror images; a
   paste key merged them and placed one end's fillet backwards. Measured on the
   CAD-exported board: output-joint count unchanged (330), a 2-pad chip and a resistor
   network now match the per-joint solve exactly (both mirror ends solved), and
   the whole board runs 23.6 → 14.5 min at 30 workers (the buggy paste-key was
   7.6 min only because it under-solved the mirror pairs).
   *Done (joint-level fan-out):* for the few footprints whose joints are
   genuinely distinct (dedup does not collapse them), `solve_footprint` fans the
   distinct solves across a thread pool — a thread pool, not a process pool,
   because the hoisted `manifold3d` component solids are shared in-process and
   not picklable; the Evolver solve is a subprocess whose blocking I/O drops the
   GIL, so the solves overlap, and the GPU collider bake is lock-serialised.
   `solver_options["joint_workers"]` (internal debugging only) forces the fan
   width; the default (0) sizes the fan to the core
   budget and lets the solve-gate semaphore do the scheduling, so users never
   touch it. Measured: a 100-joint connector with dedup off, 126 s serial
   -> 14 s at 16 threads (9x), same output-joint count.
   *Still open:* the on-disk cross-board solve cache (milestone 3) and the
   good-once / defect-only reuse split (every footprint is still re-solved each
   run).
