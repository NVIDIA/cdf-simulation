# Defect strategies

Documented separately from the module docs on purpose: the solver is a
geometry/energy machine; everything on this page is *policy* layered on top of
it. The [module docs](../docs/method.md) never mention a defect taxonomy.

## The contract

One strategy instance per `defect_type`, registered in `__init__`. A strategy
may override four hooks (`base.DefectStrategy`):

| hook | what it decides | who overrides it |
|---|---|---|
| `options(base)` | solver knobs for this defect (collider, minimiser, skirt…) | everyone |
| `cluster(rects, gap)` | how paste deposits group into joints | a future `bridge` (fuses instead of splitting) |
| `joint_ids(all)` | which joints get solder at all | a future partial-missing |
| `target_volume(nominal, j)` | how much solder | the fill defects |

**An unregistered name resolves to `None`, never to nominal.** A defect that
silently produced a good joint would be mislabelled training data — the caller
reports INELIGIBLE instead. (Pinned by `test_api.py`.)

## The registered set, with the tuned numbers and why

| defect | volume | collider | minimiser | footprint |
|---|---|---|---|---|
| `nominal` | ×1.0 | off | Newton ladder (`hess_stage=10`) | `blob_scale 1.6`, `skirt 0.10` |
| `insufficient` | ×0.5 | **on** | **gradient only** | nominal footprint |
| `excess` | ×2.6 | off | Newton ladder | nominal footprint |
| `missing_solder` | 0 | — | — | — |
| `shift`/`sideflip`/`reverse_polarity` | as nominal | as nominal | as nominal | as nominal — the *placement* is the defect |

Why those numbers (all measured on the reference board):

* **`blob_scale 1.6`** — the footprint knob that actually sets the meniscus
  angle (θ is inert while pinned): 1.0 → 86°, 1.25 → 55°, **1.6 → ~35°**,
  2.2 → 11° on an 0805. A stand-in for the real pad until `board_pad` reaches
  the work item.
* **`skirt 0.10`** — flange at the pad plane so the *pinned* contact line lands
  at the pad edge, not the paste aperture ("flood the pad" with a prescribed
  footprint). Rounded pads keep their outline (`JoinType.Round`). Useful range
  is small — 0.25 turns nominal volume into a pancake.
* **`insufficient ×0.5`, not 0.3** — a third of nominal reads as *missing*
  rather than starved. Collider on + gradient-only because a starved meniscus
  is pressed along the keep-out, which is exactly where Newton stalls with a
  wrong shape.
* **`excess ×2.6`, not 1.8** — on a pinned footprint the contact line cannot
  spread, so extra volume only bulges the free surface; excess has to be gross
  before it reads. Risk not detected anywhere: at fine pitch the bulge can
  reach a neighbouring joint — that is a *bridge*, a different defect.
* **Fill separation is real but volume-carried**: on SMT, height barely
  separates the classes (a starved joint sags mid-surface while the pinned rim
  stays put); area *rises* as volume falls. `scratch/defect_matrix.py` measures
  all of this and is the tool to re-run after touching any number here.

## Through-hole fills

THT volume has **no datum** (no paste), so the fills are paired knobs, tuned on
a header pin (long protrusion; TO92's 0.138 mm poke-through is the worst case
and clips the crown via `lead_clear`):

| class | `barrel_fill` | `solve_fill` | `lead_clear` |
|---|---|---|---|
| underfill | 0.25 | 0.75 | 0.7 |
| nominal | 0.75 | 0.90 | 0.7 |
| overfill | 1.00 | 1.70 | 1.6 (covers the tip) |

`barrel_fill` maps 1:1 onto the wetted barrel wall, top-down; 0.75 is
IPC-A-610 class 2's minimum. The underfill pair went through both failure
modes before landing: bore-only (0.35/0.90) is the physically truthful defect
and indistinguishable on a panel; bore+fillet starved (0.35/0.45) reads as
missing. Genus is the class check that cannot lie: nominal must ring the pin
(genus 1), `dry_contact` must not (genus 0) — a "thinner fillet" that still
rings the pin is not an interrupted joint however plausible it looks.

The `dry_contact` construction (in `tht.interrupt_region`): subtract a *large*
circle from the far side of the pad so the pad's own outline survives as the
outer boundary and the receding front is a gentle concave curve —
`front` slides it (signed mm), `bite` bends it (×pad radius; 2.5 ≈ straight,
1.2 curls).

## Dialling numbers in

Use `python -m pcb_aoi.solder_evolver.solve --capture-dir <dir>` to export
intermediate USDs and `scripts/isaac_render.py` for RTX renders of exported
variants. Both read the same numbers as the registered strategies, so what is
dialled in is what ships. (The retired defect_matrix / export_board_variants studies that
originally measured class separation live in git history of
scripts/surface_evolver/scratch.)
