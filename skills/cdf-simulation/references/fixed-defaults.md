# Fixed defaults

Reference for `skills/cdf-simulation`. Because the skill never asks a
clarifying question, every value here is a documented, consistent choice rather
than a guess made fresh each run. Read this before building any script.

Because this skill never asks, every one of these is a documented,
consistent choice — not a guess made fresh each time.

> **Not shipped in the public release.** In the public DIG 1.2 release
> `tests/functional/resources/` holds no board USD and no component sidecar:
> none of the fixtures this file names are present. What it says about them
> is the record of the boards the templates were verified on, not a list of
> files to look for. Every run takes the user-supplied attachment, and a
> missing attachment is refusal case 5 in `SKILL.md`.

## Board identity and paths

The only Allegro-family board with confirmed, real metadata in this repo is
**CAD_BOARD** (used by every `pcb_aoi/examples/allegro_*.py`
example and by the tracked legacy THT/SMT coupon test fixtures). Resolve
`root_path` from the attachment itself, in this order. **Inspecting the
attachment is a required step, not an option**: an attachment that was
never opened has no confirmed root.

1. a `<stem>.components.json` sidecar beside the attachment naming the root;
2. embedded USD metadata: the stage default prim is the board root.
   Read it from the file -- `Usd.Stage.Open(path).GetDefaultPrim().GetPath()`
   with `pxr` available, else the `defaultPrim = "..."` token in the
   `.usda` header (a `.usd`/`.usdc` attachment needs `pxr`). It is `/Board`
   on every maintained coupon under `tests/functional/resources/` and on
   the pcb-builder board exports (see "Repository coupons as attachments").
   Exception -- a panel: when the stage composes several USD instances of one
   board under a `StepRepeats` scope, the default prim (`/World`) is only a
   container; the root is the carrier Xform that holds `StepRepeats`
   (`reference-flows.md`, "Panels");
3. if the attachment cannot be opened at all (path unreadable, sandbox
   forbids it, binary file and no `pxr`), **stop and report it** -- refusal
   case 5 with the reason "attachment root unresolvable" -- rather than
   guessing. A run that could not read its attachment has failed to resolve
   a required input; it must not hand off a plan built on another board's
   root.

`payload-contract.md` (Row 1) states the same order. The
`CAD_BOARD` values below are the confirmed paths of **that
board's own fixtures** (`CAD_BOARD_layout.usd`, `CAD_BOARD_aoi.usd`,
the legacy THT/SMT coupons in `tests/functional/resources/`). Use them when
the attachment resolves to that root; never as a stand-in for a board
whose root was not read:

```python
root_path         = "/Board_CAD_BOARD"
camera_path       = "/World/camera_light/Camera"
camera_xform_path = "/World/camera_light"
ring_light_root   = "/World/camera_light/aoi_ring_light"
```

These four values are NOT mechanically derivable from the USD filename
(confirmed: `CAD_BOARD_layout.usd` and `CAD_BOARD_aoi.usd` are
different files but share the same `root_path`). If the attached USD is
verifiably a different board (it ships its own `<name>.components.json`
naming a different root, or its embedded metadata resolves to a different
default prim such as `/Board`), use that board's own metadata instead and
note the source in the run summary — do not silently force the
CAD_BOARD paths onto a board that carries its own. Every board
carries its own: a USD stage always has a default prim or a first root
prim, so "nothing confirms a root" can only mean the file was not read.

Component source: default to `source="usd"` (embedded USD metadata,
`job.scene.select_components(types=[...], source="usd")`) unless a
`<usd_stem>.components.json` sidecar sits next to the attached USD, in
which case attach it (`job.scene.attach_sidecar(name="components",
kind="pcb.components", path=...)`) and use `source="components"` instead
(matches `pcb_aoi/examples/allegro_defect_job.py`). The choice follows the
attachment, not the row: the legacy THT fixture Row 1 was verified on ships
a sidecar, the maintained THT coupon does not, and the same rule decides
for the SMT rows.

## Named components must resolve on the attachment

A component the prompt names -- a type or footprint (`C_0805_2012Metric`) or a
refdes (`C22`) -- is used exactly as named, and only if the attached board
has it. There is no substitution. A name the attachment cannot resolve fails
validation (refusal case 6 in `SKILL.md`): say what was asked, list what the
attachment has, build nothing. Rendering some other component in its place
would hand back a successful dataset for the wrong part, and the mismatch
does not show in the images.

| Prompt | Scope used |
|---|---|
| Names a component the attachment's own metadata confirms (sidecar entry, or embedded component metadata) | That component, as named, however unfamiliar it looks (`C_0805_2012Metric`, `R_0603_1608Metric`, `SOT-723` ... are pcb-builder footprint names) |
| Names a component the attachment does not have | None: refusal case 6, nothing built |
| Names no component | The row's documented default scope (`SKILL.md`, Inputs, "Component types"), resolved against the attachment like a named scope; if it selects nothing there, refuse the same way rather than hand off an empty selection |

`PASSIVE_SMD_0805` is the SMT passive family of the CAD_BOARD fixtures (refdes
`C18`, `tests/functional/resources/SMT_0805_C18_coupon.components.json`). It
is a valid scope on those boards; it is not a fallback for a name another
board does not have.

**Check the attachment, not an impression.** Enumerate its component types
before deciding either way:

- sidecar present: the `type` (or `footprint`) field of each entry in
  `<stem>.components.json`;
- no sidecar: the embedded metadata, attribute `pcba:component:footprintId`
  on each component prim of the composed stage, e.g. with `pxr`
  `{p.GetAttribute("pcba:component:footprintId").Get() for p in stage.Traverse() if p.HasAttribute("pcba:component:footprintId")}`.
  Without `pxr`, text-search the attachment's whole folder, not only the
  root file: a pcb-builder export keeps its components in a sublayer
  (`usd/components.usda`) that the root `.usda` merely references.

The run summary says which check was made ("`C_0805_2012Metric` found in
the attachment's component types, used as named"); a refusal lists the
types that were found.

## THT board / USD fixture

**Row-1-vs-Row-2 fixture choice is not interchangeable — verified by actually
running both against every candidate.** Two THT fixtures were used to verify
the templates (neither ships in the public release):

- `tests/functional/resources/THT_TO254P762H300_U34_coupon.usdc`
  (`root_path="/Board_CAD_BOARD"`, refdes `U34`) — correct for
  **Row 1 (Golden board)**, confirmed by a real render. It **cannot** be used
  for Row 2 (pin variation): `job.operations.pin_variation(...)` failed at
  execution time with `ValidationError: pin_variation found no through-hole
  leads: this board authors no 'pcba:mesh:role' = 'pkg_lead' meshes...` — its
  own generator script's docstring confirms this fixture "predates
  `pcba:mesh:role`/`pcba:board:holeRadiusMm`, so it cannot exercise the
  discovery half of `pin_variation`."
- `tests/functional/resources/THT_SCHEMA_U1_coupon.usdc`
  (`root_path="/Coupon"`, refdes/component selector `U1`, no sidecar) — the
  fixture purpose-built with the `pcba:mesh:role` schema `pin_variation`
  needs. Confirmed by a real render (all 4 magnitude-range defaults compile
  and execute against it). **Use this one for Row 2, not the Row 1 fixture.**

In production the attachment path (and the output root) is Airflow-provided
and **required**: a prompt with no attachment stops with refusal case 5 in
`SKILL.md`, and a missing output root stops the same way. The two fixtures
above stand in **only when SQA / dry-run validation is explicitly
selected** in a checkout that carries them, in which case use whichever matches the row; they are never
silently substituted for a missing or a real attachment, and a synthesized
per-run output directory is likewise a dry-run convenience, not a production
default. For an arbitrary real attached
USD, do not assume it authors the `pcba:mesh:role` schema — if Row 2's
`pin_variation` call fails with the "found no through-hole leads" error
above, report that the attached board's THT metadata is insufficient for
pin-geometry variation rather than retrying with invented parameters.

## Repository coupons as attachments (SQA and dry runs)

These coupons are not part of the public release (see the note at the top
of this file); the section applies only to a checkout that carries them.

The two legacy fixtures above carry sidecars. The maintained coupons under
`tests/functional/resources/` instead carry their component metadata **in the
USD** (`pcba:*` schema), which changes two resolved inputs: there is no
sidecar to attach, so selection is `select_components(types=[...],
source="usd")`, and the board root comes from the coupon, not from the
`CAD_BOARD` fallback. Attaching one of these is the case the SQA
functional prompt fixtures exercise.

| Coupon | `root_path` | Component family / refdes | Rows it serves |
|---|---|---|---|
| `ARTIX_PINHEADER_J13_coupon.usdc` | `/Board` | `PinHeader_1x3_P2.54mm_Drill1.1mm` / `J13` | 1 (golden THT, procedural solder), 2 (`pin_variation`: authors `pcba:mesh:role`) |
| `ICT_C0805_C22_coupon.usdc` | `/Board` | `C_0805_2012Metric` / `C22` | 3, 4, 5 (one SMT passive) |

Each coupon is a single-component pool. A requested defect that modifies zero
components is **not** an acceptable outcome on a non-empty pool, and the
executing side agrees: proportional targeting selects
`max(1, int(pool * ratio))`, clamped to the eligible pool (pose operations in
`scripts/sdg/standalone/defects_utils/selection.py`, solder defects in
`cdf/execution/isaac/job_order_runtime.py`), and a pose operation that
selects from a non-empty pool and applies to none of it raises instead of
exiting clean. So `ratio=0.2` (the no-coverage-phrase default) on one
component yields one defect, exactly as `ratio=1.0` does; the skill states
the expected count (`max(1, int(1 * 0.2)) = 1`) in its summary. `count=1` on
`add_shift`/`add_tombstone`/`add_sideflip` is the explicit spelling of the
same thing for a one-component selection (the solder `add_*` calls take
`ratio` only). A clean render with exit 0 on a defect request therefore
means the selection matched nothing, which is a failed case, never a default
doing its job. SQA case N07 and its twin
`tests/functional/skill_prompts/row4_tombstone_default_coverage.yaml` run
exactly this variant. On a production board with tens of parts the same
default selects 20% of them, as intended.

The `output_dir` in each twin is an SQA harness argument standing in for
the output root Airflow provides in production.

## THT pin-variation magnitude ranges

`job.operations.pin_variation(...)` takes a `(lo, hi)` **range** for each
axis, not a single value or boolean — a bare "generate images with tilt
variation" names no magnitude, so use these fixed ranges (all inside
`PIN_VARIATION_BOUNDS`, chosen as representative mid-range values, not
derived from any specific board spec — note this in the run summary):

| PM term | API param | Fixed default range |
|---|---|---|
| protrusion | `height_variation` (mm) | `(0.3, 1.0)` |
| tilt | `tilt_range` (degrees) | `(5.0, 15.0)` |
| offset | `offset_range` (mm) | `(0.1, 0.4)` |
| cut | `cut_angle` (degrees) | `(5.0, 15.0)` |

Selector: prefer `components=[...]` (explicit refdes list) when the prompt
names a component; otherwise `package_types=[...]` is only usable if the
board's package metadata is confirmed dependable (per the API's own
docstring preference) — for the `THT_SCHEMA_U1_coupon.usdc` fixture (the one
that actually supports `pin_variation`, see "THT board / USD fixture"
above), default to `components=["U1"]` when no component is named. Do not
default to `U34` for this row — that refdes belongs to the Row-1 fixture,
which `pin_variation` cannot use.

## Coverage ratio — applies to BOTH structural and solder defects

- `"full board coverage"` → `ratio=1.0` against whatever pool is currently
  selected (ratio is literally "fraction of the selected/eligible pool").
- A bare count-free request with no "full board coverage" phrase → a fixed
  moderate default of `ratio=0.2` (20% of the selected pool), matching the
  `ratio` scale used across the repo's own worked examples. **This default
  applies identically to every `add_*` call in this skill** — structural
  (`add_shift`/`add_tombstone`/`add_sideflip`) and solder
  (`add_excess`/`add_missing_solder`/etc.) alike; there is nothing
  solder-specific about it.
- `ratio` never rounds a non-empty pool down to zero: the executing side
  selects `max(1, int(pool * ratio))` for pose and solder defects alike. A
  request that resolves to a non-empty selection therefore always produces
  at least one defect -- state the expected count in the run summary. A
  selection that is empty (`types=` matched nothing on the attachment) is
  not served by any ratio; report it instead of handing off a plan that
  renders a clean board.

## Structural-defect magnitudes — required, and never in the prompt

`ratio` is not enough to build a row-4 call. Each structural operation has
**required** magnitude kwargs with no defaults in the API, and a PM prompt
("generate tombstone structural defects") never supplies them. Omitting one is
a `TypeError` at the call, before compilation. Use these fixed values, drawn
from the repo's own Allegro-board defect example
(`pcb_aoi/examples/allegro_defect_job.py`) so they agree with the board
identity this skill defaults to:

| PM term | API method | Fixed default magnitudes |
|---|---|---|
| shift / offset (pose) | `add_shift` | `translate_range=0.2` (mm), `rotate_z_range=15` (degrees) |
| tombstone | `add_tombstone` | `angle_range=(70.0, 90.0)` (degrees) |
| sideflip | `add_sideflip` | `angle_range=(70.0, 90.0)` (degrees) |

Newer examples in the repo (`allegro_v7_defects_with_solder.py`,
`distributable_board_defects.py`) use larger values — `translate_range=1.0`,
`rotate_z_range=25.0`, sideflip `(60.0, 80.0)`. Those are for other boards;
this skill picks the Allegro set for consistency with its own board default,
and the run summary should say which set was applied.

**Only `add_tombstone` takes `component_types` directly, and even there it
is reserved for the legacy sidecar path.** `add_shift` and `add_sideflip` do
not — to scope either to a named component, call
`job.scene.select_components(types=[...], source=...)` BEFORE the
`add_shift`/`add_sideflip` call, narrowing the pool first. Do not attempt to
pass a `component_types=` kwarg to `add_shift`/`add_sideflip`; it does not
exist and will raise. On an embedded-metadata board (the maintained
coupons) omit `component_types` from `add_tombstone` as well and scope it
through the narrowed selection with `eligibility_source="usd"`: supplying
the list there applied the operation to nothing (2026-09-11). See
`payload-contract.md`, Row 4, "scoping".

## Combining multiple supported defect types in one request

A prompt can name more than one supported defect in the same request (e.g.
"generate excess solder and missing defects", or in principle two
structural defects together) — this is different from mixing a supported
type with unsupported `bridge` (see below, which is about dropping the
unsupported clause). For two or more **supported** types: declare
`solder_joints(...)` once (solder defects only need it declared a single
time, not once per defect call), then chain each mapped `add_*` call back to
back against the same `component_types` scope, each with its own
independent `ratio` resolved by the rule above. Order between the `add_*`
calls does not matter for solder defects (each targets independent joints);
for structural defects, `pin_variation`-style "last group wins" ordering
does not apply either — `add_shift`/`add_tombstone`/`add_sideflip` are
independent operations and can all be declared in one job.

**But read that together with "One `job.capture.render(...)` call per process"
below, because the two answer different questions.** Declaring several
structural `add_*` calls in one job puts all of those defects in the SAME
images: one job, one render, one process. That is right when the prompt wants a
board carrying several defects at once. It is NOT how you produce one image set
per defect type — that is several variants, and each variant needs its own
process, because a second `render()` in one process segfaults Kit. So: several
defects in one picture is one job; several pictures each with one defect is
several processes.

## Solder texture-defect mapping

| PM term | API method | Notes |
|---|---|---|
| "excess solder" | `add_excess(ratio, component_types=...)` | Direct match |
| "missing" (solder/texture) | `add_missing_solder(ratio, component_types=...)` | Direct match, no shape params |
| "insufficient" | `add_insufficient(ratio, ...)` | Not in the PM's sample prompts but available |
| "bright" | `add_bright(ratio, ...)` | THT-only |
| "interrupted" / "dewetting" | `add_interrupted(ratio, ...)` | Both THT/SMT |
| "dry contact" | `add_dry_contact(ratio, ...)` | THT-only |
| **"bridge"** | **No API method exists.** | **Fail loudly** — see below |

All solder defects require `job.operations.solder_joints(solver="surface_evolver", ...)`
declared first — `solver="procedural"` plus any solder-defect call is a
validation error by design (the API rejects `procedural` there
deliberately). Use `source="components"` (with a sidecar attached) for
`solder_joints(...)` when one exists, consistent with the component-source
default above.

**Row 1 is the one place `procedural` is the right solver.** The catalog's
golden board is "procedural solder, no defects", so the Row 1 template
declares `solder_joints(solver="procedural", source="components")` and no
`add_*` call — which is exactly the combination the API accepts. A golden
board rendered without that declaration shows bare pads, which is not what the
catalog row describes. Do not pair `procedural` with any solder defect, and do
not switch Row 1 to `surface_evolver` just because Row 5 needs it: the
procedural solver has no environment prerequisites, which is also why Row 1
can be dry-run validated anywhere.

**`surface_evolver` needs TWO things the environment may not have, and — this
was re-checked against current `main` on 2026-09-09 and the earlier claim in
this file was wrong — BOTH of them fail at the same, earliest possible stage:
inside the `job.operations.solder_joints(...)` call itself.**
`pcb_aoi/sections.py` calls `resolve_solver(solver)` in the body of
`solder_joints`, so neither missing dependency waits for `compile_orders()` or
for `render()`. A Row 5 script cannot even be dry-run validated without them:

1. **The `manifold3d` Python package.** Without it, `job.operations.solder_joints(solver="surface_evolver", ...)`
   raises `ValidationError: unknown solder solver 'surface_evolver'
   (registered: ['none', 'procedural'])` **immediately at that call** — not
   at `render()`, not even at `compile_orders()`, but the moment
   `solder_joints(...)` itself is declared. Confirmed: the repo's own
   `requirements.txt` lists `manifold3d`, but the minimal four-package
   validation flow used elsewhere in this doc (`numpy`/`Pillow`/`pydantic`/`PyYAML`
   only, matching this repo's own CI `unit-tests` job) does **not** install
   it — so a Row-5 script that compiles cleanly in a full `cdf-simulation`
   environment will fail to even declare `solder_joints()` in a bare
   validation container unless `manifold3d` is installed too.
2. **The native `evolver` binary.** With `manifold3d` present but no binary,
   the same call raises `ValidationError: solder solver 'surface_evolver' is
   unavailable: Surface Evolver backend unimportable: Surface Evolver binary
   not found` — again at `solder_joints(...)`, not at `compile_orders()` and
   not at `render()`. Verified 2026-09-09 on `main` at `e738f48b`
   (`pcb_aoi/sections.py:597`). An earlier version of this file said this
   second case surfaced later, at solver resolution; it does not, and never
   did — the `resolve_solver` call has been in `solder_joints` since before
   this skill was written.

The shipped `cdf-simulation` container has both preinstalled. Outside it,
install `manifold3d` (`pip install manifold3d`) AND either build the binary
from `pcb_aoi/solder_evolver/third_party/evolver/build.sh` or point
`SOLDER_EVOLVER_BIN` at an already-built `evolver` binary. Verified working
end-to-end (solder solve + `add_excess` + crop, all real output, both
dependencies present) — but also confirmed that with *either* one missing,
even a compile-only dry-run validation of a Row 5 script is impossible, not
merely a render-time risk. Before running or even *validating* any Row 5
job, confirm the executing environment has both — an Airflow worker image
or CI validation container without them will reject every solder-defect
script at declaration time, not just fail to render.

**Bridge defects are unsupported.** If a prompt asks for "bridge defects"
(alone or mixed with other solder-defect requests), do not generate
anything for the bridge portion and do not silently substitute
`add_interrupted` or any other method — respond:

> Bridge (solder short between adjacent pads/leads) is not supported by the
> current `cdf` API. Available solder defect types: insufficient, excess,
> bright, interrupted, missing_solder, dry_contact.

If the prompt mixed bridge with a supported type (e.g. "generate bridge and
excess solder defects"), still generate the supported part and report the
bridge portion as unsupported in the same response — do not fail the whole
request over one unsupported clause.
