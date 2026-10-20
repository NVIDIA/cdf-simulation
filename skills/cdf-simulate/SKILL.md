---
name: cdf-simulate
description: Component-owned guidance for the cdf-simulation half of the Airflow-triggered PM prompt templates (DIG Flow Catalog) — golden THT board, THT pin-geometry variation, good/clean SMT images, structural pose defects, and solder texture defects. Turns one matched prompt into one validated `cdf.CDFJob` build and its compiled JobOrder payload, then hands that to the SDG workflow-operations skill; it never executes a render itself. Never asks a clarifying question; resolves every unspecified detail with a fixed, documented default. Not for interactive requests (route those to `skills/cdf` instead), not for real-photo input, not for AnomalyGen fine-tuning.
license: Apache-2.0
owner: NVIDIA
service: cdf-simulation
reviewed: '2026-09-09'
metadata:
  version: 1.0.0
  author: NVIDIA CDF <cdf@nvidia.com>
  tags:
  - physical-ai
  - synthetic-data-generation
  - simulation
  - pcba
  - aoi
  - airflow
---

# cdf-simulate

Turns the PM's "DIG Flow Catalog" prompt templates into `cdf` jobs through
the declarative Python API, with **zero interactive turns**. In the SDG
Workflow Skill Architecture this is the **component-owned guidance** for
`cdf-simulation` (SDD §6.3, §7): the DIG workflow skill resolves it to learn
how a catalog row maps onto this component's API, builds the job, and passes
the compiled payload to the workflow-operations skill, which is what submits
it to Airflow. Nothing here is invoked by a human chat, so there is no one to
ask a clarifying question — every ambiguity is resolved with a fixed,
documented default instead. If a request truly cannot be served (an
unsupported defect type, a non-USD input), the guidance says so loudly rather
than guessing at a fake capability or silently doing nothing.

**This skill does not execute.** Retrieved guidance is read, not run (SDD
§7.3): it produces a validated `CDFJob` build and its compiled JobOrder
payload, and stops. Rendering happens in the Airflow DAG the payload targets.

**Not this skill:** interactive requests where asking one question is
possible and cheap — route those to `skills/cdf/SKILL.md` instead. This
skill is specifically for the fixed prompt shapes in the catalog below.

## Catalog coverage

Of the PM's 7-row "DIG Flow Catalog", this skill implements 5 rows. The
other 2 have no corresponding capability anywhere in this repo and are
explicitly out of scope — see "Unsupported rows" below.

| # | Row | Status |
|---|---|---|
| 1 | Day 0 - Golden board (THT) | Implemented |
| 2 | Day 0 - Variation type (THT pin geometry) | Implemented |
| 3 | Day 0 - Good images (SMT) | Implemented |
| 4 | Day 0 - Structural defects (SMT) | Implemented |
| 5 | Day 0 - Texture defects (SMT solder, USD input) | Implemented, except `bridge` (unsupported — see below) |
| 6 | Day 1 - Texture defects (real photo input) | **Out of scope** — no code in this repo takes a real photo + defect-type and produces a defect-augmented image |
| 7 | Finetune & Inference (AnomalyGen) | **Out of scope** — no AnomalyGen training/inference code exists in this repo |

For rows 6 and 7, if a prompt clearly matches their shape ("Run Day 1 for
component ... with the attached image ...", "Fine-tune AnomalyGen ..."),
respond that this repo has no implementation for that capability — do not
attempt a best-effort substitute.

### Reference flows (config parity)

Beside the catalog rows, the skill serves the five **reference flows** in
[`references/reference-flows.md`](references/reference-flows.md): the release
SQA jobs, each phrased as one Airflow prompt whose compiled plan must equal
the corresponding SQA YAML config. A prompt carrying a flow marker -- a scan
grid and "stitched", a defect on a **named refdes** ("shift on C33",
"insufficient solder on J17", "missing-component defect on LS1"),
"component-focus" or "pin-focus" crops -- is a flow, not a catalog row, and
takes its fixed values from that file (Marc's SQA numbers, which differ from
the catalog defaults: shift is `translate_range=0.5, rotate_z_range=5.0` with
no `variant_index`, crops are ROI captures rather than `postprocess.crop`).

| Flow | Prompt marker | Config twins |
|---|---|---|
| G golden scan | "golden board dataset ... NxM scan ... stitched" | A1, A2, A3 (panel) |
| C component defects | "shift/tombstone/sideflip/reverse polarity on <refdes>" + "component-focus crops" | A4, A5 |
| S SMT solder defects | "insufficient/excess/interrupted/missing solder on <refdes>" + "component-focus crops" | A6, A7 |
| M missing component | "missing-component defect on <refdes>" | A8, A9 |
| T THT solder defects | six THT solder defects on refdes + "pin-focus crops from the bottom" | A10, A11, A12b (panel) |

## Fixed defaults

Every unspecified detail resolves to a documented default rather than a
question. The full table -- board identity and paths, the rule that a named component
must resolve on the attachment, the THT fixture, pin-variation magnitude ranges,
**the required structural-defect magnitudes** (`translate_range`,
`rotate_z_range`, `angle_range` -- the API has no defaults for these and a PM
prompt never states them), the coverage ratio, how multiple defect types
combine, and the solder texture-defect mapping -- is in
[`references/fixed-defaults.md`](references/fixed-defaults.md).

**Read it before building any script.** The values are not obvious and
several of them are load-bearing: the coverage ratio applies to structural
*and* solder defects, and the crop `class_filter` taxonomy is not the same as
`component_types`.

## Payload contract

What this skill hands to the workflow layer, what it accepts, and how each
value is validated is in
[`references/payload-contract.md`](references/payload-contract.md) (the
component-owner contract the SDD asks for in §7.2). The short version: the
output is a `CDFJob` build plus the compiled `JobOrderPlan` from
`job.compile_orders(...)`, serialized with `JobOrderCodec.dumps(plan)` (a
bare `plan.model_dump()` is not JSON-serializable), together with a
resolved-inputs summary naming every default that applied. Airflow supplies
the attachment path and the output root -- both **required**, with no
production fallback -- and the one-process-per-variant fan-out; this skill
supplies everything else from the catalog row and the fixed defaults.

## Instructions

Four steps, in order. Step 1 decides everything downstream, so do not start
writing a script before the row is settled.

1. **Match the prompt to exactly one catalog row or one reference flow.**
   Check the flow markers first (`references/reference-flows.md`, "Routing"):
   a named refdes target, a scan grid with "stitched", "component-focus" /
   "pin-focus" crops. With a marker present the prompt is that flow; without
   one it is a catalog row. Two different misses, two different answers: a
   prompt matching **rows 6-7** stops here and says the repo has no
   implementation, because it does not; a prompt matching **no row and no
   flow** is routed to [`skills/cdf`](../cdf/SKILL.md), because it may
   well be servable, just not by the fixed shapes this skill owns. Neither case
   assembles a best-effort substitute out of the pieces that do exist.
2. **Resolve every input from the precedence order in [Inputs](#inputs).**
   Anything still unset takes its documented default from
   [`references/fixed-defaults.md`](references/fixed-defaults.md).
3. **Build one `cdf.CDFJob` script**, following the row's shape in
   [Workflow](#workflow). One job per variant: the executing side must launch
   each variant as its own process, because sequential
   `job.capture.render(...)` calls in one process crash the second Kit
   subprocess -- so the payload says one variant, never a loop.
4. **Validate, then hand off** -- compile the job with
   `job.compile_orders(...)`, read the result, and return the build plus the
   compiled plan to the workflow-operations skill. Do not call `render()`
   here. Report what was assumed alongside what was handed off.

## Inputs

Nothing here is prompted for. Each value is resolved from the first source that
supplies it:

**state file -> explicit prompt arguments -> agent context -> user prompt text**

| Input | Required | Where it comes from if unset |
|---|---|---|
| Catalog row | yes | Matched from the prompt shape; no default. Rows 6-7 stop with "no implementation"; no match at all routes to `skills/cdf` |
| Board USD (attachment path) | yes, Airflow-provided | **No production fallback.** A prompt with no attachment stops with refusal case 5 below. The repository coupons named in `references/fixed-defaults.md` stand in only when SQA / dry-run validation is explicitly selected in a checkout that carries them; the public release ships none, so there a missing attachment is always refusal case 5 |
| Output root | yes, Airflow-provided | **No production fallback.** A missing output root stops the same way; a synthesized per-run directory is a dry-run convenience only |
| Component sidecar | when the attachment ships one | Conditional on the attachment, not on the row: attach `<stem>.components.json` when it sits beside the USD (the Row 1 legacy THT fixture ships one; the maintained coupons do not), else `source="usd"` |
| Board `root_path` | no | Read from the attachment: its sidecar if it ships one, else the stage default prim of the USD (`/Board` on every maintained coupon and on the pcb-builder boards). Opening the attachment is a required step; if it cannot be opened, stop with refusal case 5 ("attachment root unresolvable"). `/Board_CAD_BOARD` is the root of the CAD_BOARD fixtures only, never a fallback for an unread board |
| Named component | from the prompt | Passed through as named when it is present in the attachment's own component types (sidecar, else embedded metadata -- list them, do not guess); a name absent from that list is refusal case 6, never substituted. State the check in the run summary |
| Component types | no | The row's documented set |
| Seed, scan grid, board side, resolution, pin numbers (flows) | no | Explicit prompt arguments in the forms listed in `references/reference-flows.md` ("Seed 7", "a 5x4 scan", "of the bottom side", "at 2000x2000", "(pin SH)"); otherwise that file's per-flow defaults |
| Panel instances | from the attachment | When the attachment composes one board as several USD instances (`<root>/StepRepeats/SR_*`), register all of them with `job.scene.instances(...)`; image budgets multiply by the instance count (`references/reference-flows.md`, "Panels") |
| Coverage ratio | no | The single documented ratio, applied to structural *and* solder defects; the executing side selects `max(1, int(pool * ratio))`, so a non-empty selection always yields at least one defect |
| Defect magnitudes | no | The per-type ranges in `references/fixed-defaults.md` |
| `class_filter` | no | `None`. Never a component-type string -- different taxonomy |

State what was resolved and from where in the run summary. A caller reading an
Airflow log has no other way to know which defaults applied.

## Workflow

### 1. Match the prompt to a catalog row

Use the sample-prompt phrasing in the table under "Catalog coverage" (and
the fuller phrasing in the PM's original catalog) to classify the request
into exactly one of rows 1-5. If it matches row 6 or 7's shape, respond
"out of scope" per that section — do not proceed to script generation.

If a prompt does not clearly match any of the 7 rows, treat it as an
interactive-style request and route to `skills/cdf/SKILL.md` instead
(this skill only handles the fixed catalog shapes).

### 2. Resolve inputs deterministically

Extract from the prompt: attached USD path, output root, named component(s),
variation/defect type, "full board coverage" flag. Then **open the
attachment** and read two things from it: the board root (sidecar, else the
stage default prim) and its component types. A named component that
appears in those types is used as named; a name that is absent is refusal
case 6 -- never a substitution. Never ask; apply the fixed defaults above
for anything the prompt does not specify, and record both reads (root
source, component check) in the run summary.

The attachment path and the output root are the two inputs with **no
default**: Airflow provides both. If either is absent, stop and report it
(refusal case 5) rather than substituting a repository coupon or inventing a
directory -- those stand-ins exist only for explicitly selected SQA /
dry-run validation. Whether a sidecar is attached follows from the
attachment (does `<stem>.components.json` sit beside it?), not from the
row: the legacy THT fixture Row 1 was verified on ships one.

### 3. Build the `CDFJob` script

**`import pcb_aoi` before constructing the job.** The `"pcb-aoi"` domain
registers on import; without that import `CDFJob(domain="pcb-aoi", ...)`
raises `unknown domain 'pcb-aoi' (registered: none)` on the very first line.
Every template below assumes `import pcb_aoi` and `from cdf import CDFJob`
at the top of the script.

**Every row requires `job.capture.renderer(mode="pathtracing", resolution=..., spp=..., total_spp=..., max_bounces=...)` before `capture.camera`/`scan_grid`/`outputs` -- it is a required field, `compile_orders()` rejects a plan missing it ("capture render pass 'default' is missing: renderer"). Every template below assumes this call is present even where not shown explicitly.**

**`capture.camera(rotation_range=...)` is required, not optional** -- only
`side=` has a default. A top-side capture with no rotation asked for still
needs `rotation_range={"z": 0}`; omitting the kwarg is a `TypeError` before
anything is compiled.

**Several `operations.add_*` calls have required magnitude kwargs with no
defaults** (`add_shift` needs `translate_range` *and* `rotate_z_range`;
`add_tombstone` and `add_sideflip` each need `angle_range`). A PM prompt never
states them, so they come from the fixed table in
[`references/fixed-defaults.md`](references/fixed-defaults.md) -- do not invent
a number and do not ask.

**Declaration order matters and is enforced.** `job.scene.select_components(...)`
must be declared before `job.annotations.label_components(...)` --
`compile_orders()` rejects the reverse order with
`annotations.components.apply requires component selection to be declared
first`. Every template below follows select → (operations) → annotate.

**`scene.select_components(source=..., types=None)` without `types=` needs
either USD-embedded component metadata or `source="components"` pointing at
an attached sidecar** -- bare `source="usd"` with no `types` and no sidecar
fails at runtime with `component selection needs explicit types or board
metadata` (confirmed against the on-repo THT coupon fixture, which has no
embedded metadata). Prefer attaching the fixture's own `.components.json`
sidecar (`job.scene.attach_sidecar(name="components", kind="pcb.components",
path=...)`, then `source="components"`) whenever one exists alongside the
attached USD, over guessing at `source="usd"`.

Follow `skills/cdf/SKILL.md`'s "Generate the script" shape (imports →
`build_job() -> CDFJob` → CLI → `main()`), using the per-row template
below -- or, for a reference flow, the flow's template and shared conventions
in `references/reference-flows.md` (explicit `source="usd"`, `output_start(0)`,
`limits(...)`, `realtime_subframes=1`, the three-key camera rotation), which
are the spellings the YAML interpreter produces. Place generated PCB scripts under `pcb_aoi/examples/` only if the run is
meant to be durable/reusable; for a one-off Airflow-triggered run, build
and compile the job in-memory, then emit the payload
([`references/payload-contract.md`](references/payload-contract.md))
without necessarily writing a file to disk. Execution is the Airflow
DAG's, never this skill's.

**Row 1 — Golden board (THT):** verified against
`THT_TO254P762H300_U34_coupon.usdc` + its own `.components.json` sidecar
(a verification fixture; not shipped in the public release).

```python
job = CDFJob(domain="pcb-aoi", random_seed=0)
job.scene.configure(usd_path=..., root_path="/Board_CAD_BOARD")  # the attachment's own default prim; CAD_BOARD because this example uses the CAD_BOARD fixture
job.scene.attach_sidecar(name="components", kind="pcb.components", path=...)  # only when <stem>.components.json ships with the attachment
job.scene.select_components(types=[<the attachment's component family>], source="usd")  # embedded-metadata board (every maintained coupon, every pcb-builder export); with a sidecar: attach it and use source="components"
job.appearance.lighting(mode="white_light", exposure_range=(0.0, 0.0),
                         cone_angle_range=(120.0, 120.0), cone_softness_range=(1.0, 1.0),
                         dome={"intensity": (1000.0, 1000.0), "visible": True})
# The catalog's golden board carries PROCEDURAL solder: joints are generated,
# no defect is applied. `procedural` is valid here precisely because no
# add_* solder-defect call follows (those require surface_evolver).
job.operations.solder_joints(solver="procedural", component_types=[<the same family>], source="usd")  # component_types names the family the joints are generated for; the YAML twin (row1_golden_tht.yaml) carries it. With a sidecar attached, this reads source="components" too -- all three source= arguments in a row must name the same metadata source
job.annotations.label_components(source="usd")  # likewise source="components" when the sidecar is attached
job.capture.renderer(mode="pathtracing", resolution=(1920, 1080), spp=1, total_spp=32, max_bounces=4)
job.capture.camera(side="bottom", rotation_range={"z": -90})
job.capture.scan_grid(x_count=1, y_count=1)
job.capture.outputs(rgb=True, bounding_box_2d_tight=True,
                     semantic_segmentation=True, colorize_semantic_segmentation=True)
# No postprocess.crop() call -- explicitly no Qwen/crop chain for this row.
```

`camera_path`/`camera_xform_path`/`ring_light_root` are optional
(`scene.configure`'s default is `None`) -- omit them rather than assuming
the `/World/camera_light/...` Allegro rig exists verbatim on every board;
the pipeline falls back to a synthesized camera when unset, confirmed
working by this row's actual render.

**Row 2 — Variation type (THT pin geometry):**

```python
job = CDFJob(domain="pcb-aoi", random_seed=0)
job.scene.configure(usd_path=..., root_path="/Coupon")  # verified on THT_SCHEMA_U1_coupon.usdc (not shipped publicly) -- see below
job.appearance.lighting(mode="white_light", exposure_range=(0.0, 0.0),
                         cone_angle_range=(120.0, 120.0), cone_softness_range=(1.0, 1.0),
                         dome={"intensity": (1000.0, 1000.0), "visible": True})
job.operations.pin_variation(components=["U1"], offset_range=(0.1, 0.4))  # or tilt_range/cut_angle/height_variation
job.capture.renderer(mode="pathtracing", resolution=(1920, 1080), spp=1, total_spp=32, max_bounces=4)
job.capture.camera(side="bottom", rotation_range={"z": -90})
job.capture.scan_grid(x_count=1, y_count=1)
job.capture.outputs(rgb=True)
```

DIFFERENT scene from Row 1 (see "THT board / USD fixture" above --
`THT_TO254P762H300_U34_coupon.usdc` cannot exercise `pin_variation` at all;
use `THT_SCHEMA_U1_coupon.usdc`, `root_path="/Coupon"`). `pin_variation(...)`
targets leads directly by `components=`/`package_types=` -- it does **not**
need `scene.select_components(...)` or `annotations.label_components(...)`
declared first (unlike every other row); this fixture has no sidecar and
`select_components(source="usd")` would fail on it the same way the "no
explicit types or board metadata" pitfall describes. Selector defaults to
`components=["U1"]` (this fixture's refdes) unless the prompt names a
different part on a different, verified-compatible board.

**Row 3 — Good images (SMT):** board-level (`"20 good PCB images"`) uses
`capture.scan_grid(x_count=X, y_count=Y)` where **`X * Y`** is the requested
count, not `N` in each axis -- `scan_grid(x_count=20, y_count=20)` is 400
renders, not 20. Pick the most square factorisation at or just above the ask
(20 -> 5x4; 30 -> 6x5; a prime like 17 -> 5x4 = 20 and say so), no
crop, `select_components(types=[<every component family the attachment
carries>], source="usd")` -- the families are read from the attachment
(`pcba:component:footprintId`), so on a one-family coupon the list has one
entry (`row3_good_board.yaml` twin: `types: [C_0805_2012Metric]`); a bare
`select_components(source="usd")` with no `types=` compiles to a different
plan (`types: null`) and is not the twin. Component-level (`"good images of
<component>, max 3"`) narrows first with
`select_components(types=[<component_type>], source="usd")`, then
after the same capture block adds
`job.postprocess.crop(method="bbox_tight", types=["rgb", "semantic_segmentation", "component_instance"], class_filter=None, offset=10, xform_depth=7, output_subdir="cropped")`
and caps `capture.limits(max_images=3)` (or the requested max). **`class_filter=None`, never the component-type string** -- see "Crop `class_filter` is a different taxonomy than `component_types`" pitfall below; this was verified end-to-end: `class_filter=["PASSIVE_SMD_0805"]` compiled and executed with `"error": ""` but silently produced zero crop files, because `annotations.label_components` assigned the semantic class `"capacitor"` to that component, not the literal type string. No solder or structural-defect operations for this row -- clean/golden only. No Qwen step (per the "Fixed defaults" note above); output stops at the rendered/cropped RGB.

Board-level Row 3, complete (the `row3_good_board.yaml` twin; only the grid and
the render size come from the prompt):

```python
job = CDFJob(domain="pcb-aoi", random_seed=0)
job.scene.configure(usd_path=..., root_path="/Board")                       # the attachment's default prim
job.scene.select_components(types=[<all families of the attachment>], source="usd")
job.appearance.lighting(mode="white_light", exposure_range=(0.0, 0.0),
                         cone_angle_range=(120.0, 120.0), cone_softness_range=(1.0, 1.0),
                         dome={"intensity": (1000.0, 1000.0), "visible": True})
job.annotations.label_components(source="usd")                              # always: the good images ship their labels
job.capture.renderer(mode="pathtracing", resolution=(1920, 1080), spp=1, total_spp=32, max_bounces=4)
job.capture.camera(side="top", rotation_range={"z": 0})
job.capture.scan_grid(x_count=X, y_count=Y)                                 # X * Y = the requested count
job.capture.outputs(rgb=True, bounding_box_2d_tight=True,
                     semantic_segmentation=True, colorize_semantic_segmentation=True)
# no operations, no postprocess.crop
```

**Row 4 — Structural defects (SMT):** the same SMT scene as Row 3, with pose
operations declared after `select_components` and before
`label_components`, then operation labels and the per-component defect
crop. **Rows 4 and 5 always emit per-component defect crops** -- this is
the one output contract for both rows (`references/class-filter-taxonomy.md`):
`annotations.enable_operation_labels(namespace="defect")` puts the defect
class in the segmentation output, and the same `postprocess.crop(...)` block
as Row 3's component-level case follows the capture. A record with no file
under `cropped/` is a failed record, not a board-level variant. The
narrowed `select_components(types=[...])` is the scope of every pose
operation; whether `add_tombstone` also takes `component_types=` depends on
the attachment's metadata source -- see "Scoping by metadata source" below.

```python
job = CDFJob(domain="pcb-aoi", random_seed=0)
job.scene.configure(usd_path=..., root_path="/Board_CAD_BOARD")
job.scene.attach_sidecar(name="components", kind="pcb.components", path=...)  # legacy sidecar board
job.scene.select_components(types=["PASSIVE_SMD_0805"], source="components")  # the scope of every pose op
job.appearance.lighting(...)                              # same block as Row 1
job.operations.add_tombstone(ratio=0.2, angle_range=(70.0, 90.0))
# job.operations.add_shift(ratio=0.2, translate_range=0.2, rotate_z_range=15.0)
# job.operations.add_sideflip(ratio=0.2, angle_range=(70.0, 90.0))
job.annotations.label_components(source="components")
job.annotations.enable_operation_labels(namespace="defect")
job.capture.renderer(mode="pathtracing", resolution=(1920, 1080), spp=1, total_spp=32, max_bounces=4)
job.capture.camera(side="top", rotation_range={"z": 0})
job.capture.scan_grid(x_count=1, y_count=1)
job.capture.outputs(rgb=True, bounding_box_2d_tight=True,
                    semantic_segmentation=True, colorize_semantic_segmentation=True)
job.postprocess.crop(method="bbox_tight", types=["rgb", "semantic_segmentation", "component_instance"],
                     class_filter=None, offset=10, xform_depth=7, output_subdir="cropped")
```

One defect type per script when the prompt wants one image set per type (see
"One `job.capture.render(...)` call per process" below); all three in one
script only when the prompt wants a board carrying all three at once.

**Scoping by metadata source.** `add_shift` and `add_sideflip` have no
`component_types=` kwarg, so the narrowed `select_components(types=[...])`
is always their scope. `add_tombstone` does have one, and whether to use it
is decided by where the attachment's component metadata lives:

- **Embedded USD metadata** (`pcba:*` schema, no sidecar -- every maintained
  coupon under `tests/functional/resources/`): narrow with
  `select_components(types=[...], source="usd")`, pass
  `eligibility_source="usd"`, and **omit** `component_types`. Supplying it
  applied the operation to nothing: the list resolved to prim paths the
  stage does not carry and the run failed with `tombstone selected from 1
  components but applied none of them` (2026-09-11).
  `tests/functional/skill_prompts/row4_tombstone.yaml` is that build, run
  to completion.
- **Legacy sidecar board** (`<stem>.components.json` attached -- the board
  the template above was first verified on): `component_types=[...]` on
  `add_tombstone` is verified there and may accompany the narrowed
  selection. It is never a substitute for narrowing the selection, and it
  is never carried over to an embedded-metadata board.

**Row 5 — Texture defects (SMT solder):** as Row 4 -- same operation labels,
same outputs, same per-component defect crop -- but
`operations.solder_joints(...)` is declared once before any `add_*` solder
call, and the solver must actually be available in the executing environment.
The solder `add_*` calls take `ratio` only (no `count`) and their
`component_types=` is verified on both metadata sources.

```python
job = CDFJob(domain="pcb-aoi", random_seed=0)
job.scene.configure(usd_path=..., root_path="/Board_CAD_BOARD")
job.scene.attach_sidecar(name="components", kind="pcb.components", path=...)  # legacy sidecar board
job.scene.select_components(types=["PASSIVE_SMD_0805"], source="components")   # embedded-metadata board: types=[<family>], source="usd"
job.appearance.lighting(...)                              # same block as Row 1
job.operations.solder_joints(solver="surface_evolver", component_types=["PASSIVE_SMD_0805"], source="components")  # component_types = the selected family (row5_excess_solder.yaml twin)
job.operations.add_excess(ratio=0.2, component_types=["PASSIVE_SMD_0805"])
# chain further supported types here, e.g.
# job.operations.add_missing_solder(ratio=0.2, component_types=["PASSIVE_SMD_0805"])
job.annotations.label_components(source="components")
job.annotations.enable_operation_labels(namespace="defect")
job.capture.renderer(mode="pathtracing", resolution=(1920, 1080), spp=1, total_spp=32, max_bounces=4)
job.capture.camera(side="top", rotation_range={"z": 0})
job.capture.scan_grid(x_count=1, y_count=1)
job.capture.outputs(rgb=True, bounding_box_2d_tight=True,
                    semantic_segmentation=True, colorize_semantic_segmentation=True)
job.postprocess.crop(method="bbox_tight", types=["rgb", "semantic_segmentation", "component_instance"],
                     class_filter=None, offset=10, xform_depth=7, output_subdir="cropped")
```

`solder_joints(solver="surface_evolver")` resolves the solver **inside that
call**: with `manifold3d` or the native `evolver` binary missing it raises
there, before `compile_orders()` is reached. See
[`references/fixed-defaults.md`](references/fixed-defaults.md) — a Row 5 script
cannot even be dry-run validated in an environment that lacks them.

### Pitfall: `class_filter` is not `component_types`

`postprocess.crop(class_filter=...)` takes a derived semantic class, not a
component-type string. Given the latter it compiles, reports `"error": ""`,
and writes zero files. **Default to `class_filter=None`** and filter with
`types=`. Worked example and the two taxonomies side by side:
[`references/class-filter-taxonomy.md`](references/class-filter-taxonomy.md).

### 4. Validate, then hand off -- do not execute

Run `job.compile_orders(output_dir=...)` (per `skills/cdf/SKILL.md` Step 5)
to catch a malformed script before anyone spends Isaac boot time. A clean
compile is where this skill's job ends: return the `CDFJob` build, the
compiled `JobOrderPlan`, and the resolved-inputs summary to the
workflow-operations skill, in the shape
[`references/payload-contract.md`](references/payload-contract.md) defines.
Do **not** call `job.capture.render(...)` from here -- retrieved guidance is
read, not run (SDD §7.3), and execution is the Airflow DAG's. Report what was
handed off; do not claim images were generated, because at this point none
have been.

**Tell the workflow layer: one `render()` per process.** Confirmed by running
three structural-defect variants (shift/tombstone/sideflip) sequentially in
one Python script during verification: the first `render()` call succeeded,
the second segfaulted the Kit subprocess mid-run (`carb.crashreporter-breakpad`
fatal trace). Each variant as its own process worked cleanly every time. This
matches documented Isaac/replicator fragility around rebuilding the render
graph mid-session. So when a prompt implies several output variants
("generate shift, tombstone, and sideflip defects"), emit one payload per
variant and mark them as independent launches -- never a loop of `render()`
calls inside one script.

## Examples

| Prompt | Row | Key resolved values |
|---|---|---|
| "Generate a golden board image using the attached USD" | 1 | bottom camera, `solder_joints(solver="procedural")` with no defect calls, rgb+bbox+semantic, no crop |
| "Using the attached USD, generate images with tilt variation" | 2 | `tilt_range=(5.0, 15.0)`, `components=["U1"]` |
| "Using the attached USD, generate images with cut variation" | 2 | `cut_angle=(5.0, 15.0)` |
| "Generate 20 good PCB images using the attached USD" | 3 | board-level, `scan_grid(x_count=5, y_count=4)` = 20 images, no crop |
| "Generate good images of PASSIVE_SMD_0805, max 3" | 3 | `select_components(types=["PASSIVE_SMD_0805"])`, component-level crop, `max_images=3` (a component the attachment does not have is refused, case 6 -- never substituted) |
| "Using the attached USD, generate tombstone structural defects" | 4 | `add_tombstone(ratio=0.2, angle_range=(70.0, 90.0))` (no coverage phrase → default ratio, which still yields at least one defect on a non-empty selection; `angle_range` is required and comes from the fixed table); `enable_operation_labels` + per-component crop as on every row 4/5 record |
| "Generate shift structural defects for PASSIVE_SMD_0805, with full board coverage" | 4 | `select_components(types=["PASSIVE_SMD_0805"])` then `add_shift(ratio=1.0, translate_range=0.2, rotate_z_range=15.0)`; labels + crop |
| "Using the attached USD, generate excess solder on the passives" | 5 | `solder_joints(solver="surface_evolver")` + `add_excess(ratio=0.2, component_types=["PASSIVE_SMD_0805"])`; labels + crop |
| "Run Day 0 for component PASSIVE_SMD_0805 to generate excess solder and missing defects" | 5 | one `solder_joints(...)` declaration, then `add_excess(ratio=0.2, ...)` and `add_missing_solder(ratio=0.2, ...)` chained (see "Combining multiple supported defect types") |
| "Using the attached USD, generate bridge defects on the ICs" | 5 | Fail loudly — bridge unsupported |
| "Run Day 1 for component X with the attached image to generate bridge defects" | 6 | Out of scope — no real-photo defect capability in this repo |
| "Fine-tune AnomalyGen on my labeled dataset, then generate defects" | 7 | Out of scope — no AnomalyGen code in this repo |

## When to refuse

Being non-interactive makes refusal *more* important, not less: there is no
follow-up turn in which a wrong guess gets corrected, and an Airflow log is
the only place the caller finds out what happened. Six cases, with the
response to give. Each states plainly what was not done and why, produces no
substitute, and — where part of the request is servable — still delivers that
part.

**1. Row 6 (real-photo input).** Refuse the whole request; there is nothing
partial to deliver.

> Row 6 (Day 1, real-photo texture defects) has no implementation in this
> repo: no code takes a captured photo plus a defect type and returns a
> defect-augmented image. I have not run a USD-input job in its place, because
> that would answer a different question than the one asked. Rows 1-5 (USD
> input) are available if a USD of this board exists.

**2. Row 7 (AnomalyGen fine-tune / inference).** Same shape, same reason.

> Row 7 (AnomalyGen fine-tune and inference) has no implementation in this
> repo — no training or inference code for it exists here. Nothing was
> generated.

**3. `bridge` mixed with supported solder defects.** Partial fulfilment, not
refusal: generate exactly the supported types named, drop the bridge clause,
report it.

> Requested: bridge and excess solder. Generated: excess solder only
> (`add_excess(ratio=0.2, component_types=["PASSIVE_SMD_0805"])`, 1 variant,
> output at `<dir>`). Not generated: bridge — solder shorting between adjacent
> pads/leads has no method in the current `cdf` API. Available texture
> defects: insufficient, excess, bright, interrupted, missing_solder,
> dry_contact. I did not add any of those to make up the count, and did not
> approximate a bridge from the ones that exist.

**4. A board whose real metadata is being overridden by the defaults.** The
`CAD_BOARD` values are a fallback for when nothing better is known.

> Not run. The attached USD carries its own metadata (a `<name>.components.json`
> sidecar, or embedded `pcba:*` metadata whose default prim is the board root)
> naming a different board root, so applying the `CAD_BOARD`
> defaults would silently render the wrong board identity — a mismatch that
> does not show up in the image. Re-run with that board's own `root_path`, or
> confirm the defaults are genuinely intended for it.

**5. No attachment path, or no output root.** Both are Airflow-provided,
required production inputs. Stop; do not substitute a repository coupon for
the attachment or synthesize a directory for the output root -- those
stand-ins are for explicitly selected SQA / dry-run validation only, and
the public release ships no repository coupons at all.

> Not built. This request carries no attached USD path (or: no output root).
> Both are required inputs the Airflow DAG supplies; the repository coupons
> and a per-run directory are validation stand-ins, not production defaults.
> Re-run with the attachment path and the output root set.

**6. A named component the attachment does not have.** A validation
failure, not a substitution. Check the attachment's component types (sidecar
entries, else the embedded component metadata) and refdes; if the named one
is not there, build nothing.

> Not built. The prompt names `<name>`, which the attached board does not
> have (checked: <sidecar types | embedded component types>: `<list>`).
> Rendering another component in its place would deliver a dataset for the
> wrong part. Re-run naming one of the listed components, or attach the board
> that carries `<name>`.

What none of these do: assemble a lookalike out of the capabilities that do
exist, or report success for a run that did not happen. This skill hands off
a payload and never renders, so it never has grounds to say images were
generated -- that report comes back from the workflow layer, or not at all.

## Guardrails

- Never ask a clarifying question. Apply the fixed defaults above and state
  what was assumed in the run summary.
- Never invent a component name, prim path, or API method that was not
  confirmed in this repo's own code (`pcb_aoi/sections.py`,
  `pcb_aoi/examples/*.py`, `cdf/API_SPEC.md`). If something needed does not
  exist (bridge defects, real-photo input, AnomalyGen), say so plainly.
- Never build on a board root that was not read from the attachment. The
  CAD_BOARD paths belong to the CAD_BOARD fixtures; an attachment
  that could not be opened is refusal case 5, not a reason to assume that
  board. Confirmed metadata includes the embedded USD default prim, not
  only a sidecar.
- Never substitute a component for one the prompt names. Check the
  attachment's component types first: pcb-builder footprint names
  (`C_0805_2012Metric`, `R_0603_1608Metric`, `SOT-723`) look unfamiliar and
  are real, and a name the attachment does not have is refusal case 6.
- Never substitute a repository coupon for a missing attachment, or a
  synthesized directory for a missing output root, outside explicitly
  selected SQA / dry-run validation. Both are required Airflow-provided
  inputs; stop and say so (refusal case 5).
- Never hand off a defect record whose selection is empty, and never
  describe a zero-defect outcome as the default doing its job: on a
  non-empty selection the executing side applies at least one defect
  (`max(1, int(pool * ratio))`), and a clean board on a defect request is a
  failed record.
- Never claim Qwen lighting-transfer, or any other postprocessing step that
  does not exist in this repo, was applied.
- Never execute from this skill. Compile, then hand the build and the
  `JobOrderPlan` to the workflow-operations skill (SDD §7.3: retrieved
  guidance is read, not run). A `render()` call here is a bug, not a
  shortcut.
- Never route a request matching rows 1-5 to `skills/cdf/SKILL.md`'s
  interactive path — this skill owns the guidance for those shapes,
  non-interactively.
- Never pass a component-type string to `postprocess.crop(class_filter=...)`
  — it is a different taxonomy than `component_types`/`types=` and will
  silently produce zero crop files with no error. Default to
  `class_filter=None`.

## Verification status

All five implemented rows were executed end to end, crops included, and the
run surfaced four defects that are now fixed. The record is in
[`references/verification-status.md`](references/verification-status.md).

## SQA: the prompt set and its YAML twins

Checked-in YAML under `tests/functional/skill_prompts/` covers eight catalog
prompts (P01-P08) plus the default-coverage tombstone variant (N07). Two checks
keep the YAML honest, and they check different things:
`tests/unittests/api_parity/test_skill_prompt_yaml_api_parity.py` asserts
each YAML compiles to the same plan as a hand-written Python API build of
the same job (**YAML-versus-Python-API parity** -- it does not read the
templates in this file, so a template can drift while it stays green), and
the functional CI matrix executes every YAML in the candidate container. When
a template in this file changes, update the YAML twin and the Python builder
together.

Two conventions those twins make explicit, because a plan comparison is whole:
the YAML interpreter always emits `capture.output_start(0)` and
`capture.limits(triggers=1, max_images=-1)`, so a template build compared
against a YAML declares those defaults too; and on an embedded-metadata board
`add_tombstone(...)` carries `eligibility_source="usd"`, the scene's metadata
source, which the YAML derives on its own.

## Limitations

- **Two of the seven catalog rows have no implementation anywhere in this
  repo.** Row 6 (Day 1, real-photo input) and row 7 (AnomalyGen fine-tune and
  inference) are out of scope, not merely unimplemented here.
- **`bridge` solder defects are unsupported**; the other texture defects in
  row 5 are.
- **One variant per process, on the executing side.** Sequential
  `job.capture.render(...)` calls in a single Python process segfault the
  second Kit subprocess, so each payload this skill emits is one variant and
  is marked as an independent launch.
- **Guidance only.** This skill compiles and hands off; it does not render.
  The workflow-operations skill and the Airflow DAG own execution, and this
  skill has no view of the run's outcome beyond what they report back.
- **Requires a USD input.** There is no photo-to-defect path.
- **Non-interactive by construction.** Every ambiguity resolves to a default,
  which is right for Airflow and wrong for a human who could have been asked
  one question -- those requests belong to `skills/cdf`.
- **The defaults are documented, not universal.** They were chosen against the
  fixtures in this repo. A different board needs its own confirmed metadata,
  not the `CAD_BOARD` fallback.

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `"error": ""` but zero crop files | A component-type string was passed to `postprocess.crop(class_filter=...)`, which is a different taxonomy from `component_types` | Use `class_filter=None`; filter by `types=` instead |
| Second render segfaults (reported back by the workflow layer) | Two `job.capture.render(...)` calls in one Python process | Emit one payload per variant, each marked as an independent launch; never a `render()` loop |
| This skill was asked to render | A caller treated component guidance as an executor | Compile and hand off the `JobOrderPlan`; execution belongs to the workflow-operations skill and the DAG |
| `select_components` returns nothing | The sidecar and the board USD disagree, or the source is wrong for that fixture | Confirm the sidecar belongs to that board; check `source=` against the row |
| A defect request rendered a clean board with exit 0 | The selection was empty (`types=` matched nothing on that attachment); a non-empty selection always yields at least one defect | Check `types=` against the attachment's own metadata; do not hand off a plan whose selection is empty |
| Row 4/5 record with no file under `cropped/` | `enable_operation_labels(...)` or `postprocess.crop(...)` was left off the record, or `class_filter` carried a component-type string | Both calls are part of the Row 4/5 contract; `class_filter=None` |
| Compile error naming an unknown operation | An API method was invented rather than confirmed | Check `pcb_aoi/sections.py` and `cdf/API_SPEC.md`; if it is absent, the capability does not exist |
| Prompt matches nothing in the catalog | Rows 6-7, or a genuinely new shape | Say the repo has no implementation; do not substitute |
| Pin variation produces no visible change | Wrong board metadata for that fixture | Use the fixture the row documents, not the row-1 one |
