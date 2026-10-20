# Payload contract

Reference for `skills/cdf-simulate`. This is the component-owner
contract the SDG Workflow Skill Architecture asks for (§7.2): the parameters
this component supports for each catalog row, how each one is validated, and
the exact shape this skill hands to the workflow-operations skill. The skill
never executes; everything below describes what it *emits* and what the
executing side must supply.

## What this skill emits

One matched prompt produces one **variant**, and one variant is one hand-off
record. A prompt that implies several variants ("shift, tombstone and
sideflip") produces several records, each marked independent -- never a loop
of `render()` calls inside one script (a second `render()` in one process
segfaults the Kit subprocess).

| Field | Type | Meaning |
|---|---|---|
| `row` | `1..5` | The catalog row matched in Step 1 |
| `build` | Python source | `import pcb_aoi` + `from cdf import CDFJob` + a `build_job() -> CDFJob` following the row's template in `SKILL.md` |
| `plan` | `JobOrderPlan`, serialized with `JobOrderCodec.dumps(plan)` | Output of `job.compile_orders(output_dir=...)` -- the transport-safe truth the Isaac child executes. `JobOrderPlan` holds `mappingproxy` and tuple values, so a bare `plan.model_dump()` is **not** JSON-serializable; the codec (`from cdf import JobOrderCodec`) is the representation the executor's own subprocess boundary uses, and `JobOrderCodec.loads(payload)` on the receiving side restores an equal plan (`tests/unittests/api_parity/test_skill_prompt_yaml_api_parity.py`, `test_emitted_plan_round_trips_through_codec`) |
| `resolved_inputs` | mapping | Every input below with its value **and its source** (`state file`, `prompt argument`, `agent context`, `prompt text`, `fixed default`) |
| `independent_launch` | `true` | Always; tells the DAG this record needs its own process |
| `refusals` | list | Any clause of the prompt not served, with the response text from `SKILL.md` "When to refuse" (e.g. a `bridge` clause dropped from a Row 5 request) |

The record contains no rendered artifacts and makes no claim about them.

## What Airflow supplies

| Airflow-supplied | Required | Used for |
|---|---|---|
| Attachment path (board USD) | **yes -- no production fallback.** A missing attachment stops the record (`SKILL.md`, refusal case 5); the repository coupons stand in only for explicitly selected SQA / dry-run validation in a checkout that carries them (the public release ships none) | `scene.configure(usd_path=...)` |
| `<stem>.components.json` sidecar beside the attachment | when present -- conditional on the attachment, not on the row (Row 1's legacy THT fixture ships one; the maintained coupons do not) | `scene.attach_sidecar(...)` and `source="components"`; otherwise `source="usd"` with explicit `types=` |
| Output root | **yes -- no production fallback.** A synthesized per-run directory is a dry-run convenience only | `compile_orders(output_dir=...)` and the capture pass `output_dir` |
| Fan-out | yes | One process per emitted record |
| Execution environment | yes | For Row 5: `manifold3d` **and** the `evolver` binary (or `SOLDER_EVOLVER_BIN`); their absence fails at `solder_joints(...)` declaration time, before compile |

## Supported parameters, by row

Reference flows (`reference-flows.md`) add explicit prompt arguments on top
of the rows below: seed, scan grid, board side, resolution, named refdes
targets and pin numbers, plus panel instance registration read from the
attachment. Their fixed values are the SQA config values and are listed per
flow in that file; the record's `resolved_inputs` names each one with source
`prompt argument` or `fixed default (flow <letter>)`.

Values come from the precedence order in `SKILL.md` "Inputs"; anything unset
takes the documented default from `fixed-defaults.md`. "Required" means the
API has no default and compile or declaration fails without it.

### Row 1 -- Golden board (THT)

| Parameter | Required | Default / rule |
|---|---|---|
| `usd_path`, `root_path` | yes | `usd_path` is the required attachment. `root_path` is read from the attachment: its sidecar, else the embedded USD metadata (the stage default prim -- `/Board` on every maintained coupon and pcb-builder board). An attachment that cannot be opened is refusal case 5, not a `/Board_CAD_BOARD` fallback. Same rule as `fixed-defaults.md`, "Board identity and paths" |
| `select_components(source=)` | yes | `"components"` when a sidecar is attached, else `"usd"` with explicit `types=` |
| `solder_joints(solver=)` | yes | `"procedural"` -- the catalog's golden board; **no** `add_*` solder call may follow |
| `lighting` | no | `white_light`, exposure `(0,0)`, cone `(120,120)`/`(1,1)`, dome intensity `(1000,1000)` visible |
| `camera(side=, rotation_range=)` | yes | `"bottom"`, `{"z": -90}` |
| `renderer(...)` | yes | pathtracing, `(1920,1080)`, spp 1, total_spp 32, max_bounces 4 |
| `outputs` | no | rgb + bounding_box_2d_tight + semantic_segmentation (colorized) |
| `postprocess.crop` | forbidden | no crop chain on this row |

### Row 2 -- Variation type (THT pin geometry)

| Parameter | Required | Default / rule |
|---|---|---|
| `usd_path`, `root_path` | yes | a board authoring `pcba:mesh:role`; fixture `THT_SCHEMA_U1_coupon.usdc`, `/Coupon` |
| `pin_variation(components=)` | yes (one selector) | `["U1"]` unless the prompt names a verified part; `package_types=` only if package metadata is dependable |
| exactly one of `tilt_range`, `offset_range`, `height_variation`, `cut_angle` | yes | `(5,15)` deg, `(0.1,0.4)` mm, `(0.3,1.0)` mm, `(5,15)` deg |
| `select_components` / `label_components` | forbidden | `pin_variation` targets leads directly |
| camera / renderer / outputs | as Row 1 | rgb only by default |

Validation: a board without `pkg_lead` meshes fails at execution with
"pin_variation found no through-hole leads"; the skill reports insufficient
THT metadata rather than retrying with invented parameters.

### Row 3 -- Good images (SMT)

| Parameter | Required | Default / rule |
|---|---|---|
| mode | yes | board-level (`"N good PCB images"`) or component-level (`"good images of <component>, max M"`) |
| `scan_grid(x_count=, y_count=)` | board-level | most-square factorisation with `x*y >= N`, reported (20 -> 5x4) |
| `select_components(types=)` | component-level | the named component type, which must resolve on the attachment (`fixed-defaults.md`, "Named components must resolve on the attachment"; otherwise refusal case 6) |
| `postprocess.crop(...)` | component-level | `method="bbox_tight"`, `types=["rgb","semantic_segmentation","component_instance"]`, **`class_filter=None`**, `offset=10`, `xform_depth=7` |
| `capture.limits(max_images=)` | component-level | the requested max |
| operations | forbidden | no defects on this row |

Validation: `class_filter` takes the derived semantic class, never a
component-type string -- the latter compiles, reports `"error": ""`, and
writes zero crop files.

### Row 4 -- Structural defects (SMT)

| Parameter | Required | Default / rule |
|---|---|---|
| defect type(s) | yes | one or more of `shift`, `tombstone`, `sideflip` |
| `ratio` | no | `1.0` for "full board coverage", else `0.2`. The executing side selects `max(1, int(pool * ratio))`, so a non-empty selection always yields at least one defect; `count=1` is the explicit spelling for a one-component selection. A record whose selection is empty is not emitted |
| `add_shift(translate_range=, rotate_z_range=)` | yes when shift | `0.2` mm, `15` deg |
| `add_tombstone(angle_range=)` | yes when tombstone | `(70,90)` deg |
| `add_sideflip(angle_range=)` | yes when sideflip | `(70,90)` deg |
| scoping | rule | the narrowed `select_components(types=)` is the scope of every pose operation. **Embedded-metadata board** (`pcba:*` schema, no sidecar): `add_tombstone(..., eligibility_source="usd")` and **no** operation-level `component_types` -- supplying it applied the operation to nothing (verification record, 2026-09-11). **Legacy sidecar board**: `component_types=` on `add_tombstone` is verified and may accompany the narrowed selection; it never replaces it |
| `enable_operation_labels(namespace=)` | yes | `"defect"`, declared after `label_components` -- the defect class must reach the segmentation output |
| `postprocess.crop(...)` | yes | the Row 3 component-level block: `method="bbox_tight"`, `types=["rgb","semantic_segmentation","component_instance"]`, **`class_filter=None`**, `offset=10`, `xform_depth=7`, `output_subdir="cropped"`. Rows 4 and 5 are always reported as per-component defect crops; a record that leaves no file under `cropped/` is a failed record |
| `outputs` | no | rgb + bounding_box_2d_tight + semantic_segmentation (colorized) -- the tight boxes feed the crop |
| ordering | rule | `select_components` -> `add_*` -> `label_components` -> `enable_operation_labels` -> capture -> `postprocess.crop` (compile rejects annotation before selection) |

One picture carrying several defects is one record; one picture per defect
type is several records.

### Row 5 -- Texture defects (SMT solder, USD input)

| Parameter | Required | Default / rule |
|---|---|---|
| defect type(s) | yes | any of `excess`, `missing_solder`, `insufficient`, `bright`, `interrupted`, `dry_contact`; **`bridge` is refused** (partial fulfilment when mixed) |
| `solder_joints(solver=, source=)` | yes, once | `"surface_evolver"`, `"components"` -- `procedural` here is a validation error by design |
| `add_<type>(ratio=, component_types=)` | yes per type | ratio as Row 4 (`max(1, int(pool * ratio))` on the executing side; these calls take `ratio` only, no `count`); types as named (or the row's default scope), resolved on the attachment and verified on both metadata sources |
| `enable_operation_labels(namespace=)` | yes | `"defect"`, as Row 4 |
| `postprocess.crop(...)` | yes | the Row 4 block, `class_filter=None`; a record with no file under `cropped/` is a failed record |
| `outputs` | no | as Row 4 |
| ordering | rule | `solder_joints` before any `add_*`; same select -> operations -> annotate -> capture -> crop order as Row 4 |

Validation: with `manifold3d` or the `evolver` binary missing,
`solder_joints(solver="surface_evolver")` raises at declaration, so a Row 5
record cannot even be produced in a bare environment; the skill reports the
missing prerequisite instead of downgrading the solver.

## Cross-row rules

- **Never ask.** Every unset value takes its default and is listed in
  `resolved_inputs` with source `fixed default`.
- **Never override confirmed board metadata** with the `CAD_BOARD`
  fallback; a sidecar root or the embedded USD default prim wins, and the
  source used is reported.
- **Never substitute a missing attachment or output root.** Both are
  required Airflow inputs; the record stops with refusal case 5. Repository
  coupons and synthesized directories are for explicitly selected SQA /
  dry-run validation only.
- **Never emit a defect record whose selection is empty**, and never
  describe a zero-defect outcome as success: a non-empty selection yields at
  least one defect on the executing side, and a clean board on a defect
  request is a failed record.
- **Never claim a step that does not exist** (Qwen lighting transfer, real-photo
  input, AnomalyGen) -- rows 6-7 are refused, not approximated.
- **Never execute.** The record ends at a clean `compile_orders(...)`.
