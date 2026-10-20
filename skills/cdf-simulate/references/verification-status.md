# Verification status

Reference for `skills/cdf-simulate`. Every one of the five
implemented catalog rows was actually executed, crops included -- this is the
record of those runs and the defects they surfaced, not a plan. The
fixtures it names are not shipped in the public release.

These executions were done **by the maintainer, to verify the templates**.
The skill itself does not execute: under the SDG Workflow Skill Architecture
it is component-owned guidance that compiles a job and hands the payload to
the workflow-operations skill (see `SKILL.md`, "Validate, then hand off"). So
"verified end to end" below means the *templates* were proven to render,
which is what makes handing them off safe -- not that the skill renders.

Every row below was built as real Python and run end-to-end through
`job.capture.render(...)` against a real on-repo fixture (not just
`compile_orders()` dry-run validation), including the `postprocess.crop`
step where the row has one. This surfaced and fixed four real bugs that
compile-only validation missed entirely:

1. Row 2's originally-documented fixture cannot run `pin_variation` at all
   (wrong board metadata) — switched to the correct one.
2. Row 3's declaration order (`select_components` before
   `label_components`) is enforced at compile time, not just a style
   preference.
3. `postprocess.crop(class_filter=...)` silently drops every crop when
   given the component-type string instead of the derived semantic class —
   compiles and executes with `"error": ""` and produces zero files while
   reporting success -- the failure is silent in every channel a caller
   normally checks. Fixed by defaulting to `class_filter=None`.
4. Sequential `job.capture.render(...)` calls in one Python process crash
   the second Kit subprocess (segfault) — each variant needs its own
   process.

| Row | Fixture used | Result |
|---|---|---|
| 1. Golden board | `THT_TO254P762H300_U34_coupon.usdc` + sidecar | `"error": ""`, real rgb+bbox+semantic output |
| 2. Variation type | `THT_SCHEMA_U1_coupon.usdc` (not the Row-1 fixture) | `"error": ""`, real render showing tilted leads in their barrels |
| 3. Good images (board + component) | `SMT_0805_C18_coupon.usdc` + sidecar | Both variants `"error": ""`; component-level crop produces a real per-component crop file once `class_filter=None` |
| 4. Structural defects (shift/tombstone/sideflip) | `SMT_0805_C18_coupon.usdc` + sidecar | All 3 variants `"error": ""` with real renders (tombstone visibly shows the part standing on end) **and** a real crop file each, run as 3 separate process launches |
| 5. Texture defects (solder, excess) | `SMT_0805_C18_coupon.usdc` + sidecar, `SOLDER_EVOLVER_BIN` pointed at a built `evolver` binary | `"error": ""`, real solder solve + defect + render + crop file |

## Re-verified after the rebase onto `main` (2026-09-09)

The branch was rebased from its original base (`2c62e347`) onto `main`
`e738f48b`, 15 commits later, which touched `cdf/sections.py`,
`pcb_aoi/sections.py`, the config models now under `pcb_aoi/config/*`, and added the dataset-manifest
pipeline. Every row was rebuilt from the templates in `SKILL.md` and run
through `compile_orders()` — no Isaac, no render — on the rebased tree:
**12/12 compiled**, covering all four `pin_variation` axes, both Row 3 modes,
three Row 4 structural variants, and both Row 5 solder cases (with
`SOLDER_EVOLVER_BIN` pointed at a built `evolver` binary and `manifold3d`
installed).

No signature the skill depends on changed in those 15 commits. `visibility()`
gained a `show=` kwarg and `operations.pin_tip(...)` was added; neither is used
by any catalog row.

The rebase check did surface three real doc defects, all now fixed:

1. **`import pcb_aoi` was never stated.** Following `SKILL.md` alone and
   writing only `from cdf import CDFJob` fails on the first line with
   `unknown domain 'pcb-aoi' (registered: none)`. The templates now say so.
2. **Rows 4 and 5 had no code template at all** — only prose and an entry in
   the Examples table. That forced an agent to derive the capture block from
   Rows 1-2 and to invent the required magnitude kwargs. Both rows now carry a
   full template.
3. **The required structural magnitudes were undocumented.**
   `add_shift(translate_range=, rotate_z_range=)` and
   `add_tombstone`/`add_sideflip`(`angle_range=`) have no API defaults, and a
   PM prompt never supplies them — so the skill's "every unspecified detail has
   a documented default" contract was broken for all of row 4. Fixed values are
   now in `fixed-defaults.md`, taken from `pcb_aoi/examples/allegro_defect_job.py`.

Also corrected: this skill previously claimed a missing `evolver` binary
surfaces later than a missing `manifold3d`. Both raise inside
`operations.solder_joints(...)` (`pcb_aoi/sections.py:597` calls
`resolve_solver`); the claim was wrong when written, not broken by the rebase.

## SQA prompt set executed in the candidate container (2026-09-11)

Eight prompts (P01-P08) use the maintained embedded-metadata coupons and each
has a YAML twin under
`tests/functional/skill_prompts/`. All eight twins were executed through the
packaged `config` entrypoint in the MR's own candidate image
(`cdf-simulation@sha256:25eee2c5…`, source commit `58bd2c25`) on an L40S,
one process per case, 640x360 at 4 spp:

| Case | Row | Result | What the run reported |
|---|---|---|---|
| P01 golden THT | 1 | pass, 51 s | `solder_joints.attach`: 3 generated joints on `J13`; rgb + bbox + semantic written |
| P02 pin tilt | 2 | pass, 50 s | `operation.pin_variation`: 3 leads, 0 breaches; requested 13.9 deg, delivered 4.75 deg, `limited_by: barrel` on all three |
| P03 good board, 4 images | 3 | pass, 54 s | 2x2 grid -> `rgb_0000..0003` with semantic and bbox per frame |
| P04 good component, max 3, crop | 3 | pass, 47 s | `postprocess.crop` executed; `cropped/capacitor/rgb/frame0000_C22.png` and its semantic twin |
| P05 tombstone, full coverage | 4 | **fail, then pass** | see below; after the fix `operation.tombstone`: count 1, `metadata.defects` carries one `tombstone` on `C22`, angle -81.9 deg |
| P06 shift, full coverage | 4 | pass, 47 s | `operation.shift`: count 1; the part is visibly off its pads |
| P07 sideflip, full coverage | 4 | pass, 49 s | `operation.sideflip`: count 1, 81.9 deg; the part lies on its side |
| P08 excess solder, full coverage | 5 | pass, 60 s | `solder_defects.apply`: count 1; `solder_joints.generate` on `surface_evolver`, `status: generated`, no fallback, 0 warnings logged |

The YAML-versus-Python-API parity test (then
`tests/unittests/api_parity/test_skill_prompt_parity.py`, now
`test_skill_prompt_yaml_api_parity.py`) passed 8/8 with the container's
interpreter, where `surface_evolver` is registered; in the api-parity CI
lane P08 skips under the solver rule in `tests/skip_policy.json`. That test
compares each YAML with a hand-written Python API build, not with `SKILL.md`.

**The one failure was a template defect, now fixed in `SKILL.md` Row 4.**
`add_tombstone(component_types=["C_0805_2012Metric"])` on the embedded-metadata
coupon selected the component and applied it to none: `ValidationError:
tombstone selected from 1 components but applied none of them. Their prim
paths do not resolve under '/Board' on this stage`. The same job scoped by
`select_components(types=[...])` with no operation-level list applied cleanly.
The original verification ran on the legacy sidecar boards, where the
operation-level list works, so the template carried an assumption the
maintained coupons do not share. Shift and sideflip, which never had the
kwarg, were unaffected.

Still not exercised by this set: rows 1-5 from a *raw prompt through an
agent* -- the twins fix what the agent must arrive at, but the raw
prompt-to-plan path still needs separate SQA coverage.

## Row 4/5 contract: operation labels and per-component crops (2026-09-15)

Review of this MR found two job definitions for rows 4 and 5: the
reference (`class-filter-taxonomy.md`) required
`enable_operation_labels(namespace="defect")` and the per-component crop,
while the `SKILL.md` templates, the YAML twins P05-P08, the Python builders
and the functional rows had neither. The contract is now the reference's:
every Row 4/5 record carries operation labels, rgb + bounding_box_2d_tight +
semantic_segmentation outputs, and the `class_filter=None` crop, and the
functional rows `skill-row4-*` / `skill-row5-*` fail when no non-empty file
lands under `cropped/`. The 2026-09-11 table above predates that change --
those runs rendered rows 4 and 5 without labels or crops -- so the
labels + crop half of the contract on the maintained coupons is verified by
the functional pipeline of this MR, not by the table.

The same review corrected the coverage claim for one-component coupons: the
executing side selects `max(1, int(pool * ratio))`
(`scripts/sdg/standalone/defects_utils/selection.py`;
`cdf/execution/isaac/job_order_runtime.py` for solder), so `ratio=0.2` on
the one-component SMT coupon applies one tombstone, not zero, and a pose
operation that selects from a non-empty pool and applies nothing raises.
SQA case N07 and its new twin
`tests/functional/skill_prompts/row4_tombstone_default_coverage.yaml`
(functional row `skill-row4-tombstone-default-coverage`) exercise that
variant; the earlier statement that it renders a clean coupon with exit 0
had not been run and was wrong.

Not exercised: the other 5 solder-defect methods beyond `add_excess`
(insufficient/bright/interrupted/missing_solder/dry_contact — same call
shape, reasonable to assume they work but unverified); multi-component
(`>1` refdes) scenes; boards other than the two coupon fixtures and
`THT_SCHEMA_U1_coupon.usdc`; the `count=` targeting mode (only `ratio=` was
exercised); board-level crops at production resolution/SPP (all runs used a
small 640x360 smoke-test resolution, not the 1920x1080 the Row-1 template
defaults to); running this skill's actual routing/prompt-matching logic
end-to-end from a raw PM prompt string (all verification here built the
`CDFJob` scripts directly from the row templates, not by having an agent
read a prompt and follow this skill's Workflow section unassisted).
