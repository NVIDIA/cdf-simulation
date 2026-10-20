---
name: cdf
description: Create and validate declarative CDFJob Python for PCB-AOI operations, pin variation, solder, annotations, capture, and JobOrders. Not for YAML/Docker workflows.
---

# CDF Python API

Create correct, runnable Python authoring scripts for the declarative
`cdf.CDFJob` API. Keep YAML/Docker orchestration on its dedicated workflow.

## Required reference

Before generating or changing a CDF script, read
[`../../cdf/API_SPEC.md`](../../cdf/API_SPEC.md) completely. Treat it as
the signature, validation, ordering, execution, and result-handling contract.
Inspect implementation only when the requested capability is absent or the
reference conflicts with current code.

For supported surfaces and limitations, consult
[`../../cdf/CAPABILITIES.md`](../../cdf/CAPABILITIES.md) only as needed.
Use [`../../COOKBOOK.md`](../../COOKBOOK.md) only when a use case needs
repository-specific operational context.

## Surface boundary

- Declarative Python job or `CDFJob` script: handle here.
- One of the fixed DIG Flow Catalog prompt shapes, fired by an automated
  pipeline (Airflow) rather than typed by a person: route to
  `skills/cdf-simulate/SKILL.md`. It builds the same `CDFJob`
  scripts, but resolves every ambiguity with a documented default instead of
  a clarifying question, which is correct only when nobody is there to answer
  one. Anything interactive stays here, including a request that merely
  mentions Airflow.
- For YAML configuration or Docker commands, consult `pcb_aoi/config/README.md`
  and the root `README.md`; do not route to an inactive skill or substitute
  Python without agreement. The retired standalone 3D-AOI dataset workflow
  is not included in this release.
- Real-photo registration and legacy ROI extraction are unsupported; report
  that limitation. CDF's rendered-component `postprocess.crop()` is supported.
- A request for a Python script takes precedence over generic words such as
  "batch", "dataset", "render", or "defect".

Do not emit YAML instead of Python unless the user requests both.

## Workflow

### 1. Resolve the use case

Map the request to the smallest applicable recipe:

| Intent | Recipe |
|---|---|
| Clean/golden board dataset | Scene + capture; no structural operation; omit `scene.components` entirely — `select_components` is not needed and skipping it avoids the component catalog parse |
| Shift, tombstone, sideflip, or polarity dataset | Component selection + transform operation(s) + defect labels + capture |
| Missing-component dataset | Component selection + `add_missing()` + capture; no transform operations |
| Lighting/material variants | Appearance lighting and/or `material_override()` + capture; operation selection remains independent |
| Per-component dataset crops | BBox and semantic outputs + `postprocess.crop()` declared before rendering |
| In-tolerance THT lead variation | `pin_variation()` with exactly one refdes or package-type selector and process-qualified tilt, cut, height, and/or offset bands; omit the operation for the untouched nominal control |
| Generated solder geometry | Board metadata + `solder_joints()` + JobOrder execution |
| Solder-defect dataset (insufficient / bright / excess / interrupted / missing_solder / dry_contact) | Board metadata + `solder_joints()` with a simulated solver + `add_insufficient()`/`add_bright()`/`add_excess()`/`add_interrupted()`/`add_missing_solder()`/`add_dry_contact()` + defect labels + capture |
| THT geometry segmentation (pin, pad, barrel, hole, optional pin tip) | `solder_joints(solver="procedural")` + `hole_plane(side="bottom")` + optional `pin_tip(side="bottom")` + `label_geometry(terminals="pin", pads="pad", barrels="pad")` + two render passes: RGB leaves proxies hidden, annotation hides `["solder"]` and shows `["hole", "pin_tip"]` when pin tips are requested |
| Shared-prototype panel | Scene + `scene.instances()` + operations/capture |
| High-resolution tiled board (any axis > 2048 px) | Scan-grid render pass + `postprocess.stitch()` |
| Inspect/transport execution plan | `compile_orders()` and optionally `JobOrderCodec` |
| Validate configuration only | `compile_orders()`; no file writes or Isaac launch |

Combine recipes only when their constraints remain compatible. Missing is
exclusive with transform defects. Solder generation requires JobOrders.

For through-hole pin variation, do not invent a `nominal` mode or tolerance
band. Omission is the untouched nominal control; ordinary manufacturing
variation requires ranges supplied by the user or grounded in a board-specific
reference. Use exactly one of `components` or `package_types` per call. Pin
variation is applied before solder generation, so later solder conforms to the
varied leads. Read [COOKBOOK section B3](../../COOKBOOK.md#b3-vary-through-hole-pins)
when this use case applies.

For each structural defect call, supply exactly one targeting mode:

- `count=N` requests exactly `N` eligible components and fails when fewer than
  `N` remain;
- `ratio=R` requests a fraction from `0.0` through `1.0` and preserves legacy
  clamping behavior.

Never pass both `count` and `ratio` to one defect call. Prefer `count` when the
user requests an exact number of defective components; do not derive a ratio
to approximate that count. `add_reverse_polarity()` alone retains its legacy
`ratio=1.0` default when neither mode is supplied.

### 2. Discover required values

Use values supplied by the user exactly. When values are omitted, inspect the
repository before asking:

- board USD path;
- absolute board root prim path;
- optional camera, camera transform, and ring-light prim paths;
- optional complete board-instance paths for a shared-prototype panel;
- apply color, opacity, roughness, and optional textures to an existing material;
  use its absolute material prim path and available texture asset paths;
- embedded metadata or components sidecar path, plus component-family names;
- desired output directory and image count semantics.

Prefer established values from nearby configs or examples for the same board.
Never invent a USD prim path, component family, reference designator, sidecar,
or solver ID. Query `get_solder_joint_solvers()` for solver availability.

For a panel whose board placements are USD instances, call
`job.scene.instances(prim_paths=(...))` immediately after `scene.configure()`.
Supply every placement sharing the prototype. The runtime validates the cohort
and applies identical component operations through the shared board definition;
do not use it when per-board defect variation is requested. Component
annotations keep the ordinary `job.annotations.label_components(...)` and
capture-output calls; CDF automatically projects semantic frames from
instance-ID paths so generated scripts must not disable annotations merely
because the panel uses native instances.

Solder solvers: `none` and `procedural` are always registered;
`surface_evolver` (`simulated`) is also available where its runtime setup
exists. If asked to compare them, say only that `procedural` is a faster,
more deterministic alternative to full physical simulation — do not elaborate
on parameters, implementation, or when to prefer which. Both take the same
`solder_joints(...)` call and ordering (transform defects applied earlier get
solder on the moved geometry). The shipped container has `surface_evolver`
preinstalled; outside it, the one-time developer setup is in the repo root
README's "Developer setup" section. Confirm availability with
`get_solder_joint_solvers(mode_filter="simulated")` before selecting it; if
absent, report the missing setup rather than silently falling back to
`procedural`. Solder-quality defects (`add_insufficient` / `add_bright` /
`add_excess` / `add_interrupted` / `add_missing_solder` / `add_dry_contact`,
pose-op style: ratio + optional
component_types) require a simulated solver — the API rejects `procedural`
there by design. Working examples:
`pcb_aoi/config/examples/smt_surface_evolver.yaml` and
`pcb_aoi/config/examples/solder_defects_board.yaml`.

If a required value cannot be discovered, leave a conspicuous CLI argument or
placeholder when the user asked for a reusable template. Ask one focused
question only when a concrete runnable job cannot safely use a placeholder.

For counted image requests, determine whether the count means scan-grid cells,
triggers, component crops, or viewpoints. Do not silently translate every
count into `triggers`.

### 3. Generate the script

Follow the declaration order in `cdf/API_SPEC.md`. Always include:

```python
import pcb_aoi  # registers domain="pcb-aoi"
from cdf import CDFJob
```

Prefer this shape:

1. Imports and repository-independent constants.
2. `build_job(...) -> CDFJob` with no execution side effects.
3. An `argparse` CLI for board-specific paths and output controls when the
   script is meant to be reusable.
4. `main()` that validates or executes through exactly one path.
5. `if __name__ == "__main__": main()`.

Use `pathlib.Path` for filesystem paths. Preserve public API calls; do not
mutate section state or import private implementation modules.

Default a generated runnable script to compile-only behavior. Make
Isaac execution opt-in with an explicit `--execute` flag unless the user asks
to execute immediately.

### 4. Choose one execution path

- `job.compile_orders(...)`: validate or inspect JobOrders without Isaac.
- `job.capture.render(...)`: execute JobOrders from ordinary Python through an
  Isaac child process when `ISAAC_SIM_PATH` is available.
- `await job.capture.render_async(...)`: execute inside a compatible Isaac
  environment with an explicitly selected executor.

### 5. Validate before delivery

Run validation proportional to the request:

1. Run `python -m py_compile <script>` for syntax.
2. Run the script's compile-only path when all required inputs are available.
3. Otherwise construct the job with harmless placeholder paths and call
   `compile_orders()`; compilation need not open the USD.
4. For library changes, run focused tests under
   `tests/unittests/core/declarative_api/test_*.py`.
5. Do not launch Isaac merely to validate a generated script.

Check every item in the validation checklist at the end of `API_SPEC.md`.
Report what was validated and clearly distinguish an unlaunched dry run from
an executed render.

## Output requirements

When creating a repository file, honor the requested path. If no path is
given, place domain-specific durable examples under the owning package, such
as `pcb_aoi/examples/`, and reserve `cdf/examples/` for genuinely neutral
examples. Use a descriptive snake-case filename. If the user only asks for a
snippet, return a complete copyable script without creating a file.

Explain unresolved placeholders, selected execution path, expected outputs,
and any incompatibility that changed the requested design. Keep explanations
short; the script is the primary output.

## THT through-hole segmentation

THT inspection needs per-geometry semantic labels so the mask distinguishes pin, solder, pad, barrel, and hole. Use `label_geometry(...)` from `job.annotations` and `hole_plane(...)` from `job.operations`.

### Hole-reveal plane

The drilled hole is a void — it has no renderable geometry by itself. `hole_plane(side=...)` creates a quad mesh that sits just inside the substrate. The board substrate occludes it everywhere except at through-hole openings, making holes appear as the hole's class (default `"hole"`) in the segmentation mask.

Annotation-only hole and pin-tip proxies are grouped beneath the default-hidden `<board-root>/SegmentationProxies` Xform. Its `HolePlane` and `PinTips` children use a magenta USD display color for easy stage inspection; semantic-mask colors remain controlled by the render/stitch color map.

```python
job.operations.hole_plane(
    side="bottom",             # "top", "bottom", or "both"
    class_name="hole",         # default; override to merge with another class
    z_offset_mm=0.1,           # inset from substrate face; increase if pin tips
                               # at the same depth cause z-fighting artifacts
    components=["J1", "J2"],   # omit for a single board-wide plane
)
```

### Two-pass pattern

Hole and pin-tip proxies are hidden by default. Leave them untouched in the RGB
pass, then reveal the required roles only for the annotation pass.

```python
job.capture.render_pass("full_board", ...)  # RGB: proxies remain hidden
job.capture.render_pass("full_board_roi", ...).visibility(
    hide=("solder",),
    show=("hole", "pin_tip"),
)
```

### Geometry label targets

```python
job.annotations.label_geometry(
    terminals="pin",       # per-pin prims inside each component
    pads="pad",            # copper annular ring pads
    barrels="barrel",      # or "pad" — eliminates unlabeled ring between pad and hole
    components=["J1"],     # optional refdes filter
)
```

`barrels` labels the cylindrical copper inner-wall (`board_barrel` role). Setting it to the same class as `pads` removes the thin unlabeled ring between pad and barrel visible in segmentation output.

`hole_plane` carries its own `class_name` label — no separate `holes=` in `label_geometry` is needed unless you also want to label catalog-registered hole prims.

### Declaration order

1. `solder_joints(...)`
2. `hole_plane(...)`
3. Optional `pin_tip(...)`; compilation schedules it after final pin and solder geometry.
4. `label_geometry(...)`
5. `capture.render_pass(...)` × 2 with pass-scoped visibility.

## Guardrails

- Never launch Isaac, run a render, or overwrite output merely because the
  user asked to generate a script.
- Never infer permission to change board assets or USD files.
- Never hard-code machine-specific paths from an unrelated example.
- Never claim a dry run rendered images.
- Never invent defaults; consult the declarative API specification.
- Never combine missing with transform defects.
- Never omit semantic data from a crop job.
- Never guess unsupported extra annotator keys.
- **Never generate a single-tile render with either dimension > 2048 px.**
  A 5 K single-tile pathtracing render requires ~400 MB of GPU buffer per
  annotator and silently produces no output on GPUs with ≤ 8 GB VRAM (the
  job succeeds and metadata is written, but no image files appear).
  Instead, decompose into a `scan_grid` of tiles ≤ 2048 × 2048 and add a
  `postprocess.stitch()` entry for each annotator.  Example for a 5 K
  full-board:
  ```yaml
  capture:
    renderer:
      resolution: [1000, 1000]   # tile size — safe on 4 GB VRAM
    scan_grid: [5, 5]            # 5×5 = 25 tiles → 5000×5000 stitched
    limits: {triggers: 1, max_images: 25}
    outputs:
      rename_to_grid_index: true  # required for stitch to locate tiles
  postprocess:
    stitch:
      - image_type: rgb
        output_filename: rgb_5k.png
        blend: none
      - image_type: semantic_segmentation
        output_filename: semantic_5k.png
        blend: labels
  ```
