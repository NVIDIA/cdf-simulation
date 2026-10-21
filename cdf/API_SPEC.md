# CDF Python API specification for script authors

This is the implementation-level contract for generating Python scripts that
use the declarative PCB-AOI API. It is written for both human authors and
code-generating skills. When this document and an example disagree, follow
this document and the public signatures in `cdf/` and `pcb_aoi/`.

## Script-generation contract

Generated scripts MUST:

1. Import `pcb_aoi` before constructing the job. The import registers the
   `"pcb-aoi"` domain.
2. Import public API objects from `cdf`, not from private modules.
3. Construct `CDFJob` with keyword arguments.
4. Declare the scene before every other area.
5. Attach a sidecar before referring to its `name`; omit
   `attach_sidecar(...)` when the board uses embedded USD metadata.
6. Call `scene.select_components(...)` before structural operations or
   `annotations.label_components(...)`.
7. When rendering, configure the required capture calls: `renderer`, `camera`,
   `scan_grid`, and `outputs`. A serialization-only job may omit capture.
8. Use `pathlib.Path` for local paths and placeholders or CLI arguments for
   board-specific values. Do not invent USD prim paths or component families.
9. Use exactly one execution path from [Execution paths](#execution-paths).
10. Catch `CDFError` at the script boundary when a friendly diagnostic is
    needed. Do not suppress the exception inside a reusable `build_job()`.

Generated scripts MUST NOT:

- mix `add_missing()` with shift, tombstone, sideflip, or reverse-polarity;
- call the same structural operation more than once on one job;
- pass `source`, `dry_run`, or `runner` to `postprocess.crop(...)`; declare the
  crop before `capture.render()` so it executes in the same JobOrder plan;
- use crop types without at least one name containing `"semantic"`;
- mutate `job` section attributes such as `scene.config` or `operations.ops`;
- assume that a declaration immediately changes a USD stage. Declarations record
  intent until an executor runs the job.

Use this declaration order unless a task explicitly requires fewer sections:

```text
import/register domain
-> create job
-> scene.configure
-> scene.instances (optional shared-prototype board cohort)
-> scene.attach_sidecar(s) (only for legacy sidecar metadata)
-> scene.select_components
-> appearance
-> structural operation(s)
-> solder_joints (optional)
-> annotations
-> capture pass declarations (optional when serializing only)
   -> capture.output_start (optional)
   -> capture.renderer
   -> capture.camera
   -> capture.scan_grid
   -> capture.limits (optional)
   -> capture.augmentation (optional)
   -> capture.outputs
-> postprocess.crop without source (optional; requires capture)
-> postprocess.stitch (optional; requires an orthographic scan-grid capture)
-> compile or execute
```

## Minimal valid render job

This is the preferred skeleton for a generated script. Values marked as
placeholders must come from the user, a known board manifest, or repository
configuration.

```python
from pathlib import Path

import pcb_aoi  # required side effect: registers domain="pcb-aoi"
from cdf import CDFJob


def build_job(board_usd: Path, board_root: str) -> CDFJob:
    job = CDFJob(domain="pcb-aoi", random_seed=0)
    job.scene.configure(usd_path=board_usd, root_path=board_root)
    job.scene.select_components(types=None)
    job.capture.renderer(
        mode="pathtracing",
        resolution=(1920, 1080),
        spp=1,
        total_spp=32,
        max_bounces=4,
    )
    job.capture.camera(
        rotation_range={"x": (0, 0), "y": (0, 0), "z": -90},
    )
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.outputs(rgb=True, image_format="png")
    return job
```

Without `attach_sidecar(...)`, component selection reads embedded metadata from
the USD stage. Calling `attach_sidecar(...)` anchors the job to the legacy
sidecar source instead; the two metadata sources are not mixed.
Prefer explicit component types when a job should operate on only part of the
board.

## Execution paths

Choose one path; they are alternatives.

### Validate without starting Isaac Sim

```python
plan = job.compile_orders(output_dir="output")
print(plan.metadata["api_version"])
print([order.order_type for order in plan.ordered()])
```

`CDFJob.compile_orders(...)` validates the description and returns the same
immutable JobOrder plan consumed by execution. It neither writes YAML nor
launches Isaac Sim.

### Execute JobOrders synchronously

```python
result = job.execute(
    output_dir="output",
    windowed=False,
)
print(result.success)
```

This method runs from ordinary Python. It compiles and serializes JobOrders,
launches Isaac Sim's bundled Python, executes the plan through the default
in-process executor in that child, and transports `JobResult` back to the
caller. It does not generate flow/PCBA YAML or invoke `sdg_pipeline.py`. Set
`ISAAC_SIM_PATH` to the Isaac installation directory or bundled Python
launcher. Use `compile_orders()` for a dry run.

```python
job.execute(
    *,
    output_dir: str | Path | None = None,
    serialization_path: str | Path | None = None,
    windowed: bool = False,
) -> JobResult
```

`output_dir` is required only when the job declares capture passes. The
optional `serialization_path` exports the flattened composed stage after
deterministic stage mutations and before capture begins. The backwards-compatible
`job.capture.render(output_dir=..., serialization_path=..., windowed=...)`
convenience delegates to `job.execute(...)`. It requires at least one render
pass or a `serialization_path`.

| Render passes | `serialization_path` | Result |
|---|---|---|
| configured | set | Export the final stage, then render all passes. |
| configured | omitted | Render all passes; `output_dir` is required. |
| absent | set | Export the final stage without creating render products. |
| absent | omitted | Reject the call before launching Isaac Sim. |

A partially configured render pass is invalid in every row, including when a
serialization path is supplied. Supplying `output_dir` without a render pass
is also rejected.

### Execute several jobs in one Isaac process

A dataset is several variants of one board, and running them one
`job.execute()` at a time pays the same fixed cost per variant. Measured on a
KiCad board at 1024x1024: 101.5s for a render in a fresh process, 4.8s for the
second render in the same one. Roughly 40s of the difference is booting Kit and
roughly 98s is RTX pipeline and shader compilation, which the replicator
orchestrator waits out on the first render and never again.

```python
from cdf import BatchJob, render_batch

results = render_batch([
    BatchJob(golden_job, output_dir=out / "golden"),
    BatchJob(tombstone_job, output_dir=out / "tombstone"),
])
for result in results:
    print(result.success, result.total_duration_s)
```

```python
render_batch(
    items: Sequence[BatchJob],
    *,
    windowed: bool = False,
    experience: str | None = None,
    isaac_path: str | Path | None = None,
    timeout: float | None = None,
) -> tuple[JobResult, ...]
```

`BatchJob(job, output_dir=..., serialization_path=...)` carries exactly the
destinations `job.execute` takes, and each item is validated against the rows
of the table above before the child launches -- a batch that cannot run costs
nothing rather than failing minutes in.

Every job runs with its own execution context, so one job's stage and operation
state does not reach the next. A job that FAILS does not end the batch: the
boot and the compilation the remaining jobs need are already paid for, so the
failure is reported in that job's own result, in the position it ran, and the
rest still run. Check `result.success` per item rather than expecting a raise.

The Isaac window belongs to the process, not the job, so its resolution is the
largest any job in the batch asks for. Results carry the boot cost in
`startup_s` on the first item only; the others did not pay it.

### Execute JobOrders in-process

Use this path for code already running inside the environment required by the
chosen executor.

```python
# Inside a running Isaac Sim Python environment:
from cdf import create_default_isaac_executor

result = await job.capture.render_async(
    output_dir="output",
    executor=create_default_isaac_executor(),
)
if result.capture_output:
    print(result.capture_output["output_dir"])
```

```python
job.compile_orders(
    output_dir: str | Path | None = None,
    serialization_path: str | Path | None = None,
) -> JobOrderPlan

await job.capture.render_async(
    output_dir: str | Path | None = None,
    *,
    executor: JobExecutor,
    serialization_path: str | Path | None = None,
) -> JobResult
```

The JobOrder path enforces declaration order and prerequisites. In particular,
the scene must be declared first; referenced sidecars and component selection
must already have been declared.

## Constructor and compile API

```python
CDFJob(*, domain, random_seed=0)
job.compile_orders(
    output_dir: str | Path | None = None,
    serialization_path: str | Path | None = None,
) -> JobOrderPlan
```

- `domain`: registered domain name. For this repository use `"pcb-aoi"` after
  `import pcb_aoi`.
- `random_seed`: converted to `int`; it controls randomized job behavior.
- Section attributes: `scene`, `appearance`, `operations`, `annotations`,
  `capture`, and `postprocess`.
- `JobOrderPlan` has immutable `orders`, immutable `metadata`, `version`, and
  `ordered()`.

## Component filters

Every call that selects a subset of the board accepts `component_filter=`, an
exact metadata-field selector. Import it from `cdf`:

```python
from cdf import ComponentFilter

ComponentFilter(
    refdes=None,           # exact component reference designators
    component_type=None,   # exact semantic component types
    package_type=None,     # exact package classifications
    footprint_id=None,     # exact source footprint identities
    side=None,             # exact board-side values, e.g. ("Top",)
    process_class=None,    # exact scalar manufacturing process classification
    process_classes=None,  # members of the plural process-class property
)
```

- Every field is a tuple of non-empty, unique strings, or `None` when unused.
  A list is accepted and normalized; an empty or duplicated field is an error.
  The model is frozen and rejects unknown fields.
- Values within one field are alternatives (OR). Populated fields are combined
  as an intersection (AND). A filter on an operation, annotation, or ROI call
  is further intersected with the component population that
  `scene.select_components(...)` resolved.
- A mapping of the same shape is accepted wherever a `ComponentFilter` is, for
  example `component_filter={"side": ["Top"], "process_classes": ["SMT"]}`.
- `ComponentFilter(...).to_transport()` returns the JSON-compatible mapping
  that is recorded in the compiled plan.

These calls accept `component_filter=`:

| Call | Mutually exclusive with |
| --- | --- |
| `scene.select_components` | `types` |
| `operations.add_shift`, `add_sideflip`, `add_missing` | — (no legacy selector) |
| `operations.add_tombstone`, `add_reverse_polarity` | `component_types` |
| `operations.pin_variation` | `components`, `package_types` (supply exactly one of the three) |
| `operations.solder_joints` | `component_types` |
| `operations.hole_plane`, `operations.pin_tip` | `components` |
| `operations.add_insufficient`, `add_excess`, `add_bright`, `add_interrupted`, `add_missing_solder`, `add_dry_contact` | `component_types` |
| `annotations.label_geometry` | `components` |
| `capture.roi_selection` | every other selector (supply exactly one) |

Supplying both members of a mutually exclusive pair raises `ValidationError`
at declaration time. The YAML configuration layer rejects the same
combinations before it reaches these calls.

## Scene API

All methods return `job.scene` for chaining.

```python
job.scene.configure(
    *,
    usd_path,
    root_path,
    camera_path=None,
    camera_xform_path=None,
    ring_light_root=None,
    horizontal_aperture=None,
)
```

- `usd_path`: required non-empty path.
- `root_path`: required absolute USD prim path beginning with `/`.
- Camera and ring-light paths are optional strings.
- `horizontal_aperture` is an optional numeric camera override.

```python
job.scene.instances(
    *,
    prim_paths: list[str] | tuple[str, ...],
)
```

- Registers the complete ordered set of board placements that share one USD
  prototype. The first path is the representative instance.
- Every path must be an absolute, non-root, unique prim path below
  `scene.configure(root_path=...)`.
- Runtime validation requires every path to be a USD instance, all paths to
  resolve to the same prototype and stage-local prototype source, and the list
  to include every matching instance in the composed stage.
- Component selection, defects, annotations, and solder are calculated once
  against the shared board definition and propagate identically to all
  registered instances. Per-instance variation is not supported.
- Instance cohorts require embedded USD component metadata. A legacy
  `pcb.components` sidecar describes a standalone board and cannot be combined
  with `scene.instances(...)`.
- ROI selection expands each selected component across every registered board
  placement. Full-board capture continues to frame the configured scene root.
- Omit this call for a standalone board. It is supported by JobOrder execution,
  not the legacy `compile()` renderer.

```python
job.scene.attach_sidecar(*, name, kind, path)
```

- `name`: unique job-local name used by later calls.
- `kind`: exactly `"pcb.components"` or `"pcb.sim_manifest"`.
- `path`: sidecar path. Attaching does not prove that the file exists; runtime
  loading may require a readable file.

```python
job.scene.select_components(*, source=None, types=None, component_filter=None)
```

- `source`: previously attached sidecar name, or `"usd"` for the board's
  embedded metadata. Omitting it records `None`, which resolves to embedded
  metadata at execution but is a *different declaration* from an explicit
  `"usd"` — see [Matching a YAML config](#matching-a-yaml-config-plan-for-plan).
- `types`: list or tuple of component-family strings.
- `component_filter`: a `ComponentFilter` or equivalent mapping. Mutually
  exclusive with `types`; supplying both raises `ValidationError`.
- When both `source` and `types` are supplied, explicit `types` win.
- When `types` is omitted, a readable selected `pcb.components` sidecar
  supplies family names; otherwise selection falls back to `"ALL"`.

**When to call `select_components` (and when to skip it)**

`select_components` must be called before any feature that reads component
metadata: `annotations.label_components(...)`, `annotations.label_geometry(...)`,
structural defect operations (`add_tombstone`, `add_shift`, etc.), and
`operations.solder_joints(...)` with a refdes filter. Skipping it is valid and
cheaper for jobs that need none of these — for example a pure golden-board RGB
render, a lighting/material variant dataset, or a USD geometry export. In those
cases omit the `select_components` call (or set `components: null` in the YAML)
and the component catalog parse is skipped entirely, which is a measurable saving
on large boards with hundreds of components.

## Appearance API

All methods return `job.appearance` for chaining.

```python
job.appearance.lighting(
    *,
    mode,
    preserve_scene_light_color=False,
    exposure_range=None,
    cone_angle_range=None,
    cone_softness_range=None,
    dome_fill=None,
    layers=None,
    dome=None,
)
```

- Range values are ordered `(low, high)` numeric pairs.
- `mode` is required and names the rig: `rgb_light` (takes `layers`, plus an
  optional `dome_fill`), `white_light` and `dome_light` (take `dome`), or
  `usd_light` (takes neither; the board USD's own lights are reused).
- Handing a rig another rig's payload is an error rather than a silent no-op.
  `white_light` pins the dome to white, so authoring a colour there is rejected;
  use `dome_light` to author one.
- `dome_fill`, each entry in `layers`, and `dome` are mappings whose
  allowed keys are `intensity`, `color_r`, `color_g`, and `color_b`.
- Every value in those mappings is an ordered numeric pair, including a
  constant one: `{"intensity": (1000.0, 1000.0)}` is how a fixed intensity is
  written. A bare scalar is rejected by the Python API; the YAML surface
  accepts one and expands it to `(v, v)` before calling here.
- `dome` additionally accepts `visible` (a bool), so the white/dome rigs are
  written as `dome={"intensity": (1000.0, 1000.0), "visible": True}`. `layers`
  and `dome_fill` entries reject `visible`. The YAML interpreter includes the
  key only when it is true — `visible: false` produces a dome mapping with no
  `visible` key at all, not `visible: False`.
- `exposure_range`, `cone_angle_range`, and `cone_softness_range` are pairs for
  the same reason: a YAML `exposure: 0.0` reaches this call as
  `exposure_range=(0.0, 0.0)`.

```python
job.appearance.components(*, vantablack=False)
```

```python
job.appearance.material_override(
    name,
    *,
    material_path,
    diffuse_texture=None,
    roughness_texture=None,
    normal_texture=None,
    diffuse_color=(0.2, 0.2, 0.2),
    diffuse_tint=(1.0, 1.0, 1.0),
    albedo_brightness=1.0,
    opacity=1.0,
    roughness=0.5,
    roughness_texture_influence=1.0,
    metallic=0.0,
    specular_level=0.5,
    normal_strength=1.0,
    normal_format="opengl",
    texture_scale=(1.0, 1.0),
    texture_rotation=0.0,
    project_uvw=False,
    world_space_projection=False,
)
```

- Replaces the universal surface output of an existing bound USD material with
  an OmniPBR shader. Other materials retain their existing PreviewSurface or
  MDL outputs, so mixed-material stages remain renderable.
- `name` is a unique job-local identifier, while `material_path` is the
  existing absolute USD material prim path.
- Texture paths may provide albedo, scalar roughness, and tangent-space normal
  maps. `normal_format` accepts `"opengl"` or `"directx"` and controls the
  normal-map Y convention.
- `opacity` accepts `0.0` through `1.0`; values below `1.0` enable OmniPBR
  transparency so underlying geometry remains visible. It is intentionally a
  constant-only parameter and is not accepted in `variation`; declare separate
  material overrides when distinct opacity values are required.
- Set `project_uvw=True` for geometry without authored UVs. Set
  `world_space_projection=True` to keep the projection continuous across
  multiple meshes; `texture_scale` controls projected tiling.
- Material overrides execute after scene loading and before operations, final
  stage serialization, and capture.

## PCB-AOI operations API

Structural operation methods return `job.operations` for chaining. Supply
exactly one targeting mode: `ratio` is numeric and inclusive in `[0.0, 1.0]`,
or `count` is an integer greater than zero. A structural operation name may be
declared only once. Exact-count execution fails when fewer eligible components
remain; ratio execution preserves the legacy clamping behavior.

```python
job.operations.add_shift(
    *, ratio: float | None = None, count: int | None = None,
    translate_range: float, rotate_z_range: float,
    variant_index: int | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
)

job.operations.add_tombstone(
    *,
    ratio: float | None = None,
    count: int | None = None,
    angle_range: tuple[float, float],
    component_types: list[str] | tuple[str, ...] | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
    auto_filter: bool = True,
    eligibility_source: str | None = None,
)

job.operations.add_sideflip(
    *, ratio: float | None = None, count: int | None = None,
    angle_range: tuple[float, float],
    variant_index: int | None = None,
    rotate_z_jitter_deg: float = 0.0,
    slide_range_mm: float = 0.0,
    component_filter: ComponentFilter | Mapping | None = None,
)

job.operations.add_reverse_polarity(
    *,
    ratio: float | None = None,
    count: int | None = None,
    component_types: list[str] | tuple[str, ...] | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
)

job.operations.add_missing(
    *, ratio: float | None = None, count: int | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
)
```

- `translate_range` is a symmetric XY bound in millimeters;
  `rotate_z_range` is a symmetric degree bound. Both must be non-negative for
  JobOrder execution.
- Angle ranges are ordered `(minimum, maximum)` degree pairs.
- `component_filter` scopes the operation to exactly the components the filter
  matches, intersected with the current selection. It is the usual way to aim a
  defect at a named set of refdes without narrowing `select_components(...)`
  for the whole job. On `add_tombstone` and `add_reverse_polarity` it is
  mutually exclusive with `component_types`; `add_shift`, `add_sideflip`, and
  `add_missing` have no legacy selector to conflict with. See
  [Component filters](#component-filters).
- If `eligibility_source` is set, it must name an already attached
  `pcb.components` sidecar, or `"usd"` for embedded metadata. It must match the
  source the scene selected; the YAML interpreter always writes it explicitly,
  so a Python twin that leaves it `None` compiles to a different plan.
- `add_missing()` is exclusive: do not combine it with any other structural
  operation. The four transform operations may be combined.
- `rotate_z_jitter_deg` and `slide_range_mm` widen a sideflip's resting pose:
  the first rotates the flipped part about the board normal (the same axis
  `add_shift` calls `rotate_z_range`), the second slides it along its own long
  axis. Both are symmetric bounds and both default to 0.0, which reproduces the
  previous behaviour exactly.

#### `variant_index` — stratified coverage

`variant_index` replaces the random draw for a sample with an **assignment**:
sample *n* is told which region of the parameter space it belongs to and samples
only inside it, so coverage becomes a property of the run rather than something
you measure afterwards. Pass the sample's own loop counter:

```python
for n in range(count):
    job.operations.add_shift(count=1, translate_range=0.2,
                             rotate_z_range=15.0, variant_index=n)
```

It is **not a seed and not an identifier**. Two samples sharing an index share a
region, not a value — the exact numbers inside the region are still random, so a
stratified run is not a run of duplicates. Omit it and the legacy uniform
sampling is unchanged.

What each index selects, and how often each axis advances:

| Operation | Axis | Cycles over | Advances every | Period |
|---|---|---|---|---|
| `add_shift` | mode | `translate`, `rotate`, `combined` | 1 | **288** |
| | direction sector | 8 sectors of 45° in the component frame | 3 | |
| | translate bin | 15–30 %, 30–60 %, 60–100 % of `translate_range` | 24 | |
| | rotation bin | 20–60 %, 60–100 % of `rotate_z_range` | 72 | |
| | rotation sign | `+`, `−` | 144 | |
| `add_sideflip` | angle bin | 3 equal sub-bins of `angle_range` | 1 | **6** |
| | hinge edge | the two long support edges | 3 | |

The strides form a mixed-radix odometer — a digit advances only once every digit
below it has wrapped — which is what keeps the axes independent. Any 288
consecutive indices cover the shift grid exactly once; a count that is not a
multiple of the period leaves the tail partly covered without skewing the earlier
samples.

Full treatment, including how to verify the coverage a finished run achieved:
[`docs/defect-variants.md`](../docs/defect-variants.md).

### Solder-joint generation

```python
job.operations.solder_joints(
    *,
    solver: str = "none",
    component_types: list[str] | tuple[str, ...] | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
    source: str | None = None,
    solver_options: dict[str, object] | None = None,
    output_root_path: str | None = None,
    serialization: str | None = None,
    serialization_path: str | None = None,
)
```

- Uses embedded USD metadata when no component sidecar is attached. Calling
  `scene.attach_sidecar(...)` anchors the job to that legacy metadata source.
  `source` must name the same metadata source the scene selected; the YAML
  interpreter writes it explicitly (`"usd"` for an embedded-metadata board)
  rather than leaving it `None`.
- `component_filter` selects the components to solder by exact metadata fields
  and is mutually exclusive with `component_types`. When a filter is given,
  `component_types` must stay `None`; the YAML interpreter forces it to `None`
  in that case, and otherwise falls back to the `scene.components.types` list.
- Query solver IDs instead of inventing one:
  `job.operations.get_solder_joint_solvers(mode_filter=None)`.
- `mode_filter` is `None`, `"none"`, `"procedural"`, or `"simulated"`.
- The query returns dictionaries containing `id`, `type`, `available`, and an
  optional `reason`. Choose only an available solver.
- `solver_options` must be JSON-compatible with non-empty string keys.
- The procedural and Surface Evolver solvers accept `components` in
  `solver_options` as an exact refdes or component-instance prim-path
  allowlist. Omit it or use an empty list to retain every component admitted
  by `component_types` discovery.
- The procedural solver accepts `resolution` for angular tessellation and
  `solder_material_prim_path` for an explicit composed USD material source.
- THT meshes default to 64 angular segments and 16 radial rings, with smooth
  vertex normals. Use `tht_angular_segments` and `tht_radial_segments` to tune
  them independently. An explicit shared `resolution` remains the THT angular
  value when `tht_angular_segments` is omitted.
- For THT joints, `tht_pad_edge_inset_mm` sets the radial clearance from the
  associated pad edge. It defaults to `0.0`, replacing the former fixed
  maximum-annulus behavior. The generated outer boundary follows the associated
  pad contour instead of inscribing a circle in non-circular pads.
- `output_root_path`, when set, must be an absolute, non-root USD prim path.
- `serialization_path` exports the solder edit layer and therefore implies
  `"solder-joints-only"`. The legacy `serialization` keyword remains accepted
  for compatibility; `"all"` is superseded by execution-level final-stage
  serialization.
- A solder-layer export may omit capture declarations and execute with
  `job.execute()`. Use `compile_orders()` for validation only.

### Solder defects

Six adders shape the solder the solver produces. Each is its own operation
name and each selects its own `ratio` of the remaining eligible pool, so the
defect types do not overlap one another or the structural operations.

```python
job.operations.add_insufficient(
    *,
    ratio: float,
    volume_ratio=None,
    lead_climb=None,
    fullness=None,
    component_types: list[str] | tuple[str, ...] | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
)

job.operations.add_excess(
    *,
    ratio: float,
    volume_ratio=None,
    lead_climb=None,
    fullness=None,
    component_types: list[str] | tuple[str, ...] | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
)

job.operations.add_bright(
    *,
    ratio: float,
    lead_climb=None,
    fullness=None,
    component_types: list[str] | tuple[str, ...] | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
)

job.operations.add_interrupted(
    *,
    ratio: float,
    pad_exposure=None,
    lead_climb=None,
    fullness=None,
    component_types: list[str] | tuple[str, ...] | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
)

job.operations.add_missing_solder(
    *,
    ratio: float,
    component_types: list[str] | tuple[str, ...] | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
)

job.operations.add_dry_contact(
    *,
    ratio: float,
    gap_fraction=None,
    lead_climb=None,
    fullness=None,
    component_types: list[str] | tuple[str, ...] | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
)
```

- All six require `operations.solder_joints(...)` to be declared first, and
  that solver must be a **simulated** one (for example `"surface_evolver"`).
  A `"none"` or procedural solver ignores defect intent and would emit nominal
  joints under defect labels, so it is rejected at declaration time rather than
  silently honoured. Declare `solder_joints(...)` before the adders.
- `ratio` is required and inclusive in `[0.0, 1.0]`. These adders have no
  `count` mode.
- `component_types` and `component_filter` are mutually exclusive. Leaving both
  `None` draws from the pool `scene.select_components(...)` established.
- Every shape parameter takes either a single number or a `[min, max]` range
  sampled per joint; `None` keeps the solver or defect default. All of them are
  post-solve observables — measurable on the produced geometry, not solver
  internals.

| Adder | Parameter | Domain | Notes |
| --- | --- | --- | --- |
| `add_insufficient` | `volume_ratio` | `(0, 1)` fraction of nominal volume | `None` uses the solver default (SMT severity) |
| | `lead_climb` | `(0, 2)` wetted lead height over measured pin protrusion | THT shape; `> 1` buries the tip; nominal 0.8 |
| | `fullness` | `(0, 2)` fillet volume over the straight-cone fillet | THT always (nominal 0.85); SMT only when given |
| `add_excess` | `volume_ratio` | `> 1` multiple of nominal volume | same shape parameters as above |
| | `lead_climb`, `fullness` | as above | `lead_climb > 1` buries the pin tip |
| `add_bright` | `lead_climb` | `(0, 2)` | default `[0.6, 0.9]`; **through-hole only** |
| | `fullness` | `(0, 2)` | default `[0.35, 0.6]` |
| `add_interrupted` | `pad_exposure` | `(0, 1)` exposed pad area over pad area | default `[0.2, 0.5]` |
| | `lead_climb`, `fullness` | `(0, 2)` | shape overrides, as on `add_insufficient` |
| `add_missing_solder` | none | — | the volume ratio is definitionally zero |
| `add_dry_contact` | `gap_fraction` | `(0, 1)` barrel circumference NOT wetted | **through-hole only** |
| | `lead_climb`, `fullness` | `(0, 2)` | shape overrides |

- `add_bright` and `add_dry_contact` are through-hole only: an SMT component
  selected into them is reported INELIGIBLE by the solver instead of being
  silently soldered nominal.
- `add_interrupted` leaves the THT ring around the pin closed — a broken ring
  is `add_dry_contact`, a separate class. The exposure azimuth of
  `add_interrupted` and the gap azimuth of `add_dry_contact` are always
  randomized per joint and are never configurable.
- `add_missing_solder` is not the `add_missing` component operation: here the
  part is present and its solder (and printed paste) is not.

```python
job.operations.solder_joints(solver="surface_evolver", source="usd")
job.operations.add_insufficient(ratio=0.05, volume_ratio=[0.3, 0.7])
job.operations.add_dry_contact(
    ratio=1.0,
    gap_fraction=[0.25, 0.5],
    component_filter=ComponentFilter(refdes=("J2",)),
)
```

### Hole-reveal plane generation

```python
job.operations.hole_plane(
    *,
    class_name: str = "hole",
    side: str,                           # "top", "bottom", or "both"
    z_offset_mm: float = 0.1,
    output_prim_path: str | None = None,
    components: list[str] | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
)
```

- `component_filter` names the same kind of subset as `components` by exact
  metadata fields instead of refdes, and is mutually exclusive with it. Like
  `components`, it never widens back to a board-wide plane.
- `side` is required. Choose `"bottom"` for AOI bottom-side captures, `"top"` for top-side, `"both"` for boards inspected from both faces.
- Without `components`: creates a single quad mesh spanning the full board. The board substrate occludes it everywhere except at through-hole voids, so only hole openings are visible.
- With `components`: creates one small quad per selected-component pad/hole
  site instead of a board-wide plane.
- An explicit `components` filter never widens to a board-wide plane. Unknown
  refdes, components without eligible pads, and pads whose proxy cannot be
  generated are reported as runtime warnings and in the operation output;
  other matched components continue.
- `output_prim_path` names an Xform parent containing the generated mesh or
  meshes. The default parent is
  `<board-root>/SegmentationProxies/HolePlane`; pin-tip proxies use the sibling
  `<board-root>/SegmentationProxies/PinTips` hierarchy.
- `z_offset_mm` offsets the plane inward from the FR4 substrate face (not from the solder-mask layer). The default `0.1` keeps it just inside the substrate so it is fully occluded by board material and only "shows through" open holes. Increase it when through-hole pins or solder geometry at the same depth cause z-fighting.
- The plane carries the semantic class `class_name` (default `"hole"`). It does not require a separate `label_geometry(holes=...)` call.
- Hole-plane meshes have a magenta USD display color so synthetic reveal
  geometry is visually distinct during stage inspection. Semantic-mask color
  remains controlled independently by capture and stitching configuration.
- **Pass management**: the hole-plane Xform is hidden by default. Temporarily
  reveal it in annotation-only passes with `visibility.show: ["hole"]`; it
  remains hidden in RGB renders and in serialized stages.
- Generate the hole plane before annotations and capture. Its order relative
  to solder attachment does not change the proxy geometry.
- Requires board metadata that exposes substrate Z-range (`pcba:mesh:role = "board_substrate"` prims in the board USD).

Example for a THT board inspected from the bottom:

```python
job.operations.solder_joints(solver="procedural", ...)
job.operations.hole_plane(side="bottom", components=["J1", "J2"])
job.operations.pin_tip(side="bottom", components=["J1", "J2"])

# RGB pass — annotation proxies remain hidden; solder visible
job.capture.render_pass(...)

# Annotation pass — hide solder and reveal hole and pin-tip proxies
job.capture.render_pass(...).visibility(
    hide=["solder"],
    show=["hole", "pin_tip"],
)
```

### Pin-tip annotation proxies

```python
job.operations.pin_tip(
    *,
    side: str,                           # "top" or "bottom"
    class_name: str = "pin_tip",
    offset_mm: float = 0.002,
    components: list[str] | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
)
```

- `component_filter` restricts the proxies by exact metadata fields and is
  mutually exclusive with `components`.
- Each proxy copies the selected terminal's connected outward tip surface.
  Flat, steeply angled, triangulated, and curved tips are supported. Traversal
  stops at vertical lead walls, and also where the surface tilts more than 30
  degrees away from the tip's own outward normal. On a lead whose end is
  rounded rather than flat, that bound leaves the chamfer flanking the tip with
  the terminal's own `pin` class instead of absorbing it into the proxy; a flat
  or slanted cut is unaffected however finely it is triangulated, because its
  facets share that normal.
- Generation is ordered after pin variation, pin cutting, and solder attachment,
  so the proxy describes the geometry that will be captured.
- `offset_mm` moves only the annotation proxy outward to prevent z-fighting. It
  does not alter the physical terminal or solder mesh.
- The proxy carries `pcba:mesh:role = "board_pin_tip"` and the semantic class
  `class_name`. It does not require `annotations.label_geometry(...)`.
- The proxy has a magenta USD display color so all annotation-only geometry is
  visually distinct during stage inspection. Stitching `color_map` controls
  the semantic-mask color independently.
- The shared `SegmentationProxies` Xform and its `PinTips` child are hidden by
  default. Reveal the proxy only in a solder-hidden
  annotation pass with `visibility.show: ["pin_tip"]` so the tip surface is
  labeled `pin_tip` while the remaining terminal pixels retain their `pin`
  class.
- When solder is visible, normal depth occlusion applies. The proxy is not raised
  through solder to fabricate a visible pin tip.
- Unknown refdes and individual leads without a selectable tip surface emit a
  runtime warning and appear in `unmatched_components` or `failed_prim_paths`
  in the operation output. Other matched leads continue rendering.

To make the stitched semantic class cyan, use label-aware stitching:

```yaml
postprocess:
  stitch:
    source_pass: full_board_roi
    image_type: semantic_segmentation
    blend: labels
    color_map:
      pin_tip: [0, 255, 255, 255]
```

## Annotations API

Both methods return `job.annotations` for chaining.

```python
job.annotations.label_components(*, source=None, classes=None)
job.annotations.enable_operation_labels(*, namespace="defect")
```

- Call `scene.select_components(...)` first for JobOrder scripts.
- `source` is an optional metadata source name. Omit it for embedded USD
  metadata, or pass the name supplied to `attach_sidecar(...)` for a legacy
  sidecar-backed board. It is never authored in YAML: the interpreter derives
  it from the scene and always passes it explicitly (`"usd"` for an
  embedded-metadata board), so `source=None` and `source="usd"` are the same
  board but two different compiled plans.
- `classes` is `None` or `{class_name: [refdes, ...]}`.
- The operation-label namespace must be a non-empty string.
- If `capture.outputs()` omits `semantic_types`, the default becomes the
  operation namespace; without operation labels the default is `["class"]`.
- Ordinary board hierarchies receive semantics directly through Replicator.
  When `scene.instances(...)` registers a shared-prototype panel, CDF does
  not author semantics onto the prototype. It captures instance IDs and
  projects their composed prim paths through `BoardMetadataCatalog` after each
  frame, producing the same `semantic_segmentation_*.png` and labels JSON
  artifacts without changing the public annotation or capture API. Associated
  board-pad paths are included, allowing bottom-side THT captures to retain
  component labels even when the pad occludes the package lead geometry.
- The instance-safe path is selected only for a validated `scene.instances`
  cohort. Baked boards, referenced non-instance boards, and legacy sidecar
  boards continue using direct semantics. If instance IDs were not explicitly
  requested, their temporary image and mapping artifacts are removed after a
  successful projection.
- CDF does not infer a panel from its filename or hierarchy. Omitting
  `scene.instances(...)` deliberately retains the ordinary direct-semantics
  behavior.

### Geometry-level semantic labels

```python
job.annotations.label_geometry(
    *,
    components: list[str] | tuple[str, ...] | None = None,
    component_filter: ComponentFilter | Mapping | None = None,
    solder: str | None = None,
    terminals: str | None = None,
    pads: str | None = None,
    holes: str | None = None,
    barrels: str | None = None,
    body: str | None = None,
    marking: str | None = None,
)
```

- Labels geometry that is more granular than a whole component scope. Useful for THT inspection where a pad, pin, solder fillet, and barrel around the same hole each need a distinct class, and for component-level ground truth where the package body and its pin-1 / polarity marking must be told apart.
- Each keyword argument maps a geometry target to a semantic class name string. Omit a keyword to skip that target.
- `solder` labels the solder-joint output root. The annotation is recorded at declaration time and applied when `solder_joints(...)` executes, so declaring `solder=` here is valid even before `solder_joints(...)` is declared.
- `terminals` labels per-component pin/lead prims from the board metadata catalog.
- `pads` labels pad prims (copper annular ring) from the catalog.
- `holes` labels drilled-hole prims from the catalog. This is distinct from `hole_plane`; both can be used together.
- `barrels` labels the cylindrical copper inner-wall prims (`pcba:mesh:role = "board_barrel"`). Setting this to the same class as `pads` eliminates the unlabeled ring between pad and hole visible in segmentation output.
- `body` labels the package body mesh of every component prototype (`pcba:mesh:role = "pkg_body"`).
- `marking` labels the package's pin-1 dot / polarity band / notch mesh (`pcba:mesh:role = "pkg_marking"`). Only packages whose prototype carries such a mesh are labelled; a package with no marking mesh contributes nothing and is not an error.
- `component_filter` restricts the same targets by exact metadata fields
  instead of by refdes, and is mutually exclusive with `components`. See
  [Component filters](#component-filters).
- `components` restricts pads, terminals, holes, barrels, bodies and markings to the listed refdes. Solder is not filtered here (it is scoped by `solder_joints` options). Body and marking meshes belong to the component prim that encloses them.
- Geometry labels require an instanced capture-stage refresh for component sub-geometry (e.g., pin prims inside a de-instanced component). This happens automatically.

The Python declaration is identical for standalone and instanced inputs:

```python
job.annotations.label_components(
    source="usd",
    classes={"connector": ("J1", "J2")},
)
```

### Component annotation execution paths

Register every placement sharing a prototype with `job.scene.instances(...)`
to enable instance-safe component annotations. CDF validates the complete
cohort; it does not infer panels from filenames or prim names. Without a
registered cohort, component labels use direct Replicator semantics.

For a registered cohort, CDF captures instance IDs and projects component
and defect labels after capture without modifying shared-prototype semantics.
Component membership includes package bodies, contacts/leads, and associated
board pads, including pads that occlude leads in bottom-side THT captures.
The metadata catalog comes from either the attached sidecar or embedded USD
metadata; the sources are alternatives and are not merged.

Both paths produce `semantic_segmentation_*.png` and
`semantic_segmentation_labels_*.json`. Intermediate instance-ID images and
mappings are removed after successful projection unless the capture explicitly
requests them. Projection supports the `class` and `defect` semantic types;
unsupported types fail validation before the affected capture pass.

When solder geometry changes a registered shared prototype, CDF refreshes
the composed capture stage before its first capture so materials remain
correct. This is automatic and preserves the validated instance cohort.

## Capture API

Direct configuration methods return `job.capture` and configure the
backward-compatible implicit default render pass. Multiple named passes share
the same prepared scene, appearance, operations, and generated solder:

```python
with job.capture.render_pass("full_board") as full_board:
    full_board.renderer(...)
    full_board.camera(...)
    full_board.scan_grid(x_count=1, y_count=1)
    full_board.outputs(rgb=True)

with job.capture.render_pass("component_rois") as component_rois:
    component_rois.renderer(...)
    component_rois.camera(side="bottom", rotation_range={"z": -90})
    component_rois.scan_grid(x_count=1, y_count=1)
    component_rois.visibility(hide=("solder",))
    component_rois.roi_selection(refdes=("J1", "J2"), padding=0.5)
    component_rois.roi_selection(component_types=("CAPACITOR",), count=2)
    component_rois.roi_selection(
        process_classes=("through_hole",),
        focus="pin",
    )
    component_rois.roi_selection(defects=("shift", "tombstone"))
    component_rois.outputs(rgb=True)
```

Calls must be made on the yielded pass object. A direct
`job.capture.renderer(...)` call always configures the implicit default pass;
the context manager does not redirect it. Named-pass outputs are written under
`<output_dir>/<pass_name>`.

```python
job.capture.output_start(start)
```

`start` is an integer >= 0 and controls the first written frame index. It is
optional for a hand-written job, but the YAML interpreter always emits
`output_start(0)` for every pass, so a Python build aiming at plan equality
with a YAML config must call it too. The same holds for `limits(...)`, which is
always emitted with its defaults `triggers=1, max_images=-1`.

```python
job.capture.renderer(
    *, mode, resolution, spp, total_spp, max_bounces, realtime_subframes=1
)
```

- `mode`: `"pathtracing"` or `"realtime"`.
- `resolution`: positive `(width, height)` integers.
- `spp` and `total_spp`: integers >= 1. Path tracing only.
- `max_bounces`: integer >= 0. Path tracing only.
- `realtime_subframes`: integer >= 1, the `rt_subframes` used by `"realtime"`.
  Ignored by path tracing.

`"realtime"` renders about 1.47x faster on a scan grid and is a measurably
different picture (37.8 dB against a 128-sample reference, where path tracing
is 46.7 dB), so the two are not interchangeable within one dataset.

```python
job.capture.camera(*, rotation_range, side="top")
```

`side` is `"top"` or `"bottom"` and names the board face the camera views; any
other value raises `ValidationError`. It defaults to `"top"`, so a bottom-side
capture must state `side="bottom"` — it also decides which face the batch
lighting rig is placed on.

`rotation_range` accepts only keys `x`, `y`, and `z`, and an omitted axis is
simply left out of the mapping. Values for `x` and `y` are ordered numeric
pairs. `z` is one fixed numeric angle, not a pair.

```python
job.capture.scan_grid(*, x_count, y_count, scan_area_mm=None, centered_on=None,
                      scan_center_mm=None, image_size_px=None, guard_px=0)
job.capture.visibility(*, hide=(), show=())
job.capture.limits(*, triggers=1, max_images=-1)
```

- Grid counts and `triggers` are integers >= 1.
- `scan_area_mm` is the world rectangle `(length_x, breadth_y)` in millimetres
  the grid spans, centered on the board; `None` spans the whole board.
  `centered_on` labels what that rectangle is centered on and requires
  `scan_area_mm`. `scan_center_mm` is the world `(x, y)` in millimetres to
  center `scan_area_mm` on instead of the board; it requires `scan_area_mm` and
  excludes `centered_on`. All three are converted with the stage's scale at
  render time. `fov` sets them; a pass with a `fov` does not state them.
- `image_size_px` crops the stitched canvas to `(width, height)` in image
  axes, centered; `capture.fov` sets it. A mosaic smaller than it is refused.
- `guard_px` renders every cell that many pixels larger on each side, with the
  aperture widened by the same number of pixel pitches and the cell centres
  unchanged, and crops them from every per-tile output in the writer, before
  anything is encoded. The renderer's outermost column and row are unreliable
  (measured 0.74-0.89x and ~0.90x of the interior) and the cells of a stated
  `scan_area_mm` abut, so without it every seam shows them. Tiles, the pixel
  layout, `calibration.json` and the recorded renderer resolution describe the
  delivered tile whatever its value; `metadata.json` records it as `guard_px`.
  It requires `scan_area_mm`. Cropped: `rgb`, segmentation masks (label
  sidecars unchanged), depth/normals/motion vectors, and 2D boxes (shifted by
  `-guard_px`, clipped to the tile, dropped when wholly in the band, prim-path
  lists filtered to match). `bounding_box_3d`, `occlusion`, `pointcloud`,
  `skeleton_data` and `camera_params` have no faithful crop and are refused.
  `0` (the default here; `fov` defaults to 8) records the order unchanged.
- Planned frames before limits are `x_count * y_count * triggers` per pass.
- `max_images=-1` means unlimited; otherwise use an integer >= 0.

```python
job.capture.fov(*, scan_area_mm=None, corners_mm=None, dpi=None, mm_per_px=None,
                image_size_px=None, max_tile_px=(4096, 4096), centered_on="unspecified",
                guard_px=None)
```

`fov` (pcb-aoi domain) states the capture the way the AOI scanner is
configured -- a rectangle of board in millimetres at a resolution -- and
derives `renderer.resolution` and the scan-grid counts from it. Give the area
as `scan_area_mm` (a size, centered on the board) or as `corners_mm` (two
opposite corners in world millimetres, `(x, y)` or `(x, y, z)` each, `z`
ignored -- the lower-left and upper-right points a machine test program
exports), and exactly one of `dpi`, `mm_per_px` (the same resolution as a
pixel pitch: 0.042333 for 600 dpi) or `image_size_px`. Corners also place the
rectangle: its center is their midpoint rather than the board's, and it is
recorded as `scan_grid(scan_center_mm=...)`; `centered_on` applies to
`scan_area_mm` only. A pass with a `fov`
must not state `renderer(resolution=...)` or `scan_grid(...)`; either is
refused in both call orders. Orthographic cameras only. Typical AOI line
scanners image at 600 or 1200 dpi; a request outside that is
rendered and recorded in the plan's `assumptions`, not refused. The stitched
image is `area / pitch` pixels per axis, rounded once (down), so
it is the size the scanner states for the same rectangle -- 100 x 200 mm at
600 dpi is 2362 x 4724 -- whatever the tiling; the cells render a mosaic up to
one pixel per cell larger and the stitch crops it back. Each cell also renders
a `guard_px` guard band (default 8, recorded as `scan_grid(guard_px=...)`)
that is cropped before its tile is written -- see `scan_grid` above; the
delivered tiles and every contract describing them are the same size at any
`guard_px`, and `max_tile_px` bounds the guarded render, so a tile that fits
only without its band tiles into one more cell. `guard_px=0` renders the tile
directly.

Pixels are square: an orthographic camera renders one pitch on both image
axes. `dpi` and `mm_per_px` take a number or an equal pair; an unequal pair is
refused (`ValidationError` from the Python API, a parse error from YAML). With
`image_size_px` the pixel count is kept exactly and both axes use the coarser
pitch, `max(area_x / width, area_y / height)`, so the whole requested
rectangle is imaged; when the canvas does not have the area's aspect the other
axis covers more than was asked, centred on the requested rectangle (or the
corners' midpoint), and the plan reports it as `scan_area_mm_effective`
beside the requested `scan_area_mm`. A hand-authored `scan_grid` with
`scan_area_mm` whose cell footprint does not have the aspect of
`renderer.resolution` is refused at render time.

`visibility(hide=("solder",))` hides generated solder only while that pass
renders. `visibility(show=("hole", "pin_tip"))` reveals default-hidden
annotation proxies for the same duration. Both restore the exact prior
authored visibility afterward. Hiding solder without an attached placement
manifest fails instead of silently rendering an occluded annotation pass.

### ROI selection

`roi_selection()` creates focused component or physical-pin frames from board
metadata and applied-defect records:

```python
render_pass.roi_selection(
    *,
    refdes=None,
    pins=None,
    component_types=None,
    process_classes=None,
    component_filter=None,
    all_components=False,
    defects=None,
    focus=None,
    count=None,
    padding=0.5,
)
```

Each call supplies exactly one selector. Calls stack within a render pass.

| Selector | Selection behavior | `count` | Supported focus |
| --- | --- | --- | --- |
| `refdes` | Components in the supplied explicit order | Not accepted | Component or pin |
| `pins` | Exact `(refdes, pin_number)` identities | Not accepted | Always pin |
| `component_types` | Components matching type, family, package, or footprint metadata | Defaults to one deterministic match | Component or pin |
| `process_classes` | Components matching aggregate or component process metadata | All matches by default; deterministic subset when supplied | Component or pin |
| `component_filter` | Components matching an exact metadata-field filter (`ComponentFilter` or equivalent mapping) | All matches by default; deterministic subset when supplied | Component or pin |
| `all_components=True` | Every component under the board metadata root | Not accepted | Component or pin |
| `defects` | Components targeted by the requested structural defects | All matches by default; deterministic subset when supplied | Component or pin |

`focus="pin"` expands every matched component into its cataloged physical
terminals. Omitting `focus` preserves component framing. Exact `pins` selection
is intrinsically pin-focused. For mixed-process components, process-class pin
selection uses terminal-level process metadata; it fails safely when that
metadata is unavailable because the matching pins cannot be distinguished.

Stacked selectors are evaluated in declaration order. Duplicate component or
physical-terminal prim paths at the same focus level are rendered once. On a
registered board-instance cohort, each selected source target expands into the
corresponding target on every registered board instance.

Pin frames use the following resolution order:

1. Resolve pads whose contact prim directly identifies the physical terminal.
2. Fall back to pads sharing its logical `pcba:pin:number`.
3. Use the sole pad immediately, without reading geometry.
4. When multiple pads remain, select the nearest pad using world-space bounds.
5. If no pad is available, use the nearest generated solder joint, then the
   physical pin geometry.

Before step 4, a metadata-only preflight groups selected terminals by component
and `pcba:pin:number`. Duplicate logical numbers with ambiguous pad associations
emit a runtime warning before geometric scanning starts. The warning is also
stored in capture-result and trigger metadata `warnings`. Each pin frame records
the resolved `pin_path`, pad paths, generated solder-joint path, component path,
selector identity, and panel-instance identity when applicable.

Every component- or pin-focused frame also records the selected component's
source-neutral `component_type`, `package_name`, `package_type`, `footprint_id`,
`side`, `process_class`, and `process_classes` in both
`frame.roi_selection` and `frame.camera.roi_selection`. The trigger-level
compatibility `roi_selections` projection contains the same values. Catalog
fields that are unavailable remain `null`; `process_classes` remains an array.

`padding` is a nonnegative fractional framing margin applied independently to
each selected component or pin site.

Pin-focused selectors can start from exact pins or expand any component filter:

```python
pin_rois.roi_selection(refdes=("J1",), focus="pin")
pin_rois.roi_selection(
    component_types=("<component-type>",),
    focus="pin",
)
pin_rois.roi_selection(process_classes=("through_hole",), focus="pin")
pin_rois.roi_selection(all_components=True, focus="pin")
pin_rois.roi_selection(pins=(("J1", "4"),))
```

```python
job.capture.augmentation(*, motion_blur=None)
```

`motion_blur` is a mapping. When present, `probability` must be within
`[0.0, 1.0]` and `alpha_range` must be an ordered numeric pair. Omission
compiles to a disabled motion-blur block.

```python
job.capture.outputs(
    *,
    rgb=True,
    image_format="png",
    bounding_box_2d_tight=False,
    bounding_box_2d_loose=False,
    bounding_box_3d=False,
    semantic_segmentation=False,
    colorize_semantic_segmentation=False,
    instance_id_segmentation=False,
    colorize_instance_id_segmentation=False,
    semantic_types=None,
    frame_padding=4,
    rename_to_grid_index=False,
    **extra_annotators,
)
```

`extra_annotators` passes additional writer keys through by name, for example
`distance_to_camera=True`. Use it only when the exact pipeline writer key is
known. For scripts that crop output, enable `bounding_box_2d_tight`, enable a
semantic output, and include the same semantic artifact name in crop `types`.
`image_format` must be one of `png`, `jpeg`, `jpg` or `bmp`; any other value
raises `ValidationError` at declaration time, on the YAML surface and here
alike. The writer encodes only png and jpeg itself, so `bmp` is produced by
re-encoding the rgb frames after each pass flushes; segmentation masks,
bounding boxes and label sidecars keep their native formats. The postprocess
steps that open rgb frames look for the writer's png/jpeg names, so a pass
declaring `bmp` with rgb enabled cannot feed a `crop` whose `types` include
`rgb`, a `stitch` with `image_type="rgb"`, or `format_roi_output` as its
`rgb_pass`; compilation rejects those combinations. Steps that read only
masks, bounding boxes or stitched images -- `close_openings`,
`generate_joint_rois`, a semantic-only `crop` or `stitch` -- stay valid, as
does any postprocessing of a `bmp` pass declared with `rgb=False`. Use
non-empty `semantic_types` entries and `frame_padding >= 1`
for JobOrder execution.
Set `rename_to_grid_index=True` to rename completed writer artifacts from
numeric frame suffixes such as `rgb_0000.png` to scan coordinates such as
`rgb_x0_y0.png`. The runtime preserves the numeric `frame_to_grid` and
`pixel_layout` metadata used to relate each file to its capture frame.

## Postprocess API

```python
job.postprocess.crop(
    *,
    method="bbox_tight",
    types=None,
    offset=10,
    min_coverage=0.2,
    class_filter=None,
    xform_depth=7,
    output_subdir="cropped",
    mode=None,
    input_subdir=None,
    reference_subdir=None,
) -> None
```

Declare cropping before rendering so it runs after capture in the same
JobOrder plan:

```python
job.postprocess.crop(types=["rgb", "semantic_segmentation"])
result = job.capture.render(output_dir="output")
print(result.crop_output)
```

Rules:

- Only `method="bbox_tight"` is supported by this in-process API.
- `types=None` becomes `["rgb", "semantic_segmentation"]`.
- Explicit `types` must contain at least one value containing `"semantic"`.
- `offset` and `xform_depth` must be integers >= 0 for JobOrder execution.
- `min_coverage` must be within `[0.0, 1.0]` for JobOrder execution.
- `class_filter` is `None` or a list/tuple of non-empty class names.
- If `mode` is not `None`, both `input_subdir` and `reference_subdir` are
  required.
- The call declares work and returns `None`.

```python
job.postprocess.stitch(
    *,
    source_pass="default",
    image_type="rgb",
    output_subdir="stitched",
    output_filename="rgb.png",
    blend="none",
    write_labels=False,
    recolor_tiles=False,
    color_map=None,
) -> None
```

Declare stitching after the selected render pass. The postprocess operation
uses the capture's canonical pixel-layout metadata and runs after rendering:

**Parameters**

- `source_pass` — named capture pass, or `"default"` for legacy capture.
- `image_type` — BasicWriter image prefix to stitch (e.g. `"rgb"`,
  `"semantic_segmentation"`).
- `output_subdir` — directory created beneath the capture-pass output root.
- `output_filename` — filename for a single-trigger result.
- `blend` — overlap policy: `"none"` pastes tiles; `"feather"` cross-fades
  over the aperture overlap; `"labels"` unifies per-tile segmentation colors
  and writes a remapped `*_labels.json` sidecar.
- `write_labels` — when `True` and `blend` is not `"labels"`, merge the
  per-tile label sidecars alongside the stitched image without color remapping.
  Cannot be combined with `blend="labels"`.
- `recolor_tiles` — when `True` with `blend="labels"`, rewrite each source
  segmentation tile and its label sidecar with the unified stitched colors.
- `color_map` — optional `dict[str, tuple[int,int,int,int]]` mapping class
  names to explicit RGBA values in `[0, 255]`.  Only applies when
  `blend="labels"`.  Overrides the hash-derived color for the named class.
  `BACKGROUND` and `UNLABELLED` are always `(0,0,0,0)` and `(0,0,0,255)`
  respectively and cannot be overridden.

```python
with job.capture.render_pass("full_board") as full_board:
    full_board.renderer(
        mode="pathtracing",
        resolution=(2000, 2000),
        spp=1,
        total_spp=8,
        max_bounces=4,
    )
    full_board.camera(side="bottom", rotation_range={"z": -90})
    full_board.scan_grid(x_count=5, y_count=4)
    full_board.outputs(rgb=True)

job.postprocess.stitch(
    source_pass="full_board",
    output_filename="panel_10000x8000.png",
)
```

Stitching requires an orthographic camera with zero X/Y tilt and a cardinal
Z rotation. Component ROI passes use independent apertures and are therefore
not stitchable.

### Close ring-shaped labels

```python
job.postprocess.close_openings(
    *,
    source_pass="default",
    class_name,
    output_subdir="stitched",
    output_filename="rgb.png",
    max_diameter_px=48,
    min_circularity=0.55,
) -> None
```

`close_openings()` modifies the stitched label image identified by
`source_pass`, `output_subdir`, and `output_filename`. It fills enclosed
unlabelled pixels with `class_name`, while preserving pixels already assigned
to another class. `max_diameter_px` limits the opening size and
`min_circularity` must be in `(0, 1]`. The compiler requires exactly one
matching `stitch()` producer and schedules closure after it.

### Format AOI ROI output

```python
job.postprocess.format_roi_output(
    *,
    rgb_pass="full_board",
    roi_pass="full_board_roi",
) -> None
```

Both named scan-grid passes must set `outputs(rename_to_grid_index=True)`.
For each trigger, the formatter:

- renames ROI semantic images to `semantic_mask_x<col>_y<row>.png`;
- merges per-tile label sidecars into
  `semantic_segmentation_labels.json`, retaining the required
  `BACKGROUND`, `UNLABELLED`, `pad`, `hole`, and `pin` classes plus optional
  `pin_tip`;
- writes canonical pixel-layout `metadata.json` to both pass directories;
- verifies matching RGB/ROI tile counts, grid coordinates, and dimensions.

### Generate full-board joint ROIs

```python
job.postprocess.generate_joint_rois(
    *,
    roi_pass="full_board_roi",
    rgb_pass="full_board",
) -> None
```

The ROI pass must request `bounding_box_2d_tight` and set
`outputs(rename_to_grid_index=True)`. The postprocessor transforms each
per-tile tight box through the capture pixel layout, unions duplicate joint
boxes across overlapping tiles, and writes identical `joint_rois.json`
manifests beneath every trigger directory and beside the stitched output for
both the named ROI and RGB passes. A single trigger uses
`stitched/joint_rois.json`; multiple triggers use collision-safe
`stitched/joint_rois_trigger_XXXX.json` names. Each record contains the board
instance, component refdes, explicit `pin_number`, exact contributing
`source_prim_paths`, and stitched-image `bbox_xyxy`. Component metadata adds
`component_type`, `component_type_source`, `package_type`, `package_name`,
`footprint_id`, and `process_class`.
The type prefers an authored semantic type, then family, package type, footprint
identity, and package name; unavailable values remain `null`. The `joint_id`
retains all three identity segments for compatibility. Nested pin-tip proxy paths
are accepted and resolve to the underlying component lead. Joints are defined
by entries with a `StepRepeats/<instance>/.../Components/<refdes>/<lead>`
hierarchy; `bbox_xyxy` covers those entries only (the pin).

`joint_bbox_xyxy` is the whole joint: the pin box united with the boxes of the
board pads and holes the selected board metadata catalog associates with that
component pin (the exporter-stated refdes + pin number pair, never XY
proximity). A shared-prototype catalog is re-rooted under every board
placement, so each instance's pads join that instance's joint. Pad boxes come
from the same ROI pass and pixel layout as the pin, so the result is identical
for `scan_grid`, `fov(scan_area_mm=...)` and `fov(corners_mm=...)` captures. The
ROI pass must label the pads, for example `annotations.geometry.pads`; pads it
does not label, or that the catalog does not associate, leave
`joint_bbox_xyxy` equal to `bbox_xyxy` with `joint_bbox_source` set to
`pin_only`, and the postprocess warns with the count. A pad never creates a
joint on its own. Crop margins remain the consumer's choice.

#### `joint_rois.json` schema 1.0

Top-level properties:

| Property | Type | Meaning |
|---|---|---|
| `schema_version` | string | Joint-ROI document version. The initial contract writes `1.0`. |
| `coordinate_system` | string | Pixel-coordinate convention. `pixel_top_left_xyxy_half_open` uses a top-left origin with exclusive maximum coordinates. |
| `source_pass` | string | RGB capture pass containing the corresponding joint pixels. |
| `source_trigger` | string | Trigger directory associated with the ROI records. |
| `items` | array | Joint records, sorted by compound joint identity. |

Each item contains:

| Property | Type | Nullable | Meaning |
|---|---|---|---|
| `joint_id` | string | No | Compound identity in `<board_instance>/<component_ref>/<pin_number>` form. |
| `board_instance` | string | No | Stable board-instance label within the panel or cohort. |
| `component_ref` | string | No | Component reference designator, such as `R1`. |
| `pin_number` | string | No | Pin or lead identifier parsed from the annotated component-lead path. It remains a string because source identifiers need not be numeric. |
| `component_type` | string | Yes | Best available grouping value. Resolution prefers authored component type, family, package type, footprint ID, then package name. |
| `component_type_source` | string | Yes | Catalog field that supplied `component_type`: `component_type`, `family`, `package_type`, `footprint_id`, or `package_name`. |
| `package_type` | string | Yes | Native package classification from the selected board metadata catalog. |
| `package_name` | string | Yes | Native package or model name from the selected board metadata catalog. |
| `footprint_id` | string | Yes | Source footprint identity independently of which value supplied `component_type`. |
| `process_class` | string | Yes | Manufacturing process classification, normally `through_hole` or `SMT`. |
| `source_prim_paths` | array of strings | No | Sorted, unique USD prim paths whose annotations contributed to this joint ROI. The list may contain both a physical lead and its segmentation proxy. |
| `bbox_xyxy` | array of four integers | No | Pin bounds `[x_min, y_min, x_max, y_max]` in the stitched RGB image, from the lead and pin-tip entries only. Maximum coordinates are exclusive. |
| `pad_prim_paths` | array of strings | No | Sorted, unique USD prim paths of the associated pads and holes whose boxes widened `joint_bbox_xyxy`. Empty for `pin_only` joints. |
| `joint_bbox_xyxy` | array of four integers | No | Whole-joint bounds: the union of `bbox_xyxy` and the boxes of `pad_prim_paths`, in the same coordinates. Equals `bbox_xyxy` for `pin_only` joints. |
| `joint_bbox_source` | string | No | `pin_and_pads` when at least one associated pad box was found, otherwise `pin_only`. |

Unavailable catalog values and ambiguous metadata for a refdes are written as
`null`; the postprocessor does not infer a category from the refdes prefix.
For example:

```json
{
  "schema_version": "1.0",
  "coordinate_system": "pixel_top_left_xyxy_half_open",
  "source_pass": "full_board",
  "source_trigger": "trigger_0000",
  "items": [
    {
      "joint_id": "0/R1/3",
      "board_instance": "0",
      "component_ref": "R1",
      "pin_number": "3",
      "component_type": "R_Axial_DIN0207_L6.3mm",
      "component_type_source": "footprint_id",
      "package_type": null,
      "package_name": "R_Axial_DIN0207_L6.3mm",
      "footprint_id": "R_Axial_DIN0207_L6.3mm",
      "process_class": "through_hole",
      "source_prim_paths": [
        "/World/Carrier/StepRepeats/SR_0_0/Components/R1/pkg_lead_3"
      ],
      "bbox_xyxy": [8403, 6786, 8425, 6808],
      "pad_prim_paths": [
        "/World/Carrier/StepRepeats/SR_0_0/Layers/Bottom/<layer>/Pad_<n>"
      ],
      "joint_bbox_xyxy": [8384, 6767, 8444, 6827],
      "joint_bbox_source": "pin_and_pads"
    }
  ]
}
```

## Results and errors

JobOrder execution returns `JobResult` with `success`, `order_results`,
`outputs`, `output_for_type(order_type)`, `capture_output`, `crop_output`,
`stitch_output`, and `failed_order`.

A result also says how long it took. `JobOrderResult.duration_s` is the
wall-clock of that handler -- recorded for a failed order too, since how far a
run got before it died is the interesting part -- and `JobResult.startup_s` is
the application boot. `timings` maps order id to seconds,
`total_duration_s` sums startup and every order, and `slowest_orders(limit=5)`
returns `(order_id, order_type, seconds)` longest first, omitting orders that
were not timed rather than reporting them as free. Both fields default to zero,
so a result decoded from a child process that predates them is still valid.

Handlers may additionally attribute their own time by name. `capture.render`
does, reporting `phases` in its output mapping -- each entry a `seconds` and a
`calls` count, costliest first:

```python
capture = result.capture_output
for name, phase in capture["phases"].items():
    print(name, phase["seconds"], phase["calls"])
```

The call count is not decoration: a warm-up that runs once per process and one
that runs once per pass cost the same on paper and mean opposite things.

Public exceptions:

- `CDFError`: base package error.
- `ValidationError`: invalid or incomplete job description; also a
  `ValueError`.
- `JobExecutionError`: a JobOrder handler failed. Inspect `.order`, `.result`,
  and `.cause`.
- Pydantic validation errors may be raised when directly constructing or
  decoding JobOrder transport models.

## Complete defect-job pattern

```python
from pathlib import Path

import pcb_aoi
from cdf import CDFJob


def build_defect_job(
    board_usd: Path,
    components_json: Path,
    board_root: str,
) -> CDFJob:
    job = CDFJob(domain="pcb-aoi", random_seed=0)
    job.scene.configure(usd_path=board_usd, root_path=board_root)
    job.scene.attach_sidecar(
        name="components",
        kind="pcb.components",
        path=components_json,
    )
    job.scene.select_components(source="components")
    job.appearance.lighting(mode="usd_light")
    job.operations.add_tombstone(
        ratio=0.2,
        angle_range=(70, 90),
        auto_filter=True,
        eligibility_source="components",
    )
    job.annotations.label_components(source="components")
    job.annotations.enable_operation_labels(namespace="defect")
    job.capture.renderer(
        mode="pathtracing",
        resolution=(1920, 1080),
        spp=1,
        total_spp=32,
        max_bounces=4,
    )
    job.capture.camera(
        rotation_range={"x": (0, 0), "y": (0, 0), "z": -90},
    )
    job.capture.scan_grid(x_count=10, y_count=10)
    job.capture.limits(triggers=1, max_images=-1)
    job.capture.outputs(
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        semantic_types=["defect"],
    )
    job.postprocess.crop(
        types=["rgb", "semantic_segmentation"],
        method="bbox_tight",
    )
    return job
```

## Matching a YAML config plan-for-plan

A declarative YAML config is interpreted by `cdf/config/*.py`, which calls
exactly the API above. A Python build that must compile to the *same*
`JobOrderPlan` as a YAML file — a twin, a regression fixture, a port — has to
spell out what the interpreter supplies on the author's behalf. None of the
following is optional in that situation, and each one is a whole order or a
whole argument of difference.

**Arguments derived from the scene, never authored in YAML.** The interpreter
resolves one component-metadata source for the job — `"usd"` for a board with
embedded metadata, the sidecar name for a `pcb.components` board — and passes
it explicitly to:

- `scene.select_components(source=...)`
- `operations.solder_joints(source=...)`
- `annotations.label_components(source=...)`
- `operations.add_tombstone(eligibility_source=...)`

Each of these defaults to `None` in Python, and `None` is recorded as `None`,
not resolved to `"usd"` at declaration time. Pass the string.

**Orders that are always emitted.** For every render pass, including the
implicit default one, the interpreter calls `output_start(0)` and
`limits(triggers=1, max_images=-1)` even when the YAML says nothing about
either. `renderer(...)` likewise always carries `realtime_subframes=1`.

**Scalars expanded to ranges.** The YAML surface accepts a scalar wherever the
Python API wants an ordered pair, and expands it to `(v, v)`: `exposure: 0.0`
becomes `exposure_range=(0.0, 0.0)`, `cone_angle: 120.0` becomes
`cone_angle_range=(120.0, 120.0)`, and each lighting-layer `intensity` becomes
a constant pair. Camera rotation is expanded per axis, except `z`, which stays
a scalar. Solder-defect observables (`volume_ratio`, `lead_climb`, `fullness`,
`pad_exposure`, `gap_fraction`) accept a scalar on both surfaces and are not
expanded.

**Fixed order within `operations`.** Section order is scene -> appearance ->
operations -> annotations -> capture -> postprocess, and inside `operations`
the interpreter applies a fixed sequence that ignores the order the keys were
written in:

```text
shift -> tombstone -> sideflip -> reverse_polarity -> missing
-> pin_variation -> solder_joints -> hole_plane -> pin_tip
-> insufficient -> excess -> bright -> interrupted -> missing_solder
-> dry_contact
```

A YAML file that lists `solder_joints` or `hole_plane` first still compiles in
this order, so write the Python calls in this order rather than in the YAML's
own. The solder defects come last on purpose: each one validates that the
declared solder solver honours defect intent, so `solder_joints(...)` must
already be declared. Multiple `pin_variation` groups are applied in list order,
and later groups override earlier ones for a lead named by both.

**Paths.** The interpreter resolves every config path through
`Path(...).resolve()` relative to the config's directory, so a container path
such as `/work/boards/board.usd` compiles on Windows to `C:\work\boards\board.usd`.
Reproduce it with `str(Path("/work/boards/board.usd").resolve())` rather than
hardcoding a drive letter.

**Blocks whose absence is not the same as an empty call.** Omitting
`scene.components` entirely means no `select_components(...)` call, which also
means `solder_joints` receives `component_types=None`; an explicit
`select_components(types=[...])` would have been reused there. Likewise
`appearance.vantablack_components` emits `appearance.components(...)` only when
the key is present, and `label_geometry(...)` is skipped when no label remains
after dropping absent entries.

## Validation checklist for generated scripts

Before returning a script, a skill should check:

- [ ] `import pcb_aoi` appears before `CDFJob(...)` executes.
- [ ] User-provided paths and USD prim paths were preserved exactly.
- [ ] `root_path` and any `output_root_path` begin with `/`.
- [ ] Referenced sidecars are attached earlier in the script.
- [ ] Component selection precedes operations and component annotations.
- [ ] Missing is not combined with transform defects.
- [ ] Required capture methods are present.
- [ ] Defects specify one positive `count` or a `ratio` within `[0, 1]`.
- [ ] Pair ranges are ordered and camera `z` is a scalar.
- [ ] `component_filter` is not combined with the legacy selector it excludes.
- [ ] Solder-defect adders follow a `solder_joints(...)` with a simulated solver.
- [ ] Bottom-side captures pass `camera(side="bottom", ...)`.
- [ ] Crop inputs include a semantic artifact.
- [ ] Solder generation uses the JobOrder execution path.
- [ ] `compile_orders()` is used when the user asked only to generate or
      validate a plan.
- [ ] The script does not launch Isaac Sim unless the user requested execution.

## Scope

This document specifies declarative job authoring through `CDFJob`.
