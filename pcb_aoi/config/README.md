# PCB-AOI versioned CDF YAML job configuration

This directory defines the human-authored YAML representation of the CDF
authoring API. A YAML job and an equivalent Python `CDFJob` compile through
the same validation and dependency compiler into an immutable `JobOrderPlan`.
YAML is an authoring surface; it is not a serialized plan and it does not have
a separate execution engine.

The current contract is CDF API `1.0.0` with status `released`. See
[`../../cdf/VERSION.md`](../../cdf/VERSION.md) for the authoritative version and
[`../../cdf/API_CHANGELOG.md`](../../cdf/API_CHANGELOG.md) for compatibility policy and
history.

## Versioned JobConfig

A JobConfig is one YAML document beginning with `api_version` and `domain`.
It is interpreted into public `CDFJob` calls and compiled into JobOrders.
Use `python -m cdf.config` or the container's `config` command. All models
reject unknown fields.

## Minimal job

```yaml
api_version: 1.0.0
domain: pcb-aoi

scene:
  usd: board.usdc
  root: /Board

capture:
  renderer:
    resolution: [1920, 1080]
  camera:
    side: top
    rotation: {x: 0, y: 0, z: -90}
  scan_grid: [1, 1]
```

`capture` is optional when the job produces another artifact. To export the
final composed USD without rendering, configure execution output instead:

```yaml
execution:
  serialization_path: generated/final_stage.usda
```

Compile and print the normalized JobOrder plan without starting Isaac Sim:

```bash
uv run python -m cdf.config path/to/job.yaml
```

## Local Isaac Sim execution

From a repository checkout, use `uv run` for the authoring process and point
`ISAAC_SIM_PATH` at either the local Isaac Sim installation directory or its
bundled Python launcher. CDF resolves the launcher and executes the compiled
`JobOrderPlan` in an isolated Isaac Sim Python process; it does not invoke
`sdg_pipeline.py`.

Linux:

```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim
uv run python -m cdf.config path/to/job.yaml --execute
```

`ISAAC_SIM_PATH=/path/to/isaac-sim/python.sh` is equivalent.

Windows PowerShell:

```powershell
$env:ISAAC_SIM_PATH = "C:\path\to\IsaacSim"
uv run python -m cdf.config path\to\job.yaml --execute
```

Pointing `ISAAC_SIM_PATH` directly at `python.bat` is equivalent. If CDF is
already installed in the active Python environment, `python -m cdf.config`
may be used instead of `uv run python -m cdf.config`.

Override the configured output location with `--output-dir`. An override is
resolved relative to the current working directory; `output_dir` inside the
document is resolved relative to the document itself.

```bash
uv run python -m cdf.config path/to/job.yaml \
  --execute \
  --output-dir path/to/output
```

Use `--windowed` with `--execute` to show the Isaac Sim window.

## Container execution

The public release container exposes one YAML interface through `config`.
No local image build or Isaac Sim installation is required:

```bash
export IMAGE=nvcr.io/nvidia/cdf-simulation:1.2.0
docker pull "$IMAGE"
docker run --rm --gpus all \
  --volume /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  --volume "$PWD/job-input:/job:ro" \
  --volume "$PWD/job-output:/output" \
  "$IMAGE" \
  config /job/job.yaml \
  --execute \
  --output-dir /output
```

Compile-only mode is also available in the container by omitting `--execute`.
That behavior applies to versioned JobConfig, the only format the `config`
route accepts.

`--output-dir` and `--windowed` are versioned JobConfig options.

## Document structure

Every document contains the CDF authoring areas plus execution outputs. Only
`scene` is always required. A job must produce at least one capture,
operation-level artifact, or final-stage serialization.

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `api_version` | string | required | Exact CDF API contract used to validate and interpret the document. |
| `domain` | string | required | Domain registration name. This repository supports `pcb-aoi`. |
| `random_seed` | integer | `0` | Deterministic job seed carried into the compiled plan. |
| `output_dir` | string | `output` | Capture destination, resolved relative to the configuration. |
| `scene` | mapping | required | USD scene and optional component metadata. |
| `appearance` | mapping | `{}` | Lighting and material presentation. |
| `operations` | mapping | `{}` | PCB-AOI operations and solder generation. |
| `annotations` | mapping | `{}` | Component and operation-label semantics. |
| `capture` | mapping or null | `null` | Optional renderer, camera, scan, limits, and outputs. |
| `execution` | mapping | `{}` | Optional final composed-stage serialization. |
| `postprocess` | mapping | `{}` | Optional in-process postprocessing. |

Unknown fields are errors at every level. Structurally valid YAML is also
subject to the same semantic validation and dependency rules as Python: for
example, defect ratios must be valid, component operations require component
selection, and missing-component generation cannot be combined with transform
defects.

## Value shorthands

Fields described as a **number or range** accept either a scalar or a two-item
sequence:

```yaml
exposure: 0.25          # normalized to [0.25, 0.25]
exposure: [-0.25, 0.5] # sampled range
```

Light colors use one RGB triple, which is expanded into constant red, green,
and blue channel ranges:

```yaml
color: [1.0, 0.0, 0.0]
```

Camera `x` and `y` rotations also accept scalars or ranges. Camera `z` is a
fixed number, not a range.

## `scene`

```yaml
scene:
  usd: panel.usda
  root: /World/Panel
  instances:
    - /World/Panel/StepRepeats/SR_0_0
    - /World/Panel/StepRepeats/SR_0_1
  camera: /World/camera_light/Camera
  camera_xform: /World/camera_light
  ring_light: /World/camera_light/aoi_ring_light
  horizontal_aperture: 6.5
```

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `usd` | string | required | Local path, `s3://` URI, or ZIP containing the scene. |
| `root` | absolute USD prim path | required | Standalone board or panel-scene root. Must begin with `/`. |
| `instances` | sequence of absolute USD prim paths or null | `null` | Complete shared-prototype board cohort. |
| `camera` | absolute USD prim path or null | `null` | Existing camera prim. |
| `camera_xform` | absolute USD prim path or null | `null` | Parent transform for the camera. |
| `ring_light` | absolute USD prim path or null | `null` | Existing AOI ring-light root. |
| `horizontal_aperture` | number or null | `null` | Fixed camera horizontal aperture. |
| `components` | mapping or null | `null` | Component metadata source and optional type selection. |

Set `scene.components.types` to explicitly select component families. Add
`scene.components.sidecar` to anchor that selection to a legacy component
sidecar; without `sidecar`, the selection uses embedded USD metadata. If a
sidecar is supplied and `types` is omitted, selection is derived from that
sidecar.

For an embedded-metadata board, select types without attaching a sidecar:

```yaml
scene:
  usd: board.usd
  root: /Board
  components:
    types: [PASSIVE_SMD_0805]
```

For a standalone legacy board, select sidecar metadata instead of declaring
`instances`:

```yaml
scene:
  usd: board.usd
  root: /Board
  components:
    sidecar: board.components.json
    types: [PASSIVE_SMD_0805]
```

When `instances` is present, every path must identify a USD instance below
`root`, all paths must share one prototype, and the list must include every
matching instance in the composed stage. Operations are calculated once and
propagate identically to the cohort. YAML and Python use the same runtime validation;
the Python equivalent is `job.scene.instances(prim_paths=(...))`. Instance
cohorts require embedded USD component metadata and cannot be combined with a
legacy `pcb.components` sidecar.

Without `components`, component selection and operations use embedded USD
metadata. Supplying `components.sidecar` anchors the job to that legacy source
instead; embedded and sidecar metadata are not mixed.

### Component filters

`scene.components` is optional. When it is omitted and a downstream operation,
annotation, pin operation, or ROI selector needs components, CDF creates an
unrestricted component pool automatically. Its source is an attached
`pcb.components` sidecar when the job declares one, and the embedded USD
metadata otherwise. Therefore `components: {}` is allowed but is not required.

Use `component_filter` for explicit metadata-field matching:

```yaml
scene:
  usd: board.usd
  root: /Board
  components:
    component_filter:
      side: [Top]
      process_classes: [SMT]

operations:
  shift:
    ratio: 0.1
    translate: 0.2
    rotate_z: 15
    component_filter:
      package_type: [0603]
```

The supported fields are `refdes`, `component_type`, `package_type`,
`footprint_id`, `side`, `process_class`, and `process_classes`. Values within
one field are OR alternatives; populated fields intersect. Nested filters also
intersect their parent pool, so the shift above can only select top-side SMT
0603 parts. A nested filter never expands the scene or solder-joint pool.

The same structure is accepted by structural operations, solder-joint and
solder-defect operations, pin variation and pin-tip selection, hole planes,
geometry annotations, and capture ROI selection. Legacy `types`,
`component_types`, `components`, and `package_types` selectors remain supported
where documented, but cannot be combined with `component_filter` in the same
block. Exact filter fields require the selected metadata source to provide
those fields; missing sidecar data is a configuration error rather than an
implicit fallback to an overloaded legacy match.

### ZIP scenes

A local or S3-hosted `.zip` scene must contain exactly one root-level `.usd`,
`.usda`, or `.usdc` file:

```text
board.zip
├── board.usdc                 # selected scene
├── materials.mdl              # relative USD payload remains available
└── textures/
    └── solder.png
```

USD files below a subdirectory do not count as the root scene. Zero root-level
USD files or multiple root-level USD files produce an error. The archive is
localized for the execution scope so relative payloads remain available.

## `appearance`

```yaml
appearance:
  lighting:
    mode: rgb_light
    preserve_scene_light_color: false
    exposure: 0.25
    cone_angle: 120
    cone_softness: [0.75, 1.0]
    dome_fill:
      intensity: [500, 750]
    layers:
      Inner_Red:
        intensity: 6500
        color: [1.0, 0.0, 0.0]
  materials:
    carrier:
      material_path: /World/Looks/Carrier
      diffuse_texture: C:/textures/carrier_albedo.png
      roughness_texture: C:/textures/carrier_roughness.png
      normal_texture: C:/textures/carrier_normal_opengl.png
      diffuse_tint: [1.0, 0.3, 0.2]
      opacity: 0.5
      normal_strength: 0.6
      texture_scale: [0.006, 0.006]
      project_uvw: true
      world_space_projection: true
  vantablack_components: false
```

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `lighting` | mapping or null | `null` | Lighting behavior. No lighting declaration is authored when omitted. |
| `materials` | mapping or null | `null` | Named OmniPBR overrides for existing bound USD materials. |
| `vantablack_components` | boolean or null | `null` | Whether to apply the component presentation override. |

### `appearance.lighting`

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `mode` | string | required | Rig selector: `rgb_light`, `white_light`, `dome_light`, `modus_light`, `usd_light`. |
| `preserve_scene_light_color` | boolean | `false` | Preserve authored scene-light colors (applies to `usd_light`). |
| `exposure` | number or range, or null | `null` | Exposure range sampled per trigger. |
| `cone_angle` | number or range, or null | `null` | Light cone angle range. |
| `cone_softness` | number or range, or null | `null` | Light cone softness range. |
| `dome_fill` | light layer or null | `null` | Optional neutral-white fill for `rgb_light` or batch `modus_light`; intensity is sampled per trigger and recorded in lighting metadata. |
| `layers` | mapping of names to light layers, or null | `null` | Named ring-light layers; requires `mode: rgb_light`. |
| `dome` | light layer or null | `null` | DomeLight payload; requires `mode: white_light` or `mode: dome_light`. |
| `modus` | mapping or null | `null` | Modus rig knobs (`arc_radius_scale`, `width_frac`, `height_scale`, `elevations`/`intensities`/`colors` per `white`/`red`/`blue`); requires `mode: modus_light`. All optional -- `{}` accepts the rig defaults. |

A light layer requires `intensity`, a number or range. For a fixed light color,
use the optional three-number `color` RGB value. For independently randomized
channels, use `color_r`, `color_g`, and `color_b`, each as a number or range;
do not combine either form. `visible` defaults to `false`. Only `dome` accepts
`visible: true`; dome fill and ring layers are not authored as visible light
geometry. Under `mode: white_light` the dome colour is pinned to white, so
authoring either color form there is rejected -- use `mode: dome_light` for a
tint.

For `modus_light`, batch capture uses `capture.camera.side` (or the camera
inside a named pass) to place the fixed six-bar rig above the board's maximum
Z face or below its minimum Z face. The scene must be Z-up with the board in
the XY plane. The editor's live preview remains top-side only. This mode
approximates the illumination layout; its render-tuned defaults do not import
scanner PWM calibration, line-scan acquisition, color correction, or sharpening.

### `appearance.materials.<name>`

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `material_path` | string | required | Absolute path of the existing USD material to override. |
| `diffuse_texture` | string or null | `null` | Albedo texture asset path. |
| `roughness_texture` | string or null | `null` | Scalar roughness texture asset path. |
| `normal_texture` | string or null | `null` | Tangent-space normal texture asset path. |
| `diffuse_color` | RGB triple | `[0.2, 0.2, 0.2]` | Base color used without an albedo texture. |
| `diffuse_tint` | RGB triple | `[1, 1, 1]` | Multiplier applied to the albedo result. |
| `albedo_brightness` | number | `1.0` | Albedo brightness multiplier. |
| `opacity` | number | `1.0` | Constant surface opacity from `0.0` (transparent) to `1.0` (opaque). |
| `roughness` | number | `0.5` | Constant roughness value. |
| `roughness_texture_influence` | number | `1.0` | Blend weight for the roughness texture. |
| `metallic` | number | `0.0` | Metallic response. |
| `specular_level` | number | `0.5` | Dielectric/metal specular intensity. |
| `normal_strength` | number | `1.0` | Normal-map strength. |
| `normal_format` | string | `opengl` | Normal convention, `opengl` or `directx`. |
| `texture_scale` | number pair | `[1, 1]` | Projected texture tiling. |
| `texture_rotation` | number | `0.0` | Projected texture rotation in degrees. |
| `project_uvw` | boolean | `false` | Generate UVW coordinates for UV-less geometry. |
| `world_space_projection` | boolean | `false` | Use continuous world-space rather than object-space projection. |

## `operations`

All operation blocks are optional. Operations are authored in a fixed order,
regardless of YAML mapping order: shift, tombstone, sideflip, reverse polarity,
missing, pin variation, solder joints, hole plane, pin tip, then solder defects.

### Shift

```yaml
operations:
  shift:
    count: 2
    translate: 0.2
    rotate_z: 15
```

- `count`: exact positive number of eligible components to modify.
- `ratio`: fraction of eligible components to modify, from `0` to `1`.
- Supply exactly one of `count` or `ratio`.
- `translate`: required symmetric XY translation bound in millimeters.
- `rotate_z`: required symmetric board-normal rotation bound in degrees.
- `variant_index`: optional non-negative integer. Replaces the random draw with
  a **stratified assignment** — see [Stratified variants](#stratified-variants)
  below. Omit it and sampling is unchanged.

### Tombstone

```yaml
operations:
  tombstone:
    ratio: 0.2
    angle: [70, 90]
    component_types: [PASSIVE_SMD_0805]
    auto_filter: true
    eligibility_source: components
```

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `count` | integer | mutually exclusive with `ratio` | Exact number of eligible components to modify. |
| `ratio` | number | mutually exclusive with `count` | Fraction of eligible components to modify. |
| `angle` | two numbers | required | Minimum and maximum tombstone angles in degrees. |
| `component_types` | sequence of strings or null | `null` | Optional operation-specific type allow-list. |
| `auto_filter` | boolean | `true` | Apply metadata-catalog eligibility filtering. |
| `eligibility_source` | string or null | scene-selected source | Active component-metadata API source used for filtering. |

The scene selects exactly one component-metadata source for the job. Omitting
`scene.components.sidecar` selects embedded USD metadata; supplying it selects
the legacy component sidecar. An explicit operation-level
`eligibility_source` or solder-joint `source` must match that scene selection;
mixing the two sources in one job is rejected.

### Sideflip

```yaml
operations:
  sideflip:
    ratio: 0.1
    angle: [60, 80]
```

Supply exactly one positive `count` or a `ratio` between `0` and `1`; `angle`
is the required minimum and maximum rotation in degrees.

- `variant_index`: optional non-negative integer — see
  [Stratified variants](#stratified-variants) below.
- `rotate_z_jitter_deg`: optional symmetric bound, default `0.0`. Rotates the
  flipped part about the board normal — the same axis `shift` calls `rotate_z`
  — so a run of sideflips is not every part in the identical pose.
- `slide_range_mm`: optional symmetric bound, default `0.0`. Slides the flipped
  part along its own long axis.

Both jitter bounds default to `0.0`, which reproduces the previous behaviour
exactly.

### Stratified variants

Ask for a hundred shifts and you get a hundred independent draws, and
independent draws clump: a run can end up with eighty small translations in
roughly one direction and no large rotations at all, and nothing tells you.
`variant_index` replaces the draw with an **assignment** — sample *n* is told
which region of the space it belongs to — so coverage is a property of the run
rather than something measured after the GPU hours are spent.

It is **not a seed and not an identifier**. Two samples sharing an index share a
region, not a value; the numbers inside the region are still random.

| Operation | Axis | Advances every | Period |
|---|---|---|---|
| `shift` | mode (`translate` / `rotate` / `combined`) | 1 | **288** |
| | direction sector (8 x 45°) | 3 | |
| | translate magnitude bin (3) | 24 | |
| | rotation magnitude bin (2) | 72 | |
| | rotation sign | 144 | |
| `sideflip` | angle bin (3) | 1 | **6** |
| | hinge edge (2) | 3 | |

The strides form a mixed-radix odometer, which is what keeps the axes
independent: any 288 consecutive indices cover the shift grid exactly once. A
count that is not a multiple of the period leaves the tail partly covered
without skewing the earlier samples.

Because the value is per sample, a YAML job that declares one operation carries
one index. Sweeping the sequence is a Python-API pattern — see
[`cdf/API_SPEC.md`](../../cdf/API_SPEC.md) — or one job per index.

Full treatment, including how to verify the coverage a finished run achieved:
[`docs/defect-variants.md`](../../docs/defect-variants.md).

### Reverse polarity

```yaml
operations:
  reverse_polarity:
    ratio: 0.25
    component_types: [IC_SMD]
```

Supply either a positive `count` or a `ratio` between `0` and `1`.
`ratio` defaults to `1.0` when neither is supplied. `component_types`
optionally narrows the selected component families.

### Missing components

```yaml
operations:
  missing:
    count: 2
```

Missing components accepts exactly one positive `count` or a `ratio` between
`0` and `1`. An exact count fails when too few eligible components remain.
Missing-component generation must be the only defect family in a job. It
cannot be combined with shift, tombstone, sideflip, or reverse-polarity
operations.

### Through-hole pin variation

```yaml
operations:
  pin_variation:
    - components: [C1910, C1911]
      tilt: [0.0, 15.0]
      cut_angle: [0.0, 30.0]
      height: [0.0, 1.5]
      offset: [0.0, 0.30]
```

`pin_variation` is a sequence of groups. Each group selects through-hole leads
with exactly one of `components` (reference designators) or `package_types` and
supplies at least one variation band:

| Field | Unit | Bounds | Purpose |
|---|---|---|---|
| `tilt` | degrees | `0 <= min <= max <= 89` | Lead angle away from the board-normal axis. |
| `cut_angle` | degrees | `0 <= min <= max <= 89` | Trim-plane angle away from a cut perpendicular to the lead. |
| `height` | millimeters | `0 <= min <= max` | Protrusion beyond the far board face. |
| `offset` | millimeters | `0 <= min <= max` | Lateral displacement within the plated barrel. |

Bands are requests rather than guarantees. Barrel clearance can constrain tilt
and offset, while the available protrusion can constrain the cut angle. The
operation result reports the requested and delivered values and the binding
constraint for each lead. A selector that matches no eligible leads fails
instead of silently producing an unchanged board. If multiple groups select
the same lead, the last matching group wins.

Pin variation is compiled before solder generation, so generated solder is
built around the varied lead geometry. Verify the result from the board's
bottom side, where protrusion and cut geometry are visible. The board must
author lead and hole geometry roles plus hole-radius metadata.

There is no `nominal` mode. Omitting `pin_variation` preserves the untouched
nominal geometry. To model normal, in-tolerance manufacturing variation, use
process-qualified or measured-good bands; CDF does not classify a requested
band as nominal or defective. The panelised THT board worked example in
`examples/tht_pin_variation.yaml` documents refdes selection and ranges
verified on the 08/27 and 09/01 panel drops.

### Hole-reveal annotations

```yaml
operations:
  hole_plane:
    side: bottom
    class_name: hole
    z_offset_mm: 0.1
    components: [X204, X205]
```

`hole_plane` creates annotation-only geometry for drilled voids, which cannot
otherwise receive pixels from a semantic annotator. With `components`, it
creates one small quad for each selected component pad/hole site; without it, one
board-wide quad is occluded by the substrate everywhere except open holes.
An explicit component filter never falls back to the board-wide form. Unknown
refdes, components without pads, and individual proxy failures are reported as
warnings and in the operation result while other matched components continue.
`side` accepts `top`, `bottom`, or `both`. `output_prim_path`, when supplied,
must be an absolute non-root USD path and names the generated Xform parent.

The default parent is
`<board-root>/SegmentationProxies/HolePlane`. Its meshes carry the `hole`
semantic class and magenta USD display color, while the shared
`SegmentationProxies` hierarchy remains hidden by default. Reveal the proxies
only in an annotation pass with `visibility.show: [hole]`. The rendered class
color is controlled separately by Replicator or a stitched `color_map`.

### Through-hole pin-tip annotations

```yaml
operations:
  pin_tip:
    side: bottom
    class_name: pin_tip
    offset_mm: 0.002
    components: [X204, X205]
```

`pin_tip` generates annotation-only meshes by copying each selected terminal's
connected outward tip surface. Flat, steeply angled, triangulated, and curved
tips are supported; traversal stops at vertical lead walls and where the
surface tilts more than 30 degrees away from the tip's own outward normal. On a
rounded lead end that bound keeps the chamfer flanking the tip in the
terminal's `pin` class rather than absorbing it into the proxy. The
operation runs after pin variation, pin cutting, and solder attachment.
`offset_mm` displaces only the proxy toward the selected board side to avoid
z-fighting; it does not change the pin or solder geometry. Unknown refdes and
individual lead failures are warnings recorded in the operation result, so
other matched leads continue rendering.

Pin-tip proxies are grouped under
`<board-root>/SegmentationProxies/PinTips` and use a magenta USD display color.
The shared `SegmentationProxies` Xform and its generated children are hidden by
default. Reveal pin tips only in a solder-hidden semantic pass with
`visibility.show: [pin_tip]` to label the tip surface separately while the rest of
the terminal remains `pin`. Use label-aware stitching to choose an exact output
color:

```yaml
postprocess:
  stitch:
    source_pass: full_board_roi
    image_type: semantic_segmentation
    output_filename: board_sem.png
    blend: labels
    recolor_tiles: true
    color_map:
      pin_tip: [0, 255, 255, 255]
```

The RGBA color map applies to the stitched semantic image and its labels JSON.
With `recolor_tiles: true`, the source semantic masks and their per-tile label
sidecars are also rewritten with those unified colors. When
`recolor_tiles: false`, the raw masks retain the identifiers assigned by
Replicator.

### Solder joints

```yaml
operations:
  solder_joints:
    solver: procedural
    component_types: [PASSIVE_SMD_0805]
    source: components
    options:
      components: [C1, C2]
      height_ratio: 0.9
      resolution: 64
      solid: true
      tht_angular_segments: 64
      tht_radial_segments: 16
      tht_pad_edge_inset_mm: 0.0
      tht_solder_bath: true
      solder_material_prim_path: /World/Looks/SolderPaste
    output_root_path: /Board/GeneratedSolder
    serialization_path: generated/solder.usda
```

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `solver` | string | `none` | Registered solder solver. Always-present built-ins are `none` and `procedural`; `procedural` is a faster, more deterministic alternative to full physical simulation. `surface_evolver` (`simulated`, the full physical simulation) ships preinstalled in the container; outside it, see the repo root README's "Developer setup" section and `examples/smt_surface_evolver.yaml`. |
| `component_types` | sequence of strings or null | selected scene types | Optional type selection for solder generation. |
| `component_filter` | mapping or null | `null` | Exact metadata-field filter intersected with the scene component pool. |
| `source` | string or null | embedded USD metadata or attached sidecar | Board metadata source. |
| `options` | mapping | `{}` | Solver-specific JSON-compatible parameters. |
| `output_root_path` | absolute USD prim path or null | `null` | Root for generated solder prims. |
| `serialization_path` | string or null | `null` | Optional solder edit-layer destination. Its presence implies solder-joints-only serialization. |
| `serialization` | legacy string or null | `null` | Deprecated compatibility field. Prefer operation-level `serialization_path` for a solder layer or `execution.serialization_path` for the final composed stage. |

For the procedural solver, `options.components` is an exact refdes or component
instance prim-path allowlist. Omitting it or using an empty list selects every
component admitted by the effective scene, `component_filter`, and legacy
`component_types` filters. The allowlist is applied during discovery, and an
identifier outside that effective pool is a configuration error.

Procedural THT meshes default to 64 angular segments and 16 radial rings and
author smooth vertex normals. `tht_angular_segments` and
`tht_radial_segments` tune those axes independently. The shared `resolution`
option remains a backward-compatible THT angular override when the dedicated
option is absent. The primary effect of `tht_solder_bath: true` is to extend
each generated THT solder joint so it envelops the exposed solder-side portion
of its associated pin. This models the PCB assembly solder-bath process and
transitions the pin envelope smoothly into the generated volcano at its neck;
pin geometry on the opposite side of the neck is not coated. The option
defaults to `false`, preserving the volcano-only geometry.

Solder generation reads the source-neutral board metadata catalog, backed by
either embedded USD metadata or an explicitly attached legacy component
sidecar. It runs after component-transform operations and before final-stage
serialization and capture.

The `simulated` solver (`surface_evolver`) is a drop-in replacement for
`procedural` in the job: same `solder_joints` block, same ordering (transform
operations run first, so poses like tombstone get simulated solder on the moved
geometry). The shipped container has it preinstalled (prebuilt Evolver binary,
wheels in Isaac's interpreter). Outside the container it registers only where
its dependencies import — the one-time developer setup (build the Evolver
binary, install the wheels into the runtime's Python) is in the repo root
README's "Developer setup" section. Where the setup is missing, selecting it
fails fast with an unknown-solver error rather than falling back. `examples/smt_surface_evolver.yaml` is the working in-job example
(tombstone + `solver: surface_evolver` + `serialization: all`, which also writes
a self-contained solder-augmented USD).

Generated solder output, on both solvers: meshes whose material source does
not resolve bind the packaged reflowed-metal default
(`pcb_aoi/solder_generation/assets/solder.usda`; an explicit
`solder_material_prim_path` in `options` still wins), and simulated meshes
carry crease-aware vertex normals, so solder renders smooth and metallic
without per-job setup. On schema-carrying boards (`pcba:*` metadata),
through-hole joints are discovered from the board metadata's hole records and
SMT contact footprints are clipped to the exposed pad outlines — solder wets
the copper, never past it — while pre-schema boards keep the legacy raw-layer
and paste-footprint behavior unchanged.

### Solder defects

```yaml
operations:
  solder_joints:
    solver: surface_evolver          # solder defects require a simulated solver
    source: components
  insufficient:                      # each defect type is its own operation,
    ratio: 0.05                      # exactly like the pose defects
    volume_ratio: [0.3, 0.7]         # every scalar accepts a [min, max] range
    lead_climb: [0.25, 0.5]          # THT shape: wetted height / protrusion
    fullness: [0.4, 0.7]             # shape (both kinds): volume / straight cone
  bright:                            # through-hole only: steep starved meniscus
    ratio: 0.03
    lead_climb: [0.6, 0.9]
    fullness: [0.35, 0.6]
  excess:
    ratio: 0.02
    volume_ratio: [1.8, 2.6]
    lead_climb: [1.05, 1.35]         # > 1 buries the pin tip (hidden/excess)
    fullness: [1.0, 1.3]
  interrupted:                       # pad partially exposed, both kinds
    ratio: 0.03
    pad_exposure: [0.2, 0.5]
  missing_solder:
    ratio: 0.02
  dry_contact:                       # through-hole only
    ratio: 1.0
    component_types: [TO254]
    gap_fraction: [0.25, 0.5]
```

Per-component solder-quality defects on the generated joints. Each type
independently selects `ratio` of the (remaining) eligible pool —
non-overlapping across types and with the pose operations. Every parameter is
a **post-solve observable** — measurable on the produced geometry, never a
solver internal — and generation verifies it after the solve (a warning on
deviation; the measured values are then discarded, never exported).

| Type | Parameters | Meaning |
|---|---|---|
| `insufficient` | `volume_ratio` in (0, 1); `lead_climb`, `fullness` in (0, 2) | Starved joints. `volume_ratio` scales the nominal, sampled per component. The shape pair (sampled per joint) is the "less solder" valley on THT. |
| `bright` | `lead_climb`, `fullness` in (0, 2) | Steep starved THT meniscus: a real climb ([0.6, 0.9] default) with a necked-in fill ([0.35, 0.6]). Through-hole only. |
| `excess` | `volume_ratio` > 1; `lead_climb`, `fullness` in (0, 2) | Over-filled joints. On THT, `lead_climb` > 1 buries the pin tip (hidden/excess). |
| `interrupted` | `pad_exposure` in (0, 1); `lead_climb`, `fullness` | The pad itself is partially exposed: the foot stops short of the pad edge along a noisy dewetting front (azimuth always randomized). Sampled per joint. Both kinds; on THT the ring stays closed (a broken ring is `dry_contact`). |
| `missing_solder` | — | No solder forms; the printed paste is suppressed with it. The component itself stays (distinct from the `missing` operation). |
| `dry_contact` | `gap_fraction` in (0, 1); `lead_climb`, `fullness` | Fraction of the barrel circumference NOT wetted. Sampled per joint; the gap's azimuth is always randomized and never configured. Through-hole only. |

The two SHAPE parameters are dimensionless and component-invariant, sampled
per joint (values or `[min, max]` ranges): `lead_climb` (THT-only) is the
wetted lead height over the measured pin protrusion; `fullness` (both kinds)
is the fillet volume over that joint's own straight-cone fillet -- 1.0 is a
flat silhouette, < 1 concave, > 1 convex. When either is given it drives the
target volume directly and `volume_ratio` becomes measured-only on the joints
it shapes. Nominal values ride in `solder_joints.options` (`lead_climb: 0.8`,
`fullness: 0.85`, THT); an SMT joint with no explicit `fullness` keeps the
paste-driven nominal (the printed deposit is mass conservation).

Every type also takes `ratio` (selected fraction of the pool, like the pose
operations) and an optional `component_types` family allow-list; without one,
the type draws from the scene's selected component pool. Omitting an
observable uses the solver's default severity.

Constraints: solder defects require a `solder_joints` block declared first,
with a solver that honours defect intent (`procedural` silently ignores defect
records, so combining it with a solder defect is a validation error).
`dry_contact` on a component without plated holes is INELIGIBLE coverage, not
a silent nominal joint. `examples/solder_defects_board.yaml` is the worked
example.

## `annotations`

```yaml
annotations:
  components: true
  component_classes:
    passive: [R1, C1]
  operation_labels: defect
```

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `components` | boolean | `false` | Apply component labels. |
| `component_classes` | mapping of class names to reference-designator lists, or null | `null` | Explicit component classes; otherwise derive them from selection. |
| `operation_labels` | string or null | `null` | Enable operation labels under this semantic namespace. |

When `components: true`, the interpreter uses the job's selected board metadata
catalog as the annotation source: an explicitly attached sidecar selects the
legacy sidecar path; otherwise metadata comes from the USD stage. Enabling
operation labels also affects the default semantic type used by the Python
capture API; an explicit `capture.outputs.semantic_types` value takes
precedence.

For a validated `scene.instances` cohort, `components: true` uses the same YAML
contract but resolves labels after capture from instance-ID prim paths and the
embedded board metadata catalog. This avoids mutating the shared prototype and
preserves authored panel materials. Non-instanced boards continue to receive
Replicator semantics directly. Temporary instance-ID artifacts are retained
only when the capture output explicitly requests them.

```yaml
scene:
  usd: Panel.usda
  root: /World/Panel
  instances:
    - /World/Panel/Boards/Board_0
    - /World/Panel/Boards/Board_1
annotations:
  components: true
  component_classes:
    connector: [X204, X205]
capture:
  outputs:
    annotators: [rgb, semantic_segmentation]
```

`scene.instances` is the explicit behavior gate. CDF never guesses that a
stage is an instanced panel from its filename or prim hierarchy. Without this
field, YAML jobs retain direct Replicator semantic authoring. With a validated
cohort, CDF captures instance IDs internally and writes the same semantic
image and labels artifacts by post-capture projection. The metadata catalog is
selected once from either the attached sidecar or embedded USD metadata; the
sources are not combined.

See [component annotation execution paths](../../cdf/API_SPEC.md#component-annotation-execution-paths)
for output guarantees, THT board-pad membership, and intermediate instance-ID
cleanup.

## `execution`

```yaml
execution:
  serialization_path: generated/final_stage.usda
```

The path exports a flattened snapshot after deterministic scene, defect,
annotation, appearance, and solder mutations, but before capture begins. It may
be used with or without `capture`. Local relative paths resolve against the
configuration directory. The destination must end in `.usd`, `.usda`, or
`.usdc`.

## `capture`

The existing fields configure an implicit default render pass. To render
multiple views without reloading the scene, use the `passes` mapping instead:

```yaml
capture:
  passes:
    full_board:
      renderer:
        resolution: [1920, 1080]
      camera:
        side: bottom
        rotation: {z: -90}
      scan_grid: [1, 1]
    component_rois:
      renderer:
        resolution: [1920, 1080]
      camera:
        side: bottom
        rotation: {z: -90}
      scan_grid: [1, 1]
      visibility:
        hide: [solder]
      roi_selection:
        - refdes: [J1, J2]
          padding: 0.5
        - pins: [[J1, "1"], [J1, "2"]]
          padding: 0.25
        - component_types: [CAPACITOR]
          count: 2
        - process_classes: [through_hole]
          focus: pin
        - all_components: true
          focus: pin
        - defects: [shift, tombstone]
          padding: 0.75
```

Each named pass executes against the same prepared stage and writes beneath
`<output_dir>/<pass_name>`. The default-pass fields and `passes` may coexist;
direct fields retain their original output location.

### `visibility`

`visibility.hide` lists geometry hidden only while that render pass executes.
`visibility.show` lists board-role geometry revealed for the pass. Generated
`HolePlane` and `PinTips` Xforms are grouped beneath the default-hidden
`<board-root>/SegmentationProxies` Xform, so annotation passes should use
`show: [hole, pin_tip]`; RGB passes need no proxy-specific visibility configuration.
All affected geometry is restored to its exact prior authored visibility after the
pass, including when capture fails.

### `roi_selection`

`roi_selection` is a list of focused-capture selectors. Each entry must set
exactly one selector, and entries stack in declaration order.

| Selector | Selection behavior | `count` | Supported focus |
| --- | --- | --- | --- |
| `refdes` | Components in the supplied explicit order | Not accepted | Component or pin |
| `pins` | Exact `[refdes, pin_number]` identities | Not accepted | Always pin |
| `component_types` | Components matching type, family, package, or footprint metadata | Defaults to one deterministic match | Component or pin |
| `process_classes` | Components matching process metadata | All matches by default; deterministic subset when supplied | Component or pin |
| `component_filter` | Components matching explicit metadata fields within the scene pool | All matches by default; deterministic subset when supplied | Component or pin |
| `all_components: true` | Every component under the board metadata root | Not accepted | Component or pin |
| `defects` | Components targeted by the requested structural defects | All matches by default; deterministic subset when supplied | Component or pin |

Set `focus: pin` on any component selector to expand each selected component
into its cataloged physical terminals. Omitting `focus` preserves component
framing, while exact `pins` is always pin-focused. Mixed-process components use
terminal process metadata during process-class pin selection and fail safely
when that metadata cannot distinguish the requested pins.

Duplicate component or physical-terminal prim paths at the same focus level are
rendered once. Registered board instances receive one expanded target per source
target and instance.

Pin frames resolve their focus in this order:

1. A pad directly associated with the physical terminal.
2. A pad sharing the terminal's logical `pcba:pin:number`.
3. The nearest pad selected geometrically when multiple candidates remain.
4. The nearest generated solder joint when no pad exists.
5. The physical pin geometry when neither pad nor joint is available.

A metadata-only preflight detects duplicate `pcba:pin:number` values before any
geometry lookup. Ambiguous pad associations emit a runtime warning before the
world-space scan and retain that message in capture-result and trigger metadata
`warnings`. Capture metadata also records resolved pin, pad, joint, component,
selector, and panel-instance identities. `padding` supplies a nonnegative
fractional framing margin for each target.

The equivalent YAML pin-focused forms are:

```yaml
roi_selection:
  - refdes: [K2010]
    focus: pin
  - component_types: [ODD14901300.1000.7]
    focus: pin
  - process_classes: [through_hole]
    focus: pin
  - all_components: true
    focus: pin
  - pins: [[K2010, "4"]]
```

```yaml
capture:
  output_start: 0
  renderer:
    mode: pathtracing
    resolution: [1920, 1080]
    spp: 1
    total_spp: 32
    max_bounces: 4
  camera:
    side: bottom
    rotation:
      x: [-2, 2]
      y: 0
      z: -90
  scan_grid: [1, 1]
  visibility:
    hide: []
  limits:
    triggers: 1
    max_images: 1
  motion_blur:
    probability: 0.25
    alpha_range: [0.1, 0.2]
  outputs:
    annotators: [rgb, semantic_segmentation]
    image_format: png
    semantic_types: [class, defect]
    colorize: [semantic_segmentation]
    frame_padding: 4
    rename_to_grid_index: false
```

### Renderer

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `mode` | `pathtracing` | `pathtracing` | Renderer mode. |
| `resolution` | two integers | required | Output width and height. |
| `spp` | integer | `1` | Samples per pixel per accumulation step. |
| `total_spp` | integer | `32` | Total path-tracing samples per pixel. |
| `max_bounces` | integer | `4` | Maximum path-tracing bounce count. |

### Camera and scan

- `camera.side` is `top` by default and accepts `top` or `bottom`.
- `camera.rotation` is required. Its `x` and `y` values are optional numbers
  or ranges; `z` is an optional fixed number.
- `scan_grid` is the required grid, either the `[x_count, y_count]` shorthand
  or an object with the same fields as `capture.scan_grid(...)`:

  ```yaml
  scan_grid:
    x_count: 2
    y_count: 3
    scan_area_mm: [120.0, 80.0]   # optional; world rectangle the grid spans
    centered_on: panel            # optional; requires scan_area_mm
    image_size_px: [2834, 1889]   # optional; stitched canvas is cropped to it
  ```

  `scan_center_mm: [x, y]` (world millimetres) may replace `centered_on` to
  place the rectangle explicitly instead of centering it on the board; it also
  requires `scan_area_mm`. Both counts must be at least one. A pass with a `fov`
  section states neither form: `fov` derives the grid.
- `output_start` defaults to `0` and controls the first written frame index.

### Limits and augmentation

- `limits.triggers` defaults to `1` and must be at least one.
- `limits.max_images` defaults to `-1`, meaning unlimited.
- `motion_blur` is optional. Its recognized normalized fields include a
  `probability` ratio and an `alpha_range`; other JSON-compatible parameters
  pass through to the runtime augmentation implementation.

### Outputs

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `annotators` | sequence of strings | `[rgb]` | Enabled annotators. |
| `image_format` | string | `png` | Image writer format: one of `png`, `jpeg`, `jpg`, `bmp`. Any other value is rejected. `bmp` is produced by re-encoding the rgb frames after capture (the writer encodes only png and jpeg); masks, bounding boxes and label sidecars keep their native formats. Only postprocess steps that open the rgb frames themselves (`crop` with `rgb` in `types`, `stitch` of `rgb`, `format_roi_output` as `rgb_pass`) are rejected for a `bmp` pass with rgb enabled. |
| `semantic_types` | sequence of strings or null | `null` | Semantic namespaces requested from the writer. |
| `colorize` | sequence | `[]` | May contain `semantic_segmentation` and/or `instance_id_segmentation`. |
| `frame_padding` | integer | `4` | Numeric filename padding. |
| `rename_to_grid_index` | boolean | `false` | Rename sequential output suffixes to `x{x}_y{y}` grid coordinates after writing. |

Known annotators are `rgb`, `bounding_box_2d_tight`,
`bounding_box_2d_loose`, `bounding_box_3d`, `semantic_segmentation`, and
`instance_id_segmentation`. Additional annotator names are passed through as
enabled writer flags.

`rename_to_grid_index` applies to every artifact emitted by that capture pass
and preserves the numeric frame identifiers in `metadata.json`'s
`frame_to_grid` mapping. Use it for scan-grid passes; focused ROI captures can
map multiple frames to the same grid coordinate and should retain sequential
filenames. Scan-grid stitching accepts either sequential filenames or files
renamed with `rename_to_grid_index`.

## `postprocess`

```yaml
postprocess:
  close_openings:
    source_pass: full_board_roi
    class_name: hole
    output_subdir: stitched
    output_filename: board_sem.png
  stitch:
    source_pass: full_board_roi
    image_type: semantic_segmentation
    output_subdir: stitched
    output_filename: board_sem.png
    blend: labels
    color_map:
      pin: [255, 0, 0, 255]
      pad: [0, 255, 0, 255]
      hole: [0, 0, 255, 255]
      pin_tip: [0, 255, 255, 255]
  crop:
    method: bbox_tight
    types: [rgb, semantic_segmentation]
    offset: 10
    min_coverage: 0.2
    class_filter: [tombstone]
    xform_depth: 7
    output_subdir: cropped
  format_roi_output:
    rgb_pass: full_board
    roi_pass: full_board_roi
  generate_joint_rois:
    rgb_pass: full_board
    roi_pass: full_board_roi
```

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `method` | string | `bbox_tight` | In-process crop method. Only `bbox_tight` is currently supported. |
| `types` | sequence of strings or null | `[rgb, semantic_segmentation]` | Artifact types included in each crop. Must contain a semantic type. |
| `offset` | integer | `10` | Pixel padding around the tight box. |
| `min_coverage` | number | `0.2` | Minimum semantic coverage. |
| `class_filter` | sequence of strings or null | `null` | Optional class allow-list. |
| `xform_depth` | integer | `7` | Transform path depth used for grouping. |
| `output_subdir` | string | `cropped` | Crop destination below the capture output. |
| `mode` | string or null | `null` | Optional paired/missing workflow mode. |
| `input_subdir` | string or null | `null` | Required when `mode` is set. |
| `reference_subdir` | string or null | `null` | Required when `mode` is set. |

`stitch` fields:

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `source_pass` | string | `default` | Capture pass whose scan-grid frames are stitched. |
| `image_type` | string | `rgb` | BasicWriter image prefix to stitch. |
| `output_subdir` | string | `stitched` | Destination below the capture-pass output. |
| `output_filename` | filename | `rgb.png` | Filename used for a single-trigger result. |
| `blend` | string | `none` | `none` pastes tiles, `feather` blends RGB overlap, and `labels` unifies segmentation class colors. |
| `write_labels` | boolean | `false` | Merge label sidecars without remapping colors. Cannot be combined with `blend: labels`. |
| `recolor_tiles` | boolean | `false` | Rewrite segmentation tiles and their label sidecars with the unified colors from `blend: labels`. |
| `color_map` | mapping or null | `null` | Per-class RGBA overrides for `blend: labels`; `BACKGROUND` and `UNLABELLED` are fixed. |

Stitching consumes the canonical pixel layout emitted by an orthographic
scan-grid capture. It does not rerender frames and does not support component
ROI passes, tilted cameras, or non-cardinal Z rotations.

`close_openings` fields:

| Field | Type | Default | Purpose |
|---|---|---:|---|
| `source_pass` | string | `default` | Capture pass whose stitched label image is modified. |
| `class_name` | string | required | Semantic class whose closed rings are filled. |
| `output_subdir` | string | `stitched` | Directory containing the matching stitch output. |
| `output_filename` | filename | `rgb.png` | Matching stitched image filename. |
| `max_diameter_px` | integer | `48` | Largest opening eligible for filling. |
| `min_circularity` | number | `0.55` | Minimum filled-area-to-enclosing-disc ratio in `(0, 1]`. |

The compiler requires exactly one stitch with the same pass, directory, and
filename and schedules closure after that stitch.

`format_roi_output` names the RGB and semantic ROI passes. Both passes must
set `outputs.rename_to_grid_index: true`. It renames semantic tiles to
`semantic_mask_x<col>_y<row>.png`, merges label JSON, writes canonical
`metadata.json` to both pass directories, and checks that RGB and ROI grids
align. The label output retains the five required classes plus optional
`pin_tip`.

`generate_joint_rois` names the ROI pass and its coordinate-aligned RGB pass.
The ROI pass must request `bounding_box_2d_tight` and set
`outputs.rename_to_grid_index: true`. It converts per-tile boxes into stitched
full-board pixel coordinates and writes the same `joint_rois.json` beneath each
trigger directory and beside the stitched output for both named passes. A
single trigger writes `stitched/joint_rois.json`; multiple triggers use
`stitched/joint_rois_trigger_XXXX.json`. Each item includes an explicit pin
number, the exact contributing USD prim path or paths, plus component type,
type-source, native package type, native package name, footprint, and
process-class metadata from the selected board metadata catalog. When no
semantic type is authored, type resolution falls back
through family, package type, footprint identity, and package name; unavailable
values remain `null`. Joints come only from paths containing a
`StepRepeats/<instance>/.../Components/<refdes>/<lead>` hierarchy, and
`bbox_xyxy` is that pin box. `joint_bbox_xyxy` widens it to the whole joint by
adding the boxes of the pads and holes the board metadata catalog associates
with the same component pin; label pads in the ROI pass
(`annotations.geometry.pads`) for this to take effect. Joints whose pad is
missing keep `joint_bbox_xyxy == bbox_xyxy` and report
`joint_bbox_source: pin_only`. See the
[`joint_rois.json` schema reference](../../cdf/API_SPEC.md#joint_roisjson-schema-10)
for property types, nullability, coordinate semantics, and an example document.

## Complete examples

Runnable copies of these configurations are available under
[`examples/`](examples/README.md). They reference the checked-in functional
fixtures and include local Isaac Sim and container commands.

### Golden render without defects or a sidecar

```yaml
api_version: 1.0.0
domain: pcb-aoi
random_seed: 11
output_dir: output/golden-bottom

scene:
  usd: board.zip
  root: /Board

appearance:
  lighting:
    mode: white_light
    exposure: 0
    cone_angle: 120
    cone_softness: 1
    dome:
      intensity: 1000
      visible: true

capture:
  renderer:
    resolution: [1920, 1080]
    total_spp: 32
  camera:
    side: bottom
    rotation: {x: 0, y: 0, z: -90}
  scan_grid: [1, 1]
  limits: {triggers: 1, max_images: 1}
  outputs:
    annotators: [rgb]
```

### Defect render with component annotations and crops

```yaml
api_version: 1.0.0
domain: pcb-aoi
random_seed: 7
output_dir: output/tombstone

scene:
  usd: board.usdc
  root: /Board
  components:
    sidecar: board.components.json
    types: [PASSIVE_SMD_0805]

operations:
  tombstone:
    ratio: 0.25
    angle: [70, 90]

annotations:
  components: true
  operation_labels: defect

capture:
  renderer:
    resolution: [640, 360]
    total_spp: 4
  camera:
    side: top
    rotation: {x: 0, y: 0, z: -90}
  scan_grid: [1, 1]
  limits: {triggers: 1, max_images: 1}
  outputs:
    annotators: [rgb, semantic_segmentation]
    semantic_types: [class, defect]
    colorize: [semantic_segmentation]

postprocess:
  crop:
    types: [rgb, semantic_segmentation]
    class_filter: [tombstone]
    output_subdir: tombstone_closeups
```

### Missing-component render

```yaml
api_version: 1.0.0
domain: pcb-aoi

scene:
  usd: board.usdc
  root: /Board
  components:
    sidecar: board.components.json
    types: [PASSIVE_SMD_0805]

operations:
  missing:
    ratio: 0.5

annotations:
  components: true
  operation_labels: defect

capture:
  renderer:
    resolution: [640, 360]
  camera:
    side: top
    rotation: {x: 0, y: 0, z: -90}
  scan_grid: [1, 1]
  limits: {triggers: 1, max_images: 1}
  outputs:
    annotators: [rgb, semantic_segmentation]
    semantic_types: [class, defect]
```

## Local, S3, and ZIP path behavior

Path resolution depends on where the configuration itself lives:

| Configuration location | Relative `scene.usd`, sidecar, serialization, and `output_dir` values resolve against |
|---|---|
| `/jobs/board/job.yaml` | `/jobs/board/` |
| `C:\jobs\board\job.yaml` | `C:\jobs\board\` |
| `s3://pcb-inputs/jobs/board/job.yaml` | `s3://pcb-inputs/jobs/board/` |

Absolute local paths and explicit URIs are preserved. For example, a config in
one S3 bucket may point to an asset in another:

```yaml
scene:
  usd: s3://pcb-assets/releases/board-a.zip
  root: /Board
```

At execution time:

- An exact S3 object such as `s3://pcb-inputs/uploads/job-123/board-a.usdc`
  downloads only that object.
- An exact ZIP object such as
  `s3://pcb-inputs/uploads/job-123/board-a.zip` downloads only the archive,
  then extracts it locally for the execution scope.
- An S3 prefix such as `s3://pcb-inputs/uploads/job-123/` downloads only the
  objects below that prefix, never adjacent sibling objects.

S3 authentication and region selection use boto3's standard provider chain;
the YAML format does not accept credentials. Remote outputs are staged locally
and uploaded only after successful execution.

The configuration document itself may also live in S3:

```bash
python -m cdf.config \
  s3://pcb-inputs/jobs/job-123/job.yaml \
  --execute
```

Relative paths in that document resolve beneath
`s3://pcb-inputs/jobs/job-123/`.

## Schema, versions, and diagnostics

Generate the accepted JSON Schema from the selected domain models:

```bash
python -m cdf.config --print-schema --domain pcb-aoi > cdf-job-config.schema.json
```

The generated schema contains an API-version-specific `$id`. Do not hand-edit
generated schema output; the Pydantic models in this directory are the source.

Report both the installed package version and the independent authoring API:

```bash
python -m cdf.config --version
# cdf-simulation package: 1.2.0
# CDF authoring API:      1.0.0 (released)
```

`api_version` is mandatory. An unsupported value reports both the supplied and
accepted versions. CLI errors include the active CDF API version, and every
compiled plan and capture metadata record the API version for diagnostics.

The package release and authoring API version are intentionally independent.
After the API is published, contract changes follow the semantic-version policy
in [`../../cdf/API_CHANGELOG.md`](../../cdf/API_CHANGELOG.md).

## YAML and Python parity

The Python API is authoritative for semantic validation, declaration order,
dependency construction, and execution. The YAML interpreter calls that public
API in the following order:

```text
scene → appearance → operations → annotations → capture → postprocess
```

Equivalent YAML and Python descriptions are expected to produce exactly equal
normalized `JobOrderPlan` objects. The parity suite covers defaults, all
current PCB transform operations, the exclusive missing workflow, solder
generation, implicit sidecar behavior, annotations, capture settings, extra
annotators, and crop settings. The functional suite additionally executes a
small versioned YAML job through the packaged `config` entrypoint.

The parity suite also contains a coverage guard that reconciles registered PCB
operations, `OperationsConfig` fields, generated-schema properties, and the
operations named by exact-plan parity tests. Adding a Python operation without
a corresponding YAML model and behavioral parity case therefore fails CI. An
intentionally Python-only operation must be added to the suite's explicit
exception map with a reason pointing to the documented limitation.

See:

- [`../../tests/unittests/api_parity/test_job_config_parity.py`](../../tests/unittests/api_parity/test_job_config_parity.py)
- [`../../tests/functional/job_config_smoke.yaml`](../../tests/functional/job_config_smoke.yaml)
- [`../../cdf/API_SPEC.md`](../../cdf/API_SPEC.md) for the Python API contract

## Troubleshooting

### The config is reported as incompatible

Run `python -m cdf.config --version` and compare its authoring API version to
the document's `api_version`. Package version equality is not relevant.

### An asset path points somewhere unexpected

Remember that paths inside the document are relative to the document, while a
CLI `--output-dir` override is relative to the current working directory. Print
the compiled plan without `--execute` to inspect normalized paths.

### A ZIP scene is rejected

Place exactly one `.usd`, `.usda`, or `.usdc` file at the archive root. Nested
USD payloads and companion files are allowed, but nested files do not select
the root scene.

### A crop produces a validation error

Only `bbox_tight` is supported in this in-process path, and `types` must include
a semantic artifact so coverage can be measured. Omit `types` to use the safe
`[rgb, semantic_segmentation]` default.

### A component operation cannot find components

Attach `scene.components.sidecar`, then either provide `types` or allow the
selection to derive from the sidecar. Embedded USD property metadata is not yet
a substitute for the sidecar in this workflow.

### I only want to inspect the job

Omit `--execute`. Compilation validates the description and prints the complete
JobOrder plan without launching Isaac Sim or writing capture artifacts.
