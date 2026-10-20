# cdf — Declarative CDF Job API

For the exact, skill-oriented Python authoring contract, including signatures,
validation rules, execution choices, and copyable script patterns, see
[`API_SPEC.md`](API_SPEC.md). Natural-language Python script requests route to
the dedicated [`cdf` skill](../skills/cdf/SKILL.md).

## Documentation map

| Document | Purpose |
| --- | --- |
| [`pcb_aoi/config/examples/README.md`](../pcb_aoi/config/examples/README.md) | Start here for runnable PCB-AOI YAML examples using local, S3, and ZIP inputs. |
| [`config/README.md`](config/README.md) | Understand the domain-neutral versioned YAML envelope and plugin boundary. |
| [`API_SPEC.md`](API_SPEC.md) | Author jobs in Python using the authoritative contract and script patterns. |
| [`CAPABILITIES.md`](CAPABILITIES.md) | Check feature support across the YAML, Python, skill, preview, and documentation surfaces. |
| [`execution/README.md`](execution/README.md) | Execution architecture and local Isaac Sim usage. |
| [`VERSION.md`](VERSION.md) | Authoritative public API version and release status. |
| [`API_CHANGELOG.md`](API_CHANGELOG.md) | Public API compatibility policy and version history. |

The declarative job description and execution API resolves a job described in
Python and its versioned YAML counterpart to the **same normalized
`JobOrderPlan`**. Executors consume that plan directly; the normal Python and
versioned YAML paths do not generate legacy flow/PCBA YAML or invoke
`sdg_pipeline.py`.

```python
import pcb_aoi                      # registers the "pcb-aoi" domain
from cdf import CDFJob

job = CDFJob(domain="pcb-aoi", random_seed=0)
job.scene.configure(usd_path=..., root_path="/Board_...", ...)
job.appearance.lighting(mode="rgb_light", layers={...}, ...)
job.operations.add_tombstone(count=2, angle_range=(70, 90), ...)
job.annotations.enable_operation_labels(namespace="defect")
job.capture.renderer(mode="pathtracing", resolution=(1920, 1080), ...)
job.postprocess.crop(method="bbox_tight", ...)
plan = job.compile_orders(output_dir=...)                  # compile + validate
result = job.capture.render(output_dir=...)                # execute JobOrders
```

A complete defect example is available at
[`pcb_aoi/examples/allegro_defect_job.py`](../pcb_aoi/examples/allegro_defect_job.py).

### YAML job configuration

`cdf.config` provides a versioned, strict YAML authoring format that is
interpreted through the same public Python API. It is an input format, not a
serialized `JobOrderPlan`: the normal Python validation, dependency compiler,
and executor remain authoritative.

See the complete [PCB-AOI YAML API guide](../pcb_aoi/config/README.md) for every field
and default, local/S3/ZIP path behavior, container usage, examples, legacy
migration boundaries, diagnostics, and YAML/Python parity guarantees.

The configuration groups intent by section and supplies readable shorthands:
constant scalar values expand to two-value ranges, RGB colors expand to the
Python API's channel ranges, component sidecars drive selection defaults, and
capture outputs are listed by name.

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
  tombstone:
    ratio: 0.2
    angle: [70, 90]
annotations:
  operation_labels: defect
capture:
  renderer:
    resolution: [1920, 1080]
  camera:
    rotation: {x: 0, y: 0, z: -90}
  scan_grid: [1, 1]
  outputs:
    annotators: [rgb]
```

Multiple captures can share one scene setup by placing complete capture
configurations under `capture.passes`. Each key becomes an output subdirectory
and corresponds to Python's `job.capture.render_pass(name)` context manager.

Paths inside a configuration are resolved relative to the configuration file.
Compile locally and print the resulting JobOrder plan with:

```bash
python -m cdf.config path/to/job.yaml
```

The configuration itself may also be stored in S3. Relative asset and output
paths then resolve against the object's S3 prefix. Credentials and region use
boto3's standard AWS configuration chain.

```bash
python -m cdf.config s3://my-bucket/jobs/board/job.yaml
```

The container exposes the same command through its unified `config` mode.
Versioned documents use this API; recognizable legacy flow YAML delegates to
`sdg_pipeline.py` with a migration warning. Versioned JobConfig remains
compile-only by default; pass `--execute` to boot Isaac Sim and execute the
compiled plan. Legacy routing requires `--execute` because the legacy pipeline
has no compile-only mode. See the
[format-detection rules](../pcb_aoi/config/README.md#versioned-jobconfig-versus-legacy-yaml).

```bash
docker run --rm --gpus all \
  -v "$PWD/job-input:/job" \
  "$IMAGE" \
  config s3://my-bucket/jobs/board/job.yaml --execute
```

Generate a JSON Schema for validating the accepted YAML structure with:

```bash
python -m cdf.config --print-schema --domain pcb-aoi
```

Report the installed package release and independent authoring API contract:

```bash
python -m cdf.config --version
# cdf-simulation package: 1.2.0
# CDF authoring API:      1.0.0 (released)
```

`api_version` uses `major.minor.patch` and targets the contract identified by
[`VERSION.md`](VERSION.md). The envelope under `cdf/config` and the Pydantic
models contributed by selected domain plugins define the YAML representation
of that API; the JSON Schema is generated from those models. The public Python
API remains authoritative for semantic validation and for translating intent
into JobOrders. See
[`API_CHANGELOG.md`](API_CHANGELOG.md) for compatibility rules and history.

Unsupported config versions report both the supplied and active API versions.
Every compiled `JobOrderPlan` also carries `metadata.api_version`; that value
survives subprocess transport and is written into capture metadata. CLI errors
include the active API version for reproducible diagnostics.

### In-process JobOrder execution

Code already running inside Isaac Sim can bypass YAML generation and execute a
compiled `JobOrderPlan` directly:

```python
from cdf import create_default_isaac_executor

result = await job.capture.render_async(
    output_dir="output",
    executor=create_default_isaac_executor(),
)
print(result.capture_output)
print(result.crop_output)
```

`render_async()` compiles the authored calls once, validates their hard
dependencies, and delegates the plan to the selected executor. It returns the
public `JobResult`, whose `order_results` preserve execution order and whose
`failed_order` exposes failure diagnostics.

The default in-process Isaac executor supports all existing PCB structural
operations: shift, tombstone, sideflip, reverse polarity, and missing. Transform
operations may be combined and select distinct components. Missing remains a
separate capture mode and cannot be combined with transform operations; it
writes aligned `reference/` and `defective/` passes for each scan position.

To run a Python script that builds and executes a `CDFJob` in the container,
use the `job-script` entrypoint mode. For example, from the repository root:

```bash
export IMAGE="nvcr.io/nv-metropolis-dev/metropolis-sdg/pcb-aoi-ov-sdg:<TAG>"
mkdir -p "$PWD/sdg_test_output/cdf-example"

docker run --rm --gpus all --network host \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v "$PWD/sdg_test_output/cdf-example:/output" \
  "$IMAGE" \
  job-script tests/functional/test_smt_tombstone.py \
  --execute \
  --output-dir /output \
  --resolution 640 360 \
  --total-spp 4
```

The `job-script` mode launches the script with Isaac Sim's Python environment.
This example authors a `CDFJob`, starts Isaac Sim, and executes the compiled
plan through the in-process executor.

The synchronous `job.execute()` method is intended for ordinary Python. It
compiles and serializes a `JobOrderPlan`, launches Isaac Sim's bundled Python,
and transports the resulting `JobResult` back to the caller. It does not write
flow/PCBA YAML or invoke `sdg_pipeline.py`. Set `ISAAC_SIM_PATH` to the Isaac
installation directory or bundled Python launcher. Use `compile_orders()` for
a non-executing dry run. Capture is optional; `output_dir` is required only
when capture passes are declared. Pass `serialization_path` to `job.execute()`
to export the flattened composed stage without rendering, or pass it alongside
`output_dir` to `capture.render()` to render and export in one job. Calling
`capture.render()` without either a render pass or `serialization_path` fails
before Isaac launches. `render_async()` remains available when the caller
already owns the Isaac application.

For the command used to launch a JobOrder script through a local Isaac Sim
installation, see
[Run a script with a local Isaac Sim installation](execution/README.md#run-a-script-with-a-local-isaac-sim-installation).

### Remote storage

Scene paths, attached sidecars, and capture output may use `s3://` URIs in
both synchronous and asynchronous JobOrder execution. Isaac and
crop implementations continue to receive local filesystem paths; the storage
layer downloads inputs, stages outputs, publishes only successful runs, and
cleans up its temporary trees.

An exact `s3://` object URI downloads only that key. If no exact object exists,
keys beneath `<uri>/` are downloaded as a prefix tree; a trailing slash requests
prefix behavior explicitly. Scene paths should identify a USD object or ZIP
archive, not a prefix directory.

Scene input may also be a local or S3 ZIP archive. USD candidates must be at
the archive root. A single `.usd`, `.usda`, or `.usdc` is selected
automatically; when multiple candidates exist, name one explicitly:

```python
job.scene.configure(
    usd_path="s3://bucket/input/boards.zip",
    archive_member="panel_top_assembly.usda",
    root_path="/World",
)
```

The complete archive is extracted so relative payloads and companion assets
remain available. Nested USD files are not scene candidates.

See [Remote storage dataflow](execution/README.md#remote-storage-dataflow) for
the lifecycle diagram and the function responsible for each transition.

## The seven areas

Public description calls only add validated specifications to the job plan —
they never mutate the USD stage. `compile_orders()` performs a
non-executing dry run. `job.execute()` transports that plan to an Isaac child
process, where the in-process executor runs it.

| Area | Calls | Compiles to |
|---|---|---|
| **scene** | `configure`, `attach_sidecar`, `select_components` | `scene.load`, `scene.sidecar.load`, `scene.components.select` |
| **appearance** | `lighting`, `components`, `material_override` | `appearance.lighting.configure`, `appearance.components.configure`, `appearance.material.override` |
| **operations** | `add_shift`, `add_tombstone`, `add_sideflip`, `add_reverse_polarity`, `add_missing`, `pin_variation`, `solder_joints`, `hole_plane`, `pin_tip` | `operation.*`, `hole_plane.generate`, and internal `solder_joints.*` orders |
| **annotations** | `label_components`, `enable_operation_labels` | `annotations.components.apply`, `annotations.operations.configure` |
| **capture** | `render_pass`, `output_start`, `renderer`, `camera`, `scan_grid`, `fov` (pcb-aoi: scan rectangle at a dpi or mm/px, derives resolution and grid), `visibility`, `roi_selection`, `limits`, `augmentation`, `outputs`, `render` | One namespaced `capture.*.configure` sequence and `capture.render` per pass |
| **execution output** | `execute(serialization_path=...)` or `capture.render(serialization_path=...)` | `stage.serialize` before capture |
| **postprocess** | `stitch`, `crop`, `format_roi_output`, `generate_joint_rois`, `close_openings` | `postprocess.*` orders dependent on their capture artifacts |

## Domains

The core package is domain-agnostic. A domain module registers its operation
vocabulary and sidecar kinds on import (`pcb_aoi` → structural defects, pin
variation, solder generation, hole planes, and pin-tip proxies;
`pcb.components`, `pcb.sim_manifest`). Unknown
domains or operations fail validation with the supported set listed.

## Non-goals

- Does not replace USD or Replicator APIs (defect math stays in
  `scripts/sdg/standalone/defects_utils/api.py`; crop in `scripts/crop/`).
- No YAML↔Python merge/precedence semantics — they are alternative authoring
  paths to the same plan.

## Tests

`tests/unittests/core/declarative_api/test_declarative_api.py` uses stdlib
`unittest` and is CI-safe. Run the core group through the lane-aware runner:

```sh
python scripts/ci/run_unittests.py \
  --lane core-linux \
  --groups core \
  --report-file test-results/core-linux.json
```

See `tests/unittests/README.md` for group placement and complete lane
composition.
