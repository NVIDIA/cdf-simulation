# Cosmos Data Factory — Simulation

**Package:** `cdf-simulation` · **Version:** 1.2.0 · **CDF authoring API:** 1.0.0 (released)

Cosmos Data Factory (CDF) Simulation generates labeled synthetic data
with NVIDIA Isaac Sim. Its PCB-AOI domain creates golden and defective printed
circuit board imagery, annotations, crops, stitched scans, and derived dataset
artifacts from USD assemblies.

Describe a job once in versioned YAML or Python. Both authoring surfaces
compile to the same validated `JobOrderPlan`; execution does not generate the
archived flow/PCBA YAML pair or invoke the legacy `sdg_pipeline.py`.

![Example PCB defect render under AOI ring lighting](docs/images/batch_render_aoi.png)

## What it does

- Loads local, S3, or ZIP-packaged USD scenes and board metadata.
- Configures AOI ring, Modus, dome, white, and authored USD lighting.
- Generates shift, tombstone, sideflip, reverse-polarity, and missing-component
  defects.
- Varies through-hole pins and generates procedural or Surface Evolver solder,
  including solder-quality defects.
- Captures RGB, segmentation, bounding boxes, and metadata over single views,
  scan grids, and scanner-defined fields of view.
- Post-processes captures into component crops, stitched images, joint ROIs,
  image-to-USD sidecars, and dataset layouts.

## Entry points

| Interface | Use it for | Start here |
|---|---|---|
| Versioned YAML | Reproducible jobs, validation, container execution | [`pcb_aoi/config/examples/`](pcb_aoi/config/examples/README.md) |
| Python API | Programmatic job construction and composition | [`cdf/API_SPEC.md`](cdf/API_SPEC.md) |
| Agent skills | Natural-language job authoring and simulation workflows | [Agent skills](#agent-skills) |

## Documentation

| Need | Document |
|---|---|
| Copy-paste task recipes | [`COOKBOOK.md`](COOKBOOK.md) |
| YAML fields, defaults, and path behavior | [PCB-AOI JobConfig guide](pcb_aoi/config/README.md) |
| Python signatures and validation contract | [`cdf/API_SPEC.md`](cdf/API_SPEC.md) |
| API overview and execution choices | [`cdf/README.md`](cdf/README.md) |
| Runnable YAML and Python examples | [`pcb_aoi/config/examples/`](pcb_aoi/config/examples/README.md) · [`pcb_aoi/examples/`](pcb_aoi/examples/README.md) |
| Functional scenarios and expected artifacts | [`tests/functional/README.md`](tests/functional/README.md) |
| Surface Evolver solver setup and design | [`pcb_aoi/solder_evolver/README.md`](pcb_aoi/solder_evolver/README.md) |
| Stratified defect coverage (`variant_index`) | [`docs/defect-variants.md`](docs/defect-variants.md) |
| Stitched image to USD sidecar | [`docs/image-to-usd-sidecar.md`](docs/image-to-usd-sidecar.md) |
| Fine-tune SMT component-crop appearance with Qwen-Image-Edit | [Sim2Real tutorial](docs/tutorials/sim2real-qwen-image-edit/README.md) |

---

## Requirements and access

| Component | Requirement |
|---|---|
| Authoring and tests | Python 3.12 or newer; [`uv`](https://docs.astral.sh/uv/) is recommended |
| Rendering | NVIDIA Isaac Sim 6.1.0, through this repository's image or a local installation |
| Container execution | Linux, Docker, NVIDIA Container Toolkit, and an NVIDIA GPU supported by Isaac Sim |
| Container image | Public release: `nvcr.io/nvidia/cdf-simulation:1.2.0`; no local build required |
| Board inputs | A prepared USD assembly and its referenced assets or sidecars; this release does not bundle board-derived sample USDs |

The Dockerfile is based on `nvcr.io/nvidia/isaac-sim:6.1.0`. Use the
[Isaac Sim compatibility checker and installation guide](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/index.html)
to qualify the host GPU, driver, operating system, and available resources.

### Get the repository

```bash
git clone https://github.com/NVIDIA/cdf-simulation.git
cd cdf-simulation
```

### Get the container image

Pull the public release image; rebuilding is not required for normal use:

```bash
export IMAGE=nvcr.io/nvidia/cdf-simulation:1.2.0
docker pull "$IMAGE"
```

For development, optionally build the image from this checkout. The build pulls
the Isaac Sim base image and compiles the vendored Surface Evolver binary:

```bash
docker build -t cdf-simulation:local .
export IMAGE=cdf-simulation:local
```

### Developer setup: simulated solder solver (optional)

Container users need no additional setup: the image includes the solver binary
and Python dependencies. Developers running the solver from a checkout must
build the vendored binary and install the pinned dependencies as described in
the [Surface Evolver setup guide](pcb_aoi/solder_evolver/README.md).

To confirm the solver is usable from a given interpreter:

```bash
python -c "from pcb_aoi.solder_generation import registered_solver_ids; \
           print(registered_solver_ids(available_only=True))"
# expect 'surface_evolver' in the tuple
```

### Container networking

Docker's default bridge network is sufficient for SDG rendering, registration,
and cropping in isolation — host networking is not a functional requirement of
those steps by themselves.

The validated container recipe below uses `--network host` because Isaac Sim
may need an extension registry or a Nucleus service bound to the host. Keep it
unless all required extensions are cached and no host service is needed.

`--network host` removes network-namespace isolation (every co-located
container/process on the host can then reach ports the containerized process
binds), so enable it only for a trusted image and environment. See the
[Isaac Sim container documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)
for deployment-specific networking requirements.

---

## Quick start: render a prepared board

A job is one declaration across six authoring areas (`scene` / `appearance` /
`operations` / `annotations` / `capture` / `postprocess`) plus execution
output, written either as versioned YAML or as Python. Both compile to the same
`JobOrderPlan`, so the two are interchangeable spellings of one job. Runnable
YAML examples live in
[`pcb_aoi/config/examples/`](pcb_aoi/config/examples/README.md) and Python
scripts in [`pcb_aoi/examples/`](pcb_aoi/examples/README.md). Several examples
retain the declarations used during development, but their board-derived test
fixtures are intentionally not distributed in this release: their `scene.usd`
is the placeholder `/path/to/your_board.usdc`. Update `scene.usd` and the
board-specific selectors before executing them.

Supply prepared USD assets and their board-specific metadata and configuration.
CAD-to-USD preparation tooling is not included in this release.

After adapting `minimal.yaml` to your board, mount both the configuration and
its asset directory into the container:

```bash
export OUTPUT="$PWD/sdg_test_output/quickstart"
export KIT_CACHE="$HOME/.cache/cdf-kit"
export JOB_CONFIG=/absolute/path/to/job.yaml
export BOARD_ASSETS=/absolute/path/to/board-assets

mkdir -p "$OUTPUT" "$KIT_CACHE"
# The image runs as isaac-sim (UID/GID 1234:1234).
sudo chown -R 1234:1234 "$OUTPUT" "$KIT_CACHE"
sudo chmod -R u=rwX,go=rX "$OUTPUT" "$KIT_CACHE"
test -f /usr/share/nvidia/nvoptix.bin

docker run --rm --gpus all --network host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v "$KIT_CACHE:/isaac-sim/kit/cache" \
  -v "$OUTPUT:/output" \
  -v "$JOB_CONFIG:/workspace/job.yaml:ro" \
  -v "$BOARD_ASSETS:/input:ro" \
  "$IMAGE" \
  config /workspace/job.yaml \
  --config-format job \
  --execute \
  --output-dir /output
```

The environment flags accept the Isaac Sim container EULA and privacy-consent
setting. Review both before running the command in your environment.

**Success:** the command exits with status 0 and this prints at least one RGB
artifact:

```bash
find "$OUTPUT" -type f -name 'rgb_*.png' -print
```

The `config` entrypoint validates the declared API version, compiles the YAML,
and executes the normalized plan through Isaac Sim. The Kit cache preserves
compiled RTX shaders between containers; without it, a fresh container can
spend roughly 110 seconds compiling before its first render. For an annotated
defect and component crop, substitute
`pcb_aoi/config/examples/smt_tombstone.yaml`.

## Other ways to run

### Compile YAML without starting Isaac Sim

Compilation validates the job and prints its normalized plan. It does not need
a GPU:

```bash
uv sync
uv run python -m cdf.config pcb_aoi/config/examples/minimal.yaml
```

Generate a JSON Schema for editor or CI validation with:

```bash
uv run python -m cdf.config --print-schema --domain pcb-aoi
```

### YAML with a local Isaac Sim

```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim
uv run python -m cdf.config \
  pcb_aoi/config/examples/minimal.yaml \
  --execute
```

### Python

```bash
uv run python pcb_aoi/examples/environment_sensor_defects.py
uv run python pcb_aoi/examples/environment_sensor_defects.py --execute
```

Python uses the same normalized plan and executor as versioned YAML. The
checked-in scripts compile by default and render with `--execute`. Full
functional scenarios and their expected artifacts are described in
[`tests/functional/README.md`](tests/functional/README.md).

### Your own board

Point `scene.usd` at the board's USD (a local path, an `s3://` URI, or a ZIP
holding one root-level USD) and `scene.root` at its assembly prim. The pipeline
never names a board: prim paths, asset names, and part numbers belong to the
job and its board-local assets. Camera and assembly paths must be configured or
derived from USD composition; ambiguous multi-board scenes require an explicit
path.

These sample PCBA USD assemblies follow the
[SimReady PCB specification](https://github.com/NVIDIA/simready-foundation/tree/feature/pcb)
for the metadata that identifies each board and its individual components.

Sample PCBA USD packages for the **Artix DC-SCM** and **ICT Baseboard** are
available from the [CDF PCB USD Dataset](https://huggingface.co/datasets/nvidia/cdf-pcb-usd-dataset).
After downloading and extracting a package, use its root stage as `scene.usd`:

- Artix DC-SCM: `artix-dc-scm/artix-dc-scm.usda`
- ICT Baseboard: `ict-baseboard/ict-baseboard.usda`

Start from one of these templates:

- [`minimal.yaml`](pcb_aoi/config/examples/minimal.yaml) for a local USD.
- [`local_zip.yaml`](pcb_aoi/config/examples/local_zip.yaml) for a ZIP scene.
- [`s3_usd_input.yaml`](pcb_aoi/config/examples/s3_usd_input.yaml) for an S3 USD.
- [`s3_zip_input.yaml`](pcb_aoi/config/examples/s3_zip_input.yaml) for an S3 ZIP.
- [`tht_pin_variation.yaml`](pcb_aoi/config/examples/tht_pin_variation.yaml) for
  a panelized through-hole assembly.

## Agent skills

| Skill | Use for |
|---|---|
| [`skills/cdf`](skills/cdf/SKILL.md) | Generate and validate declarative `CDFJob` Python scripts |
| [`skills/cdf-simulation`](skills/cdf-simulation/SKILL.md) | Author CDF Python jobs for automated DIG Flow Catalog requests |

## Project structure

```text
cdf-simulation/
├── cdf/                  Domain-neutral authoring, planning, and execution API
├── pcb_aoi/                PCB-AOI domain, config models, geometry, and solvers
│   ├── config/examples/    Versioned YAML jobs
│   └── examples/           Python-authored jobs
├── scripts/                Runtime adapters, post-processing, and utilities
├── apps/                   Isaac Sim application configuration
├── skills/                 Natural-language workflow skills
├── tests/unittests/        CPU/unit and API-parity coverage
├── tests/functional/       Isaac Sim end-to-end sample scenarios
├── docs/                   Supported workflow documentation
├── Dockerfile              Isaac Sim 6.1.0 runtime image
└── pyproject.toml          Package metadata and Python compatibility
```

## Troubleshooting

- **`IMAGE` is empty or the pull is denied:** set
  `IMAGE=nvcr.io/nvidia/cdf-simulation:1.2.0` and retry `docker pull "$IMAGE"`.
  If access is still denied, check registry connectivity and any cached Docker
  credentials for `nvcr.io`.
- **`nvoptix.bin` is missing:** verify the NVIDIA driver installation on the
  host before starting a render.
- **The first render is much slower:** retain `/isaac-sim/kit/cache` across
  runs.
- **The output directory is unwritable:** assign the bind-mounted output and
  Kit cache to the image's `isaac-sim` user (`1234:1234`) as shown in the quick
  start. Do not make shared-host directories world-writable.
- **A config version is rejected:** run `python -m cdf.config --version` and
  consult [`cdf/VERSION.md`](cdf/VERSION.md) and
  [`cdf/API_CHANGELOG.md`](cdf/API_CHANGELOG.md).
- **A component operation resolves nothing or the stage has multiple boards:**
  supply the correct board root and metadata source; see the
  [JobConfig troubleshooting guide](pcb_aoi/config/README.md#troubleshooting).

## Contributing

External contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for the pull-request process and required DCO sign-off.

## License

Source code in this repository is licensed under the Apache License,
Version 2.0; see [LICENSE](LICENSE). Third-party runtime dependencies and
their licenses are documented in the
[`third_party` inventory](third_party/README.md). The corresponding full
license and attribution texts are collected in
[`third_party/LICENSE-3rd-party.txt`](third_party/LICENSE-3rd-party.txt).
