# Functional container tests

The public DIG 1.2 release does not distribute the three board-derived USD
coupon fixtures previously used by these workflows. Coupon-backed Python
entrypoints are retained as templates and exit with skip status 77 until an
approved replacement asset is supplied. Their YAML counterparts and prompt
fixtures are compile-time examples only.

The release branch disables the coupon-backed GPU functional matrix. The
asset-independent `test_render_subprocess.py` smoke workflow remains available
for manual execution. Restore automated coupon coverage only after the
replacement fixtures and their redistribution terms have been approved.

## Workflows

| Script | Coverage |
|---|---|
| `test_smt_tombstone.py` | Select one SMT 0805 component, apply one tombstone defect, render annotations, and crop the defect. |
| `test_smt_shift.py` | Shift one SMT 0805 component, regenerate its solder with the procedural solver, render annotations, and crop the defect. |
| `test_smt_sideflip.py` | Sideflip one SMT 0805 component, verify the applied component, render aligned defect semantics, and crop the defect. |
| `test_smt_reverse_polarity.py` | Reverse one polarized Artix LED, verify the applied component, render aligned defect semantics, and crop the defect. |
| `test_smt_solder_defect.py` | Generate insufficient SMT solder with Surface Evolver and verify volume, mesh, fallback, warning, and semantic-output invariants. |
| `test_tht.py` | Select one THT component and render RGB, bounding-box, and class-segmentation output. THT defect coverage is a TODO. |
| `test_tht_surface_evolver.py` | Generate THT solder with Surface Evolver and verify mesh, fallback, warning, and semantic-output invariants. |
| `test_pin_variation.py` | Render nominal and varied THT controls; ROI variants also verify that default-hidden hole and curved/angled pin-tip proxies reach semantic capture when explicitly revealed. |
| `test_render_subprocess.py` | Call synchronous `CDFJob.capture.render()` from ordinary Python and verify plan/result transport through a real Isaac subprocess. |
| `job_config_smoke.yaml` | Load the versioned YAML API through the packaged `config` entrypoint and render one small RGB frame. Its exact `JobOrderPlan` parity with Python is covered by the unit suite. |
| `skill_prompts/*.yaml` | Nine versioned YAML files -- one per DIG Flow Catalog prompt the `cdf-simulation` skill serves (P01-P08) plus the default-coverage variant of the tombstone prompt (N07) -- each the job the skill hands to Airflow for that prompt on a repository coupon, with a hand-written Python API twin in `tests/unittests/api_parity/test_skill_prompt_yaml_api_parity.py`. They run through the same `config` entrypoint. |

`smt_scenario.py` contains the shared SMT coupon, lighting, capture, crop, CLI,
and Isaac execution harness. The two SMT test scripts provide only their
defect-specific operations and scenario metadata.

CI defines these workflows as explicit rows of one `parallel:matrix` job. Add a
new row with its entrypoint command, target, arguments, and expected artifact;
the shared template owns candidate-image verification, container cleanup,
artifact copying, and retention.

The coupon scripts default to a 1920 x 1080 render with 32 total samples per
pixel. The subprocess smoke test uses a fixed 64 x 64, one-sample capture. Local
executions write to `sdg_test_output/functional/<scenario>/`, which is ignored
by Git alongside the other SDG run output. CI overrides the output directory for
every workflow and the coupon resolution and sampling settings; its retained
artifacts remain under `functional-output/`.

## Compile a JobOrder locally

Compile-only mode does not start Isaac Sim. Run it from the repository root to
inspect the generated JobOrder:

```bash
python tests/functional/test_smt_tombstone.py \
  --resolution 640 360 \
  --total-spp 4

python tests/functional/test_smt_shift.py \
  --resolution 640 360 \
  --total-spp 4

python tests/functional/test_smt_sideflip.py \
  --resolution 640 360 \
  --total-spp 4

python tests/functional/test_smt_reverse_polarity.py \
  --resolution 640 360 \
  --total-spp 4

python tests/functional/test_tht.py \
  --resolution 640 360 \
  --total-spp 4

python tests/functional/test_render_subprocess.py
```

## Fixture policy

`ARTIX_LED_D1_coupon.usdc`, `ARTIX_PINHEADER_J13_coupon.usdc`,
`ICT_C0805_C22_coupon.usdc`, `SMT_0805_C18_coupon.usdc`,
`THT_TO254P762H300_U34_coupon.usdc` (with their `.components.json` sidecars and
preview images), `THT_SCHEMA_U1_coupon.usdc`, and the renders under
`resources/example_output/` are intentionally absent from this release. Do not
restore them, their Git LFS declarations, or their provenance manifest without
the corresponding redistribution approval. Supply an approved board asset and
update the test configuration when re-enabling a coupon-backed workflow.

The one file left in `resources/`, `make_tht_schema_coupon.py`, generates a
synthetic THT coupon locally; its output is not committed.

## Run the candidate container

Use the immutable digest-qualified candidate image produced by CI whenever
possible:

Only the `render-subprocess` invocation below is runnable without an approved
external board asset. The coupon-backed commands are retained as restoration
recipes and exit with status 77 in this release.

```bash
export CANDIDATE_IMAGE="nvcr.io/example/cdf-simulation@sha256:<digest>"
export FUNCTIONAL_OUTPUT="$PWD/functional-output"

mkdir -p "$FUNCTIONAL_OUTPUT/smt-tombstone" \
  "$FUNCTIONAL_OUTPUT/smt-shift" \
  "$FUNCTIONAL_OUTPUT/tht" \
  "$FUNCTIONAL_OUTPUT/render-subprocess" \
  "$FUNCTIONAL_OUTPUT/config-smoke"
chmod 777 "$FUNCTIONAL_OUTPUT/smt-tombstone" \
  "$FUNCTIONAL_OUTPUT/smt-shift" \
  "$FUNCTIONAL_OUTPUT/tht" \
  "$FUNCTIONAL_OUTPUT/render-subprocess" \
  "$FUNCTIONAL_OUTPUT/config-smoke"

docker run --rm --gpus all --network host \
  --volume /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  --volume "$FUNCTIONAL_OUTPUT/smt-tombstone:/functional-output" \
  "$CANDIDATE_IMAGE" \
  job-script tests/functional/test_smt_tombstone.py \
  --execute \
  --output-dir /functional-output \
  --resolution 640 360 \
  --total-spp 4

docker run --rm --gpus all --network host \
  --volume /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  --volume "$FUNCTIONAL_OUTPUT/smt-shift:/functional-output" \
  "$CANDIDATE_IMAGE" \
  job-script tests/functional/test_smt_shift.py \
  --execute \
  --output-dir /functional-output \
  --resolution 640 360 \
  --total-spp 4

docker run --rm --gpus all --network host \
  --volume /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  --volume "$FUNCTIONAL_OUTPUT/tht:/functional-output" \
  "$CANDIDATE_IMAGE" \
  job-script tests/functional/test_tht.py \
  --execute \
  --output-dir /functional-output \
  --resolution 640 360 \
  --total-spp 4

docker run --rm --gpus all --network host \
  --volume /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  --volume "$FUNCTIONAL_OUTPUT/render-subprocess:/functional-output" \
  "$CANDIDATE_IMAGE" \
  job-script tests/functional/test_render_subprocess.py \
  --execute \
  --output-dir /functional-output

docker run --rm --gpus all --network host \
  --volume /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  --volume "$FUNCTIONAL_OUTPUT/config-smoke:/functional-output" \
  "$CANDIDATE_IMAGE" \
  config tests/functional/job_config_smoke.yaml \
  --execute \
  --output-dir /functional-output
```

The SMT workflows produce capture output plus `tombstone_closeups` or
`shift_closeups`. The shift workflow's execution summary also reports solder
generation and attachment manifests. The THT placeholder produces capture
output without a defect or postprocessing stage. The subprocess workflow proves
that synchronous `render()` can launch Isaac, return a structured result, and
preserve output across the process boundary. The YAML smoke workflow proves that
the packaged YAML entrypoint produces a non-empty RGB artifact. CI records the
candidate image tag, digest, and source commit beside each job's retained output.
