# API parity tests

This package contains the CI tests that enforce 1-to-1 correspondence between
the Python authoring API (`CDFJob` and its six section classes) and the
versioned YAML JobConfig API (Pydantic config models).

These tests run in the dedicated `api-parity` CI job, separate from the general
`core-tests` job. The job selects only the `api_parity` capability group; see
`tests/unittests/README.md` for the complete lane-to-group mapping.

## What is tested

Three layers of enforcement live in `test_job_config_parity.py`:

1. **Domain registration guard** (`test_every_registered_operation_has_yaml_and_semantic_parity`)
   — every `add_*` method on `PcbAoiOperationsSection` must be registered in
   the `DOMAIN` frozenset, have a corresponding YAML model field, a JSON schema
   field, and a `@_covers_operations` behavioral test tag.

2. **API surface snapshot** (`test_api_surface_snapshot`)
   — committed JSON baselines (`python_api_surface.json`, `yaml_api_surface.json`)
   record every public method parameter and every Pydantic model field across
   all six authoring sections. Any unintentional change fails CI.

3. **Behavioral plan parity** (the `assert_plan_parity` family of tests)
   — verifies that YAML round-trips through the Python API produce equivalent
   compiled job plans.

## When you change the API

If you add, rename, or remove a public method parameter or a YAML Pydantic
field, the snapshot test will fail with a message telling you exactly what
changed. Regenerate the baselines by running:

```bash
PYTHONPATH="." python tests/unittests/api_parity/test_job_config_parity.py --update-snapshot
```

Then commit the updated `python_api_surface.json` and/or `yaml_api_surface.json`
alongside your API change.

## Python-only parameters

Some Python method parameters have no YAML equivalent (e.g. execution-only
flags, runtime-inferred values). Add them to the `_PYTHON_ONLY_PARAMS` table in
`test_job_config_parity.py` with a short reason string — an empty reason is
rejected.
