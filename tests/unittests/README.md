# Unit-test organization

Unit tests are assigned to capability groups at the `.py` file level. The
first directory beneath `tests/unittests` identifies the dependency group; an
optional second level preserves the subsystem that owns the test. CI lanes
compose one or more groups, so a module has one clear dependency owner even
when it runs in several environments.

## Capability groups

| Directory | Put a module here when it requires |
|---|---|
| `core/` | Only the baseline Python dependencies, mocks, or stage-free logic |
| `usd/` | A real `usd-core`/`pxr` runtime |
| `imaging/` | OpenCV, SciPy, Pillow-backed image operations, or image datasets |
| `solver/` | Surface Evolver, `usd-core`, or its heavy geometry dependencies |
| `api_parity/` | Python-to-YAML public API correspondence and snapshots |

Use the narrowest group that can execute the entire module. Preserve an
existing subsystem directory beneath that group instead of flattening its name
into every file. For example, declarative API tests live in
`core/declarative_api/test_job_config.py`, while crop tests live in
`imaging/crop/test_region.py`. A test that has no subsystem home may remain at
the group root.

If one file mixes independent tests with incompatible requirements, split it
into separate modules instead of assigning individual test methods to
different groups. Do not create a new top-level directory for a subsystem;
top-level directories are reserved for dependency groups used by CI.

## CI lane composition

A **lane** is one named CI test execution with a defined operating system,
installed dependency set, and test-group composition. A lane is not a test
tag or a directory: the group directories describe what a module requires,
while a lane describes the environment in which one or more groups run. For
example, the `usd` lane installs `usd-core` and runs both the `core` and `usd`
groups on Linux.

| CI lane | Groups executed |
|---|---|
| `core-linux` | `core` |
| `usd` | `core`, `usd` |
| `imaging` | `core`, `imaging` |
| `solver-linux` | `core`, `solver` (including coupon-derived USD geometry) |
| `api-parity` | `api_parity` |
| `windows` | `core`, `usd`, `imaging` |

Runtime lane jobs are configured in `pipelines/stage-test.yml`; the
contract-focused `api-parity` lane is configured in
`pipelines/stage-validate.yml` and reuses the same lane template. The runner in
`scripts/ci/run_unittests.py` discovers the selected directories, emits JUnit
and JSON reports, and validates every skip against `tests/skip_policy.json`.

## Running tests

Run a CI-equivalent group composition from an environment containing that
lane's pinned dependencies:

```powershell
python scripts/ci/run_unittests.py `
  --lane windows `
  --groups core,usd,imaging `
  --report-file test-results/windows.json
```

Run one module directly while developing it:

```powershell
python -m unittest `
  tests.unittests.core.test_smt_functional_scenarios -v
```

Moving or adding a test module requires updating its documentation references
when applicable. The CI structure tests enforce that every `test_*.py` module
remains inside one of the five capability directories; nested subsystem
directories do not create new lanes.
