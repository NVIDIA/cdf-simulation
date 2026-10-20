# CDF JobOrder Execution

This package converts declarative `CDFJob` calls into executable intent and
runs that intent through a selected backend.

```text
CDFJob and sections
  -> JobOrder compiler
  -> JobOrderPlan
  -> JobExecutor
  -> JobOrder handler
  -> backend runtime
  -> JobResult
```

The core package is domain-neutral. Isaac-specific contracts and execution live
under [`isaac/`](isaac/). See the API specification for
[component annotation execution paths](../API_SPEC.md#component-annotation-execution-paths).

## Core files

| File | Responsibility |
| --- | --- |
| [`job_order_types.py`](job_order_types.py) | Immutable Pydantic `JobOrder` and `JobOrderPlan` models, transport-safe parameter normalization, dependency validation, and stable ordering. |
| [`job_order_codec.py`](job_order_codec.py) | Deterministic JSON serialization and reconstruction of complete plans. In-process callers pass plan objects directly. |
| [`job_order_compiler.py`](job_order_compiler.py) | Lowers the declaration log owned by `CDFJob` into orders, validates known prerequisites, and adds terminal capture and postprocess ordering. |
| [`executor_base.py`](executor_base.py) | Common synchronous and asynchronous `JobExecutor` interface. Executors orchestrate complete plans. |
| [`handler_registry.py`](handler_registry.py) | Maps semantic order types to backend callables. Handlers are the dispatch boundary between an order and its implementation. |
| [`job_result_types.py`](job_result_types.py) | Per-order and aggregate execution results, including output lookup and failure diagnostics. |
| [`__init__.py`](__init__.py) | Public exports for the domain-neutral execution API. |

## Isaac backend

| File | Responsibility |
| --- | --- |
| [`isaac/in_process_executor.py`](isaac/in_process_executor.py) | Preflights handlers, sequences orders, awaits asynchronous work, and returns structured or partial results inside the current Isaac process. |
| [`isaac/execution_context.py`](isaac/execution_context.py) | Mutable state shared by one execution, including the stage, sidecars, selected components, plan metadata, and handler outputs. |
| [`pcb_aoi/execution/board_instances.py`](../../pcb_aoi/execution/board_instances.py) | Validates shared-board cohorts, owns typed cohort and semantic-projection state, expands source paths across placements, and selects instance-safe semantic capture. |
| [`isaac/job_order_parameters.py`](isaac/job_order_parameters.py) | Versioned Pydantic parameter types consumed by the built-in handlers. |
| [`isaac/default_handler_registry.py`](isaac/default_handler_registry.py) | Validates each order against its parameter type, invokes the runtime adapter, records output, and assembles the default registry. |
| [`isaac/job_order_runtime.py`](isaac/job_order_runtime.py) | Adapts validated semantic operations to existing scene, sidecar, annotation, defect, capture, and crop APIs. |
| [`isaac/capture_execution.py`](isaac/capture_execution.py) | Concrete Replicator capture lifecycle, including camera and lighting setup, writers, frame execution, metadata, and cleanup. |
| [`isaac/executor_factory.py`](isaac/executor_factory.py) | Constructs the default context, registry, and `IsaacInProcessExecutor` for normal callers. |
| [`isaac/subprocess_executor.py`](isaac/subprocess_executor.py) | Serializes plans, resolves Isaac Sim's bundled Python, launches the child, and reconstructs `JobResult`. |
| [`isaac/subprocess_entrypoint.py`](isaac/subprocess_entrypoint.py) | Deserializes plans and persists child completion before Kit shutdown. |
| [`isaac/app_runner.py`](isaac/app_runner.py) | Owns `SimulationApp` inside the child and executes the received plan directly. |
| [`isaac/__init__.py`](isaac/__init__.py) | Public exports for the Isaac backend. |

The default Isaac registry owns backend-generic handlers. Importing `pcb_aoi`
registers its Isaac domain adapter, which contributes PCB operations including
pin variation and annotation-only pin-tip proxy generation. Transform
operations share an aggregate record set so they select different components
and can be restored together after capture. Missing uses its existing
visibility APIs and produces aligned `reference/` and `defective/` passes; it
is intentionally rejected when combined with a transform operation.

## Public execution

Code already running inside Isaac Sim selects the in-process backend explicitly:

```python
from cdf import create_default_isaac_executor

result = await job.capture.render_async(
    output_dir="output",
    executor=create_default_isaac_executor(),
)
```

The synchronous `job.capture.render()` path runs from ordinary Python. It sends
the same serialized JobOrder plan to an Isaac child process and returns the
transported result. It does not write legacy YAML or launch `sdg_pipeline.py`.
`cdf.execution.job_result_types.JobResult` is returned by both synchronous
and asynchronous JobOrder execution.

## Remote storage dataflow

Both execution paths give Isaac and the crop implementation local filesystem
paths. The storage layer owns transfer and cleanup around that local-only
processing:

```text
s3://input/...                    s3://output/...
       |                                 ^
       | localize_tree()                 | output_tree() publishes on success
       v                                 |
temporary local input --processing--> temporary local output
```

An exact object URI such as `s3://bucket/jobs/123/board.usdc` or
`s3://bucket/jobs/123/board.zip` downloads only that object. A prefix URI such
as `s3://bucket/jobs/123/` (or the same nonexistent object path without the
trailing slash) downloads only descendants of that prefix. ZIP inputs are
extracted within the plan scope and must contain exactly one root-level `.usd`,
`.usda`, or `.usdc`; nested companion assets remain available while the plan
runs.

For JobOrder execution inside the Isaac process, ownership is divided as follows:

| Step | Owner |
| --- | --- |
| Complete-plan lifecycle | [`IsaacInProcessExecutor.execute_async()`](isaac/in_process_executor.py) enters [`IsaacExecutionContext.storage_scope()`](isaac/execution_context.py). |
| S3 input to local input | [`IsaacSdgRuntime.load_scene()`](isaac/job_order_runtime.py) calls `context.localize_tree()`, which delegates to [`file_utils.localize_tree()`](../../scripts/file_utils/api.py). |
| Local processing | [`execute_capture()`](isaac/capture_execution.py) writes capture artifacts. `IsaacSdgRuntime.crop_capture()` reads and writes the same staged output tree. |
| Local output to S3 | `IsaacSdgRuntime.render_capture()` calls `context.output_tree()`. Successful storage-scope exit publishes through [`s3_storage.output_tree()`](../../scripts/file_utils/s3_storage.py). |
| Cleanup or failure | Exiting `storage_scope()` closes both tree contexts. Failed plans discard staged output instead of publishing it. |

Individual sidecars do not require a tree lifecycle. They use
`file_utils.is_file()` and `file_utils.open()` for exact local paths or remote
objects.

## Run a script with a local Isaac Sim installation

For synchronous `capture.render()`, run ordinary Python from the repository
root and set `ISAAC_SIM_PATH`. CDF launches the bundled interpreter itself.

### Windows PowerShell

```powershell
$isaacSimDir = "C:\path\to\IsaacSim"
$env:ISAAC_SIM_PATH = $isaacSimDir
python pcb_aoi\examples\panel_golden_bottom.py panel.usda --root-path /World/Panel --execute
```

For an installation path containing spaces, quote the directory:

```powershell
$isaacSimDir = "C:\Applications\NVIDIA Isaac Sim"
$env:ISAAC_SIM_PATH = $isaacSimDir
```

Quoting the environment-variable value supports Isaac Sim installation paths
containing spaces.

### Linux Bash

```bash
isaac_sim_dir="/path/to/isaac-sim"
export ISAAC_SIM_PATH="${isaac_sim_dir}"
python pcb_aoi/examples/panel_golden_bottom.py panel.usda --root-path /World/Panel --execute
```

Quoting `ISAAC_SIM_PATH` also supports installation paths containing spaces.

When `--output-dir` is omitted, functional scripts write local artifacts below
`sdg_test_output/functional/`. CI supplies an explicit output directory and
continues to retain its artifacts below `functional-output/`.

`capture.render()` is responsible for starting the child, and the child owns
`SimulationApp` and drives the asynchronous in-process executor. Scripts that
already run inside Isaac continue to use `render_async()` and own that lifecycle
themselves. A cold Isaac Sim start can take several minutes while Kit and GPU
caches are populated; subsequent starts are normally faster.

## Adding an order type

1. Record the semantic order ID, type, and parameters from the appropriate
   CDF section.
2. Add compiler prerequisite validation or explicit dependencies only when the
   backend requires them.
3. Put a backend-generic versioned parameter type under
   `isaac/job_order_parameters.py`; keep domain-owned contracts with their
   registered adapter package.
4. Implement backend-generic behavior in `isaac/job_order_runtime.py`. Put
   substantial capture lifecycle work in `isaac/capture_execution.py`, and
   domain-specific behavior behind the domain adapter.
5. Register generic handlers in `isaac/default_handler_registry.py`. Domain
   adapters contribute their handlers through `isaac_handlers()` and register
   with `register_isaac_adapter(...)`.
6. Add compiler, contract, handler, and result/failure tests as applicable.

Public JobOrder features should extend versioned YAML for Python/YAML parity.
Extend the legacy YAML surface only when compatibility requirements explicitly
call for it.

## Subprocess transport

`JobOrderCodec` is the deterministic process boundary. The ordinary-Python
parent writes a plan transport, invokes
`cdf.execution.isaac.subprocess_entrypoint` with Isaac Sim's bundled Python,
and reconstructs `JobResult` from the child envelope. The child imports Isaac,
executes the plan with `IsaacInProcessExecutor`, and writes completion before
closing `SimulationApp`, because Kit shutdown may terminate the interpreter.
