# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise a solder-quality defect (insufficient) end to end in the container.

The SMT coupon job with the simulated solver plus ``add_insufficient(
ratio=1.0, volume_ratio=0.4)``: the single selected component draws the
defect deterministically, the solve targets 0.4x its nominal volume, and the
engine's post-solve verification checks the produced mesh against that
request. The assertions therefore pin the whole solder-defects chain:

* ``solder_defects.apply`` selected and recorded exactly one component;
* generation ran on ``surface_evolver`` itself with NO procedural fallback
  (a fallback would render a fine image under a wrong label);
* the defect work item is ``generated`` and generation reported NO warnings
  -- the observable verification (volume within tolerance of the requested
  0.4x) passed, so the label matches the geometry.

Compile the JobOrder plan with ordinary Python from the repository root::

    python tests/functional/test_smt_solder_defect.py

Execute the plan inside the candidate container::

    docker run --rm --gpus all <candidate-image> \
        job-script tests/functional/test_smt_solder_defect.py --execute
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pcb_aoi  # noqa: F401 -- registers the `pcb-aoi` CDF domain
from cdf import (
    JobResult,
    CDFJob,
    create_default_isaac_executor,
)
from tests.functional.runtime import (
    _require_generated_meshes,
    _require_semantic_capture,
    execute_in_isaac,
)
from tests.functional.smt_scenario import (
    COMPONENT_TYPE,
    configure_ring_appearance,
    _configure_scene,
    component_count,
)

SOLVER = "surface_evolver"
DEFECT = "insufficient"
VOLUME_RATIO = 0.4
EXPECTED_DEFECT_COUNT = 1
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sdg_test_output" / "functional" / "smt_solder_defect"


def build_job(
    *,
    resolution: tuple[int, int] = (1920, 1080),
    total_spp: int = 32,
) -> tuple[CDFJob, int]:
    """Build the single-component insufficient-solder functional job.

    Args:
        resolution: Capture width and height.
        total_spp: Path-tracing samples per pixel.

    Returns:
        Authored CDF job and selected component count.

    Example:

    .. code-block:: python

        job, selected = build_job(resolution=(320, 180), total_spp=2)
        print(selected)
        # 1

    """
    selected = component_count()
    job = CDFJob(domain="pcb-aoi", random_seed=7)

    _configure_scene(job)
    configure_ring_appearance(job)

    # The solve the defect rides on; the simulated solver is required (the
    # API rejects procedural + a solder defect by design).
    job.operations.solder_joints(
        solver=SOLVER,
        component_types=[COMPONENT_TYPE],
        source="usd",
    )
    # ratio 1.0 over a one-component pool: deterministic selection. The
    # volume observable is verified post-solve by the engine; a deviation
    # beyond tolerance surfaces as a generation warning, which the
    # assertions below reject.
    job.operations.add_insufficient(ratio=1.0, volume_ratio=VOLUME_RATIO)

    job.annotations.label_components(source="usd")
    job.annotations.enable_operation_labels(namespace="defect")

    job.capture.output_start(0)
    job.capture.renderer(
        mode="pathtracing",
        resolution=resolution,
        spp=1,
        total_spp=total_spp,
        max_bounces=4,
    )
    job.capture.camera(rotation_range={"x": (0, 0), "y": (0, 0), "z": -90})
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.limits(triggers=1, max_images=1)
    job.capture.outputs(
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        colorize_semantic_segmentation=True,
        semantic_types=["defect"],
    )
    return job, selected


def _parse_args() -> argparse.Namespace:
    """Parse functional-test command-line arguments.

    Returns:
        Parsed command-line arguments.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute in this Isaac Sim process; default is compile-only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="capture output directory",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(1920, 1080),
        metavar=("WIDTH", "HEIGHT"),
        help="capture resolution",
    )
    parser.add_argument(
        "--total-spp",
        type=int,
        default=32,
        help="path-tracing samples per pixel",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="show the Isaac Sim window during execution",
    )
    return parser.parse_args()


def _print_plan(job: CDFJob, output_dir: Path, selected: int) -> None:
    """Compile and print the functional test's JobOrder plan.

    Args:
        job: Authored functional job.
        output_dir: Capture destination embedded in the terminal order.
        selected: Number of selected coupon components.

    """
    plan = job.compile_orders(output_dir=output_dir)
    print(f"selected {COMPONENT_TYPE} components: {selected}")
    print(f"solder defect: {DEFECT} at volume_ratio {VOLUME_RATIO} on {SOLVER}")
    print(json.dumps(plan.model_dump(mode="json", by_alias=True), indent=2))


async def _execute_job(job: CDFJob, output_dir: Path) -> JobResult:
    """Execute the functional job through the public in-process API.

    Args:
        job: Authored functional job.
        output_dir: Capture destination.

    Returns:
        Structured result from the in-process executor.

    Raises:
        TypeError: If a solder operation returns an invalid result type.
        RuntimeError: If the defect was not applied, the simulated solve did
            not run itself, or the observable verification warned.

    """
    started = time.perf_counter()
    result = await job.capture.render_async(
        output_dir=output_dir,
        executor=create_default_isaac_executor(),
    )
    elapsed_seconds = time.perf_counter() - started
    applied = result.output_for_type("solder_defects.apply")
    generation = result.output_for_type("solder_joints.generate")
    placement = result.output_for_type("solder_joints.attach")
    if (
        not isinstance(applied, Mapping)
        or not isinstance(generation, Mapping)
        or not isinstance(placement, Mapping)
    ):
        raise TypeError("solder-defect orders returned no structured output")
    coverage = generation.get("coverage")
    generated_paths = placement.get("generated_prim_paths")
    if (
        applied.get("count") != EXPECTED_DEFECT_COUNT
        or generation.get("solver") != SOLVER
        or generation.get("artifact_count") != EXPECTED_DEFECT_COUNT
        or not isinstance(coverage, list)
        or [item.get("status") for item in coverage] != ["generated"]
        or any(item.get("fallback_from") for item in coverage)
        # The verification contract: the solved volume was checked against
        # the requested ratio (then discarded). Any deviation -- or any other
        # solve trouble -- surfaces here, and a labelled defect whose
        # geometry was not verified is exactly what this test must fail on.
        or generation.get("warnings")
        or placement.get("warnings")
        or not isinstance(generated_paths, list)
    ):
        raise RuntimeError(
            "insufficient-solder defect did not verify end to end: "
            f"applied={applied!r}, generation={generation!r}",
        )
    validated_meshes = _require_generated_meshes(generated_paths)
    semantic_capture = _require_semantic_capture(
        output_dir,
        expected_label=DEFECT,
    )
    summary = {
        "success": result.success,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "executed_order_count": len(result.order_results),
        "solder_defects": applied,
        "solder_generation": generation,
        "solder_attachment": placement,
        "validated_meshes": validated_meshes,
        "semantic_validation": semantic_capture,
        "capture": result.capture_output,
    }
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return result


def _execute_in_isaac(
    job: CDFJob,
    output_dir: Path,
    *,
    windowed: bool,
    resolution: tuple[int, int],
) -> JobResult:
    """Boot Isaac Sim and drive the asynchronous public executor call.

    Args:
        job: Authored functional job.
        output_dir: Capture destination.
        windowed: Whether to show the Isaac Sim window.
        resolution: Width and height supplied to the application window.

    Returns:
        Structured result from the in-process executor.

    Raises:
        RuntimeError: If Isaac Sim exits before execution finishes.

    """
    return execute_in_isaac(
        lambda: _execute_job(job, output_dir),
        output_dir,
        windowed=windowed,
        resolution=resolution,
    )


def main() -> int:
    """Compile or execute the insufficient-solder functional job.

    Returns:
        Process exit status.

    """
    args = _parse_args()
    resolution = (int(args.resolution[0]), int(args.resolution[1]))
    output_dir = args.output_dir.expanduser().resolve()
    job, selected = build_job(resolution=resolution, total_spp=args.total_spp)
    if not args.execute:
        _print_plan(job, output_dir, selected)
        return 0

    result = _execute_in_isaac(
        job,
        output_dir,
        windowed=args.windowed,
        resolution=resolution,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
