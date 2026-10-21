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

"""Exercise procedural THT solder generation through the CDF container API.

Compile the JobOrder plan with ordinary Python from the repository root::

    python tests/functional/test_tht.py

Execute the plan inside the candidate container::

    docker run --rm --gpus all <candidate-image> \
        job-script tests/functional/test_tht.py --execute

The execution path uses ``CDFJob.capture.render_async()`` and
``create_default_isaac_executor()``. It does not generate YAML or invoke
``sdg_pipeline.py``.
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
from tests.functional.runtime import _skip_missing_fixture, execute_in_isaac

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
BOARD_USD = RESOURCE_DIR / "ARTIX_PINHEADER_J13_coupon.usdc"
COMPONENT_TYPE = "PinHeader_1x3_P2.54mm_Drill1.1mm"
EXPECTED_COMPONENT_COUNT = 1
EXPECTED_SOLDER_JOINT_COUNT = 3
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sdg_test_output" / "functional" / "tht"


def build_job(
    *,
    resolution: tuple[int, int] = (1920, 1080),
    total_spp: int = 32,
) -> tuple[CDFJob, int]:
    """Build the single-component procedural THT functional job.

    Args:
        resolution: Capture width and height.
        total_spp: Path-tracing samples per pixel.

    Returns:
        Authored CDF job and selected component count.

    Example:

    .. code-block:: python

        job, component_count = build_job(
            resolution=(320, 180),
            total_spp=2,
        )
        print(component_count)
        # 1

    """
    component_count = EXPECTED_COMPONENT_COUNT
    job = CDFJob(domain="pcb-aoi", random_seed=7)

    job.scene.configure(
        usd_path=BOARD_USD,
        root_path="/Board",
        camera_path="/World/camera_light/Camera",
        camera_xform_path="/World/camera_light",
        ring_light_root="/World/camera_light/aoi_ring_light",
    )
    job.scene.select_components(types=[COMPONENT_TYPE], source="usd")

    job.appearance.lighting(
        mode="rgb_light",
        preserve_scene_light_color=False,
        exposure_range=(0.25, 0.25),
        cone_angle_range=(120, 120),
        cone_softness_range=(1.0, 1.0),
        dome_fill={"intensity": (700, 700)},
        layers={
            "Inner_Red": {
                "intensity": (6500, 6500),
                "color_r": (1.0, 1.0),
                "color_g": (0.0, 0.0),
                "color_b": (0.0, 0.0),
            },
            "Middle_Green": {
                "intensity": (3000, 3000),
                "color_r": (0.0, 0.0),
                "color_g": (1.0, 1.0),
                "color_b": (0.0, 0.0),
            },
            "Outer_Blue": {
                "intensity": (4500, 4500),
                "color_r": (0.0, 0.0),
                "color_g": (0.0, 0.0),
                "color_b": (1.0, 1.0),
            },
        },
    )
    job.appearance.components(vantablack=False)

    # TODO: Add a THT defect operation when the `pcb-aoi` domain supports one.
    job.operations.solder_joints(
        solver="procedural",
        component_types=[COMPONENT_TYPE],
        source="usd",
        solver_options={
            "height_ratio": 0.9,
            "resolution": 20,
            "solid": True,
        },
    )
    job.annotations.label_components(source="usd")

    job.capture.output_start(0)
    job.capture.renderer(
        mode="pathtracing",
        resolution=resolution,
        spp=1,
        total_spp=total_spp,
        max_bounces=4,
    )
    job.capture.camera(
        side="bottom",
        rotation_range={"x": (0, 0), "y": (0, 0), "z": -90},
    )
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.limits(triggers=1, max_images=1)
    job.capture.outputs(
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        colorize_semantic_segmentation=True,
        semantic_types=["class"],
    )
    return job, component_count


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


def _print_plan(
    job: CDFJob,
    output_dir: Path,
    component_count: int,
) -> None:
    """Compile and print the functional test's JobOrder plan.

    Args:
        job: Authored functional job.
        output_dir: Capture destination embedded in the terminal order.
        component_count: Number of THT coupon components.

    """
    plan = job.compile_orders(output_dir=output_dir)
    print(f"selected {COMPONENT_TYPE} components: {component_count}")
    print("THT solder: procedural volcano generation enabled")
    print("THT defects: none")
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
        RuntimeError: If procedural solder does not produce all three joints.

    """
    started = time.perf_counter()
    result = await job.capture.render_async(
        output_dir=output_dir,
        executor=create_default_isaac_executor(),
    )
    elapsed_seconds = time.perf_counter() - started
    generation = result.outputs.get("solder_joints.generate")
    placement = result.outputs.get("solder_joints.attach")
    if not isinstance(generation, Mapping) or not isinstance(placement, Mapping):
        raise TypeError("procedural THT solder orders returned no structured output")
    coverage = generation.get("coverage")
    generated_paths = placement.get("generated_prim_paths")
    if (
        generation.get("artifact_count") != 1
        or not isinstance(coverage, list)
        or [item.get("status") for item in coverage] != ["generated"]
        or not isinstance(generated_paths, list)
        or len(generated_paths) != EXPECTED_SOLDER_JOINT_COUNT
    ):
        raise RuntimeError(
            "procedural THT solder did not generate the expected three joints: "
            f"generation={generation!r}, placement={placement!r}",
        )
    summary = {
        "success": result.success,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "executed_order_count": len(result.order_results),
        "solder_generation": generation,
        "solder_placement": placement,
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
    """Compile or execute the procedural THT functional job.

    Returns:
        Process exit status.

    """
    args = _parse_args()
    skip_code = _skip_missing_fixture(BOARD_USD)
    if skip_code is not None:
        return skip_code
    resolution = (int(args.resolution[0]), int(args.resolution[1]))
    output_dir = args.output_dir.expanduser().resolve()
    job, component_count = build_job(
        resolution=resolution,
        total_spp=args.total_spp,
    )
    if not args.execute:
        _print_plan(job, output_dir, component_count)
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
