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

"""Render environment-sensor with six SMT defects and component close-ups.

Compile the JobOrder plan from the repository root::

    python pcb_aoi/examples/environment_sensor_defects.py

Launch Isaac Sim and execute the job::

    python pcb_aoi/examples/environment_sensor_defects.py --execute

The job consumes only metadata embedded in the composed USD stage. It does not
attach a component sidecar or request solder-joint generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: F401
from cdf import CDFJob

_BOARD_USD = (
    _REPO_ROOT
    / "_boards"
    / "distributable-boards"
    / "environment-sensor"
    / "environment-sensor.usda"
)
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "sdg_test_output" / "environment_sensor_defects"
# Package identities authored by the board's 78 `processClass = "SMT"`
# components. The mixed-process USB-C receptacle is intentionally excluded.
_SMT_PACKAGES = (
    "BAT_BAT-HLD-012-SMT-TR",
    "C_0402_1005Metric",
    "Conn_JST_SH_1x4_SM04B-SRSS-TB-LF-SN",
    "FB_0603_1608Metric",
    "LED_0603_1608Metric_G",
    "LGA-8_2.5x2.5mm_BME280",
    "LQFP-48_7x7mm_P0.5mm",
    "QFN-32-1EP_5x5mm",
    "R_0402_1005Metric",
    "Resonator_SMD_Abracon_ABM3B_5x3.2mm",
    "Resonator_SMD_ECS_ECX-31B_3.2x1.5x0.9mm",
    "SOD-923",
    "SOP-8_3.9x5.05mm_P1.27mm",
    "SOT-23-3",
    "SOT-723",
    "Sensirion_DFN-4-1.5x1.5x0.5mm_P0.8mm",
)


def build_job(
    *,
    resolution: tuple[int, int] = (3840, 2160),
    total_spp: int = 32,
) -> CDFJob:
    """Build the sidecar-free environment-sensor defect-render job.

    Args:
        resolution: Full-board capture width and height.
        total_spp: Total path-tracing samples per pixel.

    Returns:
        Declarative CDF job ready to compile or execute.

    Example:

    .. code-block:: python

        job = build_job(resolution=(1920, 1080), total_spp=8)
        print(job.domain.name)
        pcb-aoi

    """
    job = CDFJob(domain="pcb-aoi", random_seed=24)
    job.scene.configure(
        usd_path=_BOARD_USD,
        root_path="/Board",
        camera_path="/World/camera_light/Camera",
        camera_xform_path="/World/camera_light",
        ring_light_root="/World/camera_light/aoi_ring_light",
    )
    job.scene.select_components(types=list(_SMT_PACKAGES))

    job.appearance.lighting(
        mode="white_light",
        preserve_scene_light_color=False,
        exposure_range=(0.0, 0.0),
        cone_angle_range=(120.0, 120.0),
        cone_softness_range=(1.0, 1.0),
        dome={
            "intensity": (1000.0, 1000.0),
            "visible": True,
        },
    )
    job.appearance.components(vantablack=False)

    # Reserve two known two-terminal SMT passives for a geometrically sensible
    # tombstone. Later operations exclude these and each other's selections.
    job.operations.add_tombstone(
        count=2,
        angle_range=(70.0, 90.0),
        component_types=["C8", "C5"],
        auto_filter=False,
    )
    job.operations.add_sideflip(
        count=2,
        angle_range=(60.0, 80.0),
    )
    job.operations.add_shift(
        count=2,
        translate_range=1.0,
        rotate_z_range=25.0,
    )
    job.annotations.enable_operation_labels(namespace="defect")

    job.capture.output_start(0)
    job.capture.renderer(
        mode="pathtracing",
        resolution=resolution,
        spp=1,
        total_spp=total_spp,
        max_bounces=4,
    )
    job.capture.camera(
        rotation_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": -90.0},
    )
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.limits(triggers=1, max_images=1)
    job.capture.outputs(
        rgb=True,
        image_format="png",
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        colorize_semantic_segmentation=True,
        semantic_types=["defect"],
        frame_padding=4,
    )
    job.postprocess.crop(
        method="bbox_tight",
        types=["rgb", "semantic_segmentation", "component_instance"],
        offset=64,
        min_coverage=0.0,
        class_filter=["tombstone", "sideflip", "shift"],
        xform_depth=7,
        output_subdir="defect_closeups",
    )
    return job


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for compilation or execution.

    Args:
        arguments: Optional arguments excluding the executable name.

    Returns:
        Parsed command-line namespace.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch Isaac Sim and execute; default is compile-only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="capture and close-up output directory",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(3840, 2160),
        metavar=("WIDTH", "HEIGHT"),
        help="full-board render resolution",
    )
    parser.add_argument(
        "--total-spp",
        type=int,
        default=32,
        help="total path-tracing samples per pixel",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Compile or execute the environment-sensor defect-render job.

    Args:
        arguments: Optional arguments excluding the executable name.

    Returns:
        Process exit status.

    """
    options = _parse_args(arguments)
    resolution = (int(options.resolution[0]), int(options.resolution[1]))
    output_dir = options.output_dir.expanduser().resolve()
    job = build_job(resolution=resolution, total_spp=options.total_spp)
    if not options.execute:
        plan = job.compile_orders(output_dir=output_dir)
        print(json.dumps(plan.model_dump(mode="json", by_alias=True), indent=2))
        return 0

    result = job.capture.render(output_dir=output_dir, windowed=False)
    print(json.dumps(asdict(result), indent=2, default=str))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
