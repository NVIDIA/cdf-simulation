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

"""Render a distributable PCB with six SMT defects and procedural solder."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: F401
from cdf import CDFJob


@dataclass(frozen=True)
class BoardSpec:
    """Static render inputs derived from one distributable board catalog.

    Args:
        packages: Embedded package identities for the SMT population.
        tombstone_refdes: Two known two-terminal SMT component identities.
        random_seed: Deterministic operation-selection seed.
        solder_all: Whether solder generation should include every supported
            SMT and through-hole package in the catalog.

    """

    packages: tuple[str, ...]
    tombstone_refdes: tuple[str, str]
    random_seed: int = 24
    solder_all: bool = False


_BOARD_SPECS = {
    "artix-dc-scm": BoardSpec(
        packages=("R_0402_1005Metric", "C_0402_1005Metric"),
        tombstone_refdes=("R60", "R17"),
        random_seed=19,
        solder_all=True,
    ),
    "environment-sensor": BoardSpec(
        tombstone_refdes=("C8", "C5"),
        packages=(
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
        ),
    ),
    "environment-sensor-myra-baseboard": BoardSpec(
        tombstone_refdes=("R4", "R12"),
        packages=(
            "BAT_BAT-HLD-012-SMT-TR",
            "C_0402_1005Metric",
            "Conn_JST_SH_1x4_SM04B-SRSS-TB-LF-SN",
            "LED_0603_1608Metric_G",
            "LGA-64_11.4x11.4mm_Myra",
            "LGA-8_2.5x2.5mm_BME280",
            "R_0402_1005Metric",
            "SOD-923",
            "SOT-723",
            "Sensirion_DFN-4-1.5x1.5x0.5mm_P0.8mm",
        ),
    ),
    "gmsl-serializer": BoardSpec(
        tombstone_refdes=("C3", "R14"),
        packages=(
            "C_0402_1005Metric",
            "C_0603_1608Metric",
            "Conn_FFC_WE_68715014x22",
            "DFN-10_2.5x1mm",
            "D_0402_1005Metric",
            "LED_0603_1608Metric_G",
            "L_0806_2016Metric",
            "L_Bourns-SRP3212A",
            "L_Coilcraft-1210POC",
            "L_Coilcraft-MSS6132T",
            "L_Coilcraft-PFL1609",
            "R_0402_1005Metric",
            "R_0603_1608Metric",
            "Resonator_SMD_Abracon_ABM8G_3.2x2.5mm",
            "SOT-23-5",
            "SOT-523",
            "TQFN-32-1EP_5x5mm_P0.5mm",
            "TSOT23-6",
        ),
    ),
    "ict-baseboard": BoardSpec(
        packages=("R_0402_1005Metric", "C_0402_1005Metric"),
        tombstone_refdes=("R83", "R101"),
        random_seed=1,
        solder_all=True,
    ),
    "job-oculink-expansion": BoardSpec(
        tombstone_refdes=("R35", "R1"),
        random_seed=26,
        packages=(
            "C_0402_1005Metric",
            "C_0805_2012Metric",
            "R_0402_1005Metric",
            "SOT-23-8",
            "TSOT23-6",
            "WQFN-42-1EP_3.5x9mm_P0.5mm",
        ),
    ),
    "job-storage-expansion": BoardSpec(
        tombstone_refdes=("C24", "C1"),
        random_seed=1050,
        packages=(
            "C_0402_1005Metric",
            "C_0603_1608Metric",
            "LED_0603_1608Metric_G",
            "MicroSD_Amphenol_1140084168",
            "Nexperia_DFN-10_2.5x1mm_P0.5mm",
            "QFN-48-1EP_7x7x0.9mm_P0.5mm_EP5x5mm",
            "R_0402_1005Metric",
            "Resonator_SMD_Abracon_ABM8G_3.2x2.5mm",
            "SOD-523",
            "SOT-523",
            "TSOT23-6",
            "Texas_X2QFN-12_1.6x1.6mm_P0.4mm",
        ),
    ),
}


def build_job(
    board_name: str,
    *,
    resolution: tuple[int, int] = (3840, 2160),
    total_spp: int = 32,
) -> CDFJob:
    """Build one embedded-metadata distributable-board render job.

    Args:
        board_name: Directory and main USD stem beneath distributable boards.
        resolution: Full-board capture width and height.
        total_spp: Total path-tracing samples per pixel.

    Returns:
        Declarative CDF job ready to compile or execute.

    Raises:
        KeyError: If ``board_name`` is not in the audited board set.

    """
    spec = _BOARD_SPECS[board_name]
    board_usd = (
        _REPO_ROOT
        / "_boards"
        / "distributable-boards"
        / board_name
        / f"{board_name}.usda"
    )
    job = CDFJob(domain="pcb-aoi", random_seed=spec.random_seed)
    job.scene.configure(
        usd_path=board_usd,
        root_path="/Board",
        camera_path="/World/camera_light/Camera",
        camera_xform_path="/World/camera_light",
        ring_light_root="/World/camera_light/aoi_ring_light",
    )
    job.scene.select_components(types=list(spec.packages), source="usd")
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

    tombstone_paths = [
        f"/Board/Layers/Top/Components/{refdes}" for refdes in spec.tombstone_refdes
    ]
    job.operations.add_tombstone(
        count=2,
        angle_range=(70.0, 90.0),
        component_types=tombstone_paths,
        auto_filter=False,
        eligibility_source="usd",
    )
    job.operations.add_sideflip(count=2, angle_range=(60.0, 80.0))
    job.operations.add_shift(
        count=2,
        translate_range=1.0,
        rotate_z_range=25.0,
    )
    job.operations.solder_joints(
        solver="procedural",
        component_types=None if spec.solder_all else list(spec.packages),
        source="usd",
        solver_options={"lead_local_sizing": True},
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
    """Parse one distributable-board execution request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board_name", choices=sorted(_BOARD_SPECS))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resolution", type=int, nargs=2, default=(3840, 2160))
    parser.add_argument("--total-spp", type=int, default=32)
    return parser.parse_args(arguments)


def _operation_counts(result: object) -> dict[str, int]:
    """Return defect and solder counts from a completed render result."""
    counts = {}
    for order in result.order_results:
        output = order.output
        if order.order_id.startswith("operation."):
            counts[order.order_id.removeprefix("operation.")] = int(
                output.get("count", 0),
            )
        elif order.order_id == "solder_joints.generate":
            counts["solder_artifacts"] = int(output.get("artifact_count", 0))
    return counts


def main(arguments: Sequence[str] | None = None) -> int:
    """Compile or execute one audited distributable-board render."""
    options = _parse_args(arguments)
    total_start = perf_counter()
    build_start = perf_counter()
    job = build_job(
        options.board_name,
        resolution=(int(options.resolution[0]), int(options.resolution[1])),
        total_spp=options.total_spp,
    )
    build_seconds = perf_counter() - build_start
    output_dir = options.output_dir or (
        _REPO_ROOT
        / "sdg_test_output"
        / "distributable_board_defects"
        / options.board_name
    )
    output_dir = output_dir.expanduser().resolve()
    compile_start = perf_counter()
    plan = job.compile_orders(output_dir=output_dir)
    compile_seconds = perf_counter() - compile_start
    if not options.execute:
        print(json.dumps(plan.model_dump(mode="json", by_alias=True), indent=2))
        return 0

    execute_start = perf_counter()
    result = job.capture.render(output_dir=output_dir, windowed=False)
    execute_seconds = perf_counter() - execute_start
    timing = {
        "board": options.board_name,
        "success": result.success,
        "resolution": [int(options.resolution[0]), int(options.resolution[1])],
        "total_spp": options.total_spp,
        "build_seconds": build_seconds,
        "compile_seconds": compile_seconds,
        "execute_seconds": execute_seconds,
        "total_seconds": perf_counter() - total_start,
        "counts": _operation_counts(result),
        "output_dir": str(output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "timing.json").write_text(
        json.dumps(timing, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(timing, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
