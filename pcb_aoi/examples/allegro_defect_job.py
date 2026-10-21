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

"""Build a declarative CAD-board defect job.

Run from the repo root:

    python pcb_aoi/examples/allegro_defect_job.py                # compile only
    python pcb_aoi/examples/allegro_defect_job.py --execute      # launch Kit
                                                               # (needs ISAAC_SIM_PATH)
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: F401 -- registers the "pcb-aoi" domain
from cdf import CDFJob

repo = Path(os.environ.get("PCB_AOI_ROOT", Path(__file__).resolve().parents[2]))
board_dir = repo / "_boards/cad_board/CAD_BOARD"


def build_job():
    """Build the example PCB-AOI job."""
    job = CDFJob(domain="pcb-aoi", random_seed=0)

    # 1. Scene — raw CAD data plus its domain sidecar
    job.scene.configure(
        # Prepared AOI material overlay supplied with the external board assets.
        usd_path=board_dir / "CAD_BOARD_aoi.usd",
        root_path="/Board_CAD_BOARD",
        camera_path="/World/camera_light/Camera",
        camera_xform_path="/World/camera_light",
        ring_light_root="/World/camera_light/aoi_ring_light",
    )
    job.scene.attach_sidecar(
        name="components",
        kind="pcb.components",
        path=board_dir / "CAD_BOARD.components.json",
    )
    job.scene.select_components(source="components")

    # 2. Appearance — Allegro CAD scenes have no authored lights
    job.appearance.lighting(
        mode="rgb_light",
        preserve_scene_light_color=False,
        exposure_range=(0.01, 1.0),
        cone_angle_range=(90, 150),
        cone_softness_range=(0.5, 1.0),
        dome_fill={"intensity": (600, 800)},
        layers={
            "Inner_Red": {
                "intensity": (5000, 8000),
                "color_r": (0.95, 1.0),
                "color_g": (0.0, 0.05),
                "color_b": (0.0, 0.05),
            },
            "Middle_Green": {
                "intensity": (2000, 4000),
                "color_r": (0.0, 0.05),
                "color_g": (0.95, 1.0),
                "color_b": (0.0, 0.05),
            },
            "Outer_Blue": {
                "intensity": (3000, 6000),
                "color_r": (0.0, 0.05),
                "color_g": (0.0, 0.05),
                "color_b": (0.95, 1.0),
            },
        },
    )
    job.appearance.components(vantablack=False)

    # 3. Domain operations — tombstone eligibility consumes the sidecar
    job.operations.add_shift(ratio=0.20, translate_range=0.20, rotate_z_range=15)
    job.operations.add_tombstone(
        ratio=0.20,
        angle_range=(70, 90),
        component_types=None,
        auto_filter=True,
        eligibility_source="components",
    )
    job.operations.add_sideflip(ratio=0.20, angle_range=(70, 90))
    # reverse_polarity is disabled in this Allegro example.

    # 4. Annotations — class/refdes mappings come from the component sidecar
    job.annotations.label_components(source="components")
    job.annotations.enable_operation_labels(namespace="defect")

    # 5. Capture
    job.capture.output_start(0)
    job.capture.renderer(
        mode="pathtracing", resolution=(1920, 1080), spp=1, total_spp=32, max_bounces=4
    )
    job.capture.camera(rotation_range={"x": (0, 0), "y": (0, 0), "z": -90})
    job.capture.scan_grid(x_count=10, y_count=10)
    job.capture.limits(triggers=1, max_images=-1)
    job.capture.augmentation(
        motion_blur={
            "probability": 0.8,
            "alpha_range": (0.01, 0.3),
            "kernel_choices": [5, 7],
        }
    )
    job.capture.outputs(
        rgb=True,
        image_format="png",
        bounding_box_2d_tight=True,
        bounding_box_2d_loose=False,
        bounding_box_3d=False,
        semantic_segmentation=True,
        colorize_semantic_segmentation=True,
        instance_id_segmentation=False,
        semantic_types=["defect"],
        frame_padding=4,
    )
    job.postprocess.crop(
        method="bbox_tight",
        output_subdir="cropped",
        types=["rgb", "semantic_segmentation", "component_instance"],
        offset=10,
        min_coverage=0.2,
        class_filter=None,
        xform_depth=7,
    )
    return job


def main():
    """Compile or execute the example job from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch the pipeline (requires ISAAC_SIM_PATH); "
        "default is a dry-run compile",
    )
    args = parser.parse_args()

    job = build_job()
    output_dir = repo / "sdg_test_output/allegro_defect_image"
    if not args.execute:
        plan = job.compile_orders(output_dir=output_dir)
        print(json.dumps(plan.model_dump(mode="json", by_alias=True), indent=2))
        return

    result = job.capture.render(output_dir=output_dir)
    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    main()
