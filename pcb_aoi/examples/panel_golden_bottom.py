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

r"""Render a defect-free, full-panel golden image through CDF JobOrders.

The panel is named on the command line -- its scene file and its root prim --
so the example carries no board of its own and works for any panelised export.

The default invocation compiles and prints the JobOrder plan. Set
``ISAAC_SIM_PATH`` and pass ``--execute`` from ordinary Python to render through
the Isaac subprocess transport, without legacy YAML or ``sdg_pipeline.py``.

Examples:

.. code-block:: powershell

    python pcb_aoi/examples/panel_golden_bottom.py panel.usda --root-path /World/Panel
    $env:ISAAC_SIM_PATH = "C:\\path\\to\\IsaacSim"
    python pcb_aoi/examples/panel_golden_bottom.py panel.usda --root-path /World/Panel --execute

"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: F401 -- register the pcb-aoi domain
from cdf import CDFJob

_DEFAULT_OUTPUT = _REPO_ROOT / "sdg_test_output" / "panel_golden_bottom"


def build_job(
    scene_zip: str | Path,
    *,
    root_path: str,
    resolution: tuple[int, int] = (1920, 1080),
    total_spp: int = 64,
    scan_grid: tuple[int, int] = (1, 1),
    horizontal_aperture: float | None = None,
) -> CDFJob:
    """Build the defect-free bottom-side golden-render job.

    Args:
        scene_zip: USD scene or ZIP containing one root-level USD scene.
        root_path: Root prim of the panel inside that scene.
        resolution: Output width and height in pixels.
        total_spp: Total path-traced samples per pixel.
        scan_grid: Number of orthographic scan cells along world X and Y.
        horizontal_aperture: Optional USD camera aperture override.

    Returns:
        Declarative CDF job ready to compile or execute.

    """
    job = CDFJob(domain="pcb-aoi", random_seed=0)

    job.scene.configure(
        usd_path=Path(scene_zip).resolve(),
        root_path=root_path,
        camera_path="/World/camera_light/Camera",
        camera_xform_path="/World/camera_light",
        ring_light_root="/World/camera_light/aoi_ring_light",
        horizontal_aperture=horizontal_aperture,
    )

    # A CAD-prep panel export typically has no authored camera or usable
    # lights. Use a visible neutral dome for uniform illumination over the
    # full panel.
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

    # A 1x1 auto grid grows the orthographic aperture to cover the full panel.
    # Bottom-side mode mirrors the camera below the panel and turns it upward.
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
        rotation_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": -90.0},
    )
    job.capture.scan_grid(x_count=scan_grid[0], y_count=scan_grid[1])
    job.capture.limits(triggers=1, max_images=scan_grid[0] * scan_grid[1])
    job.capture.augmentation(
        motion_blur={
            "probability": 0.0,
            "alpha_range": (0.0, 0.0),
            "kernel_choices": [3],
        }
    )
    job.capture.outputs(
        rgb=True,
        image_format="png",
        bounding_box_2d_tight=False,
        bounding_box_2d_loose=False,
        bounding_box_3d=False,
        semantic_segmentation=False,
        colorize_semantic_segmentation=False,
        instance_id_segmentation=False,
        colorize_instance_id_segmentation=False,
        frame_padding=4,
    )
    return job


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene_zip",
        type=Path,
        help="panel USD, or a ZIP when archive support is available",
    )
    parser.add_argument(
        "--root-path",
        required=True,
        help="root prim of the panel inside the scene, e.g. /World/Panel",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="render output directory",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1920, 1080),
        help="output resolution",
    )
    parser.add_argument(
        "--total-spp",
        type=int,
        default=64,
        help="total path-traced samples per pixel",
    )
    parser.add_argument(
        "--scan-grid",
        nargs=2,
        type=int,
        metavar=("X_COUNT", "Y_COUNT"),
        default=(1, 1),
        help="orthographic tile counts along world X and Y",
    )
    parser.add_argument(
        "--horizontal-aperture",
        type=float,
        help="camera aperture override in USD tenths-of-scene-unit",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute JobOrders in Isaac Sim; otherwise print the plan",
    )
    return parser.parse_args()


def main() -> None:
    """Compile or execute the bottom-side golden render."""
    args = _parse_args()
    scene_zip = args.scene_zip.resolve()
    if not scene_zip.is_file():
        raise SystemExit(f"scene ZIP does not exist: {scene_zip}")
    if scene_zip.suffix.lower() not in {".usd", ".usda", ".usdc", ".zip"}:
        raise SystemExit(f"scene input must be USD or ZIP: {scene_zip}")
    if min(args.resolution) < 1:
        raise SystemExit("resolution dimensions must be positive")
    if args.total_spp < 1:
        raise SystemExit("--total-spp must be positive")
    if min(args.scan_grid) < 1:
        raise SystemExit("--scan-grid counts must be positive")
    if args.horizontal_aperture is not None and args.horizontal_aperture <= 0:
        raise SystemExit("--horizontal-aperture must be positive")

    if not args.root_path.startswith("/") or args.root_path == "/":
        raise SystemExit("--root-path must be an absolute prim path")

    job = build_job(
        scene_zip,
        root_path=args.root_path,
        resolution=tuple(args.resolution),
        total_spp=args.total_spp,
        scan_grid=tuple(args.scan_grid),
        horizontal_aperture=args.horizontal_aperture,
    )
    output_dir = args.output_dir.resolve()
    if not args.execute:
        plan = job.compile_orders(output_dir=output_dir)
        print(json.dumps(plan.model_dump(mode="json", by_alias=True), indent=2))
        print("dry run only; set ISAAC_SIM_PATH and pass --execute")
        return

    result = job.capture.render(output_dir)
    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    main()
