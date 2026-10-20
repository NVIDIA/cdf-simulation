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

"""Exercise synchronous rendering through a real Isaac subprocess.

The functional container invokes this module through the ``job-script`` entry
point. Unlike the other functional jobs, it deliberately does not start
``SimulationApp`` itself. ``CDFJob.capture.render()`` must serialize the plan,
launch Isaac's bundled Python, and return the child's structured result.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: E402,F401 -- register the `pcb-aoi` domain
from cdf import CDF_API_VERSION, JobResult, CDFJob  # noqa: E402


_MINIMAL_BOARD = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 0.001
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "Board"
    {
        def Cube "Body"
        {
            double size = 100
            color3f[] primvars:displayColor = [(0.05, 0.35, 0.08)]
        }
    }
}
"""


def _parse_args() -> argparse.Namespace:
    """Parse the functional-test command line.

    Returns:
        Parsed functional-test arguments.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="render through the synchronous Isaac subprocess boundary",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sdg_test_output/functional/render-subprocess"),
        help="capture output directory",
    )
    return parser.parse_args()


def _build_job(scene: Path) -> CDFJob:
    """Build the minimal synchronous-render job.

    Args:
        scene: USDA scene written for this test invocation.

    Returns:
        Authored CDF job ready to compile or render.

    """
    job = CDFJob(domain="pcb-aoi", random_seed=11)
    job.scene.configure(usd_path=scene, root_path="/World/Board")
    job.appearance.lighting(
        mode="white_light",
        exposure_range=(0.0, 0.0),
        cone_angle_range=(120.0, 120.0),
        cone_softness_range=(1.0, 1.0),
        dome={
            "intensity": (1000.0, 1000.0),
            "visible": True,
        },
    )
    job.capture.renderer(
        mode="pathtracing",
        resolution=(64, 64),
        spp=1,
        total_spp=1,
        max_bounces=2,
    )
    job.capture.camera(
        side="top",
        rotation_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": 0.0},
    )
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.limits(triggers=1, max_images=1)
    job.capture.outputs(rgb=True, image_format="png")
    return job


def _validate_artifacts(output_dir: Path, result: JobResult) -> None:
    """Validate the transported result and real capture artifacts.

    Args:
        output_dir: Capture directory populated by the Isaac child.
        result: Structured result returned to the parent process.

    Raises:
        RuntimeError: If the result or generated artifacts are incomplete.

    """
    if not getattr(result, "success", False):
        raise RuntimeError("synchronous render returned an unsuccessful JobResult")
    capture_output = getattr(result, "capture_output", None)
    if not isinstance(capture_output, dict) or capture_output.get("frame_count") != 1:
        raise RuntimeError(f"unexpected capture result: {capture_output!r}")

    trigger_dir = output_dir / "trigger_0000"
    rgb = trigger_dir / "rgb_0000.png"
    metadata_path = trigger_dir / "metadata.json"
    if not rgb.is_file() or rgb.stat().st_size <= 100:
        raise RuntimeError(f"missing or empty RGB artifact: {rgb}")
    if rgb.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"RGB artifact is not a PNG: {rgb}")
    if not metadata_path.is_file():
        raise RuntimeError(f"missing capture metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("num_positions") != 1 or metadata.get("defects") != []:
        raise RuntimeError(f"unexpected capture metadata: {metadata!r}")
    plan = metadata.get("job_order_plan")
    if not isinstance(plan, dict) or plan.get("api_version") != CDF_API_VERSION:
        raise RuntimeError(f"metadata contains the wrong API version: {plan!r}")


def main() -> int:
    """Compile or execute the synchronous subprocess-render job.

    Returns:
        Process exit status.

    """
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="cdf-functional-render-") as directory:
        scene = Path(directory) / "board.usda"
        scene.write_text(_MINIMAL_BOARD, encoding="utf-8")
        job = _build_job(scene)

        if not args.execute:
            plan = job.compile_orders(output_dir=output_dir)
            print(json.dumps(plan.model_dump(mode="json", by_alias=True), indent=2))
            return 0

        result = job.capture.render(output_dir)
        _validate_artifacts(output_dir, result)
        print(
            json.dumps(
                {
                    "success": result.success,
                    "capture": result.capture_output,
                    "rgb": str(output_dir / "trigger_0000" / "rgb_0000.png"),
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
