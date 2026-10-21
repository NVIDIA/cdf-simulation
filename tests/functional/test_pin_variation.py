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

"""Render through-hole pin variation through the CDF container API.

Unit tests establish that the maths is right and that the order validates.
They cannot establish that the varied geometry survives USD authoring and
reaches the renderer, which is what this does.

Compile the JobOrder plan with ordinary Python from the repository root::

    python tests/functional/test_pin_variation.py

Execute the plan inside the candidate container::

    docker run --rm --gpus all <candidate-image> \
        job-script tests/functional/test_pin_variation.py --execute

Two jobs run against the same coupon: one nominal, one varied. The nominal pass
is the control -- without it, "the pins look tilted" is an assertion about a
picture nobody compared against anything.

**One variant per process.** Two ``render_async`` passes in a single Isaac
session return an empty second frame -- measured by running the pair in both
orders, where whichever pass went second came back black. So the control and
the varied run are separate invocations, and ``--compare`` diffs their output
afterwards with no Isaac at all::

    <container> job-script tests/functional/test_pin_variation.py --execute --variant nominal
    <container> job-script tests/functional/test_pin_variation.py --execute --variant varied
    python tests/functional/test_pin_variation.py --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pcb_aoi  # noqa: F401 -- registers the `pcb-aoi` CDF domain
from cdf import CDFJob

COUPON = REPO_ROOT / "tests/functional/resources/ARTIX_PINHEADER_J13_coupon.usdc"
BOARD_ROOT = "/Board"
COMPONENT_REFDES = "J13"
PACKAGE_TYPE = "PinHeader_1x3_P2.54mm_Drill1.1mm"
RESOLUTION = (1024, 1024)

#: Bands wide enough that the barrel, not the request, is what binds.
TILT_RANGE = (0.0, 15.0)
HEIGHT_RANGE = (0.2, 1.4)
CUT_RANGE = (0.0, 45.0)


def build_job(
    *,
    vary: bool,
    output_dir: Path,
    seed: int,
    roi: bool = False,
    solder: bool = False,
) -> CDFJob:
    """Declare one capture of the coupon, varied or nominal.

    Args:
        vary: Whether to declare the pin_variation order.
        output_dir: Where the capture writes.
        seed: Job random seed, which is what drives the variation.
        roi: Whether to capture per-terminal ROIs with labelled ground truth
            rather than one plain RGB frame.
        solder: Whether to grow solder joints before capturing. The pair of
            runs with it on and off is the workflow this operation is FOR:
            the same leads, once as an AOI camera sees them and once with the
            pad, hole and lead separable.

    Returns:
        The authored job.

    """
    job = CDFJob(domain="pcb-aoi", random_seed=seed)
    job.scene.configure(usd_path=str(COUPON), root_path=BOARD_ROOT)
    # A dome bright enough for a full board blows a 12 mm coupon to pure
    # white, and two saturated frames compare equal no matter what the
    # geometry did. The ring rig is what the other THT functional test uses.
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
    if vary:
        job.operations.pin_variation(
            package_types=(PACKAGE_TYPE,),
            tilt_range=TILT_RANGE,
            height_variation=HEIGHT_RANGE,
            cut_angle=CUT_RANGE,
        )
    if solder:
        job.operations.solder_joints(
            solver="procedural",
            solver_options={
                "height_ratio": 0.9,
                "resolution": 24,
                "tht_radial_segments": 12,
                "tht_solder_bath": True,
                "solid": True,
            },
        )
    if roi:
        job.operations.hole_plane(side="bottom", components=(COMPONENT_REFDES,))
        job.operations.pin_tip(side="bottom", components=(COMPONENT_REFDES,))
        targets = {"terminals": "pin", "holes": "hole", "pads": "pad"}
        if solder:
            targets["solder"] = "solder"
        job.annotations.label_geometry(**targets)
    job.capture.output_start(0)
    job.capture.renderer(
        mode="pathtracing",
        resolution=RESOLUTION,
        spp=1,
        total_spp=48,
        max_bounces=4,
    )
    # BOTTOM side. Tilt, protrusion and the cut plane are all below the far
    # face, so a top-down pass is exactly the view this defect hides in.
    job.capture.camera(
        side="bottom",
        rotation_range={"x": (0, 0), "y": (0, 0), "z": 0},
    )
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.limits(triggers=1, max_images=1)
    if roi:
        job.capture.visibility(show=("hole", "pin_tip"))
        # One ROI per lead of the coupon's only part, which is the view
        # AOI actually has.
        job.capture.roi_selection(
            pins=(
                (COMPONENT_REFDES, 1),
                (COMPONENT_REFDES, 2),
                (COMPONENT_REFDES, 3),
            ),
            padding=1.5,
        )
        job.capture.outputs(
            rgb=True,
            semantic_segmentation=True,
            colorize_semantic_segmentation=False,
            semantic_types=["class"],
        )
    else:
        job.capture.outputs(rgb=True)
    return job


def check_order_output(output: dict) -> list[str]:
    """Return the ways one pin_variation order output is unacceptable.

    Args:
        output: The order's returned mapping.

    Returns:
        Problems found, empty when the order behaved.

    """
    problems = []
    if output.get("leads", 0) < 1:
        problems.append("no leads were varied")
    if output.get("breaches"):
        problems.append(f"{output['breaches']} lead(s) breached their plating")
    if output.get("written") != output.get("leads"):
        problems.append("written and varied lead counts disagree")
    for lead in output.get("per_lead", ()):
        if not lead.get("trimmed"):
            problems.append(f"{lead['path']} was not trimmed despite a height band")
        if lead.get("tilt_deg", 0.0) <= 0.0:
            problems.append(f"{lead['path']} did not tilt at all")
    # The whole point: the barrel, not the request, decides.
    if all(
        lead.get("tilt_limited_by") == "request" for lead in output.get("per_lead", ())
    ):
        problems.append(
            "no lead was limited by its barrel; the bands were too timid to "
            "exercise the constraint this operation exists to enforce",
        )
    return problems


def main() -> int:
    """Compile the plan, or execute both passes in Isaac.

    Returns:
        Process exit code.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    # `--output-dir` is what the CI functional-container template passes; the
    # rest of tests/functional/ uses that spelling too. `--output` stays as an
    # alias so existing local invocations keep working.
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output",
        default=str(REPO_ROOT / "out" / "pin_variation_functional"),
        help="capture output directory",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--variant",
        choices=("nominal", "varied", "roi_solder", "roi_bare"),
        default="varied",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="diff the two variants that were already rendered",
    )
    args = parser.parse_args()

    if not COUPON.exists():
        print(f"SKIP: approved board fixture is not shipped: {COUPON.name}")
        print("Provide an approved board asset to enable this functional workflow.")
        return 77

    output_root = Path(args.output)
    if args.compare:
        if args.variant.startswith("roi_"):
            return compare_roi_pair(output_root)
        return compare(output_root)
    roi = args.variant.startswith("roi_")
    solder = args.variant == "roi_solder"
    # Every variant but the control varies the leads: the ROI pair exists to
    # show the SAME varied leads with and without solder, so holding them
    # nominal there would show nothing.
    vary = args.variant != "nominal"
    if not args.execute:
        job = build_job(
            vary=vary,
            output_dir=output_root,
            seed=args.seed,
            roi=roi,
            solder=solder,
        )
        plan = job.compile_orders()
        for order in plan.orders:
            print(f"{order.order_id:<32} {order.order_type}")
        return 0

    from cdf import create_default_isaac_executor
    from tests.functional.runtime import execute_in_isaac

    destination = output_root / args.variant
    destination.mkdir(parents=True, exist_ok=True)
    job = build_job(
        vary=vary,
        output_dir=destination,
        seed=args.seed,
        roi=roi,
        solder=solder,
    )
    report_path = destination / "pin_variation_report.json"

    async def run_variant():
        """Render one variant and decide its verdict inside the session.

        The verdict is written HERE rather than after `execute_in_isaac`
        returns: Isaac's fast shutdown ends the process, so anything printed
        afterwards never reaches the log.
        """
        result = await job.capture.render_async(
            output_dir=destination,
            executor=create_default_isaac_executor(),
        )
        order_output = result.output_for_type("operation.pin_variation.apply")
        if not vary:
            report_path.write_text(
                json.dumps({"problems": [], "order": None}, indent=2),
                encoding="utf-8",
            )
            print("OK: nominal control rendered")
            return result
        problems = (
            ["the varied run produced no pin_variation output"]
            if order_output is None
            else check_order_output(order_output)
        )
        if roi:
            pin_tip_output = result.output_for_type("operation.pin_tip.generate")
            if pin_tip_output is None:
                problems.append("the ROI run produced no pin-tip proxy output")
            else:
                if pin_tip_output.get("generated") != 3:
                    problems.append(
                        "the ROI run did not generate all three pin-tip proxies"
                    )
                if pin_tip_output.get("warnings"):
                    problems.append(
                        "pin-tip proxy generation reported incomplete targeting: "
                        f"{pin_tip_output['warnings']}"
                    )
            if not solder:
                problems.extend(check_proxy_mask(destination))
        report_path.write_text(
            json.dumps(
                {"problems": problems, "order": order_output},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        if order_output is not None:
            print(json.dumps(order_output, indent=2, default=str)[:2000])
        if problems:
            print("FAIL:")
            for problem in problems:
                print(" -", problem)
        else:
            print(f"OK: {order_output['leads']} leads varied, none breached")
        return result

    execute_in_isaac(
        run_variant,
        destination,
        windowed=False,
        resolution=RESOLUTION,
    )
    if not report_path.exists():
        print("FAIL: the Isaac session ended before a verdict was written")
        return 1
    return 1 if json.loads(report_path.read_text(encoding="utf-8"))["problems"] else 0


def check_proxy_mask(output_dir: Path) -> list[str]:
    """Return proxy-label problems found in one completed ROI capture.

    Args:
        output_dir: Capture directory containing the first trigger.

    Returns:
        Problems found, empty when proxy labels reached the semantic mask.

    """
    import numpy as np
    from PIL import Image

    trigger = output_dir / "trigger_0000"
    mask_path = trigger / "semantic_segmentation_0000.png"
    labels_path = trigger / "semantic_segmentation_labels_0000.json"
    if not mask_path.is_file() or not labels_path.is_file():
        return ["the ROI run produced no semantic mask and label sidecar"]

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    identifiers = {
        str(value.get("class", value) if isinstance(value, dict) else value): int(key)
        for key, value in labels.items()
    }
    mask = np.asarray(Image.open(mask_path))
    problems = []
    if "hole" not in identifiers or not np.any(mask == identifiers["hole"]):
        problems.append("the revealed hole proxies produced no semantic pixels")
    if "pin_tip" not in identifiers or not np.any(mask == identifiers["pin_tip"]):
        problems.append("the revealed pin-tip proxies produced no semantic pixels")
    return problems


def compare_roi_pair(output_root: Path) -> int:
    """Check the solder / no-solder ROI pair against what it claims to be.

    The two runs vary the SAME leads from the same seed and differ only in
    whether solder was grown, so the pair is the workflow: one frame as an AOI
    camera sees it, one with the pad, the hole and the lead separable. What is
    asserted is what a consumer would rely on -- the classes each run must
    carry, and that the lead's own cut face is labelled in both, since the
    trim happens whether or not there is solder to cut through.

    Args:
        output_root: Directory holding both ROI variants.

    Returns:
        Process exit code.

    """
    import numpy as np
    from PIL import Image

    problems = []
    shares = {}
    for variant in ("roi_solder", "roi_bare"):
        base = output_root / variant / "trigger_0000"
        mask = base / "semantic_segmentation_0000.png"
        labels = base / "semantic_segmentation_labels_0000.json"
        if not (mask.exists() and labels.exists()):
            print(f"missing {mask} -- run --execute --variant {variant} first")
            return 1
        names = {}
        for key, value in json.loads(labels.read_text(encoding="utf-8")).items():
            name = value.get("class", value) if isinstance(value, dict) else value
            names[str(name)] = int(key)
        array = np.asarray(Image.open(mask))
        shares[variant] = {
            name: float((array == index).mean()) for name, index in names.items()
        }

    solder = shares["roi_solder"]
    bare = shares["roi_bare"]
    if solder.get("solder", 0.0) <= 0.0:
        problems.append("the solder run labelled no solder at all")
    if bare.get("solder", 0.0) > 0.0:
        problems.append("the no-solder run labelled solder, which cannot exist")
    for variant, share in (("roi_solder", solder), ("roi_bare", bare)):
        if share.get("pin", 0.0) <= 0.0:
            problems.append(
                f"{variant} labelled no pin: the lead is trimmed in both runs, so its "
                "cut face has to be visible in both",
            )
    if bare.get("hole", 0.0) <= 0.0:
        problems.append(
            "the no-solder run labelled no hole, which is the ground truth it "
            "exists to provide",
        )

    for variant, share in sorted(shares.items()):
        visible = {
            key: round(100.0 * value, 3)
            for key, value in sorted(share.items())
            if value > 0.0
        }
        print(f"{variant:<11} {visible}")
    if problems:
        print("FAIL:")
        for problem in problems:
            print(" -", problem)
        return 1
    print(
        "OK: the pair carries solder only where solder was grown, and the "
        "lead's cut face in both"
    )
    return 0


def compare(output_root: Path) -> int:
    """Diff the control against the varied render. No Isaac required.

    Args:
        output_root: Directory holding both variants.

    Returns:
        Process exit code.

    """
    import numpy as np
    from PIL import Image

    frames = {}
    for variant in ("nominal", "varied"):
        path = output_root / variant / "trigger_0000" / "rgb_0000.png"
        if not path.exists():
            print(f"missing {path} -- run --execute --variant {variant} first")
            return 1
        frames[variant] = np.asarray(Image.open(path).convert("RGB"), float)

    for variant, frame in frames.items():
        if frame.mean() < 1.0:
            print(
                f"FAIL: the {variant} frame is black. Two render passes in one Isaac "
                "session return an empty second frame -- run each variant in "
                "its own process.",
            )
            return 1

    difference = np.abs(frames["nominal"] - frames["varied"]).mean(2)
    changed = int((difference > 4).sum())
    fraction = 100.0 * changed / difference.size
    print(
        f"changed pixels: {changed} ({fraction:.3f}%), "
        f"max difference {difference.max():.1f} levels",
    )
    if changed == 0:
        print("FAIL: the varied geometry did not reach the render")
        return 1
    print("OK: the variation is visible in the render")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
