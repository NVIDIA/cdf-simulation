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

"""Shared single-component SMT functional-test scenario harness."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pcb_aoi  # noqa: F401 -- registers the `pcb-aoi` CDF domain
from cdf import JobResult, CDFJob, create_default_isaac_executor
from tests.functional.runtime import (
    _require_semantic_capture,
    _skip_missing_fixture,
    execute_in_isaac,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
LOCAL_OUTPUT_ROOT = REPO_ROOT / "sdg_test_output" / "functional"
BOARD_USD = RESOURCE_DIR / "ICT_C0805_C22_coupon.usdc"
COMPONENT_TYPE = "C_0805_2012Metric"
SELECTED_COMPONENT_COUNT = 1

OperationConfigurator = Callable[[CDFJob, float], None]
AppearanceConfigurator = Callable[[CDFJob], None]


def component_count() -> int:
    """Return the maintained coupon's selected component count.

    The approved test coupon is expected to be a single-component crop. Runtime
    execution independently verifies that exactly one component was affected,
    so compile-only mode does not need USD Python bindings merely to count it.

    Returns:
        Number of selected component instances.

    Example:

    .. code-block:: python

        print(component_count())
        # 1

    """
    return SELECTED_COMPONENT_COUNT


def exact_ratio(selected_count: int, eligible_count: int) -> float:
    """Return a ratio whose truncated product selects an exact count.

    Args:
        selected_count: Required integer selection count.
        eligible_count: Available component population.

    Returns:
        Ratio that produces the requested truncated selection count.

    Raises:
        RuntimeError: If the requested count cannot be represented.

    Example:

    .. code-block:: python

        ratio = exact_ratio(1, 4)
        print(int(4 * ratio))
        # 1

    """
    ratio = selected_count / eligible_count
    while int(eligible_count * ratio) < selected_count:
        ratio = math.nextafter(ratio, 1.0)
    if int(eligible_count * ratio) != selected_count:
        raise RuntimeError("could not express the requested count as a ratio")
    return ratio


def _configure_scene(
    job: CDFJob,
    *,
    board_usd: Path = BOARD_USD,
    component_type: str = COMPONENT_TYPE,
) -> None:
    """Configure the common SMT coupon scene and component selection.

    Args:
        job: CDF job receiving the shared scene declarations.
        board_usd: Schema-native coupon to load.
        component_type: Exact embedded-metadata selector.

    """
    job.scene.configure(
        usd_path=board_usd,
        root_path="/Board",
        camera_path="/World/camera_light/Camera",
        camera_xform_path="/World/camera_light",
        ring_light_root="/World/camera_light/aoi_ring_light",
    )
    job.scene.select_components(types=[component_type], source="usd")


def configure_ring_appearance(job: CDFJob) -> None:
    """Configure the default deterministic AOI ring lighting.

    This is the appearance every SMT scenario gets unless it passes its
    own ``configure_appearance``; ranges collapse to single values so a
    functional run is reproducible frame for frame.

    Args:
        job: CDF job receiving the shared appearance declarations.

    """
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


def _configure_capture(
    job: CDFJob,
    *,
    resolution: tuple[int, int],
    total_spp: int,
    defect_label: str,
    output_subdir: str,
) -> None:
    """Configure the common annotated capture and defect crop.

    Args:
        job: CDF job receiving capture and postprocessing declarations.
        resolution: Capture width and height.
        total_spp: Total path-tracing samples per pixel.
        defect_label: Operation label selected by semantic cropping.
        output_subdir: Scenario-specific crop output directory.

    """
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
    job.postprocess.crop(
        method="bbox_tight",
        types=["rgb", "semantic_segmentation", "component_instance"],
        offset=32,
        min_coverage=0.2,
        class_filter=[defect_label],
        xform_depth=7,
        output_subdir=output_subdir,
    )


@dataclass(frozen=True)
class SmtFunctionalScenario:
    """Defect-specific configuration for the shared SMT coupon workflow.

    Args:
        name: Stable scenario name used for output directories.
        description: Command-line description shown by the scenario script.
        defect_label: Semantic defect label used for reporting and cropping.
        operation_type: JobOrder type used to retrieve the defect result.
        configure_operations: Callback that adds scenario-specific operations.
        output_subdir: Scenario-specific crop output directory.
        board_usd: Schema-native coupon loaded by this scenario.
        component_type: Exact embedded-metadata selector.
        configure_appearance: Callback that declares lighting and component
            appearance; ``None`` uses the shared AOI ring rig.

    Example:

    .. code-block:: python

        scenario = SmtFunctionalScenario(
            name="smt_example",
            description="Example SMT scenario",
            defect_label="shift",
            operation_type="operation.shift.apply",
            configure_operations=lambda job, ratio: None,
            output_subdir="shift_closeups",
        )
        print(scenario.name)
        # smt_example

    """

    name: str
    #: Stable scenario name used for output directories.

    description: str
    #: Command-line description shown by the scenario script.

    defect_label: str
    #: Semantic defect label used for reporting and cropping.

    operation_type: str
    #: JobOrder type used to retrieve the defect result.

    configure_operations: OperationConfigurator
    #: Callback that adds scenario-specific operations to the shared job.

    output_subdir: str
    #: Scenario-specific crop output directory.

    board_usd: Path = BOARD_USD
    #: Schema-native coupon loaded by this scenario.

    component_type: str = COMPONENT_TYPE
    #: Exact component selector read from embedded USD metadata.

    configure_appearance: AppearanceConfigurator | None = None
    #: Callback that declares lighting and component appearance; ``None``
    #: uses :func:`configure_ring_appearance`. A plain function stored as a
    #: class-level default would bind as a method on attribute access, so
    #: the fallback is resolved in ``build_job`` instead.

    @property
    def default_output_dir(self) -> Path:
        """Return the scenario's repository-local output directory.

        Returns:
            Repository-local capture and crop output directory.

        Example:

        .. code-block:: python

            print(scenario.default_output_dir.name)
            # smt_example

        """
        return LOCAL_OUTPUT_ROOT / self.name

    def build_job(
        self,
        *,
        resolution: tuple[int, int] = (1920, 1080),
        total_spp: int = 32,
    ) -> tuple[CDFJob, int, float]:
        """Build the configured single-component SMT functional job.

        Args:
            resolution: Capture width and height.
            total_spp: Total path-tracing samples per pixel.

        Returns:
            Authored job, eligible component count, and selection ratio.

        Example:

        .. code-block:: python

            job, eligible_count, ratio = scenario.build_job(
                resolution=(320, 180),
                total_spp=2,
            )
            print(eligible_count, ratio)
            # 1 1.0

        """
        eligible_count = component_count()
        selection_ratio = exact_ratio(SELECTED_COMPONENT_COUNT, eligible_count)
        job = CDFJob(domain="pcb-aoi", random_seed=7)

        _configure_scene(
            job,
            board_usd=self.board_usd,
            component_type=self.component_type,
        )
        (self.configure_appearance or configure_ring_appearance)(job)
        self.configure_operations(job, selection_ratio)
        job.annotations.label_components(source="usd")
        job.annotations.enable_operation_labels(namespace="defect")
        _configure_capture(
            job,
            resolution=resolution,
            total_spp=total_spp,
            defect_label=self.defect_label,
            output_subdir=self.output_subdir,
        )
        return job, eligible_count, selection_ratio

    def parse_args(self, argv: Sequence[str] | None = None) -> argparse.Namespace:
        """Parse the shared functional-test command-line arguments.

        Args:
            argv: Optional argument sequence, or None to read process arguments.

        Returns:
            Parsed functional-test arguments.

        Example:

        .. code-block:: python

            args = scenario.parse_args(["--resolution", "320", "180"])
            print(args.resolution)
            # [320, 180]

        """
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument(
            "--execute",
            action="store_true",
            help="execute in this Isaac Sim process; default is compile-only",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=self.default_output_dir,
            help="capture and crop output directory",
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
        return parser.parse_args(argv)

    def print_plan(
        self,
        job: CDFJob,
        output_dir: Path,
        eligible_count: int,
        selection_ratio: float,
    ) -> None:
        """Compile and print the scenario's JobOrder plan.

        Args:
            job: Authored functional job to compile.
            output_dir: Capture destination embedded in the terminal order.
            eligible_count: Number of eligible coupon components.
            selection_ratio: Ratio expected to select the target component.

        """
        plan = job.compile_orders(output_dir=output_dir)
        print(f"eligible {self.component_type} components: {eligible_count}")
        print(f"{self.defect_label} ratio: {selection_ratio:.12f}")
        print(
            f"expected {self.defect_label} defects: "
            f"{int(eligible_count * selection_ratio)}",
        )
        print(json.dumps(plan.model_dump(mode="json", by_alias=True), indent=2))

    async def execute_job(self, job: CDFJob, output_dir: Path) -> JobResult:
        """Execute the scenario through the public in-process API.

        Args:
            job: Authored functional job to execute.
            output_dir: Capture and crop destination.

        Returns:
            Structured result from the in-process executor.

        Raises:
            RuntimeError: If the structural defect, semantic capture, or crop
                does not produce the expected observable output.

        Example:

        .. code-block:: python

            result = await scenario.execute_job(job, output_dir)
            print(result.success)

        """
        started = time.perf_counter()
        result = await job.capture.render_async(
            output_dir=output_dir,
            executor=create_default_isaac_executor(),
        )
        defect_output = result.output_for_type(self.operation_type)
        component_paths = (
            defect_output.get("component_paths")
            if isinstance(defect_output, Mapping)
            else None
        )
        if (
            not isinstance(defect_output, Mapping)
            or defect_output.get("count") != SELECTED_COMPONENT_COUNT
            or not isinstance(component_paths, list)
            or len(component_paths) != SELECTED_COMPONENT_COUNT
        ):
            raise RuntimeError(
                f"{self.defect_label} did not affect exactly one component: "
                f"{defect_output!r}",
            )
        crop_output = result.crop_output
        if (
            not isinstance(crop_output, Mapping)
            or crop_output.get("executed") is not True
        ):
            raise RuntimeError(
                f"{self.defect_label} crop did not complete: {crop_output!r}",
            )
        semantic_capture = _require_semantic_capture(
            output_dir,
            expected_label=self.defect_label,
        )
        summary = {
            "success": result.success,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "executed_order_count": len(result.order_results),
            "defect": defect_output,
            "solder_generation": result.output_for_type("solder_joints.generate"),
            "solder_attachment": result.output_for_type("solder_joints.attach"),
            "capture": result.capture_output,
            "crop": crop_output,
            "semantic_validation": semantic_capture,
        }
        print(json.dumps(summary, indent=2, default=str), flush=True)
        return result

    def execute_in_isaac(
        self,
        job: CDFJob,
        output_dir: Path,
        *,
        windowed: bool,
        resolution: tuple[int, int],
    ) -> JobResult:
        """Boot Isaac Sim and drive the scenario's asynchronous execution.

        Args:
            job: Authored functional job to execute.
            output_dir: Capture and crop destination.
            windowed: Whether to display the Isaac Sim window.
            resolution: Width and height supplied to the application window.

        Returns:
            Structured result from the in-process executor.

        Raises:
            RuntimeError: If Isaac Sim exits before execution completes.

        Example:

        .. code-block:: python

            result = scenario.execute_in_isaac(
                job,
                output_dir,
                windowed=False,
                resolution=(320, 180),
            )
            print(result.success)

        """
        return execute_in_isaac(
            lambda: self.execute_job(job, output_dir),
            output_dir,
            windowed=windowed,
            resolution=resolution,
        )

    def main(self, argv: Sequence[str] | None = None) -> int:
        """Compile or execute this SMT functional scenario.

        Args:
            argv: Optional argument sequence, or None to read process arguments.

        Returns:
            Process exit status.

        """
        args = self.parse_args(argv)
        skip_code = _skip_missing_fixture(self.board_usd)
        if skip_code is not None:
            return skip_code
        resolution = (int(args.resolution[0]), int(args.resolution[1]))
        output_dir = args.output_dir.expanduser().resolve()
        job, eligible_count, selection_ratio = self.build_job(
            resolution=resolution,
            total_spp=args.total_spp,
        )
        if not args.execute:
            self.print_plan(job, output_dir, eligible_count, selection_ratio)
            return 0

        result = self.execute_in_isaac(
            job,
            output_dir,
            windowed=args.windowed,
            resolution=resolution,
        )
        return 0 if result.success else 1
