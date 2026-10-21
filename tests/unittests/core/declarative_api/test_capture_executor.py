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

"""Kit-free tests for the public executor-backed capture API."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi CDF domain
from cdf import (
    IsaacExecutionContext,
    IsaacInProcessExecutor,
    JobExecutor,
    JobResult,
    CDFJob,
    ValidationError,
    create_default_isaac_executor,
)


def _build_job() -> CDFJob:
    """Build the smallest complete plan accepted by the compiler."""
    job = CDFJob(domain="pcb-aoi", random_seed=7)
    job.scene.configure(usd_path="board.usd", root_path="/Board")
    job.capture.renderer(
        mode="pathtracing",
        resolution=(320, 180),
        spp=1,
        total_spp=2,
        max_bounces=2,
    )
    job.capture.camera(rotation_range={"x": (0, 0), "y": (0, 0), "z": -90})
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.limits(triggers=1, max_images=1)
    job.capture.outputs(rgb=True)
    return job


class _RecordingExecutor(JobExecutor):
    """Record plans received through the default asynchronous adapter."""

    def __init__(self, result: JobResult) -> None:
        self.result = result
        self.plans = []

    def execute(self, plan):
        """Record and return a controlled result."""
        self.plans.append(plan)
        return self.result


class CaptureExecutorTest(unittest.IsolatedAsyncioTestCase):
    """Verify compilation and delegation through the public capture method."""

    async def test_render_async_compiles_once_and_returns_executor_result(self):
        job = _build_job()
        expected = JobResult(success=True)
        executor = _RecordingExecutor(expected)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "artifacts"
            with mock.patch.object(
                job,
                "compile_orders",
                wraps=job.compile_orders,
            ) as compile_orders:
                result = await job.capture.render_async(
                    output_dir=output_dir,
                    executor=executor,
                )

            compile_orders.assert_called_once_with(
                output_dir=output_dir,
                serialization_path=None,
            )
            self.assertIs(result, expected)
            self.assertEqual(len(executor.plans), 1)
            self.assertEqual(executor.plans[0].ordered(), executor.plans[0].orders)
            self.assertFalse(output_dir.exists())

    async def test_render_async_rejects_non_executor_and_invalid_result(self):
        job = _build_job()
        with self.assertRaisesRegex(ValidationError, "JobExecutor"):
            await job.capture.render_async(output_dir="output", executor=object())

        executor = _RecordingExecutor(None)
        with self.assertRaisesRegex(ValidationError, "return JobResult"):
            await job.capture.render_async(output_dir="output", executor=executor)

    def test_render_compiles_and_delegates_to_subprocess_executor(self):
        job = _build_job()
        expected = object()
        executor = mock.Mock()
        executor.execute.return_value = expected
        with (
            mock.patch.object(
                job,
                "compile_orders",
                wraps=job.compile_orders,
            ) as compile_orders,
            mock.patch(
                "cdf.execution.isaac.IsaacSubprocessExecutor",
                return_value=executor,
            ) as executor_type,
        ):
            result = job.capture.render(output_dir="output", windowed=True)

        self.assertIs(result, expected)
        compile_orders.assert_called_once_with(
            output_dir="output",
            serialization_path=None,
        )
        executor_type.assert_called_once_with(
            resolution=(320, 180),
            windowed=True,
            experience=None,
        )
        executor.execute.assert_called_once()
        transported_plan = executor.execute.call_args.args[0]
        self.assertEqual(
            transported_plan.orders[-1].parameters["output_dir"],
            "output",
        )

    def test_render_rejects_legacy_execution_options(self):
        job = _build_job()

        with self.assertRaises(TypeError):
            job.capture.render(output_dir="output", dry_run=True)

    def test_execute_runs_serialized_solder_without_capture(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.operations.solder_joints(
            solver="procedural",
            source="usd",
            serialization_path="solder_layer.usda",
        )
        expected = JobResult(success=True)
        executor = mock.Mock()
        executor.execute.return_value = expected

        with mock.patch(
            "cdf.execution.isaac.IsaacSubprocessExecutor",
            return_value=executor,
        ) as executor_type:
            result = job.execute(windowed=False)

        self.assertIs(result, expected)
        executor_type.assert_called_once_with(
            resolution=(64, 64),
            windowed=False,
            experience=None,
        )
        plan = executor.execute.call_args.args[0]
        self.assertEqual(plan.orders[-1].order_id, "solder_joints.attach")
        self.assertEqual(
            plan.orders[-1].parameters["serialization"],
            "solder-joints-only",
        )
        self.assertFalse(
            any(order.order_type == "capture.render" for order in plan.orders),
        )

    def test_execute_requires_output_directory_for_capture(self):
        with self.assertRaisesRegex(ValidationError, "requires output_dir"):
            _build_job().execute()

    def test_render_requires_pass_or_final_stage_serialization(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.operations.solder_joints(
            solver="procedural",
            source="usd",
            serialization_path="solder_layer.usda",
        )

        with self.assertRaisesRegex(
            ValidationError,
            "render pass or serialization_path",
        ):
            job.capture.render()

    def test_render_serializes_final_stage_without_capture(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        expected = JobResult(success=True)
        executor = mock.Mock()
        executor.execute.return_value = expected

        with mock.patch(
            "cdf.execution.isaac.IsaacSubprocessExecutor",
            return_value=executor,
        ) as executor_type:
            result = job.capture.render(serialization_path="final_stage.usda")

        self.assertIs(result, expected)
        executor_type.assert_called_once_with(
            resolution=(64, 64),
            windowed=False,
            experience=None,
        )
        plan = executor.execute.call_args.args[0]
        self.assertEqual(plan.orders[-1].order_id, "stage.serialize")
        self.assertFalse(
            any(order.order_type == "capture.render" for order in plan.orders),
        )

    def test_execute_rejects_output_directory_without_capture(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")

        with self.assertRaisesRegex(ValidationError, "output_dir requires"):
            job.execute(
                output_dir="unused",
                serialization_path="final_stage.usda",
            )

    def test_execute_rejects_non_usd_serialization_path(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")

        with self.assertRaisesRegex(ValidationError, "must end with"):
            job.execute(serialization_path="final_stage.json")

    def test_default_factory_hides_registry_and_context_assembly(self):
        context = IsaacExecutionContext(runtime=object())

        executor = create_default_isaac_executor(context=context)

        self.assertIsInstance(executor, IsaacInProcessExecutor)
        self.assertIs(executor.context, context)
        self.assertIn("capture.render", executor.handlers.handlers)
        self.assertIn("stage.serialize", executor.handlers.handlers)
        self.assertIn("postprocess.crop", executor.handlers.handlers)
        self.assertIn("postprocess.stitch", executor.handlers.handlers)


if __name__ == "__main__":
    unittest.main()
