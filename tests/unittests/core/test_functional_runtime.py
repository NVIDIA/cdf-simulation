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

"""Test functional runtime failure reporting and output-quality checks."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cdf import JobExecutionError, JobOrder, JobOrderResult, JobResult
from tests.functional import runtime


class _FailedTask:
    """Completed task test double that raises a configured exception."""

    def __init__(self, exception: BaseException):
        self.exception = exception
        self.cancelled = False

    def done(self) -> bool:
        """Return that the task has completed."""
        return True

    def cancel(self) -> bool:
        """Record cancellation for protocol compatibility."""
        self.cancelled = True
        return True

    def result(self) -> JobResult:
        """Raise the configured task exception."""
        raise self.exception


class _StoppedTask(_FailedTask):
    """Incomplete task test double used when Isaac Sim stops early."""

    def done(self) -> bool:
        """Return that the task has not completed."""
        return False


class _FakeSimulationApp:
    """SimulationApp test double that verifies artifact-before-close order."""

    def __init__(self, output_dir: Path, *, running: bool = True):
        self.output_dir = output_dir
        self.running = running
        self.closed = False
        self.update_count = 0

    def is_running(self) -> bool:
        """Return the configured running state."""
        return self.running

    def update(self) -> None:
        """Record one application update."""
        self.update_count += 1

    def close(self) -> None:
        """Verify the failure artifact exists before shutdown."""
        artifact = self.output_dir / runtime.ERROR_ARTIFACT_NAME
        if not artifact.is_file():
            raise AssertionError("failure artifact was not written before close")
        self.closed = True


class FunctionalRuntimeTest(unittest.TestCase):
    """Validate durable diagnostics for functional-test failures."""

    def setUp(self) -> None:
        """Create an isolated functional output directory."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.output_dir = Path(self.temporary_directory.name)

    def _job_execution_error(self) -> JobExecutionError:
        """Create a structured failed capture order.

        Returns:
            Job execution error with partial successful and failed results.

        """
        order = JobOrder(
            id="capture.render",
            type="capture.render",
            parameters={},
        )
        result = JobResult(
            success=False,
            order_results=(
                JobOrderResult("scene.load", "scene.load", "succeeded"),
                JobOrderResult(
                    "capture.render",
                    "capture.render",
                    "failed",
                    error="missing LdrColorSDhost",
                ),
            ),
        )
        return JobExecutionError(
            order,
            result,
            RuntimeError("missing LdrColorSDhost"),
        )

    def _run_failure(self, exception: BaseException) -> dict[str, object]:
        """Run a failed task and return its retained JSON payload.

        Args:
            exception: Failure raised by the task test double.

        Returns:
            Parsed functional-test failure artifact.

        """
        app = _FakeSimulationApp(self.output_dir)
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(type(exception)),
        ):
            runtime._run_task(app, _FailedTask(exception), self.output_dir)
        self.assertTrue(app.closed)
        artifact = self.output_dir / runtime.ERROR_ARTIFACT_NAME
        return json.loads(artifact.read_text(encoding="utf-8"))

    def test_job_execution_error_records_order_context(self) -> None:
        """Retain the failed order, cause, partial results, and traceback."""
        payload = self._run_failure(self._job_execution_error())

        self.assertFalse(payload["success"])
        self.assertEqual(payload["failed_order"]["id"], "capture.render")
        self.assertEqual(payload["failed_order"]["type"], "capture.render")
        self.assertEqual(payload["cause"]["type"], "RuntimeError")
        self.assertEqual(len(payload["order_results"]), 2)
        self.assertIn("JobExecutionError", "".join(payload["traceback"]))

    def test_unexpected_error_records_generic_context(self) -> None:
        """Retain useful diagnostics for failures outside JobOrder execution."""
        payload = self._run_failure(ValueError("unexpected failure"))

        self.assertEqual(payload["exception"]["type"], "ValueError")
        self.assertEqual(payload["exception"]["message"], "unexpected failure")
        self.assertNotIn("failed_order", payload)

    def test_early_isaac_shutdown_records_and_reraises_error(self) -> None:
        """Turn an incomplete task into a retained runtime failure."""
        app = _FakeSimulationApp(self.output_dir, running=False)
        task = _StoppedTask(AssertionError("result must not be called"))

        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "stopped before"),
        ):
            runtime._run_task(app, task, self.output_dir)

        self.assertTrue(task.cancelled)
        self.assertTrue(app.closed)
        payload = json.loads(
            (self.output_dir / runtime.ERROR_ARTIFACT_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(payload["exception"]["type"], "RuntimeError")


class SemanticCaptureQualityTest(unittest.TestCase):
    """Verify functional semantic artifacts are present and aligned."""

    def _write_capture(
        self,
        root: Path,
        *,
        rgb_size: tuple[int, int] = (8, 6),
        mask_size: tuple[int, int] = (8, 6),
    ) -> None:
        """Write a minimal RGB, semantic mask, and label sidecar fixture."""
        trigger = root / "trigger_0000"
        trigger.mkdir(parents=True)
        Image.new("RGB", rgb_size, (10, 20, 30)).save(trigger / "rgb_0000.png")
        Image.new("L", mask_size, 1).save(
            trigger / "semantic_segmentation_0000.png",
        )
        (trigger / "semantic_segmentation_labels_0000.json").write_text(
            json.dumps({"1": {"defect": "insufficient"}}),
            encoding="utf-8",
        )

    def test_accepts_nonempty_aligned_semantic_artifacts(self) -> None:
        """Accept an aligned mask carrying the expected semantic value."""
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            self._write_capture(output_dir)

            report = runtime._require_semantic_capture(
                output_dir,
                expected_label="insufficient",
            )

        self.assertEqual((8, 6), report["size"])
        self.assertIn("insufficient", report["semantic_values"])

    def test_rejects_a_mask_with_different_dimensions(self) -> None:
        """Reject semantic masks that cannot align pixel-for-pixel with RGB."""
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            self._write_capture(output_dir, mask_size=(4, 3))

            with self.assertRaisesRegex(RuntimeError, "does not match RGB"):
                runtime._require_semantic_capture(output_dir)


if __name__ == "__main__":
    unittest.main()
