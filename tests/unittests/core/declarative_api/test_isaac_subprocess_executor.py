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

"""Verify ordinary-Python transport to the Isaac JobOrder executor."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cdf import CDFError, JobOrder, JobOrderPlan, JobResult, ValidationError
from cdf.execution.isaac import IsaacSubprocessExecutor, resolve_isaac_python
from cdf.execution.isaac.subprocess_entrypoint import main


class IsaacSubprocessExecutorTest(unittest.TestCase):
    """Verify plan and result transport without launching Isaac Sim."""

    def test_execute_transports_plan_and_result_without_legacy_pipeline(self):
        plan = JobOrderPlan(
            orders=(
                JobOrder(
                    id="capture",
                    type="capture.render",
                    parameters={"output_dir": "output"},
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "python.exe"
            launcher.touch()

            def run(command, **kwargs):
                self.assertIn("cdf.execution.isaac.subprocess_entrypoint", command)
                self.assertNotIn("sdg_pipeline.py", " ".join(command))
                plan_path = Path(command[command.index("--plan") + 1])
                transported = json.loads(plan_path.read_text(encoding="utf-8"))
                self.assertEqual(transported["orders"][0]["type"], "capture.render")
                result_path = Path(command[command.index("--result") + 1])
                result_path.write_text(
                    json.dumps(
                        {
                            "success": True,
                            "result": {
                                "success": True,
                                "order_results": [
                                    {
                                        "order_id": "capture",
                                        "order_type": "capture.render",
                                        "status": "succeeded",
                                        "output": {"output_dir": "output"},
                                        "error": "",
                                    },
                                ],
                            },
                        },
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch(
                "cdf.execution.isaac.subprocess_executor.subprocess.run",
                side_effect=run,
            ) as launch:
                result = IsaacSubprocessExecutor(
                    resolution=(320, 180),
                    windowed=True,
                    isaac_path=launcher,
                ).execute(plan)

        self.assertIsInstance(result, JobResult)
        self.assertTrue(result.success)
        self.assertEqual(result.capture_output, {"output_dir": "output"})
        self.assertIn("--windowed", launch.call_args.args[0])

    def test_execute_reports_child_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "python.exe"
            launcher.touch()

            def run(command, **kwargs):
                result_path = Path(command[command.index("--result") + 1])
                result_path.write_text(
                    json.dumps(
                        {
                            "success": False,
                            "error_type": "RuntimeError",
                            "error": "capture failed",
                        },
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 1, "", "")

            with mock.patch(
                "cdf.execution.isaac.subprocess_executor.subprocess.run",
                side_effect=run,
            ):
                executor = IsaacSubprocessExecutor(
                    resolution=(1, 1),
                    isaac_path=launcher,
                )
                with self.assertRaisesRegex(CDFError, "capture failed"):
                    executor.execute(JobOrderPlan())

    def test_resolve_isaac_python_accepts_directory_and_sim_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / ("python.bat" if sys.platform == "win32" else "python.sh")
            python.touch()
            sim = root / (
                "isaac-sim.bat" if sys.platform == "win32" else "isaac-sim.sh"
            )
            sim.touch()

            self.assertEqual(resolve_isaac_python(root), python.resolve())
            self.assertEqual(resolve_isaac_python(sim), python.resolve())

    def test_resolve_isaac_python_ignores_surrounding_whitespace(self):
        r"""`set ISAAC_SIM_PATH=C:\isaac-sim-6.0 && ...` keeps the space.

        A trailing blank in a batch assignment is invisible and the failure it
        produces names a path with a space wedged into the middle of it:
        `C:\isaac-sim-6.0 \python.bat`. Measured on Windows -- it cost a full
        Isaac boot to diagnose.
        """
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            launcher = Path(temp) / (
                "python.bat" if os.name == "nt" else "python.sh"
            )
            launcher.write_text("", encoding="utf-8")

            self.assertEqual(
                resolve_isaac_python("  %s  " % temp),
                launcher.resolve(),
            )

    def test_resolve_isaac_python_requires_configuration(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValidationError, "ISAAC_SIM_PATH"):
                resolve_isaac_python()


class IsaacSubprocessEntrypointTest(unittest.TestCase):
    """Verify child deserialization and result serialization."""

    def test_main_executes_transported_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            result_path = root / "result.json"
            plan_path.write_text('{"orders":[]}', encoding="utf-8")
            expected = JobResult(success=True)
            arguments = [
                "subprocess_entrypoint",
                "--plan",
                str(plan_path),
                "--result",
                str(result_path),
                "--resolution",
                "64",
                "32",
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch(
                    "cdf.execution.isaac.app_runner.run_plan_in_isaac",
                    return_value=expected,
                ) as execute,
            ):
                return_code = main()

            envelope = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(return_code, 0)
        self.assertTrue(envelope["success"])
        execute.assert_called_once()
        self.assertEqual(execute.call_args.kwargs["resolution"], (64, 32))
        self.assertTrue(callable(execute.call_args.kwargs["on_complete"]))


if __name__ == "__main__":
    unittest.main()
