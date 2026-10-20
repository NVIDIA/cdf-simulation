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

"""Test Docker entrypoint dispatch without starting Isaac Sim."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
ENTRYPOINT = REPO_ROOT / "scripts/docker/entrypoint.sh"


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires Bash")
class ContainerEntrypointTests(unittest.TestCase):
    """Verify Docker entrypoint mode selection with fake Isaac launchers."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.isaac_root = Path(self.temporary_directory.name)
        (self.isaac_root / "kit").mkdir()
        (self.isaac_root / "apps").mkdir()
        self._write_launcher(self.isaac_root / "python.sh", "PYTHON")
        self._write_launcher(self.isaac_root / "kit/kit", "KIT")
        self.job_config = self.isaac_root / "job.yaml"
        self.job_config.write_text(
            "api_version: 1.0.0\ndomain: pcb-aoi\n",
            encoding="utf-8",
        )
        self.environment = {
            **os.environ,
            "CDF_ISAAC_SIM_ROOT": str(self.isaac_root),
        }
        # A developer machine may export a real one; these tests are about what
        # the entrypoint derives, so start from unset.
        self.environment.pop("ISAAC_SIM_PATH", None)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_launcher(self, path: Path, name: str) -> None:
        """Create a fake launcher that reports its arguments and environment.

        It echoes ``ISAAC_SIM_PATH`` as well as the arguments, because every
        route has to export it -- every route can end up in ``capture.render``.

        Args:
            path: Destination for the launcher.
            name: Label written before the received arguments.

        """
        path.write_text(
            f'#!/bin/bash\nprintf \'{name}\\n\'\n'
            f'printf \'ISAAC_SIM_PATH=%s\\n\' "$ISAAC_SIM_PATH"\n'
            f'printf \'%s\\n\' "$@"\n',
            encoding="utf-8",
        )
        path.chmod(0o755)

    def _run_entrypoint(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run the entrypoint against the fake Isaac installation.

        Args:
            *arguments: Arguments passed to the entrypoint.

        Returns:
            Completed entrypoint process.

        """
        return subprocess.run(
            ["bash", str(ENTRYPOINT), *arguments],
            check=False,
            capture_output=True,
            env=self.environment,
            text=True,
        )

    @staticmethod
    def _launcher(result: subprocess.CompletedProcess[str]) -> str:
        """The launcher label, which is the first line the fake script prints."""
        return result.stdout.splitlines()[0]

    @staticmethod
    def _arguments(result: subprocess.CompletedProcess[str]) -> list[str]:
        """Everything after the label and the reported ISAAC_SIM_PATH."""
        return result.stdout.splitlines()[2:]

    def test_job_script_uses_python_launcher(self) -> None:
        result = self._run_entrypoint(
            "job-script",
            "example_job.py",
            "--execute",
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(self._launcher(result), "PYTHON")
        self.assertEqual(self._arguments(result),
                         ["example_job.py", "--execute"])

    def test_default_uses_existing_kit_launcher(self) -> None:
        command = "legacy.py --config example.yaml"
        result = self._run_entrypoint(command)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(self._launcher(result), "KIT")
        self.assertEqual(self._arguments(result)[-2:], ["--exec", command])

    def test_config_uses_python_module(self) -> None:
        result = self._run_entrypoint(
            "config",
            str(self.job_config),
            "--execute",
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(self._launcher(result), "PYTHON")
        self.assertEqual(
            self._arguments(result),
            ["-m", "cdf.config", str(self.job_config), "--execute"],
        )

    def test_every_route_exports_isaac_sim_path(self) -> None:
        """Regression: retiring the legacy router took this export with it.

        The router set ``ISAAC_SIM_PATH`` on its way past, so the ``config``
        route inherited it by accident and only ``job-script`` exported it for
        itself. Archiving the router left ``config --execute`` failing in CI
        with "capture.render requires ISAAC_SIM_PATH" -- the one thing that
        route needs in order to render at all.
        """
        routes = {
            "job-script": ("job-script", "example_job.py", "--execute"),
            "config": ("config", str(self.job_config), "--execute"),
            "raw command": ("legacy.py --config example.yaml",),
        }
        for label, arguments in routes.items():
            with self.subTest(route=label):
                result = self._run_entrypoint(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    "ISAAC_SIM_PATH=%s" % self.isaac_root,
                    result.stdout.splitlines(),
                )

    def test_an_explicit_isaac_sim_path_is_not_overridden(self) -> None:
        self.environment["ISAAC_SIM_PATH"] = "/somewhere/else"
        result = self._run_entrypoint("config", str(self.job_config))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ISAAC_SIM_PATH=/somewhere/else",
                      result.stdout.splitlines())

    def test_job_script_requires_script_path(self) -> None:
        result = self._run_entrypoint("job-script")

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires a Python script path", result.stderr)

    def test_config_requires_yaml_path(self) -> None:
        result = self._run_entrypoint("config")

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires a YAML path", result.stderr)


if __name__ == "__main__":
    unittest.main()
