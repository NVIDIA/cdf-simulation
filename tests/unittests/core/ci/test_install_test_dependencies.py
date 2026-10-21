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

"""Tests for shared CI test-dependency resolution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.ci import install_test_dependencies


class InstallTestDependenciesTest(unittest.TestCase):
    """Verify exact dependency-pin selection for CI lanes."""

    def test_loads_pins_and_normalizes_distribution_names(self) -> None:
        """Resolve differently styled names to the same exact pin."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text(
                "# comment\nopencv-python-headless==5.0.0.93\n",
                encoding="utf-8",
            )

            pins = install_test_dependencies._load_pins(path)

        self.assertEqual(
            ["opencv-python-headless==5.0.0.93"],
            install_test_dependencies._resolve_requested(
                ("opencv_python_headless",),
                pins,
            ),
        )

    def test_adds_requested_extras_to_the_exact_pin(self) -> None:
        """Preserve the repository version while selecting package extras."""
        resolved = install_test_dependencies._resolve_requested(
            ("trimesh[easy]",),
            {"trimesh": "trimesh==4.12.2"},
        )

        self.assertEqual(["trimesh[easy]==4.12.2"], resolved)

    def test_missing_pin_is_an_error(self) -> None:
        """Reject a lane dependency that is absent from the pinned files."""
        with self.assertRaisesRegex(ValueError, "no exact pin found"):
            install_test_dependencies._resolve_requested(("missing",), {})

    def test_rejects_unpinned_requirement_lines(self) -> None:
        """Require every selectable dependency to use an exact version."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text("example>=1.0\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not an exact package pin"):
                install_test_dependencies._load_pins(path)

    def test_main_installs_runtime_and_test_dependencies_together(self) -> None:
        """Invoke pip once with pins from both dependency files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_path = root / "requirements.txt"
            test_path = root / "requirements-test.txt"
            runtime_path.write_text("Pillow==12.3.0\n", encoding="utf-8")
            test_path.write_text(
                "unittest-xml-reporting==4.0.0\n",
                encoding="utf-8",
            )
            completed = mock.Mock(returncode=0)
            with mock.patch(
                "scripts.ci.install_test_dependencies.subprocess.run",
                return_value=completed,
            ) as run:
                result = install_test_dependencies.main(
                    [
                        "--runtime",
                        "Pillow",
                        "--test",
                        "unittest-xml-reporting",
                        "--runtime-file",
                        str(runtime_path),
                        "--test-file",
                        str(test_path),
                    ]
                )

        self.assertEqual(0, result)
        command = run.call_args.args[0]
        self.assertIn("Pillow==12.3.0", command)
        self.assertIn("unittest-xml-reporting==4.0.0", command)
        self.assertTrue(run.call_args.kwargs["check"] is False)


if __name__ == "__main__":
    unittest.main()
