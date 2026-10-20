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

"""Protect the dependency direction between CDF and domain plugins."""

import subprocess
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest import mock

from cdf import capabilities, capability_registry


class DomainPluginBoundaryTest(unittest.TestCase):
    """Verify that the CDF package root remains domain-neutral."""

    def test_import_cdf_does_not_import_pcb_aoi(self) -> None:
        """Keep an ordinary CDF import from installing the PCB plugin."""
        program = """
import sys
import cdf

loaded = sorted(
    name
    for name in sys.modules
    if name == "pcb_aoi" or name.startswith("pcb_aoi.")
)
if loaded:
    raise SystemExit("import cdf loaded PCB-AOI modules: " + ", ".join(loaded))
"""
        subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_explicit_domain_discovery_works_from_source_checkout(self) -> None:
        """Load the PCB plugin when its domain is explicitly selected."""
        program = """
from cdf.domain import get_domain

domain = get_domain("pcb-aoi")
if domain.name != "pcb-aoi":
    raise SystemExit(f"loaded unexpected domain: {domain.name}")
"""
        subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_explicit_domain_config_discovery_works_from_source_checkout(self) -> None:
        """Load the PCB config adapter without installed entry-point metadata."""
        program = """
from cdf.config.domain_adapters import get_domain_config

adapter = get_domain_config("pcb-aoi")
if adapter.operations_model.__name__ != "OperationsConfig":
    raise SystemExit(f"loaded unexpected adapter: {adapter!r}")
"""
        subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_capability_cli_loads_explicit_domain_from_source_checkout(self) -> None:
        """Generate the PCB capability matrix without package installation."""
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                "-m",
                "cdf.capabilities",
                "--domain",
                "pcb-aoi",
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        self.assertIn("scene.board", result.stdout)

    def test_capability_cli_filters_explicit_domains(self) -> None:
        """Exclude other registered plugins from an explicit domain matrix."""
        program = """
from contextlib import redirect_stdout
from io import StringIO

from cdf.capabilities import main
from cdf.capability_registry import Capability, register_capabilities

register_capabilities(
    "unrelated",
    (Capability(id="unrelated.capability", title="Unrelated"),),
)
output = StringIO()
with redirect_stdout(output):
    result = main(["--domain", "pcb-aoi"])
if result != 0:
    raise SystemExit(f"capability CLI returned {result}")
matrix = output.getvalue()
if "scene.board" not in matrix:
    raise SystemExit("PCB capability missing from filtered matrix")
if "unrelated.capability" in matrix:
    raise SystemExit("unrelated capability leaked into filtered matrix")
"""
        subprocess.run(
            [sys.executable, "-X", "utf8", "-c", program],
            check=True,
            capture_output=True,
            encoding="utf-8",
        )

    def test_capability_cli_rejects_an_empty_registry(self) -> None:
        """Refuse to replace the matrix with a header-only document."""
        stderr = StringIO()
        with (
            mock.patch.object(capabilities, "get_capabilities", return_value=()),
            redirect_stderr(stderr),
        ):
            result = capabilities.main([])

        self.assertEqual(result, 2)
        self.assertIn("no CDF capabilities are registered", stderr.getvalue())

    def test_capability_cli_reports_unknown_domain_without_traceback(self) -> None:
        """Report explicit discovery failures as ordinary CLI errors."""
        stderr = StringIO()
        with redirect_stderr(stderr):
            result = capabilities.main(["--domain", "not-a-real-domain"])

        self.assertEqual(result, 2)
        self.assertIn("unknown domain 'not-a-real-domain'", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_failed_plugin_discovery_can_retry(self) -> None:
        """Leave discovery incomplete when an entry point fails to load."""
        entry_point = mock.Mock()
        entry_point.load.side_effect = RuntimeError("plugin import failed")
        entry_points = mock.Mock()
        entry_points.select.return_value = (entry_point,)

        with (
            mock.patch.object(
                capability_registry.metadata,
                "entry_points",
                return_value=entry_points,
            ),
            mock.patch.object(capability_registry, "_PLUGINS_LOADING", False),
            mock.patch.object(capability_registry, "_PLUGINS_LOADED", False),
        ):
            with self.assertRaisesRegex(RuntimeError, "plugin import failed"):
                capability_registry.load_installed_domains()
            self.assertFalse(capability_registry._PLUGINS_LOADING)
            self.assertFalse(capability_registry._PLUGINS_LOADED)

            entry_point.load.side_effect = None
            capability_registry.load_installed_domains()

            self.assertFalse(capability_registry._PLUGINS_LOADING)
            self.assertTrue(capability_registry._PLUGINS_LOADED)
            self.assertEqual(entry_point.load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
