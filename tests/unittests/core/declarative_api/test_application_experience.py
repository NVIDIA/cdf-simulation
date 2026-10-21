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

"""Choosing which Kit experience a render boots.

A headless defect render enables **304** extensions and 86% of its 42 s boot is
their startup. `apps/pcb_aoi.sdg.kit` drops the 29 that a board cannot use --
no window, no robot, no articulation -- and produces a bit-identical RGB frame
with identical semantic segmentation and identical 2D and 3D bounding boxes.

The saving is small and the trap is not: excluding `omni.kit.test`, which costs
0.54 s, makes the boot **238 s** because every extension shipping OmniGraph
nodes imports it from `ogn/tests` at registration time and each failure logs a
traceback. So this option exists, defaults to off, and the file that goes with
it is the one that was measured.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cdf.execution.isaac.app_runner import _application_settings  # noqa: E402


class ApplicationExperienceTest(unittest.TestCase):
    """What `_application_settings` hands to `SimulationApp`."""

    def test_no_experience_by_default(self):
        """Omitting the key lets Isaac pick its own, which is the shipped path."""
        self.assertNotIn("experience", _application_settings((8, 8), False))

    def test_an_experience_is_passed_through(self):
        settings = _application_settings((8, 8), False, experience="/apps/x.kit")

        self.assertEqual(settings["experience"], "/apps/x.kit")

    def test_an_empty_experience_is_treated_as_unset(self):
        """An unset environment variable arrives as "", not as None."""
        self.assertNotIn(
            "experience", _application_settings((8, 8), False, experience=""),
        )

    def test_a_path_object_is_accepted_and_stringified(self):
        settings = _application_settings(
            (8, 8), False, experience=Path("/apps/x.kit"),
        )

        self.assertIsInstance(settings["experience"], str)
        self.assertTrue(settings["experience"].endswith("x.kit"))

    def test_the_rest_of_the_settings_are_unchanged_by_it(self):
        """Adding a key must not move the ones the renderer depends on."""
        without = _application_settings((320, 180), False)
        with_experience = _application_settings(
            (320, 180), False, experience="/apps/x.kit",
        )
        with_experience.pop("experience")

        self.assertEqual(with_experience, without)

    def test_windowed_still_controls_headless_and_ui(self):
        windowed = _application_settings((8, 8), True, experience="/apps/x.kit")

        self.assertFalse(windowed["headless"])
        self.assertFalse(windowed["hide_ui"])


class SubprocessTransportTest(unittest.TestCase):
    """The option has to survive the process boundary to mean anything."""

    def _command(self, **kwargs):
        """Return the argv a child would be launched with."""
        from unittest import mock

        from cdf.execution.isaac.subprocess_executor import (
            IsaacSubprocessExecutor,
        )

        executor = IsaacSubprocessExecutor(
            resolution=(8, 8), isaac_path="/isaac/python.sh", **kwargs,
        )
        captured = {}

        class _Captured(Exception):
            """Raised once the argv exists, which is all this test wants."""

        def _run(command, *args, **rest):
            captured["command"] = list(command)
            raise _Captured

        with mock.patch("subprocess.run", side_effect=_run),                 mock.patch(
                    "cdf.execution.isaac.subprocess_executor"
                    ".resolve_isaac_python",
                    return_value="/isaac/python.sh",
                ):
            # Catch ONLY the intentional stop. Swallowing everything would let
            # `test_the_flag_is_absent_by_default` pass on an empty list after
            # a failure before subprocess.run -- a test that cannot fail.
            with self.assertRaises(_Captured):
                executor._run_child("--plan", "{}")
        self.assertIn("command", captured, "the child argv was never built")
        return captured["command"]

    def test_the_flag_is_absent_by_default(self):
        self.assertNotIn("--experience", self._command())

    def test_the_flag_is_passed_when_set(self):
        command = self._command(experience="/apps/pcb_aoi.sdg.kit")

        self.assertIn("--experience", command)
        self.assertIn("/apps/pcb_aoi.sdg.kit", command)

    def test_the_child_parses_the_flag(self):
        """A flag the sender writes and the receiver ignores is not plumbing."""
        import argparse
        import inspect

        from cdf.execution.isaac import subprocess_entrypoint

        source = inspect.getsource(subprocess_entrypoint)
        self.assertIn('"--experience"', source)
        self.assertIn("experience=arguments.experience", source)
        del argparse


class ShippedExperienceFileTest(unittest.TestCase):
    """The experience file this repo ships is the one that was measured."""

    def setUp(self):
        self.text = (_REPO_ROOT / "apps" / "pcb_aoi.sdg.kit").read_text()

    def test_it_exists_and_declares_the_isaac_base(self):
        self.assertIn('"isaacsim.exp.base" = {}', self.text)

    def test_it_excludes_extensions(self):
        self.assertIn("excluded = [", self.text)

    def test_it_does_not_exclude_omni_kit_test(self):
        """0.54 s to start; excluding it costs 196 s in failed OGN imports."""
        self.assertNotIn('"omni.kit.test"', self.text)

    def test_it_does_not_exclude_the_ones_that_others_import(self):
        """Each of these left failed imports in the boot log when excluded."""
        for extension in ('"omni.kit.window.filepicker"',
                          '"isaacsim.sensors.experimental.rtx"',
                          '"omni.kit.property.usd"',
                          '"omni.kit.window.property"'):
            self.assertNotIn(extension, self.text)

    def test_it_does_not_anchor_extension_folders_on_the_app_token(self):
        """${app} is the experience file's OWN directory.

        The shipped experiences use it because they live in `isaac-sim/apps`.
        This one lives in a git repository, so ${app} would point at the repo,
        Kit would find no extensions at all, and the child would die with
        `SystemExit: 1` and write nothing -- which is exactly what it did
        before this was anchored on ${exe-path} instead.
        """
        section = self.text.split("[settings.app.exts.folders]")[1]
        # The comment above the list explains ${app}; only the list itself is
        # the contract, so read the quoted entries and not the prose.
        entries = [line.strip() for line in section.splitlines()
                   if line.strip().startswith('"')]

        self.assertTrue(entries, "the folder list is empty")
        for entry in entries:
            self.assertNotIn("${app}", entry)
            self.assertIn("${exe-path}", entry)

    def test_it_keeps_replicator_and_the_core_prims(self):
        """Excluding either would remove the render, not just the overhead."""
        for extension in ('"omni.replicator.core"', '"isaacsim.core.prims"'):
            self.assertNotIn(extension, self.text)


if __name__ == "__main__":
    unittest.main()


class SingleJobExecuteTest(unittest.TestCase):
    """The lever has to reach the path it is FOR.

    The experience file is the right lever only once batching and a live
    session do not apply -- a cold, one-shot render in a
    fresh process. That is `job.execute()`, and it was the one path that could
    not set it.
    """

    def test_execute_accepts_an_experience(self):
        import inspect

        from cdf.job import CDFJob

        signature = inspect.signature(CDFJob.execute)

        self.assertIn("experience", signature.parameters)
        self.assertIsNone(signature.parameters["experience"].default)

    def test_execute_hands_it_to_the_executor(self):
        from unittest import mock

        import pcb_aoi  # noqa: F401 -- registers the domain
        from cdf.job import CDFJob

        job = CDFJob(domain="pcb-aoi", random_seed=1)
        job.scene.configure(usd_path="board.usd", root_path="/Board")

        captured = {}

        class _Executor:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def execute(self, _plan):
                return mock.Mock(success=True)

        with mock.patch(
            "cdf.execution.isaac.IsaacSubprocessExecutor", _Executor,
        ):
            job.execute(
                serialization_path="/tmp/out.usda",
                experience="/apps/pcb_aoi.sdg.kit",
            )

        self.assertEqual(captured.get("experience"), "/apps/pcb_aoi.sdg.kit")

    def test_omitting_it_passes_none_rather_than_a_guess(self):
        from unittest import mock

        import pcb_aoi  # noqa: F401
        from cdf.job import CDFJob

        job = CDFJob(domain="pcb-aoi", random_seed=1)
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        captured = {}

        class _Executor:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def execute(self, _plan):
                return mock.Mock(success=True)

        with mock.patch(
            "cdf.execution.isaac.IsaacSubprocessExecutor", _Executor,
        ):
            job.execute(serialization_path="/tmp/out.usda")

        self.assertIsNone(captured.get("experience"))

    def test_capture_render_forwards_it(self):
        """`capture.render` is the entry point the docs use; it must carry it."""
        from unittest import mock

        import pcb_aoi  # noqa: F401
        from cdf.job import CDFJob

        job = CDFJob(domain="pcb-aoi", random_seed=1)
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        captured = {}

        class _Executor:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def execute(self, _plan):
                return mock.Mock(success=True)

        with mock.patch(
            "cdf.execution.isaac.IsaacSubprocessExecutor", _Executor,
        ):
            job.capture.render(
                serialization_path="/tmp/out.usda",
                experience="/apps/pcb_aoi.sdg.kit",
            )

        self.assertEqual(captured.get("experience"), "/apps/pcb_aoi.sdg.kit")
