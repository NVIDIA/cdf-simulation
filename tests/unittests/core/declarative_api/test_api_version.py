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

"""Tests for the CDF API version policy checker."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.ci.check_cdf_api_version import (
    VersionCheckError,
    check_api_version,
    is_guarded_api_path,
)


@unittest.skipUnless(shutil.which("git"), "requires git")
class CdfApiVersionCheckTest(unittest.TestCase):
    """Verify published and unreleased API-version policy behavior."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary_directory.name)
        self._git("init")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "user.name", "Test")
        self._write_contract("1.0.0")
        self._git("add", ".")
        self._git("commit", "-m", "baseline")
        self.base = self._git("rev-parse", "HEAD").strip()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _git(self, *arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def _write_contract(self, version: str, *, status: str | None = None) -> None:
        (self.repo / "cdf").mkdir(exist_ok=True)
        status_line = f"Current API status: `{status}`\n" if status else ""
        (self.repo / "cdf/VERSION.md").write_text(
            f"# CDF API version\n\nCurrent API version: `{version}`\n{status_line}",
            encoding="utf-8",
        )
        (self.repo / "cdf/API_CHANGELOG.md").write_text(
            f"# API changelog\n\n## {version} — test\n",
            encoding="utf-8",
        )
        (self.repo / "cdf/sections.py").write_text("API = 1\n", encoding="utf-8")

    def _commit_changes(self) -> None:
        self._git("add", ".")
        self._git("commit", "-m", "candidate")

    def _released_base(self, source: str) -> str:
        """Publish a guarded source fixture without changing the API version."""
        self._write_contract("1.0.0", status="released")
        (self.repo / "cdf/sections.py").write_text(source, encoding="utf-8")
        self._commit_changes()
        return self._git("rev-parse", "HEAD").strip()

    def _commit_source(self, source: str) -> None:
        (self.repo / "cdf/sections.py").write_text(source, encoding="utf-8")
        self._commit_changes()

    def test_comment_and_formatting_changes_do_not_require_bump(self) -> None:
        base = self._released_base("API = 1\n")
        self._commit_source("# Public API constant\n\nAPI=1  # unchanged\n")

        self.assertEqual(check_api_version(self.repo, base), [])

    def test_docstring_edits_do_not_require_bump(self) -> None:
        source = (
            '"""Old module documentation."""\n'
            'class Section:\n'
            '    """Old class documentation."""\n'
            '    def run(self):\n'
            '        """Old method documentation."""\n'
            '        return 1\n'
            '    async def async_run(self):\n'
            '        """Old async method documentation."""\n'
            '        return 1\n'
        )
        base = self._released_base(source)
        self._commit_source(source.replace("Old", "Updated"))

        self.assertEqual(check_api_version(self.repo, base), [])

    def test_adding_and_removing_docstrings_do_not_require_bump(self) -> None:
        plain = "def run():\n    return 1\n"
        documented = 'def run():\n    """Explain the existing behavior."""\n    return 1\n'
        base = self._released_base(plain)
        self._commit_source(documented)
        self.assertEqual(check_api_version(self.repo, base), [])

        documented_base = self._git("rev-parse", "HEAD").strip()
        self._commit_source(plain)
        self.assertEqual(check_api_version(self.repo, documented_base), [])

    def test_docstrings_do_not_hide_behavior_changes(self) -> None:
        base = self._released_base('"""Old docs."""\nAPI = 1\n')
        self._commit_source('"""Updated docs."""\nAPI = 2\n')

        with self.assertRaisesRegex(VersionCheckError, "cannot be reused"):
            check_api_version(self.repo, base)

    def test_function_signature_changes_still_require_bump(self) -> None:
        base = self._released_base("def run():\n    return 1\n")
        self._commit_source("def run(value=1):\n    return value\n")

        with self.assertRaisesRegex(VersionCheckError, "cannot be reused"):
            check_api_version(self.repo, base)

    def test_annotation_changes_still_require_bump(self) -> None:
        base = self._released_base("def run(value: int):\n    return value\n")
        self._commit_source("def run(value: str):\n    return value\n")

        with self.assertRaisesRegex(VersionCheckError, "cannot be reused"):
            check_api_version(self.repo, base)

    def test_string_values_are_not_treated_as_docstrings(self) -> None:
        base = self._released_base('DEFAULT = "old"\n')
        self._commit_source('DEFAULT = "updated"\n')

        with self.assertRaisesRegex(VersionCheckError, "cannot be reused"):
            check_api_version(self.repo, base)

    def test_non_docstring_expressions_still_require_bump(self) -> None:
        base = self._released_base('API = 1\n"old expression"\n')
        self._commit_source('API = 1\n"updated expression"\n')

        with self.assertRaisesRegex(VersionCheckError, "cannot be reused"):
            check_api_version(self.repo, base)

    def test_added_guarded_file_still_requires_bump(self) -> None:
        base = self._released_base("API = 1\n")
        (self.repo / "cdf/job.py").write_text("JOB = 1\n", encoding="utf-8")
        self._commit_changes()

        with self.assertRaisesRegex(VersionCheckError, "cannot be reused"):
            check_api_version(self.repo, base)

    def test_deleted_guarded_file_still_requires_bump(self) -> None:
        base = self._released_base("API = 1\n")
        (self.repo / "cdf/sections.py").unlink()
        self._commit_changes()

        with self.assertRaisesRegex(VersionCheckError, "cannot be reused"):
            check_api_version(self.repo, base)

    def test_invalid_python_does_not_bypass_version_policy(self) -> None:
        base = self._released_base("API = 1\n")
        self._commit_source("def invalid syntax\n")

        with self.assertRaisesRegex(VersionCheckError, "cannot be reused"):
            check_api_version(self.repo, base)

    def test_api_change_requires_version_increase(self) -> None:
        (self.repo / "cdf/sections.py").write_text("API = 2\n", encoding="utf-8")
        self._commit_changes()

        with self.assertRaisesRegex(VersionCheckError, "was not increased"):
            check_api_version(self.repo, self.base)

    def test_api_change_with_version_and_changelog_passes(self) -> None:
        self._write_contract("1.1.0")
        (self.repo / "cdf/sections.py").write_text("API = 2\n", encoding="utf-8")
        self._commit_changes()

        self.assertEqual(check_api_version(self.repo, self.base), ["cdf/sections.py"])

    def test_unreleased_api_change_can_keep_initial_version(self) -> None:
        self._write_contract("1.0.0", status="unreleased")
        (self.repo / "cdf/sections.py").write_text("API = 2\n", encoding="utf-8")
        self._commit_changes()

        self.assertEqual(check_api_version(self.repo, self.base), ["cdf/sections.py"])

    def test_released_api_version_cannot_be_reused(self) -> None:
        self._write_contract("1.0.0", status="released")
        self._git("add", ".")
        self._git("commit", "-m", "publish")
        released_base = self._git("rev-parse", "HEAD").strip()
        self._write_contract("1.0.0", status="unreleased")
        (self.repo / "cdf/sections.py").write_text("API = 2\n", encoding="utf-8")
        self._commit_changes()

        with self.assertRaisesRegex(VersionCheckError, "cannot be reused"):
            check_api_version(self.repo, released_base)

    def test_non_api_change_does_not_require_bump(self) -> None:
        (self.repo / "README.md").write_text("docs\n", encoding="utf-8")
        self._commit_changes()

        self.assertEqual(check_api_version(self.repo, self.base), [])

    def test_api_bump_requires_matching_changelog_entry(self) -> None:
        self._write_contract("1.1.0")
        (self.repo / "cdf/API_CHANGELOG.md").write_text(
            "# API changelog\n\n## 1.0.0 — baseline\n",
            encoding="utf-8",
        )
        (self.repo / "cdf/sections.py").write_text("API = 2\n", encoding="utf-8")
        self._commit_changes()

        with self.assertRaisesRegex(
            VersionCheckError, "no entry for API version 1.1.0"
        ):
            check_api_version(self.repo, self.base)

    def test_config_python_is_guarded_but_docs_are_not(self) -> None:
        self.assertTrue(is_guarded_api_path("cdf/config/core.py"))
        self.assertFalse(is_guarded_api_path("cdf/config/README.md"))


if __name__ == "__main__":
    unittest.main()
