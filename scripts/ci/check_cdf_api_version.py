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

"""Require a CDF API version bump when its guarded surface changes."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

VERSION_PATH = Path("cdf/VERSION.md")
CHANGELOG_PATH = Path("cdf/API_CHANGELOG.md")
VERSION_PATTERN = re.compile(
    r"^Current API version: `([0-9]+\.[0-9]+\.[0-9]+)`$",
    flags=re.MULTILINE,
)
STATUS_PATTERN = re.compile(
    r"^Current API status: `(unreleased|released)`$",
    flags=re.MULTILINE,
)
GUARDED_FILES = frozenset(
    {
        "cdf/__init__.py",
        "cdf/domain.py",
        "cdf/errors.py",
        "cdf/job.py",
        "cdf/results.py",
        "cdf/sections.py",
        "cdf/execution/job_order_compiler.py",
        "cdf/execution/job_order_types.py",
        "cdf/execution/job_result_types.py",
        "cdf/execution/isaac/job_order_parameters.py",
        "pcb_aoi/domain.py",
        "pcb_aoi/sections.py",
    },
)


class VersionCheckError(RuntimeError):
    """Report an invalid API change/version combination."""


def parse_version(contents: str, *, source: str) -> tuple[int, int, int]:
    """Extract a semantic API version from VERSION.md contents."""
    match = VERSION_PATTERN.search(contents)
    if match is None:
        raise VersionCheckError(f"{source} has no valid Current API version line")
    return tuple(int(part) for part in match.group(1).split("."))


def format_version(version: tuple[int, int, int]) -> str:
    """Format a parsed semantic version."""
    return ".".join(str(part) for part in version)


def parse_status(contents: str) -> str | None:
    """Return the explicit release status, if the contract declares one."""
    match = STATUS_PATTERN.search(contents)
    return match.group(1) if match is not None else None


def is_guarded_api_path(path: str) -> bool:
    """Return whether a changed path belongs to the versioned API contract."""
    normalized = path.replace("\\", "/")
    return normalized in GUARDED_FILES or (
        normalized.startswith("cdf/config/") and normalized.endswith(".py")
    )


def _git(repo: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise VersionCheckError(result.stderr.strip() or "git command failed")
    return result.stdout


def _base_file(repo: Path, base_ref: str, path: Path) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{path.as_posix()}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def check_api_version(repo: Path, base_ref: str) -> list[str]:
    """Validate version policy and return the guarded paths that changed."""
    changed = _git(repo, "diff", "--name-only", "--diff-filter=ACMRD", base_ref, "HEAD")
    guarded = sorted(path for path in changed.splitlines() if is_guarded_api_path(path))
    if not guarded:
        return []

    current_contents = (repo / VERSION_PATH).read_text(encoding="utf-8")
    current = parse_version(current_contents, source=str(VERSION_PATH))
    current_status = parse_status(current_contents)
    base_contents = _base_file(repo, base_ref, VERSION_PATH)
    if base_contents is not None:
        base = parse_version(base_contents, source=f"{base_ref}:{VERSION_PATH}")
        base_status = parse_status(base_contents)
        if current == base and base_status == "released":
            raise VersionCheckError(
                "CDF API version cannot be reused after publication: "
                f"base={format_version(base)}, current={format_version(current)}"
            )
        if current < base or (current == base and current_status != "unreleased"):
            raise VersionCheckError(
                "CDF API files changed but the API version was not increased: "
                f"base={format_version(base)}, current={format_version(current)}"
            )

    changelog = (repo / CHANGELOG_PATH).read_text(encoding="utf-8")
    version_text = format_version(current)
    if (
        re.search(rf"^## {re.escape(version_text)}(?:\s|$)", changelog, re.MULTILINE)
        is None
    ):
        raise VersionCheckError(
            f"{CHANGELOG_PATH} has no entry for API version {version_text}"
        )
    return guarded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", help="Git commit/ref to compare against")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    """Run the API-version policy check from command-line arguments."""
    args = _parse_args()
    base_ref = args.base_ref or os.environ.get("CI_MERGE_REQUEST_DIFF_BASE_SHA")
    if not base_ref:
        print(
            "error: provide --base-ref or CI_MERGE_REQUEST_DIFF_BASE_SHA",
            file=sys.stderr,
        )
        return 2
    try:
        guarded = check_api_version(args.repo.resolve(), base_ref)
    except (OSError, VersionCheckError) as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 1
    if guarded:
        print("CDF API version policy satisfied for:")
        for path in guarded:
            print(f"- {path}")
    else:
        print("No versioned CDF API files changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
