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

"""Install named test-lane dependencies from the repository's pinned files."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_NAME_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)(\[[A-Za-z0-9_,.-]+\])?")


def _canonical_name(name: str) -> str:
    """Return a normalized Python distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _load_pins(path: Path) -> dict[str, str]:
    """Load exact package pins from one requirements file."""
    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _NAME_PATTERN.match(line)
        if match is None or "==" not in line:
            msg = f"{path}:{line_number} is not an exact package pin: {line!r}"
            raise ValueError(msg)
        name = _canonical_name(match.group(1))
        if name in pins:
            msg = f"{path}:{line_number} duplicates package {match.group(1)!r}"
            raise ValueError(msg)
        pins[name] = line
    return pins


def _requested_names(value: str) -> tuple[str, ...]:
    """Split a comma-separated package list."""
    return tuple(name.strip() for name in value.split(",") if name.strip())


def _resolve_requested(names: tuple[str, ...], pins: dict[str, str]) -> list[str]:
    """Resolve requested package names to pinned requirement specifiers."""
    resolved: list[str] = []
    for requested in names:
        match = _NAME_PATTERN.fullmatch(requested)
        if match is None:
            msg = f"invalid requested package name: {requested!r}"
            raise ValueError(msg)
        package_name, extras = match.groups()
        canonical_name = _canonical_name(package_name)
        try:
            pinned = pins[canonical_name]
        except KeyError as exc:
            msg = f"no exact pin found for requested package {package_name!r}"
            raise ValueError(msg) from exc
        if extras:
            pinned = _NAME_PATTERN.sub(f"{package_name}{extras}", pinned, count=1)
        resolved.append(pinned)
    return resolved


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        default="",
        help="Comma-separated package names pinned in requirements.txt",
    )
    parser.add_argument(
        "--test",
        default="",
        help="Comma-separated package names pinned in requirements-test.txt",
    )
    parser.add_argument(
        "--runtime-file",
        type=Path,
        default=Path("requirements.txt"),
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("requirements-test.txt"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Resolve and install the exact dependencies requested by one CI lane."""
    args = _parse_args(argv)
    try:
        runtime = _resolve_requested(
            _requested_names(args.runtime),
            _load_pins(args.runtime_file),
        )
        test = _resolve_requested(
            _requested_names(args.test),
            _load_pins(args.test_file),
        )
    except (OSError, ValueError) as exc:
        print(f"Cannot resolve test dependencies: {exc}", file=sys.stderr)
        return 2

    requirements = runtime + test
    if not requirements:
        print("No test dependencies requested.")
        return 0
    print("Installing pinned test dependencies: " + ", ".join(requirements))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            *requirements,
        ],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
