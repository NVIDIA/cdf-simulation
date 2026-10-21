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

"""Command-line interface for YAML-authored CDF jobs."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from ..api_version import CDF_API_STATUS, CDF_API_VERSION
from ..errors import CDFError
from ._common import resolve_path
from .core import dump_job_config_schema, load_job

_DISTRIBUTION_NAME = "cdf-simulation"


def _package_version() -> str:
    """Return the installed package version or its source-tree declaration."""
    try:
        return distribution_version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        try:
            with pyproject_path.open("rb") as handle:
                project = tomllib.load(handle)["project"]
            version = project["version"]
        except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError):
            return "unknown"
        return version if isinstance(version, str) else "unknown"


def _version_report() -> str:
    """Return package and authoring API identities for diagnostics."""
    return (
        f"{_DISTRIBUTION_NAME} package: {_package_version()}\n"
        f"CDF authoring API:      {CDF_API_VERSION} ({CDF_API_STATUS})"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        help="local path or s3:// URI for the versioned JobConfig YAML",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute in Isaac Sim; otherwise print the compiled JobOrder plan",
    )
    parser.add_argument(
        "--output-dir",
        help="local path or s3:// URI overriding config output_dir",
    )
    parser.add_argument("--windowed", action="store_true", help="show Isaac Sim")
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="print the accepted JSON Schema and exit",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="domain to include during schema generation; may be repeated",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print package and CDF API versions and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the versioned CDF configuration command-line interface."""
    args = _parse_args(argv)
    if args.version:
        print(_version_report())
        return 0
    try:
        if args.print_schema:
            print(dump_job_config_schema(domains=tuple(args.domain)))
            return 0
        if args.config is None:
            print(
                f"error [CDF API {CDF_API_VERSION}]: "
                "a versioned JobConfig YAML path is required",
                file=sys.stderr,
            )
            return 2
        config_location = args.config
        config, job = load_job(config_location)
        config_base = (
            config_location.rsplit("/", 1)[0]
            if config_location.startswith("s3://")
            else str(Path(config_location).expanduser().resolve().parent)
        )
        output_dir = (
            resolve_path(
                args.output_dir if args.output_dir is not None else config.output_dir,
                Path.cwd() if args.output_dir is not None else config_base,
            )
            if config.capture is not None or args.output_dir is not None
            else None
        )
        serialization_path = (
            resolve_path(config.execution.serialization_path, config_base)
            if config.execution.serialization_path is not None
            else None
        )
        if not args.execute:
            plan = job.compile_orders(
                output_dir=output_dir,
                serialization_path=serialization_path,
            )
            print(json.dumps(plan.model_dump(mode="json", by_alias=True), indent=2))
            return 0
        result = job.execute(
            output_dir=output_dir,
            serialization_path=serialization_path,
            windowed=args.windowed,
        )
        print(json.dumps(asdict(result), indent=2, default=str))
        return 0 if result.success else 1
    except (
        OSError,
        RuntimeError,
        ValueError,
        CDFError,
        PydanticValidationError,
    ) as exception:
        print(
            f"error [CDF API {CDF_API_VERSION}]: {exception}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
