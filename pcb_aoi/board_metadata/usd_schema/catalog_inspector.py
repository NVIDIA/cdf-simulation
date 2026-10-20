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

"""Export the important metadata in a composed PCBA USD stage as JSON."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pxr import Tf, Usd

from ..catalog_json_writer import write_catalog_json
from ..metadata_provider import BoardMetadataRequest
from .metadata_provider import UsdSchemaMetadataProvider


def _parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured parser.

    """
    parser = argparse.ArgumentParser(
        description="Export a composed PCBA USD metadata catalog as JSON.",
    )
    parser.add_argument("usd_path", type=Path, help="PCBA USD stage to inspect")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path (default: <usd-stem>.board_metadata.json)",
    )
    parser.add_argument(
        "--board-root",
        help="absolute board-root prim path (default: stage default prim)",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Load a PCBA USD stage and write its normalized metadata catalog.

    Args:
        arguments: Command-line arguments excluding the executable name.

    Returns:
        Zero after the catalog is written.

    Raises:
        RuntimeError: If the USD stage cannot be opened.

    """
    options = _parser().parse_args(arguments)
    usd_path = options.usd_path.resolve()
    try:
        stage = Usd.Stage.Open(str(usd_path))
    except Tf.ErrorException as exc:
        raise RuntimeError(f"could not open USD stage {usd_path}: {exc}") from exc
    if stage is None:
        raise RuntimeError(f"could not open USD stage {usd_path}")
    output_path = options.output or usd_path.with_name(
        f"{usd_path.stem}.board_metadata.json",
    )
    catalog = UsdSchemaMetadataProvider().load(
        BoardMetadataRequest(
            board_root=options.board_root,
            source_identifier=str(usd_path),
            stage=stage,
        ),
    )
    written_path = write_catalog_json(catalog, output_path)
    print(written_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
