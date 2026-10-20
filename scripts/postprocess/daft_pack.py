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

"""Command line entry point for the DAFT packager.

The packaging logic lives in :mod:`daft`; this file is argument parsing and exit
codes, so the core stays importable without inheriting a CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from daft import load_records, pack_dir  # noqa: F401 -- re-exported for callers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", required=True,
                        help="raw capture output dir (holds *.pair.json)")
    parser.add_argument("--out", dest="dst", required=True,
                        help="DAFT dataset output dir")
    parser.add_argument("--strict", action="store_true",
                        help="fail if a referenced raw file is missing")
    args = parser.parse_args(argv)
    mani = pack_dir(Path(args.src), Path(args.dst), strict=args.strict)
    print(f"[daft] packed {mani['scene_count']} scene(s) -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
