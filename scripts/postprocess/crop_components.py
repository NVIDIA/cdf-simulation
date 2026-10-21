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

"""Thin CLI front-end for the bbox-tight per-component crop.

The implementation lives in the shared crop library at
``scripts/crop/bbox_tight.py`` and is dispatched through
``scripts/crop/facade.py::run_crop``. This file only parses CLI args and calls
it, so the single-flow crop step keeps its stable command-line contract:

    python crop_components.py --input <trigger> --output <out> \
        --crops rgb semantic_segmentation --offset 10 [--min-coverage 0.2] \
        [--reference <ref>] [--class-filter <substr...>] [--xform-depth 7]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Shared crop library (scripts/crop/). This script runs with cwd = repo root,
# so reach it by file path relative to this file.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crop"))

from crop_defaults import DEFAULT_CROPS, MIN_COVERAGE, OFFSET, XFORM_DEPTH  # noqa: E402
from facade import run_crop  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Crop components from SDG output (bbox-tight)")
    parser.add_argument("--input", type=str, required=True, help="Input trigger folder")
    parser.add_argument("--output", type=str, required=True, help="Output folder for crops")
    parser.add_argument(
        "--offset", type=int, default=OFFSET, help=f"Extra pixels around bbox (default {OFFSET})"
    )
    parser.add_argument(
        "--crops",
        type=str,
        nargs="+",
        default=list(DEFAULT_CROPS),
        help=f"Image type prefixes to crop (default: {DEFAULT_CROPS})",
    )
    parser.add_argument(
        "--class-filter",
        type=str,
        nargs="+",
        default=None,
        help="Only crop components whose class/defect contains any of these strings",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference dir (missing mode): bbox/semantic from here, RGB from --input",
    )
    parser.add_argument(
        "--xform-depth",
        type=int,
        default=XFORM_DEPTH,
        help=f"USD hierarchy depth for component Xform extraction (default {XFORM_DEPTH})",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=MIN_COVERAGE,
        help=f"Drop crops whose semantic mask covers < this fraction (default {MIN_COVERAGE})",
    )
    args = parser.parse_args()

    run_crop(
        {
            "method": "bbox_tight",
            "input": args.input,
            "output": args.output,
            "offset": args.offset,
            "crops": args.crops,
            "class_filter": args.class_filter,
            "reference": args.reference,
            "xform_depth": args.xform_depth,
            "min_coverage": args.min_coverage,
        }
    )
