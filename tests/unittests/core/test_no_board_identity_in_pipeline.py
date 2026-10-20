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

"""No single board may be named in the pipeline's source.

Board-shaped constants come back one convenience default at a time, and they
fail quietly: a prim path or a component-type name that belongs to one board
matches nothing on the next one, so the run finishes, writes frames, and
reports success over work it never did. Review does not catch that reliably —
the constant looks like a sensible default right up until someone else's board
loads.

So the rule is mechanical. Board identity lives in a ``--pcba-config`` target
shipped with the board's assets, outside this repository; the pipeline reads
it from there. This test fails when it reappears in code.

The patterns below are shapes, not a denylist of today's boards: mangled
instance names, part-library prototype names, and manufacturer part numbers
cannot be anything but one board's identity, whichever board that is.
"""

import re
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Trees that make up the pipeline. Board identity is forbidden in all of them.
_PIPELINE_TREES = ("pcb_aoi", "cdf", "scripts")

#: Paths inside those trees that may name a board, and why.
_ALLOWED = (
    # Vendored upstream sources; not ours to rewrite.
    "pcb_aoi/solder_evolver/third_party/",
)

#: (name, compiled pattern, what it is) — each matches only board identity.
_FORBIDDEN = (
    (
        "mangled instance name",
        re.compile(r"tn__\w+"),
        "an ECAD-mangled prim name; it exists in exactly one board's namespace",
    ),
    (
        "part-library prototype name",
        re.compile(r"_\d{3,4}[A-Z]*(?:_[A-Z])?_H\d{2,3}\b"),
        "a component-type name from one board's part library",
    ),
    (
        "manufacturer part number",
        # A bare 8-digit literal also matches array sizes, timestamps and
        # hashes, so the rule fired on code that names no board at all.
        # Only shapes that are recognisably part numbers: the A5E prefix,
        # or a run of digits carrying a part-number context word.
        re.compile(r"(?:A5E\d{6,}"
                   r"|(?:part[_ ]?(?:no|num|number)|mpn|p/n)\W{0,4}\d{6,})",
                   re.IGNORECASE),
        "a board part number",
    ),
    (
        "named board asset",
        re.compile(r"\bpcba_\w+\.usd[ac]?\b|\b\w+_mod\.usd[ac]?\b|\b\w+_s_detail\b"),
        "a specific board asset filename",
    ),
)


def _sources():
    """Yield every pipeline source file the rule applies to."""
    for tree in _PIPELINE_TREES:
        root = _REPO_ROOT / tree
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if any(rel.startswith(prefix) for prefix in _ALLOWED):
                continue
            # Examples are where a board belongs; they are not the pipeline.
            if "/examples/" in rel:
                continue
            yield rel, path
class NoBoardIdentityInPipelineTest(unittest.TestCase):
    """The pipeline must not be able to name a board."""

    def test_pipeline_sources_name_no_board(self):
        offences = []
        for rel, path in _sources():
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), 1):
                for label, pattern, why in _FORBIDDEN:
                    found = pattern.search(line)
                    if found:
                        offences.append(
                            f"{rel}:{line_number}: {label} {found.group(0)!r} "
                            f"— {why}. Move it to the board's pcba target."
                        )
        self.assertEqual(
            offences,
            [],
            "board identity found in pipeline source:\n  " + "\n  ".join(offences),
        )

    def test_the_scan_actually_covers_the_pipeline(self):
        # A rule that silently stops scanning is worse than no rule.
        #
        # Pick canaries that are not themselves in motion.
        scanned = [rel for rel, _ in _sources()]
        self.assertGreater(len(scanned), 100, "source discovery collapsed")
        for expected in (
            "cdf/execution/isaac/capture_execution.py",
            "scripts/sdg/editor/board_material_override.py",
            "scripts/sdg/editor/component_material_override.py",
            "scripts/sdg/editor/editor_solder_fillet_board.py",
        ):
            self.assertIn(expected, scanned)

    def test_the_patterns_would_catch_a_regression(self):
        # Each pattern is checked against the constant it was written for, so
        # a future edit that guts a pattern fails here instead of going quiet.
        samples = {
            "mangled instance name": "/World/asset/PCBA/tn__12345678BASEA01_fM9E",
            "part-library prototype name": 'comp_types = ["_0402_H055"]',
            "manufacturer part number": "Panel_A5E12345678-01",
            "named board asset": "assets_final/pcba_x_mod.usd",
        }
        for label, pattern, _ in _FORBIDDEN:
            self.assertIsNotNone(
                pattern.search(samples[label]), f"{label} pattern stopped matching"
            )


if __name__ == "__main__":
    unittest.main()
