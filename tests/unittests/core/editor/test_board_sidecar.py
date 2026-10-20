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

"""Tests for the shared board-sidecar helper.

Pins the ``<board>.components.json`` path resolution, index loading,
exact-prim and ancestor lookup, and prefix remapping conventions.
"""

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EDITOR_DIR = _REPO_ROOT / "scripts" / "sdg" / "editor"


def _load(name):
    sys.path.insert(0, str(_EDITOR_DIR))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(_EDITOR_DIR))


bs = _load("board_sidecar")

_B = "/Board_FIX/Components"
_SIDECAR_DATA = {
    "component_instances": [
        {
            "path": f"{_B}/Top/CAP_SMD_0603/C1",
            "refdes": "C1",
            "solder_paste_prims": [
                f"{_B}/Top/CAP_SMD_0603/C1/pad_l",
                f"{_B}/Top/CAP_SMD_0603/C1/pad_r",
            ],
            "solder_contact_prims": [f"{_B}/Top/CAP_SMD_0603/C1/solder_l"],
            "body_prims": [f"{_B}/Top/CAP_SMD_0603/C1/pkg_body0"],
        },
        {
            "path": f"{_B}/Bottom/RES_SMD_0603/R7",
            "refdes": "R7",
            "solder_paste_prims": [f"{_B}/Bottom/RES_SMD_0603/R7/pad0"],
        },
        {"no_path_field": True},
    ],
}


class SidecarPathResolutionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="board_sidecar_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.sidecar = os.path.join(self.tmp, "board.components.json")
        with open(self.sidecar, "w", encoding="utf-8") as handle:
            json.dump(_SIDECAR_DATA, handle)
        self.layer = os.path.join(self.tmp, "board.usd")

    def test_explicit_path_wins(self):
        self.assertEqual(
            bs.resolve_sidecar_path(self.sidecar, ("/elsewhere/x.usd",)), self.sidecar
        )

    def test_explicit_but_missing_never_falls_back(self):
        # an explicit-but-wrong path must NOT silently pick another board's
        # sidecar (matches the fillet editor's historical behavior)
        missing = os.path.join(self.tmp, "nope.components.json")
        self.assertEqual(bs.resolve_sidecar_path(missing, (self.layer,)), "")

    def test_derives_stem_next_to_layer(self):
        self.assertEqual(bs.resolve_sidecar_path(None, (self.layer,)), self.sidecar)

    def test_aoi_overlay_falls_back_to_the_base_board_sidecar(self):
        # The supplied AOI material overlay (<board>_aoi.usd) shares
        # ONE <board>.components.json with the whole board.
        # Deriving <board>_aoi.components.json found nothing, so every
        # sidecar-backed capability silently degraded to bbox math.
        overlay = os.path.join(self.tmp, "board_aoi.usd")
        self.assertEqual(bs.resolve_sidecar_path(None, (overlay,)), self.sidecar)

    def test_every_derived_layer_suffix_falls_back(self):
        for suffix in bs.DERIVED_LAYER_SUFFIXES:
            with self.subTest(suffix=suffix):
                layer = os.path.join(self.tmp, f"board{suffix}.usd")
                self.assertEqual(bs.resolve_sidecar_path(None, (layer,)), self.sidecar)

    def test_derived_sidecar_wins_over_the_base_one(self):
        # A board that really does ship its own <stem>.components.json keeps
        # it — the base-stem fallback is a second choice, never an override.
        overlay = os.path.join(self.tmp, "board_aoi.usd")
        own = os.path.join(self.tmp, "board_aoi.components.json")
        with open(own, "w", encoding="utf-8") as handle:
            json.dump(_SIDECAR_DATA, handle)
        self.assertEqual(bs.resolve_sidecar_path(None, (overlay,)), own)

    def test_suffix_match_is_on_the_stem_not_the_board_name(self):
        # "_aoi" only strips when it is the layer stem's tail; a board
        # legitimately named e.g. aoi_panel.usd must not borrow another
        # board's sidecar.
        named = os.path.join(self.tmp, "aoi_panel.usd")
        self.assertEqual(bs.resolve_sidecar_path(None, (named,)), "")

    def test_first_layer_with_sidecar_wins(self):
        other = os.path.join(self.tmp, "no_sidecar.usd")
        self.assertEqual(
            bs.resolve_sidecar_path(None, (other, self.layer)), self.sidecar
        )

    def test_legacy_path_without_sidecar_returns_empty(self):
        # Legacy authored boards ship no components.json — "" signals the
        # comp_types + bbox fallback placement path
        self.assertEqual(
            bs.resolve_sidecar_path(None, (os.path.join(self.tmp, "legacy.usd"),)), ""
        )
        self.assertEqual(bs.resolve_sidecar_path(None, ()), "")
        self.assertEqual(bs.resolve_sidecar_path(None, ("",)), "")

    def test_load_component_index(self):
        index = bs.load_component_index(self.sidecar)
        self.assertEqual(
            set(index), {f"{_B}/Top/CAP_SMD_0603/C1", f"{_B}/Bottom/RES_SMD_0603/R7"}
        )

    def test_load_component_index_missing_or_bad_returns_none(self):
        self.assertIsNone(bs.load_component_index(""))
        self.assertIsNone(
            bs.load_component_index(os.path.join(self.tmp, "absent.json"))
        )
        bad = os.path.join(self.tmp, "bad.json")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        self.assertIsNone(bs.load_component_index(bad))


class SidecarLookupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = {
            e["path"]: e for e in _SIDECAR_DATA["component_instances"] if "path" in e
        }

    def test_exact_prim_match_no_prefix(self):
        entry, prefix = bs.lookup(self.index, f"{_B}/Top/CAP_SMD_0603/C1")
        self.assertEqual(entry["refdes"], "C1")
        self.assertEqual(prefix, "")

    def test_world_mounted_prim_matches_with_prefix(self):
        entry, prefix = bs.lookup(self.index, f"/World{_B}/Top/CAP_SMD_0603/C1")
        self.assertEqual(entry["refdes"], "C1")
        self.assertEqual(prefix, "/World")

    def test_ancestor_walk_resolves_sub_prims(self):
        # a pad/body sub-prim resolves to its component's record
        entry, prefix = bs.lookup(
            self.index, f"/World{_B}/Top/CAP_SMD_0603/C1/pkg_body0"
        )
        self.assertEqual(entry["refdes"], "C1")
        self.assertEqual(prefix, "/World")

    def test_no_match_returns_none(self):
        self.assertEqual(bs.lookup(self.index, "/World/NotABoard/X"), (None, ""))

    def test_remap_record_applies_prefix_and_keeps_original(self):
        entry, prefix = bs.lookup(self.index, f"/World{_B}/Top/CAP_SMD_0603/C1")
        record = bs.remap_record(entry, prefix)
        self.assertEqual(
            record["body_prims"], [f"/World{_B}/Top/CAP_SMD_0603/C1/pkg_body0"]
        )
        self.assertEqual(
            record["solder_paste_prims"][0], f"/World{_B}/Top/CAP_SMD_0603/C1/pad_l"
        )
        # the source entry must stay board-relative (cached indexes!)
        self.assertEqual(entry["body_prims"], [f"{_B}/Top/CAP_SMD_0603/C1/pkg_body0"])
        self.assertEqual(record["refdes"], "C1")


if __name__ == "__main__":
    unittest.main()
