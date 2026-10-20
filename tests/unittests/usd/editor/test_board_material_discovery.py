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

"""Board-material discovery must be structural, never name-based.

``board_material_override`` used to find its asset layer by testing every used
layer's filename for one board's asset name, and to hardcode both the live
Looks path and the layer-internal path that pairs with it. This pins the
replacements: the Looks scope, the asset layer, and the layer-internal shader
path all come from composition, on a stage whose asset name is arbitrary.
"""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from pxr import Usd, UsdGeom, UsdShade
except ImportError:  # the CI unit-test image installs numpy only
    Usd = UsdGeom = UsdShade = None

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EDITOR_DIR = _REPO_ROOT / "scripts" / "sdg" / "editor"


def _load(name):
    sys.path.insert(0, str(_EDITOR_DIR))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(_EDITOR_DIR))


# The module under test imports pxr itself, so it is only loadable where USD is.
bmo = _load("board_material_override") if UsdGeom is not None else None

_ASSET_STEM = "an_arbitrary_board"
_BOARD_ROOT = f"/World/{_ASSET_STEM}/PCBA/tn__assembly"
_LOOKS = f"/World/{_ASSET_STEM}/Looks"


def _build_scene(tmpdir):
    """Reference an asset layer under an asset xform, as the boards do.

    Returns the in-memory scene stage and the asset layer's identifier.
    """
    asset = os.path.join(tmpdir, f"{_ASSET_STEM}.usda")
    asset_stage = Usd.Stage.CreateNew(asset)
    world = UsdGeom.Xform.Define(asset_stage, "/World")
    asset_stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(asset_stage, "/World/PCBA/tn__assembly")
    for name in bmo._MATERIALS:
        UsdShade.Material.Define(asset_stage, f"/World/Looks/{name}")
        UsdShade.Shader.Define(asset_stage, f"/World/Looks/{name}/Shader")
    asset_stage.GetRootLayer().Save()

    scene = Usd.Stage.CreateInMemory()
    scene_world = UsdGeom.Xform.Define(scene, "/World")
    scene.SetDefaultPrim(scene_world.GetPrim())
    holder = UsdGeom.Xform.Define(scene, f"/World/{_ASSET_STEM}")
    holder.GetPrim().GetReferences().AddReference(asset)
    return scene, asset


@unittest.skipUnless(UsdGeom is not None, "usd-core is not installed")
class BoardMaterialDiscoveryTest(unittest.TestCase):
    """Discovery works on a board whose asset filename is unknown here."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.scene, self.asset = _build_scene(self.tmp)
        self.assertTrue(self.scene.GetPrimAtPath(_BOARD_ROOT).IsValid())

    def _same_layer(self, layer):
        self.assertIsNotNone(layer)
        self.assertEqual(
            os.path.normcase(layer.identifier), os.path.normcase(self.asset)
        )

    def test_looks_root_found_from_the_board_root(self):
        self.assertEqual(bmo.find_board_looks_root(self.scene, _BOARD_ROOT), _LOOKS)

    def test_looks_root_found_without_a_board_root(self):
        self.assertEqual(bmo.find_board_looks_root(self.scene), _LOOKS)

    def test_board_layer_resolves_by_composition(self):
        self._same_layer(bmo.resolve_board_layer(self.scene, _BOARD_ROOT))
        self._same_layer(bmo.resolve_board_layer(self.scene))

    def test_setup_pairs_live_paths_with_layer_internal_paths(self):
        # The asset layer's namespace strips the asset xform the scene mounts
        # it under, so the two paths must differ by exactly that prefix. Both
        # used to be string constants naming one board.
        state = bmo._State()
        ok, info = state.setup(self.scene, _LOOKS, extra_blackout_enabled=False)
        self.assertTrue(ok, info)
        self._same_layer(state.pcba_layer)
        for name in bmo._MATERIALS:
            live, internal = info[name].split(" ↔ ")
            self.assertEqual(live, f"{_LOOKS}/{name}/Shader")
            self.assertEqual(internal, f"/World/Looks/{name}/Shader")

    def test_setup_inserts_exactly_one_sublayer_into_the_asset_layer(self):
        state = bmo._State()
        ok, _ = state.setup(self.scene, _LOOKS, extra_blackout_enabled=False)
        self.assertTrue(ok)
        self.assertEqual(len(state.pcba_layer.subLayerPaths), 1)
        self.assertEqual(state.pcba_layer.subLayerPaths[0], state.anon.identifier)

    def test_missing_looks_root_is_refused_not_guessed(self):
        ok, why = bmo._State().setup(self.scene, "")
        self.assertFalse(ok)
        self.assertIn("looks_root", why)

    def test_two_boards_refuse_rather_than_pick_one(self):
        # Zero-config discovery is a convenience, not a guess: with a second
        # asset on the stage there is no answer, and returning the first one
        # would style whichever board happened to be traversed first.
        second = os.path.join(self.tmp, "second")
        os.makedirs(second, exist_ok=True)
        other, _ = _build_scene(second)  # a separate asset layer on disk
        del other
        holder = UsdGeom.Xform.Define(self.scene, "/World/another_board")
        holder.GetPrim().GetReferences().AddReference(
            os.path.join(second, f"{_ASSET_STEM}.usda")
        )
        self.assertIsNone(bmo.find_board_looks_root(self.scene))
        self.assertIsNone(bmo.resolve_board_layer(self.scene))
        # Naming the board resolves it again — the escape hatch must work.
        self.assertEqual(bmo.find_board_looks_root(self.scene, _BOARD_ROOT), _LOOKS)
        self._same_layer(bmo.resolve_board_layer(self.scene, _BOARD_ROOT))

    def test_one_board_placed_twice_is_not_ambiguous(self):
        # A panel repeats the same asset. Both scopes resolve to one layer, so
        # either authors the same override and refusing would be unhelpful.
        holder = UsdGeom.Xform.Define(self.scene, "/World/second_placement")
        holder.GetPrim().GetReferences().AddReference(self.asset)
        self.assertIsNotNone(bmo.find_board_looks_root(self.scene))
        self._same_layer(bmo.resolve_board_layer(self.scene))

    def test_no_blackout_prims_are_hardcoded(self):
        # These are per-board mesh paths; a default would name meshes that
        # exist on one board and silently miss everywhere else.
        self.assertEqual(tuple(bmo.EXTRA_BLACKOUT_PRIMS), ())


if __name__ == "__main__":
    unittest.main()
