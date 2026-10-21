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

"""Test transactional capture visibility for generated PCB solder and board roles."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from cdf.errors import ValidationError
from pcb_aoi.capture_visibility import (
    _hide_board_roles,
    _hide_solder,
    _restore_visibility,
    _show_board_roles,
)


class _Attribute:
    """Minimal visibility attribute retaining authored state."""

    def __init__(self, value: str = "inherited", *, authored: bool = False) -> None:
        self.value = value
        self.authored = authored
        self.cleared = False

    def Get(self) -> str:
        """Return the current visibility token."""
        return self.value

    def Set(self, value: str) -> None:
        """Author a visibility token."""
        self.value = value
        self.authored = True
        self.cleared = False

    def Clear(self) -> None:
        """Clear the authored visibility opinion."""
        self.value = "inherited"
        self.authored = False
        self.cleared = True

    def HasAuthoredValueOpinion(self) -> bool:
        """Return whether the attribute has an authored opinion."""
        return self.authored


class _Prim:
    """Minimal imageable prim used by the visibility helper."""

    def __init__(self, *, valid: bool = True, authored: bool = False) -> None:
        self.valid = valid
        self.visibility = _Attribute(authored=authored)

    def IsValid(self) -> bool:
        """Return whether the prim resolves on the stage."""
        return self.valid


class _Imageable:
    """Minimal ``UsdGeom.Imageable`` adapter."""

    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def __bool__(self) -> bool:
        return self.prim.IsValid()

    def GetVisibilityAttr(self) -> _Attribute:
        """Return the prim visibility attribute."""
        return self.prim.visibility


class _Stage:
    """Minimal prim-path lookup stage."""

    def __init__(self, prims: dict[str, _Prim]) -> None:
        self.prims = prims

    def GetPrimAtPath(self, path: str) -> _Prim:
        """Return the requested prim or an invalid placeholder."""
        return self.prims.get(path, _Prim(valid=False))


def _pxr_module() -> types.ModuleType:
    """Build a minimal lazy ``pxr.UsdGeom`` module."""
    module = types.ModuleType("pxr")
    module.UsdGeom = types.SimpleNamespace(
        Imageable=_Imageable,
        Tokens=types.SimpleNamespace(
            inherited="inherited",
            invisible="invisible",
        ),
    )
    return module


class CaptureVisibilityTest(unittest.TestCase):
    """Pass-scoped solder visibility behavior."""

    def test_hides_output_root_and_clears_new_visibility_opinion(self) -> None:
        prim = _Prim()
        stage = _Stage({"/Board/Solder": prim})
        manifest = types.SimpleNamespace(
            output_root_path="/Board/Solder",
            generated_prim_paths=("/Board/Solder/Joint_1",),
        )

        with mock.patch.dict(sys.modules, {"pxr": _pxr_module()}):
            snapshots = _hide_solder(stage, manifest)
            self.assertEqual(prim.visibility.value, "invisible")
            _restore_visibility(stage, snapshots)

        self.assertEqual(len(snapshots), 1)
        self.assertTrue(prim.visibility.cleared)
        self.assertEqual(prim.visibility.value, "inherited")

    def test_restores_preexisting_authored_visibility(self) -> None:
        prim = _Prim(authored=True)
        prim.visibility.value = "inherited"
        stage = _Stage({"/Board/Solder": prim})
        manifest = types.SimpleNamespace(
            output_root_path="/Board/Solder",
            generated_prim_paths=(),
        )

        with mock.patch.dict(sys.modules, {"pxr": _pxr_module()}):
            snapshots = _hide_solder(stage, manifest)
            _restore_visibility(stage, snapshots)

        self.assertTrue(prim.visibility.authored)
        self.assertFalse(prim.visibility.cleared)
        self.assertEqual(prim.visibility.value, "inherited")

    def test_hides_composed_solder_roots_after_instance_flattening(self) -> None:
        source = _Prim()
        first = _Prim()
        second = _Prim()
        stage = _Stage(
            {
                "/Board/Solder": source,
                "/Panel/I0/Solder": first,
                "/Panel/I1/Solder": second,
            },
        )
        manifest = types.SimpleNamespace(
            output_root_path="/Board/Solder",
            generated_prim_paths=(),
        )
        cohort = mock.Mock()
        cohort.expand_source_path.return_value = (
            types.SimpleNamespace(composed_path="/Panel/I0/Solder"),
            types.SimpleNamespace(composed_path="/Panel/I1/Solder"),
        )

        with mock.patch.dict(sys.modules, {"pxr": _pxr_module()}):
            snapshots = _hide_solder(stage, manifest, cohort)

        self.assertEqual(
            tuple(snapshot.path for snapshot in snapshots),
            ("/Panel/I0/Solder", "/Panel/I1/Solder"),
        )
        self.assertEqual(source.visibility.value, "inherited")
        self.assertEqual(first.visibility.value, "invisible")
        self.assertEqual(second.visibility.value, "invisible")

    def test_rejects_manifest_without_resolvable_solder(self) -> None:
        stage = _Stage({})
        manifest = types.SimpleNamespace(
            output_root_path="/Board/Solder",
            generated_prim_paths=("/Board/Solder/Joint_1",),
        )

        with (
            mock.patch.dict(sys.modules, {"pxr": _pxr_module()}),
            self.assertRaisesRegex(ValidationError, "resolves no generated prims"),
        ):
            _hide_solder(stage, manifest)


class _RoleAttribute:
    """Attribute that returns a board role value or is falsy when absent."""

    def __init__(self, role: str | None) -> None:
        self._role = role

    def __bool__(self) -> bool:
        return self._role is not None

    def Get(self) -> str | None:
        return self._role


class _BoardPrim(_Prim):
    """Prim that carries an optional ``pcba:mesh:role`` attribute."""

    def __init__(
        self,
        path: str,
        *,
        role: str | None = None,
        parent: _BoardPrim | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._path = path
        self._role = role
        self._parent = parent

    def GetPath(self) -> str:
        return self._path

    def GetAttribute(self, _attr_name: str) -> _RoleAttribute:
        return _RoleAttribute(self._role)

    def GetParent(self) -> _Prim:
        """Return the parent prim or an invalid root sentinel."""
        return self._parent or _Prim(valid=False)


def _pxr_with_primrange(children: list) -> types.ModuleType:
    """Build the pxr mock extended with a ``Usd.PrimRange`` iterator."""
    module = _pxr_module()
    module.Usd = types.SimpleNamespace(PrimRange=lambda _root: iter(children))
    return module


def _metadata_keys_module() -> types.ModuleType:
    """Minimal ``pcb_aoi.board_metadata.usd_schema_metadata_keys`` stub."""
    module = types.ModuleType("pcb_aoi.board_metadata.usd_schema_metadata_keys")
    module.MESH_ROLE_ATTR = "pcba:mesh:role"
    return module


class BoardRoleVisibilityTest(unittest.TestCase):
    """Role-scan visibility behavior for ``pcba:mesh:role``-tagged prims."""

    def _mock_modules(self, children: list) -> dict:
        return {
            "pxr": _pxr_with_primrange(children),
            "pcb_aoi.board_metadata.usd_schema_metadata_keys": _metadata_keys_module(),
        }

    def test_hides_copper_prims_by_role(self) -> None:
        copper1 = _BoardPrim("/Board/Copper/Pour_1", role="board_copper")
        copper2 = _BoardPrim("/Board/Copper/Pour_2", role="board_copper")
        board_prim = _Prim()
        stage = _Stage({"/Board": board_prim})

        with mock.patch.dict(sys.modules, self._mock_modules([copper1, copper2])):
            snapshots = _hide_board_roles(stage, "/Board", {"copper"})

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(copper1.visibility.value, "invisible")
        self.assertEqual(copper2.visibility.value, "invisible")

    def test_ignores_prims_without_matching_role(self) -> None:
        copper = _BoardPrim("/Board/Copper/Pour_1", role="board_copper")
        pad = _BoardPrim("/Board/Pads/Pad_1", role="board_pad")
        board_prim = _Prim()
        stage = _Stage({"/Board": board_prim})

        with mock.patch.dict(sys.modules, self._mock_modules([copper, pad])):
            snapshots = _hide_board_roles(stage, "/Board", {"copper"})

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(copper.visibility.value, "invisible")
        self.assertEqual(pad.visibility.value, "inherited")

    def test_hides_multiple_roles_in_one_call(self) -> None:
        copper = _BoardPrim("/Board/Copper/Pour_1", role="board_copper")
        mask = _BoardPrim("/Board/Mask/Mask_1", role="board_mask")
        board_prim = _Prim()
        stage = _Stage({"/Board": board_prim})

        with mock.patch.dict(sys.modules, self._mock_modules([copper, mask])):
            snapshots = _hide_board_roles(stage, "/Board", {"copper", "mask"})

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(copper.visibility.value, "invisible")
        self.assertEqual(mask.visibility.value, "invisible")

    def test_restores_authored_opinion_after_hide(self) -> None:
        copper = _BoardPrim("/Board/Copper/Pour_1", role="board_copper", authored=True)
        copper.visibility.value = "inherited"
        board_prim = _Prim()
        stage = _Stage({"/Board": board_prim, "/Board/Copper/Pour_1": copper})

        with mock.patch.dict(sys.modules, self._mock_modules([copper])):
            snapshots = _hide_board_roles(stage, "/Board", {"copper"})
            _restore_visibility(stage, snapshots)

        self.assertTrue(copper.visibility.authored)
        self.assertFalse(copper.visibility.cleared)
        self.assertEqual(copper.visibility.value, "inherited")

    def test_shows_default_hidden_annotation_root_and_restores_it(self) -> None:
        board = _BoardPrim("/Board")
        root = _BoardPrim(
            "/Board/PinTips",
            parent=board,
            authored=True,
        )
        root.visibility.value = "invisible"
        tip = _BoardPrim(
            "/Board/PinTips/tip_1",
            role="board_pin_tip",
            parent=root,
        )
        stage = _Stage(
            {
                "/Board": board,
                "/Board/PinTips": root,
                "/Board/PinTips/tip_1": tip,
            }
        )

        with mock.patch.dict(sys.modules, self._mock_modules([root, tip])):
            snapshots = _show_board_roles(stage, "/Board", {"pin_tip"})
            self.assertEqual(root.visibility.value, "inherited")
            _restore_visibility(stage, snapshots)

        self.assertEqual(tuple(item.path for item in snapshots), ("/Board/PinTips",))
        self.assertEqual(root.visibility.value, "invisible")

    def test_rejects_role_with_no_matching_prims(self) -> None:
        board_prim = _Prim()
        stage = _Stage({"/Board": board_prim})

        with (
            mock.patch.dict(sys.modules, self._mock_modules([])),
            self.assertWarns(UserWarning),
        ):
            _hide_board_roles(stage, "/Board", {"copper"})

    def test_rejects_invalid_board_root(self) -> None:
        stage = _Stage({})

        with (
            mock.patch.dict(sys.modules, self._mock_modules([])),
            self.assertRaisesRegex(ValidationError, r"not a valid prim"),
        ):
            _hide_board_roles(stage, "/Board", {"copper"})


if __name__ == "__main__":
    unittest.main()
