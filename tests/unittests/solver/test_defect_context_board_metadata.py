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

"""What ``build_defect_context`` accepts as its board metadata.

This exists because of a defect that every other test missed. The 3D-AOI capture
handed ``build_defect_context`` a raw ``json.load`` of the components sidecar,
while ``defect_ops`` had moved to reading a normalized ``BoardMetadataCatalog``
(it indexes ``.prototypes`` and ``.components``). The whole unit suite was green
either side of the change, because nothing off-GPU built a defect context, and it
died on the first scene of the first real capture:

    AttributeError: 'dict' object has no attribute 'prototypes'

``defect_ops`` imports ``omni.usd`` at module scope, which is why there was no
off-GPU test to fail. Stubbing the Kit-only names is enough to reach the branch
that matters -- the board-metadata indexing touches neither Kit nor USD.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STANDALONE = _REPO_ROOT / "scripts" / "sdg" / "standalone"
_SIDECAR = (
    _REPO_ROOT / "tests" / "functional" / "resources"
    / "SMT_0805_C18_coupon.components.json"
)

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pcb_aoi.board_metadata import SidecarMetadataProvider  # noqa: E402


try:
    from pxr import UsdGeom as _UsdGeom
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    _UsdGeom = None


def _kit_stubs() -> dict[str, types.ModuleType]:
    """Return the Kit-only modules ``defect_ops`` imports at module scope.

    Only ``omni`` is stubbed here. ``pxr`` deliberately is **not**: unlike the
    pure-arithmetic tests that share ``tests/unittests/core/kitless.py``, the code
    under test calls into USD for real -- ``defect_ops`` reaches
    ``UsdGeom.GetStageMetersPerUnit``. Faking that would be asserting against a
    stub rather than against the library, so the class skips when usd-core is
    absent, which is what the rest of the suite does
    (see ``tests/unittests/test_write_mesh_geometry.py``).
    """
    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_usd = types.ModuleType("omni.usd")
    omni.usd = omni_usd
    return {"omni": omni, "omni.usd": omni_usd}


@unittest.skipUnless(_UsdGeom is not None, "usd-core is not installed")
class BuildDefectContextBoardMetadataTest(unittest.TestCase):
    """The board-metadata argument is a catalog, not a decoded sidecar."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._patch = mock.patch.dict(sys.modules, _kit_stubs())
        cls._patch.start()
        # Recorded so teardown removes exactly what setup added. The previous
        # form inserted unconditionally and then removed EVERY occurrence, so a
        # path a different test had legitimately put there was taken away too.
        cls._added_standalone = str(_STANDALONE) not in sys.path
        if cls._added_standalone:
            sys.path.insert(0, str(_STANDALONE))
        try:
            cls.defect_ops = importlib.import_module("defect_ops")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_added_standalone", False):
            try:
                sys.path.remove(str(_STANDALONE))
            except ValueError:
                pass
        for name in ("defect_ops",):
            sys.modules.pop(name, None)
        cls._patch.stop()

    @unittest.skipUnless(_SIDECAR.is_file(), "approved SMT coupon fixture is not shipped in the public release")
    def test_a_catalog_populates_the_contact_index(self) -> None:
        catalog = SidecarMetadataProvider().from_path(_SIDECAR)
        context = self.defect_ops.build_defect_context(None, None, catalog)

        self.assertTrue(
            context["contact_index"],
            "a catalog with components must index at least one instance",
        )
        for path, record in context["contact_index"].items():
            self.assertEqual(record["path"], path)
            self.assertIn("refdes", record)
            self.assertIn("solder_contact_prims", record)
        self.assertEqual(
            set(context["component_defs_by_path"]),
            set(context["contact_index"]),
            "every indexed instance needs a component definition beside it",
        )

    @unittest.skipUnless(_SIDECAR.is_file(), "approved SMT coupon fixture is not shipped in the public release")
    def test_the_raw_sidecar_dict_is_rejected(self) -> None:
        # The regression guard. Passing what json.load returns used to reach
        # production and fail only under Isaac; it must fail here instead.
        raw = json.loads(_SIDECAR.read_text(encoding="utf-8"))
        self.assertIsInstance(raw, dict)
        self.assertFalse(
            hasattr(raw, "prototypes"),
            "if a plain dict ever grows .prototypes this test is meaningless",
        )
        with self.assertRaises(AttributeError):
            self.defect_ops.build_defect_context(None, None, raw)

    def test_none_is_still_allowed(self) -> None:
        # Callers without a sidecar pass None, and that must stay cheap rather
        # than becoming an error alongside the stricter type.
        context = self.defect_ops.build_defect_context(None, None, None)
        self.assertEqual(context["contact_index"], {})
        self.assertEqual(context["component_defs_by_path"], {})


if __name__ == "__main__":
    unittest.main()
