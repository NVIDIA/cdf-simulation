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

"""Import a ``scripts/sdg/standalone`` module without Kit or USD present.

The unit-test CI job runs in `python:3.12-slim` with numpy, Pillow, pydantic and
PyYAML and **nothing else** -- no `omni`, and no `pxr`. A developer machine
usually has `usd-core` installed, so a test that imports one of these modules
passes locally and errors in CI with `ModuleNotFoundError: No module named
'pxr'`. A test that stubs `omni` but not `pxr` (as
`test_defect_variant_controls` and `test_yawed_hinge` would otherwise need to)
hits exactly that.

The functions under test here are pure arithmetic -- variant planning, pivot
rotation matrices. They never reach USD; only the *module scope* of
`defects_utils.transform_utils` does, via `from pxr import UsdGeom`. Stubbing it
is honest: nothing being asserted depends on the real library.

Keep the stub list here rather than in each test, so the next module that grows
a Kit-only import is fixed once.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest import mock

_STANDALONE = (Path(__file__).resolve().parents[3]
               / "scripts" / "sdg" / "standalone")


class _LazyPackage(types.ModuleType):
    """A stub package that fabricates any submodule asked of it.

    Enumerating the USD submodules by hand does not hold: the list was
    ``UsdGeom, Usd, Gf, Sdf, UsdLux, Vt`` and the next module to need a stub
    wanted ``pxr.UsdSemantics``. ``from pxr import X`` resolves ``X`` as an
    attribute of the package before it looks in ``sys.modules``, so answering
    every attribute is enough and nothing has to be kept in sync.
    """

    __path__: list[str] = []

    def __getattr__(self, name: str) -> types.ModuleType:
        if name.startswith("__"):
            raise AttributeError(name)
        submodule = types.ModuleType("%s.%s" % (self.__name__, name))
        setattr(self, name, submodule)
        return submodule


def _stubs() -> dict[str, types.ModuleType]:
    """Module objects standing in for the Kit-only imports."""
    omni = _LazyPackage("omni")
    pxr = _LazyPackage("pxr")
    return {"omni": omni, "omni.usd": omni.usd, "pxr": pxr}


def _forget_defects_utils() -> None:
    for name in [n for n in sys.modules if n.startswith("defects_utils")]:
        sys.modules.pop(name, None)


def import_standalone(*module_names: str):
    """Import one or more ``defects_utils`` modules with the stubs in place.

    The modules are dropped from ``sys.modules`` first and again by
    :func:`forget_standalone`, so a stubbed import cannot leak into a test that
    wants the real thing.

    Args:
        *module_names: Dotted names below ``scripts/sdg/standalone``.

    Returns:
        The imported module, or a tuple of them when several are asked for.

    """
    if str(_STANDALONE) not in sys.path:
        sys.path.insert(0, str(_STANDALONE))
    with mock.patch.dict(sys.modules, _stubs()):
        _forget_defects_utils()
        imported = tuple(importlib.import_module(name)
                         for name in module_names)
    return imported[0] if len(imported) == 1 else imported


def forget_standalone() -> None:
    """Drop the stub-imported modules, for tearDownClass."""
    _forget_defects_utils()
