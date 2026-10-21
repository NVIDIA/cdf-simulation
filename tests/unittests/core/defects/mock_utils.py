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

"""Omniverse and USD test doubles for component-defect unit tests."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest import mock

_STANDALONE_DIR = str(
    Path(__file__).resolve().parents[4] / "scripts" / "sdg" / "standalone"
)


class MockRange:
    """Minimal USD range test double.

    Args:
        minimum: Minimum range coordinates.
        maximum: Maximum range coordinates.
    """

    def __init__(self, minimum: tuple[float, ...], maximum: tuple[float, ...]):
        self._minimum = minimum
        self._maximum = maximum

    def GetMin(self) -> tuple[float, ...]:
        """Return the minimum range coordinates.

        Returns:
            Minimum coordinate tuple.
        """
        return self._minimum

    def GetMax(self) -> tuple[float, ...]:
        """Return the maximum range coordinates.

        Returns:
            Maximum coordinate tuple.
        """
        return self._maximum


class MockBound:
    """Minimal USD bound test double.

    Args:
        minimum: Minimum bound coordinates.
        maximum: Maximum bound coordinates.
    """

    def __init__(self, minimum: tuple[float, ...], maximum: tuple[float, ...]):
        self._range = MockRange(minimum, maximum)

    def GetRange(self) -> MockRange:
        """Return the bound range.

        Returns:
            Range test double for the bound.
        """
        return self._range


class MockPrim:
    """Minimal USD prim test double.

    Args:
        local_bounds: Component-local minimum and maximum coordinates.
        world_bounds: Optional world-space minimum and maximum coordinates.
        valid: Whether the prim reports itself as valid.
    """

    def __init__(
        self,
        local_bounds: tuple[tuple[float, ...], tuple[float, ...]],
        world_bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
        *,
        valid: bool = True,
    ):
        self.local_bounds = local_bounds
        self.world_bounds = world_bounds or local_bounds
        self._valid = valid

    def IsValid(self) -> bool:
        """Return whether the prim is valid.

        Returns:
            True when the prim should be treated as valid.
        """
        return self._valid

    def GetAttribute(self, _name: str) -> None:
        """Return no authored attribute for this test double.

        Args:
            _name: Attribute name requested by the implementation.

        Returns:
            None because this test double has no authored attributes.
        """
        return None


class MockXformAttr:
    """Minimal ``xformOp:transform`` attribute test double.

    Args:
        matrix: Row-major 4x4 transform as nested sequences.
    """

    def __init__(self, matrix: list[list[float]]):
        self._matrix = matrix

    def IsValid(self) -> bool:
        """Return whether the attribute is valid.

        Returns:
            True because this test double always models an authored attribute.
        """
        return True

    def Get(self) -> list[list[float]]:
        """Return the configured transform value.

        Returns:
            Row-major 4x4 transform as nested sequences.
        """
        return self._matrix


class MockXformPrim(MockPrim):
    """Prim test double that authors an ``xformOp:transform`` attribute.

    Args:
        local_bounds: Component-local minimum and maximum coordinates.
        world_bounds: Optional world-space minimum and maximum coordinates.
        xform: Optional row-major 4x4 transform returned for
            ``xformOp:transform`` attribute lookups.
        valid: Whether the prim reports itself as valid.
    """

    def __init__(
        self,
        local_bounds: tuple[tuple[float, ...], tuple[float, ...]],
        world_bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
        *,
        xform: list[list[float]] | None = None,
        composed_xform: list[list[float]] | None = None,
        valid: bool = True,
    ):
        super().__init__(local_bounds, world_bounds, valid=valid)
        self._xform = xform
        self.composed_xform = composed_xform

    def GetAttribute(self, name: str) -> MockXformAttr | None:
        """Return the configured transform attribute when requested.

        Args:
            name: Attribute name requested by the implementation.

        Returns:
            Transform attribute test double for ``xformOp:transform``, or
            None for any other attribute.
        """
        if name == "xformOp:transform" and self._xform is not None:
            return MockXformAttr(self._xform)
        return None


class MockXformable:
    """Minimal ``UsdGeom.Xformable`` adapter for composed transforms."""

    def __init__(self, prim: MockPrim):
        self._prim = prim

    def __bool__(self) -> bool:
        """Return whether the prim provides a composed local transform."""
        return getattr(self._prim, "composed_xform", None) is not None

    def GetLocalTransformation(self) -> list[list[float]] | None:
        """Return the prim's composed local transform."""
        return getattr(self._prim, "composed_xform", None)


class MockBBoxCache:
    """Minimal ``UsdGeom.BBoxCache`` test double.

    Args:
        *_args: Constructor arguments accepted for API compatibility.
    """

    def __init__(self, *_args: object):
        pass

    def ComputeLocalBound(self, prim: MockPrim) -> MockBound:
        """Return a prim's configured local bound.

        Args:
            prim: Prim whose local bound is requested.

        Returns:
            Configured local bound.
        """
        return MockBound(*prim.local_bounds)

    def ComputeWorldBound(self, prim: MockPrim) -> MockBound:
        """Return a prim's configured world bound.

        Args:
            prim: Prim whose world bound is requested.

        Returns:
            Configured world bound.
        """
        return MockBound(*prim.world_bounds)


class MockStage:
    """Minimal USD stage test double.

    Args:
        prims: Mapping from prim paths to test doubles.
    """

    def __init__(self, prims: dict[str, MockPrim]):
        self._prims = prims
        self._invalid_prim = MockPrim(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            valid=False,
        )

    def GetPrimAtPath(self, path: str) -> MockPrim:
        """Return the prim at a path.

        Args:
            path: Prim path to resolve.

        Returns:
            Configured prim, or an invalid prim when the path is unknown.
        """
        return self._prims.get(path, self._invalid_prim)

    def add_prim(self, path: str, prim: MockPrim) -> None:
        """Register an additional prim on the stage.

        Args:
            path: Prim path to register.
            prim: Prim test double resolved by :meth:`GetPrimAtPath`.
        """
        self._prims[path] = prim


class MockContext:
    """Minimal ``omni.usd`` context test double.

    Args:
        stage: Stage returned by the context.
    """

    def __init__(self, stage: MockStage):
        self._stage = stage

    def get_stage(self) -> MockStage:
        """Return the configured stage.

        Returns:
            Stage test double.
        """
        return self._stage


class DefectsApiMockEnvironment:
    """Load the defects API against isolated Omniverse and USD test doubles."""

    def __init__(self) -> None:
        self._modules_patch: mock._patch_dict[str, types.ModuleType] | None = None

    def _purge_defects_modules(self) -> None:
        """Remove cached defects modules so each test imports a fresh API."""
        for module_name in tuple(sys.modules):
            if module_name == "defects_utils" or module_name.startswith(
                "defects_utils."
            ):
                sys.modules.pop(module_name)

    def start(self) -> tuple[types.ModuleType, MockPrim]:
        """Install test doubles and import a fresh defects API module.

        Returns:
            Imported API module and component prim test double.

        Raises:
            RuntimeError: If the environment has already been started.
        """
        component = MockPrim(((-2.0, -1.0, -0.2), (2.0, 1.0, 0.2)))
        paste_0 = MockPrim(
            ((-0.2, -0.5, -0.05), (0.2, 0.5, 0.05)),
            ((-1.7, -0.5, -0.25), (-1.3, 0.5, -0.15)),
        )
        paste_1 = MockPrim(
            ((-0.2, -0.5, -0.05), (0.2, 0.5, 0.05)),
            ((1.3, -0.5, -0.25), (1.7, 0.5, -0.15)),
        )
        stage = MockStage({"/paste_0": paste_0, "/paste_1": paste_1})

        usd_geom = types.ModuleType("pxr.UsdGeom")
        usd_geom.BBoxCache = MockBBoxCache
        usd_geom.Tokens = types.SimpleNamespace(default_="default")
        usd_geom.Xformable = MockXformable
        pxr = types.ModuleType("pxr")
        pxr.UsdGeom = usd_geom

        omni = types.ModuleType("omni")
        omni.__path__ = []
        omni_usd = types.ModuleType("omni.usd")
        omni_usd.get_context = lambda: MockContext(stage)
        omni.usd = omni_usd

        if self._modules_patch is not None:
            raise RuntimeError("Defects API mock environment is already started")

        modules_patch = mock.patch.dict(
            sys.modules,
            {
                "pxr": pxr,
                "pxr.UsdGeom": usd_geom,
                "omni": omni,
                "omni.usd": omni_usd,
            },
        )
        self._modules_patch = modules_patch
        modules_patch.start()
        self._purge_defects_modules()
        sys.path.insert(0, _STANDALONE_DIR)
        try:
            return importlib.import_module("defects_utils.api"), component
        except Exception:
            self.stop()
            raise

    def get_stage(self) -> MockStage:
        """Return the stage installed by :meth:`start`.

        Returns:
            Stage test double shared with the imported defects API.

        Raises:
            RuntimeError: If the environment has not been started.
        """
        if self._modules_patch is None:
            raise RuntimeError("Defects API mock environment is not started")
        return sys.modules["omni.usd"].get_context().get_stage()

    def stop(self) -> None:
        """Restore the process module and import-path state."""
        while _STANDALONE_DIR in sys.path:
            sys.path.remove(_STANDALONE_DIR)
        if self._modules_patch is not None:
            self._modules_patch.stop()
            self._modules_patch = None
