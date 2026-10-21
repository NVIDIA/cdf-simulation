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

"""Runtime behaviour of traversal-resolved geometry annotation targets.

The role-name table is domain vocabulary and is asserted elsewhere. What is
asserted here is what the executor actually does with it on a stage: that the
requested class lands on the prims carrying the role and on nothing else, that
``components=`` restricts labelling to the selected components' meshes, and
that prototype geometry reaches the instanced projection while board geometry
does not.

These exercise the registered geometry-role hook rather than a hard-coded
package list, so a wrong ownership or projection declaration fails here.
"""

from __future__ import annotations

import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

from cdf.execution.isaac.execution_context import IsaacExecutionContext
from cdf.execution.isaac.job_order_parameters import GeometryAnnotationParameters
from cdf.execution.isaac.job_order_runtime import IsaacSdgRuntime

try:
    from pxr import Sdf, Usd, UsdGeom
except ImportError:  # pragma: no cover - lane without usd-core
    Usd = None

# Importing the domain registers its Isaac handlers and geometry roles.
import pcb_aoi  # noqa: F401
from pcb_aoi.board_metadata.usd_schema_metadata_keys import (
    MESH_ROLE_ATTR,
    USD_SCHEMA_METADATA_KEYS,
)

_COMP_REL = USD_SCHEMA_METADATA_KEYS["board_feature.component_relationship"][0]
_SEMANTIC_ATTR = "semantic:Semantics:class:semanticData"

BOARD = "/World/Board"


def _semantic_class(stage: object, path: str) -> str | None:
    """Return the semantic class authored on a prim, or None."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    attr = prim.GetAttribute(_SEMANTIC_ATTR)
    return attr.Get() if attr else None


def _build_stage() -> object:
    """A board with two components, each carrying package body and marking.

    Also carries one barrel mesh, which is board geometry associated with a
    component by relationship rather than by enclosure -- the other half of the
    ownership contract the hook declares.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, BOARD)

    for refdes in ("C22", "R7"):
        comp = UsdGeom.Xform.Define(stage, f"{BOARD}/{refdes}").GetPrim()
        for role, name in (("pkg_body", "Body"), ("pkg_marking", "Marking")):
            mesh = UsdGeom.Mesh.Define(stage, f"{BOARD}/{refdes}/{name}").GetPrim()
            mesh.CreateAttribute(MESH_ROLE_ATTR, Sdf.ValueTypeNames.String).Set(role)
        # A lead mesh with no role must never be labelled.
        UsdGeom.Mesh.Define(stage, f"{BOARD}/{refdes}/Lead")
        comp.CreateAttribute("pcba:board:refdes", Sdf.ValueTypeNames.String).Set(refdes)

    # Barrel geometry lives outside any component prim and names its component
    # through a relationship.
    barrel = UsdGeom.Mesh.Define(stage, f"{BOARD}/Barrels/Barrel_0").GetPrim()
    barrel.CreateAttribute(MESH_ROLE_ATTR, Sdf.ValueTypeNames.String).Set(
        "board_barrel"
    )
    barrel.CreateRelationship(_COMP_REL).SetTargets([f"{BOARD}/C22"])
    return stage


def _context(stage: object) -> IsaacExecutionContext:
    context = IsaacExecutionContext(runtime=object(), stage=stage)
    context.scene = {"root_path": BOARD, "board_root_path": BOARD}
    return context


def _common_stub() -> ModuleType:
    """Stand in for the Isaac standalone ``common`` module.

    Only ``refresh_authored_semantics`` is reached from this path, and it
    re-reads what was already authored, so recording the call is enough.
    """
    common = ModuleType("common")
    common.refreshed = []
    common.refresh_authored_semantics = lambda _stage, paths: common.refreshed.append(
        list(paths),
    )
    return common


def _catalog(*components: tuple[str, str]) -> object:
    """A component catalog stub: (refdes, prim_path) pairs."""
    return SimpleNamespace(
        components=[
            SimpleNamespace(refdes=refdes, prim_path=path)
            for refdes, path in components
        ],
    )


@unittest.skipIf(Usd is None, "usd-core is not installed")
class GeometryRoleTraversalTest(unittest.TestCase):
    """What the executor does with role-resolved geometry on a real stage."""

    def _apply(
        self,
        stage: object,
        parameters: GeometryAnnotationParameters,
        *,
        context: IsaacExecutionContext | None = None,
        catalog: object | None = None,
    ) -> dict[str, object]:
        runtime = object.__new__(IsaacSdgRuntime)
        context = context or _context(stage)
        common = _common_stub()
        with (
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
                return_value=common,
            ),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._board_metadata",
                return_value=catalog or _catalog(),
            ),
        ):
            return runtime.apply_geometry_annotations(context, parameters)

    def test_body_and_marking_prims_receive_their_requested_labels(self) -> None:
        stage = _build_stage()
        result = self._apply(
            stage,
            GeometryAnnotationParameters(body="package_body", marking="polarity"),
        )

        for refdes in ("C22", "R7"):
            self.assertEqual(
                _semantic_class(stage, f"{BOARD}/{refdes}/Body"),
                "package_body",
            )
            self.assertEqual(
                _semantic_class(stage, f"{BOARD}/{refdes}/Marking"),
                "polarity",
            )
            # A mesh carrying no role is not labelled by a role request.
            self.assertIsNone(_semantic_class(stage, f"{BOARD}/{refdes}/Lead"))

        # The barrel carries a different role and was not requested.
        self.assertIsNone(_semantic_class(stage, f"{BOARD}/Barrels/Barrel_0"))
        self.assertEqual(result["body"], 2)
        self.assertEqual(result["marking"], 2)
        self.assertEqual(result["body_records"], 2)

    def test_components_filter_restricts_labels_to_that_components_meshes(
        self,
    ) -> None:
        stage = _build_stage()
        result = self._apply(
            stage,
            GeometryAnnotationParameters(body="package_body", components=("C22",)),
            catalog=_catalog(("C22", f"{BOARD}/C22"), ("R7", f"{BOARD}/R7")),
        )

        self.assertEqual(
            _semantic_class(stage, f"{BOARD}/C22/Body"),
            "package_body",
        )
        # R7 is a component with the same role on it, excluded by the filter.
        self.assertIsNone(_semantic_class(stage, f"{BOARD}/R7/Body"))
        self.assertEqual(result["body"], 1)
        self.assertEqual(result["body_records"], 1)

    def test_barrel_ownership_follows_the_relationship_not_the_enclosure(
        self,
    ) -> None:
        """Barrels are declared relationship-owned, so enclosure must not decide.

        The barrel sits under ``/Board/Barrels``, which is not a component at
        all. If the executor resolved its owner by enclosure -- the rule the
        hook declares for package geometry -- it would find no component and
        drop the prim.
        """
        stage = _build_stage()
        result = self._apply(
            stage,
            GeometryAnnotationParameters(barrels="barrel", components=("C22",)),
            catalog=_catalog(("C22", f"{BOARD}/C22"), ("R7", f"{BOARD}/R7")),
        )

        self.assertEqual(
            _semantic_class(stage, f"{BOARD}/Barrels/Barrel_0"),
            "barrel",
        )
        self.assertEqual(result["barrels"], 1)

    def test_package_geometry_is_projected_for_an_instanced_cohort(self) -> None:
        """Prototype geometry reaches the projection; board geometry does not."""
        from pcb_aoi.execution.board_instances import InstanceSemanticProjection

        stage = _build_stage()
        context = _context(stage)
        cohort = SimpleNamespace(
            prototype_source_path=BOARD,
            instance_paths=("/World/Panel/Board_0",),
        )
        context.instance_semantic_projection = InstanceSemanticProjection(
            cohort=cohort,
            components=(),
        )

        result = self._apply(
            stage,
            GeometryAnnotationParameters(body="package_body", barrels="barrel"),
            context=context,
        )

        # body is declared projected=True by the hook; barrels is not.
        self.assertEqual(result["body_projected"], 2)
        self.assertNotIn("barrels_projected", result)
        projected = {
            record.source_path
            for record in context.instance_semantic_projection.components
        }
        self.assertIn(f"{BOARD}/C22/Body", projected)
        self.assertNotIn(f"{BOARD}/Barrels/Barrel_0", projected)


if __name__ == "__main__":
    unittest.main()
