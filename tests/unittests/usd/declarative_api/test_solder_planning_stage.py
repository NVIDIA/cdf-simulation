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

"""Solder planning against a live USD stage.

These cases need a real ``usd-core`` runtime: they read pin throats and
pad associations from an authored stage, and bind materials and semantics
onto the generated solder mesh. The stage-free solder planning tests live in
``tests/unittests/core/declarative_api/test_solder_planning.py``.
"""

import json
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pcb_aoi.board_metadata import (
    BoardMetadataRequest,
    SidecarMetadataProvider,
)
from pcb_aoi.solder_generation import (
    ProceduralSolderSolver,
    SolderCoverageStatus,
    SolderGenerationExtension,
    SolderGenerationRequest,
    discover_solder_components,
)
from pcb_aoi.solder_generation.procedural_tht_stage_analysis import (
    prepare_through_hole_site,
)


def _catalog(manifest, identifier="components.json"):
    data = dict(manifest)
    data.setdefault("schema_version", "v1")
    data.setdefault("board_name", "Board")
    data.setdefault("board_prim_path", "/Board")
    return SidecarMetadataProvider().load(
        BoardMetadataRequest(
            source_identifier=identifier,
            sidecar_data=data,
        ),
    )


class SolderPlanningStageTest(unittest.TestCase):
    """Solder planning steps that read from or author onto a USD stage."""

    def test_prepare_through_hole_site_extracts_live_square_pin_throat(self):
        """Read an offset square pin's throat and its pad radius from the stage."""
        try:
            from pxr import Gf, Usd, UsdGeom
        except ModuleNotFoundError:
            self.skipTest("usd-core not installed (CI installs numpy only)")

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 0.001)
        UsdGeom.Xform.Define(stage, "/Board")
        UsdGeom.Xform.Define(stage, "/Board/Components")
        pin = UsdGeom.Mesh.Define(stage, "/Board/Components/SquarePin")
        pin.CreatePointsAttr(
            [
                Gf.Vec3f(0.0, -0.3, -1.0),
                Gf.Vec3f(0.4, -0.3, -1.0),
                Gf.Vec3f(0.4, 0.1, -1.0),
                Gf.Vec3f(0.0, 0.1, -1.0),
                Gf.Vec3f(0.0, -0.3, 0.5),
                Gf.Vec3f(0.4, -0.3, 0.5),
                Gf.Vec3f(0.4, 0.1, 0.5),
                Gf.Vec3f(0.0, 0.1, 0.5),
            ],
        )
        pin.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
        pin.CreateFaceVertexIndicesAttr(
            [
                0,
                3,
                2,
                1,
                4,
                5,
                6,
                7,
                0,
                1,
                5,
                4,
                1,
                2,
                6,
                5,
                2,
                3,
                7,
                6,
                3,
                0,
                4,
                7,
            ],
        )
        pin.CreateExtentAttr(
            [Gf.Vec3f(0.0, -0.3, -1.0), Gf.Vec3f(0.4, 0.1, 0.5)],
        )
        UsdGeom.Xform.Define(stage, "/Board/Pads")
        pad = UsdGeom.Cube.Define(stage, "/Board/Pads/P1")
        pad.AddScaleOp().Set(Gf.Vec3f(0.8, 0.8, 0.05))
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            ["default", "render", "proxy"],
            useExtentsHint=True,
        )

        site = prepare_through_hole_site(
            stage,
            {
                "solder_contact_prim": "/Board/Components/SquarePin",
                "solder_pad_prim": "/Board/Pads/P1",
                "pin_center": (0.0, 0.0),
            },
            "Top",
            1.0,
            cache,
            solder_bath=True,
        )

        self.assertIsNotNone(site)
        assert site is not None
        self.assertEqual(site["center_xy"], (0.0, 0.0))
        self.assertAlmostEqual(site["throat_center_xy"][0], 0.2)
        self.assertAlmostEqual(site["throat_center_xy"][1], -0.1)
        self.assertEqual(len(site["throat_shape_xy"]), 4)
        self.assertAlmostEqual(site["pad_radius"], 0.8)
        self.assertEqual(len(site["pin_points"]), 8)
        self.assertEqual(site["pin_face_vertex_counts"], (4, 4, 4, 4, 4, 4))
        self.assertEqual(len(site["pin_face_vertex_indices"]), 24)
        self.assertEqual(
            set(site["pad_shape_xy"]),
            {(-0.8, -0.8), (0.8, -0.8), (0.8, 0.8), (-0.8, 0.8)},
        )

    @unittest.skipUnless(
        (
            _REPO_ROOT
            / "tests"
            / "functional"
            / "resources"
            / "ARTIX_PINHEADER_J13_coupon.usdc"
        ).is_file(),
        "approved THT coupon fixture is not shipped in the public release",
    )
    def test_procedural_prepare_consumes_tht_coupon_associations(self):
        """Build one through-hole site per coupon pin-pad association."""
        try:
            from pxr import Usd

            from pcb_aoi.board_metadata.usd_schema import UsdSchemaMetadataProvider
        except ModuleNotFoundError:
            self.skipTest("usd-core not installed (CI installs numpy only)")

        resource_dir = _REPO_ROOT / "tests" / "functional" / "resources"
        coupon_path = resource_dir / "ARTIX_PINHEADER_J13_coupon.usdc"
        stage = Usd.Stage.Open(str(coupon_path))
        catalog = UsdSchemaMetadataProvider().load(
            BoardMetadataRequest(
                stage=stage,
                board_root="/Board",
                source_identifier=str(coupon_path),
            ),
        )
        request = SolderGenerationRequest(
            request_id="/Board/SolderJoints@procedural",
            solver_id="procedural",
            source_usd_path=str(coupon_path),
            root_path="/Board",
            output_root_path="/Board/SolderJoints",
            stage=stage,
            board_metadata=catalog,
            candidates=discover_solder_components(
                catalog,
                require_sample_paste=False,
            ),
            defect_records=(),
        )
        extension = SolderGenerationExtension()
        solver = ProceduralSolderSolver()

        prepared = solver.prepare(request, extension.build_work_items(request))
        generated = solver.generate(prepared)

        sites = prepared.payload["sites_by_component"][
            "/Board/Layers/Top/Components/J13"
        ]["sites"]
        self.assertEqual(len(sites), 3)
        self.assertEqual(len(generated.artifacts), 1)
        self.assertEqual(len(generated.artifacts[0].meshes), 3)
        self.assertEqual(
            generated.coverage[0].status,
            SolderCoverageStatus.GENERATED,
        )

    def test_procedural_prepare_extracts_pad_aware_numeric_inputs(self):
        """Turn a paste site's pad and lead contact into numeric solver inputs."""
        try:
            from pxr import Gf, Usd, UsdGeom
        except ModuleNotFoundError:
            self.skipTest("usd-core not installed (CI installs numpy only)")

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 0.001)
        UsdGeom.Xform.Define(stage, "/Board")
        UsdGeom.Xform.Define(stage, "/Board/Components")
        UsdGeom.Xform.Define(stage, "/Board/Components/Top")
        UsdGeom.Xform.Define(stage, "/Board/Components/Top/C1")
        body = UsdGeom.Cube.Define(stage, "/Board/Components/Top/C1/body")
        body.AddScaleOp().Set(Gf.Vec3f(1.2, 0.8, 0.7))
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.8))
        contact = UsdGeom.Cube.Define(stage, "/Board/Components/Top/C1/lead")
        contact.AddScaleOp().Set(Gf.Vec3f(1.4, 0.7, 0.3))
        contact.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.3))
        contact_mesh = UsdGeom.Mesh.Define(
            stage,
            "/Board/Components/Top/C1/lead/mesh",
        )
        contact_mesh.CreatePointsAttr(
            [
                Gf.Vec3f(0.7, -0.2, 0.0),
                Gf.Vec3f(0.7, 0.2, 0.0),
                Gf.Vec3f(0.9, -0.2, 0.0),
                Gf.Vec3f(0.9, 0.2, 0.0),
            ],
        )
        contact_mesh.CreateExtentAttr(
            [Gf.Vec3f(0.7, -0.2, 0.0), Gf.Vec3f(0.9, 0.2, 0.0)],
        )
        UsdGeom.Xform.Define(stage, "/Board/Paste")
        paste = UsdGeom.Cube.Define(stage, "/Board/Paste/p1")
        paste.AddScaleOp().Set(Gf.Vec3f(0.5, 0.9, 0.1))
        paste.AddTranslateOp().Set(Gf.Vec3d(1.5, 0.0, 0.1))

        manifest = {
            "components": [
                {
                    "name": "PASSIVE_SMD_0805",
                    "side": "Top",
                    "instance_paths": ["/Board/Components/Top/C1"],
                    "solder_contact_meshes": ["lead"],
                },
            ],
            "component_instances": [
                {
                    "path": "/Board/Components/Top/C1",
                    "refdes": "C1",
                    "body_prims": ["/Board/Components/Top/C1/body"],
                    "solder_contact_prims": ["/Board/Components/Top/C1/lead"],
                    "solder_paste_prims": ["/Board/Paste/p1"],
                },
            ],
        }
        catalog = _catalog(manifest)
        request = SolderGenerationRequest(
            request_id="/Board/SolderJoints@procedural",
            solver_id="procedural",
            source_usd_path="board.usd",
            root_path="/Board",
            output_root_path="/Board/SolderJoints",
            stage=stage,
            board_metadata=catalog,
            candidates=discover_solder_components(catalog),
            defect_records=(),
        )
        extension = SolderGenerationExtension()
        solver = ProceduralSolderSolver()

        prepared = solver.prepare(request, extension.build_work_items(request))
        generated = solver.generate(prepared)

        component_input = prepared.payload["sites_by_component"][
            "/Board/Components/Top/C1"
        ]
        self.assertEqual(component_input["site_kind"], "paste")
        self.assertEqual(len(component_input["sites"]), 1)
        self.assertEqual(
            component_input["sites"][0]["material_source_prim_path"],
            "/Board/Paste/p1",
        )
        self.assertEqual(
            generated.artifacts[0].meshes[0].material_source_prim_path,
            "/Board/Paste/p1",
        )
        self.assertEqual(len(component_input["lead_contacts"]), 1)
        self.assertIsNotNone(component_input["lead_contacts"][0])
        self.assertFalse(component_input["area_array"])
        json.dumps(prepared.payload)
        self.assertEqual(prepared.payload["scene_units_per_mm"], 1.0)
        self.assertEqual(
            generated.coverage[0].status,
            SolderCoverageStatus.GENERATED,
        )
        self.assertGreater(len(generated.artifacts[0].meshes[0].points), 400)

    def test_generated_mesh_reuses_source_composed_material(self):
        """Bind the material bound to the source paste prim to the new mesh."""
        try:
            from pxr import Usd, UsdGeom, UsdShade
        except ModuleNotFoundError:
            self.skipTest("usd-core not installed (CI installs numpy only)")

        stage = Usd.Stage.CreateInMemory()
        material = UsdShade.Material.Define(stage, "/Board/Looks/solder")
        source = UsdGeom.Mesh.Define(stage, "/Board/Paste/p1").GetPrim()
        destination_path = "/Board/SolderJoints/joint_000"
        UsdGeom.Mesh.Define(stage, destination_path)
        UsdShade.MaterialBindingAPI.Apply(source).Bind(material)

        bound_path = SolderGenerationExtension._reuse_composed_material(
            stage,
            destination_path,
            "/Board/Paste/p1",
        )

        destination = stage.GetPrimAtPath(destination_path)
        bound_material, _relationship = UsdShade.MaterialBindingAPI(
            destination,
        ).ComputeBoundMaterial()
        self.assertEqual(bound_path, "/Board/Looks/solder")
        self.assertEqual(str(bound_material.GetPath()), "/Board/Looks/solder")

    def test_generated_mesh_binds_direct_material_source(self):
        """Bind the source path itself when it names a material."""
        try:
            from pxr import Usd, UsdGeom, UsdShade
        except ModuleNotFoundError:
            self.skipTest("usd-core not installed (CI installs numpy only)")

        stage = Usd.Stage.CreateInMemory()
        UsdShade.Material.Define(stage, "/Board/Looks/solder")
        destination_path = "/Board/SolderJoints/joint_000"
        UsdGeom.Mesh.Define(stage, destination_path)

        bound_path = SolderGenerationExtension._reuse_composed_material(
            stage,
            destination_path,
            "/Board/Looks/solder",
        )

        destination = stage.GetPrimAtPath(destination_path)
        bound_material, _relationship = UsdShade.MaterialBindingAPI(
            destination,
        ).ComputeBoundMaterial()
        self.assertEqual(bound_path, "/Board/Looks/solder")
        self.assertEqual(str(bound_material.GetPath()), "/Board/Looks/solder")

    def test_generated_mesh_falls_back_to_packaged_solder_material(self):
        """Reference the packaged solder material once and reuse it for later meshes."""
        try:
            from pxr import Usd, UsdGeom, UsdShade
        except ModuleNotFoundError:
            self.skipTest("usd-core not installed (CI installs numpy only)")
        from types import SimpleNamespace

        stage = Usd.Stage.CreateInMemory()
        destination_path = "/Board/SolderJoints/joint_000"
        UsdGeom.Mesh.Define(stage, destination_path)
        second_path = "/Board/SolderJoints/joint_001"
        UsdGeom.Mesh.Define(stage, second_path)
        request = SimpleNamespace(output_root_path="/Board/SolderJoints")

        bound_path = SolderGenerationExtension._bind_packaged_solder_material(
            stage,
            request,
            destination_path,
        )

        self.assertEqual(bound_path, "/Board/SolderJoints/Looks/Solder")
        destination = stage.GetPrimAtPath(destination_path)
        bound_material, _relationship = UsdShade.MaterialBindingAPI(
            destination,
        ).ComputeBoundMaterial()
        self.assertEqual(
            str(bound_material.GetPath()),
            "/Board/SolderJoints/Looks/Solder",
        )
        # The packaged asset composed through the reference: the preview
        # shader exists and is metallic.
        shader = stage.GetPrimAtPath(
            "/Board/SolderJoints/Looks/Solder/PreviewSurface",
        )
        self.assertTrue(shader and shader.IsValid())
        self.assertEqual(shader.GetAttribute("inputs:metallic").Get(), 1.0)
        # Second mesh reuses the once-referenced material prim.
        again = SolderGenerationExtension._bind_packaged_solder_material(
            stage,
            request,
            second_path,
        )
        self.assertEqual(again, bound_path)

    def test_generated_solder_copies_component_semantics(self):
        """Copy the component's semantic labels, but not its refdes, onto the solder."""
        try:
            from pxr import Sdf, Usd, UsdGeom
        except ModuleNotFoundError:
            self.skipTest("usd-core not installed (CI installs numpy only)")

        stage = Usd.Stage.CreateInMemory()
        source = UsdGeom.Xform.Define(stage, "/Board/Components/K2010").GetPrim()
        destination = UsdGeom.Xform.Define(
            stage,
            "/Board/SolderJoints/K2010",
        ).GetPrim()
        source.CreateAttribute(
            "semantic:Semantics:class:semanticData",
            Sdf.ValueTypeNames.String,
        ).Set("K2010")
        source.CreateAttribute(
            "semantic:Semantics:class:semanticType",
            Sdf.ValueTypeNames.String,
        ).Set("class")
        source.CreateAttribute(
            "pcba:component:refdes",
            Sdf.ValueTypeNames.String,
        ).Set("K2010")

        copied = SolderGenerationExtension._copy_semantics(source, destination)

        self.assertEqual(copied, 2)
        self.assertEqual(
            destination.GetAttribute(
                "semantic:Semantics:class:semanticData",
            ).Get(),
            "K2010",
        )
        self.assertFalse(destination.HasAttribute("pcba:component:refdes"))


if __name__ == "__main__":
    unittest.main()
