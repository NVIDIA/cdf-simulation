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

"""Tests for composed PCBA USD metadata catalog loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcb_aoi.board_metadata import BoardMetadataRequest, MetadataCapability
from pcb_aoi.solder_generation import discover_solder_components

try:
    from pxr import Sdf, Usd

    from pcb_aoi.board_metadata.usd_schema import UsdSchemaMetadataProvider
    from pcb_aoi.board_metadata.usd_schema.catalog_inspector import main
except ModuleNotFoundError:
    Usd = None
    UsdSchemaMetadataProvider = None
    main = None

_REPO_ROOT = Path(__file__).resolve().parents[5]
_RESOURCE_DIR = _REPO_ROOT / "tests" / "functional" / "resources"
_SMT_COUPON = _RESOURCE_DIR / "ICT_C0805_C22_coupon.usdc"
_POLAR_COUPON = _RESOURCE_DIR / "ARTIX_LED_D1_coupon.usdc"


@unittest.skipUnless(Usd is not None, "usd-core is not installed")
class UsdSchemaMetadataProviderTest(unittest.TestCase):
    """Verify composed PCBA USD metadata normalizes consistently."""

    def test_instance_proxy_preserves_mixed_process_metadata(self) -> None:
        """Load instance-proxy terminals and detailed process classes."""
        assert Usd is not None
        stage = Usd.Stage.CreateInMemory()
        board = stage.DefinePrim("/Board")
        stage.SetDefaultPrim(board)
        prototype = stage.DefinePrim("/Prototypes/USB_C")
        prototype.CreateAttribute(
            "pcba:component:refdes",
            Sdf.ValueTypeNames.String,
        ).Set("J1")
        prototype.CreateAttribute(
            "pcba:component:type",
            Sdf.ValueTypeNames.Token,
        ).Set("connector")
        prototype.CreateAttribute(
            "pcba:component:processClass",
            Sdf.ValueTypeNames.Token,
        ).Set("Mixed")
        prototype.CreateAttribute(
            "pcba:component:processClasses",
            Sdf.ValueTypeNames.TokenArray,
        ).Set(["through_hole", "SMT"])
        prototype.CreateAttribute(
            "pcba:component:packageType",
            Sdf.ValueTypeNames.String,
        ).Set("USB-C-receptacle")
        prototype.CreateAttribute(
            "pcba:component:footprintId",
            Sdf.ValueTypeNames.String,
        ).Set("USB_C")
        prototype.CreateAttribute(
            "pcba:component:side",
            Sdf.ValueTypeNames.Token,
        ).Set("top")
        terminal = stage.DefinePrim("/Prototypes/USB_C/pin_1")
        terminal.CreateAttribute(
            "pcba:pin:number",
            Sdf.ValueTypeNames.Int,
        ).Set(1)
        component = stage.DefinePrim("/Board/Components/J1")
        component.GetReferences().AddInternalReference(prototype.GetPath())
        component.SetInstanceable(True)

        catalog = UsdSchemaMetadataProvider().load(
            BoardMetadataRequest(stage=stage),
        )

        self.assertEqual(len(catalog.components), 1)
        self.assertEqual(len(catalog.terminals), 1)
        record = catalog.components[0]
        self.assertEqual(record.process_class, "Mixed")
        self.assertEqual(record.process_classes, ("through_hole", "SMT"))
        self.assertEqual(record.package_type, "USB-C-receptacle")

    @unittest.skipUnless(
        _POLAR_COUPON.is_file(),
        "approved polarized coupon fixture is not shipped in the public release",
    )
    def test_maintained_coupon_loads_string_package_and_baked_meshes(self) -> None:
        """Load string package metadata from a self-contained Artix coupon."""
        assert Usd is not None
        stage = Usd.Stage.Open(str(_POLAR_COUPON))
        catalog = UsdSchemaMetadataProvider().load(
            BoardMetadataRequest(stage=stage, source_identifier=str(_POLAR_COUPON)),
        )

        self.assertEqual(stage.GetRootLayer().GetExternalReferences(), ())
        self.assertEqual(len(catalog.components), 1)
        self.assertEqual(len(catalog.terminals), 2)
        self.assertEqual(catalog.components[0].refdes, "D1")
        self.assertEqual(catalog.components[0].package_type, "0603")
        self.assertEqual(catalog.components[0].footprint_id, "LED_0603_1608Metric_G")
        self.assertTrue(any(prim.GetTypeName() == "Mesh" for prim in stage.Traverse()))

    @unittest.skipUnless(
        _SMT_COUPON.is_file(),
        "approved SMT coupon fixture is not shipped in the public release",
    )
    def test_maintained_ict_coupon_loads_smt_associations(self) -> None:
        """Discover terminals and paste from the maintained ICT coupon."""
        assert Usd is not None
        stage = Usd.Stage.Open(str(_SMT_COUPON))
        catalog = UsdSchemaMetadataProvider().load(
            BoardMetadataRequest(stage=stage, source_identifier=str(_SMT_COUPON)),
        )

        component = catalog.component(board_root="/Board", refdes="C22")
        self.assertIsNotNone(component)
        assert component is not None
        self.assertEqual(component.process_class, "SMT")
        self.assertEqual(len(component.contact_prim_paths), 2)
        self.assertEqual(len(catalog.paste_for_component(component)), 2)
        self.assertIn(
            MetadataCapability.SOLDER_PASTE_ASSOCIATIONS,
            catalog.capabilities,
        )
        candidates = discover_solder_components(catalog)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].sample_refdes, "C22")

    def test_an_unassociated_hole_is_still_available(self) -> None:
        """A hole belongs to the board whether or not a terminal claims it.

        This is the case the capability split exists for. Mapping `holes` to
        TERMINAL_HOLE_ASSOCIATIONS made a hole reachable only once something
        had claimed it, so the unassociated ones -- the motivating case --
        could never be labelled at all.
        """
        assert Usd is not None
        stage = Usd.Stage.CreateInMemory()
        board = stage.DefinePrim("/Board")
        stage.SetDefaultPrim(board)
        layer = stage.DefinePrim("/Board/Layers/Holes")
        layer.CreateAttribute("pcba:layer:side", Sdf.ValueTypeNames.Token).Set(
            "top",
        )
        hole = stage.DefinePrim("/Board/Layers/Holes/Hole_1")
        hole.CreateAttribute("pcba:mesh:role", Sdf.ValueTypeNames.Token).Set(
            "board_hole",
        )
        # Deliberately no refdes and no pin number: nothing claims this hole.
        hole.CreateAttribute(
            "pcba:board:holeCenterMm",
            Sdf.ValueTypeNames.Float2,
        ).Set((1.0, 2.0))

        catalog = UsdSchemaMetadataProvider().load(
            BoardMetadataRequest(stage=stage, board_root="/Board"),
        )

        self.assertEqual(len(catalog.holes), 1)
        self.assertIsNone(catalog.holes[0].terminal_number)
        self.assertIn(MetadataCapability.HOLES, catalog.capabilities)
        self.assertNotIn(
            MetadataCapability.TERMINAL_HOLE_ASSOCIATIONS,
            catalog.capabilities,
        )

    def test_flattened_over_root_loads_defined_component_children(self) -> None:
        """Load components below an undefined flattened-prototype root."""
        assert UsdSchemaMetadataProvider is not None
        layer = Sdf.Layer.CreateAnonymous()
        root_spec = Sdf.CreatePrimInLayer(layer, "/Flattened_Prototype_1")
        root_spec.specifier = Sdf.SpecifierOver
        component_spec = Sdf.CreatePrimInLayer(
            layer,
            "/Flattened_Prototype_1/Components/J1",
        )
        component_spec.specifier = Sdf.SpecifierDef
        stage = Usd.Stage.Open(layer)
        root = stage.GetPrimAtPath("/Flattened_Prototype_1")
        self.assertFalse(root.IsDefined())
        component = stage.GetPrimAtPath(
            "/Flattened_Prototype_1/Components/J1",
        )
        component.CreateAttribute(
            "pcba:component:refdes",
            Sdf.ValueTypeNames.String,
        ).Set("J1")
        component.CreateAttribute(
            "pcba:component:footprintId",
            Sdf.ValueTypeNames.String,
        ).Set("TEST_FOOTPRINT")

        catalog = UsdSchemaMetadataProvider().load(
            BoardMetadataRequest(
                stage=stage,
                board_root="/Flattened_Prototype_1",
            ),
        )

        self.assertEqual(len(catalog.components), 1)
        self.assertEqual(catalog.components[0].refdes, "J1")
        self.assertEqual(catalog.components[0].package_name, "TEST_FOOTPRINT")

    def test_schema_native_refdes_resolves_differently_named_pad_and_hole(
        self,
    ) -> None:
        """Resolve schema-native board feature metadata without legacy fields."""
        assert Usd is not None
        stage = Usd.Stage.CreateInMemory()
        board = stage.DefinePrim("/Board")
        stage.SetDefaultPrim(board)
        component = stage.DefinePrim("/Board/Components/J1")
        component.CreateAttribute(
            "pcba:component:refdes", Sdf.ValueTypeNames.String
        ).Set(
            "J1",
        )
        component.CreateAttribute(
            "pcba:component:side",
            Sdf.ValueTypeNames.Token,
        ).Set("top")
        terminal = stage.DefinePrim("/Board/Components/J1/Pin1")
        terminal.CreateAttribute("pcba:pin:number", Sdf.ValueTypeNames.String).Set(
            "1",
        )
        terminal.CreateAttribute(
            "pcba:pin:crossSection",
            Sdf.ValueTypeNames.Token,
        ).Set("rectangular")
        top_layer = stage.DefinePrim("/Board/Layers/Top")
        top_layer.CreateAttribute("pcba:layer:side", Sdf.ValueTypeNames.Token).Set(
            "top",
        )
        bottom_layer = stage.DefinePrim("/Board/Layers/Bottom")
        bottom_layer.CreateAttribute(
            "pcba:layer:side",
            Sdf.ValueTypeNames.Token,
        ).Set("bottom")
        pad = stage.DefinePrim("/Board/Layers/Top/Pad_100")
        pad.CreateAttribute("pcba:mesh:role", Sdf.ValueTypeNames.Token).Set(
            "board_pad",
        )
        pad.CreateAttribute("pcba:board:refdes", Sdf.ValueTypeNames.String).Set("J1")
        pad.CreateAttribute(
            "pcba:board:componentPinNumber",
            Sdf.ValueTypeNames.Int,
        ).Set(1)
        bottom_pad = stage.DefinePrim("/Board/Layers/Bottom/Pad_200")
        bottom_pad.CreateAttribute("pcba:mesh:role", Sdf.ValueTypeNames.Token).Set(
            "board_pad",
        )
        bottom_pad.CreateAttribute(
            "pcba:board:refdes",
            Sdf.ValueTypeNames.String,
        ).Set("J1")
        bottom_pad.CreateAttribute(
            "pcba:board:componentPinNumber",
            Sdf.ValueTypeNames.Int,
        ).Set(1)
        bottom_pad.CreateAttribute(
            "pcba:board:padContour",
            Sdf.ValueTypeNames.Point3fArray,
        ).Set(
            [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0)],
        )
        hole = stage.DefinePrim("/Board/Layers/Holes/Hole_200")
        hole.CreateAttribute("pcba:mesh:role", Sdf.ValueTypeNames.Token).Set(
            "board_hole",
        )
        hole.CreateAttribute("pcba:board:refdes", Sdf.ValueTypeNames.String).Set(
            "J1",
        )
        hole.CreateAttribute(
            "pcba:board:componentPinNumber",
            Sdf.ValueTypeNames.Int,
        ).Set(1)
        hole.CreateAttribute(
            "pcba:board:holeCenterMm",
            Sdf.ValueTypeNames.Float2,
        ).Set((0.0, 0.0))
        hole.CreateAttribute(
            "pcba:board:holeDimsMm",
            Sdf.ValueTypeNames.Float2,
        ).Set((0.8, 0.8))
        hole.CreateAttribute(
            "pcba:board:holeShape",
            Sdf.ValueTypeNames.Token,
        ).Set("circular")
        hole.CreateRelationship("pcba:board:component").SetTargets(
            [component.GetPath()],
        )

        catalog = UsdSchemaMetadataProvider().load(
            BoardMetadataRequest(stage=stage),
        )

        self.assertEqual(catalog.pads[0].component_path, str(component.GetPath()))
        self.assertEqual(catalog.pads[0].terminal_number, "1")
        self.assertEqual(catalog.pads[0].contact_prim_path, str(terminal.GetPath()))
        self.assertEqual(catalog.holes[0].component_path, str(component.GetPath()))
        self.assertEqual(catalog.holes[0].terminal_number, "1")
        self.assertEqual(catalog.holes[0].pad_prim_path, str(bottom_pad.GetPath()))
        self.assertEqual(catalog.terminals[0].cross_section, "rectangular")
        bottom_pad_record = catalog.by_prim_path(str(bottom_pad.GetPath()))
        self.assertIsNotNone(bottom_pad_record)
        assert bottom_pad_record is not None
        self.assertTrue(bottom_pad_record.contour)
        self.assertEqual(catalog.holes[0].center_mm, (0.0, 0.0))
        # Authored through a USD `float` attribute, so 0.8 reads back as
        # 0.800000011920929. Comparing the tuple exactly asserted that
        # float32 is lossless, which it is not.
        for measured, expected in zip(
            catalog.holes[0].dimensions_mm, (0.8, 0.8), strict=True
        ):
            self.assertAlmostEqual(measured, expected, places=6)
        self.assertEqual(catalog.tht_geometry_metadata_gaps(bottom_pad_record), ())

    def test_abstract_prototype_root_loads_component_metadata(self) -> None:
        """Load components authored below an internal prototype class."""
        assert Usd is not None
        stage = Usd.Stage.CreateInMemory()
        prototype = stage.CreateClassPrim("/World/_class_Boards/Board")
        component = stage.DefinePrim(
            "/World/_class_Boards/Board/Components/C1",
        )
        component.CreateAttribute(
            "pcba:component:refdes",
            Sdf.ValueTypeNames.String,
        ).Set("C1")
        component.CreateAttribute(
            "pcba:component:side",
            Sdf.ValueTypeNames.Token,
        ).Set("top")

        catalog = UsdSchemaMetadataProvider().load(
            BoardMetadataRequest(stage=stage, board_root=str(prototype.GetPath())),
        )

        self.assertEqual(len(catalog.components), 1)
        self.assertEqual(catalog.components[0].refdes, "C1")
        self.assertEqual(catalog.components[0].board_root, str(prototype.GetPath()))

    @unittest.skipUnless(
        _SMT_COUPON.is_file(),
        "approved SMT coupon fixture is not shipped in the public release",
    )
    def test_catalog_inspector_writes_fully_loaded_json(self) -> None:
        """Export a maintained coupon through the command-line entry point."""
        assert main is not None
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "ict-c22.metadata.json"

            result = main([str(_SMT_COUPON), "-o", str(output_path)])

            document = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result, 0)
            self.assertEqual(len(document["components"]), 1)
            self.assertEqual(len(document["terminals"]), 2)
            self.assertEqual(document["boards"][0]["source"]["origin"], "usd")
