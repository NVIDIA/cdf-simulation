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

"""Tests for board metadata and legacy sidecar normalization."""

from __future__ import annotations

import unittest
from pathlib import Path

from cdf.errors import ValidationError
from pcb_aoi.board_metadata import (
    BoardMetadataCatalog,
    BoardMetadataRequest,
    BoardRecord,
    ComponentRecord,
    MetadataCapability,
    MetadataOrigin,
    MetadataSource,
    SidecarMetadataProvider,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_RESOURCE_DIR = _REPO_ROOT / "tests" / "functional" / "resources"
_SMT_SIDECAR = _RESOURCE_DIR / "SMT_0805_C18_coupon.components.json"
_THT_SIDECAR = _RESOURCE_DIR / "THT_TO254P762H300_U34_coupon.components.json"


class SidecarMetadataProviderTest(unittest.TestCase):
    """Verify v1/v1.1 sidecars converge on the canonical catalog."""

    def test_v1_sidecar_normalizes_without_v11_fields(self) -> None:
        """Normalize v1 records when optional v1.1 fields are absent."""
        data = {
            "schema_version": "v1",
            "board_name": "Example",
            "board_prim_path": "/Example",
            "components": [
                {
                    "name": "R_0805",
                    "family": "resistor",
                    "source": "R_0805.usdc",
                    "prototype_path": "/Example/Prototypes/R_0805",
                    "side": "Top",
                    "prim_path": "/Example/Components/R_0805",
                    "instance_paths": ["/Example/Components/R1"],
                    "body_meshes": ["Body"],
                    "solder_contact_meshes": ["Pin1", "Pin2"],
                },
            ],
            "component_instances": [
                {
                    "path": "/Example/Components/R1",
                    "refdes": "R1",
                    "side": "Top",
                    "family": "resistor",
                    "type": "RES_0805",
                    "body_prims": ["/Example/Components/R1/Body"],
                    "solder_contact_prims": [
                        "/Example/Components/R1/Pin1",
                        "/Example/Components/R1/Pin2",
                    ],
                    "solder_paste_prims": [],
                    "solder_paste_usdc_paths": [],
                },
            ],
        }

        catalog = SidecarMetadataProvider().load(
            BoardMetadataRequest(source_identifier="memory", sidecar_data=data),
        )

        component = catalog.component(board_root="/Example", refdes="R1")
        self.assertIsNotNone(component)
        assert component is not None
        self.assertEqual(component.process_classes, ())
        self.assertIsNone(component.process_class)
        self.assertEqual(len(catalog.terminals_for_component(component)), 2)

    @unittest.skipUnless(_SMT_SIDECAR.is_file(), "approved SMT coupon fixture is not shipped in the public release")
    def test_smt_sidecar_populates_queries_and_capabilities(self) -> None:
        """Populate component queries and SMT-related capabilities."""
        path = _SMT_SIDECAR

        catalog = SidecarMetadataProvider().from_path(path)

        component = catalog.component(
            board_root="/Board_CAD_BOARD",
            refdes="C18",
        )
        self.assertIsNotNone(component)
        assert component is not None
        self.assertEqual(component.package_name, "PASSIVE_SMD_0805")
        self.assertEqual(component.process_classes, ("SMT",))
        self.assertEqual(component.process_class, "SMT")
        self.assertEqual(len(component.body_prim_paths), 7)
        self.assertEqual(len(catalog.terminals_for_component(component)), 2)
        self.assertEqual(len(catalog.paste_for_component(component)), 2)
        self.assertIn(MetadataCapability.COMPONENTS, catalog.capabilities)
        self.assertIn(MetadataCapability.SOLDER_CONTACTS, catalog.capabilities)
        self.assertIn(
            MetadataCapability.SOLDER_PASTE_ASSOCIATIONS,
            catalog.capabilities,
        )
        self.assertEqual(catalog.components_by_type("PASSIVE_0805"), (component,))
        self.assertIs(catalog.by_prim_path(component.prim_path), component)

    @unittest.skipUnless(_THT_SIDECAR.is_file(), "approved THT coupon fixture is not shipped in the public release")
    def test_tht_sidecar_preserves_pin_to_pad_associations(self) -> None:
        """Preserve through-hole pin-to-pad association data."""
        path = _THT_SIDECAR

        catalog = SidecarMetadataProvider().from_path(path)
        component = catalog.component(
            board_root="/Board_CAD_BOARD",
            refdes="U34",
        )

        self.assertIsNotNone(component)
        assert component is not None
        self.assertEqual(component.process_classes, ("through_hole",))
        self.assertEqual(len(catalog.terminals_for_component(component)), 8)
        self.assertEqual(len(catalog.pads_for_component(component)), 8)
        self.assertIn(
            MetadataCapability.TERMINAL_PAD_ASSOCIATIONS,
            catalog.capabilities,
        )
        self.assertTrue(
            all(
                item.contact_prim_path for item in catalog.pads_for_component(component)
            ),
        )
        self.assertTrue(
            all(
                item.pin_center is not None
                for item in catalog.pads_for_component(component)
            ),
        )

    def test_catalog_rejects_duplicate_component_identity(self) -> None:
        """Reject duplicate reference designators within one board root."""
        source = MetadataSource(MetadataOrigin.SIDECAR, "memory", "v1")
        board = BoardRecord("/Board", "board", source)
        component = ComponentRecord("/Board", "C1", "/Board/C1", source)

        with self.assertRaisesRegex(ValidationError, "duplicate component identity"):
            BoardMetadataCatalog(
                boards=(board,),
                components=(
                    component,
                    ComponentRecord("/Board", "C1", "/Board/OtherC1", source),
                ),
            )

    def test_provider_rejects_unknown_schema(self) -> None:
        """Reject sidecars that declare an unsupported schema version."""
        with self.assertRaisesRegex(ValidationError, "unsupported.*schema_version"):
            SidecarMetadataProvider().load(
                request=BoardMetadataRequest(
                    sidecar_data={
                        "schema_version": "v99",
                        "board_prim_path": "/Board",
                    },
                ),
            )


if __name__ == "__main__":
    unittest.main()
