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

"""Tests for source-neutral board metadata JSON output."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcb_aoi.board_metadata import (
    BoardMetadataCatalog,
    BoardRecord,
    MetadataCapability,
    MetadataOrigin,
    MetadataSource,
    catalog_to_json_data,
    write_catalog_json,
)


class CatalogJsonWriterTest(unittest.TestCase):
    """Verify catalogs serialize without their internal query indexes."""

    def test_catalog_serializes_public_records_and_provenance(self) -> None:
        """Serialize public records, extensions, and source provenance."""
        source = MetadataSource(
            origin=MetadataOrigin.USD,
            identifier="board.usda",
            schema_version="1.0",
        )
        catalog = BoardMetadataCatalog(
            boards=(
                BoardRecord(
                    root_path="/Board",
                    name="Example",
                    source=source,
                    extensions={"pcba:test": (1, 2)},
                ),
            ),
            capabilities=frozenset({MetadataCapability.COMPONENTS}),
        )

        document = catalog_to_json_data(catalog)

        self.assertEqual(document["format"], "pcb_aoi.board_metadata.catalog")
        self.assertEqual(document["capabilities"], ["components"])
        self.assertEqual(document["boards"][0]["source"]["origin"], "usd")
        self.assertEqual(document["boards"][0]["extensions"]["pcba:test"], [1, 2])
        self.assertNotIn("_by_prim_path", document)

    def test_writer_emits_indented_json_with_terminal_newline(self) -> None:
        """Write readable JSON with the repository text-file convention."""
        catalog = BoardMetadataCatalog(
            boards=(
                BoardRecord(
                    root_path="/Board",
                    name="Example",
                    source=MetadataSource(
                        origin=MetadataOrigin.SIDECAR,
                        identifier="memory",
                    ),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"

            result = write_catalog_json(catalog, path)

            text = path.read_text(encoding="utf-8")
            self.assertEqual(result, path)
            self.assertTrue(text.endswith("\n"))
            self.assertEqual(json.loads(text)["boards"][0]["name"], "Example")
