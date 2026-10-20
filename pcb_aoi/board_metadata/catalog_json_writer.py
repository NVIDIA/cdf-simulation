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

"""Write source-neutral board metadata catalogs as inspectable JSON."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

from .metadata_catalog import BoardMetadataCatalog

CATALOG_FORMAT = "pcb_aoi.board_metadata.catalog"
CATALOG_FORMAT_VERSION = "1"


def _json_value(value: object) -> object:
    """Convert a catalog or extension value into JSON-compatible data.

    Args:
        value: Value to normalize.

    Returns:
        JSON-compatible scalar, list, or dictionary.

    """
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set | frozenset):
        normalized = (_json_value(item) for item in value)
        return sorted(normalized, key=str)
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if hasattr(value, "path"):
        return str(value.path)
    return str(value)


def catalog_to_json_data(catalog: BoardMetadataCatalog) -> dict[str, object]:
    """Build the stable JSON document for a board metadata catalog.

    Args:
        catalog: Fully loaded source-neutral catalog.

    Returns:
        JSON-compatible catalog document without private query indexes.

    """
    return {
        "format": CATALOG_FORMAT,
        "format_version": CATALOG_FORMAT_VERSION,
        "capabilities": _json_value(catalog.capabilities),
        "boards": _json_value(catalog.boards),
        "components": _json_value(catalog.components),
        "prototypes": _json_value(catalog.prototypes),
        "terminals": _json_value(catalog.terminals),
        "pads": _json_value(catalog.pads),
        "holes": _json_value(catalog.holes),
        "paste_deposits": _json_value(catalog.paste_deposits),
        "nets": _json_value(catalog.nets),
        "diagnostics": _json_value(catalog.diagnostics),
    }


def write_catalog_json(
    catalog: BoardMetadataCatalog,
    output_path: str | Path,
) -> Path:
    """Write a board metadata catalog as deterministic, indented JSON.

    Args:
        catalog: Fully loaded source-neutral catalog.
        output_path: Destination JSON path.

    Returns:
        Destination path.

    Raises:
        OSError: If the destination cannot be written.

    """
    destination = Path(output_path)
    destination.write_text(
        json.dumps(catalog_to_json_data(catalog), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
