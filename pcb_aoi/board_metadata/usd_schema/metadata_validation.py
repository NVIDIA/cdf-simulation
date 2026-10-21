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

"""Validate core PCBA metadata without inventing missing schema values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pxr import Usd

from ..metadata_catalog import Diagnostic
from .stage_traversal import iter_component_prims

_SCHEMA_VERSION_ATTRIBUTES = (
    "pcba:schema:version",
    "pcba:board:schemaVersion",
)

_BOARD_ATTRIBUTE_TYPES: Mapping[str, tuple[str, ...]] = {
    "pcba:board:name": ("string",),
    "pcba:board:generatorVersion": ("string",),
    "pcba:board:contentHash": ("string",),
    "pcba:board:packageManifest": ("asset",),
    "pcba:board:thicknessMm": ("float", "double"),
    "pcba:board:originConvention": ("token", "string"),
}

_COMPONENT_ATTRIBUTE_TYPES: Mapping[str, tuple[str, ...]] = {
    "pcba:component:refdes": ("string", "token"),
    "pcba:component:type": ("token", "string"),
    "pcba:component:processClass": ("token", "string"),
    "pcba:component:packageType": ("token", "string"),
    "pcba:component:footprintId": ("string", "token"),
    "pcba:component:side": ("token", "string"),
}


def schema_version(board_prim: Usd.Prim) -> str | None:
    """Return the explicitly authored PCBA schema version.

    Args:
        board_prim: Composed board-root prim.

    Returns:
        Authored schema version, or None when no recognized field exists.

    Example:

    .. code-block:: python

        print(schema_version(board_prim))  # doctest: +SKIP
        1.0

    """
    for name in _SCHEMA_VERSION_ATTRIBUTES:
        attribute = board_prim.GetAttribute(name)
        if attribute and attribute.HasAuthoredValueOpinion():
            value = attribute.Get()
            if value is not None and str(value):
                return str(value)
    return None


def _validate_attributes(
    prim: Usd.Prim,
    expected_types: Mapping[str, Sequence[str]],
) -> list[Diagnostic]:
    """Validate required attribute presence and accepted USD types.

    Args:
        prim: Prim carrying required metadata.
        expected_types: Accepted USD type names keyed by attribute name.

    Returns:
        Validation diagnostics for the supplied prim.

    """
    diagnostics = []
    for name, accepted_types in expected_types.items():
        attribute = prim.GetAttribute(name)
        if not attribute:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="usd.required_attribute_missing",
                    message=f"required metadata attribute {name!r} is missing",
                    entity_path=str(prim.GetPath()),
                ),
            )
            continue
        type_name = str(attribute.GetTypeName())
        if type_name not in accepted_types:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="usd.attribute_type_unsupported",
                    message=(
                        f"metadata attribute {name!r} uses {type_name!r}; "
                        f"accepted types are {list(accepted_types)!r}"
                    ),
                    entity_path=str(prim.GetPath()),
                ),
            )
    return diagnostics


def validate_core_metadata(
    stage: Usd.Stage,
    board_root: str,
) -> tuple[Diagnostic, ...]:
    """Validate the board and component metadata required by the MVP reader.

    Args:
        stage: Open composed USD stage.
        board_root: Absolute board-root prim path.

    Returns:
        Structured validation diagnostics in deterministic traversal order.

    Raises:
        ValueError: If `board_root` does not identify a valid prim.

    Example:

    .. code-block:: python

        diagnostics = validate_core_metadata(stage, "/Board")
        print([item.code for item in diagnostics])  # doctest: +SKIP

    """
    board_prim = stage.GetPrimAtPath(board_root)
    if not board_prim or not board_prim.IsValid():
        raise ValueError(f"board root {board_root!r} does not exist")

    diagnostics = _validate_attributes(board_prim, _BOARD_ATTRIBUTE_TYPES)
    if schema_version(board_prim) is None:
        diagnostics.append(
            Diagnostic(
                severity="warning",
                code="usd.schema_version_missing",
                message="no recognized PCBA schema-version attribute is authored",
                entity_path=board_root,
            ),
        )

    for component_prim in iter_component_prims(stage, board_root):
        diagnostics.extend(
            _validate_attributes(component_prim, _COMPONENT_ATTRIBUTE_TYPES),
        )
        process_class = component_prim.GetAttribute(
            "pcba:component:processClass",
        ).Get()
        process_classes = component_prim.GetAttribute(
            "pcba:component:processClasses",
        )
        if str(process_class) == "Mixed" and not process_classes:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    code="usd.mixed_process_classes_missing",
                    message=(
                        "Mixed component does not author pcba:component:processClasses"
                    ),
                    entity_path=str(component_prim.GetPath()),
                ),
            )
        if process_classes and str(process_classes.GetTypeName()) != "token[]":
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="usd.process_classes_type_unsupported",
                    message=("pcba:component:processClasses must use token[] metadata"),
                    entity_path=str(component_prim.GetPath()),
                ),
            )
    return tuple(diagnostics)
