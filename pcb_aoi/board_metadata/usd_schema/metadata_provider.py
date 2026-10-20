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

"""Load composed PCBA USD metadata into the source-neutral catalog."""

from __future__ import annotations

from collections.abc import Mapping

from pxr import Sdf, Usd, UsdGeom

from cdf.errors import ValidationError

from ..metadata_catalog import (
    HOLE_ASSOCIATION_ORIGIN_KEY,
    BoardMetadataCatalog,
    BoardRecord,
    ComponentRecord,
    HoleRecord,
    MetadataCapability,
    MetadataOrigin,
    MetadataSource,
    PadRecord,
    PasteDepositRecord,
    PrototypeRecord,
    TerminalRecord,
)
from ...geometry.transform import transform_points
from ..metadata_provider import BoardMetadataRequest
from ..usd_schema_metadata_keys import (
    MESH_ROLE_ATTR,
    USD_SCHEMA_METADATA_KEYS,
)
from .metadata_validation import schema_version, validate_core_metadata
from .relationship_resolver import referenced_asset_path, relative_prim_path
from .stage_traversal import iter_component_prims, iter_prim_subtree

_COMPONENT_FIELDS = {
    "pcba:component:refdes",
    "pcba:component:type",
    "pcba:component:packageType",
    "pcba:component:footprintId",
    "pcba:component:side",
    "pcba:component:processClass",
    "pcba:component:processClasses",
}

_TERMINAL_FIELDS = {
    "pcba:pin:number",
    "pcba:pin:name",
    "pcba:pin:net",
    "pcba:pin:solderable",
    "pcba:pin:processClass",
    "pcba:pin:crossSection",
}

_PAD_FIELDS = {
    MESH_ROLE_ATTR,
    "pcba:board:padNumber",
    "pcba:board:refdes",
    "pcba:board:componentPinNumber",
    "pcba:board:occupied",
    "pcba:board:intentionallyUnpopulated",
    "pcba:board:padContour",
}

_HOLE_FIELDS = {
    MESH_ROLE_ATTR,
    "pcba:board:refdes",
    "pcba:board:componentPinNumber",
    "pcba:board:holeCenterMm",
    "pcba:board:holeShape",
    "pcba:board:holeDimsMm",
    "pcba:board:holeSide",
    "pcba:board:plated",
    "pcba:board:occupied",
    "pcba:board:intentionallyUnpopulated",
}


def _attribute_value(prim: Usd.Prim, name: str) -> object | None:
    """Return an authored attribute value from a prim.

    Args:
        prim: Prim to inspect.
        name: Attribute name.

    Returns:
        Composed value, or None when the attribute is absent.

    """
    attribute = prim.GetAttribute(name)
    return attribute.Get() if attribute else None


def _text(prim: Usd.Prim, name: str) -> str | None:
    """Return a nonempty attribute value as text.

    Args:
        prim: Prim to inspect.
        name: Attribute name.

    Returns:
        Nonempty text, or None.

    """
    value = _attribute_value(prim, name)
    return str(value) if value is not None and str(value) else None


def _metadata_text(prim: Usd.Prim, semantic_key: str) -> str | None:
    """Return text from the expected metadata key or its fallbacks."""
    for attribute_name in USD_SCHEMA_METADATA_KEYS[semantic_key]:
        if value := _text(prim, attribute_name):
            return value
    return None


def _ancestor_text(prim: Usd.Prim, name: str) -> str | None:
    """Return the nearest authored text value on a prim or its ancestors."""
    current = prim
    while current and current.IsValid():
        if value := _text(current, name):
            return value
        current = current.GetParent()
    return None


def _xy_center(prim: Usd.Prim, attribute_name: str) -> tuple[float, float] | None:
    """Return the XY center of a point array or extent attribute."""
    raw_points = _attribute_value(prim, attribute_name)
    if raw_points is None:
        return None
    points = tuple(raw_points)
    if not points:
        return None
    xs = tuple(float(point[0]) for point in points)
    ys = tuple(float(point[1]) for point in points)
    return ((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5)


def _float_tuple(
    prim: Usd.Prim,
    attribute_name: str,
) -> tuple[float, ...] | None:
    """Return one numeric tuple attribute as source-neutral floats."""
    value = _attribute_value(prim, attribute_name)
    if value is None:
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def _point_tuple(
    prim: Usd.Prim,
    attribute_name: str,
) -> tuple[tuple[float, ...], ...]:
    """Return one point-array attribute as source-neutral float tuples."""
    value = _attribute_value(prim, attribute_name)
    if value is None:
        return ()
    try:
        return tuple(tuple(float(axis) for axis in point) for point in value)
    except (TypeError, ValueError):
        return ()


def _plain(value: object) -> object:
    """Convert common USD values into source-neutral Python data.

    Args:
        value: Composed USD value.

    Returns:
        Scalar, list, dictionary, or readable string.

    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Sdf.AssetPath):
        return value.resolvedPath or value.path
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    try:
        return [_plain(item) for item in value]  # type: ignore[union-attr]
    except TypeError:
        return str(value)


def _extensions(prim: Usd.Prim, consumed: set[str]) -> dict[str, object]:
    """Collect namespaced metadata not represented by catalog core fields.

    Args:
        prim: Prim to inspect.
        consumed: Attribute names already mapped to typed fields.

    Returns:
        Source-specific metadata keyed by full USD attribute name.

    """
    return {
        attribute.GetName(): _plain(attribute.Get())
        for attribute in prim.GetAttributes()
        if attribute.GetName().startswith(("pcb:", "pcba:"))
        and attribute.GetName() not in consumed
    }


def _board_thickness_mm(board_prim: Usd.Prim) -> float | None:
    """Return the authored board thickness in millimeters, if any."""
    value = _attribute_value(board_prim, "pcba:board:thicknessMm")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _substrate_z_range(
    board_prims: tuple[Usd.Prim, ...],
) -> tuple[float, float] | None:
    """Measure the world-space z extent of the board_substrate meshes.

    A derived convenience recorded on the BoardRecord so consumers that need
    the board faces (THT barrels, solder-side classification) do not each
    re-scan the stage. None when no substrate mesh carries points.

    Args:
        board_prims: Board-subtree prims to scan.

    Returns:
        (z_lo, z_hi) in world units, or None.

    """
    cache = UsdGeom.XformCache()
    z_lo = z_hi = None
    for prim in board_prims:
        if _metadata_text(prim, "board_feature.role") != "board_substrate":
            continue
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
        # A substrate mesh is the largest in the tree and this wants two
        # scalars out of it, so the points move in one batch rather than one
        # Gf call each. transform_points is exact against that loop.
        world_z = transform_points(matrix, points)[:, 2]
        low, high = float(world_z.min()), float(world_z.max())
        z_lo = low if z_lo is None or low < z_lo else z_lo
        z_hi = high if z_hi is None or high > z_hi else z_hi
    if z_lo is None:
        return None
    return (float(z_lo), float(z_hi))


def _board_component(
    prim: Usd.Prim,
    component_by_path: Mapping[str, ComponentRecord],
    component_by_refdes: Mapping[str, ComponentRecord],
) -> tuple[ComponentRecord | None, str | None]:
    """Resolve the component associated with a board feature prim.

    Returns:
        ``(component, origin)`` -- origin is ``"relationship"`` when the
        authored component rel matched, ``"refdes"`` when the authored
        refdes text did, and None when unresolved. Both origins are
        EXPORTER-STATED associations: the contract's S10 design note makes
        the refdes + componentPinNumber pair the primary association and
        the rel a secondary direct handle (checked first here only because
        a resolved rel needs no name lookup). Consumers that must separate
        stated associations from downstream geometric guesses (e.g. THT
        joint provenance) read the origin; everything else may ignore it.

    """
    for relationship_name in USD_SCHEMA_METADATA_KEYS[
        "board_feature.component_relationship"
    ]:
        relationship = prim.GetRelationship(relationship_name)
        targets = tuple(str(path) for path in relationship.GetTargets())
        if targets and (component := component_by_path.get(targets[0])):
            return component, "relationship"
    component = component_by_refdes.get(
        _metadata_text(prim, "board_feature.refdes") or "",
    )
    return component, ("refdes" if component else None)


def _solder_side_pad(
    pads: tuple[PadRecord, ...],
    component: ComponentRecord,
) -> PadRecord | None:
    """Return the pad on the side opposite component insertion when available."""
    if not pads:
        return None
    insertion_side = (component.side or "").lower()
    solder_side = "bottom" if insertion_side.startswith("t") else "top"
    return next(
        (pad for pad in pads if (pad.side or "").lower() == solder_side),
        pads[0],
    )


def _board_root(stage: Usd.Stage, requested: str | None) -> str:
    """Resolve and validate the requested board-root path.

    Args:
        stage: Open composed USD stage.
        requested: Explicit root path, if supplied.

    Returns:
        Absolute board-root path.

    Raises:
        ValidationError: If no valid board root can be resolved.

    """
    default_prim = stage.GetDefaultPrim()
    path = requested or (str(default_prim.GetPath()) if default_prim else "")
    if not path or not stage.GetPrimAtPath(path):
        raise ValidationError(f"USD board root {path!r} does not exist")
    return path


class UsdSchemaMetadataProvider:
    """Normalize composed PCBA USD metadata into a board catalog."""

    def load(self, request: BoardMetadataRequest) -> BoardMetadataCatalog:
        """Load metadata from an already composed USD stage.

        Args:
            request: Provider request containing an open USD stage.

        Returns:
            Immutable catalog populated from the composed stage.

        Raises:
            ValidationError: If the request has no usable stage or board root.

        """
        if not isinstance(request.stage, Usd.Stage):
            raise ValidationError("USD schema metadata provider requires a USD stage")
        stage = request.stage
        board_root = _board_root(stage, request.board_root)
        board_prim = stage.GetPrimAtPath(board_root)
        root_identifier = stage.GetRootLayer().identifier
        source = MetadataSource(
            origin=MetadataOrigin.USD,
            identifier=request.source_identifier or root_identifier,
            schema_version=schema_version(board_prim),
            generator_version=_text(board_prim, "pcba:board:generatorVersion"),
        )

        components = []
        terminals = []
        prototypes: dict[str, PrototypeRecord] = {}
        for component_prim in iter_component_prims(stage, board_root):
            component_path = str(component_prim.GetPath())
            process_class = _text(component_prim, "pcba:component:processClass")
            raw_process_classes = _attribute_value(
                component_prim,
                "pcba:component:processClasses",
            )
            process_classes = (
                tuple(str(item) for item in raw_process_classes)
                if raw_process_classes is not None
                else (process_class,)
                if process_class
                else ()
            )
            asset_path = referenced_asset_path(component_prim)
            footprint_id = _text(component_prim, "pcba:component:footprintId")
            package_type = _text(component_prim, "pcba:component:packageType")
            prototype_id = asset_path or (
                f"footprint:{footprint_id or package_type or component_prim.GetName()}"
            )
            body_paths = []
            contact_paths = []
            marking_paths = []
            relative_bodies = []
            relative_contacts = []
            relative_markings = []
            for child in iter_prim_subtree(component_prim):
                if child == component_prim:
                    continue
                child_path = str(child.GetPath())
                child_name = child.GetName().lower()
                pin_number = child.GetAttribute("pcba:pin:number")
                if pin_number:
                    contact_paths.append(child_path)
                    relative_contacts.append(relative_prim_path(component_prim, child))
                    terminals.append(
                        TerminalRecord(
                            board_root=board_root,
                            component_path=component_path,
                            prim_path=child_path,
                            source=source,
                            number=_text(child, "pcba:pin:number"),
                            name=_text(child, "pcba:pin:name"),
                            net=_text(child, "pcba:pin:net"),
                            solderable=_attribute_value(child, "pcba:pin:solderable"),
                            process_class=_text(child, "pcba:pin:processClass"),
                            cross_section=_text(child, "pcba:pin:crossSection"),
                            extensions=_extensions(child, _TERMINAL_FIELDS),
                        ),
                    )
                elif "mark" in child_name:
                    marking_paths.append(child_path)
                    relative_markings.append(relative_prim_path(component_prim, child))
                elif child_name.startswith(("pkg_body", "pkg_lid")):
                    body_paths.append(child_path)
                    relative_bodies.append(relative_prim_path(component_prim, child))

            prototype_prim = component_prim.GetPrototype()
            prototypes.setdefault(
                prototype_id,
                PrototypeRecord(
                    prototype_id=prototype_id,
                    source=source,
                    name=footprint_id or package_type or component_prim.GetName(),
                    prim_path=str(prototype_prim.GetPath()) if prototype_prim else None,
                    asset_path=asset_path,
                    body_meshes=tuple(relative_bodies),
                    contact_meshes=tuple(relative_contacts),
                    marking_meshes=tuple(relative_markings),
                ),
            )
            components.append(
                ComponentRecord(
                    board_root=board_root,
                    refdes=_text(component_prim, "pcba:component:refdes") or "",
                    prim_path=component_path,
                    source=source,
                    component_type=_text(component_prim, "pcba:component:type"),
                    package_name=footprint_id or package_type,
                    package_type=package_type,
                    footprint_id=footprint_id,
                    side=_text(component_prim, "pcba:component:side"),
                    process_class=process_class,
                    process_classes=process_classes,
                    prototype_id=prototype_id,
                    body_prim_paths=tuple(body_paths),
                    contact_prim_paths=tuple(contact_paths),
                    extensions=_extensions(component_prim, _COMPONENT_FIELDS),
                ),
            )

        component_by_path = {component.prim_path: component for component in components}
        component_by_refdes = {component.refdes: component for component in components}
        terminal_by_component_number = {
            (terminal.component_path, terminal.number): terminal
            for terminal in terminals
            if terminal.number
        }
        board_prims = tuple(iter_prim_subtree(board_prim))
        pads = []
        pads_by_name = {}
        pads_by_component_terminal = {}
        for prim in board_prims:
            if _metadata_text(prim, "board_feature.role") != "board_pad":
                continue
            component, _association_origin = _board_component(
                prim,
                component_by_path,
                component_by_refdes,
            )
            terminal_number = _metadata_text(prim, "pad.terminal_number")
            terminal = (
                terminal_by_component_number.get(
                    (component.prim_path, terminal_number),
                )
                if component and terminal_number
                else None
            )
            contour = _point_tuple(prim, "pcba:board:padContour")
            pin_center = _xy_center(prim, "pcba:board:padContour") or _xy_center(
                prim,
                "extent",
            )
            pad = PadRecord(
                board_root=board_root,
                prim_path=str(prim.GetPath()),
                source=source,
                component_path=component.prim_path if component else None,
                terminal_number=terminal_number,
                contact_prim_path=terminal.prim_path if terminal else None,
                pad_type=_metadata_text(prim, "pad.type"),
                side=_ancestor_text(prim, "pcba:layer:side"),
                occupied=_attribute_value(prim, "pcba:board:occupied"),
                intentionally_unpopulated=_attribute_value(
                    prim,
                    "pcba:board:intentionallyUnpopulated",
                ),
                pin_center=pin_center,
                contour=contour,
                extensions=_extensions(prim, _PAD_FIELDS),
            )
            pads.append(pad)
            pads_by_name[prim.GetName()] = pad
            if component and terminal_number:
                pads_by_component_terminal.setdefault(
                    (component.prim_path, terminal_number),
                    [],
                ).append(pad)

        holes = []
        for prim in board_prims:
            if _metadata_text(prim, "board_feature.role") != "board_hole":
                continue
            component, association_origin = _board_component(
                prim,
                component_by_path,
                component_by_refdes,
            )
            terminal_number = _metadata_text(prim, "hole.terminal_number")
            pad = pads_by_name.get(prim.GetName())
            if pad is None and component and terminal_number:
                pad = _solder_side_pad(
                    tuple(
                        pads_by_component_terminal.get(
                            (component.prim_path, terminal_number),
                            (),
                        ),
                    ),
                    component,
                )
            if terminal_number is None and pad:
                terminal_number = pad.terminal_number
            holes.append(
                HoleRecord(
                    board_root=board_root,
                    prim_path=str(prim.GetPath()),
                    source=source,
                    component_path=component.prim_path if component else None,
                    terminal_number=terminal_number,
                    pad_prim_path=pad.prim_path if pad else None,
                    side=_text(prim, "pcba:board:holeSide"),
                    plated=_attribute_value(prim, "pcba:board:plated"),
                    occupied=_attribute_value(prim, "pcba:board:occupied"),
                    intentionally_unpopulated=_attribute_value(
                        prim,
                        "pcba:board:intentionallyUnpopulated",
                    ),
                    site_kind=_metadata_text(prim, "site.kind"),
                    shape=_text(prim, "pcba:board:holeShape"),
                    center_mm=_float_tuple(prim, "pcba:board:holeCenterMm"),
                    dimensions_mm=_float_tuple(prim, "pcba:board:holeDimsMm"),
                    extensions=(
                        {
                            **_extensions(prim, _HOLE_FIELDS),
                            HOLE_ASSOCIATION_ORIGIN_KEY: association_origin,
                        }
                        if association_origin
                        else _extensions(prim, _HOLE_FIELDS)
                    ),
                ),
            )

        paste_deposits = []
        for prim in board_prims:
            if _metadata_text(prim, "board_feature.role") != "board_paste":
                continue
            relationship = prim.GetRelationship("pcba:board:pasteFor")
            targets = tuple(str(path) for path in relationship.GetTargets())
            target_path = targets[0] if targets else None
            component = next(
                (
                    item
                    for path, item in component_by_path.items()
                    if target_path == path or (target_path or "").startswith(f"{path}/")
                ),
                None,
            )
            if component is None:
                refdes = _metadata_text(prim, "paste.refdes")
                component = component_by_refdes.get(refdes or "")
            site_kind = _metadata_text(prim, "paste.site_kind")
            paste_deposits.append(
                PasteDepositRecord(
                    board_root=board_root,
                    prim_path=str(prim.GetPath()),
                    source=source,
                    component_path=component.prim_path if component else None,
                    terminal_number=_metadata_text(prim, "paste.terminal_number"),
                    target_path=target_path,
                    side=_text(prim, "pcba:board:solderLayer"),
                    site_kind=site_kind,
                    extensions=_extensions(
                        prim,
                        {MESH_ROLE_ATTR, "pcba:board:solderLayer"},
                    ),
                ),
            )

        capabilities = {
            MetadataCapability.COMPONENTS,
            MetadataCapability.COMPONENT_PROTOTYPES,
        }
        if terminals:
            capabilities.update(
                {MetadataCapability.TERMINALS, MetadataCapability.SOLDER_CONTACTS},
            )
        if paste_deposits:
            capabilities.add(MetadataCapability.SOLDER_PASTE_ASSOCIATIONS)
        # Physical availability first: a pad or a hole is labellable because it
        # is THERE, not because something claimed it. An unassociated hole --
        # the case this exists for -- has no terminal and must still be
        # reachable.
        if pads:
            capabilities.add(MetadataCapability.PADS)
        if holes:
            capabilities.add(MetadataCapability.HOLES)
        if any(pad.component_path and pad.terminal_number for pad in pads):
            capabilities.add(MetadataCapability.TERMINAL_PAD_ASSOCIATIONS)
        if any(hole.component_path and hole.terminal_number for hole in holes):
            capabilities.add(MetadataCapability.TERMINAL_HOLE_ASSOCIATIONS)
        return BoardMetadataCatalog(
            boards=(
                BoardRecord(
                    root_path=board_root,
                    name=_text(board_prim, "pcba:board:name") or board_prim.GetName(),
                    source=source,
                    asset_path=root_identifier,
                    meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
                    up_axis=str(UsdGeom.GetStageUpAxis(stage)),
                    thickness_mm=_board_thickness_mm(board_prim),
                    substrate_z_range=_substrate_z_range(board_prims),
                    extensions=_extensions(
                        board_prim,
                        {
                            "pcba:board:name",
                            "pcba:board:generatorVersion",
                            "pcba:schema:version",
                            "pcba:board:schemaVersion",
                            "pcba:board:thicknessMm",
                        },
                    ),
                ),
            ),
            components=tuple(components),
            prototypes=tuple(prototypes.values()),
            terminals=tuple(terminals),
            pads=tuple(pads),
            holes=tuple(holes),
            paste_deposits=tuple(paste_deposits),
            capabilities=frozenset(capabilities),
            diagnostics=validate_core_metadata(stage, board_root),
        )
