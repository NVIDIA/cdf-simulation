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

"""Source-neutral PCB metadata records and indexed board catalog."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from cdf.errors import ValidationError


class MetadataOrigin(str, Enum):
    """Physical source from which a metadata record was decoded."""

    USD = "usd"
    SIDECAR = "sidecar"
    INFERRED = "inferred"


#: ``HoleRecord.extensions`` key recording HOW a hole's component association
#: was resolved: ``"relationship"`` when the authored ``pcba:board:component``
#: rel matched, ``"refdes"`` when the authored refdes text did. BOTH are
#: exporter-stated: the PCBA-to-USD contract makes the refdes +
#: componentPinNumber PAIR the primary association and the rel a
#: convenience handle whose absence is not a failure. Absent when
#: unresolved or when the source does not track it. Consumers use it to
#: distinguish stated associations from downstream geometric guesses (the
#: contract forbids leaving association to XY coincidence).
HOLE_ASSOCIATION_ORIGIN_KEY = "catalog:componentAssociationOrigin"


class MetadataCapability(str, Enum):
    """A query capability supplied by a metadata catalog."""

    COMPONENTS = "components"
    COMPONENT_PROTOTYPES = "component_prototypes"
    TERMINALS = "terminals"
    PADS = "pads"
    HOLES = "holes"
    TERMINAL_PAD_ASSOCIATIONS = "terminal_pad_associations"
    TERMINAL_HOLE_ASSOCIATIONS = "terminal_hole_associations"
    SOLDER_CONTACTS = "solder_contacts"
    SOLDER_PASTE_ASSOCIATIONS = "solder_paste_associations"
    NETS = "nets"


@dataclass(frozen=True)
class MetadataSource:
    """Identity and format of the source that produced catalog records.

    Args:
        origin: Physical source category.
        identifier: Stable source path or in-memory identifier.
        schema_version: Metadata schema version, if declared.
        generator_version: Producer version, if declared.

    """

    origin: MetadataOrigin
    identifier: str
    schema_version: str | None = None
    generator_version: str | None = None


@dataclass(frozen=True)
class Diagnostic:
    """Structured metadata validation or normalization message.

    Args:
        severity: Diagnostic severity name.
        code: Stable machine-readable diagnostic code.
        message: Human-readable explanation.
        entity_path: Related prim path, if available.

    """

    severity: str
    code: str
    message: str
    entity_path: str | None = None


def _empty_mapping() -> Mapping[str, object]:
    """Return an immutable empty mapping.

    Returns:
        Empty read-only mapping.

    """
    return MappingProxyType({})


def freeze_mapping(value: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable shallow copy suitable for frozen records.

    Args:
        value: Mapping to copy, or None for an empty mapping.

    Returns:
        Read-only shallow copy of the supplied mapping.

    Example:

    .. code-block:: python

        from pcb_aoi.board_metadata.metadata_catalog import freeze_mapping

        frozen = freeze_mapping({"refdes": "R1"})
        print(frozen["refdes"])
        R1

    """
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class BoardRecord:
    """One PCB assembly root represented by the catalog.

    Args:
        root_path: Absolute board-root prim path.
        name: Human-readable board name.
        source: Metadata source identity.
        asset_path: Source board asset path, if known.
        meters_per_unit: Stage scale in meters per unit, if known.
        up_axis: Stage up-axis token, if known.
        thickness_mm: Authored board thickness in millimeters, if known.
        substrate_z_range: Measured world-space z extent (lo, hi) of the
            board_substrate meshes, if the source could measure it. A
            derived convenience for consumers that need the board faces
            (THT barrels, solder side) without re-scanning the stage.
        extensions: Source-specific values not represented by core fields.

    """

    root_path: str
    name: str
    source: MetadataSource
    asset_path: str | None = None
    meters_per_unit: float | None = None
    up_axis: str | None = None
    thickness_mm: float | None = None
    substrate_z_range: tuple[float, float] | None = None
    extensions: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        """Freeze extension values after dataclass initialization."""
        object.__setattr__(self, "extensions", freeze_mapping(self.extensions))


@dataclass(frozen=True)
class ComponentRecord:
    """One placed component instance.

    Args:
        board_root: Absolute path of the owning board root.
        refdes: Component reference designator.
        prim_path: Absolute component-instance prim path.
        source: Metadata source identity.
        component_type: Semantic component type, if available.
        family: Component family, if available.
        package_name: Package or model name, if available.
        package_type: Package classification, if available.
        footprint_id: Source footprint identity, if available.
        side: Board side, if available.
        process_class: Aggregate manufacturing process classification.
        process_classes: Manufacturing process classifications.
        prototype_id: Reusable prototype identity, if available.
        group_path: Legacy component-group path, if available.
        body_prim_paths: Component body prim paths.
        contact_prim_paths: Solder-contact or terminal prim paths.
        extensions: Source-specific values not represented by core fields.

    """

    board_root: str
    refdes: str
    prim_path: str
    source: MetadataSource
    component_type: str | None = None
    family: str | None = None
    package_name: str | None = None
    package_type: str | None = None
    footprint_id: str | None = None
    side: str | None = None
    process_class: str | None = None
    process_classes: tuple[str, ...] = ()
    prototype_id: str | None = None
    group_path: str | None = None
    body_prim_paths: tuple[str, ...] = ()
    contact_prim_paths: tuple[str, ...] = ()
    extensions: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        """Freeze sequence and extension values after initialization."""
        object.__setattr__(self, "process_classes", tuple(self.process_classes))
        object.__setattr__(self, "body_prim_paths", tuple(self.body_prim_paths))
        object.__setattr__(self, "contact_prim_paths", tuple(self.contact_prim_paths))
        object.__setattr__(self, "extensions", freeze_mapping(self.extensions))


@dataclass(frozen=True)
class PrototypeRecord:
    """Reusable component package or model metadata.

    Args:
        prototype_id: Stable prototype identity.
        source: Metadata source identity.
        name: Package or model name.
        family: Component family, if available.
        prim_path: Prototype prim path, if available.
        asset_path: Referenced package asset path, if available.
        body_meshes: Prototype-relative body mesh names or paths.
        contact_meshes: Prototype-relative contact mesh names or paths.
        marking_meshes: Prototype-relative marking mesh names or paths.
        extensions: Source-specific values not represented by core fields.

    """

    prototype_id: str
    source: MetadataSource
    name: str
    family: str | None = None
    prim_path: str | None = None
    asset_path: str | None = None
    body_meshes: tuple[str, ...] = ()
    contact_meshes: tuple[str, ...] = ()
    marking_meshes: tuple[str, ...] = ()
    extensions: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        """Freeze sequence and extension values after initialization."""
        object.__setattr__(self, "body_meshes", tuple(self.body_meshes))
        object.__setattr__(self, "contact_meshes", tuple(self.contact_meshes))
        object.__setattr__(self, "marking_meshes", tuple(self.marking_meshes))
        object.__setattr__(self, "extensions", freeze_mapping(self.extensions))


@dataclass(frozen=True)
class TerminalRecord:
    """A component terminal or legacy solder-contact prim.

    Args:
        board_root: Absolute path of the owning board root.
        component_path: Absolute path of the owning component.
        prim_path: Absolute terminal or contact prim path.
        source: Metadata source identity.
        number: Terminal number, if available.
        name: Terminal name, if available.
        net: Connected net identity, if available.
        solderable: Whether the terminal accepts solder, if known.
        process_class: Manufacturing process classification, if available.
        cross_section: Terminal cross-section classification, if available.
        extensions: Source-specific values not represented by core fields.

    """

    board_root: str
    component_path: str
    prim_path: str
    source: MetadataSource
    number: str | None = None
    name: str | None = None
    net: str | None = None
    solderable: bool | None = None
    process_class: str | None = None
    cross_section: str | None = None
    extensions: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        """Freeze extension values after dataclass initialization."""
        object.__setattr__(self, "extensions", freeze_mapping(self.extensions))


@dataclass(frozen=True)
class PadRecord:
    """A PCB pad associated with a component terminal or contact.

    Args:
        board_root: Absolute path of the owning board root.
        prim_path: Absolute pad prim path.
        source: Metadata source identity.
        component_path: Owning component path, if resolved.
        terminal_number: Associated terminal number, if resolved.
        contact_prim_path: Associated contact prim path, if resolved.
        pad_type: Pad classification, if available.
        side: Board side, if available.
        occupied: Whether a component occupies the pad, if known.
        intentionally_unpopulated: Whether the site is intentionally empty.
        pin_center: Pin center in board-layout coordinates, if available.
        contour: Explicit exposed-pad contour points, if available.
        extensions: Source-specific values not represented by core fields.

    """

    board_root: str
    prim_path: str
    source: MetadataSource
    component_path: str | None = None
    terminal_number: str | None = None
    contact_prim_path: str | None = None
    pad_type: str | None = None
    side: str | None = None
    occupied: bool | None = None
    intentionally_unpopulated: bool | None = None
    pin_center: tuple[float, float] | None = None
    contour: tuple[tuple[float, ...], ...] = ()
    extensions: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        """Freeze coordinate and extension values after initialization."""
        if self.pin_center is not None:
            object.__setattr__(self, "pin_center", tuple(self.pin_center))
        object.__setattr__(
            self,
            "contour",
            tuple(tuple(point) for point in self.contour),
        )
        object.__setattr__(self, "extensions", freeze_mapping(self.extensions))


@dataclass(frozen=True)
class HoleRecord:
    """A plated, non-plated, or mechanical board hole.

    Args:
        board_root: Absolute path of the owning board root.
        prim_path: Absolute hole prim path.
        source: Metadata source identity.
        component_path: Owning component path, if resolved.
        terminal_number: Associated terminal number, if resolved.
        pad_prim_path: Associated board-pad path, if resolved.
        side: Board side, if applicable.
        plated: Whether the hole is plated, if known.
        occupied: Whether a component occupies the hole, if known.
        intentionally_unpopulated: Whether the site is intentionally empty.
        site_kind: Electrical or mechanical site classification, if available.
        shape: Hole shape, if available.
        center_mm: Hole center in board-layout millimeters, if available.
        dimensions_mm: Hole dimensions in millimeters, if available.
        extensions: Source-specific values not represented by core fields.

    """

    board_root: str
    prim_path: str
    source: MetadataSource
    component_path: str | None = None
    terminal_number: str | None = None
    pad_prim_path: str | None = None
    side: str | None = None
    plated: bool | None = None
    occupied: bool | None = None
    intentionally_unpopulated: bool | None = None
    site_kind: str | None = None
    shape: str | None = None
    center_mm: tuple[float, float] | None = None
    dimensions_mm: tuple[float, float] | None = None
    extensions: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        """Freeze coordinate and extension values after initialization."""
        if self.center_mm is not None:
            object.__setattr__(self, "center_mm", tuple(self.center_mm))
        if self.dimensions_mm is not None:
            object.__setattr__(self, "dimensions_mm", tuple(self.dimensions_mm))
        object.__setattr__(self, "extensions", freeze_mapping(self.extensions))


@dataclass(frozen=True)
class PasteDepositRecord:
    """One solder-paste deposit and its resolved or unresolved target.

    Args:
        board_root: Absolute path of the owning board root.
        prim_path: Absolute solder-paste prim path.
        source: Metadata source identity.
        component_path: Owning component path, if resolved.
        terminal_number: Associated terminal number, if resolved.
        target_path: Explicit pad or terminal relationship target, if resolved.
        side: Board side, if available.
        site_kind: Fitted or unpopulated site classification, if available.
        asset_paths: Source assets containing paste geometry.
        extensions: Source-specific values not represented by core fields.

    """

    board_root: str
    prim_path: str
    source: MetadataSource
    component_path: str | None = None
    terminal_number: str | None = None
    target_path: str | None = None
    side: str | None = None
    site_kind: str | None = None
    asset_paths: tuple[str, ...] = ()
    extensions: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        """Freeze sequence and extension values after initialization."""
        object.__setattr__(self, "asset_paths", tuple(self.asset_paths))
        object.__setattr__(self, "extensions", freeze_mapping(self.extensions))


@dataclass(frozen=True)
class NetRecord:
    """A named electrical net and associated prim paths.

    Args:
        board_root: Absolute path of the owning board root.
        name: Stable net name or identity.
        source: Metadata source identity.
        terminal_paths: Associated terminal prim paths.
        feature_paths: Associated board-feature prim paths.
        extensions: Source-specific values not represented by core fields.

    """

    board_root: str
    name: str
    source: MetadataSource
    terminal_paths: tuple[str, ...] = ()
    feature_paths: tuple[str, ...] = ()
    extensions: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        """Freeze sequence and extension values after initialization."""
        object.__setattr__(self, "terminal_paths", tuple(self.terminal_paths))
        object.__setattr__(self, "feature_paths", tuple(self.feature_paths))
        object.__setattr__(self, "extensions", freeze_mapping(self.extensions))


@dataclass(frozen=True)
class BoardMetadataCatalog:
    """Immutable PCB metadata snapshot with source-neutral query indexes.

    Args:
        boards: Board roots represented by the catalog.
        components: Placed component records.
        prototypes: Reusable package and model records.
        terminals: Component terminal and contact records.
        pads: Board pad records.
        holes: Board hole records.
        paste_deposits: Solder-paste deposit records.
        nets: Electrical net records.
        capabilities: Queries supported by this catalog snapshot.
        diagnostics: Validation and normalization messages.

    Raises:
        ValidationError: If prim paths or component identities conflict.

    """

    boards: tuple[BoardRecord, ...]
    components: tuple[ComponentRecord, ...] = ()
    prototypes: tuple[PrototypeRecord, ...] = ()
    terminals: tuple[TerminalRecord, ...] = ()
    pads: tuple[PadRecord, ...] = ()
    holes: tuple[HoleRecord, ...] = ()
    paste_deposits: tuple[PasteDepositRecord, ...] = ()
    nets: tuple[NetRecord, ...] = ()
    capabilities: frozenset[MetadataCapability] = frozenset()
    diagnostics: tuple[Diagnostic, ...] = ()
    _by_prim_path: Mapping[str, object] = field(init=False, repr=False, compare=False)
    _component_by_identity: Mapping[tuple[str, str], ComponentRecord] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _components_by_type: Mapping[str, tuple[ComponentRecord, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _duplicate_terminals: Mapping[tuple[str, str], tuple[TerminalRecord, ...]] = field(
        init=False, repr=False, compare=False
    )
    _tht_geometry_metadata_gaps: Mapping[str, tuple[str, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Freeze record collections and build query indexes.

        Raises:
            ValidationError: If prim paths or component identities conflict.

        """
        for name in (
            "boards",
            "components",
            "prototypes",
            "terminals",
            "pads",
            "holes",
            "paste_deposits",
            "nets",
            "diagnostics",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))

        paths: dict[str, object] = {}
        for record in self._path_records():
            path = getattr(record, "prim_path", None) or getattr(
                record,
                "root_path",
                None,
            )
            if not path:
                continue
            existing = paths.get(path)
            if existing is not None and existing != record:
                raise ValidationError(f"duplicate metadata prim path {path!r}")
            paths[path] = record

        identities: dict[tuple[str, str], ComponentRecord] = {}
        typed: defaultdict[str, list[ComponentRecord]] = defaultdict(list)
        for component in self.components:
            identity = (component.board_root, component.refdes)
            if identity in identities:
                raise ValidationError(
                    f"duplicate component identity {identity!r} at "
                    f"{identities[identity].prim_path} and {component.prim_path}",
                )
            identities[identity] = component
            for key in (
                component.component_type,
                component.family,
                component.package_name,
                component.package_type,
                component.footprint_id,
            ):
                if key:
                    typed[key].append(component)

        object.__setattr__(self, "_by_prim_path", MappingProxyType(paths))
        object.__setattr__(self, "_component_by_identity", MappingProxyType(identities))
        object.__setattr__(
            self,
            "_components_by_type",
            MappingProxyType({key: tuple(value) for key, value in typed.items()}),
        )

        terminals_by_identity: defaultdict[tuple[str, str], list[TerminalRecord]] = (
            defaultdict(list)
        )
        for terminal in self.terminals:
            if terminal.number:
                terminals_by_identity[
                    (terminal.component_path, terminal.number)
                ].append(terminal)
        object.__setattr__(
            self,
            "_duplicate_terminals",
            MappingProxyType(
                {
                    identity: tuple(terminals)
                    for identity, terminals in terminals_by_identity.items()
                    if len(terminals) > 1
                },
            ),
        )

        holes_by_pad = {
            hole.pad_prim_path: hole for hole in self.holes if hole.pad_prim_path
        }
        tht_geometry_metadata_gaps = {}
        for pad in self.pads:
            missing = []
            terminal = paths.get(pad.contact_prim_path or "")
            if not isinstance(terminal, TerminalRecord) or not terminal.cross_section:
                missing.append("pin.cross_section")
            if not pad.contour:
                missing.append("pad.contour")
            hole = holes_by_pad.get(pad.prim_path)
            if hole is None:
                missing.append("hole.association")
            else:
                if hole.center_mm is None:
                    missing.append("hole.center_mm")
                if hole.dimensions_mm is None:
                    missing.append("hole.dimensions_mm")
                if not hole.shape:
                    missing.append("hole.shape")
            if missing:
                tht_geometry_metadata_gaps[pad.prim_path] = tuple(missing)
        object.__setattr__(
            self,
            "_tht_geometry_metadata_gaps",
            MappingProxyType(tht_geometry_metadata_gaps),
        )

    def _path_records(self) -> Iterable[object]:
        """Yield records that may be indexed by a prim path.

        Yields:
            Records carrying a prim path or board-root path.

        """
        yield from self.boards
        yield from self.components
        yield from self.prototypes
        yield from self.terminals
        yield from self.pads
        yield from self.holes
        yield from self.paste_deposits

    def require(self, *capabilities: MetadataCapability) -> None:
        """Require metadata capabilities for a downstream operation.

        Args:
            *capabilities: Capabilities the operation requires.

        Raises:
            ValidationError: If any requested capability is unavailable.

        """
        missing = sorted(
            capability.value
            for capability in capabilities
            if capability not in self.capabilities
        )
        if missing:
            raise ValidationError(
                "board metadata is missing required capabilities: "
                f"{', '.join(missing)}",
            )

    def by_prim_path(self, prim_path: str) -> object | None:
        """Return the record authored at ``prim_path``.

        Args:
            prim_path: Absolute prim path to query.

        Returns:
            Matching metadata record, or None if the path is unknown.

        """
        return self._by_prim_path.get(str(prim_path))

    def component(self, *, board_root: str, refdes: str) -> ComponentRecord | None:
        """Return one component by board root and reference designator.

        Args:
            board_root: Absolute path of the owning board root.
            refdes: Component reference designator.

        Returns:
            Matching component, or None if the identity is unknown.

        """
        return self._component_by_identity.get((board_root, refdes))

    def components_by_type(self, value: str) -> tuple[ComponentRecord, ...]:
        """Return components matching type, family, or package name.

        Args:
            value: Type, family, or package value to match exactly.

        Returns:
            Components matching the supplied value.

        """
        return self._components_by_type.get(value, ())

    def terminals_for_component(
        self, component: ComponentRecord | str
    ) -> tuple[TerminalRecord, ...]:
        """Return terminals owned by a component record or prim path.

        Args:
            component: Component record or absolute component prim path.

        Returns:
            Terminal records owned by the component.

        """
        path = (
            component.prim_path if isinstance(component, ComponentRecord) else component
        )
        return tuple(item for item in self.terminals if item.component_path == path)

    def pads_for_component(
        self, component: ComponentRecord | str
    ) -> tuple[PadRecord, ...]:
        """Return pads owned by a component record or prim path.

        Args:
            component: Component record or absolute component prim path.

        Returns:
            Pad records owned by the component.

        """
        path = (
            component.prim_path if isinstance(component, ComponentRecord) else component
        )
        return tuple(item for item in self.pads if item.component_path == path)

    def paste_for_component(
        self,
        component: ComponentRecord | str,
    ) -> tuple[PasteDepositRecord, ...]:
        """Return paste deposits associated with a component.

        Args:
            component: Component record or absolute component prim path.

        Returns:
            Paste-deposit records associated with the component.

        """
        path = (
            component.prim_path if isinstance(component, ComponentRecord) else component
        )
        return tuple(
            item for item in self.paste_deposits if item.component_path == path
        )

    def duplicate_terminal_number_groups(
        self,
        component: ComponentRecord | str,
    ) -> tuple[tuple[str, tuple[TerminalRecord, ...]], ...]:
        """Return duplicate terminal-number groups for one component.

        Args:
            component: Component record or absolute component prim path.

        Returns:
            Sorted terminal numbers and their conflicting terminal records.

        """
        path = (
            component.prim_path if isinstance(component, ComponentRecord) else component
        )
        return tuple(
            (number, terminals)
            for (component_path, number), terminals in sorted(
                self._duplicate_terminals.items(),
            )
            if component_path == path
        )

    def tht_geometry_metadata_gaps(
        self,
        pad: PadRecord | str,
    ) -> tuple[str, ...]:
        """Return missing normalized THT geometry fields for one pad.

        Args:
            pad: Pad record or absolute pad prim path.

        Returns:
            Missing semantic field names, or an empty tuple when complete.

        """
        path = pad.prim_path if isinstance(pad, PadRecord) else pad
        return self._tht_geometry_metadata_gaps.get(path, ())
