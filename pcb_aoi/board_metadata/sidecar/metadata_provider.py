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

"""Legacy ``components.json`` provider for the canonical metadata catalog."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from cdf.errors import ValidationError

from ..metadata_catalog import (
    BoardMetadataCatalog,
    BoardRecord,
    ComponentRecord,
    Diagnostic,
    MetadataCapability,
    MetadataOrigin,
    MetadataSource,
    PadRecord,
    PasteDepositRecord,
    PrototypeRecord,
    TerminalRecord,
)
from ..metadata_provider import BoardMetadataRequest

_SUPPORTED_VERSIONS = frozenset({"v1", "v1.1"})


def _object_records(
    data: Mapping[str, object],
    key: str,
) -> tuple[Mapping[str, object], ...]:
    """Return object records from a sidecar array field.

    Args:
        data: Decoded sidecar root object.
        key: Array field to read.

    Returns:
        Object records in source order.

    Raises:
        ValidationError: If the field is not an array of objects.

    """
    value = data.get(key, ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationError(f"components sidecar {key!r} must be an array")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValidationError(
            f"components sidecar {key!r} contains a non-object",
        )
    return tuple(value)


def _strings(value: object) -> tuple[str, ...]:
    """Normalize a sequence into nonempty strings.

    Args:
        value: Candidate sequence value.

    Returns:
        Nonempty string values, or an empty tuple for non-sequences.

    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if item is not None and str(item))


def _unknown(record: Mapping[str, object], known: set[str]) -> dict[str, object]:
    """Return source fields not represented by the known field set.

    Args:
        record: Source object to inspect.
        known: Field names consumed by the provider.

    Returns:
        Source-specific extension values keyed by field name.

    """
    return {str(key): value for key, value in record.items() if key not in known}


def _prototype_id(record: Mapping[str, object]) -> str:
    """Build a stable prototype identity from a component group.

    Args:
        record: Component-group sidecar record.

    Returns:
        Prototype path when available, otherwise a source-and-name identity.

    """
    path = str(record.get("prototype_path") or "")
    if path:
        return path
    source = str(record.get("source") or "")
    name = str(record.get("name") or "")
    return f"{source}#{name}"


def _rebase_document(
    data: Mapping[str, object],
    document_root: str,
    board_root: str,
) -> Mapping[str, object]:
    """Move every prim path in a sidecar onto ``board_root``.

    A sidecar records paths as they are authored in the board's OWN USD file,
    where the board root is a child of the pseudo-root
    (``/Board_X/Components/Top/R1``). Reading it against a stage that
    references the board somewhere else, such as under ``/World``, leaves
    every recorded path naming a prim that does not
    exist.

    The rewrite is done on the document rather than on the component pool
    downstream, because the pool is not the only thing keyed by these paths:
    the tombstone filter looks candidates up in a contact index built from
    the same records, and rebasing one but not the other turns "moved
    nothing" into "found nothing".

    Only strings under ``document_root`` are touched, and only when the two
    roots actually differ, so a batch run -- which opens the board AS the
    stage, where the recorded paths are already correct -- is untouched.

    Args:
        data: Decoded sidecar document.
        document_root: ``board_prim_path`` as written in the document.
        board_root: Board root the document is being read against.

    Returns:
        The document, rebased, or the original object when nothing moved.

    """
    if not document_root or document_root == board_root:
        return data
    old = document_root.rstrip("/")
    new = board_root.rstrip("/")

    def rebase(value: object) -> object:
        if isinstance(value, str):
            if value == old:
                return new
            if value.startswith(old + "/"):
                return new + value[len(old) :]
            return value
        if isinstance(value, Mapping):
            return {key: rebase(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [rebase(item) for item in value]
        return value

    rebased = rebase(data)
    return rebased if isinstance(rebased, Mapping) else data


class SidecarMetadataProvider:
    """Normalize supported v1/v1.1 component sidecars into a catalog."""

    def from_path(self, path: str | Path) -> BoardMetadataCatalog:
        """Load and normalize a sidecar from a local filesystem path.

        Args:
            path: Path to a supported components sidecar.

        Returns:
            Immutable catalog populated from the sidecar.

        Raises:
            ValidationError: If the file is unreadable, malformed, or uses an
                unsupported schema.

        Example:

        .. code-block:: python

            from pcb_aoi.board_metadata.sidecar import SidecarMetadataProvider

            provider = SidecarMetadataProvider()
            catalog = provider.from_path("board.components.json")  # doctest: +SKIP
            print(len(catalog.components))  # doctest: +SKIP

        """
        source_path = Path(path)
        try:
            document = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValidationError(
                f"could not read components sidecar {path}: {exc}",
            ) from exc
        return self.from_json(document, source_identifier=str(source_path))

    def from_json(
        self,
        document: str,
        *,
        source_identifier: str = "<components-sidecar>",
    ) -> BoardMetadataCatalog:
        """Decode and normalize a component-sidecar JSON document.

        Args:
            document: Complete JSON document text.
            source_identifier: Stable path or label used for provenance.

        Returns:
            Immutable catalog populated from the sidecar.

        Raises:
            ValidationError: If the document is malformed or its root is not
                an object.

        Example:

        .. code-block:: python

            from pcb_aoi.board_metadata.sidecar import SidecarMetadataProvider

            provider = SidecarMetadataProvider()
            catalog = provider.from_json(  # doctest: +SKIP
                document,
                source_identifier="board.components.json",
            )
            print(len(catalog.components))  # doctest: +SKIP

        """
        try:
            data = json.loads(document)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"could not decode components sidecar {source_identifier}: {exc}",
            ) from exc
        if not isinstance(data, Mapping):
            raise ValidationError(
                f"components sidecar root must be an object: {source_identifier}",
            )
        return self.load(
            BoardMetadataRequest(
                source_identifier=source_identifier,
                sidecar_data=data,
            ),
        )

    def load(self, request: BoardMetadataRequest) -> BoardMetadataCatalog:
        """Normalize request sidecar data into board metadata records.

        Args:
            request: Provider request containing a decoded sidecar mapping.

        Returns:
            Immutable catalog populated from the sidecar mapping.

        Raises:
            ValidationError: If required fields are absent, malformed, or use
                an unsupported schema.

        """
        data = request.sidecar_data
        if not isinstance(data, Mapping):
            raise ValidationError("sidecar metadata provider requires sidecar_data")
        schema_version = str(data.get("schema_version") or "")
        if schema_version not in _SUPPORTED_VERSIONS:
            raise ValidationError(
                "unsupported components sidecar schema_version "
                f"{schema_version!r} (supported: {sorted(_SUPPORTED_VERSIONS)})",
            )
        board_root = str(request.board_root or data.get("board_prim_path") or "")
        if not board_root.startswith("/"):
            raise ValidationError("components sidecar board_prim_path must be absolute")
        data = _rebase_document(
            data, str(data.get("board_prim_path") or ""), board_root
        )
        board_name = str(data.get("board_name") or board_root.rsplit("/", 1)[-1])
        source = MetadataSource(
            origin=MetadataOrigin.SIDECAR,
            identifier=request.source_identifier or "<components-sidecar>",
            schema_version=schema_version,
        )

        group_records = _object_records(data, "components")
        instance_records = _object_records(data, "component_instances")
        groups_by_instance: dict[str, Mapping[str, object]] = {}
        prototypes_by_id: dict[str, PrototypeRecord] = {}
        for group in group_records:
            prototype_id = _prototype_id(group)
            prototype = PrototypeRecord(
                prototype_id=prototype_id,
                source=source,
                name=str(group.get("name") or ""),
                family=str(group.get("family") or "") or None,
                prim_path=str(group.get("prototype_path") or "") or None,
                asset_path=str(group.get("source") or "") or None,
                body_meshes=_strings(group.get("body_meshes")),
                contact_meshes=_strings(group.get("solder_contact_meshes")),
                extensions=_unknown(
                    group,
                    {
                        "name",
                        "family",
                        "source",
                        "prototype_path",
                        "side",
                        "process_class",
                        "prim_path",
                        "instance_paths",
                        "body_meshes",
                        "solder_contact_meshes",
                    },
                ),
            )
            existing = prototypes_by_id.get(prototype_id)
            if existing is not None and existing != prototype:
                raise ValidationError(
                    f"conflicting prototype records for {prototype_id!r}",
                )
            prototypes_by_id[prototype_id] = prototype
            for path in _strings(group.get("instance_paths")):
                if path in groups_by_instance:
                    raise ValidationError(
                        f"component instance {path!r} belongs to multiple groups",
                    )
                groups_by_instance[path] = group

        components: list[ComponentRecord] = []
        terminals: list[TerminalRecord] = []
        pads: list[PadRecord] = []
        paste: list[PasteDepositRecord] = []
        diagnostics: list[Diagnostic] = []
        for instance in instance_records:
            path = str(instance.get("path") or "")
            refdes = str(instance.get("refdes") or "")
            if not path or not refdes:
                raise ValidationError("component instance requires path and refdes")
            group = groups_by_instance.get(path)
            if group is None:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        code="sidecar.component_group_missing",
                        message="component instance is not listed by a prototype group",
                        entity_path=path,
                    ),
                )
                group = {}
            raw_process = instance.get("process_class", group.get("process_class"))
            process_classes = (str(raw_process),) if raw_process else ()
            prototype_id = _prototype_id(group) if group else None
            body_paths = _strings(instance.get("body_prims"))
            contact_paths = _strings(instance.get("solder_contact_prims"))
            component = ComponentRecord(
                board_root=board_root,
                refdes=refdes,
                prim_path=path,
                source=source,
                component_type=str(instance.get("type") or "") or None,
                family=str(instance.get("family") or group.get("family") or "") or None,
                package_name=str(group.get("name") or "") or None,
                side=str(instance.get("side") or group.get("side") or "") or None,
                process_class=str(raw_process) if raw_process else None,
                process_classes=process_classes,
                prototype_id=prototype_id,
                group_path=str(group.get("prim_path") or "") or None,
                body_prim_paths=body_paths,
                contact_prim_paths=contact_paths,
                extensions=_unknown(
                    instance,
                    {
                        "path",
                        "refdes",
                        "side",
                        "process_class",
                        "family",
                        "type",
                        "body_prims",
                        "solder_contact_prims",
                        "solder_paste_prims",
                        "solder_paste_usdc_paths",
                        "solder_pin_pad_associations",
                    },
                ),
            )
            components.append(component)
            terminals.extend(
                TerminalRecord(
                    board_root=board_root,
                    component_path=path,
                    prim_path=contact_path,
                    source=source,
                    solderable=True,
                    process_class=process_classes[0] if process_classes else None,
                )
                for contact_path in contact_paths
            )
            paste_assets = _strings(instance.get("solder_paste_usdc_paths"))
            paste.extend(
                PasteDepositRecord(
                    board_root=board_root,
                    prim_path=paste_path,
                    source=source,
                    component_path=path,
                    side=component.side,
                    site_kind="fitted",
                    asset_paths=paste_assets,
                )
                for paste_path in _strings(instance.get("solder_paste_prims"))
            )
            associations = instance.get("solder_pin_pad_associations", ())
            if not isinstance(associations, Sequence) or isinstance(
                associations, (str, bytes)
            ):
                raise ValidationError("solder_pin_pad_associations must be an array")
            for association in associations:
                if not isinstance(association, Mapping):
                    raise ValidationError(
                        "solder_pin_pad_associations contains a non-object"
                    )
                pad_path = str(association.get("solder_pad_prim") or "")
                if not pad_path:
                    raise ValidationError(
                        "solder pin-to-pad association requires solder_pad_prim"
                    )
                center = association.get("pin_center")
                pin_center = None
                if (
                    isinstance(center, Sequence)
                    and not isinstance(center, (str, bytes))
                    and len(center) == 2
                ):
                    pin_center = (float(center[0]), float(center[1]))
                pads.append(
                    PadRecord(
                        board_root=board_root,
                        prim_path=pad_path,
                        source=source,
                        component_path=path,
                        contact_prim_path=str(
                            association.get("solder_contact_prim") or ""
                        )
                        or None,
                        side="Bottom"
                        if component.side == "Top"
                        else "Top"
                        if component.side == "Bottom"
                        else None,
                        occupied=True,
                        pin_center=pin_center,
                        extensions=_unknown(
                            association,
                            {"solder_contact_prim", "solder_pad_prim", "pin_center"},
                        ),
                    ),
                )

        capabilities = {
            MetadataCapability.COMPONENTS,
            MetadataCapability.COMPONENT_PROTOTYPES,
        }
        if terminals:
            capabilities.add(MetadataCapability.TERMINALS)
            capabilities.add(MetadataCapability.SOLDER_CONTACTS)
        if pads:
            # PADS says the records exist at all; the association capability
            # says they are tied to a terminal. A caller that only wants to
            # label pad geometry needs the first, not the second.
            capabilities.add(MetadataCapability.PADS)
            capabilities.add(MetadataCapability.TERMINAL_PAD_ASSOCIATIONS)
        if paste:
            capabilities.add(MetadataCapability.SOLDER_PASTE_ASSOCIATIONS)

        board_extensions = _unknown(
            data,
            {
                "schema_version",
                "added_features",
                "board_name",
                "board_prim_path",
                "components",
                "component_instances",
                "solder_paste_layers",
                "solder_paste_usdc_paths_all",
                "solder_paste_sources",
            },
        )
        if data.get("added_features") is not None:
            board_extensions["added_features"] = data["added_features"]
        for key in (
            "solder_paste_layers",
            "solder_paste_usdc_paths_all",
            "solder_paste_sources",
        ):
            if data.get(key) is not None:
                board_extensions[key] = data[key]

        return BoardMetadataCatalog(
            boards=(
                BoardRecord(
                    root_path=board_root,
                    name=board_name,
                    source=source,
                    extensions=board_extensions,
                ),
            ),
            components=tuple(components),
            prototypes=tuple(prototypes_by_id.values()),
            terminals=tuple(terminals),
            pads=tuple(pads),
            paste_deposits=tuple(paste),
            capabilities=frozenset(capabilities),
            diagnostics=tuple(diagnostics),
        )
