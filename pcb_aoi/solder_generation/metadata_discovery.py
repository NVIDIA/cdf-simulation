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

"""Discover catalog components eligible for solder-joint generation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from pcb_aoi.board_metadata import BoardMetadataCatalog, ComponentRecord


@dataclass(frozen=True)
class SolderComponentCandidate:
    """Reusable solder-generation input discovered from board metadata.

    Args:
        prototype_id: Stable human-readable prototype identity.
        family: Broader component family or package name.
        side: Board side containing the component group.
        process_class: Explicit assembly process, when available.
        prototype_path: USD path to the reusable component prototype.
        source: Asset path containing the prototype geometry.
        instance_paths: USD paths of instances that reuse the prototype.
        solder_contact_meshes: Prototype-local solder-contact mesh names.
        sample_instance_path: Instance inspected for paste associations.
        sample_refdes: Reference designator of the inspected instance.
        solder_paste_prims: Paste prims associated with the inspected instance.
        solder_paste_usdc_paths: Assets containing the paste prims.

    """

    prototype_id: str
    family: str
    side: str
    process_class: str | None
    prototype_path: str
    source: str
    instance_paths: tuple[str, ...]
    solder_contact_meshes: tuple[str, ...]
    sample_instance_path: str
    sample_refdes: str
    solder_paste_prims: tuple[str, ...]
    solder_paste_usdc_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation.

        Returns:
            Mapping of candidate field names to JSON-compatible values.

        """
        return asdict(self)


#: Process classes whose solder joints the solvers know how to build.
_SOLDER_PROCESS_CLASSES = frozenset({"SMT", "through_hole"})


def solder_process_class(component: ComponentRecord) -> str | None:
    """Return the assembly process that owns a component's solder joints.

    A mixed-process part (some exporters author
    ``pcba:component:processClass = "Mixed"`` with
    ``processClasses = [through_hole, SMT]``, e.g. a USB-C receptacle whose
    shield legs are through-hole and whose signal pins are SMT) is soldered
    like a through-hole part: the solver builds joints for its plated holes
    on the solder side, and its paste-side pins never reach the THT
    association step. Aggregate classes that carry no through-hole member
    pass through unchanged so callers can still reject them.

    Args:
        component: Placed component record.

    Returns:
        ``"SMT"``, ``"through_hole"``, None for an unclassified component,
        or the unrecognized aggregate class verbatim.

    Example:

    .. code-block:: python

        solder_process_class(mixed_usb_c)  # -> "through_hole"

    """
    process_class = component.process_class
    if process_class is None or process_class in _SOLDER_PROCESS_CLASSES:
        return process_class
    if "through_hole" in component.process_classes:
        return "through_hole"
    return process_class


def _candidate_key(component: ComponentRecord) -> tuple[str, str, str]:
    """Return the source-neutral grouping key for a placed component."""
    return (
        component.prototype_id or component.package_name or component.prim_path,
        component.side or "",
        solder_process_class(component) or "",
    )


def discover_solder_components(
    catalog: BoardMetadataCatalog,
    component_types: Iterable[str] | None = None,
    *,
    component_paths: Iterable[str] | None = None,
    require_sample_paste: bool = True,
) -> tuple[SolderComponentCandidate, ...]:
    """Discover prototype/side groups eligible for solder generation.

    Args:
        catalog: Source-neutral board metadata catalog.
        component_types: Optional package, footprint, type, or family allow-list.
        component_paths: Optional exact component-path population.
        require_sample_paste: Exclude groups without associated source paste.

    Returns:
        Eligible candidates sorted by prototype, side, and prototype path.

    """
    requested = {str(value) for value in component_types or () if str(value)}
    allowed_paths = (
        None
        if component_paths is None
        else {str(value) for value in component_paths if str(value)}
    )
    prototype_by_id = {
        prototype.prototype_id: prototype for prototype in catalog.prototypes
    }
    grouped: defaultdict[tuple[str, str, str], list[ComponentRecord]] = defaultdict(
        list,
    )
    for component in catalog.components:
        if allowed_paths is not None and component.prim_path not in allowed_paths:
            continue
        values = {
            component.component_type,
            component.family,
            component.package_name,
            component.package_type,
            component.footprint_id,
        }
        if requested and requested.isdisjoint(value for value in values if value):
            continue
        if solder_process_class(component) not in {None, *_SOLDER_PROCESS_CLASSES}:
            # An aggregate class with no through-hole member has no joint
            # recipe; a "Mixed" part resolves to through_hole above and is
            # a legitimate THT solder target (e.g. a USB-C receptacle or an M.2 socket).
            continue
        grouped[_candidate_key(component)].append(component)

    candidates = []
    for (prototype_key, side, process_class), components in grouped.items():
        prototype = prototype_by_id.get(components[0].prototype_id or "")
        contact_meshes = prototype.contact_meshes if prototype else ()
        if not contact_meshes and not process_class:
            continue
        sample = next(
            (
                component
                for component in components
                if catalog.paste_for_component(component)
            ),
            components[0],
        )
        paste = catalog.paste_for_component(sample)
        if require_sample_paste and not paste:
            continue
        prototype_name = (
            prototype.name
            if prototype
            else sample.package_name or sample.footprint_id or prototype_key
        )
        family = sample.family or sample.package_name or prototype_name
        candidates.append(
            SolderComponentCandidate(
                prototype_id=prototype_name,
                family=family,
                side=side,
                process_class=process_class or None,
                prototype_path=prototype.prim_path or "" if prototype else "",
                source=prototype.asset_path or "" if prototype else "",
                instance_paths=tuple(
                    component.prim_path
                    for component in sorted(
                        components,
                        key=lambda item: item.prim_path,
                    )
                ),
                solder_contact_meshes=contact_meshes,
                sample_instance_path=sample.prim_path,
                sample_refdes=sample.refdes,
                solder_paste_prims=tuple(item.prim_path for item in paste),
                solder_paste_usdc_paths=tuple(
                    dict.fromkeys(
                        asset_path for item in paste for asset_path in item.asset_paths
                    ),
                ),
            ),
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.prototype_id,
                candidate.side,
                candidate.prototype_path,
            ),
        ),
    )
