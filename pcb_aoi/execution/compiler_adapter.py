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

"""Own PCB-AOI declaration validation and JobOrder lowering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cdf.execution.compiler_adapters import (
    DomainCompilerAdapter,
    register_compiler_adapter,
)
from cdf.execution.job_order_types import JobOrder
from cdf.job import _JobDeclaration

_STRUCTURAL_OPERATION_TYPES = frozenset(
    {
        "operation.shift.apply",
        "operation.tombstone.apply",
        "operation.sideflip.apply",
        "operation.reverse_polarity.apply",
        "operation.missing.apply",
    },
)
_GEOMETRY_OPERATION_TYPES = frozenset({"operation.pin_variation.apply"})
_PIN_TIP_OPERATION_TYPE = "operation.pin_tip.generate"
_POST_SOLDER_OPERATION_TYPES = frozenset(
    {"operation.pin_cut.apply", _PIN_TIP_OPERATION_TYPE},
)
_SOLDER_CONFIGURATION_TYPE = "operation.solder_joints.configure"
_SOLDER_DEFECTS_CONFIGURATION_TYPE = "operation.solder_defects.configure"
_CONFIGURATION_ORDER_TYPES = frozenset(
    {_SOLDER_CONFIGURATION_TYPE, _SOLDER_DEFECTS_CONFIGURATION_TYPE},
)
_COMPONENT_CONSUMER_TYPES = frozenset(
    {
        *_STRUCTURAL_OPERATION_TYPES,
        _SOLDER_CONFIGURATION_TYPE,
        _SOLDER_DEFECTS_CONFIGURATION_TYPE,
        "annotations.components.apply",
        "annotations.geometry.apply",
        "capture.roi_selection.configure",
        "hole_plane.generate",
        "operation.pin_cut.apply",
        "operation.pin_tip.generate",
        "operation.pin_variation.apply",
    },
)


def _ensure_component_selection(
    declarations: tuple[_JobDeclaration, ...],
) -> tuple[_JobDeclaration, ...]:
    """Insert an unrestricted PCB component selection when required."""
    if any(item.order_type == "scene.components.select" for item in declarations):
        return declarations
    if not any(item.order_type in _COMPONENT_CONSUMER_TYPES for item in declarations):
        return declarations
    component_sidecar = next(
        (
            declaration
            for declaration in declarations
            if declaration.order_type == "scene.sidecar.load"
            and declaration.parameters.get("kind") == "pcb.components"
        ),
        None,
    )
    source = (
        str(component_sidecar.parameters["name"])
        if component_sidecar is not None
        else "usd"
    )
    selection = _JobDeclaration(
        order_id="scene.components.select",
        order_type="scene.components.select",
        parameters={"source": source, "types": None, "component_filter": {}},
    )
    prerequisite_id = (
        component_sidecar.order_id if component_sidecar is not None else "scene.load"
    )
    insertion = next(
        (
            index + 1
            for index, declaration in enumerate(declarations)
            if declaration.order_id == prerequisite_id
        ),
        len(declarations),
    )
    return (*declarations[:insertion], selection, *declarations[insertion:])


def _sidecar_dependency(
    declaration: _JobDeclaration,
) -> tuple[str, str] | None:
    """Return a PCB sidecar prerequisite declared by an order."""
    source = None
    if declaration.order_type in {
        "scene.components.select",
        "annotations.components.apply",
    }:
        source = declaration.parameters.get("source")
    elif declaration.order_type == "operation.tombstone.apply":
        source = declaration.parameters.get("eligibility_source")
    elif declaration.order_type == _SOLDER_CONFIGURATION_TYPE:
        source = declaration.parameters.get("source")
    if source is None or source == "usd":
        return None
    return f"scene.sidecar.{source}", "pcb.components"


def _validate_declarations(
    declarations: tuple[_JobDeclaration, ...],
    positions: Mapping[str, int],
) -> Sequence[str]:
    """Return PCB-AOI prerequisite and operation-combination problems."""
    problems = []
    by_id = {declaration.order_id: declaration for declaration in declarations}
    for declaration in declarations:
        prerequisite = _sidecar_dependency(declaration)
        if prerequisite is None:
            continue
        sidecar_id, expected_kind = prerequisite
        sidecar = by_id.get(sidecar_id)
        sidecar_name = sidecar_id.removeprefix("scene.sidecar.")
        if sidecar is None:
            problems.append(
                f"{declaration.order_type} requires sidecar {sidecar_name!r} "
                "to be attached first",
            )
            continue
        if positions[sidecar_id] > positions[declaration.order_id]:
            problems.append(
                f"{declaration.order_type} requires sidecar {sidecar_name!r} "
                "to be attached before it is declared",
            )
        if sidecar.parameters.get("kind") != expected_kind:
            problems.append(
                f"{declaration.order_type} requires sidecar {sidecar_name!r} "
                f"with kind {expected_kind!r}",
            )

    selection_position = positions.get("scene.components.select")
    operation_declarations = tuple(
        declaration
        for declaration in declarations
        if declaration.order_type in _STRUCTURAL_OPERATION_TYPES
    )
    has_missing = any(
        declaration.order_type == "operation.missing.apply"
        for declaration in operation_declarations
    )
    if has_missing and len(operation_declarations) > 1:
        problems.append(
            "operation.missing.apply cannot be combined with transform "
            "structural operations",
        )
    if operation_declarations and selection_position is None:
        problems.append(
            "structural operations require scene.select_components(...) first",
        )
    if selection_position is not None:
        for declaration in declarations:
            needs_selection = (
                declaration.order_type in _STRUCTURAL_OPERATION_TYPES
                or declaration.order_type == "annotations.components.apply"
            )
            if needs_selection and selection_position > positions[declaration.order_id]:
                problems.append(
                    f"{declaration.order_type} requires component selection "
                    "to be declared first",
                )
    return problems


def _declaration_dependencies(
    declaration: _JobDeclaration,
    positions: Mapping[str, int],
    _declarations: tuple[_JobDeclaration, ...],
) -> Sequence[str]:
    """Return PCB-AOI prerequisites for one declaration."""
    dependencies = []
    sidecar = _sidecar_dependency(declaration)
    if sidecar is not None:
        dependencies.append(sidecar[0])
    if (
        declaration.order_type in _COMPONENT_CONSUMER_TYPES
        and "scene.components.select" in positions
    ):
        dependencies.append("scene.components.select")
    solder_enabled = "operation.solder_joints" in positions
    if solder_enabled and declaration.order_type.startswith("capture."):
        dependencies.append("solder_joints.attach")
    if declaration.order_type == _PIN_TIP_OPERATION_TYPE:
        for prerequisite in ("operation.pin_variation", "operation.pin_cut"):
            if prerequisite in positions:
                dependencies.append(prerequisite)
    return tuple(dependencies)


def _produces_output(declarations: tuple[_JobDeclaration, ...]) -> bool:
    """Return whether solder configuration persists a stage artifact."""
    return any(
        declaration.order_type == _SOLDER_CONFIGURATION_TYPE
        and declaration.parameters.get("serialization") != "none"
        for declaration in declarations
    )


def _expand_orders(
    orders: list[JobOrder],
    declarations: tuple[_JobDeclaration, ...],
    positions: Mapping[str, int],
) -> list[JobOrder]:
    """Expand PCB solder configuration into its executable lifecycle."""
    solder_declaration = next(
        (
            declaration
            for declaration in declarations
            if declaration.order_type == _SOLDER_CONFIGURATION_TYPE
        ),
        None,
    )
    if solder_declaration is None:
        return orders
    solder_defects = tuple(
        declaration
        for declaration in declarations
        if declaration.order_type == _SOLDER_DEFECTS_CONFIGURATION_TYPE
    )
    solder_parameters = dict(solder_declaration.parameters)
    source = str(solder_parameters["source"])
    structural_order_ids = tuple(
        order.order_id
        for order in orders
        if order.order_type in _STRUCTURAL_OPERATION_TYPES
    )
    geometry_order_ids = tuple(
        order.order_id
        for order in orders
        if order.order_type in _GEOMETRY_OPERATION_TYPES
    )
    post_solder_orders = tuple(
        order for order in orders if order.order_type in _POST_SOLDER_OPERATION_TYPES
    )
    expanded = [order for order in orders if order not in post_solder_orders]
    discovery_dependencies = ["scene.load"]
    if source != "usd":
        discovery_dependencies.append(f"scene.sidecar.{source}")
    if "scene.components.select" in positions:
        discovery_dependencies.append("scene.components.select")
    expanded.append(
        JobOrder(
            order_id="solder_joints.discover",
            order_type="solder_joints.discover",
            parameters=solder_parameters,
            depends_on=tuple(discovery_dependencies),
        ),
    )
    solder_defect_order_ids: tuple[str, ...] = ()
    if solder_defects:
        expanded.append(
            JobOrder(
                order_id="solder_defects.apply",
                order_type="solder_defects.apply",
                parameters={
                    "entries": [dict(item.parameters) for item in solder_defects],
                    "source": source,
                },
                depends_on=("solder_joints.discover", *structural_order_ids),
            ),
        )
        solder_defect_order_ids = ("solder_defects.apply",)
    expanded.extend(
        (
            JobOrder(
                order_id="solder_joints.collect",
                order_type="solder_joints.collect",
                parameters=solder_parameters,
                depends_on=(
                    "solder_joints.discover",
                    *structural_order_ids,
                    *geometry_order_ids,
                    *solder_defect_order_ids,
                ),
            ),
            JobOrder(
                order_id="solder_joints.generate",
                order_type="solder_joints.generate",
                parameters=solder_parameters,
                depends_on=("solder_joints.collect",),
            ),
            JobOrder(
                order_id="solder_joints.attach",
                order_type="solder_joints.attach",
                parameters=solder_parameters,
                depends_on=("solder_joints.generate",),
            ),
        ),
    )
    expanded.extend(
        JobOrder(
            order_id=order.order_id,
            order_type=order.order_type,
            parameters=order.parameters,
            depends_on=(*order.depends_on, "solder_joints.attach"),
        )
        for order in post_solder_orders
    )
    return expanded


ADAPTER = DomainCompilerAdapter(
    prepare_declarations=_ensure_component_selection,
    validate_declarations=_validate_declarations,
    declaration_dependencies=_declaration_dependencies,
    configuration_order_types=_CONFIGURATION_ORDER_TYPES,
    produces_output=_produces_output,
    expand_orders=_expand_orders,
)


def register() -> None:
    """Register PCB-AOI compiler ownership with CDF."""
    register_compiler_adapter("pcb-aoi", ADAPTER)
