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

"""PCB-AOI's own Isaac handlers, registered with CDF's executor.

`DEVELOPMENT.md` names the pattern this follows: keep the Isaac executor
machinery in CDF, and migrate domain-specific contracts and handlers to
domain-owned adapters that register with it. Domain handlers are registered
here instead of being embedded in CDF's default Isaac registry. Contracts
and runtime implementations can then move behind this adapter independently.
"""

from __future__ import annotations

import inspect
import math
import warnings
from typing import TYPE_CHECKING

from pydantic import BaseModel

from cdf.errors import ValidationError
from cdf.execution.isaac.domain_adapters import (
    OWNERSHIP_ANCESTOR,
    OWNERSHIP_RELATIONSHIP,
    GeometryRole,
    register_geometry_roles,
    register_isaac_adapter,
)
from cdf.execution.isaac.execution_context import IsaacExecutionContext
from cdf.execution.isaac.job_order_parameters import (
    ComponentAnnotationParameters,
    ComponentAppearanceParameters,
    ComponentSelectionParameters,
    GeometryAnnotationParameters,
    HolePlaneParameters,
    MissingParameters,
    OperationLabelParameters,
    PostprocessFormatRoiOutputParameters,
    PostprocessGenerateJointRoisParameters,
    ReversePolarityParameters,
    ShiftParameters,
    SideflipParameters,
    SolderDefectParameters,
    SolderJointParameters,
    TombstoneParameters,
)
from pcb_aoi.execution.board_instances import mark_instanced_stage_mutation

from .pin_tip_contract import PinTipParameters
from .pin_variation_contract import PinVariationParameters

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cdf.execution.job_order_types import JobOrder

#: Semantic order type this adapter owns.
PIN_VARIATION_ORDER_TYPE = "operation.pin_variation.apply"
PIN_CUT_ORDER_TYPE = "operation.pin_cut.apply"
PIN_TIP_ORDER_TYPE = "operation.pin_tip.generate"

#: How far the joint's cut section sits behind the lead's, in millimetres.
#:
#: The blade cuts a tube -- the bath sleeve wraps the lead, so the solder's
#: section is a solid disc with the lead's face inside it, not a ring around
#: it. Cut both at exactly one plane and the two coplanar surfaces trade
#: pixels: on a panelised THT board, thousands of pixels of the lead's own face
#: came back labelled `solder`, in a crescent bitten out of the mask.
#:
#: A micron already settles it -- `pin` goes from 0.17% of the frame to the
#: full 3.78% that a render with the solder hidden gives. Two, for margin;
#: that is a fifth of a pixel at the ROI's 5 mm field. Nor is it a fiction:
#: solder is the softer of the two and a blade leaves it shy of the lead it
#: cut through.
SECTION_RECESS_MM = 0.002


def _require_context(context: object | None) -> IsaacExecutionContext:
    """Return the Isaac execution context, or say what arrived instead.

    Args:
        context: Context supplied by the executor.

    Returns:
        The typed Isaac execution context.

    Raises:
        ValidationError: If the executor supplied another context type.

    """
    if not isinstance(context, IsaacExecutionContext):
        message = (
            "pcb-aoi Isaac handlers require an IsaacExecutionContext, got "
            f"{type(context).__name__}"
        )
        raise ValidationError(message)
    return context


def _runtime_handler[Parameters: BaseModel](
    contract: type[Parameters],
    runtime_method: str,
):
    """Create a PCB-AOI handler for one validated runtime method."""

    async def handler(order: JobOrder, context: object | None) -> object:
        """Validate and dispatch one domain order to the Isaac runtime."""
        if order.version != 1:
            message = (
                f"{order.order_type} handler supports version 1, got {order.version}"
            )
            raise ValidationError(message)
        execution = _require_context(context)
        parameters = contract.model_validate(
            order.model_dump(mode="json")["parameters"],
        )
        method = getattr(execution.runtime, runtime_method, None)
        if not callable(method):
            message = (
                f"Isaac runtime {type(execution.runtime).__name__} does not "
                f"implement {runtime_method}()"
            )
            raise ValidationError(message)
        execution._active_order_id = order.order_id
        try:
            output = method(execution, parameters)
            if inspect.isawaitable(output):
                output = await output
        finally:
            execution._active_order_id = None
        execution.outputs[order.order_id] = output
        return output

    return handler


def _component_scoped_leads(
    execution: IsaacExecutionContext,
    leads: Sequence[object],
    catalogs: Sequence[object],
) -> list[object]:
    """Restrict discovered leads to the resolved scene component population.

    Args:
        execution: Active execution context.
        leads: Discovered through-hole leads.
        catalogs: Loaded component metadata catalogs.

    Returns:
        Leads whose component reference designator belongs to the scene pool.

    """
    parent = execution.component_parent_paths()
    if parent is None:
        return list(leads)
    parent_paths = set(parent)
    allowed_refdes = {
        component.refdes
        for catalog in catalogs
        for component in catalog.components
        if component.prim_path in parent_paths
    }
    return [lead for lead in leads if lead.refdes in allowed_refdes]


def _resolved_pin_groups(
    execution: IsaacExecutionContext,
    parameters: PinVariationParameters,
    catalogs: Sequence[object],
) -> list[dict[str, object]]:
    """Resolve normalized group filters to the pin engine's refdes selector.

    Args:
        execution: Active execution context.
        parameters: Validated pin-variation declaration.
        catalogs: Loaded component metadata catalogs.

    Returns:
        Pin-engine groups using exact component reference designators.

    Raises:
        ValidationError: If no metadata catalog is available for a filter.

    """
    groups = []
    for index, group in enumerate(parameters.groups):
        payload = group.model_dump(exclude_none=True)
        component_filter = payload.pop("component_filter", None)
        if component_filter is not None:
            if not catalogs:
                raise ValidationError(
                    "pin_variation.component_filter requires component metadata",
                )
            from pcb_aoi.board_metadata import select_components

            selected = select_components(
                catalogs[0],
                component_filter,
                parent_paths=execution.component_parent_paths(),
                where=f"operations.pin_variation[{index}].component_filter",
            )
            payload["components"] = [component.refdes for component in selected]
        groups.append(payload)
    return groups


async def handle_pin_variation(
    order: JobOrder,
    context: object | None,
) -> object:
    """Execute a version-1 ``operation.pin_variation.apply`` order.

    Thin by design: discovery, selection, sampling and the pose solve live
    under ``pcb_aoi.tht``. This resolves the stage and the run seed, hands them
    over, and returns the report.

    No override layer is written. In a JobOrder run the stage is already open
    and mutable, and ``stage.serialize`` is the declared way to persist it.

    Args:
        order: The JobOrder being executed.
        context: Isaac execution context.

    Returns:
        Per-lead requested-versus-delivered records and a summary.

    Raises:
        ValidationError: If the order version is unsupported, the board
            declares no through-hole leads, no declared group matches any lead,
            or a varied lead breaches its own plating.

    """
    from pcb_aoi.tht import (
        discover_through_hole_leads,
        plan_board_variation,
        summarise,
        write_lead_points,
    )

    if order.version != 1:
        message = f"{order.order_type} handler supports version 1, got {order.version}"
        raise ValidationError(message)
    execution = _require_context(context)
    parameters = PinVariationParameters.model_validate(order.parameters)

    stage = execution.require_stage()
    if inspect.isawaitable(stage):
        stage = await stage
    # Consume the catalogs the job already loaded rather than reading the
    # board a second time: a whole-panel load takes ~14 s and raises on
    # duplicate component identity, and a private read is the double
    # maintenance this operation is trying to stop.
    catalogs = tuple(execution.board_metadata.values())
    leads, rejected = discover_through_hole_leads(stage, catalogs)
    leads = _component_scoped_leads(execution, leads, catalogs)
    from pcb_aoi.tht.lead_discovery import pair_leads

    disagreements = getattr(pair_leads, "disagreements", ())
    if not leads:
        message = (
            "pin_variation found no through-hole leads: this board authors no "
            "'pcba:mesh:role' = 'pkg_lead' meshes (nor the earlier 'pcb:' "
            "spelling) paired to a plated 'board_hole' "
            f"({len(rejected)} candidates rejected)"
        )
        raise ValidationError(message)

    # The JOB owns the seed. An order-level one would make
    # CDFJob(random_seed=...) silently inert for this operation while every
    # other sampling order honoured it.
    seed = int(execution.plan_metadata.get("random_seed", 0))
    board_far_face_z = min(lead.hole.z_range[0] for lead in leads)
    groups = _resolved_pin_groups(execution, parameters, catalogs)
    edits, outcomes, skipped = plan_board_variation(
        leads,
        groups,
        board_far_face_z=board_far_face_z,
        seed=seed,
    )
    if not edits:
        message = (
            "pin_variation matched no poseable leads: the declared groups name "
            "parts this board does not have, or every matched lead was already "
            f"outside its own bore at rest ({len(skipped)} skipped)"
        )
        raise ValidationError(message)

    report = summarise(outcomes, skipped)
    if report["breaches"]:
        message = (
            f"pin_variation produced {report['breaches']} lead(s) outside their "
            "own plating; this is a solver failure, not a board limitation"
        )
        raise ValidationError(message)

    written = write_lead_points(stage, edits)
    execution.state["operation.pin_variation.report"] = report
    output = {
        "written": written,
        "discovered": len(leads),
        "rejected": len(rejected),
        "seed": seed,
        "catalogs": len(catalogs),
        "paired_from_catalog": sum(
            1 for lead in leads if lead.pairing_source == "catalog"
        ),
        # An authored association the lead's own geometry contradicts is an
        # asset defect worth naming, not something to resolve silently either
        # way.
        "association_disagreements": [
            {"lead": path, "authored_hole": hole, "distance_mm": round(d, 4)}
            for path, hole, d in disagreements
        ],
        **report,
    }
    execution.outputs[order.order_id] = output
    return output


async def handle_pin_cut(
    order: JobOrder,
    context: object | None,
) -> object:
    """Execute a version-1 ``operation.pin_cut.apply`` order.

    The trim runs after the joint exists, so it takes the lead and the joint
    together: one plane, one section across both materials. The solder
    generator never learns that a cut is coming, and the caller never asked for
    two operations -- one `pin_variation(...)` declares four axes and the
    compiler lifts the two that are a plane past the solder orders.

    Args:
        order: The JobOrder being executed.
        context: Isaac execution context.

    Returns:
        What was planned, what was cut, and what was skipped.

    Raises:
        ValidationError: If the order version is unsupported, or no declared
            group matches any lead this board has.

    """
    from pcb_aoi.geometry.plane_cut import signed_distance  # noqa: F401
    from pcb_aoi.tht import discover_through_hole_leads
    from pcb_aoi.tht.pin_cut_apply import cut_prims, offset_plane
    from pcb_aoi.tht.pin_cut_plan import (
        normal_for,
        plan_board_trim,
        summarise_trims,
    )
    from pcb_aoi.tht.pin_variation_apply import select_leads

    if order.version != 1:
        message = f"{order.order_type} handler supports version 1, got {order.version}"
        raise ValidationError(message)
    execution = _require_context(context)
    parameters = PinVariationParameters.model_validate(order.parameters)

    stage = execution.require_stage()
    if inspect.isawaitable(stage):
        stage = await stage
    catalogs = tuple(execution.board_metadata.values())
    leads, _ = discover_through_hole_leads(stage, catalogs)
    leads = _component_scoped_leads(execution, leads, catalogs)
    if not leads:
        raise ValidationError(
            "pin_cut found no through-hole leads: this board authors no "
            "'pcba:mesh:role' = 'pkg_lead' meshes paired to a plated hole",
        )

    seed = int(execution.plan_metadata.get("random_seed", 0))
    groups = _resolved_pin_groups(execution, parameters, catalogs)
    trims, skipped = plan_board_trim(
        leads,
        groups,
        seed=seed,
        select=select_leads,
    )
    if not trims:
        message = (
            "pin_cut matched no leads: the declared groups name parts this "
            f"board does not have ({len(skipped)} skipped)"
        )
        raise ValidationError(message)

    by_path = {lead.path: lead for lead in leads}
    assigned = _assign_joints(stage, execution, trims)
    applied: list[Mapping[str, object]] = []
    delivered: list[Mapping[str, object]] = []
    already_cut: set[str] = set()
    for trim in trims:
        lead = by_path[trim.lead_path]
        # The far face is where the lead EMERGES, not the lead's own top,
        # which sits inside the board. On a panelised THT board, taking the
        # lead's top put the plane more than a millimetre high and the blade
        # removed the entire joint rather than its tip.
        board_far_face_z = float(lead.hole.z_range[0])
        direction = -1.0
        centre = lead.hole.center
        # Half-width from the lead's OWN points rather than the bore's radius.
        # Several parts on this board author leads wider than their stated
        # bores, and the overhang has to describe the blade's reach across the
        # metal it actually meets.
        span = max(
            (
                float(max(abs(point[0] - centre[0]), abs(point[1] - centre[1])))
                for point in lead.points
            ),
            default=float(lead.hole.radius_mm),
        )
        # CLAMP THE ANGLE BY WHAT THE BOARD ALLOWS.
        #
        # The blade is a plane, and a plane keeps rising as it travels
        # outward. Measured here: 41.9 degrees through a lead 1.0 mm proud has
        # climbed 1.21 mm by the time it reaches the joint's outer radius of
        # 1.35 mm -- which puts it 0.21 mm INSIDE the board, and on the way it
        # took 99 of the fillet's 336 points with it. The joint then reads as
        # collapsed, and it is: a third of it was removed by a cut that could
        # not physically have happened.
        #
        # So the angle is bounded by the same rule the other axes already use.
        # `tilt` is bounded by the barrel and `offset` by the clearance; this
        # is bounded by the reach across the joint, and like them it reports
        # what was delivered rather than what was asked.
        reach = _joint_reach(stage, assigned.get(trim.lead_path), centre)
        cut_deg = trim.cut_deg
        cut_by = "request"
        if reach > 1e-6:
            limit = math.degrees(
                math.atan(max(0.0, trim.protrusion_mm - _BOARD_CLEARANCE_MM) / reach)
            )
            if cut_deg > limit:
                cut_deg, cut_by = limit, "joint reach"
        normal = normal_for(cut_deg, trim.cut_azimuth_deg, direction)
        delivered.append(
            {
                "lead": trim.lead_path,
                "requested_deg": round(trim.cut_deg, 3),
                "delivered_deg": round(cut_deg, 3),
                "limited_by": cut_by,
                "joint_reach_mm": round(reach, 4),
            }
        )
        overhang = span * math.tan(math.radians(cut_deg))
        plane = (
            (
                float(centre[0]),
                float(centre[1]),
                board_far_face_z + direction * (trim.protrusion_mm - overhang),
            ),
            normal,
        )
        # One blade per mesh. Two leads resolving to the same joint would cut
        # the second plane through geometry the first had already sectioned,
        # which sections the section.
        joint = assigned.get(trim.lead_path)
        joints = [joint] if joint and joint not in already_cut else []
        lead_targets = [trim.lead_path] if trim.lead_path not in already_cut else []
        already_cut.update(lead_targets)
        already_cut.update(joints)
        applied.extend(cut_prims(stage, lead_targets, plane))
        # Recess the joint's section behind the lead's.
        #
        # One blade leaves them exactly coplanar, which is right physically and
        # unusable as ground truth: measured on the board, the two sections
        # z-fight across the whole face and the mask comes back with `pin` and
        # `solder` interleaved pixel by pixel. A few microns is enough to
        # settle it, and it is not a fiction -- solder is the softer of the two
        # and a blade leaves it slightly shy of the lead it cut through.
        if joints:
            applied.extend(
                cut_prims(stage, joints, offset_plane(plane, -SECTION_RECESS_MM))
            )

    report = summarise_trims(trims, skipped, applied)
    report["joints_assigned"] = len(assigned)
    report["cut_angles"] = delivered[:20]
    report["angles_limited_by_the_joint"] = sum(
        1 for entry in delivered if entry["limited_by"] != "request"
    )
    execution.state["operation.pin_cut.report"] = report
    output = {"seed": seed, "discovered": len(leads), **report}
    execution.outputs[order.order_id] = output
    return output


#: How far below the board's far face the blade must stay clear.
#:
#: 0.05 was not enough. The fillet's surface RISES from the crest to meet the
#: board at its outer rim, so a plane that only just clears the board still
#: grazes the fillet along the way -- measured, it kept taking points off the
#: outer edge. The margin has to clear the fillet, not the board.
_BOARD_CLEARANCE_MM = 0.15


def _joint_reach(
    stage: object,
    joint_path: str | None,
    centre: tuple[float, float],
) -> float:
    """Return how far the joint extends from the lead's bore, in plan.

    That distance is what the blade has to clear. A lead half-width is not
    enough: the fillet is several times wider than the lead it sits on, and it
    is the fillet a rising plane reaches first.

    Args:
        stage: Open USD stage.
        joint_path: The joint assigned to this lead, or None.
        centre: The bore centre in plan.

    Returns:
        The reach in millimetres, or 0.0 when there is no joint to clear.

    """
    if not joint_path:
        return 0.0
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(joint_path)
    if not prim or not prim.IsValid():
        return 0.0
    bounds = (
        UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            ["default", "render", "proxy"],
            useExtentsHint=True,
        )
        .ComputeWorldBound(prim)
        .ComputeAlignedRange()
    )
    if bounds.IsEmpty():
        return 0.0
    low, high = bounds.GetMin(), bounds.GetMax()
    # The RADIUS to the furthest corner, not the larger of the two axis
    # extents. The blade tips toward an arbitrary azimuth, and along a diagonal
    # the joint reaches sqrt(2) further than it does along x or y -- measured,
    # a bound taken per axis still let the plane climb over the fillet on the
    # diagonal views while the axis-aligned ones came out clean.
    return max(
        ((x - centre[0]) ** 2 + (y - centre[1]) ** 2) ** 0.5
        for x in (float(low[0]), float(high[0]))
        for y in (float(low[1]), float(high[1]))
    )


def _assign_joints(
    stage: object,
    execution: object,
    trims: Sequence[object],
) -> dict[str, str]:
    """Assign each lead the joint grown on it, one to one, per component.

    Nearest-joint-per-lead is not enough on its own. It lets two leads claim
    the same joint and leaves others unclaimed, and a hard distance threshold
    to paper over that then drops joints whose centre sits further out than the
    number guessed at. On a panelised THT board, that combination left some
    joints uncut -- those joints kept their full shape while the lead through
    them was trimmed, which renders as a severed lead poking out of an
    intact fillet.

    Grouping by component first and then assigning greedily by distance gives
    every joint a lead while it has one to give, without any threshold at all:
    a component's joints belong to that component's leads and to nothing else.

    Args:
        stage: Open USD stage.
        execution: Isaac execution context, for the attach order's output.
        trims: The planned trims, which name the leads.

    Returns:
        Lead path mapped to joint path. Leads without a joint are absent.

    """
    from pxr import Usd, UsdGeom

    attach = execution.outputs.get("solder_joints.attach") or {}
    candidates = tuple(attach.get("generated_prim_paths") or ())
    if not candidates:
        return {}
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=True,
    )

    def centre(path: str) -> tuple[float, float] | None:
        """Return a prim's planar centre, or None when it has no extent."""
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bounds.IsEmpty():
            return None
        low, high = bounds.GetMin(), bounds.GetMax()
        return (0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1]))

    joints_by_component: dict[str, list[str]] = {}
    for path in candidates:
        name = path.rsplit("/", 1)[-1]
        component = name.split("_solder")[0]
        joints_by_component.setdefault(component, []).append(path)

    assigned: dict[str, str] = {}
    for component, joints in joints_by_component.items():
        leads = [
            trim.lead_path
            for trim in trims
            if _component_of(trim.lead_path) == component
        ]
        pairs = []
        for lead in leads:
            lead_centre = centre(lead)
            if lead_centre is None:
                continue
            for joint in joints:
                joint_centre = centre(joint)
                if joint_centre is None:
                    continue
                pairs.append(
                    (
                        (joint_centre[0] - lead_centre[0]) ** 2
                        + (joint_centre[1] - lead_centre[1]) ** 2,
                        lead,
                        joint,
                    )
                )
        pairs.sort()
        taken_leads: set[str] = set()
        taken_joints: set[str] = set()
        for _, lead, joint in pairs:
            if lead in taken_leads or joint in taken_joints:
                continue
            assigned[lead] = joint
            taken_leads.add(lead)
            taken_joints.add(joint)
    return assigned


def _component_of(lead_path: str) -> str:
    """Return the reference designator a lead prim belongs to."""
    marker = "/Components/"
    if marker not in lead_path:
        return ""
    return lead_path.split(marker, 1)[1].split("/", 1)[0]


async def handle_pin_tip(
    order: JobOrder,
    context: object | None,
) -> object:
    """Generate separately labelable proxies for through-hole lead tips.

    This operation runs after pin variation and solder generation.  Each proxy
    copies the terminal's final outward tip-surface polygons and is offset a small
    distance toward the selected board side.  The physical pin and solder
    meshes remain unchanged.

    Args:
        order: The JobOrder being executed.
        context: Isaac execution context.

    Returns:
        Generated proxy paths and selection counts.

    Raises:
        ValidationError: If the order version is unsupported, an unfiltered
            board has no through-hole leads, or semantic registration fails.

    """
    import importlib

    from pxr import UsdGeom

    from pcb_aoi.geometry_labels import SEGMENTATION_PROXIES_SCOPE
    from pcb_aoi.pin_tip import create_pin_tip_proxy
    from pcb_aoi.refdes_selection import select_refdes
    from pcb_aoi.tht import discover_through_hole_leads

    if order.version != 1:
        raise ValidationError(
            f"{order.order_type} handler supports version 1, got {order.version}"
        )
    execution = _require_context(context)
    parameters = PinTipParameters.model_validate(order.parameters)
    stage = execution.require_stage()
    if inspect.isawaitable(stage):
        stage = await stage
    catalogs = tuple(execution.board_metadata.values())
    leads, _rejected = discover_through_hole_leads(
        stage,
        catalogs,
    )
    leads = _component_scoped_leads(execution, leads, catalogs)
    components = parameters.components
    if parameters.component_filter is not None:
        if not catalogs:
            raise ValidationError(
                "pin_tip.component_filter requires component metadata",
            )
        from pcb_aoi.board_metadata import select_components

        selected_components = select_components(
            catalogs[0],
            parameters.component_filter,
            parent_paths=execution.component_parent_paths(),
            where="operations.pin_tip.component_filter",
        )
        components = tuple(component.refdes for component in selected_components)
    if not leads and components is None:
        raise ValidationError(
            "pin_tip found no through-hole leads with paired plated holes"
        )
    selection = select_refdes(
        leads,
        components,
        key=lambda lead: lead.refdes,
    )
    selected = selection.selected
    diagnostics = []
    if selection.unmatched:
        message = (
            "pin_tip components matched no eligible through-hole leads: "
            f"{list(selection.unmatched)}"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        diagnostics.append(message)

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    board_root = str(
        execution.scene.get("board_root_path") or execution.scene.get("root_path")
    )
    segmentation_proxies_path = f"{board_root.rstrip('/')}/{SEGMENTATION_PROXIES_SCOPE}"
    output_root_path = f"{segmentation_proxies_path}/PinTips"
    generated = []
    failed_prim_paths = []
    for lead in selected:
        path = create_pin_tip_proxy(
            stage,
            lead.path,
            output_root_path=output_root_path,
            class_name=parameters.class_name,
            side=parameters.side,
            offset_mm=parameters.offset_mm,
            meters_per_unit=meters_per_unit,
        )
        if path is not None:
            generated.append(path)
        else:
            failed_prim_paths.append(lead.path)
    if failed_prim_paths:
        message = (
            "pin_tip could not identify an outward tip surface on "
            f"{len(failed_prim_paths)} selected lead(s): {failed_prim_paths}"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        diagnostics.append(message)
    registered = 0
    if generated:
        registered = importlib.import_module("common").refresh_authored_semantics(
            stage,
            generated,
        )
    if registered != len(generated):
        raise ValidationError(
            "pin_tip registered semantic labels on "
            f"{registered} of {len(generated)} generated proxies"
        )
    if generated:
        mark_instanced_stage_mutation(execution, "operation.pin_tip")
    output = {
        "class_name": parameters.class_name,
        "side": parameters.side,
        "offset_mm": parameters.offset_mm,
        "requested_components": list(selection.requested),
        "matched_components": list(selection.matched),
        "unmatched_components": list(selection.unmatched),
        "selected": len(selected),
        "generated": len(generated),
        "failed_prim_paths": failed_prim_paths,
        "semantic_prims": registered,
        "prim_paths": generated,
        "warnings": diagnostics,
    }
    execution.outputs[order.order_id] = output
    return output


def isaac_handlers() -> Mapping[str, object]:
    """Return the Isaac handlers this domain owns.

    Called when a registry is built rather than at import time, so the domain
    can keep its heavier dependencies lazy.

    Returns:
        Order type mapped to handler.

    """
    return {
        "scene.components.select": _runtime_handler(
            ComponentSelectionParameters,
            "select_components",
        ),
        "appearance.components.configure": _runtime_handler(
            ComponentAppearanceParameters,
            "configure_component_appearance",
        ),
        "annotations.operations.configure": _runtime_handler(
            OperationLabelParameters,
            "configure_operation_labels",
        ),
        "annotations.components.apply": _runtime_handler(
            ComponentAnnotationParameters,
            "apply_component_annotations",
        ),
        "annotations.geometry.apply": _runtime_handler(
            GeometryAnnotationParameters,
            "apply_geometry_annotations",
        ),
        "operation.shift.apply": _runtime_handler(ShiftParameters, "apply_shift"),
        "operation.tombstone.apply": _runtime_handler(
            TombstoneParameters,
            "apply_tombstone",
        ),
        "operation.sideflip.apply": _runtime_handler(
            SideflipParameters,
            "apply_sideflip",
        ),
        "operation.reverse_polarity.apply": _runtime_handler(
            ReversePolarityParameters,
            "apply_reverse_polarity",
        ),
        "operation.missing.apply": _runtime_handler(
            MissingParameters,
            "apply_missing",
        ),
        "solder_defects.apply": _runtime_handler(
            SolderDefectParameters,
            "apply_solder_defects",
        ),
        "solder_joints.discover": _runtime_handler(
            SolderJointParameters,
            "discover_solder_joints",
        ),
        "solder_joints.collect": _runtime_handler(
            SolderJointParameters,
            "collect_solder_joint_inputs",
        ),
        "solder_joints.generate": _runtime_handler(
            SolderJointParameters,
            "generate_solder_joints",
        ),
        "solder_joints.attach": _runtime_handler(
            SolderJointParameters,
            "attach_solder_joints",
        ),
        "hole_plane.generate": _runtime_handler(
            HolePlaneParameters,
            "generate_hole_plane",
        ),
        "postprocess.format_roi_output": _runtime_handler(
            PostprocessFormatRoiOutputParameters,
            "format_roi_output",
        ),
        "postprocess.generate_joint_rois": _runtime_handler(
            PostprocessGenerateJointRoisParameters,
            "generate_joint_rois",
        ),
        PIN_VARIATION_ORDER_TYPE: handle_pin_variation,
        PIN_CUT_ORDER_TYPE: handle_pin_cut,
        PIN_TIP_ORDER_TYPE: handle_pin_tip,
    }


def geometry_roles() -> Mapping[str, GeometryRole]:
    """Return the traversal-resolved geometry targets this domain owns.

    ``barrels`` is board geometry -- the copper inner wall of a drilled hole --
    whose component is named by a relationship. ``body`` and ``marking`` are
    package geometry inside a component prototype, so their component is the
    enclosing prim and an instanced cohort reads them through the projection.

    Returns:
        Target name mapped to how the executor resolves it.

    """
    from pcb_aoi.geometry_labels import (
        PACKAGE_TRAVERSAL_TARGETS,
        TRAVERSAL_ROLES,
    )

    return {
        target: GeometryRole(
            role=role,
            ownership=(
                OWNERSHIP_ANCESTOR
                if target in PACKAGE_TRAVERSAL_TARGETS
                else OWNERSHIP_RELATIONSHIP
            ),
            projected=target in PACKAGE_TRAVERSAL_TARGETS,
        )
        for target, role in TRAVERSAL_ROLES.items()
    }


def register() -> None:
    """Register this domain's Isaac handlers and geometry roles with CDF."""
    register_isaac_adapter("pcb-aoi", isaac_handlers)
    register_geometry_roles("pcb-aoi", geometry_roles)
