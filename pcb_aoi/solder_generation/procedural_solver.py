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

"""Generate stage-free procedural solder meshes from PCB contact geometry."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import ClassVar

from ..board_metadata.metadata_catalog import PadRecord
from ..board_metadata.usd_schema_metadata_keys import USD_SCHEMA_METADATA_KEYS
from .data_types import (
    ComponentIdentity,
    SolderCoordinateFrame,
    SolderCoverageStatus,
    SolderGenerationRequest,
    SolderGeometryArtifact,
    SolderMesh,
    SolderReuseScope,
    SolderSolverRequest,
    SolderSolverResult,
    SolderSolverType,
    SolderWorkItem,
    SolderWorkItemCoverage,
)
from .metadata_discovery import solder_process_class
from .procedural_meniscus_geometry import (
    LeadContactMeasurement,
    MeniscusParameters,
    build_pad_meniscus,
    is_area_array,
    measure_lead_contact,
    resolve_pad_orientation,
)
from .procedural_tht_geometry import (
    ThroughHoleParameters,
    ThroughHoleSite,
    build_through_hole_volcano,
)
from .procedural_tht_stage_analysis import prepare_through_hole_site
from .solver_base import SolderSolverBase, _component_allowlist

PROCEDURAL_SOLVER_ID = "procedural"
_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


def _solder_side_pads(
    pads: Sequence[PadRecord],
    insertion_side: str,
) -> tuple[PadRecord, ...]:
    """Return only THT pads opposite the component insertion side."""
    solder_side = "bottom" if insertion_side.lower().startswith("t") else "top"
    return tuple(pad for pad in pads if (pad.side or "").lower() == solder_side)


def _missing_solder_side_terminals(
    available_pads: Sequence[PadRecord],
    solder_side_pads: Sequence[PadRecord],
) -> tuple[str, ...]:
    """Return terminals represented only by non-solder-side pads."""
    available = {pad.terminal_number for pad in available_pads if pad.terminal_number}
    selected = {pad.terminal_number for pad in solder_side_pads if pad.terminal_number}
    return tuple(
        sorted(
            available - selected,
            key=lambda number: (
                0 if number.isdigit() else 1,
                int(number) if number.isdigit() else number,
            ),
        ),
    )


def _combined_bounds(bounds: Sequence[Sequence[float]]) -> tuple[float, ...] | None:
    """Return the union of six-value aligned bounds.

    Args:
        bounds: Collection of aligned world-space bounds.

    Returns:
        Union bounds, or None when no valid bounds are supplied.

    """
    valid = tuple(item for item in bounds if len(item) == 6)
    if not valid:
        return None
    return (
        min(float(item[0]) for item in valid),
        min(float(item[1]) for item in valid),
        min(float(item[2]) for item in valid),
        max(float(item[3]) for item in valid),
        max(float(item[4]) for item in valid),
        max(float(item[5]) for item in valid),
    )


def _component_yaw(
    stage: object,
    prim_path: str,
    xform_cache: object,
) -> float:
    """Return a component's rotation about the board normal, in degrees.

    The meniscus maths reasons about axis-aligned pads, so a component at a
    non-quadrant rotation must be measured in its own yaw frame. This recovers
    that frame from the instance transform.

    Only a **planar** rotation is reported. If the prim is tilted out of the
    board plane the yaw frame is not the whole story, and returning a confident
    number for it would replace a visible error with a hidden one — so the
    tilted case reports ``0.0`` and keeps the previous world-aligned behaviour.

    Args:
        stage: Live USD stage used only for input preparation.
        prim_path: Component instance prim path.
        xform_cache: Shared USD xform cache.

    Returns:
        Yaw in degrees, or 0.0 when the prim is missing, tilted or unrotated.

    """
    import math

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return 0.0
    matrix = xform_cache.GetLocalToWorldTransform(prim)
    x_axis = (matrix[0][0], matrix[0][1], matrix[0][2])
    z_axis = (matrix[2][0], matrix[2][1], matrix[2][2])
    z_length = math.sqrt(sum(value * value for value in z_axis))
    if z_length <= 1e-12:
        return 0.0
    # Out-of-plane tilt: the local Z has drifted off the board normal.
    tilt = math.hypot(z_axis[0], z_axis[1]) / z_length
    if tilt > 1e-4:
        return 0.0
    if abs(x_axis[0]) <= 1e-12 and abs(x_axis[1]) <= 1e-12:
        return 0.0
    return math.degrees(math.atan2(x_axis[1], x_axis[0]))


def _aligned_bounds(
    stage: object,
    prim_path: str,
    cache: object | None = None,
    yaw_degrees: float = 0.0,
) -> tuple[float, ...] | None:
    """Return an aligned bounding box for one prim, optionally in a yaw frame.

    Args:
        stage: Live USD stage used only for input preparation.
        prim_path: Prim path to inspect.
        cache: Optional shared USD bounding-box cache.
        yaw_degrees: Component yaw. When non-zero the box is measured in that
            frame instead of world space, which keeps it **tight** on a rotated
            component — a world-aligned box of a rotated part is inflated by
            roughly ``L*|sin| + W*|cos|`` and mis-sizes every meniscus built
            from it.

    Returns:
        Six bounds in the requested frame, or None when the prim cannot be
        measured.

    """
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    if cache is None:
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            ["default", "render", "proxy"],
            useExtentsHint=True,
        )
    box = cache.ComputeWorldBound(prim)
    if yaw_degrees:
        # Compose the inverse yaw onto the box's own transform rather than
        # re-boxing world corners: the untransformed range is preserved, so the
        # result is exact instead of an inflated box of an inflated box.
        rotation = Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), -yaw_degrees),
        )
        box = Gf.BBox3d(box.GetRange(), box.GetMatrix() * rotation)
    bounds = box.ComputeAlignedRange()
    if bounds.IsEmpty():
        return None
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(minimum[2]),
        float(maximum[0]),
        float(maximum[1]),
        float(maximum[2]),
    )


def _world_mesh_points(
    stage: object,
    prim_paths: Sequence[str],
    xform_cache: object,
    point_cache: dict[str, tuple[tuple[float, float, float], ...]],
    yaw_degrees: float = 0.0,
) -> tuple[tuple[float, float, float], ...]:
    """Return mesh points below the supplied prim paths, in the yaw frame.

    Args:
        stage: Live USD stage used only for input preparation.
        prim_paths: Prim paths whose meshes contribute points.
        xform_cache: Shared USD xform cache.
        point_cache: Per-path memo of world-space points.
        yaw_degrees: Component yaw. Points are rotated into that frame so they
            can be compared against bounds measured the same way.

    Returns:
        Points in the requested frame.

    """
    import math

    import numpy as np
    from pxr import Usd, UsdGeom

    chunks = []
    for path in dict.fromkeys(prim_paths):
        if path not in point_cache:
            arrays = []
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                for mesh_prim in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                    if not mesh_prim.IsA(UsdGeom.Mesh):
                        continue
                    points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
                    if not points:
                        continue
                    matrix = np.asarray(
                        xform_cache.GetLocalToWorldTransform(mesh_prim),
                        dtype=float,
                    )
                    arrays.append(
                        np.asarray(points, dtype=float) @ matrix[:3, :3] + matrix[3, :3]
                    )
            if arrays:
                combined = np.vstack(arrays)
                point_cache[path] = tuple(
                    (float(point[0]), float(point[1]), float(point[2]))
                    for point in combined
                )
            else:
                point_cache[path] = ()
        chunks.extend(point_cache[path])
    if not yaw_degrees or not chunks:
        return tuple(chunks)
    # The cache stays world-space and shared; only this view is rotated.
    angle = math.radians(-yaw_degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return tuple(
        (x * cos_a - y * sin_a, x * sin_a + y * cos_a, z) for x, y, z in chunks
    )


def _mesh_name(component_path: str, site_index: int) -> str:
    """Build a stable USD-safe procedural mesh name.

    Args:
        component_path: Component-instance prim path.
        site_index: Zero-based solder-site index.

    Returns:
        Stable mesh name.

    """
    component_name = component_path.rstrip("/").rsplit("/", 1)[-1]
    safe_name = _NAME_PATTERN.sub("_", component_name)
    return f"{safe_name}_solder_{site_index:03d}"


class ProceduralSolderSolver(SolderSolverBase):
    """Generate pad-aware menisci for paste or contact footprints."""

    solver_id = PROCEDURAL_SOLVER_ID
    solver_type = SolderSolverType.PROCEDURAL
    available = True
    unavailable_reason = None
    requires_source_paste = False

    _MENISCUS_OPTIONS: ClassVar[set[str]] = {
        "long_edge",
        "x_scale",
        "platform_start",
        "platform_end",
        "smoothness_y",
        "bump",
        "delta_x_decay",
        "pad_decay",
        "noise_scale",
        "noise_octaves",
        "noise_amplitude",
        "resolution",
        "solid",
        "solid_depth_fraction",
        "wetting_overlap_ratio",
        "z_offset_mm",
        "tht_angular_segments",
        "tht_crease_angle_deg",
        "tht_radial_segments",
        "tht_pad_edge_inset_mm",
        "tht_profile_exponent",
        "tht_solder_bath",
        "lead_local_sizing",
        "solder_material_prim_path",
    }

    @staticmethod
    def _options(options: Mapping[str, object]) -> tuple[float, float, float]:
        """Validate procedural shape configuration.

        Args:
            options: Solver-specific configuration.

        Returns:
            Height, inset, and center-shift ratios.

        Raises:
            ValueError: If an option is unknown or outside its valid range.

        """
        supported = {
            "components",
            "height_ratio",
            "inset_ratio",
            "shift_ratio",
            *ProceduralSolderSolver._MENISCUS_OPTIONS,
        }
        unknown = sorted(set(options) - supported)
        if unknown:
            raise ValueError(f"unknown procedural solver option(s): {unknown}")
        height_ratio = float(options.get("height_ratio", 0.9))
        inset_ratio = float(options.get("inset_ratio", 0.2))
        shift_ratio = float(options.get("shift_ratio", 0.12))
        if not 0.0 < height_ratio <= 2.0:
            raise ValueError("height_ratio must be greater than 0 and at most 2")
        if not 0.0 <= inset_ratio < 0.5:
            raise ValueError("inset_ratio must be at least 0 and less than 0.5")
        if not 0.0 <= shift_ratio <= 1.0:
            raise ValueError("shift_ratio must be between 0 and 1")
        material_path = options.get("solder_material_prim_path")
        if material_path is not None and (
            not isinstance(material_path, str)
            or not material_path.startswith("/")
            or material_path == "/"
            or "//" in material_path
        ):
            raise ValueError(
                "solder_material_prim_path must be an absolute non-root prim path",
            )
        return height_ratio, inset_ratio, shift_ratio

    @staticmethod
    def _component_is_selected(
        component: ComponentIdentity,
        allowlist: frozenset[str],
    ) -> bool:
        """Return whether a component matches an exact refdes or prim path."""
        return (
            not allowlist
            or component.refdes in allowlist
            or component.instance_path in allowlist
        )

    @classmethod
    def _meniscus_parameters(
        cls,
        options: Mapping[str, object],
        scene_units_per_mm: float,
    ) -> MeniscusParameters:
        """Translate public solver options into stage-free meniscus settings.

        Args:
            options: Solver-specific configuration supplied by the public API.
            scene_units_per_mm: Conversion factor from millimeters to scene units.

        Returns:
            Validated stage-free meniscus settings.

        Raises:
            ValueError: If an option is unknown or invalid.

        """
        height_ratio, inset_ratio, shift_ratio = cls._options(options)
        solid = options.get("solid", True)
        if not isinstance(solid, bool):
            raise ValueError("solid must be a boolean")  # noqa: TRY004
        lead_local_sizing = options.get("lead_local_sizing", True)
        if not isinstance(lead_local_sizing, bool):
            raise ValueError("lead_local_sizing must be a boolean")  # noqa: TRY004
        return MeniscusParameters(
            long_edge=float(options.get("long_edge", 9.0)),
            x_scale=float(options.get("x_scale", 0.2)),
            platform_start=float(options.get("platform_start", inset_ratio)),
            platform_end=float(options.get("platform_end", 1.0 - inset_ratio)),
            smoothness_y=float(options.get("smoothness_y", 6.44)),
            bump=float(options.get("bump", 0.38)),
            decay=float(options.get("delta_x_decay", 10.8)),
            pad_decay=float(options.get("pad_decay", 5.0)),
            noise_scale=float(options.get("noise_scale", 4.0)),
            noise_octaves=int(options.get("noise_octaves", 4)),
            noise_amplitude=float(options.get("noise_amplitude", 0.01)),
            resolution=int(options.get("resolution", 20)),
            solid=solid,
            solid_depth_fraction=float(
                options.get("solid_depth_fraction", 0.15),
            ),
            termination_height_ratio=height_ratio,
            wetting_overlap_ratio=float(
                options.get("wetting_overlap_ratio", shift_ratio),
            ),
            z_offset=(float(options.get("z_offset_mm", 0.0)) * scene_units_per_mm),
            lead_local_sizing=lead_local_sizing,
            scene_units_per_mm=scene_units_per_mm,
        )

    @classmethod
    def _through_hole_parameters(
        cls,
        options: Mapping[str, object],
        scene_units_per_mm: float,
    ) -> ThroughHoleParameters:
        """Translate public solver options into THT volcano settings.

        Args:
            options: Solver-specific configuration supplied by the public API.
            scene_units_per_mm: Conversion factor from millimeters to scene units.

        Returns:
            Validated stage-free through-hole geometry settings.

        Raises:
            ValueError: If an option is unknown or invalid.

        """
        height_ratio, _inset_ratio, _shift_ratio = cls._options(options)
        solid = options.get("solid", True)
        if not isinstance(solid, bool):
            raise ValueError("solid must be a boolean")  # noqa: TRY004
        solder_bath = options.get("tht_solder_bath", False)
        if not isinstance(solder_bath, bool):
            raise ValueError("tht_solder_bath must be a boolean")  # noqa: TRY004
        if "tht_angular_segments" in options:
            resolution = int(options["tht_angular_segments"])
        elif "resolution" in options:
            resolution = int(options["resolution"])
        else:
            resolution = 64
        radial_segments = int(
            options.get("tht_radial_segments", max(2, resolution // 4)),
        )
        crease_angle_deg = options.get("tht_crease_angle_deg")
        return ThroughHoleParameters(
            resolution=max(8, resolution),
            radial_segments=radial_segments,
            height_ratio=height_ratio,
            solid=solid,
            solid_depth_fraction=float(
                options.get("solid_depth_fraction", 0.15),
            ),
            z_offset=float(options.get("z_offset_mm", 0.0)) * scene_units_per_mm,
            pad_edge_inset=(
                float(options.get("tht_pad_edge_inset_mm", 0.0)) * scene_units_per_mm
            ),
            solder_bath=solder_bath,
            profile_exponent=float(options.get("tht_profile_exponent", 1.0)),
            crease_angle_deg=(
                None if crease_angle_deg is None else float(crease_angle_deg)
            ),
        )

    def prepare(
        self,
        request: SolderGenerationRequest,
        work_items: Sequence[SolderWorkItem],
    ) -> SolderSolverRequest:
        """Extract transport-safe world-space footprint bounds.

        Paste geometry is preferred when available. Contact geometry provides a
        fallback for through-hole parts and procedural configurations that do
        not require paste.

        Args:
            request: Extension-level generation request.
            work_items: Solver-neutral work selected by the extension.

        Returns:
            Stage-free request containing component and solder-site bounds.

        """
        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            ["default", "render", "proxy"],
            useExtentsHint=True,
        )
        catalog = request.board_metadata
        component_by_path = {
            component.prim_path: component for component in catalog.components
        }
        hole_by_pad_path = {
            hole.pad_prim_path: hole for hole in catalog.holes if hole.pad_prim_path
        }
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(request.stage) or 0.01)
        scene_units_per_mm = 0.001 / meters_per_unit
        parameters = self._meniscus_parameters(
            request.solver_options,
            scene_units_per_mm,
        )
        through_hole_parameters = self._through_hole_parameters(
            request.solver_options,
            scene_units_per_mm,
        )
        # Select during preparation to avoid unnecessary stage geometry reads.
        # ``generate()`` reapplies the filter because selection also determines
        # emitted artifacts and coverage when consuming a stage-free request.
        component_allowlist = _component_allowlist(request.solver_options)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        point_cache: dict[str, tuple[tuple[float, float, float], ...]] = {}
        sites_by_component: dict[str, object] = {}
        preparation_warnings = []
        geometry_fallback_associations = 0
        missing_geometry_metadata: set[str] = set()
        for work_item in work_items:
            for component in work_item.components:
                if not self._component_is_selected(component, component_allowlist):
                    continue
                component_path = component.instance_path
                component_record = component_by_path[component_path]
                process_class = (
                    solder_process_class(component_record) or component.process_class
                )
                if process_class == "through_hole":
                    associations = []
                    solder_side = (
                        "bottom" if component.side.lower().startswith("t") else "top"
                    )
                    available_component_pads = catalog.pads_for_component(
                        component_record,
                    )
                    component_pads = _solder_side_pads(
                        available_component_pads,
                        component.side,
                    )
                    missing_solder_side_terminals = _missing_solder_side_terminals(
                        available_component_pads,
                        component_pads,
                    )
                    for pad in component_pads:
                        if pad.pad_type in {"np_thru_hole", "np_through_hole"}:
                            continue
                        hole = hole_by_pad_path.get(pad.prim_path)
                        associations.append(
                            {
                                "solder_contact_prim": pad.contact_prim_path,
                                "solder_pad_prim": pad.prim_path,
                                "solder_hole_prim": hole.prim_path if hole else None,
                                "terminal_number": pad.terminal_number,
                                "pin_center": pad.pin_center,
                                "pad_contour": pad.contour,
                            },
                        )
                    sites = []
                    for association in associations:
                        pad = catalog.by_prim_path(
                            str(association["solder_pad_prim"]),
                        )
                        missing_metadata = catalog.tht_geometry_metadata_gaps(pad)
                        if missing_metadata:
                            geometry_fallback_associations += 1
                            missing_geometry_metadata.update(
                                "associated board_hole prim"
                                if field == "hole.association"
                                else USD_SCHEMA_METADATA_KEYS[field][0]
                                for field in missing_metadata
                            )
                        site = prepare_through_hole_site(
                            request.stage,
                            association,
                            component.side,
                            scene_units_per_mm,
                            cache,
                            height_ratio=through_hole_parameters.height_ratio,
                            z_offset=through_hole_parameters.z_offset,
                            pad_edge_inset=through_hole_parameters.pad_edge_inset,
                            solder_bath=through_hole_parameters.solder_bath,
                        )
                        if site is None:
                            terminal_number = association.get("terminal_number")
                            terminal_suffix = (
                                f".{terminal_number}" if terminal_number else ""
                            )
                            preparation_warnings.append(
                                f"{component.refdes}{terminal_suffix}: invalid THT "
                                "pin-to-pad association",
                            )
                            continue
                        sites.append(site)
                    if missing_solder_side_terminals:
                        joined_terminals = ", ".join(missing_solder_side_terminals)
                        preparation_warnings.append(
                            f"{component.refdes}: missing {solder_side}-side THT "
                            f"pad associations for terminal(s) {joined_terminals}; "
                            "refusing same-side fallback",
                        )
                    elif not component_pads and available_component_pads:
                        available_sides = sorted(
                            {
                                (pad.side or "unknown").lower()
                                for pad in available_component_pads
                            },
                        )
                        preparation_warnings.append(
                            f"{component.refdes}: no {solder_side}-side THT pad "
                            "associations; refusing same-side fallback from "
                            f"{', '.join(available_sides)} pad associations",
                        )
                    elif not associations:
                        preparation_warnings.append(
                            f"{component.refdes}: through-hole solder generation "
                            "requires solder_pin_pad_associations",
                        )
                    sites_by_component[component_path] = {
                        "process_class": process_class,
                        "side": component.side,
                        "site_kind": "pin-pad",
                        "sites": tuple(sites),
                    }
                    continue

                source_paths = tuple(
                    work_item.source_paste_prims.get(component_path, ()),
                )
                site_kind = "paste"
                if not source_paths:
                    source_paths = tuple(
                        work_item.solder_contact_prims.get(component_path, ()),
                    )
                    site_kind = "contact"
                # Everything below is measured in the component's yaw frame, so
                # a rotated part presents axis-aligned pads to the placement
                # maths. `generate` rotates the finished meniscus back.
                yaw_degrees = _component_yaw(
                    request.stage,
                    component_path,
                    xform_cache,
                )
                sites = tuple(
                    {
                        "bounds": bounds,
                        "material_source_prim_path": path,
                    }
                    for path in source_paths
                    if (
                        bounds := _aligned_bounds(
                            request.stage,
                            path,
                            cache,
                            yaw_degrees,
                        )
                    )
                    is not None
                )
                site_bounds = tuple(site["bounds"] for site in sites)
                component_bounds = _aligned_bounds(
                    request.stage,
                    component_path,
                    cache,
                    yaw_degrees,
                )
                body_paths = component_record.body_prim_paths
                contact_paths = tuple(
                    work_item.solder_contact_prims.get(component_path, ()),
                )
                body_bounds = _combined_bounds(
                    tuple(
                        bounds
                        for path in (*body_paths, *contact_paths)
                        if (
                            bounds := _aligned_bounds(
                                request.stage,
                                path,
                                cache,
                                yaw_degrees,
                            )
                        )
                        is not None
                    ),
                )
                if body_bounds is None:
                    body_bounds = component_bounds
                lead_contacts = ()
                area_array = False
                if (
                    parameters.lead_local_sizing
                    and component_bounds is not None
                    and body_bounds is not None
                ):
                    point_paths = tuple(dict.fromkeys((*contact_paths, *body_paths)))
                    lead_points = _world_mesh_points(
                        request.stage,
                        point_paths,
                        xform_cache,
                        point_cache,
                        yaw_degrees,
                    )
                    area_array = is_area_array(site_bounds)
                    measurements = []
                    for bounds in site_bounds:
                        _, outward_axis, outward_sign = resolve_pad_orientation(
                            bounds,
                            body_bounds,
                            component_bounds,
                        )
                        measurement = measure_lead_contact(
                            bounds,
                            lead_points,
                            component.side,
                            outward_axis,
                            outward_sign,
                            0.5 * scene_units_per_mm,
                            2.5 * scene_units_per_mm,
                        )
                        measurements.append(
                            measurement.as_payload()
                            if measurement is not None
                            else None
                        )
                    lead_contacts = tuple(measurements)
                sites_by_component[component_path] = {
                    "component_bounds": component_bounds,
                    "body_bounds": body_bounds,
                    "process_class": process_class,
                    "side": component.side,
                    "site_kind": site_kind,
                    "sites": sites,
                    "lead_contacts": lead_contacts,
                    "area_array": area_array,
                    "yaw_degrees": yaw_degrees,
                }

        if missing_geometry_metadata:
            missing_fields = ", ".join(sorted(missing_geometry_metadata))
            preparation_warnings.append(
                "THT dimensional metadata is incomplete for "
                f"{geometry_fallback_associations} pin-to-pad association(s); "
                f"missing {missing_fields}. The procedural solver will perform "
                "heavier geometric analysis of the associated USD meshes to "
                "infer solder-joint dimensions.",
            )

        return SolderSolverRequest(
            request_id=request.request_id,
            solver_id=self.solver_id,
            source_usd_path=request.source_usd_path,
            root_path=request.root_path,
            work_items=tuple(work_items),
            solver_options=dict(request.solver_options),
            payload={
                "sites_by_component": sites_by_component,
                "scene_units_per_mm": scene_units_per_mm,
                "preparation_warnings": tuple(preparation_warnings),
            },
        )

    def generate(self, request: SolderSolverRequest) -> SolderSolverResult:
        """Generate procedural meshes from prepared footprint bounds.

        Args:
            request: Stage-free procedural request.

        Returns:
            World-space meshes and per-work-item coverage.

        Raises:
            TypeError: If prepared procedural payload data is malformed.

        """
        raw_sites = request.payload.get("sites_by_component")
        if not isinstance(raw_sites, Mapping):
            raise TypeError("procedural request is missing sites_by_component")
        scene_units_per_mm = float(request.payload.get("scene_units_per_mm", 1.0))
        parameters = self._meniscus_parameters(
            request.solver_options,
            scene_units_per_mm,
        )
        through_hole_parameters = self._through_hole_parameters(
            request.solver_options,
            scene_units_per_mm,
        )

        artifacts = []
        coverage = []
        warnings = [
            str(message)
            for message in (request.payload.get("preparation_warnings") or ())
            if message
        ]
        material_override = (
            str(request.solver_options.get("solder_material_prim_path") or "") or None
        )
        # Reapply selection here because this stage-free entry point owns both
        # artifact generation and the coverage status for excluded components.
        component_allowlist = _component_allowlist(request.solver_options)
        for work_item in request.work_items:
            artifact_ids = []
            selected_components = tuple(
                component
                for component in work_item.components
                if self._component_is_selected(component, component_allowlist)
            )
            for component in selected_components:
                prepared = raw_sites.get(component.instance_path)
                if not isinstance(prepared, Mapping):
                    continue
                process_class = prepared.get("process_class")
                sites = prepared.get("sites")
                if not isinstance(sites, (list, tuple)):
                    continue
                if process_class == "through_hole":
                    meshes = []
                    for index, raw_site in enumerate(sites):
                        if not isinstance(raw_site, Mapping):
                            continue
                        try:
                            site = ThroughHoleSite(
                                center_xy=tuple(
                                    float(value)
                                    for value in raw_site.get("center_xy", ())
                                ),
                                pin_radius=float(raw_site.get("pin_radius", 0.0)),
                                pad_radius=float(raw_site.get("pad_radius", 0.0)),
                                base_z=float(raw_site.get("base_z", 0.0)),
                                grow_toward_negative_z=bool(
                                    raw_site.get("grow_toward_negative_z"),
                                ),
                                pad_shape_xy=tuple(
                                    tuple(float(value) for value in point)
                                    for point in raw_site.get("pad_shape_xy", ())
                                ),
                                throat_center_xy=(
                                    tuple(
                                        float(value)
                                        for value in raw_site.get(
                                            "throat_center_xy",
                                            (),
                                        )
                                    )
                                    or None
                                ),
                                throat_shape_xy=tuple(
                                    tuple(float(value) for value in point)
                                    for point in raw_site.get(
                                        "throat_shape_xy",
                                        (),
                                    )
                                ),
                                pin_points=tuple(
                                    tuple(float(value) for value in point)
                                    for point in raw_site.get("pin_points", ())
                                ),
                                pin_face_vertex_counts=tuple(
                                    int(value)
                                    for value in raw_site.get(
                                        "pin_face_vertex_counts",
                                        (),
                                    )
                                ),
                                pin_face_vertex_indices=tuple(
                                    int(value)
                                    for value in raw_site.get(
                                        "pin_face_vertex_indices",
                                        (),
                                    )
                                ),
                            )
                            generated = build_through_hole_volcano(
                                site,
                                through_hole_parameters,
                            )
                        except (TypeError, ValueError) as exc:
                            # One unusable lead must not retire the board.
                            # The site constructor was already guarded here;
                            # the mesh builder beside it was not, and it
                            # rejects a throat that reaches the pad with the
                            # SAME message the constructor uses. An escape
                            # reaches the extension's solver isolation, which
                            # falls the WHOLE request back to pass-through:
                            # on a panelised THT board, a single overwide
                            # lead would leave every through-hole
                            # component with no solder while the job
                            # still reported success.
                            warnings.append(
                                f"{component.instance_path} site {index} skipped: {exc}",
                            )
                            continue
                        meshes.append(
                            SolderMesh(
                                name=_mesh_name(component.instance_path, index),
                                points=generated.points,
                                face_vertex_counts=generated.face_vertex_counts,
                                face_vertex_indices=generated.face_vertex_indices,
                                normals=generated.normals,
                                transform=generated.transform,
                                material_source_prim_path=(
                                    material_override
                                    or str(raw_site.get("solder_contact_prim") or "")
                                    or None
                                ),
                            ),
                        )
                    meshes = tuple(meshes)
                    if meshes:
                        artifact_id = f"{component.instance_path}#procedural"
                        artifact_ids.append(artifact_id)
                        artifacts.append(
                            SolderGeometryArtifact(
                                artifact_id=artifact_id,
                                work_item_ids=(work_item.work_item_id,),
                                component_paths=(component.instance_path,),
                                coordinate_frame=SolderCoordinateFrame.WORLD,
                                reuse_scope=SolderReuseScope.INSTANCE,
                                meshes=meshes,
                                metadata={
                                    "process_class": process_class,
                                    "site_kind": prepared.get("site_kind"),
                                },
                            ),
                        )
                    continue

                component_bounds = prepared.get("component_bounds")
                body_bounds = prepared.get("body_bounds")
                if (
                    not isinstance(component_bounds, (list, tuple))
                    or len(component_bounds) != 6
                    or not isinstance(body_bounds, (list, tuple))
                    or len(body_bounds) != 6
                    or not isinstance(sites, (list, tuple))
                ):
                    continue
                if parameters.lead_local_sizing and prepared.get("area_array"):
                    continue
                lead_contacts = prepared.get("lead_contacts")
                if not isinstance(lead_contacts, (list, tuple)):
                    lead_contacts = ()
                # `prepare` measured this component in its own yaw frame; the
                # meniscus is built there and rotated back on the way out. The
                # shared settings are otherwise untouched, and yaw 0 leaves the
                # object identical to the one that came in.
                yaw_degrees = float(prepared.get("yaw_degrees") or 0.0)
                site_parameters = (
                    replace(parameters, yaw_degrees=yaw_degrees)
                    if yaw_degrees
                    else parameters
                )
                meshes = []
                for index, raw_site in enumerate(sites):
                    material_source_prim_path = None
                    bounds = raw_site
                    if isinstance(raw_site, Mapping):
                        bounds = raw_site.get("bounds")
                        material_source_prim_path = (
                            material_override
                            or str(raw_site.get("material_source_prim_path") or "")
                            or None
                        )
                    if not isinstance(bounds, (list, tuple)) or len(bounds) != 6:
                        continue
                    generated = build_pad_meniscus(
                        tuple(float(value) for value in bounds),
                        tuple(float(value) for value in body_bounds),
                        tuple(float(value) for value in component_bounds),
                        str(prepared.get("side") or component.side),
                        len(sites),
                        site_parameters,
                        (
                            LeadContactMeasurement.from_payload(
                                tuple(float(value) for value in lead_contacts[index]),
                            )
                            if (
                                index < len(lead_contacts)
                                and isinstance(lead_contacts[index], (list, tuple))
                                and len(lead_contacts[index]) == 5
                            )
                            else None
                        ),
                    )
                    if generated is None:
                        continue
                    meshes.append(
                        SolderMesh(
                            name=_mesh_name(component.instance_path, index),
                            points=generated.points,
                            face_vertex_counts=generated.face_vertex_counts,
                            face_vertex_indices=generated.face_vertex_indices,
                            transform=generated.transform,
                            material_source_prim_path=material_source_prim_path,
                        ),
                    )
                meshes = tuple(meshes)
                if not meshes:
                    continue
                artifact_id = f"{component.instance_path}#procedural"
                artifact_ids.append(artifact_id)
                artifacts.append(
                    SolderGeometryArtifact(
                        artifact_id=artifact_id,
                        work_item_ids=(work_item.work_item_id,),
                        component_paths=(component.instance_path,),
                        coordinate_frame=SolderCoordinateFrame.WORLD,
                        reuse_scope=SolderReuseScope.INSTANCE,
                        meshes=meshes,
                        metadata={
                            "site_kind": prepared.get("site_kind"),
                        },
                    ),
                )

            coverage.append(
                SolderWorkItemCoverage(
                    work_item_id=work_item.work_item_id,
                    component_paths=work_item.component_paths,
                    status=(
                        SolderCoverageStatus.GENERATED
                        if artifact_ids
                        else (
                            SolderCoverageStatus.NO_SOURCE_GEOMETRY
                            if any(
                                isinstance(
                                    raw_sites.get(component.instance_path), Mapping
                                )
                                and raw_sites.get(component.instance_path, {}).get(
                                    "process_class",
                                )
                                == "through_hole"
                                for component in selected_components
                            )
                            else SolderCoverageStatus.INELIGIBLE
                        )
                    ),
                    artifact_ids=tuple(artifact_ids),
                    message=(
                        None
                        if artifact_ids
                        else (
                            "component not in the configured component allowlist"
                            if not selected_components
                            else (
                                "procedural solver found no usable THT pin-to-pad "
                                "associations"
                                if any(
                                    isinstance(
                                        raw_sites.get(component.instance_path), Mapping
                                    )
                                    and raw_sites.get(
                                        component.instance_path,
                                        {},
                                    ).get("process_class")
                                    == "through_hole"
                                    for component in selected_components
                                )
                                else "procedural solver found no measurable paste or contact sites"
                            )
                        )
                    ),
                ),
            )

        return SolderSolverResult(
            request_id=request.request_id,
            solver_id=self.solver_id,
            artifacts=tuple(artifacts),
            coverage=tuple(coverage),
            warnings=tuple(dict.fromkeys(warnings)),
            metadata={
                "implementation": "pad-aware-meniscus",
                "through_hole_implementation": "volcano",
            },
        )
