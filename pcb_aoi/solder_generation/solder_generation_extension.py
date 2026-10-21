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

"""Orchestrate solver preparation, fallback, and live-stage placement."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from cdf.errors import ValidationError

from .data_types import (
    IDENTITY_MATRIX_4X4,
    ComponentIdentity,
    SolderCoordinateFrame,
    SolderCoverageStatus,
    SolderDefect,
    SolderGenerationRequest,
    SolderGeometryArtifact,
    SolderMesh,
    SolderPlacementManifest,
    SolderSerializationMode,
    SolderSolverRequest,
    SolderSolverResult,
    SolderWorkItem,
    SolderWorkItemCoverage,
)
from .metadata_discovery import solder_process_class
from .pass_through_solver import PASS_THROUGH_SOLVER_ID
from .solver_base import _component_allowlist_warning
from .solver_registry import resolve_solver

_PRIM_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


def _normalized_delta(record: Mapping[str, object]) -> tuple[float, ...]:
    """Compute the normalized component-local defect transform.

    Args:
        record: Applied structural-defect record.

    Returns:
        Row-major component-local transform. The identity is returned when an
        older record does not contain both original and final transforms.

    Raises:
        ValidationError: If a supplied transform is malformed or singular.

    """
    defect_type = str(record.get("defect") or record.get("operation") or "defect")
    original = record.get("original_flat")
    final = record.get(f"{defect_type}_final_flat")
    if original is None or final is None:
        return IDENTITY_MATRIX_4X4
    if not isinstance(original, (list, tuple)) or len(original) != 16:
        raise ValidationError("defect original_flat must contain 16 values")
    if not isinstance(final, (list, tuple)) or len(final) != 16:
        raise ValidationError(
            f"{defect_type}_final_flat must contain 16 values",
        )

    import numpy as np

    original_matrix = np.asarray(original, dtype=float).reshape((4, 4))
    final_matrix = np.asarray(final, dtype=float).reshape((4, 4))
    try:
        delta = np.linalg.inv(original_matrix) @ final_matrix
    except np.linalg.LinAlgError as exc:
        raise ValidationError("defect original transform is singular") from exc
    return tuple(float(value) for value in delta.reshape(16))


def _join_prim_path(parent: str, child: str) -> str:
    """Join a component path with one prototype-local prim name.

    Args:
        parent: Parent prim path.
        child: Child name or already-absolute prim path.

    Returns:
        Absolute prim path.

    """
    if child.startswith("/"):
        return child
    return "{}/{}".format(parent.rstrip("/"), child.lstrip("/"))


def _safe_prim_name(value: str) -> str:
    """Return a stable USD-compatible prim name.

    Args:
        value: Artifact or mesh identifier.

    Returns:
        Identifier containing only alphanumeric and underscore characters.

    """
    result = _PRIM_NAME_PATTERN.sub("_", value).strip("_")
    if not result:
        return "Solder"
    if result[0].isdigit():
        return f"_{result}"
    return result


def _coverage_by_id(
    result: SolderSolverResult,
) -> dict[str, SolderWorkItemCoverage]:
    """Index and validate solver coverage by work-item ID.

    Args:
        result: Solver result to validate.

    Returns:
        Coverage indexed by unique work-item ID.

    Raises:
        ValidationError: If duplicate coverage records are returned.

    """
    coverage = {}
    for item in result.coverage:
        if item.work_item_id in coverage:
            raise ValidationError(
                f"solver returned duplicate coverage for {item.work_item_id!r}",
            )
        coverage[item.work_item_id] = item
    return coverage


def _validate_solver_result(
    request: SolderSolverRequest,
    result: SolderSolverResult,
) -> dict[str, SolderWorkItemCoverage]:
    """Validate solver artifact and coverage correlation.

    Args:
        request: Stage-free request supplied to the solver.
        result: Solver result to validate.

    Returns:
        Coverage indexed by work-item ID.

    Raises:
        ValidationError: If artifacts or coverage do not match requested work.

    """
    work_by_id = {item.work_item_id: item for item in request.work_items}
    artifact_by_id = {}
    for artifact in result.artifacts:
        if artifact.artifact_id in artifact_by_id:
            raise ValidationError(
                f"solver returned duplicate artifact {artifact.artifact_id!r}",
            )
        unknown_work = set(artifact.work_item_ids) - set(work_by_id)
        if unknown_work:
            raise ValidationError(
                f"solver artifact {artifact.artifact_id!r} references unknown work items {sorted(unknown_work)}",
            )
        artifact_by_id[artifact.artifact_id] = artifact

    coverage = _coverage_by_id(result)
    unknown_coverage = set(coverage) - set(work_by_id)
    if unknown_coverage:
        raise ValidationError(
            f"solver returned coverage for unknown work items {sorted(unknown_coverage)}",
        )
    for work_item_id, item in coverage.items():
        expected_paths = set(work_by_id[work_item_id].component_paths)
        if set(item.component_paths) != expected_paths:
            raise ValidationError(
                f"solver coverage for {work_item_id!r} does not match requested components",
            )
        unknown_artifacts = set(item.artifact_ids) - set(artifact_by_id)
        if unknown_artifacts:
            raise ValidationError(
                f"solver coverage for {work_item_id!r} references unknown artifacts {sorted(unknown_artifacts)}",
            )
        if item.status is SolderCoverageStatus.GENERATED and not item.artifact_ids:
            raise ValidationError(
                f"generated coverage for {work_item_id!r} requires an artifact",
            )
    return coverage


class SolderGenerationExtension:
    """Own solder work construction, fallback, and live-stage edits."""

    def __init__(self) -> None:
        """Initialize extension-layer tracking."""
        self._edit_layer_identifier: str | None = None
        #: Identifier of the extension layer from the previous invocation.

        self._edit_layer: object | None = None
        #: Extension-owned layer object retained for reversible reruns.

    def build_work_items(
        self,
        request: SolderGenerationRequest,
    ) -> tuple[SolderWorkItem, ...]:
        """Build nominal and defect-specific work from catalog identities.

        Args:
            request: Extension-level generation request.

        Returns:
            Deterministically ordered solder work items.

        """
        catalog = request.board_metadata
        component_by_path = {
            component.prim_path: component for component in catalog.components
        }
        prototype_by_id = {
            prototype.prototype_id: prototype for prototype in catalog.prototypes
        }

        defects_by_path: dict[str, list[tuple[int, Mapping[str, object]]]] = (
            defaultdict(list)
        )
        for index, record in enumerate(request.defect_records):
            component_path = record.get("path")
            if component_path:
                defects_by_path[str(component_path)].append((index, record))

        work_items = []
        for candidate in request.candidates:
            paste_by_path: dict[str, tuple[str, ...]] = {}
            contacts_by_path: dict[str, tuple[str, ...]] = {}
            identities = {}
            for instance_path in candidate.instance_paths:
                component = component_by_path[instance_path]
                prototype = prototype_by_id.get(component.prototype_id or "")
                definition_path = (
                    prototype.prim_path
                    if prototype and prototype.prim_path
                    else candidate.prototype_path or candidate.prototype_id
                )
                identity = ComponentIdentity(
                    instance_path=instance_path,
                    refdes=component.refdes,
                    side=component.side or candidate.side,
                    component_definition_path=definition_path,
                    prototype_path=(
                        str(candidate.prototype_path)
                        if candidate.prototype_path
                        else None
                    ),
                    process_class=(
                        solder_process_class(component) or candidate.process_class
                    ),
                )
                identities[instance_path] = identity
                paste_by_path[instance_path] = tuple(
                    item.prim_path for item in catalog.paste_for_component(component)
                )
                instance_contacts = component.contact_prim_paths
                contacts_by_path[instance_path] = (
                    instance_contacts
                    if instance_contacts
                    else tuple(
                        _join_prim_path(instance_path, mesh_name)
                        for mesh_name in candidate.solder_contact_meshes
                    )
                )
                if instance_path not in defects_by_path:
                    work_items.append(
                        SolderWorkItem(
                            work_item_id=f"{instance_path}#nominal",
                            components=(identity,),
                            source_paste_prims={
                                instance_path: paste_by_path[instance_path],
                            },
                            solder_contact_prims={
                                instance_path: contacts_by_path[instance_path],
                            },
                        ),
                    )

            for instance_path in candidate.instance_paths:
                identity = identities[instance_path]
                for defect_index, record in defects_by_path.get(instance_path, ()):
                    defect_type = str(
                        record.get("defect") or record.get("operation") or "defect"
                    )
                    defect_id = f"{defect_type}-{defect_index:03d}"
                    defect = SolderDefect(
                        defect_id=defect_id,
                        defect_type=defect_type,
                        component_path=instance_path,
                        delta_transform=_normalized_delta(record),
                        metadata=dict(record),
                    )
                    work_items.append(
                        SolderWorkItem(
                            work_item_id=f"{instance_path}#defect-{defect_id}",
                            components=(identity,),
                            source_paste_prims={
                                instance_path: paste_by_path[instance_path],
                            },
                            solder_contact_prims={
                                instance_path: contacts_by_path[instance_path],
                            },
                            defect=defect,
                        ),
                    )

        return tuple(work_items)

    def prepare(
        self,
        request: SolderGenerationRequest,
    ) -> SolderSolverRequest:
        """Build work items and invoke solver-owned input adaptation.

        Args:
            request: Extension-level generation request.

        Returns:
            Stage-free request prepared by the selected solver.

        """
        solver = resolve_solver(request.solver_id)
        work_items = self.build_work_items(request)
        try:
            return solver.prepare(request, work_items)
        except Exception as exc:  # noqa: BLE001 - adapter failures also fall back.
            return SolderSolverRequest(
                request_id=request.request_id,
                solver_id=solver.solver_id,
                source_usd_path=request.source_usd_path,
                root_path=request.root_path,
                work_items=work_items,
                solver_options=dict(request.solver_options),
                payload={
                    "preparation_error": f"{type(exc).__name__}: {exc}",
                },
            )

    def _fallback(
        self,
        request: SolderGenerationRequest,
        work_items: Sequence[SolderWorkItem],
        *,
        failed_solver_id: str,
        reason: str,
    ) -> SolderSolverResult:
        """Generate pass-through coverage for failed or missing work.

        Args:
            request: Extension-level request used to prepare pass-through.
            work_items: Work items requiring fallback.
            failed_solver_id: Solver that did not provide final coverage.
            reason: User-facing fallback reason.

        Returns:
            Pass-through result with fallback correlation.

        """
        pass_through = resolve_solver(PASS_THROUGH_SOLVER_ID)
        fallback_request = pass_through.prepare(request, work_items)
        result = pass_through.generate(fallback_request)
        warning = (
            f"solver {failed_solver_id!r} fell back for "
            f"{len(work_items)} work item(s): {reason}"
        )
        return replace(
            result,
            solver_id=failed_solver_id,
            coverage=tuple(
                replace(
                    item,
                    message=(f"{reason}; {item.message}" if item.message else reason),
                    fallback_from=failed_solver_id,
                )
                for item in result.coverage
            ),
            warnings=(warning,),
            metadata={"fallback_solver": PASS_THROUGH_SOLVER_ID},
        )

    def generate(
        self,
        request: SolderGenerationRequest,
        solver_request: SolderSolverRequest,
    ) -> SolderSolverResult:
        """Invoke the selected solver and fill incomplete work by pass-through.

        Args:
            request: Extension-level generation request.
            solver_request: Stage-free request prepared by the solver.

        Returns:
            Complete result containing generated and fallback coverage.

        """
        solver = resolve_solver(request.solver_id)
        allowlist_warning = _component_allowlist_warning(
            solver_request.solver_options,
            solver_request.work_items,
        )
        allowlist_warnings = (allowlist_warning,) if allowlist_warning else ()
        if not solver_request.work_items:
            return SolderSolverResult(
                request_id=request.request_id,
                solver_id=solver.solver_id,
                artifacts=(),
                coverage=(),
                warnings=(
                    "no solder-generation work items were discovered",
                    *allowlist_warnings,
                ),
            )
        preparation_error = solver_request.payload.get("preparation_error")
        if isinstance(preparation_error, str):
            fallback = self._fallback(
                request,
                solver_request.work_items,
                failed_solver_id=solver.solver_id,
                reason=f"solver input preparation failed with {preparation_error}",
            )
            return replace(
                fallback,
                warnings=allowlist_warnings + fallback.warnings,
            )
        try:
            result = solver.generate(solver_request)
            if allowlist_warnings:
                result = replace(
                    result,
                    warnings=tuple(
                        dict.fromkeys(result.warnings + allowlist_warnings),
                    ),
                )
            if result.request_id != request.request_id:
                raise ValidationError("solver result request_id does not match request")
            if result.solver_id != solver.solver_id:
                raise ValidationError("solver result solver_id does not match registry")
            coverage = _validate_solver_result(solver_request, result)
        except Exception as exc:  # noqa: BLE001 - solver isolation is intentional.
            fallback = self._fallback(
                request,
                solver_request.work_items,
                failed_solver_id=solver.solver_id,
                reason=f"solver failed with {type(exc).__name__}: {exc}",
            )
            return replace(
                fallback,
                warnings=allowlist_warnings + fallback.warnings,
            )

        successful_statuses = {
            SolderCoverageStatus.GENERATED,
            SolderCoverageStatus.PASSTHROUGH,
            SolderCoverageStatus.NO_SOURCE_GEOMETRY,
        }
        fallback_items = tuple(
            work_item
            for work_item in solver_request.work_items
            if (
                work_item.work_item_id not in coverage
                or coverage[work_item.work_item_id].status not in successful_statuses
            )
        )
        if not fallback_items:
            return result

        fallback_reasons = []
        for work_item in fallback_items:
            item_coverage = coverage.get(work_item.work_item_id)
            fallback_reasons.append(
                item_coverage.message
                if item_coverage and item_coverage.message
                else "solver returned incomplete coverage"
            )
        reason = "; ".join(dict.fromkeys(fallback_reasons))
        fallback = self._fallback(
            request,
            fallback_items,
            failed_solver_id=solver.solver_id,
            reason=reason,
        )
        fallback_ids = {item.work_item_id for item in fallback_items}
        retained_coverage = tuple(
            item for item in result.coverage if item.work_item_id not in fallback_ids
        )
        retained_artifact_ids = {
            artifact_id
            for item in retained_coverage
            for artifact_id in item.artifact_ids
        }
        retained_artifacts = tuple(
            artifact
            for artifact in result.artifacts
            if artifact.artifact_id in retained_artifact_ids
        )
        return SolderSolverResult(
            request_id=request.request_id,
            solver_id=solver.solver_id,
            artifacts=retained_artifacts + fallback.artifacts,
            coverage=retained_coverage + fallback.coverage,
            warnings=result.warnings + fallback.warnings,
            metadata=result.metadata,
        )

    @staticmethod
    def _write_mesh_geometry(
        stage: object,
        prim_path: str,
        mesh: SolderMesh,
    ) -> object:
        """Write one mesh's points, topology, and extent into the edit layer.

        Frame-agnostic: the caller decides whether a placement transform is
        applied to the mesh (world frame) or to a referencing instance
        (component-local frame).

        Args:
            stage: Live USD stage.
            prim_path: Destination mesh prim path.
            mesh: ``SolderMesh`` payload to author.

        Returns:
            The defined ``UsdGeom.Mesh``.

        """
        import numpy as np
        from pxr import Gf, UsdGeom, Vt

        # A solved joint is thousands of points and this used to walk them five
        # times in Python -- once building a Gf.Vec3f each for the points, once
        # more for normals, and three times for the extent. Profiling the
        # surface_evolver path in-process put this at the top of everything
        # that is not the solver's own subprocess.
        usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
        points = np.asarray(mesh.points, dtype=np.float64).reshape(-1, 3)
        usd_mesh.CreatePointsAttr().Set(
            Vt.Vec3fArray.FromNumpy(points.astype(np.float32)),
        )
        usd_mesh.CreateFaceVertexCountsAttr().Set(list(mesh.face_vertex_counts))
        usd_mesh.CreateFaceVertexIndicesAttr().Set(list(mesh.face_vertex_indices))
        usd_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        if mesh.normals:
            normals = np.asarray(mesh.normals, dtype=np.float32).reshape(-1, 3)
            usd_mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
            usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        if mesh.points:
            # min/max in double and narrow after, as the scalar form did.
            low, high = points.min(axis=0), points.max(axis=0)
            usd_mesh.CreateExtentAttr(
                [Gf.Vec3f(*low.tolist()), Gf.Vec3f(*high.tolist())],
            )
        return usd_mesh

    @classmethod
    def _author_mesh(
        cls,
        stage: object,
        prim_path: str,
        artifact: SolderGeometryArtifact,
        mesh: SolderMesh,
    ) -> None:
        """Write one world-frame solver mesh into the extension edit layer.

        Args:
            stage: Live USD stage.
            prim_path: Destination mesh prim path.
            artifact: Artifact declaring the mesh coordinate frame.
            mesh: ``SolderMesh`` payload to author.

        Raises:
            ValidationError: If the coordinate frame is unsupported.

        """
        from pxr import Gf, UsdGeom

        if artifact.coordinate_frame is not SolderCoordinateFrame.WORLD:
            raise ValidationError(
                "attachment currently requires world-frame inline meshes",
            )
        usd_mesh = cls._write_mesh_geometry(stage, prim_path, mesh)
        if mesh.transform != IDENTITY_MATRIX_4X4:
            matrix = Gf.Matrix4d(*mesh.transform)
            UsdGeom.Xformable(usd_mesh).AddTransformOp().Set(matrix)

    def _author_component_local_artifact(
        self,
        stage: object,
        request: SolderGenerationRequest,
        artifact: SolderGeometryArtifact,
        generated_paths: list[str],
        placed_names: set[str],
        xform_cache: object,
    ) -> None:
        """Place a COMPONENT_LOCAL artifact as a shared prototype and instances.

        The artifact's meshes are in the SOLVER frame and are authored ONCE under
        a prototype scope. Each ``component_path`` then gets a placement Xform
        that references the prototype and carries the per-instance transform:
        ``solver_to_output_transform`` (SOLVER -> component-local) composed with
        the instance's component-local -> world transform read live from the
        stage. USD's row-vector convention makes the composed transform
        ``S * L`` (point' = point * S * L).

        Args:
            stage: Live USD stage with the edit layer as its edit target.
            request: Extension request providing the output root.
            artifact: COMPONENT_LOCAL artifact to place.
            generated_paths: Accumulator for authored prim paths.
            placed_names: Names already used under the ``Placed`` scope.
            xform_cache: Shared ``UsdGeom.XformCache`` for world transforms.

        """
        from pxr import Gf, Sdf, UsdGeom

        output_root = request.output_root_path.rstrip("/")
        prototypes_scope = UsdGeom.Scope.Define(stage, f"{output_root}/Prototypes")
        # The prototypes are instancing templates, not solder to render on their
        # own -- otherwise a duplicate of every footprint renders at the solver
        # origin. Hiding the SCOPE (the parent) suppresses that stray copy without
        # affecting the placements: each references the child prototype prim and
        # resolves its own visibility from the visible Placed branch, so the
        # invisibility authored here does not compose into the instances.
        prototypes_scope.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        proto_path = f"{output_root}/Prototypes/{_safe_prim_name(artifact.artifact_id)}"
        UsdGeom.Scope.Define(stage, proto_path)
        generated_paths.append(proto_path)
        for mesh in artifact.meshes:
            mesh_path = f"{proto_path}/{_safe_prim_name(mesh.name)}"
            usd_mesh = self._write_mesh_geometry(stage, mesh_path, mesh)
            if mesh.transform != IDENTITY_MATRIX_4X4:
                UsdGeom.Xformable(usd_mesh).AddTransformOp().Set(
                    Gf.Matrix4d(*mesh.transform),
                )
            if self._reuse_composed_material(
                stage,
                mesh_path,
                mesh.material_source_prim_path,
            ) is None:
                self._bind_packaged_solder_material(stage, request, mesh_path)
            generated_paths.append(mesh_path)

        solver_to_local = Gf.Matrix4d(*artifact.solver_to_output_transform)
        placed_scope = f"{output_root}/Placed"
        UsdGeom.Scope.Define(stage, placed_scope)
        for component_path in artifact.component_paths:
            prim = stage.GetPrimAtPath(component_path)
            if not prim or not prim.IsValid():
                continue
            local_to_world = xform_cache.GetLocalToWorldTransform(prim)
            placement = solver_to_local * local_to_world
            base_name = _safe_prim_name(
                component_path.rstrip("/").rsplit("/", 1)[-1],
            )
            name = base_name
            suffix = 1
            while name in placed_names:
                suffix += 1
                name = f"{base_name}_{suffix}"
            placed_names.add(name)
            placed_path = f"{placed_scope}/{name}"
            placed = UsdGeom.Xform.Define(stage, placed_path)
            placed.AddTransformOp().Set(placement)
            placed_prim = placed.GetPrim()
            placed_prim.GetReferences().AddInternalReference(Sdf.Path(proto_path))
            placed_prim.SetInstanceable(True)
            self._copy_semantics(prim, placed_prim)
            generated_paths.append(placed_path)

    @staticmethod
    def _copy_semantics(source_prim: object, destination_prim: object) -> int:
        """Copy composed Replicator semantic attributes between prims.

        Args:
            source_prim: Prim providing composed semantic attributes.
            destination_prim: Prim receiving authored semantic attributes.

        Returns:
            Number of semantic attributes copied.

        """
        copied = 0
        for source_attribute in source_prim.GetAttributes():
            name = source_attribute.GetName()
            if not name.startswith("semantic:"):
                continue
            value = source_attribute.Get()
            if value is None:
                continue
            destination_prim.CreateAttribute(
                name,
                source_attribute.GetTypeName(),
                custom=source_attribute.IsCustom(),
            ).Set(value)
            copied += 1
        return copied

    @staticmethod
    def _reuse_composed_material(
        stage: object,
        destination_prim_path: str,
        source_prim_path: str | None,
    ) -> str | None:
        """Bind a source material or prim's effective material to a mesh.

        Args:
            stage: Live USD stage.
            destination_prim_path: Generated mesh prim receiving the binding.
            source_prim_path: Source material, paste, or contact prim.

        Returns:
            Bound material path, or None when no composed binding is available.

        """
        from pxr import UsdShade

        if not source_prim_path:
            return None
        source_prim = stage.GetPrimAtPath(source_prim_path)
        destination_prim = stage.GetPrimAtPath(destination_prim_path)
        if (
            not source_prim
            or not source_prim.IsValid()
            or not destination_prim
            or not destination_prim.IsValid()
        ):
            return None
        if source_prim.IsA(UsdShade.Material):
            material = UsdShade.Material(source_prim)
        else:
            material, _relationship = UsdShade.MaterialBindingAPI(
                source_prim,
            ).ComputeBoundMaterial()
        if not material or not material.GetPrim().IsValid():
            return None
        UsdShade.MaterialBindingAPI.Apply(destination_prim).Bind(material)
        return str(material.GetPath())

    @staticmethod
    def _bind_packaged_solder_material(
        stage: object,
        request: SolderGenerationRequest,
        destination_prim_path: str,
    ) -> str | None:
        """Bind the packaged default solder material to a generated mesh.

        The fallback when a mesh names no material source: every solder mesh
        -- procedural or simulated -- gets the same reflowed-metal look from
        the repo-tracked ``assets/solder.usda``. The asset is referenced once
        under ``<output root>/Looks/Solder`` in the edit layer, so ``all``
        serialization flattens it into the output and the stage stays
        self-describing.

        Args:
            stage: Live USD stage with the edit layer as its edit target.
            request: Extension request providing the output root.
            destination_prim_path: Generated mesh prim receiving the binding.

        Returns:
            Bound material path, or None when the asset or prim is missing.

        """
        from pxr import UsdShade

        asset = Path(__file__).resolve().parent / "assets" / "solder.usda"
        if not asset.is_file():                             # pragma: no cover
            return None
        looks_path = request.output_root_path.rstrip("/") + "/Looks/Solder"
        material_prim = stage.GetPrimAtPath(looks_path)
        if not material_prim or not material_prim.IsValid():
            material_prim = stage.DefinePrim(looks_path)
            material_prim.GetReferences().AddReference(str(asset))
        destination_prim = stage.GetPrimAtPath(destination_prim_path)
        material = UsdShade.Material(material_prim)
        if (
            not destination_prim
            or not destination_prim.IsValid()
            or not material
            or not material.GetPrim().IsValid()
        ):
            return None
        UsdShade.MaterialBindingAPI.Apply(destination_prim).Bind(material)
        return looks_path

    @staticmethod
    def _serialize(
        request: SolderGenerationRequest,
        edit_layer: object,
    ) -> tuple[str, ...]:
        """Serialize extension output according to the request policy.

        Args:
            request: Extension request containing serialization policy.
            edit_layer: Extension-owned USD edit layer.

        Returns:
            Serialized output paths.

        Raises:
            ValidationError: If serialization is requested without a path.
            RuntimeError: If USD export fails.

        """
        if request.serialization is SolderSerializationMode.NONE:
            return ()
        if not request.serialization_path:
            raise ValidationError(
                "solder serialization requires serialization_path",
            )
        output_path = Path(request.serialization_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if request.serialization is SolderSerializationMode.SOLDER_JOINTS_ONLY:
            succeeded = edit_layer.Export(str(output_path))
        else:
            succeeded = request.stage.Flatten().Export(str(output_path))
        if not succeeded:
            raise RuntimeError(f"could not export solder USD to {output_path}")
        return (str(output_path),)

    def attach(
        self,
        request: SolderGenerationRequest,
        solver_request: SolderSolverRequest,
        result: SolderSolverResult,
    ) -> SolderPlacementManifest:
        """Write validated solver output into an extension-owned USD layer.

        Args:
            request: Extension-level request containing the live stage.
            solver_request: Prepared request containing all work items.
            result: Complete solver and fallback result.

        Returns:
            Placement manifest and reversible change log.

        Raises:
            ValidationError: If returned artifacts cannot be placed safely.

        """
        from pxr import Sdf, UsdGeom

        stage = request.stage
        root_layer = stage.GetRootLayer()
        previous_layer = self._edit_layer
        previous_layer_identifier = self._edit_layer_identifier
        if previous_layer_identifier in root_layer.subLayerPaths:
            root_layer.subLayerPaths.remove(previous_layer_identifier)

        edit_layer = Sdf.Layer.CreateAnonymous("cdf_solder_generation.usda")
        root_layer.subLayerPaths.insert(0, edit_layer.identifier)
        previous_target = stage.GetEditTarget()
        generated_paths = []
        suppressed_paths = []
        restored_paths = []
        all_source_by_component = tuple(
            (component_path, source_path)
            for work_item in solver_request.work_items
            for component_path, source_paths in work_item.source_paste_prims.items()
            for source_path in source_paths
        )
        # A component is "generated" when ANY artifact covers it -- including an
        # artifact with zero meshes. An empty artifact is a legitimate solder
        # outcome (the missing_solder defect deliberately authors no geometry),
        # and its source paste must still be suppressed: leaving the printed
        # deposit visible would render paste exactly where the defect says no
        # solder exists.
        generated_components = {
            component_path
            for artifact in result.artifacts
            for component_path in artifact.component_paths
        }

        try:
            stage.SetEditTarget(edit_layer)
            UsdGeom.Xform.Define(stage, request.output_root_path)
            xform_cache = UsdGeom.XformCache()
            placed_names: set[str] = set()

            for artifact in result.artifacts:
                if artifact.coordinate_frame is SolderCoordinateFrame.COMPONENT_LOCAL:
                    self._author_component_local_artifact(
                        stage,
                        request,
                        artifact,
                        generated_paths,
                        placed_names,
                        xform_cache,
                    )
                    continue
                artifact_root = "{}/{}".format(
                    request.output_root_path.rstrip("/"),
                    _safe_prim_name(artifact.artifact_id),
                )
                artifact_root_prim = UsdGeom.Xform.Define(
                    stage,
                    artifact_root,
                ).GetPrim()
                component_prim = None
                if artifact.component_paths:
                    component_prim = stage.GetPrimAtPath(artifact.component_paths[0])
                    if component_prim and component_prim.IsValid():
                        self._copy_semantics(component_prim, artifact_root_prim)
                if artifact.usd_path:
                    external_prim = artifact_root_prim
                    if artifact.usd_prim_path:
                        external_prim.GetReferences().AddReference(
                            artifact.usd_path,
                            artifact.usd_prim_path,
                        )
                    else:
                        external_prim.GetReferences().AddReference(
                            artifact.usd_path,
                        )
                    generated_paths.append(artifact_root)
                for mesh in artifact.meshes:
                    mesh_path = f"{artifact_root}/{_safe_prim_name(mesh.name)}"
                    self._author_mesh(stage, mesh_path, artifact, mesh)
                    if component_prim and component_prim.IsValid():
                        generated_prim = stage.GetPrimAtPath(mesh_path)
                        if generated_prim and generated_prim.IsValid():
                            self._copy_semantics(component_prim, generated_prim)
                    if self._reuse_composed_material(
                        stage,
                        mesh_path,
                        mesh.material_source_prim_path,
                    ) is None:
                        self._bind_packaged_solder_material(
                            stage, request, mesh_path,
                        )
                    generated_paths.append(mesh_path)

            for component_path, source_path in all_source_by_component:
                if component_path not in generated_components:
                    restored_paths.append(source_path)
                    continue
                prim = stage.GetPrimAtPath(source_path)
                if not prim or not prim.IsValid():
                    continue
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                    UsdGeom.Tokens.invisible,
                )
                suppressed_paths.append(source_path)

            serialized_paths = self._serialize(request, edit_layer)
        except Exception:
            stage.SetEditTarget(previous_target)
            if edit_layer.identifier in root_layer.subLayerPaths:
                root_layer.subLayerPaths.remove(edit_layer.identifier)
            if (
                previous_layer is not None
                and previous_layer_identifier not in root_layer.subLayerPaths
            ):
                root_layer.subLayerPaths.insert(0, previous_layer_identifier)
            raise
        stage.SetEditTarget(previous_target)
        self._edit_layer = edit_layer
        self._edit_layer_identifier = edit_layer.identifier
        return SolderPlacementManifest(
            request_id=request.request_id,
            solver_id=request.solver_id,
            output_root_path=request.output_root_path,
            edit_layer_identifier=edit_layer.identifier,
            generated_prim_paths=tuple(generated_paths),
            suppressed_source_prims=tuple(dict.fromkeys(suppressed_paths)),
            restored_source_prims=tuple(dict.fromkeys(restored_paths)),
            coverage=result.coverage,
            warnings=result.warnings,
            serialized_paths=serialized_paths,
        )
