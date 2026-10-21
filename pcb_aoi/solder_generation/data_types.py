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

"""Define solder-generation requests, results, artifacts, and shared data types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from .metadata_discovery import SolderComponentCandidate

if TYPE_CHECKING:
    from pcb_aoi.board_metadata import BoardMetadataCatalog

IDENTITY_MATRIX_4X4 = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


class SolderSolverType(str, Enum):
    """Classify a registered solder solver."""

    NONE = "none"
    PROCEDURAL = "procedural"
    SIMULATED = "simulated"


class SolderSerializationMode(str, Enum):
    """Select optional solder-generation USD serialization."""

    NONE = "none"
    SOLDER_JOINTS_ONLY = "solder-joints-only"
    ALL = "all"


class SolderCoordinateFrame(str, Enum):
    """Describe the coordinate frame of returned mesh points."""

    COMPONENT_LOCAL = "component-local"
    BOARD_LOCAL = "board-local"
    WORLD = "world"


class SolderReuseScope(str, Enum):
    """Describe where one solver artifact may be reused."""

    INSTANCE = "instance"
    COMPONENT_DEFINITION = "component-definition"
    PROTOTYPE = "prototype"


class SolderCoverageStatus(str, Enum):
    """Describe the final outcome of one solder work item."""

    GENERATED = "generated"
    PASSTHROUGH = "passthrough"
    NO_SOURCE_GEOMETRY = "no-source-geometry"
    INELIGIBLE = "ineligible"
    SOLVER_FAILED = "solver-failed"
    PLACEMENT_FAILED = "placement-failed"


@dataclass(frozen=True)
class ComponentIdentity:
    """Stable identity for one component instance.

    Args:
        instance_path: Authoritative component-instance prim path.
        refdes: Human-readable reference designator.
        side: Board side containing the instance.
        component_definition_path: Metadata-defined component-group prim path.
        prototype_path: Optional reusable USD prototype prim path.
        process_class: Explicit assembly process, when supplied by metadata.

    """

    instance_path: str
    #: Authoritative component-instance prim path.

    refdes: str
    #: Human-readable component reference designator.

    side: str
    #: Board side containing the component.

    component_definition_path: str
    #: Metadata-defined component-group prim path.

    prototype_path: str | None = None
    #: Reusable USD prototype prim path, if provided by metadata.

    process_class: str | None = None
    #: Explicit assembly process, when supplied by metadata.


@dataclass(frozen=True)
class SolderDefect:
    """Normalized defect variant associated with one component.

    Args:
        defect_id: Stable defect identifier within the generation request.
        defect_type: Structural defect operation name.
        component_path: Component-instance prim path.
        delta_transform: Component-local row-major 4x4 transform.
        metadata: Additional defect diagnostics retained for the solver.

    Raises:
        ValueError: If the transform does not contain 16 values.

    """

    defect_id: str
    defect_type: str
    component_path: str
    delta_transform: tuple[float, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the normalized defect transform length."""
        if len(self.delta_transform) != 16:
            raise ValueError("delta_transform must contain 16 row-major values")


@dataclass(frozen=True)
class SolderWorkItem:
    """Solver-neutral nominal or defect-specific generation work.

    Args:
        work_item_id: Stable path-readable work-item identifier.
        components: Component instances targeted by the result.
        source_paste_prims: Original paste prims keyed by component path.
        solder_contact_prims: Contact prims keyed by component path.
        defect: Defect variant, or None for nominal work.

    """

    work_item_id: str
    components: tuple[ComponentIdentity, ...]
    source_paste_prims: Mapping[str, tuple[str, ...]]
    solder_contact_prims: Mapping[str, tuple[str, ...]]
    defect: SolderDefect | None = None

    @property
    def component_paths(self) -> tuple[str, ...]:
        """Return component-instance paths targeted by this work item.

        Returns:
            Component-instance paths in deterministic order.

        """
        return tuple(component.instance_path for component in self.components)


@dataclass(frozen=True)
class SolderGenerationRequest:
    """Extension-level request collected from the six-area workflow.

    The request is in-process and may contain a live USD stage. A solver may
    inspect the stage only while preparing a stage-free ``SolderSolverRequest``.
    It must never modify the stage.

    Args:
        request_id: Stable human-readable request identifier.
        solver_id: Selected solver registry ID.
        source_usd_path: Source PCBA USD identity used for provenance.
        root_path: PCBA root prim path.
        output_root_path: Extension-owned solder output prim path.
        stage: Live stage containing already-applied structural defects.
        board_metadata: Source-neutral component and assembly metadata.
        candidates: Discovered component-definition candidates.
        defect_records: Applied structural-defect records.
        solver_options: Solver-specific configuration.
        serialization: Optional serialization policy.
        serialization_path: Optional USD output path.

    """

    request_id: str
    solver_id: str
    source_usd_path: str
    root_path: str
    output_root_path: str
    stage: object = field(repr=False, compare=False)
    board_metadata: BoardMetadataCatalog = field(repr=False, compare=False)
    candidates: tuple[SolderComponentCandidate, ...]
    defect_records: tuple[Mapping[str, object], ...]
    solver_options: Mapping[str, object] = field(default_factory=dict)
    serialization: SolderSerializationMode = SolderSerializationMode.NONE
    serialization_path: str | None = None

    def summary(self) -> dict[str, object]:
        """Return transport-safe extension-request metadata.

        Returns:
            Summary of selected solver and collected workflow inputs.

        """
        return {
            "request_id": self.request_id,
            "solver": self.solver_id,
            "source_usd_path": self.source_usd_path,
            "root_path": self.root_path,
            "output_root_path": self.output_root_path,
            "metadata_origin": self.board_metadata.boards[0].source.origin.value,
            "metadata_source": self.board_metadata.boards[0].source.identifier,
            "candidate_count": len(self.candidates),
            "defect_count": len(self.defect_records),
            "serialization": self.serialization.value,
            "serialization_path": self.serialization_path,
        }


@dataclass(frozen=True)
class SolderSolverRequest:
    """Stage-free request prepared for one registered solver.

    Args:
        request_id: Correlation identifier shared with the extension request.
        solver_id: Solver registry ID.
        source_usd_path: Source PCBA USD identity used for provenance.
        root_path: PCBA root prim path.
        work_items: Normalized nominal and defect-specific work.
        solver_options: Validated solver-specific configuration.
        payload: Solver-conditioned, transport-safe input data.

    """

    request_id: str
    solver_id: str
    source_usd_path: str
    root_path: str
    work_items: tuple[SolderWorkItem, ...]
    solver_options: Mapping[str, object] = field(default_factory=dict)
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SolderMesh:
    """Serializable mesh returned by a solder solver.

    Args:
        name: Stable mesh name within its artifact.
        points: Mesh points in the artifact coordinate frame.
        face_vertex_counts: Vertex count for each polygon.
        face_vertex_indices: Flattened polygon vertex indices.
        transform: Row-major 4x4 placement transform.
        material_source_prim_path: Optional source prim whose composed material
            binding should be reused for the generated mesh.
        normals: Optional per-point normals for smooth surface shading.

    Raises:
        ValueError: If the transform does not contain 16 values.

    """

    name: str
    points: tuple[tuple[float, float, float], ...]
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]
    transform: tuple[float, ...] = IDENTITY_MATRIX_4X4
    material_source_prim_path: str | None = None
    normals: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        """Validate the mesh transform and optional normals."""
        if len(self.transform) != 16:
            raise ValueError("mesh transform must contain 16 row-major values")
        if self.normals and len(self.normals) != len(self.points):
            raise ValueError("mesh normals must contain one value per point")


@dataclass(frozen=True)
class SolderGeometryArtifact:
    """Geometry or source references returned by one solver.

    Args:
        artifact_id: Stable identifier within the solver result.
        work_item_ids: Work items covered by the artifact.
        component_paths: Component instances receiving the artifact.
        coordinate_frame: Coordinate frame used by returned mesh points.
        reuse_scope: Declared artifact reuse applicability.
        meshes: Inline meshes supplied for extension-owned USD authoring.
        source_prim_paths: Existing source prims retained by pass-through.
        usd_path: Optional external solver-generated USD artifact.
        usd_prim_path: Optional prim within the external USD artifact.
        source_to_solver_transform: Row-major input normalization transform.
        solver_to_output_transform: Row-major output placement transform.
        metadata: Solver-specific artifact metadata.

    """

    artifact_id: str
    work_item_ids: tuple[str, ...]
    component_paths: tuple[str, ...]
    coordinate_frame: SolderCoordinateFrame
    reuse_scope: SolderReuseScope
    meshes: tuple[SolderMesh, ...] = ()
    source_prim_paths: tuple[str, ...] = ()
    usd_path: str | None = None
    usd_prim_path: str | None = None
    source_to_solver_transform: tuple[float, ...] = IDENTITY_MATRIX_4X4
    solver_to_output_transform: tuple[float, ...] = IDENTITY_MATRIX_4X4
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SolderWorkItemCoverage:
    """Final solver coverage for one work item.

    Args:
        work_item_id: Covered work-item identifier.
        component_paths: Component instances covered by the outcome.
        status: Final generation or fallback status.
        artifact_ids: Artifacts associated with the outcome.
        message: Optional user-facing diagnostic.
        fallback_from: Solver ID that failed to provide final coverage.

    """

    work_item_id: str
    component_paths: tuple[str, ...]
    status: SolderCoverageStatus
    artifact_ids: tuple[str, ...] = ()
    message: str | None = None
    fallback_from: str | None = None


@dataclass(frozen=True)
class SolderSolverResult:
    """Validated geometry result and solver artifact manifest.

    Args:
        request_id: Correlation identifier from the solver request.
        solver_id: Solver that produced the result.
        artifacts: Generated geometry and source-geometry references.
        coverage: Final outcome for each handled work item.
        warnings: Structured user-facing warnings.
        metadata: Backend-specific result metadata.

    """

    request_id: str
    solver_id: str
    artifacts: tuple[SolderGeometryArtifact, ...]
    coverage: tuple[SolderWorkItemCoverage, ...]
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SolderPlacementManifest:
    """Record extension-owned stage edits for one generation request.

    Args:
        request_id: Correlation identifier from the generation request.
        solver_id: Configured solver ID.
        output_root_path: Extension-owned solder output prim path.
        edit_layer_identifier: USD layer containing extension-authored edits.
        generated_prim_paths: Generated solder prims authored by the extension.
        suppressed_source_prims: Original paste prims suppressed by the extension.
        restored_source_prims: Original paste prims retained or restored.
        coverage: Final placement coverage by work item.
        warnings: Generation, fallback, and placement warnings.
        serialized_paths: Optional serialized USD artifacts.

    """

    request_id: str
    solver_id: str
    output_root_path: str
    edit_layer_identifier: str | None
    generated_prim_paths: tuple[str, ...]
    suppressed_source_prims: tuple[str, ...]
    restored_source_prims: tuple[str, ...]
    coverage: tuple[SolderWorkItemCoverage, ...]
    warnings: tuple[str, ...] = ()
    serialized_paths: tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        """Return JSON-compatible placement metadata.

        Returns:
            Placement counts, warnings, and output identities.

        """
        return {
            "request_id": self.request_id,
            "solver": self.solver_id,
            "output_root_path": self.output_root_path,
            "edit_layer_identifier": self.edit_layer_identifier,
            "generated_prim_paths": list(self.generated_prim_paths),
            "suppressed_source_prims": list(self.suppressed_source_prims),
            "restored_source_prims": list(self.restored_source_prims),
            "coverage": [
                {
                    "work_item_id": item.work_item_id,
                    "component_paths": list(item.component_paths),
                    "status": item.status.value,
                    "artifact_ids": list(item.artifact_ids),
                    "message": item.message,
                    "fallback_from": item.fallback_from,
                }
                for item in self.coverage
            ],
            "warnings": list(self.warnings),
            "serialized_paths": list(self.serialized_paths),
        }
