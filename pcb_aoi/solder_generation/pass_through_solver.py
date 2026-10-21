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

"""Retain source solder-paste geometry as the no-generation solver."""

from __future__ import annotations

from collections.abc import Sequence

from .data_types import (
    SolderCoordinateFrame,
    SolderCoverageStatus,
    SolderGenerationRequest,
    SolderGeometryArtifact,
    SolderReuseScope,
    SolderSolverRequest,
    SolderSolverResult,
    SolderSolverType,
    SolderWorkItem,
    SolderWorkItemCoverage,
)
from .solver_base import SolderSolverBase

PASS_THROUGH_SOLVER_ID = "none"


class PassThroughSolderSolver(SolderSolverBase):
    """Preserve source solder-paste geometry without generating meshes."""

    solver_id = PASS_THROUGH_SOLVER_ID
    solver_type = SolderSolverType.NONE
    available = True
    unavailable_reason = None
    requires_source_paste = False

    def prepare(
        self,
        request: SolderGenerationRequest,
        work_items: Sequence[SolderWorkItem],
    ) -> SolderSolverRequest:
        """Build a stage-free pass-through request.

        Args:
            request: Extension-level generation request.
            work_items: Work items that should retain source paste.

        Returns:
            Stage-free request containing catalog-derived source prim paths.

        """
        return SolderSolverRequest(
            request_id=request.request_id,
            solver_id=self.solver_id,
            source_usd_path=request.source_usd_path,
            root_path=request.root_path,
            work_items=tuple(work_items),
            solver_options={},
        )

    def generate(self, request: SolderSolverRequest) -> SolderSolverResult:
        """Map work items to source solder-paste prims.

        Args:
            request: Stage-free pass-through request.

        Returns:
            Source-geometry artifacts and per-work-item coverage.

        """
        artifacts = []
        coverage = []
        for work_item in request.work_items:
            artifact_ids = []
            for component in work_item.components:
                source_prims = tuple(
                    work_item.source_paste_prims.get(
                        component.instance_path,
                        (),
                    ),
                )
                if not source_prims:
                    continue
                artifact_id = f"{component.instance_path}#source-paste"
                artifact_ids.append(artifact_id)
                artifacts.append(
                    SolderGeometryArtifact(
                        artifact_id=artifact_id,
                        work_item_ids=(work_item.work_item_id,),
                        component_paths=(component.instance_path,),
                        coordinate_frame=SolderCoordinateFrame.WORLD,
                        reuse_scope=SolderReuseScope.INSTANCE,
                        source_prim_paths=source_prims,
                    ),
                )

            status = (
                SolderCoverageStatus.PASSTHROUGH
                if artifact_ids
                else SolderCoverageStatus.NO_SOURCE_GEOMETRY
            )
            message = (
                None
                if artifact_ids
                else "no source solder-paste geometry is associated with this work item"
            )
            coverage.append(
                SolderWorkItemCoverage(
                    work_item_id=work_item.work_item_id,
                    component_paths=work_item.component_paths,
                    status=status,
                    artifact_ids=tuple(artifact_ids),
                    message=message,
                ),
            )

        return SolderSolverResult(
            request_id=request.request_id,
            solver_id=self.solver_id,
            artifacts=tuple(artifacts),
            coverage=tuple(coverage),
        )
