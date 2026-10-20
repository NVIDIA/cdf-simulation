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

"""PCB-specific solder-generation discovery, data types, and solvers."""

from .data_types import (
    ComponentIdentity,
    SolderCoordinateFrame,
    SolderCoverageStatus,
    SolderDefect,
    SolderGenerationRequest,
    SolderGeometryArtifact,
    SolderMesh,
    SolderPlacementManifest,
    SolderReuseScope,
    SolderSerializationMode,
    SolderSolverRequest,
    SolderSolverResult,
    SolderSolverType,
    SolderWorkItem,
    SolderWorkItemCoverage,
)
from .metadata_discovery import (
    SolderComponentCandidate,
    discover_solder_components,
    solder_process_class,
)
from .pass_through_solver import PassThroughSolderSolver
from .procedural_solver import ProceduralSolderSolver
from .solder_generation_extension import SolderGenerationExtension
from .solver_base import SolderSolverBase
from .solver_registry import (
    register_solver,
    registered_solver_ids,
    registered_solvers,
    resolve_solver,
)

register_solver(PassThroughSolderSolver())
register_solver(ProceduralSolderSolver())


def _register_surface_evolver() -> str | None:
    """Register the Surface Evolver simulated solver, if it is importable.

    Import failure is NOT fatal -- an environment lacking the solver's
    dependencies (numpy, pxr, manifold3d) must still be able to use the
    procedural and pass-through solvers. But it is reported rather than
    swallowed: a silently absent solver looks exactly like one that was never
    written. A missing Evolver *binary* is a different case that handles itself,
    registering as unavailable with a reason.

    Returns:
        None when registered, otherwise a human-readable reason.

    """
    # The solver lives in-tree now (pcb_aoi/solder_generation/
    # surface_evolver_solver.py), so the sys.path bootstrap and the
    # dual-import-order race guard that used to live here are gone with the
    # out-of-tree package they served.
    # SystemExit is caught alongside Exception on purpose: solder_evolver's
    # boolean.py raises SystemExit (not ImportError) when manifold3d is absent,
    # and SystemExit does not inherit from Exception. Without this, a numpy-only
    # environment aborts this whole module import mid-way -- after the two base
    # solvers are registered -- so a later re-import re-runs the registration
    # and trips "solder solver 'none' is already registered".
    try:
        from .surface_evolver_solver import register as _register
    except (Exception, SystemExit) as exc:   # noqa: BLE001 -- reported, not raised
        return "%s: %s" % (type(exc).__name__, exc)
    _register(register_solver)
    return None


SURFACE_EVOLVER_UNAVAILABLE = _register_surface_evolver()
if SURFACE_EVOLVER_UNAVAILABLE:  # pragma: no cover - environment dependent
    import warnings

    warnings.warn(
        "surface_evolver solder solver not registered: %s"
        % SURFACE_EVOLVER_UNAVAILABLE,
        RuntimeWarning,
        stacklevel=2,
    )

__all__ = [
    "ComponentIdentity",
    "PassThroughSolderSolver",
    "SolderComponentCandidate",
    "SolderCoordinateFrame",
    "SolderCoverageStatus",
    "SolderDefect",
    "SolderGenerationExtension",
    "SolderGenerationRequest",
    "SolderGeometryArtifact",
    "SolderMesh",
    "SolderPlacementManifest",
    "SolderReuseScope",
    "SolderSerializationMode",
    "SolderSolverBase",
    "SolderSolverRequest",
    "SolderSolverResult",
    "SolderSolverType",
    "SolderWorkItem",
    "SolderWorkItemCoverage",
    "discover_solder_components",
    "register_solver",
    "registered_solver_ids",
    "registered_solvers",
    "resolve_solver",
    "solder_process_class",
]
