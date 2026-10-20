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

"""Kit-free tests for PCB solder candidate discovery."""

import sys
import types
import unittest
from collections import Counter
from dataclasses import replace
from math import exp
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pcb_aoi.board_metadata import (
    BoardMetadataCatalog,
    BoardMetadataRequest,
    BoardRecord,
    ComponentRecord,
    HoleRecord,
    MetadataOrigin,
    MetadataSource,
    PadRecord,
    SidecarMetadataProvider,
    TerminalRecord,
)
from pcb_aoi.solder_generation import (
    ComponentIdentity,
    ProceduralSolderSolver,
    SolderCoverageStatus,
    SolderGenerationExtension,
    SolderGenerationRequest,
    SolderSolverRequest,
    SolderWorkItem,
    discover_solder_components,
    registered_solver_ids,
    registered_solvers,
    solder_process_class,
)
from pcb_aoi.solder_generation.procedural_meniscus_geometry import (
    LeadContactMeasurement,
    MeniscusParameters,
    build_pad_meniscus,
    is_area_array,
    measure_lead_contact,
)
from pcb_aoi.solder_generation.procedural_solver import (
    _missing_solder_side_terminals,
    _solder_side_pads,
)
from pcb_aoi.solder_generation.procedural_tht_geometry import (
    PIN_BATH_TRANSITION_RINGS,
    ThroughHoleParameters,
    ThroughHoleSite,
    _smooth_resampled_transition_normals,
    build_through_hole_volcano,
)
from pcb_aoi.solder_generation.procedural_tht_stage_analysis import (
    _cross_section_fits_pad,
    fallback_connected_pin_cross_section_hull,
)
from pcb_aoi.solder_generation.solver_base import _component_allowlist


def _catalog(manifest, identifier="components.json"):
    data = dict(manifest)
    data.setdefault("schema_version", "v1")
    data.setdefault("board_name", "Board")
    data.setdefault("board_prim_path", "/Board")
    return SidecarMetadataProvider().load(
        BoardMetadataRequest(
            source_identifier=identifier,
            sidecar_data=data,
        ),
    )


class SolderDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "schema_version": "v1",
            "board_name": "Board",
            "board_prim_path": "/Board",
            "components": [
                {
                    "name": "PASSIVE_SMD_0805",
                    "family": "PASSIVE_0805",
                    "side": "Top",
                    "prototype_path": "/Board/_Prototypes/PASSIVE_SMD_0805",
                    "source": "payloads/PASSIVE_SMD_0805.usdc",
                    "instance_paths": ["/Board/Components/Top/C1"],
                    "solder_contact_meshes": ["lead_left", "lead_right"],
                },
                {
                    "name": "NO_CONTACTS",
                    "family": "NO_CONTACTS",
                    "side": "Top",
                    "process_class": "through_hole",
                    "instance_paths": ["/Board/Components/Top/U1"],
                    "solder_contact_meshes": [],
                },
                {
                    "name": "NO_PASTE",
                    "family": "NO_PASTE",
                    "side": "Bottom",
                    "instance_paths": ["/Board/Components/Bottom/U2"],
                    "solder_contact_meshes": ["lead"],
                },
            ],
            "component_instances": [
                {
                    "path": "/Board/Components/Top/C1",
                    "refdes": "C1",
                    "side": "Top",
                    "solder_paste_prims": ["/Board/Paste/p1", "/Board/Paste/p2"],
                    "solder_paste_usdc_paths": ["board/paste.usdc"],
                },
                {
                    "path": "/Board/Components/Top/U1",
                    "refdes": "U1",
                    "side": "Top",
                    "process_class": "through_hole",
                    "solder_paste_prims": [],
                },
                {
                    "path": "/Board/Components/Bottom/U2",
                    "refdes": "U2",
                    "side": "Bottom",
                    "solder_paste_prims": [],
                },
            ],
        }
        self.catalog = _catalog(self.manifest)

    def test_default_requires_contact_meshes_and_sample_instance_paste(self):
        candidates = discover_solder_components(self.catalog)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.prototype_id, "PASSIVE_SMD_0805")
        self.assertEqual(candidate.family, "PASSIVE_0805")
        self.assertEqual(candidate.sample_refdes, "C1")
        self.assertEqual(
            candidate.solder_paste_prims,
            (
                "/Board/Paste/p1",
                "/Board/Paste/p2",
            ),
        )

    def test_explicit_filter_accepts_prototype_name_or_family(self):
        by_name = discover_solder_components(
            self.catalog,
            ["PASSIVE_SMD_0805"],
        )
        by_family = discover_solder_components(
            self.catalog,
            ["PASSIVE_0805"],
        )
        excluded = discover_solder_components(self.catalog, ["BGA_0084"])

        self.assertEqual(by_name, by_family)
        self.assertEqual(len(by_name), 1)
        self.assertEqual(excluded, ())

    def test_solver_can_discover_contact_geometry_without_source_paste(self):
        candidates = discover_solder_components(
            self.catalog,
            ["NO_PASTE"],
            require_sample_paste=False,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].prototype_id, "NO_PASTE")
        self.assertEqual(candidates[0].solder_paste_prims, ())

    def test_explicit_tht_without_contacts_reaches_solver_discovery(self):
        candidates = discover_solder_components(
            self.catalog,
            ["NO_CONTACTS"],
            require_sample_paste=False,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].process_class, "through_hole")
        self.assertEqual(candidates[0].solder_contact_meshes, ())

    def test_mixed_process_component_is_discovered_as_through_hole(self):
        # pcb-builder exports author processClass="Mixed" with
        # processClasses=[through_hole, SMT] for parts such as a USB-C
        # receptacle (SQA A10 J1) or an M.2 socket (SQA A11 J2). Their
        # through-hole legs are legitimate THT solder targets, so the part
        # must reach the discovery pool as a through-hole candidate.
        source = MetadataSource(MetadataOrigin.USD, "board.usda")

        def _component(refdes, process_class, process_classes):
            return ComponentRecord(
                board_root="/Board",
                refdes=refdes,
                prim_path=f"/Board/Components/{refdes}",
                source=source,
                component_type="connector",
                side="top",
                process_class=process_class,
                process_classes=process_classes,
            )

        catalog = BoardMetadataCatalog(
            boards=(),
            components=(
                _component("J17", "through_hole", ("through_hole",)),
                _component("J1", "Mixed", ("through_hole", "SMT")),
                _component("X1", "Unknown", ("Unknown",)),
            ),
        )

        candidates = discover_solder_components(catalog, require_sample_paste=False)

        pool = {
            path: candidate.process_class
            for candidate in candidates
            for path in candidate.instance_paths
        }
        self.assertEqual(pool["/Board/Components/J1"], "through_hole")
        self.assertEqual(pool["/Board/Components/J17"], "through_hole")
        # An aggregate class without a through-hole member is still rejected.
        self.assertNotIn("/Board/Components/X1", pool)
        self.assertEqual(solder_process_class(catalog.components[1]), "through_hole")
        self.assertEqual(solder_process_class(catalog.components[2]), "Unknown")

    def test_solvers_are_registered_by_stable_id_and_type(self):
        # 'none' and 'procedural' are always registered. The SIMULATED solver
        # (surface_evolver) registers only where its heavy deps import -- present
        # on a full dev machine, absent in the CI numpy-only image -- so the
        # simulated slot is environment-gated, not a fixed part of the contract.
        # It sorts last by id, so it appends to the deterministic ordering.
        simulated = [solver.solver_id for solver in registered_solvers("simulated")]
        self.assertIn(simulated, ([], ["surface_evolver"]))
        simulated_available = [
            solver.solver_id
            for solver in registered_solvers("simulated", available_only=True)
        ]
        self.assertEqual(
            registered_solver_ids(),
            ("none", "procedural", *simulated),
        )
        self.assertEqual(
            registered_solver_ids(available_only=True),
            ("none", "procedural", *simulated_available),
        )
        self.assertEqual(
            [solver.solver_id for solver in registered_solvers("procedural")],
            ["procedural"],
        )

    def test_extension_builds_multiple_defect_work_items_per_component(self):
        candidates = discover_solder_components(self.catalog)
        identity = (
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
        request = SolderGenerationRequest(
            request_id="/Board/SolderJoints@none",
            solver_id="none",
            source_usd_path="board.usd",
            root_path="/Board",
            output_root_path="/Board/SolderJoints",
            stage=object(),
            board_metadata=self.catalog,
            candidates=candidates,
            defect_records=(
                {
                    "path": "/Board/Components/Top/C1",
                    "defect": "shift",
                    "original_flat": identity,
                    "shift_final_flat": identity,
                },
                {
                    "path": "/Board/Components/Top/C1",
                    "defect": "tombstone",
                    "original_flat": identity,
                    "tombstone_final_flat": identity,
                },
            ),
        )

        work_items = SolderGenerationExtension().build_work_items(request)

        self.assertEqual(len(work_items), 2)
        self.assertEqual(
            [item.defect.defect_id for item in work_items],
            ["shift-000", "tombstone-001"],
        )
        self.assertTrue(
            all(
                item.work_item_id.startswith(
                    "/Board/Components/Top/C1#defect-",
                )
                for item in work_items
            ),
        )

    def test_solver_preparation_failure_falls_back_per_work_item(self):
        candidates = discover_solder_components(self.catalog)
        request = SolderGenerationRequest(
            request_id="/Board/SolderJoints@procedural",
            solver_id="procedural",
            source_usd_path="board.usd",
            root_path="/Board",
            output_root_path="/Board/SolderJoints",
            stage=object(),
            board_metadata=self.catalog,
            candidates=candidates,
            defect_records=(),
        )
        extension = SolderGenerationExtension()
        work_items = extension.build_work_items(request)
        solver_request = SolderSolverRequest(
            request_id=request.request_id,
            solver_id="procedural",
            source_usd_path=request.source_usd_path,
            root_path=request.root_path,
            work_items=work_items,
            payload={"preparation_error": "RuntimeError: backend unavailable"},
        )

        result = extension.generate(request, solver_request)

        self.assertEqual(result.solver_id, "procedural")
        self.assertEqual(result.coverage[0].status, SolderCoverageStatus.PASSTHROUGH)
        self.assertEqual(result.coverage[0].fallback_from, "procedural")
        self.assertEqual(len(result.artifacts), 1)
        self.assertTrue(result.warnings)

    def test_procedural_solver_returns_stage_free_mesh_artifact(self):
        component = ComponentIdentity(
            instance_path="/Board/Components/Top/C1",
            refdes="C1",
            side="Top",
            component_definition_path="/Board/Components/Top/PASSIVE_SMD_0805",
            prototype_path="/Board/_Prototypes/PASSIVE_SMD_0805",
        )
        work_item = SolderWorkItem(
            work_item_id=f"{component.instance_path}#nominal",
            components=(component,),
            source_paste_prims={
                component.instance_path: ("/Board/Paste/p1",),
            },
            solder_contact_prims={component.instance_path: ()},
        )
        request = SolderSolverRequest(
            request_id="/Board/SolderJoints@procedural",
            solver_id="procedural",
            source_usd_path="board.usd",
            root_path="/Board",
            work_items=(work_item,),
            payload={
                "sites_by_component": {
                    component.instance_path: {
                        "component_bounds": (-0.8, -0.5, 0.1, 0.8, 0.5, 0.8),
                        "body_bounds": (-0.6, -0.4, 0.1, 0.6, 0.4, 0.8),
                        "side": "Top",
                        "site_kind": "paste",
                        "sites": ((0.5, -0.5, 0.0, 1.0, 0.5, 0.1),),
                    },
                },
                "scene_units_per_mm": 1.0,
            },
        )

        result = ProceduralSolderSolver().generate(request)

        self.assertEqual(result.coverage[0].status, SolderCoverageStatus.GENERATED)
        self.assertEqual(len(result.artifacts), 1)
        self.assertEqual(len(result.artifacts[0].meshes), 1)
        mesh = result.artifacts[0].meshes[0]
        self.assertGreater(len(mesh.points), 8)
        self.assertNotEqual(
            mesh.transform,
            (1.0, 0.0, 0.0, 0.0) * 4,
        )
        self.assertEqual(
            result.metadata["implementation"],
            "pad-aware-meniscus",
        )

    def test_procedural_solver_filters_exact_refdes_or_component_path(self):
        component = ComponentIdentity(
            instance_path="/Board/Components/Top/C1",
            refdes="C1",
            side="Top",
            component_definition_path="/Board/Components/Top/PASSIVE_SMD_0805",
        )
        work_item = SolderWorkItem(
            work_item_id=f"{component.instance_path}#nominal",
            components=(component,),
            source_paste_prims={component.instance_path: ("/Board/Paste/p1",)},
            solder_contact_prims={component.instance_path: ()},
        )
        request = SolderSolverRequest(
            request_id="/Board/SolderJoints@procedural",
            solver_id="procedural",
            source_usd_path="board.usd",
            root_path="/Board",
            work_items=(work_item,),
            solver_options={"components": ["C10"]},
            payload={
                "sites_by_component": {
                    component.instance_path: {
                        "component_bounds": (-0.8, -0.5, 0.1, 0.8, 0.5, 0.8),
                        "body_bounds": (-0.6, -0.4, 0.1, 0.6, 0.4, 0.8),
                        "side": "Top",
                        "site_kind": "paste",
                        "sites": ((0.5, -0.5, 0.0, 1.0, 0.5, 0.1),),
                    },
                },
                "scene_units_per_mm": 1.0,
            },
        )

        excluded = ProceduralSolderSolver().generate(request)
        by_refdes = ProceduralSolderSolver().generate(
            replace(request, solver_options={"components": ["C1"]}),
        )
        by_path = ProceduralSolderSolver().generate(
            replace(
                request,
                solver_options={"components": [component.instance_path]},
            ),
        )
        unrestricted = ProceduralSolderSolver().generate(
            replace(request, solver_options={"components": []}),
        )

        self.assertEqual(excluded.coverage[0].status, SolderCoverageStatus.INELIGIBLE)
        self.assertEqual(excluded.artifacts, ())
        self.assertEqual(by_refdes.coverage[0].status, SolderCoverageStatus.GENERATED)
        self.assertEqual(by_path.coverage[0].status, SolderCoverageStatus.GENERATED)
        self.assertEqual(
            unrestricted.coverage[0].status,
            SolderCoverageStatus.GENERATED,
        )

        generation_request = SolderGenerationRequest(
            request_id=request.request_id,
            solver_id="procedural",
            source_usd_path=request.source_usd_path,
            root_path=request.root_path,
            output_root_path="/Board/SolderJoints",
            stage=object(),
            board_metadata=self.catalog,
            candidates=(),
            defect_records=(),
            solver_options={"components": ["C1", "C10"]},
        )
        partial_match = SolderGenerationExtension().generate(
            generation_request,
            replace(
                request,
                solver_options=generation_request.solver_options,
            ),
        )

        self.assertEqual(
            partial_match.coverage[0].status, SolderCoverageStatus.GENERATED
        )
        self.assertEqual(
            partial_match.warnings,
            ("components allowlist entries matched no discovered component: 'C10'",),
        )

    def test_procedural_prepare_skips_excluded_component_geometry_reads(self):
        component = ComponentIdentity(
            instance_path="/Board/Components/Top/C1",
            refdes="C1",
            side="Top",
            component_definition_path="/Board/Components/Top/PASSIVE_SMD_0805",
        )
        work_item = SolderWorkItem(
            work_item_id=f"{component.instance_path}#nominal",
            components=(component,),
            source_paste_prims={component.instance_path: ("/Board/Paste/p1",)},
            solder_contact_prims={component.instance_path: ()},
        )
        request = SolderGenerationRequest(
            request_id="/Board/SolderJoints@procedural",
            solver_id="procedural",
            source_usd_path="board.usd",
            root_path="/Board",
            output_root_path="/Board/SolderJoints",
            stage=object(),
            board_metadata=self.catalog,
            candidates=(),
            defect_records=(),
            solver_options={"components": ["C10"]},
        )
        pxr = types.ModuleType("pxr")
        pxr.Usd = types.SimpleNamespace(
            TimeCode=types.SimpleNamespace(Default=mock.Mock(return_value=object())),
        )
        pxr.UsdGeom = types.SimpleNamespace(
            BBoxCache=mock.Mock(return_value=mock.Mock()),
            GetStageMetersPerUnit=mock.Mock(return_value=0.001),
            XformCache=mock.Mock(return_value=mock.Mock()),
        )

        with (
            mock.patch.dict(sys.modules, {"pxr": pxr}),
            mock.patch(
                "pcb_aoi.solder_generation.procedural_solver._aligned_bounds",
            ) as aligned_bounds,
            mock.patch(
                "pcb_aoi.solder_generation.procedural_solver._world_mesh_points",
            ) as world_mesh_points,
        ):
            prepared = ProceduralSolderSolver().prepare(request, (work_item,))

        self.assertEqual(prepared.payload["sites_by_component"], {})
        aligned_bounds.assert_not_called()
        world_mesh_points.assert_not_called()

    def test_procedural_solver_filters_components_within_one_work_item(self):
        selected = ComponentIdentity(
            instance_path="/Board/Components/Top/C1",
            refdes="C1",
            side="Top",
            component_definition_path="/Board/Components/Top/PASSIVE_SMD_0805",
        )
        excluded = ComponentIdentity(
            instance_path="/Board/Components/Top/C2",
            refdes="C2",
            side="Top",
            component_definition_path="/Board/Components/Top/PASSIVE_SMD_0805",
        )
        work_item = SolderWorkItem(
            work_item_id="passives#nominal",
            components=(selected, excluded),
            source_paste_prims={
                selected.instance_path: ("/Board/Paste/p1",),
                excluded.instance_path: ("/Board/Paste/p2",),
            },
            solder_contact_prims={
                selected.instance_path: (),
                excluded.instance_path: (),
            },
        )
        prepared_site = {
            "component_bounds": (-0.8, -0.5, 0.1, 0.8, 0.5, 0.8),
            "body_bounds": (-0.6, -0.4, 0.1, 0.6, 0.4, 0.8),
            "side": "Top",
            "site_kind": "paste",
            "sites": ((0.5, -0.5, 0.0, 1.0, 0.5, 0.1),),
        }
        request = SolderSolverRequest(
            request_id="/Board/SolderJoints@procedural",
            solver_id="procedural",
            source_usd_path="board.usd",
            root_path="/Board",
            work_items=(work_item,),
            solver_options={"components": ["C1"]},
            payload={
                "sites_by_component": {
                    selected.instance_path: prepared_site,
                    excluded.instance_path: prepared_site,
                },
                "scene_units_per_mm": 1.0,
            },
        )

        result = ProceduralSolderSolver().generate(request)

        self.assertEqual(
            tuple(artifact.component_paths for artifact in result.artifacts),
            ((selected.instance_path,),),
        )
        self.assertEqual(result.coverage[0].component_paths, work_item.component_paths)
        self.assertEqual(result.coverage[0].status, SolderCoverageStatus.GENERATED)

    def test_component_allowlist_warns_when_discovery_finds_no_work(self):
        generation_request = SolderGenerationRequest(
            request_id="/Board/SolderJoints@procedural",
            solver_id="procedural",
            source_usd_path="board.usd",
            root_path="/Board",
            output_root_path="/Board/SolderJoints",
            stage=object(),
            board_metadata=self.catalog,
            candidates=(),
            defect_records=(),
            solver_options={"components": ["C1910"]},
        )
        solver_request = SolderSolverRequest(
            request_id=generation_request.request_id,
            solver_id="procedural",
            source_usd_path="board.usd",
            root_path="/Board",
            work_items=(),
            solver_options=generation_request.solver_options,
        )

        result = SolderGenerationExtension().generate(
            generation_request,
            solver_request,
        )

        self.assertEqual(
            result.warnings,
            (
                "no solder-generation work items were discovered",
                "components allowlist entries matched no discovered component: 'C1910'",
            ),
        )

    def test_shared_component_allowlist_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "components"):
            _component_allowlist({"components": "C1"})

    def test_public_lead_local_option_uses_internal_scaled_calibration(self):
        parameters = ProceduralSolderSolver._meniscus_parameters(
            {"lead_local_sizing": True},
            scene_units_per_mm=2.0,
        )

        self.assertTrue(parameters.lead_local_sizing)
        self.assertEqual(parameters.scene_units_per_mm, 2.0)

    def test_public_lead_local_boolean_rejects_truthy_non_boolean_values(self):
        with self.assertRaisesRegex(ValueError, "lead_local_sizing"):
            ProceduralSolderSolver._meniscus_parameters(
                {"lead_local_sizing": "true"},
                scene_units_per_mm=1.0,
            )

    def test_pad_meniscus_places_bottom_geometry_downward(self):
        mesh = build_pad_meniscus(
            pad_bounds=(0.5, -0.5, -0.1, 1.0, 0.5, 0.0),
            body_bounds=(-0.6, -0.4, -0.8, 0.6, 0.4, -0.1),
            component_bounds=(-0.8, -0.5, -0.8, 0.8, 0.5, -0.1),
            side="Bottom",
            pad_count=2,
            parameters=MeniscusParameters(noise_amplitude=0.0),
        )

        self.assertIsNotNone(mesh)
        assert mesh is not None
        self.assertLess(mesh.transform[10], 0.0)
        self.assertEqual(mesh.transform[14], -0.1)

    def test_lead_local_multi_pin_height_uses_contact_foot_not_package(self):
        parameters = MeniscusParameters(noise_amplitude=0.0)
        mesh = build_pad_meniscus(
            pad_bounds=(1.0, -0.3, 0.0, 2.0, 0.3, 0.1),
            body_bounds=(-1.0, -1.0, 0.1, 1.2, 1.0, 3.0),
            component_bounds=(-1.0, -1.0, 0.1, 1.2, 1.0, 3.0),
            side="Top",
            pad_count=16,
            parameters=parameters,
            lead_contact=LeadContactMeasurement(0.2, 1.7, 0.2, 0.0, 0.2),
        )

        self.assertIsNotNone(mesh)
        assert mesh is not None
        peak = (
            parameters.long_edge
            * parameters.bump
            * (
                1.0
                / (
                    1.0
                    + exp(-parameters.smoothness_y * (0.5 - parameters.platform_start))
                )
            )
            * (
                1.0
                - 1.0
                / (
                    1.0
                    + exp(-parameters.smoothness_y * (0.5 - parameters.platform_end))
                )
            )
        )
        self.assertAlmostEqual(abs(mesh.transform[10]) * peak, 0.18, places=6)
        self.assertAlmostEqual(abs(mesh.transform[5]) * 5.4, 0.24, places=6)
        self.assertAlmostEqual(mesh.transform[12], 1.664, places=6)

    def test_lead_local_covered_pad_without_contact_has_no_visible_fillet(self):
        mesh = build_pad_meniscus(
            pad_bounds=(-0.2, -0.2, 0.0, 0.2, 0.2, 0.1),
            body_bounds=(-1.0, -1.0, 0.1, 1.0, 1.0, 2.0),
            component_bounds=(-1.0, -1.0, 0.1, 1.0, 1.0, 2.0),
            side="Top",
            pad_count=16,
            parameters=MeniscusParameters(noise_amplitude=0.0),
        )

        self.assertIsNone(mesh)

    def test_two_pad_tall_termination_uses_pad_width_and_partial_climb(self):
        parameters = MeniscusParameters(noise_amplitude=0.0)
        mesh = build_pad_meniscus(
            pad_bounds=(0.5, -0.7, 0.0, 1.5, 0.7, 0.1),
            body_bounds=(-0.5, -0.4, 0.1, 0.8, 0.4, 3.0),
            component_bounds=(-0.5, -0.4, 0.1, 0.8, 0.4, 3.0),
            side="Top",
            pad_count=2,
            parameters=parameters,
            lead_contact=LeadContactMeasurement(0.3, 1.0, 0.5, 0.0, 2.0),
        )

        self.assertIsNotNone(mesh)
        assert mesh is not None
        self.assertAlmostEqual(abs(mesh.transform[5]) * 5.4, 1.4, places=6)
        self.assertAlmostEqual(abs(mesh.transform[0]) * 1.8, 1.008, places=6)

    def test_area_array_detection_separates_grid_from_peripheral_pads(self):
        grid = tuple(
            (x - 0.1, y - 0.1, 0.0, x + 0.1, y + 0.1, 0.1)
            for x in range(4)
            for y in range(4)
        )
        peripheral = tuple(
            (x - 0.1, y - 0.1, 0.0, x + 0.1, y + 0.1, 0.1)
            for x in range(4)
            for y in range(4)
            if x in (0, 3) or y in (0, 3)
        )

        self.assertTrue(is_area_array(grid))
        self.assertFalse(is_area_array(peripheral))

    def test_contact_measurement_uses_toe_strip_and_ignores_high_heel(self):
        contact = measure_lead_contact(
            pad_bounds=(1.0, -0.4, 0.0, 2.0, 0.4, 0.1),
            points=(
                (1.8, -0.1, 0.2),
                (1.8, 0.1, 0.2),
                (1.1, -0.1, 1.8),
                (1.1, 0.1, 1.8),
            ),
            side="Top",
            outward_axis=0,
            outward_sign=1.0,
            contact_band=0.5,
            wall_band=2.5,
        )

        self.assertIsNotNone(contact)
        assert contact is not None
        self.assertAlmostEqual(contact.foot_height, 0.1)
        self.assertAlmostEqual(contact.toe, 1.8)
        self.assertAlmostEqual(contact.width, 0.2)

    def test_solid_meniscus_is_watertight(self):
        mesh = build_pad_meniscus(
            pad_bounds=(0.5, -0.5, 0.0, 1.0, 0.5, 0.1),
            body_bounds=(-0.6, -0.4, 0.1, 0.6, 0.4, 0.8),
            component_bounds=(-0.8, -0.5, 0.1, 0.8, 0.5, 0.8),
            side="Top",
            pad_count=2,
            parameters=MeniscusParameters(noise_amplitude=0.0),
        )

        self.assertIsNotNone(mesh)
        assert mesh is not None
        edges = Counter()
        offset = 0
        for count in mesh.face_vertex_counts:
            face = mesh.face_vertex_indices[offset : offset + count]
            offset += count
            for start, end in zip(face, face[1:] + face[:1], strict=True):
                edges[tuple(sorted((start, end)))] += 1
        self.assertTrue(edges)
        self.assertEqual(set(edges.values()), {2})

    def test_through_hole_volcano_is_watertight_and_grows_downward(self):
        mesh = build_through_hole_volcano(
            ThroughHoleSite(
                center_xy=(2.0, 3.0),
                pin_radius=0.3,
                pad_radius=0.8,
                base_z=0.0,
                grow_toward_negative_z=True,
            ),
            ThroughHoleParameters(
                resolution=16,
                radial_segments=4,
                solid=True,
            ),
        )

        self.assertLess(min(point[2] for point in mesh.points), 0.0)
        self.assertEqual(mesh.transform[0], 1.0)
        edges = Counter()
        offset = 0
        for count in mesh.face_vertex_counts:
            face = mesh.face_vertex_indices[offset : offset + count]
            offset += count
            for start, end in zip(face, face[1:] + face[:1], strict=True):
                edges[tuple(sorted((start, end)))] += 1
        self.assertTrue(edges)
        self.assertEqual(set(edges.values()), {2})
        self.assertEqual(len(mesh.normals), len(mesh.points))
        for normal in mesh.normals:
            self.assertAlmostEqual(
                sum(component**2 for component in normal),
                1.0,
                places=6,
            )

    def test_through_hole_volcano_uses_concave_radial_profile(self):
        resolution = 8
        radial_segments = 4
        mesh = build_through_hole_volcano(
            ThroughHoleSite(
                center_xy=(0.0, 0.0),
                pin_radius=0.2,
                pad_radius=1.0,
                base_z=0.0,
                grow_toward_negative_z=False,
            ),
            ThroughHoleParameters(
                resolution=resolution,
                radial_segments=radial_segments,
                height_ratio=1.0,
                solid=False,
            ),
        )

        ring_heights = tuple(
            mesh.points[ring * resolution][2] for ring in range(radial_segments + 1)
        )
        crest_height = 0.8
        self.assertAlmostEqual(ring_heights[0], crest_height)
        self.assertAlmostEqual(ring_heights[-1], 0.0)
        for ring, height in enumerate(ring_heights[1:-1], start=1):
            fraction = ring / radial_segments
            linear_height = crest_height * (1.0 - fraction)
            self.assertLess(height, linear_height)

    def test_through_hole_defaults_use_smooth_tessellation(self):
        parameters = ProceduralSolderSolver._through_hole_parameters({}, 1.0)

        self.assertEqual(parameters.resolution, 64)
        self.assertEqual(parameters.radial_segments, 16)

    def test_through_hole_tessellation_can_be_overridden(self):
        parameters = ProceduralSolderSolver._through_hole_parameters(
            {
                "tht_angular_segments": 48,
                "tht_radial_segments": 10,
            },
            1.0,
        )

        self.assertEqual(parameters.resolution, 48)
        self.assertEqual(parameters.radial_segments, 10)

    def test_through_hole_solder_bath_defaults_off_and_can_be_enabled(self):
        default_parameters = ProceduralSolderSolver._through_hole_parameters({}, 1.0)
        bath_parameters = ProceduralSolderSolver._through_hole_parameters(
            {"tht_solder_bath": True},
            1.0,
        )

        self.assertFalse(default_parameters.solder_bath)
        self.assertTrue(bath_parameters.solder_bath)

        with self.assertRaisesRegex(ValueError, "tht_solder_bath must be a boolean"):
            ProceduralSolderSolver._through_hole_parameters(
                {"tht_solder_bath": 1},
                1.0,
            )

    def test_through_hole_solder_bath_envelops_pin(self):
        pin_points = (
            (-0.2, -0.2, -1.0),
            (0.2, -0.2, -1.0),
            (0.2, 0.2, -1.0),
            (-0.2, 0.2, -1.0),
            (-0.2, -0.2, 1.0),
            (0.2, -0.2, 1.0),
            (0.2, 0.2, 1.0),
            (-0.2, 0.2, 1.0),
        )
        pin_counts = (4, 4, 4, 4, 4, 4)
        pin_indices = (
            0,
            3,
            2,
            1,
            4,
            5,
            6,
            7,
            0,
            1,
            5,
            4,
            1,
            2,
            6,
            5,
            2,
            3,
            7,
            6,
            3,
            0,
            4,
            7,
        )
        site = ThroughHoleSite(
            center_xy=(0.0, 0.0),
            pin_radius=0.2,
            pad_radius=0.8,
            base_z=0.0,
            grow_toward_negative_z=True,
            pin_points=pin_points,
            pin_face_vertex_counts=pin_counts,
            pin_face_vertex_indices=pin_indices,
        )

        default_mesh = build_through_hole_volcano(
            site,
            ThroughHoleParameters(resolution=8, radial_segments=2),
        )
        bath_parameters = ThroughHoleParameters(
            resolution=64,
            radial_segments=16,
            solder_bath=True,
        )
        bath_mesh = build_through_hole_volcano(
            site,
            bath_parameters,
        )

        pin_min_z = min(point[2] for point in pin_points)
        self.assertGreater(min(point[2] for point in default_mesh.points), pin_min_z)
        self.assertLessEqual(min(point[2] for point in bath_mesh.points), pin_min_z)

        edges = Counter()
        offset = 0
        for count in bath_mesh.face_vertex_counts:
            face = bath_mesh.face_vertex_indices[offset : offset + count]
            offset += count
            for start, end in zip(face, face[1:] + face[:1], strict=True):
                edges[tuple(sorted((start, end)))] += 1
        self.assertEqual(set(edges.values()), {2})

        neck_normals = bath_mesh.normals[: bath_parameters.resolution]
        self.assertLess(
            max(abs(normal[2]) for normal in neck_normals),
            0.04,
        )

        transition_start = len(bath_mesh.points) - (
            PIN_BATH_TRANSITION_RINGS * bath_parameters.resolution
        )
        first_transition = range(
            transition_start,
            transition_start + bath_parameters.resolution,
        )
        first_dots = [
            sum(
                bath_mesh.normals[segment][axis]
                * bath_mesh.normals[transition_index][axis]
                for axis in range(3)
            )
            for segment, transition_index in enumerate(first_transition)
        ]
        self.assertGreater(min(first_dots), 0.995)

    def test_through_hole_solder_bath_caps_chamfered_pin_tip(self):
        pin_points = (
            (-0.32, -0.32, 0.0),
            (0.32, -0.32, 0.0),
            (0.32, 0.32, 0.0),
            (-0.32, 0.32, 0.0),
            (-0.32, -0.32, -2.75),
            (0.32, -0.32, -2.75),
            (0.32, 0.32, -2.75),
            (-0.32, 0.32, -2.75),
            (-0.1984, -0.1984, -3.1),
            (0.1984, -0.1984, -3.1),
            (0.1984, 0.1984, -3.1),
            (-0.1984, 0.1984, -3.1),
        )
        pin_counts = (4,) * 10
        pin_indices = (
            0,
            1,
            2,
            3,
            0,
            1,
            5,
            4,
            1,
            2,
            6,
            5,
            2,
            3,
            7,
            6,
            3,
            0,
            4,
            7,
            4,
            5,
            9,
            8,
            5,
            6,
            10,
            9,
            6,
            7,
            11,
            10,
            7,
            4,
            8,
            11,
            11,
            10,
            9,
            8,
        )
        site = ThroughHoleSite(
            center_xy=(0.0, 0.0),
            pin_radius=0.32,
            pad_radius=0.85,
            base_z=0.0,
            grow_toward_negative_z=True,
            pin_points=pin_points,
            pin_face_vertex_counts=pin_counts,
            pin_face_vertex_indices=pin_indices,
        )

        mesh = build_through_hole_volcano(
            site,
            ThroughHoleParameters(
                resolution=64,
                radial_segments=16,
                solder_bath=True,
            ),
        )

        cap_z = -3.1 - 0.32 * 0.10
        self.assertAlmostEqual(min(point[2] for point in mesh.points), cap_z)
        cap_points = []
        offset = 0
        for count in mesh.face_vertex_counts:
            face = mesh.face_vertex_indices[offset : offset + count]
            offset += count
            if all(abs(mesh.points[index][2] - cap_z) < 1e-8 for index in face):
                cap_points.extend(mesh.points[index] for index in face)
        self.assertTrue(cap_points)
        self.assertLessEqual(min(point[0] for point in cap_points), -0.1984)
        self.assertGreaterEqual(max(point[0] for point in cap_points), 0.1984)
        self.assertLessEqual(min(point[1] for point in cap_points), -0.1984)
        self.assertGreaterEqual(max(point[1] for point in cap_points), 0.1984)

    def test_transition_normal_blend_aligns_reversed_bath_normals(self):
        points = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.5),
            (0.0, 1.0, 0.5),
            (-1.0, 0.0, 0.5),
            (0.0, -1.0, 0.5),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (-1.0, 0.0, 1.0),
            (0.0, -1.0, 1.0),
        ]
        outward = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        ]
        normals = (
            outward
            + [(0.0, 0.0, 1.0)] * 4
            + [tuple(-value for value in normal) for normal in outward]
        )

        _smooth_resampled_transition_normals(
            points,
            normals,
            (tuple(range(4)), tuple(range(4, 8)), tuple(range(8, 12))),
            (0.0, 0.0),
        )

        for segment, transition_index in enumerate(range(4, 8)):
            self.assertGreater(
                sum(
                    outward[segment][axis] * normals[transition_index][axis]
                    for axis in range(3)
                ),
                0.99,
            )

    def test_through_hole_volcano_follows_offset_square_pin_throat(self):
        resolution = 32
        throat_center = (0.2, -0.1)
        mesh = build_through_hole_volcano(
            ThroughHoleSite(
                center_xy=(0.0, 0.0),
                pin_radius=0.2,
                pad_radius=0.8,
                base_z=0.0,
                grow_toward_negative_z=True,
                throat_center_xy=throat_center,
                throat_shape_xy=(
                    (-0.2, -0.2),
                    (0.2, -0.2),
                    (0.2, 0.2),
                    (-0.2, 0.2),
                ),
            ),
            ThroughHoleParameters(
                resolution=resolution,
                radial_segments=4,
                solid=True,
            ),
        )

        inner_ring = mesh.points[:resolution]
        outer_ring = mesh.points[4 * resolution : 5 * resolution]
        local_inner = tuple(
            (point[0] - throat_center[0], point[1] - throat_center[1])
            for point in inner_ring
        )
        self.assertTrue(
            all(
                abs(max(abs(point[0]), abs(point[1])) - 0.2) < 1e-6
                for point in local_inner
            ),
        )
        inner_radii = tuple((x**2 + y**2) ** 0.5 for x, y in local_inner)
        self.assertGreater(max(inner_radii), min(inner_radii) * 1.3)
        self.assertTrue(
            all(
                abs((point[0] ** 2 + point[1] ** 2) ** 0.5 - 0.8) < 1e-6
                for point in outer_ring
            ),
        )

    def test_through_hole_volcano_follows_oblong_pad_contour(self):
        resolution = 64
        mesh = build_through_hole_volcano(
            ThroughHoleSite(
                center_xy=(0.0, 0.0),
                pin_radius=0.2,
                pad_radius=0.9,
                base_z=0.0,
                grow_toward_negative_z=True,
                pad_shape_xy=(
                    (-1.1, -0.9),
                    (1.1, -0.9),
                    (1.1, 0.9),
                    (-1.1, 0.9),
                ),
            ),
            ThroughHoleParameters(
                resolution=resolution,
                radial_segments=4,
                solid=True,
            ),
        )

        outer_ring = mesh.points[4 * resolution : 5 * resolution]
        self.assertAlmostEqual(min(point[0] for point in outer_ring), -1.1)
        self.assertAlmostEqual(max(point[0] for point in outer_ring), 1.1)
        self.assertAlmostEqual(min(point[1] for point in outer_ring), -0.9)
        self.assertAlmostEqual(max(point[1] for point in outer_ring), 0.9)

    def test_cross_section_rejects_throat_spanning_neighboring_pads(self):
        self.assertFalse(
            _cross_section_fits_pad(
                (
                    (0.0, 2.5),
                    0.2,
                    (
                        (-0.2, -2.7),
                        (0.2, -2.7),
                        (0.2, 2.7),
                        (-0.2, 2.7),
                    ),
                ),
                (0.0, 0.0),
                0.7,
            ),
        )

    def test_cross_section_accepts_offset_square_inside_one_pad(self):
        self.assertTrue(
            _cross_section_fits_pad(
                (
                    (0.15, -0.1),
                    0.2,
                    (
                        (-0.2, -0.2),
                        (0.2, -0.2),
                        (0.2, 0.2),
                        (-0.2, 0.2),
                    ),
                ),
                (0.0, 0.0),
                0.7,
            ),
        )

    def test_cross_section_localizes_connected_multi_pin_geometry(self):
        hull = fallback_connected_pin_cross_section_hull(
            (
                (-0.2, -0.2),
                (0.2, -0.2),
                (0.2, 0.2),
                (-0.2, 0.2),
                (-0.2, 2.3),
                (0.2, 2.3),
                (0.2, 2.7),
                (-0.2, 2.7),
            ),
            (0.0, 0.0),
            0.7,
        )

        self.assertEqual(
            set(hull),
            {(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)},
        )

    def test_one_unbuildable_lead_does_not_retire_the_other_sites(self):
        """An overwide lead is skipped alone, not with the rest of the board.

        `ThroughHoleSite` and `build_through_hole_volcano` reject a throat that
        reaches the pad with the same message, but only the constructor used to
        sit inside the solver's per-site guard. The builder's rejection escaped
        to the extension's solver isolation, which retires the WHOLE request to
        pass-through: on a production THT panel one overwide lead left
        all sixteen work items with no solder while the job reported success.

        The pin here is deliberately shaped the way that board's are -- a
        nominal `pin_radius` far smaller than the mesh it describes -- so the
        constructor passes and only the builder objects.
        """

        def pin_box(half_width):
            """Return the points, counts and indices of a square lead."""
            return (
                (
                    (-half_width, -half_width, -1.0),
                    (half_width, -half_width, -1.0),
                    (half_width, half_width, -1.0),
                    (-half_width, half_width, -1.0),
                    (-half_width, -half_width, 0.5),
                    (half_width, -half_width, 0.5),
                    (half_width, half_width, 0.5),
                    (-half_width, half_width, 0.5),
                ),
                (4, 4, 4, 4, 4, 4),
                (
                    0,
                    1,
                    2,
                    3,
                    4,
                    7,
                    6,
                    5,
                    0,
                    4,
                    5,
                    1,
                    1,
                    5,
                    6,
                    2,
                    2,
                    6,
                    7,
                    3,
                    3,
                    7,
                    4,
                    0,
                ),
            )

        def raw_site(half_width):
            """Return one through-hole site carrying a square lead."""
            points, counts, indices = pin_box(half_width)
            return {
                "center_xy": (0.0, 0.0),
                "pin_radius": 0.2,
                "pad_radius": 0.8,
                "base_z": 0.0,
                "grow_toward_negative_z": True,
                "pin_points": points,
                "pin_face_vertex_counts": counts,
                "pin_face_vertex_indices": indices,
            }

        component = ComponentIdentity(
            instance_path="/Board/Components/Top/X1",
            refdes="X1",
            side="Top",
            component_definition_path="/Board/Components/Top/THT_CONNECTOR",
            prototype_path="/Board/_Prototypes/THT_CONNECTOR",
        )
        work_item = SolderWorkItem(
            work_item_id=f"{component.instance_path}#nominal",
            components=(component,),
            source_paste_prims={component.instance_path: ()},
            solder_contact_prims={component.instance_path: ()},
        )
        request = SolderSolverRequest(
            request_id="/Board/SolderJoints@procedural",
            solver_id="procedural",
            source_usd_path="board.usd",
            root_path="/Board",
            work_items=(work_item,),
            solver_options={"tht_solder_bath": True},
            payload={
                "sites_by_component": {
                    component.instance_path: {
                        "process_class": "through_hole",
                        "site_kind": "through_hole",
                        # 0.70 half-width overruns the 0.8 pad on the
                        # diagonal once the bath expands it; 0.35 does not.
                        "sites": (raw_site(0.70), raw_site(0.35)),
                    },
                },
                "scene_units_per_mm": 1.0,
            },
        )

        result = ProceduralSolderSolver().generate(request)

        self.assertEqual(result.coverage[0].status, SolderCoverageStatus.GENERATED)
        self.assertEqual(len(result.artifacts), 1)
        self.assertEqual(len(result.artifacts[0].meshes), 1)
        skipped = [
            message for message in result.warnings if "site 0 skipped" in message
        ]
        self.assertEqual(len(skipped), 1, result.warnings)
        self.assertIn(component.instance_path, skipped[0])
        self.assertIn("inside the pad boundary", skipped[0])

    def test_through_hole_site_rejects_cross_pad_throat(self):
        with self.assertRaisesRegex(ValueError, "inside the pad boundary"):
            ThroughHoleSite(
                center_xy=(0.0, 0.0),
                pin_radius=0.2,
                pad_radius=0.8,
                base_z=0.0,
                grow_toward_negative_z=True,
                throat_center_xy=(0.0, 2.5),
            )

    def test_procedural_solver_generates_through_hole_volcanoes(self):
        component = ComponentIdentity(
            instance_path="/Board/Components/Top/U1",
            refdes="U1",
            side="Top",
            component_definition_path="/Board/Components/Top/TO254",
            process_class="through_hole",
        )
        work_item = SolderWorkItem(
            work_item_id=f"{component.instance_path}#nominal",
            components=(component,),
            source_paste_prims={component.instance_path: ()},
            solder_contact_prims={component.instance_path: ("/Board/U1/pin",)},
        )
        request = SolderSolverRequest(
            request_id="/Board/SolderJoints@procedural",
            solver_id="procedural",
            source_usd_path="board.usd",
            root_path="/Board",
            work_items=(work_item,),
            solver_options={
                "resolution": 64,
                "solder_material_prim_path": "/World/Looks/SolderPaste",
            },
            payload={
                "sites_by_component": {
                    component.instance_path: {
                        "process_class": "through_hole",
                        "side": "Top",
                        "site_kind": "pin-pad",
                        "sites": (
                            {
                                "center_xy": (2.0, 3.0),
                                "pin_radius": 0.3,
                                "pad_radius": 0.8,
                                "base_z": 0.0,
                                "grow_toward_negative_z": True,
                                "solder_contact_prim": "/Board/U1/pin",
                            },
                        ),
                    },
                },
                "scene_units_per_mm": 1.0,
            },
        )

        result = ProceduralSolderSolver().generate(request)

        self.assertEqual(result.coverage[0].status, SolderCoverageStatus.GENERATED)
        self.assertEqual(len(result.artifacts), 1)
        self.assertEqual(len(result.artifacts[0].meshes), 1)
        self.assertEqual(
            result.artifacts[0].meshes[0].material_source_prim_path,
            "/World/Looks/SolderPaste",
        )
        self.assertGreater(len(result.artifacts[0].meshes[0].points), 1000)
        self.assertEqual(
            result.artifacts[0].metadata["process_class"],
            "through_hole",
        )
        self.assertEqual(result.metadata["through_hole_implementation"], "volcano")

    def test_tht_metadata_check_reports_geometry_fallback_fields(self):
        source = MetadataSource(MetadataOrigin.USD, "board.usda")
        terminal = TerminalRecord(
            "/Board",
            "/Board/Components/U1",
            "/Board/Components/U1/pin_1",
            source,
            number="1",
            cross_section="rectangular",
        )
        pad = PadRecord(
            "/Board",
            "/Board/Pads/P1",
            source,
            component_path="/Board/Components/U1",
            terminal_number="1",
            contact_prim_path=terminal.prim_path,
            contour=((-1.0, -1.0), (1.0, 1.0)),
        )
        hole = HoleRecord(
            "/Board",
            "/Board/Holes/H1",
            source,
            component_path="/Board/Components/U1",
            terminal_number="1",
            pad_prim_path=pad.prim_path,
            center_mm=(0.0, 0.0),
        )
        catalog = BoardMetadataCatalog(
            boards=(BoardRecord("/Board", "Board", source),),
            terminals=(terminal,),
            pads=(pad,),
            holes=(hole,),
        )

        self.assertEqual(
            catalog.tht_geometry_metadata_gaps(pad),
            (
                "hole.dimensions_mm",
                "hole.shape",
            ),
        )

    def test_procedural_solver_rejects_relative_solder_material_path(self):
        with self.assertRaisesRegex(ValueError, "solder_material_prim_path"):
            ProceduralSolderSolver._options(
                {"solder_material_prim_path": "Looks/SolderPaste"},
            )

    def test_procedural_solver_converts_tht_pad_edge_inset(self):
        parameters = ProceduralSolderSolver._through_hole_parameters(
            {"tht_pad_edge_inset_mm": 0.05},
            scene_units_per_mm=2.0,
        )

        self.assertAlmostEqual(parameters.pad_edge_inset, 0.1)

    def test_procedural_solver_rejects_negative_tht_pad_edge_inset(self):
        with self.assertRaisesRegex(ValueError, "pad_edge_inset"):
            ProceduralSolderSolver._through_hole_parameters(
                {"tht_pad_edge_inset_mm": -0.05},
                scene_units_per_mm=1.0,
            )

    def test_procedural_solver_warns_when_tht_associations_are_missing(self):
        component = ComponentIdentity(
            instance_path="/Board/Components/Top/U1",
            refdes="U1",
            side="Top",
            component_definition_path="/Board/Components/Top/TO254",
            process_class="through_hole",
        )
        work_item = SolderWorkItem(
            work_item_id=f"{component.instance_path}#nominal",
            components=(component,),
            source_paste_prims={component.instance_path: ()},
            solder_contact_prims={component.instance_path: ()},
        )
        request = SolderSolverRequest(
            request_id="/Board/SolderJoints@procedural",
            solver_id="procedural",
            source_usd_path="board.usd",
            root_path="/Board",
            work_items=(work_item,),
            payload={
                "sites_by_component": {
                    component.instance_path: {
                        "process_class": "through_hole",
                        "side": "Top",
                        "site_kind": "pin-pad",
                        "sites": (),
                    },
                },
                "scene_units_per_mm": 1.0,
                "preparation_warnings": (
                    (
                        "U1: through-hole solder generation requires "
                        "solder_pin_pad_associations"
                    ),
                ),
            },
        )

        result = ProceduralSolderSolver().generate(request)

        self.assertEqual(
            result.coverage[0].status,
            SolderCoverageStatus.NO_SOURCE_GEOMETRY,
        )
        self.assertIn("U1", result.warnings[0])

    def test_solder_side_pads_refuses_same_side_fallback(self):
        source = MetadataSource(MetadataOrigin.USD, "board.usda")
        top_pad = PadRecord(
            "/Board",
            "/Board/Layers/Top/Pad_1",
            source,
            component_path="/Board/Components/U1",
            terminal_number="1",
            side="top",
        )
        bottom_pad = PadRecord(
            "/Board",
            "/Board/Layers/Bottom/Pad_1",
            source,
            component_path="/Board/Components/U1",
            terminal_number="1",
            side="bottom",
        )

        self.assertEqual(_solder_side_pads((top_pad,), "Top"), ())
        self.assertEqual(
            _solder_side_pads((top_pad, bottom_pad), "Top"),
            (bottom_pad,),
        )
        self.assertEqual(
            _missing_solder_side_terminals((top_pad,), ()),
            ("1",),
        )
        self.assertEqual(
            _missing_solder_side_terminals((top_pad, bottom_pad), (bottom_pad,)),
            (),
        )


if __name__ == "__main__":
    unittest.main()
