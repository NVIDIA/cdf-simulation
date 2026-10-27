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

"""The solder-defect operations: observable parameters end to end.

Each solder defect type is its own operation in the style of the pose
defects: it independently selects ``ratio`` of the (remaining) eligible pool,
non-overlapping across types. Covers the four layers the operations cross:
the Python API's validation, the JobOrder compiler's expansion, the YAML
JobConfig models, and the Isaac runtime handler that selects components and
produces the defect records the solder collect step merges into the
generation request.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError as PydanticValidationError

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi domain
from cdf import (
    ComponentFilter,
    IsaacExecutionContext,
    IsaacSdgRuntime,
    JobConfig,
    CDFJob,
    ValidationError,
    build_job_from_config,
)
from cdf.execution.isaac.job_order_parameters import (
    SolderDefectParameters,
    SolderJointParameters,
)
from pcb_aoi.board_metadata import (
    BoardMetadataCatalog,
    BoardMetadataRequest,
    ComponentRecord,
    MetadataOrigin,
    MetadataSource,
    SidecarMetadataProvider,
)
from pcb_aoi.solder_generation import registered_solver_ids

#: Declaring a solder defect requires a SIMULATED solver, and surface_evolver
#: is usable only where its geometry deps import and the Evolver binary is
#: built -- deliberately absent in the deps-starved generic test job. Tests that declare it skip there; the
#: handler tests below stay (they exercise the runtime with solver 'none').
_requires_simulated = unittest.skipUnless(
    "surface_evolver" in registered_solver_ids(available_only=True),
    "surface_evolver is not registered in this environment",
)


def _job(solver: str = "surface_evolver") -> CDFJob:
    job = CDFJob(domain="pcb-aoi")
    job.scene.configure(usd_path="board.usd", root_path="/Board")
    job.scene.attach_sidecar(
        name="components",
        kind="pcb.components",
        path="components.json",
    )
    job.operations.solder_joints(solver=solver, source="components")
    return job


def _configure_capture(job: CDFJob) -> None:
    job.capture.output_start(0)
    job.capture.renderer(
        mode="pathtracing",
        resolution=(320, 180),
        spp=1,
        total_spp=2,
        max_bounces=4,
    )
    job.capture.camera(rotation_range={"x": (0, 0), "y": (0, 0), "z": -90})
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.limits(triggers=1, max_images=1)
    job.capture.outputs(rgb=True)


def _add_all_four(job: CDFJob) -> None:
    job.operations.add_insufficient(ratio=0.05, volume_ratio=[0.3, 0.7])
    job.operations.add_excess(ratio=0.02, volume_ratio=[1.8, 2.6])
    job.operations.add_missing_solder(ratio=0.02)
    job.operations.add_dry_contact(
        ratio=1.0,
        gap_fraction=[0.25, 0.5],
        component_types=["TO254"],
    )


@_requires_simulated
class SolderDefectsApiTest(unittest.TestCase):
    def test_adders_record_per_type_specs(self) -> None:
        job = _job()
        _add_all_four(job)
        specs = job.operations.solder_defect_specs
        self.assertEqual(
            sorted(specs),
            ["dry_contact", "excess", "insufficient", "missing_solder"],
        )
        self.assertEqual(specs["insufficient"]["volume_ratio"], [0.3, 0.7])
        self.assertEqual(specs["dry_contact"]["component_types"], ["TO254"])
        self.assertIsNone(specs["missing_solder"]["component_types"])

    def test_redeclaring_a_type_replaces_it(self) -> None:
        job = _job()
        job.operations.add_insufficient(ratio=0.1, volume_ratio=0.4)
        job.operations.add_insufficient(ratio=0.2, volume_ratio=0.5)
        self.assertEqual(
            job.operations.solder_defect_specs["insufficient"]["ratio"],
            0.2,
        )
        matching = [
            d
            for d in job._declarations
            if d.order_id == "operation.solder_defects.insufficient"
        ]
        self.assertEqual(len(matching), 1)

    def test_requires_solder_joints_first(self) -> None:
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        with self.assertRaisesRegex(ValidationError, "solder_joints"):
            job.operations.add_insufficient(ratio=0.1)

    def test_defect_ignoring_solver_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "honour defect intent"):
            _job(solver="procedural").operations.add_insufficient(ratio=0.1)

    def test_rejections(self) -> None:
        cases = [
            # Exclusive bounds: the boundary values are different classes.
            (
                lambda j: j.operations.add_insufficient(ratio=0.1, volume_ratio=1.0),
                "within",
            ),
            (
                lambda j: j.operations.add_excess(ratio=0.1, volume_ratio=[0.5, 2.0]),
                "within",
            ),
            (
                lambda j: j.operations.add_dry_contact(ratio=0.1, gap_fraction=0.0),
                "within",
            ),
            # Selection ratio is a [0, 1] fraction of the pool.
            (lambda j: j.operations.add_insufficient(ratio=1.5), "within"),
            # Families must be non-empty names.
            (
                lambda j: j.operations.add_excess(ratio=0.1, component_types=[""]),
                "component_types",
            ),
        ]
        for add, pattern in cases:
            with self.assertRaisesRegex(ValidationError, pattern):
                add(_job())


@_requires_simulated
class SolderDefectsCompilerTest(unittest.TestCase):
    def test_expansion_merges_types_in_call_order(self) -> None:
        job = _job()
        _add_all_four(job)
        _configure_capture(job)
        plan = job.compile_orders()
        by_id = {order.order_id: order for order in plan.orders}

        apply_order = by_id["solder_defects.apply"]
        self.assertEqual(apply_order.order_type, "solder_defects.apply")
        self.assertEqual(
            [entry["defect"] for entry in apply_order.parameters["entries"]],
            ["insufficient", "excess", "missing_solder", "dry_contact"],
        )
        self.assertEqual(apply_order.parameters["source"], "components")
        self.assertIn("solder_joints.discover", apply_order.depends_on)
        self.assertIn(
            "scene.sidecar.components",
            by_id["solder_joints.discover"].depends_on,
        )
        # Collect merges the records, so it must wait for apply.
        self.assertIn("solder_defects.apply", by_id["solder_joints.collect"].depends_on)
        # The configure declarations never become runtime orders themselves.
        self.assertNotIn("operation.solder_defects.insufficient", by_id)

    def test_structural_operations_precede_apply(self) -> None:
        job = _job()
        job.scene.select_components(types=["CAP_SMD_0603"])
        job.operations.add_shift(
            ratio=0.5,
            translate_range=0.2,
            rotate_z_range=2.0,
        )
        job.operations.add_insufficient(ratio=0.1)
        _configure_capture(job)
        plan = job.compile_orders()
        by_id = {order.order_id: order for order in plan.orders}
        self.assertIn("operation.shift", by_id["solder_defects.apply"].depends_on)


@_requires_simulated
class SolderDefectsConfigTest(unittest.TestCase):
    def _payload(self, operations: dict[str, object]) -> dict[str, object]:
        return {
            "api_version": "1.0.0",
            "domain": "pcb-aoi",
            "output_dir": "output",
            "scene": {
                "usd": "board.usdc",
                "root": "/Board",
                "components": {"sidecar": "board.components.json"},
            },
            "operations": {
                "solder_joints": {"solver": "surface_evolver"},
                **operations,
            },
            "capture": {
                "renderer": {"resolution": [320, 180], "total_spp": 2},
                "camera": {"side": "top", "rotation": {"x": 0, "y": 0, "z": -90}},
                "scan_grid": [1, 1],
                "limits": {"max_images": 1},
                "outputs": {"annotators": ["rgb"]},
            },
        }

    def test_yaml_blocks_reach_the_declarations(self) -> None:
        config = JobConfig.model_validate(
            self._payload(
                {
                    "insufficient": {"ratio": 0.05, "volume_ratio": [0.3, 0.7]},
                    "excess": {"ratio": 0.02, "volume_ratio": 2.0},
                    "missing_solder": {"ratio": 0.02},
                    "dry_contact": {
                        "ratio": 1.0,
                        "gap_fraction": [0.25, 0.5],
                        "component_types": ["TO254"],
                    },
                }
            )
        )
        job = build_job_from_config(config, base_dir=Path("fixture"))
        specs = job.operations.solder_defect_specs
        self.assertEqual(len(specs), 4)
        self.assertEqual(specs["dry_contact"]["gap_fraction"], [0.25, 0.5])
        self.assertEqual(specs["excess"]["volume_ratio"], 2.0)

    def test_unknown_block_fields_are_rejected(self) -> None:
        with self.assertRaises(PydanticValidationError):
            JobConfig.model_validate(
                self._payload(
                    {
                        "insufficient": {"ratio": 0.1, "blob": 2},
                    }
                )
            )

    def test_wrong_observable_is_rejected(self) -> None:
        # dry_contact takes gap_fraction, never volume_ratio -- the model has
        # no such field, so pydantic's extra=forbid rejects it.
        with self.assertRaises(PydanticValidationError):
            JobConfig.model_validate(
                self._payload(
                    {
                        "dry_contact": {"ratio": 0.1, "volume_ratio": 0.4},
                    }
                )
            )


class _Prim:
    def __init__(self) -> None:
        self.valid = True

    def IsValid(self) -> bool:
        return self.valid

    def __bool__(self) -> bool:
        return self.valid


class _Stage:
    def __init__(self) -> None:
        self.prims: dict[str, _Prim] = {}

    def GetPrimAtPath(self, path: str) -> _Prim:
        return self.prims.setdefault(path, _Prim())


def _instances() -> list[dict[str, object]]:
    return [
        {
            "path": f"/Board/Components/Top/{refdes}",
            "refdes": refdes,
            "family": family,
        }
        for refdes, family in (
            ("C10", "CAP_SMD_0603"),
            ("C11", "CAP_SMD_0603"),
            ("C12", "CAP_SMD_0603"),
            ("C13", "CAP_SMD_0603"),
            ("U34", "TO254"),
            ("L3", "IND_TOROID"),
        )
    ]


class SolderDefectsHandlerTest(unittest.TestCase):
    def _context(self) -> IsaacExecutionContext:
        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            stage=_Stage(),
            scene={"usd_path": "board.usd", "root_path": "/Board"},
        )
        context.board_metadata["components"] = SidecarMetadataProvider().load(
            BoardMetadataRequest(
                source_identifier="components.json",
                sidecar_data={
                    "schema_version": "v1",
                    "board_name": "Board",
                    "board_prim_path": "/Board",
                    "components": [],
                    "component_instances": _instances(),
                },
            ),
        )
        context.plan_metadata["random_seed"] = 7
        # The scene-selected pool includes every component. Solder discovery
        # establishes the narrower parent used by solder-defect filters.
        context.component_pool = tuple(
            str(instance["path"]) for instance in _instances()
        )
        context.state["solder_joints.component_pool"] = context.component_pool
        return context

    def _apply(self, context, entries):
        parameters = SolderDefectParameters(
            entries=entries,
            source="components",
        )
        defect_ops = mock.Mock()
        with mock.patch(
            "cdf.execution.isaac.job_order_runtime._standalone_module",
            return_value=defect_ops,
        ):
            output = context.runtime.apply_solder_defects(context, parameters)
        return output, defect_ops

    def test_pool_selection_produces_observable_records(self) -> None:
        context = self._context()
        output, defect_ops = self._apply(
            context,
            (
                {
                    "defect": "insufficient",
                    "ratio": 1.0,
                    "component_types": ("CAP_SMD_0603",),
                    "volume_ratio": (0.3, 0.7),
                },
                {
                    "defect": "dry_contact",
                    "ratio": 1.0,
                    "component_types": ("TO254",),
                    "gap_fraction": (0.25, 0.5),
                },
            ),
        )
        records = context.state["operation.solder_defects.records"]
        self.assertEqual(output["count"], 5)  # 4 caps + 1 TO254
        by_defect: dict[str, list] = {}
        for record in records:
            by_defect.setdefault(record["defect"], []).append(record)
        self.assertEqual(len(by_defect["insufficient"]), 4)
        for record in by_defect["insufficient"]:
            # Sampled per component: the range collapses to one scalar.
            self.assertIsInstance(record["volume_ratio"], float)
            self.assertTrue(0.3 <= record["volume_ratio"] <= 0.7)
            self.assertIsInstance(record["seed"], int)
        # Per-JOINT observable: the range passes through whole.
        self.assertEqual(
            by_defect["dry_contact"][0]["gap_fraction"],
            [0.25, 0.5],
        )
        # Every selected component was labelled.
        self.assertEqual(defect_ops.apply_defect_semantic.call_count, 5)

    def test_types_are_non_overlapping(self) -> None:
        context = self._context()
        output, _ = self._apply(
            context,
            (
                {"defect": "insufficient", "ratio": 1.0},
                {"defect": "excess", "ratio": 1.0},
            ),
        )
        # insufficient drained the shared pool; excess found it empty.
        records = context.state["operation.solder_defects.records"]
        self.assertEqual(output["count"], 6)
        self.assertEqual({r["defect"] for r in records}, {"insufficient"})
        self.assertEqual(len({r["path"] for r in records}), 6)

    def test_structural_claims_are_excluded(self) -> None:
        context = self._context()
        context.state["operation.structural.records"] = [
            {"path": "/Board/Components/Top/C10", "defect": "shift"},
        ]
        output, _ = self._apply(context, ({"defect": "excess", "ratio": 1.0},))
        self.assertEqual(output["count"], 5)
        self.assertNotIn(
            "/Board/Components/Top/C10",
            output["component_paths"],
        )

    def test_missing_pool_is_an_error(self) -> None:
        context = self._context()
        context.component_pool = ()
        context.state["solder_joints.component_pool"] = ()
        with self.assertRaisesRegex(ValidationError, "discovery pool"):
            self._apply(context, ({"defect": "insufficient", "ratio": 0.5},))

    def test_component_filter_resolves_mixed_process_component(self) -> None:
        # ict-baseboard J1 (USB-C) and artix-dc-scm J2 (M.2 socket):
        # pcb-builder authors processClass="Mixed" with
        # processClasses=[through_hole, SMT]. Solder discovery used to drop
        # such parts, so the per-operation component_filter -- whose parent
        # population is the discovery pool -- failed with "matched no
        # components" although the through-hole joints exist.
        source = MetadataSource(MetadataOrigin.USD, "board.usda")

        def _component(refdes, process_class, process_classes):
            return ComponentRecord(
                board_root="/Board",
                refdes=refdes,
                prim_path=f"/Board/Layers/Top/Components/{refdes}",
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
            ),
        )
        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            stage=_Stage(),
            scene={"usd_path": "board.usd", "root_path": "/Board"},
        )
        context.board_metadata["components"] = catalog
        context.plan_metadata["random_seed"] = 7
        context.component_pool = tuple(
            component.prim_path for component in catalog.components
        )
        discovered = context.runtime.discover_solder_joints(
            context,
            SolderJointParameters(
                solver="procedural",
                source="components",
                component_filter=ComponentFilter(refdes=("J17", "J1")),
            ),
        )
        self.assertEqual(
            context.state["solder_joints.component_pool"],
            (
                "/Board/Layers/Top/Components/J1",
                "/Board/Layers/Top/Components/J17",
            ),
        )
        # One candidate per prototype; the mixed part solders as through-hole.
        self.assertEqual(discovered["count"], 2)
        self.assertEqual(
            {candidate["process_class"] for candidate in discovered["candidates"]},
            {"through_hole"},
        )

        output, _ = self._apply(
            context,
            (
                {
                    "defect": "interrupted",
                    "ratio": 1.0,
                    "component_filter": {"refdes": ["J1"]},
                    "pad_exposure": 0.4,
                },
            ),
        )
        self.assertEqual(output["component_paths"], ["/Board/Layers/Top/Components/J1"])

        # A refdes that is genuinely absent from the board is still an error.
        with self.assertRaisesRegex(ValidationError, "matched no components"):
            self._apply(
                context,
                (
                    {
                        "defect": "missing_solder",
                        "ratio": 1.0,
                        "component_filter": {"refdes": ["J99"]},
                    },
                ),
            )

    def test_selection_is_deterministic_per_seed(self) -> None:
        entries = ({"defect": "excess", "ratio": 0.5, "volume_ratio": (1.8, 2.6)},)
        one, _ = self._apply(self._context(), entries)
        two, _ = self._apply(self._context(), entries)
        self.assertEqual(one["component_paths"], two["component_paths"])

    def test_collect_merges_solder_defect_records(self) -> None:
        context = self._context()
        self._apply(
            context, ({"defect": "insufficient", "ratio": 0.5, "volume_ratio": 0.4},)
        )
        context.state["operation.structural.records"] = [
            {"path": "/Board/Components/Top/C13", "defect": "shift"},
        ]
        from cdf.execution.isaac.job_order_parameters import (
            SolderJointParameters,
        )

        parameters = SolderJointParameters(solver="none", source="components")
        context.runtime.discover_solder_joints(context, parameters)
        context.runtime.collect_solder_joint_inputs(context, parameters)
        request = context.state["solder_joints.generation_request"]
        defects = {record["defect"] for record in request.defect_records}
        self.assertEqual(defects, {"shift", "insufficient"})


class SolderDefectEntryParametersTest(unittest.TestCase):
    """The JobOrder parameter model enforces the observable contract itself.

    Orders can arrive over transport without passing through the Python
    API's ``_observable_spec``, so the pydantic layer must hold each defect
    type's observable to the same exclusive bounds independently.
    """

    def _entry(self, **kwargs):
        from cdf.execution.isaac.job_order_parameters import (
            SolderDefectEntryParameters,
        )

        return SolderDefectEntryParameters(**kwargs)

    def test_valid_entries_pass(self) -> None:
        self._entry(defect="insufficient", ratio=0.1, volume_ratio=0.4)
        self._entry(defect="insufficient", ratio=0.1, volume_ratio=(0.3, 0.7))
        self._entry(defect="excess", ratio=0.1, volume_ratio=(1.8, 2.6))
        self._entry(defect="dry_contact", ratio=1.0, gap_fraction=0.35)
        self._entry(defect="missing_solder", ratio=0.1)
        # None observables always pass -- the solver default applies.
        self._entry(defect="insufficient", ratio=0.1)

    def test_exclusive_bounds_rejected_per_type(self) -> None:
        rejected = (
            # boundary values are different classes, so bounds are exclusive
            {"defect": "insufficient", "ratio": 0.1, "volume_ratio": 1.0},
            {"defect": "insufficient", "ratio": 0.1, "volume_ratio": 0.0},
            {"defect": "excess", "ratio": 0.1, "volume_ratio": 1.0},
            {"defect": "dry_contact", "ratio": 0.1, "gap_fraction": 0.0},
            {"defect": "dry_contact", "ratio": 0.1, "gap_fraction": 1.0},
            # a range must sit entirely within the open interval
            {
                "defect": "insufficient",
                "ratio": 0.1,
                "volume_ratio": (0.5, 1.0),
            },
            {"defect": "excess", "ratio": 0.1, "volume_ratio": (0.9, 2.0)},
        )
        for kwargs in rejected:
            with (
                self.subTest(kwargs),
                self.assertRaises(
                    PydanticValidationError,
                ),
            ):
                self._entry(**kwargs)

    def test_unordered_range_rejected(self) -> None:
        with self.assertRaises(PydanticValidationError):
            self._entry(defect="insufficient", ratio=0.1, volume_ratio=(0.7, 0.3))

    def test_wrong_observable_for_type_rejected(self) -> None:
        with self.assertRaises(PydanticValidationError):
            self._entry(defect="dry_contact", ratio=0.1, volume_ratio=0.4)
        with self.assertRaises(PydanticValidationError):
            self._entry(defect="insufficient", ratio=0.1, gap_fraction=0.4)
        with self.assertRaises(PydanticValidationError):
            self._entry(defect="missing_solder", ratio=0.1, volume_ratio=0.4)
        with self.assertRaises(PydanticValidationError):
            self._entry(defect="missing_solder", ratio=0.1, gap_fraction=0.4)


if __name__ == "__main__":
    unittest.main()
