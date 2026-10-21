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

"""Kit-free contract tests for the first built-in Isaac handlers."""

import asyncio
import contextlib
import json
import math
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_STANDALONE = _REPO_ROOT / "scripts" / "sdg" / "standalone"
if str(_STANDALONE) not in sys.path:
    sys.path.insert(0, str(_STANDALONE))

from defects_utils.selection import describe_targeting

from cdf import (
    IsaacExecutionContext,
    IsaacInProcessExecutor,
    IsaacSdgRuntime,
    JobExecutionError,
    JobOrder,
    JobOrderPlan,
    ValidationError,
    create_default_isaac_handler_registry,
)
from cdf.execution.isaac.capture_execution import (
    _aim_along_view,
    _capture_reference_plane_z,
    _component_roi_metadata,
    _nearest_prim_path,
    _pin_roi_preflight_warnings,
    _pin_roi_selection_specs,
    _positions_within_image_budget,
    _require_one_write_per_cell,
    _resolve_pin_roi_selection,
    _resolve_roi_selection,
    _roi_apertures,
    _roi_selection_specs,
    _trigger_camera_rotation,
    _warmup,
    _write_settled_frame,
    _writer_armed_at_attach,
    build_capture_config,
    execute_capture,
)
from cdf.execution.isaac.job_order_parameters import (
    CaptureRenderParameters,
    ComponentAnnotationParameters,
    ComponentSelectionParameters,
    GeometryAnnotationParameters,
    MaterialOverrideParameters,
    MissingParameters,
    PostprocessCropParameters,
    PostprocessGenerateJointRoisParameters,
    PostprocessStitchParameters,
    ReversePolarityParameters,
    ShiftParameters,
    SidecarLoadParameters,
    SideflipParameters,
    SolderJointParameters,
    StageSerializationParameters,
    TombstoneParameters,
)
from pcb_aoi.board_metadata import (
    BoardMetadataCatalog,
    BoardMetadataRequest,
    BoardRecord,
    ComponentRecord,
    MetadataOrigin,
    MetadataSource,
    PadRecord,
    SidecarMetadataProvider,
    TerminalRecord,
)
from pcb_aoi.execution.board_instances import (
    BoardInstanceCohort,
    ComponentProjectionRecord,
    InstanceSemanticProjection,
    _internal_reference_source,
    mark_instanced_stage_mutation,
    refresh_instanced_capture_stage,
    resolve_board_instance_cohort,
)
from pcb_aoi.solder_generation import (
    SolderGenerationExtension,
    SolderGenerationRequest,
    SolderPlacementManifest,
    SolderSerializationMode,
    SolderSolverRequest,
    SolderSolverResult,
)


def _catalog(data=None, identifier="components.json"):
    document = {
        "schema_version": "v1",
        "board_name": "Board",
        "board_prim_path": "/Board",
        "components": [],
        "component_instances": [],
    }
    document.update(data or {})
    return SidecarMetadataProvider().load(
        BoardMetadataRequest(
            source_identifier=identifier,
            sidecar_data=document,
        ),
    )


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    async def load_scene(self, context, parameters):
        self.calls.append(("scene", parameters))
        context.stage = object()
        context.scene = parameters.model_dump(mode="python")
        return {"root_path": parameters.root_path}

    def load_sidecar(self, context, parameters):
        self.calls.append(("sidecar", parameters))
        context.board_metadata[parameters.name] = _catalog(
            identifier=parameters.path,
        )
        return {"name": parameters.name}

    def select_components(self, context, parameters):
        self.calls.append(("selection", parameters))
        context.component_types = tuple(parameters.types or ())
        context.component_pool = ("/Board/Components/R1",)
        return {"count": 1}

    def apply_component_annotations(self, context, parameters):
        self.calls.append(("annotations", parameters))
        return {"count": len(context.component_pool)}

    def apply_tombstone(self, context, parameters):
        self.calls.append(("tombstone", parameters))
        return {"count": 1, "component_paths": [context.component_pool[0]]}

    def apply_shift(self, context, parameters):
        self.calls.append(("shift", parameters))
        return {"count": 1, "component_paths": [context.component_pool[0]]}

    def apply_sideflip(self, context, parameters):
        self.calls.append(("sideflip", parameters))
        return {"count": 1, "component_paths": [context.component_pool[0]]}

    def apply_reverse_polarity(self, context, parameters):
        self.calls.append(("reverse_polarity", parameters))
        return {"count": 1, "component_paths": [context.component_pool[0]]}

    def apply_missing(self, context, parameters):
        self.calls.append(("missing", parameters))
        return {"count": 1, "component_paths": [context.component_pool[0]]}

    async def render_capture(self, context, parameters):
        self.calls.append(("capture", parameters))
        return {"output_dir": parameters.output_dir, "frame_count": 1}


class _FakeUsdPrim:
    def __init__(
        self,
        path,
        *,
        prototype=None,
        instance=False,
        valid=True,
    ):
        self.path = path
        self.prototype = prototype
        self.instance = instance
        self.valid = valid

    def GetPath(self):
        return self.path

    def GetPrototype(self):
        return self.prototype

    def IsInstance(self):
        return self.instance

    def IsValid(self):
        return self.valid


class _FakeReferenceList:
    """Sdf reference list operation carrying internal and external items."""

    def __init__(self, internal=(), external=()):
        self.explicitItems = tuple(
            types.SimpleNamespace(assetPath="", primPath=path) for path in internal
        ) + tuple(
            types.SimpleNamespace(assetPath="board.usd", primPath=path)
            for path in external
        )
        self.prependedItems = ()
        self.addedItems = ()
        self.appendedItems = ()


class _FakePrimSpec:
    """One authored spec, at the path it has in ITS OWN layer."""

    def __init__(self, path, internal=(), external=()):
        self.path = path
        self.referenceList = _FakeReferenceList(internal, external)


class _FakeStackPrim:
    """A prim whose authored namespace may differ from its composed one."""

    def __init__(self, composed_path, prim_stack):
        self._composed_path = composed_path
        self._prim_stack = prim_stack

    def GetPath(self):
        return self._composed_path

    def GetPrimStack(self):
        return self._prim_stack


def _scene_plan() -> JobOrderPlan:
    return JobOrderPlan(
        orders=(
            JobOrder(
                order_id="scene.load",
                order_type="scene.load",
                parameters={
                    "usd_path": "board.usd",
                    "root_path": "/Board",
                    "archive_member": None,
                    "camera_path": None,
                    "camera_xform_path": None,
                    "ring_light_root": None,
                    "horizontal_aperture": None,
                },
            ),
            JobOrder(
                order_id="scene.sidecar.components",
                order_type="scene.sidecar.load",
                parameters={
                    "name": "components",
                    "kind": "pcb.components",
                    "path": "components.json",
                },
                depends_on=("scene.load",),
            ),
            JobOrder(
                order_id="scene.components.select",
                order_type="scene.components.select",
                parameters={
                    "source": None,
                    "types": ["PASSIVE_SMD_0805"],
                },
                depends_on=("scene.load",),
            ),
        ),
    )


def _write_frame_on_step(replicator, common, *, skip_pass=None):
    """Make each capture step write an ``rgb`` file, as an attached writer does.

    ``skip_pass`` names a pass directory whose writer writes nothing, the way a
    writer attached before the renderer delivers frames does.
    """
    active = {}

    def _start(writer, _frame_index):
        active["dir"] = Path(writer.initialize.call_args.kwargs["output_dir"])
        # Attached right after this; Replicator arms a writer attached while the
        # orchestrator is STARTED or STEPPED for its next step.
        orchestrator = replicator.orchestrator
        active["scheduled"] = orchestrator.get_status() in (
            orchestrator.Status.STARTED,
            orchestrator.Status.STEPPED,
        )

        def _schedule():
            active["scheduled"] = True

        writer.schedule_write.side_effect = _schedule

    def _seed(frame):
        active["frame"] = int(frame)

    async def _step(**_kwargs):
        # A manually triggered writer writes only the step it was scheduled for.
        if not active.pop("scheduled", False):
            return
        if active["dir"].name != skip_pass:
            (active["dir"] / f"rgb_{active['frame']:04d}.png").write_bytes(b"")

    common.set_writer_start_frame_index.side_effect = _start
    common.set_replicator_seed_for_output_frame.side_effect = _seed
    replicator.orchestrator.step_async = mock.AsyncMock(side_effect=_step)


class IsaacAsyncExecutorTest(unittest.TestCase):
    def test_default_handlers_execute_against_injected_runtime(self):
        runtime = _FakeRuntime()
        context = IsaacExecutionContext(runtime=runtime)
        executor = IsaacInProcessExecutor(
            create_default_isaac_handler_registry(),
            context=context,
        )

        result = asyncio.run(executor.execute_async(_scene_plan()))

        self.assertTrue(result.success)
        self.assertEqual(
            [name for name, _ in runtime.calls],
            ["scene", "sidecar", "selection"],
        )
        self.assertEqual(result.outputs["scene.components.select"], {"count": 1})
        self.assertEqual(context.component_types, ("PASSIVE_SMD_0805",))
        self.assertEqual(
            context.output_view,
            result.outputs,
        )

    def test_sync_execute_rejects_async_handlers_before_mutation(self):
        runtime = _FakeRuntime()
        executor = IsaacInProcessExecutor(
            create_default_isaac_handler_registry(),
            context=IsaacExecutionContext(runtime=runtime),
        )

        with self.assertRaisesRegex(ValidationError, "execute_async"):
            executor.execute(_scene_plan())
        self.assertEqual(runtime.calls, [])

    def test_registry_preflight_rejects_unimplemented_orders(self):
        runtime = _FakeRuntime()
        plan = JobOrderPlan(
            orders=(
                _scene_plan().orders[0],
                JobOrder(
                    order_id="future",
                    order_type="capture.future",
                    depends_on=("scene.load",),
                ),
            ),
        )
        executor = IsaacInProcessExecutor(
            create_default_isaac_handler_registry(),
            context=IsaacExecutionContext(runtime=runtime),
        )

        with self.assertRaisesRegex(ValidationError, "capture.future"):
            asyncio.run(executor.execute_async(plan))
        self.assertEqual(runtime.calls, [])

    def test_stage_serialization_flattens_and_exports_composed_stage(self):
        runtime = IsaacSdgRuntime()
        stage = mock.Mock()
        flattened_stage = stage.Flatten.return_value
        flattened_stage.Export.return_value = True
        context = IsaacExecutionContext(runtime=runtime, stage=stage)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "final_stage.usda"
            output = runtime.serialize_stage(
                context,
                StageSerializationParameters(
                    serialization_path=str(destination),
                ),
            )

        stage.Flatten.assert_called_once_with()
        flattened_stage.Export.assert_called_once_with(str(destination.resolve()))
        self.assertEqual(output, {"serialization_path": str(destination)})
        self.assertEqual(
            context.state["stage.serialization_local_path"],
            str(destination.resolve()),
        )

    def test_instance_refresh_leaves_unmodified_stage_open(self):
        stage = object()
        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            stage=stage,
        )
        open_stage = mock.AsyncMock()

        result = asyncio.run(
            refresh_instanced_capture_stage(context, open_stage),
        )

        self.assertIs(result, stage)
        open_stage.assert_not_awaited()

    def test_instance_refresh_reopens_flattened_stage_after_mutation(self):
        stage = mock.Mock()
        stage.Flatten.return_value.Export.return_value = True
        stage.GetPrimAtPath.return_value.HasAuthoredMetadata.return_value = False
        refreshed_stage = mock.Mock()
        refreshed_stage.GetPrimAtPath.return_value.IsValid.return_value = True
        usd_context = mock.Mock()
        usd_context.get_stage.return_value = refreshed_stage
        app = object()
        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            stage=stage,
            board_instance_cohort=BoardInstanceCohort(
                instance_paths=("/World/Panel/I0",),
                representative_instance_path="/World/Panel/I0",
                prototype_path="/__Prototype_1",
                prototype_source_path="/Board",
            ),
        )
        open_stage = mock.AsyncMock()
        mark_instanced_stage_mutation(context, "annotations.geometry")

        omni = types.ModuleType("omni")
        omni_kit = types.ModuleType("omni.kit")
        omni_kit_app = types.ModuleType("omni.kit.app")
        omni_kit_app.get_app = mock.Mock(return_value=app)
        omni_usd = types.ModuleType("omni.usd")
        omni_usd.get_context = mock.Mock(return_value=usd_context)
        omni.kit = omni_kit
        omni.usd = omni_usd
        omni_kit.app = omni_kit_app

        with (
            context.storage_scope(),
            mock.patch.dict(
                sys.modules,
                {
                    "omni": omni,
                    "omni.kit": omni_kit,
                    "omni.kit.app": omni_kit_app,
                    "omni.usd": omni_usd,
                },
            ),
        ):
            result = asyncio.run(
                refresh_instanced_capture_stage(context, open_stage),
            )

        self.assertIs(result, refreshed_stage)
        self.assertIs(context.stage, refreshed_stage)
        stage.Flatten.assert_called_once_with()
        instance_root = stage.GetPrimAtPath.return_value
        instance_root.SetInstanceable.assert_called_once_with(False)
        instance_root.ClearInstanceable.assert_called_once_with()
        exported_path = stage.Flatten.return_value.Export.call_args.args[0]
        self.assertIn("cdf-annotated-capture-", exported_path)
        open_stage.assert_awaited_once_with(app, exported_path)
        self.assertNotIn("capture.instanced_stage_mutations", context.state)
        self.assertEqual(
            context.state["capture.instanced_stage_refresh_reasons"],
            ("annotations.geometry",),
        )

    def test_capture_handler_delegates_to_injected_runtime(self):
        runtime = _FakeRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        plan = JobOrderPlan(
            orders=(
                JobOrder(
                    order_id="capture.render",
                    order_type="capture.render",
                    parameters={"output_dir": "output"},
                ),
            ),
        )
        executor = IsaacInProcessExecutor(
            create_default_isaac_handler_registry(),
            context=context,
        )

        result = asyncio.run(executor.execute_async(plan))

        self.assertTrue(result.success)
        self.assertEqual(runtime.calls[0][0], "capture")
        self.assertEqual(result.outputs["capture.render"]["frame_count"], 1)

    def test_executor_publishes_storage_only_after_plan_success(self):
        events = []

        @contextlib.contextmanager
        def staged_output(_uri):
            with tempfile.TemporaryDirectory() as temporary_directory:
                try:
                    yield temporary_directory
                except BaseException:
                    events.append("discard")
                    raise
                else:
                    events.append("publish")

        def write_output(_order, context):
            output = context.output_tree("s3://bucket/run")
            Path(output, "result.txt").write_text("done", encoding="utf-8")
            return {"output_dir": "s3://bucket/run"}

        context = IsaacExecutionContext(runtime=object())
        executor = IsaacInProcessExecutor(
            {"capture.render": write_output},
            context=context,
        )
        plan = JobOrderPlan(
            orders=(
                JobOrder(
                    order_id="capture.render",
                    order_type="capture.render",
                ),
            ),
        )

        with mock.patch(
            "cdf.execution.isaac.execution_context.file_utils.output_tree",
            side_effect=staged_output,
        ):
            result = executor.execute(plan)

        self.assertTrue(result.success)
        self.assertEqual(events, ["publish"])

    def test_executor_discards_storage_after_handler_failure(self):
        events = []

        @contextlib.contextmanager
        def staged_output(_uri):
            with tempfile.TemporaryDirectory() as temporary_directory:
                try:
                    yield temporary_directory
                except BaseException:
                    events.append("discard")
                    raise
                else:
                    events.append("publish")

        def fail_after_write(_order, context):
            output = context.output_tree("s3://bucket/run")
            Path(output, "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("capture failed")

        context = IsaacExecutionContext(runtime=object())
        executor = IsaacInProcessExecutor(
            {"capture.render": fail_after_write},
            context=context,
        )
        plan = JobOrderPlan(
            orders=(
                JobOrder(
                    order_id="capture.render",
                    order_type="capture.render",
                ),
            ),
        )

        with (
            mock.patch(
                "cdf.execution.isaac.execution_context.file_utils.output_tree",
                side_effect=staged_output,
            ),
            self.assertRaises(JobExecutionError),
        ):
            executor.execute(plan)

        self.assertEqual(events, ["discard"])

    def test_configuration_handlers_retain_typed_intent_and_plan_metadata(self):
        context = IsaacExecutionContext(runtime=IsaacSdgRuntime(), stage=object())
        plan = JobOrderPlan(
            metadata={
                "domain": "pcb-aoi",
                "random_seed": 7,
                "smoke_overrides": {"resolution": [320, 180]},
            },
            orders=(
                JobOrder(
                    order_id="lighting",
                    order_type="appearance.lighting.configure",
                    parameters={"mode": "rgb_light"},
                ),
                JobOrder(
                    order_id="renderer",
                    order_type="capture.renderer.configure",
                    parameters={
                        "mode": "pathtracing",
                        "resolution": [1920, 1080],
                        "spp": 1,
                        "total_spp": 32,
                        "max_bounces": 4,
                    },
                ),
                JobOrder(
                    order_id="outputs",
                    order_type="capture.outputs.configure",
                    parameters={
                        "rgb": True,
                        "image_format": "png",
                        "distance_to_camera": True,
                    },
                ),
            ),
        )
        executor = IsaacInProcessExecutor(
            create_default_isaac_handler_registry(),
            context=context,
        )

        result = asyncio.run(executor.execute_async(plan))

        self.assertTrue(result.success)
        self.assertEqual(context.plan_metadata["random_seed"], 7)
        self.assertEqual(
            context.plan_metadata["smoke_overrides"],
            {"resolution": [320, 180]},
        )
        self.assertIsInstance(context.plan_metadata["smoke_overrides"], dict)
        self.assertEqual(context.state["appearance.lighting"]["mode"], "rgb_light")
        self.assertEqual(
            context.state["capture.renderer"]["resolution"],
            (1920, 1080),
        )
        self.assertTrue(context.state["capture.outputs"]["distance_to_camera"])

    def test_lighting_apply_handler_authors_the_configured_rig(self):
        context = IsaacExecutionContext(runtime=IsaacSdgRuntime(), stage=object())
        plan = JobOrderPlan(
            orders=(
                JobOrder(
                    order_id="lighting",
                    order_type="appearance.lighting.configure",
                    parameters={"mode": "modus_light", "modus": {}},
                ),
                JobOrder(
                    order_id="appearance.lighting.default.apply",
                    order_type="appearance.lighting.apply",
                    parameters={"camera_side": "bottom"},
                    depends_on=("lighting",),
                ),
            ),
        )
        executor = IsaacInProcessExecutor(
            create_default_isaac_handler_registry(),
            context=context,
        )
        expected = {
            "mode": "modus_light",
            "root_path": "/World/_aoi_modus_light",
            "camera_side": "bottom",
        }

        with mock.patch(
            "cdf.execution.isaac.capture_execution._apply_lighting_appearance",
            return_value=expected,
        ) as apply_lighting:
            result = asyncio.run(executor.execute_async(plan))

        self.assertTrue(result.success)
        apply_lighting.assert_called_once_with(context, "bottom")
        self.assertEqual(
            result.outputs["appearance.lighting.default.apply"],
            expected,
        )

    def test_annotation_and_tombstone_handlers_use_validated_contracts(self):
        runtime = _FakeRuntime()
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=object(),
            scene={"root_path": "/Board"},
            component_types=("PASSIVE_SMD_0805",),
            component_pool=("/Board/Components/R1", "/Board/Components/R2"),
        )
        context.board_metadata["components"] = _catalog()
        plan = JobOrderPlan(
            metadata={"random_seed": 7},
            orders=(
                JobOrder(
                    order_id="annotations",
                    order_type="annotations.components.apply",
                    parameters={"source": "components", "classes": None},
                ),
                JobOrder(
                    order_id="tombstone",
                    order_type="operation.tombstone.apply",
                    parameters={
                        "ratio": 0.5,
                        "angle_range": [70, 90],
                        "component_types": ["PASSIVE_SMD_0805"],
                        "auto_filter": True,
                        "eligibility_source": "components",
                    },
                ),
            ),
        )
        executor = IsaacInProcessExecutor(
            create_default_isaac_handler_registry(),
            context=context,
        )

        result = asyncio.run(executor.execute_async(plan))

        self.assertTrue(result.success)
        self.assertEqual(
            [name for name, _ in runtime.calls],
            ["annotations", "tombstone"],
        )
        self.assertEqual(result.outputs["tombstone"]["count"], 1)

    def test_registry_executes_every_transform_operation_contract(self):
        runtime = _FakeRuntime()
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=object(),
            component_pool=("/Board/Components/R1",),
        )
        plan = JobOrderPlan(
            orders=(
                JobOrder(
                    order_id="shift",
                    order_type="operation.shift.apply",
                    parameters={
                        "ratio": 0.1,
                        "translate_range": 0.2,
                        "rotate_z_range": 15,
                    },
                ),
                JobOrder(
                    order_id="sideflip",
                    order_type="operation.sideflip.apply",
                    parameters={"ratio": 0.2, "angle_range": [60, 80]},
                ),
                JobOrder(
                    order_id="reverse",
                    order_type="operation.reverse_polarity.apply",
                    parameters={
                        "ratio": 0.3,
                        "component_types": ["IC_SMD"],
                    },
                ),
            ),
        )
        executor = IsaacInProcessExecutor(
            create_default_isaac_handler_registry(),
            context=context,
        )

        result = asyncio.run(executor.execute_async(plan))

        self.assertTrue(result.success)
        self.assertEqual(
            [name for name, _ in runtime.calls],
            ["shift", "sideflip", "reverse_polarity"],
        )

    def test_registry_executes_missing_operation_contract(self):
        runtime = _FakeRuntime()
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=object(),
            component_pool=("/Board/Components/R1",),
        )
        plan = JobOrderPlan(
            orders=(
                JobOrder(
                    order_id="missing",
                    order_type="operation.missing.apply",
                    parameters={"ratio": 0.25},
                ),
            ),
        )
        executor = IsaacInProcessExecutor(
            create_default_isaac_handler_registry(),
            context=context,
        )

        result = asyncio.run(executor.execute_async(plan))

        self.assertTrue(result.success)
        self.assertEqual(runtime.calls[0][0], "missing")

    def test_contract_failure_is_reported_on_the_failed_order(self):
        runtime = _FakeRuntime()
        plan = JobOrderPlan(
            orders=(
                JobOrder(
                    order_id="scene.load",
                    order_type="scene.load",
                    parameters={"usd_path": "board.usd", "root_path": "relative"},
                ),
            ),
        )
        executor = IsaacInProcessExecutor(
            create_default_isaac_handler_registry(),
            context=IsaacExecutionContext(runtime=runtime),
        )

        with self.assertRaises(JobExecutionError) as caught:
            asyncio.run(executor.execute_async(plan))

        self.assertEqual(caught.exception.order.order_id, "scene.load")
        self.assertFalse(caught.exception.result.success)
        self.assertEqual(runtime.calls, [])


class IsaacSdgRuntimeTest(unittest.TestCase):
    def test_material_override_authors_translucent_omnipbr_inputs(self):
        authored_inputs = {}
        shader = mock.Mock()

        def create_input(name, value_type):
            shader_input = mock.Mock()
            shader_input.Set.side_effect = lambda value: authored_inputs.__setitem__(
                name,
                (value, value_type),
            )
            return shader_input

        shader.CreateInput.side_effect = create_input
        material = mock.Mock()
        material_prim = mock.Mock()
        material_prim.IsValid.return_value = True
        material_prim.IsA.return_value = True
        stage = mock.Mock()
        stage.GetPrimAtPath.return_value = material_prim

        pxr = types.ModuleType("pxr")
        pxr.Gf = types.SimpleNamespace(
            Vec2f=lambda *values: values,
            Vec3f=lambda *values: values,
        )
        pxr.Sdf = types.SimpleNamespace(
            AssetPath=lambda value: value,
            ValueTypeNames=types.SimpleNamespace(
                Asset="Asset",
                Bool="Bool",
                Color3f="Color3f",
                Float="Float",
                Float2="Float2",
            ),
        )
        pxr.Usd = types.SimpleNamespace(
            EditContext=lambda *_args: contextlib.nullcontext(),
            EditTarget=lambda _layer: object(),
        )
        pxr.UsdShade = types.SimpleNamespace(
            Material=mock.Mock(return_value=material),
            Shader=types.SimpleNamespace(Define=mock.Mock(return_value=shader)),
        )
        context = IsaacExecutionContext(runtime=IsaacSdgRuntime(), stage=stage)

        with mock.patch.dict(sys.modules, {"pxr": pxr}):
            IsaacSdgRuntime().apply_material_override(
                context,
                MaterialOverrideParameters(
                    material_path="/World/Looks/SolderMask",
                    opacity=0.5,
                ),
            )

        self.assertEqual(authored_inputs["enable_opacity"], (True, "Bool"))
        self.assertEqual(authored_inputs["opacity_constant"], (0.5, "Float"))

    def test_instance_cohort_requires_every_shared_prototype_instance(self):
        prototype = _FakeUsdPrim("/__Prototype_1")
        source = _FakeUsdPrim("/World/BoardDefinition")
        first_path = "/World/Panel/Board_0"
        second_path = "/World/Panel/Board_1"
        first = _FakeUsdPrim(first_path, prototype=prototype, instance=True)
        second = _FakeUsdPrim(second_path, prototype=prototype, instance=True)
        stage = mock.Mock()
        stage.GetPrimAtPath.side_effect = {
            first_path: first,
            second_path: second,
            source.path: source,
        }.__getitem__
        stage.GetPseudoRoot.return_value = _FakeUsdPrim("/")
        pxr = types.ModuleType("pxr")
        pxr.Usd = types.SimpleNamespace(PrimRange=lambda _root: (first, second))

        with (
            mock.patch.dict(sys.modules, {"pxr": pxr}),
            mock.patch(
                "pcb_aoi.execution.board_instances._internal_reference_source",
                return_value=source.path,
            ),
        ):
            cohort = resolve_board_instance_cohort(
                stage,
                "/World/Panel",
                (first_path, second_path),
            )
            with self.assertRaisesRegex(ValidationError, "omitted paths"):
                resolve_board_instance_cohort(
                    stage,
                    "/World/Panel",
                    (first_path,),
                )

        self.assertEqual(cohort.prototype_source_path, source.path)
        self.assertEqual(cohort.representative_instance_path, first_path)

    def test_instance_cohort_rejects_malformed_instancing(self):
        first_path = "/World/Panel/Board_0"
        second_path = "/World/Panel/Board_1"
        source = _FakeUsdPrim("/World/BoardDefinition")
        first = _FakeUsdPrim(
            first_path,
            prototype=_FakeUsdPrim("/__Prototype_1"),
            instance=True,
        )
        second = _FakeUsdPrim(
            second_path,
            prototype=_FakeUsdPrim("/__Prototype_2"),
            instance=True,
        )
        stage = mock.Mock()
        stage.GetPrimAtPath.side_effect = {
            first_path: first,
            second_path: second,
            source.path: source,
        }.__getitem__
        pxr = types.ModuleType("pxr")
        pxr.Usd = types.SimpleNamespace(PrimRange=lambda _root: (first, second))

        with (
            mock.patch.dict(sys.modules, {"pxr": pxr}),
            mock.patch(
                "pcb_aoi.execution.board_instances._internal_reference_source",
                return_value=source.path,
            ),
            self.assertRaisesRegex(ValidationError, "do not share one USD prototype"),
        ):
            resolve_board_instance_cohort(
                stage,
                "/World/Panel",
                (first_path, second_path),
            )

    def test_internal_reference_source_keeps_path_when_opened_as_root(self):
        # Batch OPENS the board, so the authoring layer's namespace and the
        # stage's are the same and the authored path is already correct.
        prim = _FakeStackPrim(
            "/World/Carrier/StepRepeats/SR_0_0",
            (
                _FakePrimSpec(
                    "/World/Carrier/StepRepeats/SR_0_0",
                    ("/World/_class_Boards/PCA",),
                ),
            ),
        )

        self.assertEqual(
            _internal_reference_source(prim),
            "/World/_class_Boards/PCA",
        )

    def test_internal_reference_source_rebases_path_under_reference_mount(self):
        # A parent stage may reference the board under /World/Board_x, which
        # shifts every prim in the layer down one level. The authored target
        # still names the pre-shift path, and reading it verbatim asked the
        # stage for a prim that exists only in the other namespace -- the
        # "references missing source" failure on the panelised THT board.
        prim = _FakeStackPrim(
            "/World/Board_x/Carrier/StepRepeats/SR_0_0",
            (
                _FakePrimSpec(
                    "/World/Carrier/StepRepeats/SR_0_0",
                    ("/World/_class_Boards/PCA",),
                ),
            ),
        )

        self.assertEqual(
            _internal_reference_source(prim),
            "/World/Board_x/_class_Boards/PCA",
        )

    def test_internal_reference_source_ignores_external_references(self):
        prim = _FakeStackPrim(
            "/World/Board_x/Carrier/StepRepeats/SR_0_0",
            (
                _FakePrimSpec(
                    "/World/Carrier/StepRepeats/SR_0_0",
                    (),
                    external=("/World/_class_Boards/PCA",),
                ),
            ),
        )

        self.assertIsNone(_internal_reference_source(prim))

    def test_scene_zip_auto_selects_sole_root_usd(self):
        context = IsaacExecutionContext(runtime=IsaacSdgRuntime())
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "board.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("board.usda", "#usda 1.0")
                archive.writestr("payloads/mesh.usdc", "payload")
            with context.storage_scope():
                selected = Path(context.localize_scene(archive_path))
                self.assertEqual(selected.name, "board.usda")
                self.assertTrue(selected.is_file())
                self.assertTrue((selected.parent / "payloads" / "mesh.usdc").is_file())
            self.assertFalse(selected.parent.exists())

    def test_scene_zip_requires_selection_for_multiple_root_usds(self):
        context = IsaacExecutionContext(runtime=IsaacSdgRuntime())
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "boards.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("board.usda", "board")
                archive.writestr("panel.usdc", "panel")
            with (
                context.storage_scope(),
                self.assertRaisesRegex(ValidationError, "multiple.*archive_member"),
            ):
                context.localize_scene(archive_path)

        context = IsaacExecutionContext(runtime=IsaacSdgRuntime())
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "boards.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("board.usda", "board")
                archive.writestr("panel.usdc", "panel")
            with context.storage_scope():
                selected = context.localize_scene(archive_path, "panel.usdc")
                self.assertEqual(Path(selected).read_text(), "panel")

    def test_scene_zip_ignores_nested_usd_candidates(self):
        context = IsaacExecutionContext(runtime=IsaacSdgRuntime())
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "nested.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("nested/board.usda", "board")
            with (
                context.storage_scope(),
                self.assertRaisesRegex(ValidationError, "no root-level USD"),
            ):
                context.localize_scene(archive_path)

    def test_archive_member_is_rejected_for_non_zip_scene(self):
        context = IsaacExecutionContext(runtime=IsaacSdgRuntime())
        with self.assertRaisesRegex(ValidationError, "only valid.*ZIP"):
            context.localize_scene("board.usda", "board.usda")

    def test_scene_loader_localizes_remote_tree_for_stage_open(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime)
        context.localize_scene = mock.Mock(return_value="C:/staged/board.usd")
        root = mock.Mock()
        root.IsValid.return_value = True
        stage = mock.Mock()
        stage.GetPrimAtPath.return_value = root
        usd_context = mock.Mock()
        usd_context.get_stage.return_value = stage
        common = mock.Mock()
        common.open_usd_stage = mock.AsyncMock()
        catalog = mock.Mock()
        catalog.components = (mock.Mock(), mock.Mock())

        omni = types.ModuleType("omni")
        omni_kit = types.ModuleType("omni.kit")
        omni_kit_app = types.ModuleType("omni.kit.app")
        omni_kit_app.get_app = mock.Mock(return_value=object())
        omni_usd = types.ModuleType("omni.usd")
        omni_usd.get_context = mock.Mock(return_value=usd_context)
        omni.kit = omni_kit
        omni.usd = omni_usd
        omni_kit.app = omni_kit_app

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "omni": omni,
                    "omni.kit": omni_kit,
                    "omni.kit.app": omni_kit_app,
                    "omni.usd": omni_usd,
                },
            ),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
                return_value=common,
            ),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._load_usd_board_metadata",
                return_value=catalog,
            ) as load_metadata,
        ):
            output = asyncio.run(
                runtime.load_scene(
                    context,
                    mock.Mock(
                        usd_path="s3://bucket/boards/board.usd",
                        archive_member=None,
                        root_path="/Board",
                        instance_paths=None,
                        model_dump=mock.Mock(
                            return_value={
                                "usd_path": "s3://bucket/boards/board.usd",
                                "root_path": "/Board",
                            },
                        ),
                    ),
                ),
            )

        context.localize_scene.assert_called_once_with(
            "s3://bucket/boards/board.usd",
            None,
        )
        common.open_usd_stage.assert_awaited_once_with(
            omni_kit_app.get_app.return_value,
            "C:/staged/board.usd",
        )
        self.assertEqual(context.scene["usd_path"], "C:/staged/board.usd")
        self.assertEqual(output["usd_path"], "s3://bucket/boards/board.usd")
        self.assertEqual(output["metadata_components"], 2)
        self.assertIs(context.board_metadata["usd"], catalog)
        load_metadata.assert_called_once_with(
            stage,
            "/Board",
            "C:/staged/board.usd",
        )

    def test_scene_loader_uses_shared_instance_prototype_source(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime)
        context.localize_scene = mock.Mock(return_value="C:/staged/panel.usda")
        root = mock.Mock()
        root.IsValid.return_value = True
        stage = mock.Mock()
        stage.GetPrimAtPath.return_value = root
        usd_context = mock.Mock()
        usd_context.get_stage.return_value = stage
        common = mock.Mock()
        common.open_usd_stage = mock.AsyncMock()
        catalog = mock.Mock()
        catalog.components = (mock.Mock(),)
        instance_paths = (
            "/World/Panel/StepRepeats/SR_0_0",
            "/World/Panel/StepRepeats/SR_0_1",
        )
        cohort = BoardInstanceCohort(
            instance_paths=instance_paths,
            representative_instance_path=instance_paths[0],
            prototype_path="/__Prototype_1",
            prototype_source_path="/World/_class_Boards/Board",
        )

        omni = types.ModuleType("omni")
        omni_kit = types.ModuleType("omni.kit")
        omni_kit_app = types.ModuleType("omni.kit.app")
        omni_kit_app.get_app = mock.Mock(return_value=object())
        omni_usd = types.ModuleType("omni.usd")
        omni_usd.get_context = mock.Mock(return_value=usd_context)
        omni.kit = omni_kit
        omni.usd = omni_usd
        omni_kit.app = omni_kit_app

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "omni": omni,
                    "omni.kit": omni_kit,
                    "omni.kit.app": omni_kit_app,
                    "omni.usd": omni_usd,
                },
            ),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
                return_value=common,
            ),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime.resolve_board_instance_cohort",
                return_value=cohort,
            ) as resolve_cohort,
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._load_usd_board_metadata",
                return_value=catalog,
            ) as load_metadata,
        ):
            output = asyncio.run(
                runtime.load_scene(
                    context,
                    mock.Mock(
                        usd_path="panel.usda",
                        archive_member=None,
                        root_path="/World/Panel",
                        instance_paths=instance_paths,
                        model_dump=mock.Mock(
                            return_value={
                                "usd_path": "panel.usda",
                                "root_path": "/World/Panel",
                                "instance_paths": instance_paths,
                            },
                        ),
                    ),
                ),
            )

        resolve_cohort.assert_called_once_with(
            stage,
            "/World/Panel",
            instance_paths,
        )
        load_metadata.assert_called_once_with(
            stage,
            "/World/_class_Boards/Board",
            "C:/staged/panel.usda",
        )
        self.assertEqual(
            context.scene["representative_instance_path"],
            instance_paths[0],
        )
        self.assertIs(context.board_instance_cohort, cohort)
        self.assertEqual(output["prototype_path"], "/__Prototype_1")

    def test_capture_config_is_derived_from_typed_context_state(self):
        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            stage=object(),
            scene={
                "root_path": "/Board",
                "camera_path": "/World/Rig/Camera",
                "camera_xform_path": "/World/Rig",
                "ring_light_root": "/World/Rig/Lights",
                "horizontal_aperture": None,
            },
            component_types=("PASSIVE_SMD_0805",),
            plan_metadata={"random_seed": 7},
        )
        context.state.update(
            {
                "capture.renderer": {
                    "mode": "pathtracing",
                    "resolution": (1920, 1080),
                    "spp": 1,
                    "total_spp": 32,
                    "max_bounces": 4,
                },
                "capture.camera": {
                    "side": "bottom",
                    "rotation_range": {"x": (0, 0), "y": (0, 0), "z": -90},
                },
                "capture.scan_grid": {"x_count": 1, "y_count": 1},
                "capture.visibility": {
                    "hide": ("solder",),
                    "show": ("hole", "pin_tip"),
                },
                "capture.outputs": {
                    "rgb": True,
                    "image_format": "png",
                    "frame_padding": 4,
                    "rename_to_grid_index": True,
                },
                "appearance.lighting": {
                    "use_scene_lights": False,
                    "preserve_scene_light_color": False,
                    "ring_light": False,
                    "layers": None,
                    "white_light": {"intensity": (1000.0, 1000.0)},
                },
            },
        )

        config = build_capture_config(context, "output")

        self.assertEqual(config["camera_rotation"]["z_fixed"], -90.0)
        self.assertEqual(config["camera_side"], "bottom")
        self.assertEqual(config["pathtracing"]["total_spp"], 32)
        self.assertEqual(config["scan_grid"], {"x_num": 1, "y_num": 1})
        self.assertEqual(config["writer"]["image_output_format"], "png")
        self.assertNotIn("rename_to_grid_index", config["writer"])
        self.assertTrue(config["rename_to_grid_index"])
        self.assertEqual(config["hidden_geometry"], ["solder"])
        self.assertEqual(config["shown_geometry"], ["hole", "pin_tip"])
        self.assertEqual(config["random_seed"], 7)
        self.assertNotIn("layers", config["lighting"])
        self.assertEqual(
            config["lighting"]["white_light"]["intensity"],
            (1000.0, 1000.0),
        )

    def test_stacked_roi_selectors_are_deterministic_and_deduplicated(self):
        paths = [f"/Board/Components/Top/C{index}" for index in range(1, 4)]
        catalog = _catalog(
            {
                "components": [
                    {
                        "name": "CAPACITOR",
                        "family": "PASSIVE",
                        "side": "Top",
                        "prototype_path": "/Board/_Prototypes/CAPACITOR",
                        "source": "payloads/CAPACITOR.usdc",
                        "instance_paths": paths,
                    },
                ],
                "component_instances": [
                    {"path": path, "refdes": f"C{index}"}
                    for index, path in enumerate(paths, 1)
                ],
            },
        )
        context = IsaacExecutionContext(runtime=IsaacSdgRuntime())
        context.state["operation.structural.records"] = [
            {"path": paths[1], "defect": "shift"},
            {"path": paths[2], "defect": "tombstone"},
        ]
        components_by_path = {
            component.prim_path: component for component in catalog.components
        }
        selections = (
            {"refdes": ("C1",), "count": None, "padding": 0.5},
            {"component_types": ("CAPACITOR",), "count": 1, "padding": 0.5},
            {
                "defects": ("shift", "tombstone"),
                "count": None,
                "padding": 0.5,
            },
        )

        first = _resolve_roi_selection(
            context,
            catalog,
            "/Board",
            selections,
            components_by_path,
            random_seed=17,
        )
        second = _resolve_roi_selection(
            context,
            catalog,
            "/Board",
            selections,
            components_by_path,
            random_seed=17,
        )

        first_refdes = [component.refdes for component, *_ in first]
        second_refdes = [component.refdes for component, *_ in second]
        self.assertEqual(first_refdes, second_refdes)
        self.assertEqual(set(first_refdes), {"C1", "C2", "C3"})
        self.assertEqual(len(first_refdes), len(set(first_refdes)))

    def test_component_roi_specs_include_catalog_metadata(self):
        source = MetadataSource(MetadataOrigin.USD, "board.usda")
        component = ComponentRecord(
            board_root="/Board",
            refdes="U20",
            prim_path="/Board/Components/U20",
            source=source,
            component_type="ic",
            package_name="USON-10",
            package_type="USON",
            footprint_id="USON-10_2.5x1mm_P0.5mm",
            side="top",
            process_class="SMT",
            process_classes=("SMT",),
        )
        catalog = BoardMetadataCatalog(
            boards=(BoardRecord("/Board", "Board", source),),
            components=(component,),
        )
        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            stage=mock.Mock(),
            scene={"board_root_path": "/Board"},
        )
        context.board_metadata["usd"] = catalog
        bounds = mock.Mock()
        bounds.IsEmpty.return_value = False
        bounds.GetMidpoint.return_value = (1.0, 2.0, 3.0)
        bounds.GetSize.return_value = (4.0, 5.0, 1.0)
        cache = mock.Mock()
        cache.ComputeWorldBound.return_value.ComputeAlignedRange.return_value = bounds
        pxr = types.SimpleNamespace(
            Usd=types.SimpleNamespace(
                TimeCode=types.SimpleNamespace(Default=mock.Mock(return_value=0)),
            ),
            UsdGeom=types.SimpleNamespace(BBoxCache=mock.Mock(return_value=cache)),
        )
        config = {
            "roi_selection": {
                "selections": (
                    {
                        "refdes": ("U20",),
                        "count": None,
                        "padding": 0.5,
                    },
                ),
            },
            "pcba_root": "/Board",
            "resolution": (100, 100),
            "random_seed": 0,
            "camera_rotation": {"z_fixed": 0.0},
            "camera_side": "top",
        }

        with mock.patch.dict(sys.modules, {"pxr": pxr}):
            specs = _roi_selection_specs(context, context.stage, config, 100.0)

        self.assertEqual(len(specs), 1)
        self.assertEqual(
            {key: specs[0][key] for key in _component_roi_metadata(component)},
            _component_roi_metadata(component),
        )

    def test_pin_roi_selection_resolves_each_physical_terminal(self):
        source = MetadataSource(MetadataOrigin.USD, "board.usda")
        component = ComponentRecord(
            board_root="/Board",
            refdes="T410",
            prim_path="/Board/Components/T410",
            source=source,
        )
        terminals = tuple(
            TerminalRecord(
                board_root="/Board",
                component_path=component.prim_path,
                prim_path=f"{component.prim_path}/{name}",
                source=source,
                number="6",
            )
            for name in ("pkg_lead_6", "pkg_lead_6_1")
        )
        catalog = BoardMetadataCatalog(
            boards=(BoardRecord("/Board", "Board", source),),
            components=(component,),
            terminals=terminals,
        )

        selected = _resolve_pin_roi_selection(
            IsaacExecutionContext(runtime=IsaacSdgRuntime()),
            catalog,
            "/Board",
            ({"pins": (("T410", "6"),), "padding": 0.25},),
            {component.prim_path: component},
            random_seed=0,
        )

        self.assertEqual(
            [terminal.prim_path for _component, terminal, *_rest in selected],
            [terminal.prim_path for terminal in terminals],
        )

    def test_pin_roi_selection_expands_component_filters(self):
        source = MetadataSource(MetadataOrigin.USD, "board.usda")
        component = ComponentRecord(
            board_root="/Board",
            refdes="K2010",
            prim_path="/Board/Components/K2010",
            source=source,
            component_type="ODD14901300.1000.7",
            process_class="Mixed",
            process_classes=("through_hole", "SMT"),
        )
        terminals = tuple(
            TerminalRecord(
                board_root="/Board",
                component_path=component.prim_path,
                prim_path=f"{component.prim_path}/pkg_lead_{number}",
                source=source,
                number=number,
                process_class=process_class,
            )
            for number, process_class in (
                ("1", "through_hole"),
                ("2", "through_hole"),
                ("3", "SMT"),
            )
        )
        catalog = BoardMetadataCatalog(
            boards=(BoardRecord("/Board", "Board", source),),
            components=(component,),
            terminals=terminals,
        )
        context = IsaacExecutionContext(runtime=IsaacSdgRuntime())
        components_by_path = {component.prim_path: component}

        selector_cases = (
            ({"refdes": ("K2010",), "focus": "pin"}, terminals),
            (
                {
                    "component_types": ("ODD14901300.1000.7",),
                    "focus": "pin",
                },
                terminals,
            ),
            (
                {"process_classes": ("through_hole",), "focus": "pin"},
                terminals[:2],
            ),
            ({"all_components": True, "focus": "pin"}, terminals),
        )
        for selector, expected in selector_cases:
            with self.subTest(selector=selector):
                selected = _resolve_pin_roi_selection(
                    context,
                    catalog,
                    "/Board",
                    (selector,),
                    components_by_path,
                    random_seed=0,
                )
                self.assertEqual(
                    [terminal.prim_path for _component, terminal, *_ in selected],
                    [terminal.prim_path for terminal in expected],
                )

    def test_nearest_prim_path_skips_bounds_for_unambiguous_candidate(self):
        stage = mock.Mock()
        cache = mock.Mock()

        selected = _nearest_prim_path(
            stage,
            cache,
            ("/Board/Layers/Bottom/Pad_1",),
            "/Board/Components/K2010/pkg_lead_1",
        )

        self.assertEqual(selected, "/Board/Layers/Bottom/Pad_1")
        stage.assert_not_called()
        cache.assert_not_called()

    def test_nearest_prim_path_uses_bounds_for_ambiguous_candidates(self):
        bounds_by_path = {
            "/Board/Components/T410/pkg_lead_6": (
                (10.0, 10.0, 0.0),
                (12.0, 12.0, 1.0),
            ),
            "/Board/Layers/Bottom/Pad_6786": (
                (10.0, 10.0, 0.0),
                (12.0, 12.0, 1.0),
            ),
            "/Board/Layers/Bottom/Pad_6789": (
                (20.0, 20.0, 0.0),
                (22.0, 22.0, 1.0),
            ),
        }

        with mock.patch(
            "cdf.execution.isaac.capture_execution._combined_prim_bounds",
            side_effect=lambda _stage, _cache, paths: bounds_by_path[paths[0]],
        ) as combined_bounds:
            selected = _nearest_prim_path(
                object(),
                object(),
                (
                    "/Board/Layers/Bottom/Pad_6786",
                    "/Board/Layers/Bottom/Pad_6789",
                ),
                "/Board/Components/T410/pkg_lead_6",
            )

        self.assertEqual(selected, "/Board/Layers/Bottom/Pad_6786")
        self.assertEqual(combined_bounds.call_count, 3)

    def test_reference_plane_prefers_the_measured_substrate_range(self):
        """The catalog measured the PCB; every bbox route measures the scene.

        A world-space AABB over the board's scope includes the tallest
        component and the longest protruding pin, so a plane derived from it is
        not the board surface it is presented as. When the metadata catalog
        carries a measured substrate_z_range, that wins.
        """
        from cdf.execution.isaac.capture_execution import (
            _capture_reference_plane_z,
        )

        class _Board:
            substrate_z_range = (1.0, 1.6)

        class _Metadata:
            board = _Board()

        class _Context:
            board_metadata = _Metadata()
            board_instance_cohort = None

        z_top, source_top = _capture_reference_plane_z(
            _Context(), object(), {}, top_side=True
        )
        self.assertEqual((z_top, source_top), (1.6, "substrate_z_range"))
        z_bot, source_bot = _capture_reference_plane_z(
            _Context(), object(), {}, top_side=False
        )
        self.assertEqual((z_bot, source_bot), (1.0, "substrate_z_range"))

    def test_reference_plane_prefers_board_instance_over_capture_root(self):
        """A carrier-scoped root must not leak fixture thickness into Z."""
        context = mock.Mock()
        context.board_instance_cohort = BoardInstanceCohort(
            instance_paths=("/World/Carrier/StepRepeats/SR_0_0",),
            representative_instance_path="/World/Carrier/StepRepeats/SR_0_0",
            prototype_path="/World/Carrier/StepRepeats/SR_0_0/Prototype",
            prototype_source_path="/Prototypes/Board",
        )
        # The instance (bare board) is thinner and sits above the carrier's
        # own underside -- the carrier bbox alone would report a lower
        # (wrong) bottom face than the actual board.
        instance_bounds = ((0.0, 0.0, -2.0), (100.0, 80.0, 0.0))
        pxr = types.ModuleType("pxr")
        pxr.Usd = mock.Mock()
        pxr.UsdGeom = mock.Mock()

        with (
            mock.patch.dict(sys.modules, {"pxr": pxr}),
            mock.patch(
                "cdf.execution.isaac.capture_execution._combined_prim_bounds",
                return_value=instance_bounds,
            ) as combined_bounds,
        ):
            top_z, top_source = _capture_reference_plane_z(
                context,
                stage=object(),
                scan_grid={
                    "bbox_min": [0.0, 0.0, -5.0],
                    "bbox_max": [100.0, 80.0, 5.0],
                },
                top_side=True,
            )
            bottom_z, bottom_source = _capture_reference_plane_z(
                context,
                stage=object(),
                scan_grid={
                    "bbox_min": [0.0, 0.0, -5.0],
                    "bbox_max": [100.0, 80.0, 5.0],
                },
                top_side=False,
            )

        self.assertEqual((top_z, top_source), (0.0, "board_instance"))
        self.assertEqual((bottom_z, bottom_source), (-2.0, "board_instance"))
        combined_bounds.assert_called_with(
            mock.ANY,
            mock.ANY,
            ("/World/Carrier/StepRepeats/SR_0_0",),
        )

    def test_reference_plane_falls_back_to_capture_root_without_cohort(self):
        context = mock.Mock()
        context.board_instance_cohort = None

        z, source = _capture_reference_plane_z(
            context,
            stage=object(),
            scan_grid={"bbox_min": [0.0, 0.0, -5.0], "bbox_max": [100.0, 80.0, 5.0]},
            top_side=False,
        )

        self.assertEqual((z, source), (-5.0, "capture_root"))

    def test_pin_roi_preflight_warns_before_duplicate_pin_geometry_scan(self):
        source = MetadataSource(MetadataOrigin.USD, "board.usda")
        component = ComponentRecord(
            board_root="/Board",
            refdes="T410",
            prim_path="/Board/Components/T410",
            source=source,
        )
        terminals = tuple(
            TerminalRecord(
                board_root="/Board",
                component_path=component.prim_path,
                prim_path=f"{component.prim_path}/{name}",
                source=source,
                number="6",
            )
            for name in ("pkg_lead_6", "pkg_lead_6_1")
        )
        pads = tuple(
            PadRecord(
                board_root="/Board",
                prim_path=f"/Board/Layers/Bottom/Pad_{number}",
                source=source,
                component_path=component.prim_path,
                terminal_number="6",
                contact_prim_path=terminals[1].prim_path,
                side="bottom",
            )
            for number in (6786, 6789)
        )
        catalog = BoardMetadataCatalog(
            boards=(BoardRecord("/Board", "Board", source),),
            components=(component,),
            terminals=terminals,
            pads=pads,
        )
        resolved = tuple(
            (component, terminal, 0, {"pins": (("T410", "6"),)})
            for terminal in terminals
        )

        messages = _pin_roi_preflight_warnings(catalog, resolved, "bottom")

        self.assertEqual(len(messages), 1)
        self.assertIn("pcba:pin:number '6'", messages[0])
        self.assertIn("world-space geometric scan", messages[0])

    def test_pin_roi_selection_anchors_bounds_to_associated_pad(self):
        source = MetadataSource(MetadataOrigin.USD, "board.usda")
        component = ComponentRecord(
            board_root="/Board",
            refdes="T410",
            prim_path="/Board/Components/T410",
            source=source,
            component_type="ic",
            package_name="TSSOP-16",
            package_type="TSSOP",
            footprint_id="TSSOP-16_4.4x5mm_P0.65mm",
            side="top",
            process_class="SMT",
            process_classes=("SMT",),
        )
        terminal = TerminalRecord(
            board_root="/Board",
            component_path=component.prim_path,
            prim_path=f"{component.prim_path}/pkg_lead_6",
            source=source,
            number="6",
        )
        catalog = BoardMetadataCatalog(
            boards=(BoardRecord("/Board", "Board", source),),
            components=(component,),
            terminals=(terminal,),
        )
        context = IsaacExecutionContext(runtime=IsaacSdgRuntime(), stage=object())
        context.state["solder_joints.placement_manifest"] = mock.Mock(
            generated_prim_paths=("/Board/Solder/T410_solder_002",),
        )
        pad = mock.Mock(prim_path="/Board/Layers/Bottom/Pad_6786")
        pad_bounds = ((10.0, 20.0, 0.0), (12.0, 22.0, 1.0))

        with (
            mock.patch(
                "cdf.execution.isaac.capture_execution._terminal_pads",
                return_value=(pad,),
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._nearest_prim_path",
                return_value=pad.prim_path,
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._nearest_generated_solder_path",
                return_value="/Board/Solder/T410_solder_002",
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._pin_roi_preflight_warnings",
                return_value=("geometric scan required",),
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution.warnings.warn",
            ) as emit_warning,
            mock.patch(
                "cdf.execution.isaac.capture_execution._combined_prim_bounds",
                return_value=pad_bounds,
            ) as combined_bounds,
        ):
            specs = _pin_roi_selection_specs(
                context,
                context.stage,
                catalog,
                "/Board",
                ({"pins": (("T410", "6"),), "padding": 0.5},),
                {component.prim_path: component},
                object(),
                100.0,
                aspect=1.0,
                camera_side="bottom",
                camera_z_rotation=0.0,
                random_seed=0,
            )

        self.assertEqual(len(specs), 1)
        self.assertEqual(
            {
                key: specs[0][key]
                for key in (
                    "component_type",
                    "package_name",
                    "package_type",
                    "footprint_id",
                    "side",
                    "process_class",
                    "process_classes",
                )
            },
            _component_roi_metadata(component),
        )
        self.assertEqual(specs[0]["pad_paths"], (pad.prim_path,))
        self.assertEqual(
            specs[0]["solder_joint_path"],
            "/Board/Solder/T410_solder_002",
        )
        emit_warning.assert_called_once_with(
            "geometric scan required",
            RuntimeWarning,
            stacklevel=2,
        )
        self.assertEqual(
            context.state["capture.roi_selection.warnings"],
            ("geometric scan required",),
        )
        self.assertEqual(
            [call.args[2] for call in combined_bounds.call_args_list],
            [(pad.prim_path,), (pad.prim_path,)],
        )

    def test_roi_apertures_follow_camera_z_rotation(self):
        unswapped = _roi_apertures(
            23.3,
            7.5,
            padding=0.5,
            aspect=1080 / 1920,
            camera_z_rotation=0.0,
        )
        swapped = _roi_apertures(
            23.3,
            7.5,
            padding=0.5,
            aspect=1080 / 1920,
            camera_z_rotation=-90.0,
        )

        self.assertAlmostEqual(unswapped[0], 34.95)
        self.assertAlmostEqual(unswapped[1], 19.659375)
        self.assertAlmostEqual(swapped[0], 62.13333333333333)
        self.assertAlmostEqual(swapped[1], 34.95)

    def test_tilted_roi_camera_looks_through_its_target(self):
        # Regression: camera.rotation x: 45 at z = 27.8 mm looked ~26 mm past each part.
        target = (28.115, 110.535, 1.6)
        tilt = math.radians(45.0)
        view = (0.0, math.sin(tilt), -math.cos(tilt))
        camera = _aim_along_view((28.115, 110.535, 27.79), target, view)

        self.assertAlmostEqual(camera[2], 27.79)
        distance = (target[2] - camera[2]) / view[2]
        hit = [camera[axis] + distance * view[axis] for axis in range(3)]
        for axis in range(3):
            self.assertAlmostEqual(hit[axis], target[axis])
        self.assertAlmostEqual(camera[1], 110.535 - (27.79 - 1.6))

    def test_untilted_roi_camera_is_left_exactly_where_it_was(self):
        position = (28.115, 110.535, 27.79)
        self.assertEqual(
            _aim_along_view(position, (1.0, 2.0, 1.6), (0.0, 0.0, -1.0)),
            position,
        )
        # A bottom-side camera (x + 180) carries float dust in its direction.
        self.assertEqual(
            _aim_along_view(position, (1.0, 2.0, 1.6), (0.0, 1.2e-16, 1.0)),
            position,
        )

    def test_roi_camera_that_cannot_see_the_board_is_refused(self):
        with self.assertRaisesRegex(ValidationError, "parallel"):
            _aim_along_view((0.0, 0.0, 27.8), (0.0, 0.0, 1.6), (1.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValidationError, "looks away"):
            _aim_along_view((0.0, 0.0, 27.8), (0.0, 0.0, 1.6), (0.0, 0.5, 0.8))

    def test_roi_trigger_rotation_is_fixed_exact_and_ranged_reproducible(self):
        fixed = {"x_range": (45.0, 45.0), "y_range": (0.0, 0.0), "z_fixed": -90.0}
        self.assertEqual(_trigger_camera_rotation(fixed, 7, 0), (45.0, 0.0, -90.0))

        ranged = {"x_range": (-10.0, 10.0), "y_range": (0.0, 5.0), "z_fixed": 0.0}
        first = _trigger_camera_rotation(ranged, 7, 0)
        self.assertEqual(first, _trigger_camera_rotation(ranged, 7, 0))
        self.assertNotEqual(first, _trigger_camera_rotation(ranged, 7, 1))
        self.assertTrue(-10.0 <= first[0] <= 10.0 and 0.0 <= first[1] <= 5.0)

    def test_roi_apertures_support_oblique_camera_z_rotation(self):
        width, height = _roi_apertures(
            4.0,
            2.0,
            padding=0.0,
            aspect=1.0,
            camera_z_rotation=45.0,
        )

        expected = 3.0 * 2.0**0.5
        self.assertAlmostEqual(width, expected)
        self.assertAlmostEqual(height, expected)

    def test_concrete_runtime_delegates_capture_to_extracted_lifecycle(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        expected = {"output_dir": "output", "frame_count": 1}

        with mock.patch(
            "cdf.execution.isaac.capture_execution.execute_capture",
            new=mock.AsyncMock(return_value=expected),
        ) as execute_capture:
            output = asyncio.run(
                runtime.render_capture(
                    context,
                    CaptureRenderParameters(output_dir="output"),
                ),
            )

        self.assertEqual(output, expected)
        execute_capture.assert_awaited_once()

    def test_remote_capture_and_crop_share_local_output_tree(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        adapter = mock.Mock()
        remote_output = "s3://bucket/runs/job-1"

        with tempfile.TemporaryDirectory() as temporary_directory:
            context.output_tree = mock.Mock(return_value=temporary_directory)
            context.local_output_tree = mock.Mock(return_value=temporary_directory)
            with mock.patch(
                "cdf.execution.isaac.capture_execution.execute_capture",
                new=mock.AsyncMock(
                    return_value={
                        "output_dir": temporary_directory,
                        "pipeline_type": "defect",
                        "trigger_directories": [
                            str(Path(temporary_directory) / "trigger_0000"),
                        ],
                    },
                ),
            ) as execute_capture:
                capture_output = asyncio.run(
                    runtime.render_capture(
                        context,
                        CaptureRenderParameters(output_dir=remote_output),
                    ),
                )

            context.outputs["capture.render"] = capture_output
            with mock.patch(
                "cdf.execution.isaac.job_order_runtime.importlib.import_module",
                return_value=adapter,
            ):
                crop_output = runtime.crop_capture(
                    context,
                    PostprocessCropParameters(
                        method="bbox_tight",
                        types=("rgb", "semantic_segmentation"),
                        offset=32,
                        min_coverage=0.2,
                        xform_depth=7,
                        output_subdir="cropped",
                    ),
                )

        local_parameters = execute_capture.await_args.args[1]
        self.assertEqual(local_parameters.output_dir, temporary_directory)
        self.assertEqual(capture_output["output_dir"], remote_output)
        self.assertEqual(
            capture_output["trigger_directories"],
            [remote_output + "/trigger_0000"],
        )
        crop_config = adapter.run_pipeline_crop.call_args.args[0]
        self.assertEqual(crop_config["output"], temporary_directory)
        self.assertEqual(crop_output["output_dir"], remote_output + "/cropped")

    def test_joint_rois_receive_selected_catalog_component_metadata(self):
        """Pass component classifications from the catalog to ROI generation."""
        source = MetadataSource(MetadataOrigin.USD, "board.usda")
        catalog = BoardMetadataCatalog(
            boards=(BoardRecord("/Board", "Board", source),),
            components=(
                ComponentRecord(
                    board_root="/Board",
                    refdes="R1",
                    prim_path="/Board/Components/R1",
                    source=source,
                    component_type="RESISTOR",
                    package_type="THT_AXIAL",
                    package_name="AXIAL_RESISTOR",
                    footprint_id="R_AXIAL_10MM",
                    process_class="through_hole",
                ),
            ),
        )
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        context.board_metadata["usd"] = catalog
        context.outputs["capture.full_board.render"] = {
            "output_dir": "rgb-output",
            "trigger_metadata": [],
        }
        context.outputs["capture.full_board_roi.render"] = {
            "output_dir": "roi-output",
            "trigger_metadata": [],
        }

        expected = {"joint_count": 0, "artifacts": [], "executed": True}
        with mock.patch(
            "pcb_aoi.postprocess.joint_rois.generate_joint_rois",
            return_value=expected,
        ) as generate:
            result = runtime.generate_joint_rois(
                context,
                PostprocessGenerateJointRoisParameters(),
            )

        self.assertEqual(result, expected)
        self.assertEqual(
            generate.call_args.kwargs["output_roots_by_pass"],
            {
                "full_board": "rgb-output",
                "full_board_roi": "roi-output",
            },
        )
        self.assertEqual(
            generate.call_args.kwargs["component_metadata"]["R1"],
            {
                "component_type": "RESISTOR",
                "component_type_source": "component_type",
                "package_type": "THT_AXIAL",
                "package_name": "AXIAL_RESISTOR",
                "footprint_id": "R_AXIAL_10MM",
                "process_class": "through_hole",
            },
        )

    def test_joint_rois_allow_missing_board_metadata(self):
        """Preserve ROI generation when no board metadata catalog is loaded."""
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        context.outputs["capture.full_board.render"] = {
            "output_dir": "rgb-output",
            "trigger_metadata": [],
        }
        context.outputs["capture.full_board_roi.render"] = {
            "output_dir": "roi-output",
            "trigger_metadata": [],
        }

        expected = {"joint_count": 0, "artifacts": [], "executed": True}
        with mock.patch(
            "pcb_aoi.postprocess.joint_rois.generate_joint_rois",
            return_value=expected,
        ) as generate:
            result = runtime.generate_joint_rois(
                context,
                PostprocessGenerateJointRoisParameters(),
            )

        self.assertEqual(result, expected)
        self.assertEqual(generate.call_args.kwargs["component_metadata"], {})
        self.assertEqual(generate.call_args.kwargs["pad_joint_keys"], {})

    def test_joint_rois_receive_pad_associations_for_every_placement(self):
        """Re-root stated pad associations from the prototype to each placement."""
        source = MetadataSource(MetadataOrigin.USD, "panel.usda")
        board = "/World/_class_Board"
        catalog = BoardMetadataCatalog(
            boards=(BoardRecord(board, "Board", source),),
            components=(
                ComponentRecord(
                    board_root=board,
                    refdes="J1",
                    prim_path=f"{board}/Components/J1",
                    source=source,
                ),
            ),
            pads=(
                PadRecord(
                    board_root=board,
                    prim_path=f"{board}/Layers/Bottom/Pad_3",
                    source=source,
                    component_path=f"{board}/Components/J1",
                    terminal_number="2",
                    contact_prim_path=f"{board}/Components/J1/pkg_lead_2",
                ),
                PadRecord(
                    board_root=board,
                    prim_path=f"{board}/Layers/Bottom/Pad_4",
                    source=source,
                ),
            ),
        )
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        context.board_metadata["usd"] = catalog
        context.board_instance_cohort = BoardInstanceCohort(
            instance_paths=("/World/Panel/StepRepeats/A", "/World/Panel/StepRepeats/B"),
            representative_instance_path="/World/Panel/StepRepeats/A",
            prototype_path="/__Prototype_1",
            prototype_source_path=board,
        )
        context.outputs["capture.full_board.render"] = {
            "output_dir": "rgb-output",
            "trigger_metadata": [],
        }
        context.outputs["capture.full_board_roi.render"] = {
            "output_dir": "roi-output",
            "trigger_metadata": [],
        }

        expected = {
            "joint_count": 2,
            "pin_only_joint_count": 1,
            "artifacts": [],
            "executed": True,
        }
        with (
            mock.patch(
                "pcb_aoi.postprocess.joint_rois.generate_joint_rois",
                return_value=expected,
            ) as generate,
            self.assertWarnsRegex(RuntimeWarning, "1 of 2 joint"),
        ):
            runtime.generate_joint_rois(
                context,
                PostprocessGenerateJointRoisParameters(),
            )

        self.assertEqual(
            generate.call_args.kwargs["pad_joint_keys"],
            {
                "/World/Panel/StepRepeats/A/Layers/Bottom/Pad_3": ("J1", "2"),
                "/World/Panel/StepRepeats/B/Layers/Bottom/Pad_3": ("J1", "2"),
            },
        )

    def test_joint_rois_register_artifacts_with_each_capture_pass(self):
        """Register each copied manifest with the capture pass containing it."""
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        context.outputs["capture.full_board.render"] = {
            "output_dir": "rgb-output",
            "trigger_metadata": [],
        }
        context.outputs["capture.full_board_roi.render"] = {
            "output_dir": "roi-output",
            "trigger_metadata": [],
        }
        artifacts = [
            {
                "capture_pass": "full_board",
                "kind": "joint_rois",
                "path": "trigger_0000/joint_rois.json",
                "trigger_idx": 0,
            },
            {
                "capture_pass": "full_board_roi",
                "kind": "joint_rois_stitched",
                "path": "stitched/joint_rois.json",
                "trigger_idx": 0,
            },
        ]
        expected = {"joint_count": 1, "artifacts": artifacts, "executed": True}

        with (
            mock.patch(
                "pcb_aoi.postprocess.joint_rois.generate_joint_rois",
                return_value=expected,
            ),
            mock.patch.object(
                context.manifest,
                "record_derived_artifacts",
            ) as record_artifacts,
        ):
            result = runtime.generate_joint_rois(
                context,
                PostprocessGenerateJointRoisParameters(),
            )

        self.assertEqual(result, expected)
        self.assertEqual(
            record_artifacts.call_args_list,
            [
                mock.call(
                    "capture.full_board.render",
                    [
                        {
                            "kind": "joint_rois",
                            "path": "trigger_0000/joint_rois.json",
                            "trigger_idx": 0,
                        }
                    ],
                ),
                mock.call(
                    "capture.full_board_roi.render",
                    [
                        {
                            "kind": "joint_rois_stitched",
                            "path": "stitched/joint_rois.json",
                            "trigger_idx": 0,
                        }
                    ],
                ),
            ],
        )

    def test_capture_lifecycle_writes_one_frame_and_cleans_up(self):
        class _Camera:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            stage=object(),
            scene={"root_path": "/Board"},
            component_types=("PASSIVE_SMD_0805",),
            plan_metadata={"random_seed": 7},
        )
        context.state.update(
            {
                "capture.renderer": {
                    "mode": "pathtracing",
                    "resolution": (64, 64),
                    "spp": 1,
                    "total_spp": 2,
                    "max_bounces": 1,
                },
                "capture.camera": {
                    "rotation_range": {"x": (0, 0), "y": (0, 0), "z": -90},
                },
                "capture.scan_grid": {"x_count": 1, "y_count": 1},
                "capture.outputs": {
                    "rgb": True,
                    "image_format": "png",
                    "frame_padding": 4,
                    "rename_to_grid_index": True,
                },
                "capture.limits": {"triggers": 1, "max_images": 1},
                "capture.output": {"start": 0},
                "operation.structural.records": [
                    {
                        "path": "/Board/Components/R1",
                        "defect": "tombstone",
                    },
                ],
            },
        )

        render_product = mock.Mock(path="render-product")
        writer = mock.Mock()
        replicator = types.ModuleType("omni.replicator.core")
        replicator.create = mock.Mock()
        replicator.create.render_product.return_value = render_product
        replicator.WriterRegistry = mock.Mock()
        replicator.WriterRegistry.get.return_value = writer
        replicator.get = mock.Mock()
        replicator.get.prim_at_path.return_value = _Camera()
        replicator.modify = mock.Mock()
        replicator.distribution = mock.Mock()
        replicator.distribution.uniform.return_value = "rotation"
        replicator.orchestrator = mock.Mock()
        replicator.orchestrator.step_async = mock.AsyncMock()
        replicator.orchestrator.wait_until_complete_async = mock.AsyncMock()
        # As after the warm-up: attaching now arms the writer for one step.
        replicator.orchestrator.get_status.return_value = (
            replicator.orchestrator.Status.STEPPED
        )

        omni = types.ModuleType("omni")
        omni_replicator = types.ModuleType("omni.replicator")
        pxr = types.ModuleType("pxr")
        pxr.Gf = mock.Mock()
        pxr.Gf.Vec3d.side_effect = lambda x, y, z: (x, y, z)

        translate_op = mock.Mock()
        common = mock.Mock()
        common.configure_pathtracing.return_value = 2
        common.build_scan_positions.return_value = [(1.0, 2.0, 3.0)]
        common.find_translate_op.return_value = translate_op
        common.randomize_lighting.return_value = {"ring_light": False}
        common.setup_augmentation.return_value = {"applied": False}
        common.scan_positions_for_trigger.side_effect = (
            lambda positions, trigger, seed: positions
        )
        common.cap_positions_for_max_outputs.side_effect = (
            lambda positions, pipeline, maximum, cursor: positions[: maximum - cursor]
        )
        common.grid_index_for_position.return_value = (0, 0)
        defect_ops = mock.Mock()
        scan_grid_geometry = mock.Mock()
        expected_pixel_layout = {
            "supported": True,
            "canvas_size_px": [64, 64],
        }
        scan_grid_geometry.compute_pixel_layout.return_value = expected_pixel_layout
        _write_frame_on_step(replicator, common)

        warmup = mock.AsyncMock()
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.dict(
                sys.modules,
                {
                    "omni": omni,
                    "omni.replicator": omni_replicator,
                    "omni.replicator.core": replicator,
                    "pxr": pxr,
                },
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._script_module",
                side_effect=lambda directory, name: {
                    "common": common,
                    "defect_ops": defect_ops,
                }[name],
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution.scan_grid_geometry",
                scan_grid_geometry,
            ),
            mock.patch("cdf.execution.isaac.capture_execution._ensure_camera"),
            mock.patch("cdf.execution.isaac.capture_execution._setup_lighting"),
            mock.patch(
                "cdf.execution.isaac.capture_execution._setup_component_appearance",
                return_value={"applied": False},
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._warmup",
                new=warmup,
            ),
        ):
            output = asyncio.run(
                execute_capture(
                    context,
                    CaptureRenderParameters(output_dir=temporary_directory),
                ),
            )

        self.assertEqual(output["frame_count"], 1)
        self.assertEqual(output["pipeline_type"], "defect")
        writer.attach.assert_called_once_with([render_product], trigger=None)
        self.assertNotIn("rename_to_grid_index", writer.initialize.call_args.kwargs)
        self.assertTrue(writer.initialize.call_args.kwargs["bounding_box_2d_tight"])
        writer.detach.assert_called_once_with()
        render_product.destroy.assert_called_once_with()
        self.assertEqual(
            translate_op.Set.call_args_list,
            # Pass warm-up, then cell 0 applied and settled before attach; the
            # armed writer writes cell 0 without moving again.
            [mock.call((1.0, 2.0, 3.0))] * 2,
        )
        self.assertEqual(warmup.await_count, 2)
        defect_ops.reapply_defects.assert_called_once()
        defect_ops.restore_defects.assert_called_once()
        common.save_metadata.assert_not_called()
        metadata = output["trigger_metadata"][0]
        self.assertEqual(metadata["schema_name"], "cdf.dataset_manifest")
        expected_frame_to_grid = [{"frame": 0, "x_idx": 0, "y_idx": 0}]
        self.assertEqual(metadata["frame_to_grid"], expected_frame_to_grid)
        self.assertEqual(metadata["pixel_layout"], expected_pixel_layout)
        self.assertEqual(output["trigger_metadata"], [metadata])
        self.assertEqual(
            output["pixel_layouts"],
            [{"trigger_idx": 0, **expected_pixel_layout}],
        )
        scan_grid_geometry.compute_pixel_layout.assert_called_once_with(
            {"x_num": 1, "y_num": 1},
            (64, 64),
            {"x_range": (0.0, 0.0), "y_range": (0.0, 0.0), "z_fixed": -90.0},
            [{"frame": 0, "x_idx": 0, "y_idx": 0}],
            camera_side="top",
        )
        common.rename_outputs_to_grid_index.assert_called_once_with(
            writer.initialize.call_args.kwargs["output_dir"],
            [(0, 0, 0)],
            frame_padding=4,
        )

    def test_scan_grid_image_size_reaches_the_pixel_layout(self):
        """``scan_grid.image_size_px`` is what the stitch crops to; dropping it
        between the job order and the pixel layout would silently skip the crop."""

        class _Camera:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            stage=object(),
            scene={"root_path": "/Board"},
            component_types=("PASSIVE_SMD_0805",),
            plan_metadata={"random_seed": 7},
        )
        context.state.update(
            {
                "capture.renderer": {
                    "mode": "pathtracing",
                    "resolution": (64, 64),
                    "spp": 1,
                    "total_spp": 2,
                    "max_bounces": 1,
                },
                "capture.camera": {
                    "rotation_range": {"x": (0, 0), "y": (0, 0), "z": 0},
                },
                "capture.scan_grid": {
                    "x_count": 1,
                    "y_count": 1,
                    "image_size_px": [60, 50],
                },
                "capture.outputs": {
                    "rgb": True,
                    "image_format": "png",
                    "frame_padding": 4,
                },
                "capture.limits": {"triggers": 1, "max_images": 1},
                "capture.output": {"start": 0},
            },
        )
        self.assertEqual(
            build_capture_config(context, "output")["scan_image_size_px"], [60, 50]
        )

        render_product = mock.Mock(path="render-product")
        writer = mock.Mock()
        replicator = types.ModuleType("omni.replicator.core")
        replicator.create = mock.Mock()
        replicator.create.render_product.return_value = render_product
        replicator.WriterRegistry = mock.Mock()
        replicator.WriterRegistry.get.return_value = writer
        replicator.get = mock.Mock()
        replicator.get.prim_at_path.return_value = _Camera()
        replicator.modify = mock.Mock()
        replicator.distribution = mock.Mock()
        replicator.distribution.uniform.return_value = "rotation"
        replicator.orchestrator = mock.Mock()
        replicator.orchestrator.step_async = mock.AsyncMock()
        replicator.orchestrator.wait_until_complete_async = mock.AsyncMock()

        omni = types.ModuleType("omni")
        omni_replicator = types.ModuleType("omni.replicator")
        pxr = types.ModuleType("pxr")
        pxr.Gf = mock.Mock()
        pxr.Gf.Vec3d.side_effect = lambda x, y, z: (x, y, z)

        common = mock.Mock()
        common.configure_pathtracing.return_value = 2
        common.build_scan_positions.return_value = [(1.0, 2.0, 3.0)]
        common.find_translate_op.return_value = mock.Mock()
        common.randomize_lighting.return_value = {"ring_light": False}
        common.setup_augmentation.return_value = {"applied": False}
        common.scan_positions_for_trigger.side_effect = (
            lambda positions, trigger, seed: positions
        )
        common.cap_positions_for_max_outputs.side_effect = (
            lambda positions, pipeline, maximum, cursor: positions[: maximum - cursor]
        )
        common.grid_index_for_position.return_value = (0, 0)
        scan_grid_geometry = mock.Mock()
        scan_grid_geometry.compute_pixel_layout.return_value = {
            "supported": True,
            "canvas_size_px": [64, 64],
        }
        _write_frame_on_step(replicator, common)

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.dict(
                sys.modules,
                {
                    "omni": omni,
                    "omni.replicator": omni_replicator,
                    "omni.replicator.core": replicator,
                    "pxr": pxr,
                },
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._script_module",
                side_effect=lambda directory, name: {
                    "common": common,
                    "defect_ops": mock.Mock(),
                }[name],
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution.scan_grid_geometry",
                scan_grid_geometry,
            ),
            mock.patch("cdf.execution.isaac.capture_execution._ensure_camera"),
            mock.patch("cdf.execution.isaac.capture_execution._setup_lighting"),
            mock.patch(
                "cdf.execution.isaac.capture_execution._setup_component_appearance",
                return_value={"applied": False},
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._warmup",
                new=mock.AsyncMock(),
            ),
        ):
            output = asyncio.run(
                execute_capture(
                    context,
                    CaptureRenderParameters(output_dir=temporary_directory),
                ),
            )

        layout = output["trigger_metadata"][0]["pixel_layout"]
        self.assertEqual(layout["image_size_px"], [60, 50])
        self.assertEqual(output["pixel_layouts"][0]["image_size_px"], [60, 50])

    def test_missing_capture_writes_aligned_reference_and_defective_passes(self):
        class _Camera:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        stage = object()
        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            stage=stage,
            scene={"root_path": "/Board"},
            component_types=("PASSIVE_SMD_0805",),
            plan_metadata={"random_seed": 7},
        )
        context.state.update(
            {
                "capture.renderer": {
                    "mode": "pathtracing",
                    "resolution": (64, 64),
                    "spp": 1,
                    "total_spp": 2,
                    "max_bounces": 1,
                },
                "capture.camera": {
                    "rotation_range": {"x": (0, 0), "y": (0, 0), "z": -90},
                },
                "capture.scan_grid": {"x_count": 1, "y_count": 1},
                "capture.outputs": {
                    "rgb": True,
                    "image_format": "png",
                    "frame_padding": 4,
                },
                "capture.limits": {"triggers": 1, "max_images": 2},
                "capture.output": {"start": 0},
                "operation.missing.paths": ["/Board/Components/R1"],
            },
        )

        render_product = mock.Mock(path="render-product")
        reference_writer = mock.Mock()
        defective_writer = mock.Mock()
        replicator = types.ModuleType("omni.replicator.core")
        replicator.create = mock.Mock()
        replicator.create.render_product.return_value = render_product
        replicator.WriterRegistry = mock.Mock()
        replicator.WriterRegistry.get.side_effect = [
            reference_writer,
            defective_writer,
        ]
        replicator.get = mock.Mock()
        replicator.get.prim_at_path.return_value = _Camera()
        replicator.modify = mock.Mock()
        replicator.distribution = mock.Mock()
        replicator.distribution.uniform.return_value = "rotation"
        replicator.orchestrator = mock.Mock()
        replicator.orchestrator.step_async = mock.AsyncMock()
        replicator.orchestrator.wait_until_complete_async = mock.AsyncMock()

        omni = types.ModuleType("omni")
        omni_replicator = types.ModuleType("omni.replicator")
        pxr = types.ModuleType("pxr")
        pxr.Gf = mock.Mock()
        pxr.Gf.Vec3d.side_effect = lambda x, y, z: (x, y, z)

        common = mock.Mock()
        common.configure_pathtracing.return_value = 2
        common.build_scan_positions.return_value = [(1.0, 2.0, 3.0)]
        common.find_translate_op.return_value = mock.Mock()
        common.randomize_lighting.return_value = {"ring_light": False}
        common.setup_augmentation.return_value = {"applied": False}
        common.scan_positions_for_trigger.side_effect = (
            lambda positions, trigger, seed: positions
        )
        common.cap_positions_for_max_outputs.side_effect = (
            lambda positions, pipeline, maximum, cursor: positions[:1]
        )
        common.grid_index_for_position.return_value = (0, 0)
        defect_ops = mock.Mock()
        missing_ops = mock.Mock()
        scan_grid_geometry = mock.Mock()
        expected_pixel_layout = {
            "supported": True,
            "canvas_size_px": [64, 64],
        }
        scan_grid_geometry.compute_pixel_layout.return_value = expected_pixel_layout
        _write_frame_on_step(replicator, common)

        warmup = mock.AsyncMock()
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.dict(
                sys.modules,
                {
                    "omni": omni,
                    "omni.replicator": omni_replicator,
                    "omni.replicator.core": replicator,
                    "pxr": pxr,
                },
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._script_module",
                side_effect=lambda directory, name: {
                    "common": common,
                    "defect_ops": defect_ops,
                    "missing_ops": missing_ops,
                }[name],
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution.scan_grid_geometry",
                scan_grid_geometry,
            ),
            mock.patch("cdf.execution.isaac.capture_execution._ensure_camera"),
            mock.patch("cdf.execution.isaac.capture_execution._setup_lighting"),
            mock.patch(
                "cdf.execution.isaac.capture_execution._setup_component_appearance",
                return_value={"applied": False},
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._warmup",
                new=warmup,
            ),
        ):
            output = asyncio.run(
                execute_capture(
                    context,
                    CaptureRenderParameters(output_dir=temporary_directory),
                ),
            )

        self.assertEqual(output["pipeline_type"], "missing")
        self.assertEqual(output["frame_count"], 2)
        self.assertEqual(
            [frame["role"] for frame in output["frames"]],
            ["reference", "defective"],
        )
        self.assertEqual(
            [
                call.args[1]
                for call in common.set_writer_start_frame_index.call_args_list
            ],
            [0, 0],
        )
        missing_ops.hide_components.assert_called_once_with(
            stage,
            ["/Board/Components/R1"],
        )
        self.assertEqual(missing_ops.restore_components.call_count, 2)
        reference_dir = reference_writer.initialize.call_args.kwargs["output_dir"]
        defective_dir = defective_writer.initialize.call_args.kwargs["output_dir"]
        self.assertEqual(Path(reference_dir).name, "reference")
        self.assertEqual(Path(defective_dir).name, "defective")
        self.assertEqual(warmup.await_count, 3)
        common.save_metadata.assert_not_called()
        metadata = output["trigger_metadata"][0]
        self.assertEqual(metadata["schema_version"], "1.0.0")
        self.assertEqual(metadata["pixel_layout"], expected_pixel_layout)
        self.assertEqual(output["trigger_metadata"], [metadata])
        self.assertEqual(
            output["pixel_layouts"],
            [{"trigger_idx": 0, **expected_pixel_layout}],
        )
        common.rename_outputs_to_grid_index.assert_not_called()

    def test_positions_within_image_budget_counts_image_sets(self):
        positions = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)]
        self.assertEqual(_positions_within_image_budget(positions, -1, 5), positions)
        # max_images=1 is one image set even though the missing pipeline
        # renders it twice (reference + defective).
        self.assertEqual(_positions_within_image_budget(positions, 1, 0), positions[:1])
        self.assertEqual(_positions_within_image_budget(positions, 2, 1), positions[:1])
        self.assertEqual(_positions_within_image_budget(positions, 1, 1), [])
        self.assertEqual(_positions_within_image_budget(positions, 0, 0), [])

    @staticmethod
    def _warmup_replicator(frames):
        annotator = mock.Mock()
        annotator.get_data.side_effect = frames
        replicator = mock.Mock()
        replicator.AnnotatorRegistry.get_annotator.return_value = annotator
        replicator.orchestrator.step_async = mock.AsyncMock()
        return replicator, annotator

    def test_warmup_steps_until_renderer_delivers_a_frame(self):
        empty = types.SimpleNamespace(size=0)
        rendered = types.SimpleNamespace(size=512 * 512 * 4)
        # A cold shader cache: steps return empty data until the pipeline has
        # compiled, well past the minimum step count.
        replicator, annotator = self._warmup_replicator([empty] * 8 + [rendered])
        render_product = types.SimpleNamespace(path="/Render/RP")

        asyncio.run(_warmup(render_product, replicator))

        self.assertEqual(replicator.orchestrator.step_async.await_count, 9)
        annotator.detach.assert_called_once_with("/Render/RP")

    def test_warmup_keeps_minimum_steps_when_renderer_is_ready(self):
        rendered = types.SimpleNamespace(size=512 * 512 * 4)
        replicator, _annotator = self._warmup_replicator([rendered] * 5)

        asyncio.run(_warmup(types.SimpleNamespace(path="/Render/RP"), replicator))

        self.assertEqual(replicator.orchestrator.step_async.await_count, 5)

    def test_each_cell_is_settled_before_its_one_written_step(self):
        events = []
        replicator = mock.Mock()

        async def _step(**kwargs):
            events.append(("step", kwargs["rt_subframes"]))

        replicator.orchestrator.step_async = mock.AsyncMock(side_effect=_step)
        writer = mock.Mock()
        writer.schedule_write.side_effect = lambda: events.append(("schedule_write",))
        timer = mock.Mock()
        timer.phase.side_effect = lambda name: contextlib.nullcontext()

        for cell in (0, 1, 2):
            asyncio.run(
                _write_settled_frame(
                    replicator,
                    writer,
                    move=lambda cell=cell: events.append(("move", cell)),
                    reseed=lambda cell=cell: events.append(("seed", cell)),
                    rt_subframes=4,
                    timer=timer,
                ),
            )

        expected = []
        for cell in (0, 1, 2):
            expected += [
                ("move", cell),
                ("seed", cell),
                ("step", 4),
                ("seed", cell),
                ("schedule_write",),
                ("step", 4),
            ]
        self.assertEqual(events, expected)
        self.assertEqual(writer.schedule_write.call_count, 3)

    @staticmethod
    def _replicator_writer(status):
        """Fakes that follow Replicator 1.13.36's write rules for a manual writer.

        Nothing is written before attach. After it, a step is written when
        ``schedule_write()`` preceded it, or -- once -- when the writer was
        attached while the orchestrator was ``STARTED`` or ``STEPPED``
        (``OgnScheduleWriterInternalState.is_triggered``).
        """
        status_enum = types.SimpleNamespace(STARTED="started", STEPPED="stepped")
        state = {"pending": False, "attached": False, "writes": [], "events": []}
        replicator = mock.Mock()
        replicator.orchestrator.Status = status_enum
        replicator.orchestrator.get_status.return_value = status

        async def _step(**_kwargs):
            state["events"].append(("step", state["attached"]))
            if state["attached"] and state["pending"]:
                state["writes"].append(state["pose"])
                state["pending"] = False

        def _attach():
            state["attached"] = True
            state["pending"] = status in ("started", "stepped")

        replicator.orchestrator.step_async = mock.AsyncMock(side_effect=_step)
        writer = mock.Mock()
        writer.schedule_write.side_effect = lambda: state.update(pending=True)
        timer = mock.Mock()
        timer.phase.side_effect = lambda name: contextlib.nullcontext()
        return replicator, writer, timer, state, _attach

    def _run_cells(self, status, cells):
        """Drive one pass the way execute_capture does."""
        replicator, writer, timer, state, attach = self._replicator_writer(status)
        state["pose"] = "warm-up"

        def move(cell):
            state["events"].append(("move", cell))
            state["pose"] = cell

        # Cell 0 applied and settled before the writer exists.
        move(cells[0])
        asyncio.run(replicator.orchestrator.step_async(rt_subframes=4))
        armed = _writer_armed_at_attach(replicator)
        attach()
        for index, cell in enumerate(cells):
            asyncio.run(
                _write_settled_frame(
                    replicator,
                    writer,
                    move=lambda cell=cell: move(cell),
                    reseed=lambda: None,
                    rt_subframes=4,
                    timer=timer,
                    armed=armed and index == 0,
                ),
            )
        return state

    def test_one_write_per_cell_when_attach_arms_the_writer(self):
        cells = ["c0", "c1", "c2", "c3", "c4"]
        state = self._run_cells("stepped", cells)
        self.assertEqual(state["writes"], cells)
        # Cell 0 is settled by a step taken before the writer was attached.
        self.assertEqual(state["events"][:2], [("move", "c0"), ("step", False)])
        self.assertEqual(state["events"][2], ("step", True))

    def test_one_write_per_cell_when_attach_does_not_arm_the_writer(self):
        cells = ["c0", "c1", "c2"]
        self.assertEqual(self._run_cells("stopped", cells)["writes"], cells)

    def test_writer_armed_at_attach_follows_the_orchestrator_status(self):
        for status, armed in (("started", True), ("stepped", True), ("stopped", False)):
            with self.subTest(status=status):
                replicator, *_fakes = self._replicator_writer(status)
                self.assertEqual(_writer_armed_at_attach(replicator), armed)

    def test_a_write_count_that_is_not_one_per_cell_fails_the_pass(self):
        with self.assertRaisesRegex(RuntimeError, "21 frame.*20 camera position"):
            _require_one_write_per_cell(21, 20, "/out/full_board")
        _require_one_write_per_cell(20, 20, "/out/full_board")
        _require_one_write_per_cell(mock.Mock(), 20, "/out/full_board")

    def test_warmup_raises_when_renderer_never_delivers_a_frame(self):
        empty = types.SimpleNamespace(size=0)
        replicator, annotator = self._warmup_replicator([empty] * 5)

        with self.assertRaisesRegex(RuntimeError, "no frame for /Render/RP"):
            asyncio.run(
                _warmup(
                    types.SimpleNamespace(path="/Render/RP"),
                    replicator,
                    timeout_s=-1.0,
                ),
            )
        annotator.detach.assert_called_once_with("/Render/RP")

    def _run_missing_capture(
        self,
        *,
        limits,
        output_start=0,
        roi_specs=None,
        skip_pass=None,
        aim_offset=(0.0, 0.0, 0.0),
    ):
        """Drive ``execute_capture`` for a one-component missing pipeline.

        Mirrors a missing-component job (ict-baseboard, ``operations.missing`` on one
        component, one named pass focused via ``roi_selection: defects:
        [missing]``).  Returns the output plus the mocks the assertions need.
        ``skip_pass`` names a pass directory whose writer writes nothing.
        """

        class _Camera:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        stage = object()
        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            stage=stage,
            scene={"root_path": "/Board"},
            component_types=("Buzzer_SMD_CMI-9705-0380-SMT-TR",),
            plan_metadata={"random_seed": 7},
        )
        context.state.update(
            {
                "capture.renderer": {
                    "mode": "pathtracing",
                    "resolution": (64, 64),
                    "spp": 1,
                    "total_spp": 2,
                    "max_bounces": 1,
                },
                "capture.camera": {
                    "rotation_range": {"x": (0, 0), "y": (0, 0), "z": -90},
                },
                "capture.scan_grid": {"x_count": 1, "y_count": 1},
                "capture.outputs": {
                    "rgb": True,
                    "image_format": "png",
                    "frame_padding": 4,
                },
                "capture.limits": dict(limits),
                "capture.output": {"start": output_start},
                "operation.missing.paths": ["/Board/Layers/Top/Components/LS1"],
            },
        )
        if roi_specs is not None:
            context.state["capture.roi_selection"] = {
                "selections": [{"defects": ["missing"], "padding": 0.75}],
            }

        render_product = mock.Mock(path="render-product")
        replicator = types.ModuleType("omni.replicator.core")
        replicator.create = mock.Mock()
        replicator.create.render_product.return_value = render_product
        replicator.WriterRegistry = mock.Mock()
        replicator.WriterRegistry.get.side_effect = lambda name: mock.Mock()
        replicator.get = mock.Mock()
        replicator.get.prim_at_path.return_value = _Camera()
        replicator.modify = mock.Mock()
        replicator.distribution = mock.Mock()
        replicator.distribution.uniform.return_value = "rotation"
        replicator.orchestrator = mock.Mock()
        replicator.orchestrator.step_async = mock.AsyncMock()
        replicator.orchestrator.wait_until_complete_async = mock.AsyncMock()

        omni = types.ModuleType("omni")
        omni_replicator = types.ModuleType("omni.replicator")
        pxr = types.ModuleType("pxr")
        pxr.Gf = mock.Mock()
        pxr.Gf.Vec3d.side_effect = lambda x, y, z: (x, y, z)
        pxr.UsdGeom = mock.Mock()

        common = mock.Mock()
        common.configure_pathtracing.return_value = 2
        common.build_scan_positions.return_value = [(1.0, 2.0, 3.0)]
        common.find_translate_op.return_value = mock.Mock()
        common.randomize_lighting.return_value = {"ring_light": False}
        common.setup_augmentation.return_value = {"applied": False}
        common.scan_positions_for_trigger.side_effect = (
            lambda positions, trigger, seed: positions
        )
        common.grid_index_for_position.return_value = (0, 0)
        defect_ops = mock.Mock()
        missing_ops = mock.Mock()
        # Record the interleaving of ROI resolution with hide/restore so the
        # test can prove the target was visible while its bounds were read.
        events: list[str] = []
        missing_ops.hide_components.side_effect = lambda *_: events.append("hide")
        missing_ops.restore_components.side_effect = lambda *_: events.append("restore")
        scan_grid_geometry = mock.Mock()
        scan_grid_geometry.compute_pixel_layout.return_value = {
            "supported": True,
            "canvas_size_px": [64, 64],
        }
        _write_frame_on_step(replicator, common, skip_pass=skip_pass)

        def _roi_specs(*_args, **_kwargs):
            events.append("roi.select")
            return tuple(roi_specs or ())

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.dict(
                sys.modules,
                {
                    "omni": omni,
                    "omni.replicator": omni_replicator,
                    "omni.replicator.core": replicator,
                    "pxr": pxr,
                },
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._script_module",
                side_effect=lambda directory, name: {
                    "common": common,
                    "defect_ops": defect_ops,
                    "missing_ops": missing_ops,
                }[name],
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution.scan_grid_geometry",
                scan_grid_geometry,
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._roi_selection_specs",
                side_effect=_roi_specs,
            ),
            # The stub stage has no camera to aim; the aim has its own tests.
            mock.patch(
                "cdf.execution.isaac.capture_execution._aimed_roi_translate",
                side_effect=lambda _stage, _path, translate, _target: tuple(
                    value + offset for value, offset in zip(translate, aim_offset)
                ),
            ),
            mock.patch("cdf.execution.isaac.capture_execution._ensure_camera"),
            mock.patch("cdf.execution.isaac.capture_execution._setup_lighting"),
            mock.patch(
                "cdf.execution.isaac.capture_execution._setup_component_appearance",
                return_value={"applied": False},
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._warmup",
                new=mock.AsyncMock(),
            ),
        ):
            output = asyncio.run(
                execute_capture(
                    context,
                    CaptureRenderParameters(output_dir=temporary_directory),
                ),
            )
        return output, events, render_product

    def test_focused_frame_records_where_the_aimed_camera_stood(self):
        # A tilted crop camera is moved off the spec position to look at its
        # part; the frame must say where it was, as the calibration does.
        roi_spec = {
            "position": (4.0, 5.0, 6.0),
            "target": (4.0, 5.0, 0.0),
            "horizontal_aperture": 2.0,
            "vertical_aperture": 2.0,
        }
        output, _events, _render_product = self._run_missing_capture(
            limits={"triggers": 1, "max_images": 1},
            roi_specs=[roi_spec],
            aim_offset=(-6.0, 0.0, 0.0),
        )
        for frame in output["frames"]:
            self.assertEqual(frame["position"], [-2.0, 5.0, 6.0])
            self.assertEqual(frame["camera"]["position"], [-2.0, 5.0, 6.0])

    def test_missing_capture_with_one_image_budget_writes_one_aligned_pair(self):
        # Missing-component job: operations.missing on one component, roi_selection
        # defects: [missing], limits {triggers: 1, max_images: 1}.  Before the
        # fix the missing pipeline halved the budget (1 // 2 == 0 positions)
        # and the pass succeeded with frame_count 0 and no warning.
        roi_spec = {
            "position": (4.0, 5.0, 6.0),
            "target": (4.0, 5.0, 0.0),
            "horizontal_aperture": 2.0,
            "vertical_aperture": 2.0,
        }
        output, events, render_product = self._run_missing_capture(
            limits={"triggers": 1, "max_images": 1},
            roi_specs=[roi_spec],
        )

        self.assertEqual(output["pipeline_type"], "missing")
        self.assertEqual(output["frame_count"], 2)
        self.assertEqual(
            [frame["role"] for frame in output["frames"]],
            ["reference", "defective"],
        )
        self.assertEqual(
            [frame["position"] for frame in output["frames"]],
            [[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
        )
        self.assertEqual(len(output["trigger_directories"]), 1)
        self.assertEqual(len(output["trigger_metadata"]), 1)
        self.assertEqual(output["warnings"], [])
        # ROI bounds are read with the component restored, then it is hidden
        # again; the reference pass restores, the defective pass hides, and
        # the final cleanup restores.
        self.assertEqual(
            events,
            ["restore", "roi.select", "hide", "restore", "hide", "restore"],
        )
        render_product.destroy.assert_called_once_with()

    def test_missing_capture_budget_spans_triggers_by_image_set(self):
        # max_images=2 with two triggers yields two pairs (four frames), not
        # one pair: the budget counts positions, not writer frames.
        output, _events, _render_product = self._run_missing_capture(
            limits={"triggers": 3, "max_images": 2},
        )
        self.assertEqual(output["frame_count"], 4)
        self.assertEqual(len(output["trigger_directories"]), 2)
        self.assertEqual(
            [frame["role"] for frame in output["frames"]],
            ["reference", "defective", "reference", "defective"],
        )

    def test_missing_capture_raises_when_reference_writer_writes_nothing(self):
        # A writer attached before the renderer delivers frames writes nothing;
        # the pair must fail rather than report two captured frames.
        with self.assertRaisesRegex(RuntimeError, "no output") as raised:
            self._run_missing_capture(
                limits={"triggers": 1, "max_images": 1},
                skip_pass="reference",
            )
        message = str(raised.exception)
        self.assertIn(str(Path("trigger_0000") / "reference"), message)
        self.assertIn("frame:0000:reference", message)
        self.assertNotIn("defective", message)

    def test_capture_with_no_renderable_position_raises_instead_of_empty_pass(
        self,
    ):
        # output.start beyond the resolved ROI targets leaves every trigger
        # without a position; the pass must fail loudly, not succeed empty.
        roi_spec = {
            "position": (4.0, 5.0, 6.0),
            "target": (4.0, 5.0, 0.0),
            "horizontal_aperture": 2.0,
            "vertical_aperture": 2.0,
        }
        with self.assertRaises(ValidationError) as raised:
            self._run_missing_capture(
                limits={"triggers": 1, "max_images": 1},
                output_start=1,
                roi_specs=[roi_spec],
            )
        message = str(raised.exception)
        self.assertIn("rendered no frames", message)
        self.assertIn("pipeline_type=missing", message)
        self.assertIn("max_images=1", message)
        self.assertIn("output.start=1", message)
        self.assertIn("roi_targets=1", message)

    def test_sidecar_loader_is_kit_free(self):
        omni_was_loaded = "omni" in sys.modules
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "components.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "v1.1",
                        "board_name": "Board",
                        "board_prim_path": "/Board",
                        "components": [
                            {
                                "name": "PASSIVE_SMD_0805",
                                "family": "PASSIVE_0805",
                                "source": "passive.usdc",
                                "prototype_path": "/Board/Prototypes/PASSIVE_SMD_0805",
                                "side": "Top",
                                "process_class": "SMT",
                                "prim_path": "/Board/Components/PASSIVE_SMD_0805",
                                "instance_paths": ["/Board/Components/C1"],
                                "body_meshes": ["Body"],
                                "solder_contact_meshes": ["Pin1", "Pin2"],
                            },
                        ],
                        "component_instances": [
                            {
                                "path": "/Board/Components/C1",
                                "refdes": "C1",
                                "side": "Top",
                                "process_class": "SMT",
                                "family": "PASSIVE_0805",
                                "type": "CAPACITOR",
                                "body_prims": ["/Board/Components/C1/Body"],
                                "solder_contact_prims": [
                                    "/Board/Components/C1/Pin1",
                                    "/Board/Components/C1/Pin2",
                                ],
                                "solder_paste_prims": [],
                                "solder_paste_usdc_paths": [],
                            },
                        ],
                    },
                ),
                encoding="utf-8",
            )
            runtime = IsaacSdgRuntime()
            context = IsaacExecutionContext(runtime=runtime, stage=object())

            output = runtime.load_sidecar(
                context,
                SidecarLoadParameters(
                    name="components",
                    kind="pcb.components",
                    path=str(path),
                ),
            )

            self.assertEqual(output["records"], 5)
            catalog = context.board_metadata["components"]
            self.assertEqual(output["metadata_components"], 1)
            self.assertEqual(
                catalog.component(board_root="/Board", refdes="C1").prim_path,
                "/Board/Components/C1",
            )
            self.assertEqual("omni" in sys.modules, omni_was_loaded)

    @unittest.skipUnless(
        (
            _REPO_ROOT
            / "tests"
            / "functional"
            / "resources"
            / "SMT_0805_C18_coupon.components.json"
        ).is_file(),
        "approved SMT coupon fixture is not shipped in the public release",
    )
    def test_component_selection_uses_catalog_exact_paths(self):
        resource = (
            _REPO_ROOT
            / "tests"
            / "functional"
            / "resources"
            / "SMT_0805_C18_coupon.components.json"
        )
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=object(),
            scene={"root_path": "/Board_CAD_BOARD"},
        )
        runtime.load_sidecar(
            context,
            SidecarLoadParameters(
                name="components",
                kind="pcb.components",
                path=str(resource),
            ),
        )

        with mock.patch(
            "cdf.execution.isaac.job_order_runtime._standalone_module",
        ) as standalone_module:
            output = runtime.select_components(
                context,
                ComponentSelectionParameters(
                    source="components",
                    types=("PASSIVE_SMD_0805",),
                ),
            )

        standalone_module.assert_not_called()
        self.assertEqual(output["count"], 1)
        self.assertEqual(
            output["component_paths"],
            [
                "/Board_CAD_BOARD/Components/Top/PASSIVE_SMD_0805/C18",
            ],
        )
        with self.assertRaisesRegex(ValidationError, "no components found"):
            runtime.select_components(
                context,
                ComponentSelectionParameters(
                    source="components",
                    types=("NOT_A_REAL_PACKAGE",),
                ),
            )

    @unittest.skipUnless(
        (
            _REPO_ROOT
            / "tests"
            / "functional"
            / "resources"
            / "SMT_0805_C18_coupon.components.json"
        ).is_file(),
        "approved SMT coupon fixture is not shipped in the public release",
    )
    def test_attached_sidecar_anchors_job_metadata_source(self):
        resource = (
            _REPO_ROOT
            / "tests"
            / "functional"
            / "resources"
            / "SMT_0805_C18_coupon.components.json"
        )
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=object(),
            scene={"root_path": "/Board_CAD_BOARD"},
        )
        context.board_metadata["usd"] = _catalog(identifier="coupon.usdc")
        runtime.load_sidecar(
            context,
            SidecarLoadParameters(
                name="components",
                kind="pcb.components",
                path=str(resource),
            ),
        )
        self.assertNotIn("usd", context.board_metadata)
        self.assertEqual(context.state["board_metadata.source"], "components")

        output = runtime.select_components(
            context,
            ComponentSelectionParameters(
                source=None,
                types=("PASSIVE_SMD_0805",),
            ),
        )

        self.assertEqual(output["count"], 1)
        self.assertEqual(
            output["component_paths"],
            [
                "/Board_CAD_BOARD/Components/Top/PASSIVE_SMD_0805/C18",
            ],
        )
        with self.assertRaisesRegex(
            ValidationError,
            "attached component sidecar selects metadata source 'components'",
        ):
            runtime.select_components(
                context,
                ComponentSelectionParameters(
                    source="usd",
                    types=("PASSIVE_SMD_0805",),
                ),
            )

    def test_sidecar_loader_reads_remote_object_through_file_api(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        remote_path = "s3://bucket/boards/components.json"

        with (
            mock.patch(
                "cdf.execution.isaac.job_order_runtime.file_utils.is_file",
                return_value=True,
            ),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime.file_utils.open",
                mock.mock_open(
                    read_data=json.dumps(
                        {
                            "schema_version": "v1",
                            "board_name": "Board",
                            "board_prim_path": "/Board",
                            "components": [],
                            "component_instances": [],
                        },
                    ),
                ),
            ) as open_file,
        ):
            output = runtime.load_sidecar(
                context,
                SidecarLoadParameters(
                    name="components",
                    kind="pcb.components",
                    path=remote_path,
                ),
            )

        self.assertEqual(output["records"], 5)
        self.assertEqual(
            context.board_metadata["components"].boards[0].source.identifier,
            remote_path,
        )
        self.assertNotIn("components", context.json_documents)
        open_file.assert_called_once_with(remote_path, encoding="utf-8")

    def test_solder_discovery_groups_reusable_prototype_work(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        context.board_metadata["components"] = _catalog(
            {
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
                ],
                "component_instances": [
                    {
                        "path": "/Board/Components/Top/C1",
                        "refdes": "C1",
                        "solder_paste_prims": ["/Board/Paste/p1", "/Board/Paste/p2"],
                        "solder_paste_usdc_paths": ["board/paste.usdc"],
                    },
                ],
            },
        )

        output = runtime.discover_solder_joints(
            context,
            SolderJointParameters(
                solver="procedural",
                source="components",
            ),
        )

        self.assertEqual(output["count"], 1)
        self.assertEqual(output["solver"], "procedural")
        self.assertEqual(output["solver_type"], "procedural")
        self.assertFalse(output["requires_source_paste"])
        self.assertEqual(
            output["candidates"][0]["prototype_id"],
            "PASSIVE_SMD_0805",
        )
        self.assertEqual(
            context.state["solder_joints.candidates"][0].prototype_id,
            "PASSIVE_SMD_0805",
        )

    def test_solder_component_allowlist_filters_during_discovery(self):
        """Apply the exact solver allowlist before prototype grouping."""
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        context.board_metadata["components"] = _catalog(
            {
                "components": [
                    {
                        "name": "PASSIVE_SMD_0805",
                        "family": "PASSIVE_0805",
                        "side": "Top",
                        "prototype_path": "/Board/_Prototypes/PASSIVE_SMD_0805",
                        "source": "payloads/PASSIVE_SMD_0805.usdc",
                        "instance_paths": [
                            "/Board/Components/Top/C1",
                            "/Board/Components/Top/C2",
                        ],
                        "solder_contact_meshes": ["lead_left", "lead_right"],
                    },
                ],
                "component_instances": [
                    {
                        "path": f"/Board/Components/Top/C{index}",
                        "refdes": f"C{index}",
                    }
                    for index in (1, 2)
                ],
            },
        )
        context.component_pool = (
            "/Board/Components/Top/C1",
            "/Board/Components/Top/C2",
        )
        context.component_selection_resolved = True

        output = runtime.discover_solder_joints(
            context,
            SolderJointParameters(
                solver="procedural",
                source="components",
                solver_options={"components": ["C1"]},
            ),
        )

        self.assertEqual(
            output["candidates"][0]["instance_paths"],
            ("/Board/Components/Top/C1",),
        )
        self.assertEqual(
            context.state["solder_joints.component_pool"],
            ("/Board/Components/Top/C1",),
        )
        with self.assertRaisesRegex(ValidationError, "matched no component"):
            runtime.discover_solder_joints(
                context,
                SolderJointParameters(
                    solver="procedural",
                    source="components",
                    solver_options={"components": ["C3"]},
                ),
            )

    def test_solder_collection_retains_stage_catalog_and_applied_defects(self):
        runtime = IsaacSdgRuntime()
        stage = object()
        sidecar_data = {
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
            ],
            "component_instances": [
                {
                    "path": "/Board/Components/Top/C1",
                    "refdes": "C1",
                    "solder_paste_prims": ["/Board/Paste/p1", "/Board/Paste/p2"],
                    "solder_paste_usdc_paths": ["board/paste.usdc"],
                },
            ],
        }
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=stage,
            scene={
                "usd_path": "board.usd",
                "root_path": "/Board",
            },
        )
        catalog = _catalog(sidecar_data)
        context.board_metadata["components"] = catalog
        parameters = SolderJointParameters(
            solver="none",
            source="components",
        )
        runtime.discover_solder_joints(context, parameters)
        defect_record = {
            "path": "/Board/Components/Top/C1",
            "operation": "shift",
        }
        context.state["operation.structural.records"] = [defect_record]

        output = runtime.collect_solder_joint_inputs(context, parameters)

        generation_request = context.state["solder_joints.generation_request"]
        self.assertIsInstance(generation_request, SolderGenerationRequest)
        self.assertIs(generation_request.stage, stage)
        self.assertIs(generation_request.board_metadata, catalog)
        self.assertEqual(generation_request.defect_records, (defect_record,))
        self.assertEqual(output["candidate_count"], 1)
        self.assertEqual(output["defect_count"], 1)
        self.assertEqual(output["solver"], "none")
        self.assertEqual(output["solver_type"], "none")
        generated = runtime.generate_solder_joints(context, parameters)
        self.assertEqual(generated["artifact_count"], 1)
        self.assertEqual(generated["coverage"][0]["status"], "passthrough")

        procedural_parameters = SolderJointParameters(
            solver="procedural",
            source="components",
        )
        runtime.discover_solder_joints(context, procedural_parameters)
        procedural_output = runtime.collect_solder_joint_inputs(
            context,
            procedural_parameters,
        )
        self.assertEqual(procedural_output["solver_type"], "procedural")
        procedural_generated = runtime.generate_solder_joints(
            context,
            procedural_parameters,
        )
        self.assertEqual(procedural_generated["artifact_count"], 1)
        self.assertTrue(procedural_generated["warnings"])
        self.assertEqual(
            procedural_generated["coverage"][0]["fallback_from"],
            "procedural",
        )

    def test_solder_attachment_refreshes_authored_semantics(self):
        """Newly attached solder labels are registered before capture."""
        runtime = IsaacSdgRuntime()
        stage = object()
        context = IsaacExecutionContext(runtime=runtime, stage=stage)
        generation_request = SolderGenerationRequest(
            request_id="request",
            solver_id="none",
            source_usd_path="board.usd",
            root_path="/Board",
            output_root_path="/Board/SolderJoints",
            stage=stage,
            board_metadata=_catalog(),
            candidates=(),
            defect_records=(),
            serialization=SolderSerializationMode.NONE,
        )
        solver_request = SolderSolverRequest(
            request_id="request",
            solver_id="none",
            source_usd_path="board.usd",
            root_path="/Board",
            work_items=(),
        )
        solver_result = SolderSolverResult(
            request_id="request",
            solver_id="none",
            artifacts=(),
            coverage=(),
        )
        manifest = SolderPlacementManifest(
            request_id="request",
            solver_id="none",
            output_root_path="/Board/SolderJoints",
            edit_layer_identifier="solder.usda",
            generated_prim_paths=("/Board/SolderJoints/J1",),
            suppressed_source_prims=(),
            restored_source_prims=(),
            coverage=(),
        )
        extension = SolderGenerationExtension()
        context.state.update(
            {
                "solder_joints.extension": extension,
                "solder_joints.generation_request": generation_request,
                "solder_joints.solver_request": solver_request,
                "solder_joints.solver_result": solver_result,
            }
        )
        common = mock.Mock()

        with (
            mock.patch.object(extension, "attach", return_value=manifest),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
                return_value=common,
            ),
        ):
            output = runtime.attach_solder_joints(
                context,
                SolderJointParameters(solver="none", source="components"),
            )

        self.assertEqual(output["generated_prim_paths"], ["/Board/SolderJoints/J1"])
        common.refresh_authored_semantics.assert_called_once_with(
            stage,
            ["/Board/SolderJoints/J1"],
        )

        common.reset_mock()
        context.board_instance_cohort = BoardInstanceCohort(
            instance_paths=("/World/Panel/I0", "/World/Panel/I1"),
            representative_instance_path="/World/Panel/I0",
            prototype_path="/__Prototype_1",
            prototype_source_path="/Board",
        )
        with (
            mock.patch.object(extension, "attach", return_value=manifest),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
                return_value=common,
            ),
        ):
            runtime.attach_solder_joints(
                context,
                SolderJointParameters(solver="none", source="components"),
            )

        common.refresh_authored_semantics.assert_not_called()
        self.assertEqual(
            context.state["capture.instanced_stage_mutations"],
            ("solder_joints.attach",),
        )

    def test_geometry_annotations_mark_instanced_stage_mutation(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=object(),
            board_instance_cohort=BoardInstanceCohort(
                instance_paths=("/World/Panel/I0",),
                representative_instance_path="/World/Panel/I0",
                prototype_path="/__Prototype_1",
                prototype_source_path="/Board",
            ),
        )
        catalog = mock.Mock()
        catalog.terminals = (mock.Mock(prim_path="/Board/Components/R1/Pin1"),)

        with (
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
            ),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._board_metadata",
                return_value=catalog,
            ),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._label_geometry_prims",
                return_value=1,
            ),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._project_geometry_class",
                return_value=1,
            ),
        ):
            output = runtime.apply_geometry_annotations(
                context,
                GeometryAnnotationParameters(terminals="pin"),
            )

        self.assertEqual(output["terminals"], 1)
        self.assertEqual(
            context.state["capture.instanced_stage_mutations"],
            ("annotations.geometry",),
        )

    def test_geometry_annotations_keep_unowned_records_in_scene_scope(self):
        # The pin-variation coupon's holes carry a terminal number but no
        # component association; the scene pool must not label them away.
        catalog = mock.Mock()
        catalog.components = (
            mock.Mock(prim_path="/Board/Components/J13", refdes="J13"),
        )
        catalog.holes = (
            mock.Mock(prim_path="/Board/Holes/J13_1", component_path=None),
            mock.Mock(
                prim_path="/Board/Holes/R1_1",
                component_path="/Board/Components/R1",
            ),
        )
        labeled = []

        def label(stage, common, paths, class_name):
            labeled.append(tuple(paths))
            return len(paths)

        def run(parameters):
            runtime = IsaacSdgRuntime()
            context = IsaacExecutionContext(runtime=runtime, stage=object())
            context.component_pool = ("/Board/Components/J13",)
            context.component_selection_resolved = True
            labeled.clear()
            with (
                mock.patch(
                    "cdf.execution.isaac.job_order_runtime._standalone_module",
                ),
                mock.patch(
                    "cdf.execution.isaac.job_order_runtime._board_metadata",
                    return_value=catalog,
                ),
                mock.patch(
                    "cdf.execution.isaac.job_order_runtime._label_geometry_prims",
                    side_effect=label,
                ),
                mock.patch(
                    "cdf.execution.isaac.job_order_runtime._project_geometry_class",
                    return_value=0,
                ),
            ):
                return runtime.apply_geometry_annotations(context, parameters)

        output = run(GeometryAnnotationParameters(holes="hole"))
        self.assertEqual(output["holes"], 1)
        self.assertEqual(labeled, [("/Board/Holes/J13_1",)])

        with self.assertRaisesRegex(ValidationError, "found no holes records"):
            run(GeometryAnnotationParameters(holes="hole", components=("J13",)))

    def test_component_annotations_call_existing_common_api(self):
        runtime = IsaacSdgRuntime()
        stage = object()
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=stage,
            scene={"root_path": "/Board"},
            component_types=("PASSIVE_SMD_0805",),
            component_pool=("/Board/Components/R1",),
        )
        common = mock.Mock()
        common.apply_semantics.return_value = 1

        with mock.patch(
            "cdf.execution.isaac.job_order_runtime._standalone_module",
            return_value=common,
        ):
            output = runtime.apply_component_annotations(
                context,
                ComponentAnnotationParameters(source="components"),
            )

        self.assertEqual(output, {"count": 1})
        common.apply_semantics.assert_called_once_with(
            stage,
            "/Board",
            ["PASSIVE_SMD_0805"],
            None,
            component_paths=["/Board/Components/R1"],
        )

    def test_component_annotations_defer_native_instance_labels(self):
        runtime = IsaacSdgRuntime()
        instance_paths = ("/World/Panel/I0", "/World/Panel/I1")
        component_path = "/Board/Components/R1"
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=object(),
            board_instance_cohort=BoardInstanceCohort(
                instance_paths=instance_paths,
                representative_instance_path=instance_paths[0],
                prototype_path="/__Prototype_1",
                prototype_source_path="/Board",
            ),
            scene={
                "root_path": "/World/Panel",
                "board_root_path": "/Board",
                "instance_paths": instance_paths,
            },
            component_types=("RESISTOR",),
            component_pool=(component_path,),
        )
        context.board_metadata["usd"] = _catalog(
            {
                "components": [
                    {
                        "name": "RESISTOR",
                        "instance_paths": [component_path],
                    },
                ],
                "component_instances": [
                    {"path": component_path, "refdes": "R1"},
                ],
            },
        )

        with mock.patch(
            "cdf.execution.isaac.job_order_runtime._standalone_module",
        ) as standalone_module:
            output = runtime.apply_component_annotations(
                context,
                ComponentAnnotationParameters(
                    source="usd",
                    classes={"resistor": ("R1",)},
                ),
            )

        self.assertEqual(output["mode"], "instance_id_projection")
        self.assertEqual(output["count"], 2)
        standalone_module.assert_not_called()
        projection = context.instance_semantic_projection
        self.assertIsNotNone(projection)
        self.assertEqual(projection.cohort.instance_paths, instance_paths)
        self.assertEqual(
            projection.components[0].class_name,
            "resistor",
        )

    def test_capture_config_projects_only_before_deinstanced_refresh(self):
        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            scene={"root_path": "/World/Panel"},
            instance_semantic_projection=InstanceSemanticProjection(
                cohort=BoardInstanceCohort(
                    instance_paths=("/World/Panel/I0",),
                    representative_instance_path="/World/Panel/I0",
                    prototype_path="/__Prototype_1",
                    prototype_source_path="/Board",
                ),
                components=(
                    ComponentProjectionRecord(
                        source_path="/Board/Components/R1",
                        member_paths=("/Board/Components/R1",),
                        refdes="R1",
                        class_name="resistor",
                    ),
                ),
            ),
        )
        context.state.update(
            {
                "capture.renderer": {
                    "mode": "pathtracing",
                    "resolution": (320, 180),
                    "spp": 1,
                    "total_spp": 2,
                    "max_bounces": 2,
                },
                "capture.camera": {
                    "side": "bottom",
                    "rotation_range": {"x": 0, "y": 0, "z": -90},
                },
                "capture.scan_grid": {"x_count": 1, "y_count": 1},
                "capture.outputs": {
                    "rgb": True,
                    "image_format": "png",
                    "semantic_segmentation": True,
                    "colorize_semantic_segmentation": True,
                    "instance_id_segmentation": False,
                    "colorize_instance_id_segmentation": False,
                    "semantic_types": ("class",),
                    "frame_padding": 4,
                },
            },
        )

        config = build_capture_config(context, "output")

        self.assertFalse(config["writer"]["semantic_segmentation"])
        self.assertTrue(config["writer"]["instance_id_segmentation"])
        self.assertTrue(config["writer"]["colorize_instance_id_segmentation"])
        self.assertTrue(config["semantic_projection"]["colorized"])
        self.assertTrue(
            config["semantic_projection"]["remove_instance_artifacts"],
        )

        stage = mock.Mock()
        stage.GetPrimAtPath.return_value.IsInstance.return_value = False
        context.stage = stage
        context.board_instance_cohort = context.instance_semantic_projection.cohort

        native_config = build_capture_config(context, "output")

        self.assertTrue(native_config["writer"]["semantic_segmentation"])
        self.assertFalse(native_config["writer"]["instance_id_segmentation"])
        self.assertNotIn("semantic_projection", native_config)
        stage.GetPrimAtPath.assert_called_once_with("/World/Panel/I0")

    def test_tombstone_calls_existing_defect_api_and_retains_records(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=object(),
            scene={"root_path": "/Board"},
            component_types=("PASSIVE_SMD_0805",),
            component_pool=("/Board/Components/R1", "/Board/Components/R2"),
            plan_metadata={"random_seed": 7},
        )
        context.board_metadata["components"] = _catalog()
        defect_ops = mock.Mock()
        defect_ops.build_defect_context.return_value = {"contacts": True}
        defect_ops.prepare_defects.return_value = [
            {"path": "/Board/Components/R1", "defect": "tombstone"},
        ]
        # The real one: what an operation asked for is reported back, and a
        # ratio request is not the same as what it delivered.
        defect_ops.describe_targeting = describe_targeting
        numpy = mock.Mock()

        with (
            mock.patch.dict(sys.modules, {"numpy": numpy}),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
                return_value=defect_ops,
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._record_placed_bounds",
            ),
        ):
            output = runtime.apply_tombstone(
                context,
                TombstoneParameters(
                    ratio=0.5,
                    angle_range=(70, 90),
                    component_types=("PASSIVE_SMD_0805",),
                    eligibility_source="components",
                ),
            )

        self.assertEqual(
            output,
            {
                "count": 1,
                "requested": "50.0% requested",
                "component_paths": ["/Board/Components/R1"],
            },
        )
        defects_cfg = defect_ops.prepare_defects.call_args.args[1]
        numpy.random.seed.assert_called_once_with(7)
        self.assertEqual(defects_cfg["tombstone"]["angle_min"], 70)
        self.assertEqual(defects_cfg["tombstone"]["angle_max"], 90)
        self.assertEqual(
            defect_ops.prepare_defects.call_args.kwargs["excluded_paths"],
            set(),
        )
        self.assertEqual(
            context.state["operation.structural.records"][0]["path"],
            "/Board/Components/R1",
        )

    def _empty_transform_error(
        self,
        *,
        pool,
        used_records,
        component_types,
        eligible_ignoring_used,
        scene=None,
    ):
        """Run tombstone over ``pool`` with nothing applied; return the error text.

        ``prepare_defects`` returns no records, as it does when the eligible
        pool is empty, and ``_eligible_components`` answers what the
        operation COULD have used had nothing been taken yet.
        """
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=object(),
            scene=scene or {"root_path": "/Board", "board_root_path": "/Board"},
            component_types=("C_0603", "FB_0603"),
            component_pool=tuple(pool),
            plan_metadata={"random_seed": 7},
        )
        context.board_metadata["components"] = _catalog()
        context.state["operation.structural.records"] = list(used_records)
        context.state["operation.structural.count"] = len(used_records)
        defect_ops = mock.Mock()
        defect_ops.build_defect_context.return_value = {"contacts": True}
        defect_ops.prepare_defects.return_value = []
        defect_ops._eligible_components.return_value = list(eligible_ignoring_used)
        defect_ops.describe_targeting = describe_targeting
        with (
            mock.patch.dict(sys.modules, {"numpy": mock.Mock()}),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
                return_value=defect_ops,
            ),
            self.assertRaises(ValidationError) as raised,
        ):
            runtime.apply_tombstone(
                context,
                TombstoneParameters(
                    count=1,
                    angle_range=(70, 90),
                    component_types=component_types,
                    eligibility_source="components",
                ),
            )
        return str(raised.exception)

    def test_empty_transform_names_the_component_an_earlier_operation_took(self):
        """A pinned target already consumed by shift is reported as such.

        Case: shift (seed-driven, first in order) draws FB3, then tombstone
        pinned to FB3 has an empty eligible pool. The message must not blame
        unresolvable prim paths and send the user to check the board when the
        board is fine.
        """
        message = self._empty_transform_error(
            pool=["/Board/Components/C33", "/Board/Components/FB3"],
            used_records=[{"path": "/Board/Components/FB3", "defect": "shift"}],
            component_types=("/Board/Components/FB3",),
            eligible_ignoring_used=["/Board/Components/FB3"],
        )
        self.assertIn("earlier structural operations already took", message)
        self.assertIn("/Board/Components/FB3", message)
        self.assertIn("shift, sideflip", message)
        self.assertNotIn("do not resolve", message)

    def test_empty_transform_without_a_selector_does_not_name_component_types(self):
        """A selector-less operation is not blamed on an empty component_types.

        apply_tombstone/apply_reverse_polarity pass component_types=None when
        the caller leaves the selector unset, so the "already took" branch must
        describe the consumed eligible components without naming a knob the
        config never carried.
        """
        message = self._empty_transform_error(
            pool=["/Board/Components/C33", "/Board/Components/FB3"],
            used_records=[{"path": "/Board/Components/FB3", "defect": "shift"}],
            component_types=None,
            eligible_ignoring_used=["/Board/Components/FB3"],
        )
        self.assertIn("already taken by an earlier structural operation", message)
        self.assertIn("/Board/Components/FB3", message)
        self.assertNotIn("component_types", message)
        self.assertNotIn("do not resolve", message)

    def test_empty_transform_reports_an_exhausted_pool(self):
        message = self._empty_transform_error(
            pool=["/Board/Components/C33", "/Board/Components/FB3"],
            used_records=[
                {"path": "/Board/Components/C33", "defect": "shift"},
                {"path": "/Board/Components/FB3", "defect": "sideflip"},
            ],
            component_types=None,
            eligible_ignoring_used=[],
        )
        self.assertIn("Every component in the pool was already assigned", message)
        self.assertNotIn("do not resolve", message)

    def test_empty_transform_reports_a_selector_that_matches_nothing(self):
        message = self._empty_transform_error(
            pool=["/Board/Components/C33", "/Board/Components/FB3"],
            used_records=[],
            component_types=("/Board/Components/R99",),
            eligible_ignoring_used=[],
        )
        self.assertIn("match no component in the pool", message)
        self.assertIn("/Board/Components/R99", message)

    def test_empty_transform_keeps_the_metadata_mismatch_diagnosis(self):
        """Selectable components that still applied nothing: the original cause."""
        message = self._empty_transform_error(
            pool=["/Board/Components/C33", "/Board/Components/FB3"],
            used_records=[],
            component_types=None,
            eligible_ignoring_used=["/Board/Components/C33", "/Board/Components/FB3"],
        )
        self.assertIn("do not resolve under '/Board'", message)

    def test_empty_transform_diagnosis_falls_back_to_root_path(self):
        """A context that only carries scene.root_path still names the board root.

        ``scene.load`` sets ``board_root_path``, but the structural handlers
        are also reached from contexts built with ``root_path`` alone (every
        other setup in this file), and the diagnosis must not print ``None``.
        """
        message = self._empty_transform_error(
            pool=["/Board/Components/C33", "/Board/Components/FB3"],
            used_records=[],
            component_types=None,
            eligible_ignoring_used=["/Board/Components/C33", "/Board/Components/FB3"],
            scene={"root_path": "/Board"},
        )
        self.assertIn("do not resolve under '/Board'", message)
        self.assertNotIn("None", message)

    def test_transform_operation_authors_final_transform_to_usd(self):
        """The pose must be written to the USD stage, not just Fabric.

        defect_ops moves components with Replicator's pose_ops, which the USD
        stage never sees -- solder generation, attach placement, and stage
        serialization all read USD and would silently get NOMINAL poses. This
        locks the write-back: the record's final transform is authored onto
        the component prim as a matrix xform.
        """
        runtime = IsaacSdgRuntime()
        prim = mock.Mock()
        prim.IsValid.return_value = True
        stage = mock.Mock()
        stage.GetPrimAtPath.return_value = prim
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=stage,
            scene={"root_path": "/Board"},
            component_types=("PASSIVE_SMD_0805",),
            component_pool=("/Board/Components/R1",),
            plan_metadata={"random_seed": 7},
        )
        final = [float(v) for v in range(16)]
        defect_ops = mock.Mock()
        defect_ops.build_defect_context.return_value = {}
        defect_ops.prepare_defects.return_value = [
            {
                "path": "/Board/Components/R1",
                "defect": "tombstone",
                "tombstone_final_flat": final,
            },
        ]
        numpy = mock.Mock()
        pxr = types.ModuleType("pxr")
        pxr.Gf = mock.Mock()
        pxr.UsdGeom = mock.Mock()

        with (
            mock.patch.dict(sys.modules, {"numpy": numpy, "pxr": pxr}),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
                return_value=defect_ops,
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._record_placed_bounds",
            ),
        ):
            runtime.apply_tombstone(
                context,
                TombstoneParameters(
                    ratio=1.0,
                    angle_range=(70, 90),
                    component_types=None,
                ),
            )

        stage.GetPrimAtPath.assert_called_once_with("/Board/Components/R1")
        pxr.Gf.Matrix4d.assert_called_once_with(*final)
        pxr.UsdGeom.Xformable.assert_called_once_with(prim)
        (
            pxr.UsdGeom.Xformable.return_value.MakeMatrixXform.return_value.Set
        ).assert_called_once_with(pxr.Gf.Matrix4d.return_value)

    def test_placed_bounds_are_recorded_before_the_defect_is_authored(self):
        """Measure each part where it was placed before authoring its defect.

        Focused ROI crops frame a defect-moved part from these bounds. Once the
        final transform is authored onto the stage the placed pose is gone, so
        measuring afterwards would frame the crop around the defect instead.
        """
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=object(),
            scene={"root_path": "/Board"},
            component_types=("PASSIVE_SMD_0805",),
            component_pool=("/Board/Components/R1",),
            plan_metadata={"random_seed": 7},
        )
        records = [
            {
                "path": "/Board/Components/R1",
                "defect": "tombstone",
                "original_flat": [float(v) for v in range(16)],
                "tombstone_final_flat": [float(v) for v in range(16, 32)],
            },
        ]
        defect_ops = mock.Mock()
        defect_ops.build_defect_context.return_value = {}
        defect_ops.prepare_defects.return_value = records
        steps = mock.Mock()

        with (
            mock.patch.dict(sys.modules, {"numpy": mock.Mock()}),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
                return_value=defect_ops,
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._record_placed_bounds",
                steps.record_placed_bounds,
            ),
            mock.patch.object(
                IsaacSdgRuntime,
                "_author_defect_transforms",
                steps.author_defect_transforms,
            ),
        ):
            runtime.apply_tombstone(
                context,
                TombstoneParameters(
                    ratio=1.0,
                    angle_range=(70, 90),
                    component_types=None,
                ),
            )

        self.assertEqual(
            [name for name, _args, _kwargs in steps.mock_calls],
            ["record_placed_bounds", "author_defect_transforms"],
        )
        steps.record_placed_bounds.assert_called_once_with(
            context,
            context.stage,
            records,
        )

    def test_transform_operations_aggregate_non_overlapping_records(self):
        runtime = IsaacSdgRuntime()
        paths = [
            "/Board/Components/R1",
            "/Board/Components/C1",
            "/Board/Components/U1",
        ]
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=object(),
            scene={"root_path": "/Board"},
            component_types=("PASSIVE_SMD_0805", "IC_SMD"),
            component_pool=tuple(paths),
            plan_metadata={"random_seed": 7},
        )
        defect_ops = mock.Mock()
        defect_ops.build_defect_context.return_value = {}

        def prepare_defects(
            component_pool,
            defects,
            defect_context,
            excluded_paths,
            *,
            apply_semantics,
        ):
            self.assertTrue(apply_semantics)
            operation_name = next(iter(defects))
            available = [path for path in component_pool if path not in excluded_paths]
            return [{"path": available[0], "defect": operation_name}]

        defect_ops.prepare_defects.side_effect = prepare_defects
        numpy = mock.Mock()

        with (
            mock.patch.dict(sys.modules, {"numpy": numpy}),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
                return_value=defect_ops,
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._record_placed_bounds",
            ),
        ):
            runtime.apply_shift(
                context,
                ShiftParameters(
                    ratio=0.25,
                    translate_range=0.2,
                    rotate_z_range=15,
                ),
            )
            runtime.apply_sideflip(
                context,
                SideflipParameters(ratio=0.25, angle_range=(60, 80)),
            )
            runtime.apply_reverse_polarity(
                context,
                ReversePolarityParameters(ratio=0.25),
            )

        records = context.state["operation.structural.records"]
        self.assertEqual([record["path"] for record in records], paths)
        self.assertEqual(
            [record["defect"] for record in records],
            ["shift", "sideflip", "reverse_polarity"],
        )
        self.assertEqual(
            [call.args[0] for call in numpy.random.seed.call_args_list],
            [7, 8, 9],
        )

    def test_transform_exact_count_reaches_defect_utility(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=object(),
            scene={"root_path": "/Board"},
            component_pool=("/Board/Components/R1", "/Board/Components/R2"),
        )
        defect_ops = mock.Mock()
        defect_ops.build_defect_context.return_value = {}
        defect_ops.prepare_defects.return_value = [
            {"path": "/Board/Components/R1", "defect": "shift"},
            {"path": "/Board/Components/R2", "defect": "shift"},
        ]

        with (
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
                return_value=defect_ops,
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._record_placed_bounds",
            ),
        ):
            runtime.apply_shift(
                context,
                ShiftParameters(
                    count=2,
                    translate_range=0.2,
                    rotate_z_range=15,
                ),
            )

        config = defect_ops.prepare_defects.call_args.args[1]["shift"]
        self.assertEqual(config["count"], 2)
        self.assertNotIn("ratio", config)

    def test_missing_uses_existing_visibility_api_and_retains_paths(self):
        runtime = IsaacSdgRuntime()
        stage = object()
        context = IsaacExecutionContext(
            runtime=runtime,
            stage=stage,
            component_pool=("/Board/Components/R1", "/Board/Components/C1"),
            plan_metadata={"random_seed": 7},
        )
        missing_ops = mock.Mock()
        missing_ops.select_missing_components.return_value = [
            "/Board/Components/R1",
        ]
        numpy = mock.Mock()

        with (
            mock.patch.dict(sys.modules, {"numpy": numpy}),
            mock.patch(
                "cdf.execution.isaac.job_order_runtime._standalone_module",
                return_value=missing_ops,
            ),
        ):
            output = runtime.apply_missing(
                context,
                MissingParameters(ratio=0.5),
            )

        self.assertEqual(
            output,
            {"count": 1, "component_paths": ["/Board/Components/R1"]},
        )
        missing_ops.apply_missing_semantics.assert_called_once_with(
            stage,
            ["/Board/Components/R1"],
        )
        missing_ops.hide_components.assert_called_once_with(
            stage,
            ["/Board/Components/R1"],
        )
        self.assertEqual(
            context.state["operation.missing.paths"],
            ["/Board/Components/R1"],
        )

    def test_crop_handler_uses_structured_capture_output(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        context.outputs["capture.render"] = {
            "output_dir": "output",
            "pipeline_type": "defect",
        }
        adapter = mock.Mock()

        with mock.patch(
            "cdf.execution.isaac.job_order_runtime.importlib.import_module",
            return_value=adapter,
        ):
            output = runtime.crop_capture(
                context,
                PostprocessCropParameters(
                    method="bbox_tight",
                    types=("rgb", "semantic_segmentation"),
                    offset=32,
                    min_coverage=0.2,
                    class_filter=("tombstone",),
                    xform_depth=7,
                    output_subdir="tombstone_closeups",
                ),
            )

        self.assertTrue(output["executed"])
        self.assertEqual(
            output["output_dir"],
            str(Path("output") / "tombstone_closeups"),
        )
        crop_config = adapter.run_pipeline_crop.call_args.args[0]
        self.assertEqual(crop_config["pipeline_type"], "defect")
        self.assertTrue(crop_config["crop"]["enabled"])

    def test_crop_handler_runs_every_manifest_trigger(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        context.outputs["capture.render"] = {
            "output_dir": "output",
            "pipeline_type": "defect",
            "trigger_metadata": [
                {"trigger_idx": 0},
                {"trigger_idx": 2},
            ],
        }
        adapter = mock.Mock()

        with mock.patch(
            "cdf.execution.isaac.job_order_runtime.importlib.import_module",
            return_value=adapter,
        ):
            runtime.crop_capture(
                context,
                PostprocessCropParameters(
                    method="bbox_tight",
                    types=("rgb", "semantic_segmentation"),
                    offset=32,
                    min_coverage=0.2,
                    xform_depth=7,
                    output_subdir="crops",
                ),
            )

        configs = [call.args[0] for call in adapter.run_pipeline_crop.call_args_list]
        self.assertEqual([config["trigger_index"] for config in configs], [0, 2])
        self.assertEqual(
            [config["crop"]["output_subdir"] for config in configs],
            ["crops/trigger_0000", "crops/trigger_0002"],
        )

    def _crop_parameters(self, output_subdir="cropped"):
        return PostprocessCropParameters(
            method="bbox_tight",
            types=("rgb", "semantic_segmentation", "component_instance"),
            offset=10,
            min_coverage=0.2,
            xform_depth=7,
            output_subdir=output_subdir,
        )

    def test_crop_handler_uses_single_named_capture_output(self):
        # Regression: ``capture.passes.component_focus`` plus
        # ``postprocess.crop`` rendered four frames and then failed with
        # "requires a successful capture.render output" because the handler
        # only ever looked up the legacy default pass id.
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        context.outputs["capture.component_focus.renderer"] = {"mode": "pathtracing"}
        context.outputs["capture.component_focus.render"] = {
            "output_dir": "output/component_focus",
            "pipeline_type": "defect",
            "trigger_metadata": [{"trigger_idx": 0}],
        }
        context.manifest.begin_capture_pass(
            "capture.component_focus.render",
            output_uri="output/component_focus",
            local_root="output/component_focus",
        )
        adapter = mock.Mock()

        with mock.patch(
            "cdf.execution.isaac.job_order_runtime.importlib.import_module",
            return_value=adapter,
        ):
            output = runtime.crop_capture(context, self._crop_parameters())

        self.assertTrue(output["executed"])
        self.assertEqual(
            Path(output["output_dir"]),
            Path("output/component_focus/cropped"),
        )
        crop_config = adapter.run_pipeline_crop.call_args.args[0]
        self.assertEqual(crop_config["pipeline_type"], "defect")
        self.assertEqual(crop_config["output"], "output/component_focus")
        self.assertEqual(crop_config["crop"]["output_subdir"], "cropped")
        derived = context.manifest_view.capture_pass("capture.component_focus.render")[
            "derived_artifacts"
        ]
        self.assertEqual(
            [(artifact["kind"], artifact["path"]) for artifact in derived],
            [("crop_collection", "cropped")],
        )

    def test_crop_handler_prefers_default_pass_when_named_passes_coexist(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        context.outputs["capture.full_board.render"] = {
            "output_dir": "output/full_board",
            "pipeline_type": "defect",
        }
        context.outputs["capture.render"] = {
            "output_dir": "output",
            "pipeline_type": "defect",
        }
        adapter = mock.Mock()

        with mock.patch(
            "cdf.execution.isaac.job_order_runtime.importlib.import_module",
            return_value=adapter,
        ):
            output = runtime.crop_capture(context, self._crop_parameters())

        self.assertEqual(output["output_dir"], str(Path("output") / "cropped"))
        self.assertEqual(
            adapter.run_pipeline_crop.call_args.args[0]["output"],
            "output",
        )

    def test_crop_handler_refuses_ambiguous_named_passes(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        for name in ("full_board", "component_rois"):
            context.outputs[f"capture.{name}.render"] = {
                "output_dir": f"output/{name}",
                "pipeline_type": "defect",
            }
        adapter = mock.Mock()

        with (
            mock.patch(
                "cdf.execution.isaac.job_order_runtime.importlib.import_module",
                return_value=adapter,
            ),
            self.assertRaisesRegex(
                ValidationError,
                r"cannot choose between several named capture passes "
                r"\['capture.component_rois.render', 'capture.full_board.render'\]",
            ),
        ):
            runtime.crop_capture(context, self._crop_parameters())

        adapter.run_pipeline_crop.assert_not_called()

    def test_crop_handler_still_requires_a_render_output(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        context.outputs["capture.component_focus.renderer"] = {"mode": "pathtracing"}

        with self.assertRaisesRegex(
            ValidationError,
            "requires a successful capture.render output",
        ):
            runtime.crop_capture(context, self._crop_parameters())

    def test_stitch_handler_uses_named_capture_output(self):
        runtime = IsaacSdgRuntime()
        context = IsaacExecutionContext(runtime=runtime, stage=object())
        context.outputs["capture.full_board.render"] = {
            "output_dir": "output/full_board",
            "trigger_metadata": [{"trigger_idx": 0}],
        }
        artifacts = (
            {
                "trigger_idx": 0,
                "path": "output/full_board/stitched/panel.png",
                "canvas_size_px": [10000, 8000],
                "tile_count": 20,
            },
        )

        with mock.patch(
            "cdf.postprocess.image_stitching.stitch_capture_grid",
            return_value=artifacts,
        ) as stitch:
            output = runtime.stitch_capture(
                context,
                PostprocessStitchParameters(
                    source_pass="full_board",
                    image_type="semantic_segmentation",
                    output_filename="panel.png",
                    blend="labels",
                    recolor_tiles=True,
                ),
            )

        stitch.assert_called_once_with(
            "output/full_board",
            [{"trigger_idx": 0}],
            image_type="semantic_segmentation",
            output_subdir="stitched",
            output_filename="panel.png",
            blend="labels",
            write_labels=False,
            recolor_tiles=True,
            color_map=None,
        )
        self.assertTrue(output["executed"])
        self.assertEqual(
            output["artifacts"][0]["canvas_size_px"],
            [10000, 8000],
        )


if __name__ == "__main__":
    unittest.main()
