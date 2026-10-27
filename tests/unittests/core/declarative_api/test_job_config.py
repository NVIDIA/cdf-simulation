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

"""Tests for the YAML JobConfig-to-Python-API interpreter."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml
from pydantic import ValidationError as PydanticValidationError
from pydantic import create_model

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi domain
from cdf import (
    JobConfig,
    JobResult,
    CDFJob,
    build_job_from_config,
    dump_job_config_schema,
    load_job,
)
from cdf.config import CDF_API_STATUS, CDF_API_VERSION
from cdf.config._common import ConfigModel
from cdf.config.cli import main as config_main
from cdf.config.job_execution import (
    EXECUTION_API_CONFIG_FIELDS,
    ExecutionConfig,
)
from cdf.errors import CDFError
from cdf.sections import (
    AppearanceSection,
    PostprocessSection,
    SceneSection,
)
from pcb_aoi.config.appearance import APPEARANCE_API_CONFIG_FIELDS, AppearanceConfig
from pcb_aoi.config.capture import (
    CAPTURE_API_CONFIG_FIELDS,
    RENDER_PASS_API_CONFIG_FIELDS,
    CaptureConfig,
    RenderPassConfig,
)
from pcb_aoi.config.postprocess import (
    POSTPROCESS_API_CONFIG_FIELDS,
    PostprocessConfig,
)
from pcb_aoi.config.scene import SCENE_API_CONFIG_FIELDS, SceneConfig
from pcb_aoi.sections import PcbAoiCapturePassSection, PcbAoiCaptureSection


def _config_payload() -> dict[str, object]:
    return {
        "api_version": CDF_API_VERSION,
        "domain": "pcb-aoi",
        "random_seed": 7,
        "output_dir": "output",
        "scene": {
            "usd": "board.usdc",
            "root": "/Board",
            "camera": "/World/Camera",
            "components": {
                "sidecar": "board.components.json",
                "types": ["THT_8"],
            },
        },
        "appearance": {
            "lighting": {
                "mode": "rgb_light",
                "exposure": 0.25,
                "layers": {
                    "Red": {"intensity": 6500, "color": [1, 0, 0]},
                },
            },
            "vantablack_components": False,
        },
        "operations": {
            "solder_joints": {
                "solver": "procedural",
                "options": {"solid": True},
            },
        },
        "annotations": {"components": True},
        "capture": {
            "renderer": {"resolution": [320, 180], "total_spp": 2},
            "camera": {
                "side": "bottom",
                "rotation": {"x": 0, "y": [-2, 2], "z": -90},
            },
            "scan_grid": [1, 1],
            "visibility": {"hide": ["solder"]},
            "limits": {"max_images": 1},
            "outputs": {
                "annotators": ["rgb", "semantic_segmentation"],
                "semantic_types": ["class"],
                "colorize": ["semantic_segmentation"],
                "rename_to_grid_index": True,
            },
        },
    }


class _ConfigS3Client:
    """Serve one YAML object through the boto3 download interface."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.contents = yaml.safe_dump(payload).encode("utf-8")
        self.downloads: list[tuple[str, str]] = []

    def download_fileobj(
        self,
        bucket: str,
        key: str,
        handle: io.BufferedIOBase,
    ) -> None:
        self.downloads.append((bucket, key))
        handle.write(self.contents)


class JobConfigTest(unittest.TestCase):
    def test_appearance_python_api_and_yaml_fields_remain_in_lockstep(self) -> None:
        public_methods = {
            name
            for name, value in inspect.getmembers(AppearanceSection)
            if not name.startswith("_") and callable(value)
        }

        self.assertEqual(public_methods, set(APPEARANCE_API_CONFIG_FIELDS))
        self.assertEqual(
            set(APPEARANCE_API_CONFIG_FIELDS.values()),
            set(AppearanceConfig.model_fields),
        )

    def test_postprocess_python_api_and_yaml_fields_remain_in_lockstep(self) -> None:
        public_methods = {
            name
            for name, value in inspect.getmembers(PostprocessSection)
            if not name.startswith("_") and callable(value)
        }

        self.assertEqual(public_methods, set(POSTPROCESS_API_CONFIG_FIELDS))
        self.assertEqual(
            set(POSTPROCESS_API_CONFIG_FIELDS.values()),
            set(PostprocessConfig.model_fields),
        )

    def test_scene_python_api_and_yaml_fields_remain_in_lockstep(self) -> None:
        public_methods = {
            name
            for name, value in inspect.getmembers(SceneSection)
            if not name.startswith("_") and callable(value)
        }
        mapped_fields = {
            field for fields in SCENE_API_CONFIG_FIELDS.values() for field in fields
        }

        self.assertEqual(
            public_methods,
            set(SCENE_API_CONFIG_FIELDS),
            "Every public scene declaration must have an explicit YAML mapping",
        )
        self.assertEqual(
            mapped_fields,
            set(SceneConfig.model_fields),
            "Every YAML scene field must map to a public Python declaration",
        )

    def test_capture_python_api_and_yaml_fields_remain_in_lockstep(self) -> None:
        non_config_methods = {"render", "render_async", "render_with_executor"}
        # The domain's capture section, not the core one. A domain may add a
        # way of stating a capture in its own terms -- pcb-aoi adds ``fov`` --
        # and the YAML contract under test is the one a pcb-aoi job writes.
        # This is the class the registry hands a job, so it is the surface a
        # caller actually has.
        public_methods = {
            name
            for name, value in inspect.getmembers(PcbAoiCaptureSection)
            if not name.startswith("_") and callable(value)
        }
        render_pass_methods = {
            name
            for name, value in inspect.getmembers(PcbAoiCapturePassSection)
            if not name.startswith("_") and callable(value)
        }

        self.assertEqual(
            public_methods - non_config_methods,
            set(CAPTURE_API_CONFIG_FIELDS),
            "Every public capture declaration must have an explicit YAML mapping",
        )
        self.assertEqual(
            set(CAPTURE_API_CONFIG_FIELDS.values()),
            set(CaptureConfig.model_fields),
            "Every YAML capture field must map to a public Python declaration",
        )
        self.assertEqual(
            render_pass_methods,
            set(RENDER_PASS_API_CONFIG_FIELDS),
            "Every render-pass declaration must have an explicit YAML mapping",
        )
        self.assertEqual(
            set(RENDER_PASS_API_CONFIG_FIELDS.values()),
            set(RenderPassConfig.model_fields),
            "Every YAML render-pass field must map to a Python declaration",
        )

    def test_execution_python_api_and_yaml_fields_remain_in_lockstep(self) -> None:
        execution_parameters = {
            name
            for name, parameter in inspect.signature(CDFJob.execute).parameters.items()
            # `windowed` and `experience` are properties of the machine the
            # job runs on, not of the job: one is whether there is a screen,
            # the other is a path to a Kit `.kit` file in a particular Isaac
            # install. A JobConfig is meant to travel, so neither belongs in
            # it -- unlike the outputs below, which describe the work.
            if name not in {"self", "windowed", "experience"}
            and parameter.kind is inspect.Parameter.KEYWORD_ONLY
        }

        self.assertEqual(
            execution_parameters,
            set(EXECUTION_API_CONFIG_FIELDS),
            "Every execution output must have an explicit YAML mapping",
        )
        self.assertEqual(
            set(ExecutionConfig.model_fields),
            {"serialization_path"},
        )
        self.assertIn("output_dir", JobConfig.model_fields)

    def test_exact_operation_count_compiles_without_ratio(self) -> None:
        payload = _config_payload()
        payload["operations"] = {
            "shift": {
                "count": 2,
                "translate": 0.2,
                "rotate_z": 15,
            }
        }

        config = JobConfig.model_validate(payload)
        job = build_job_from_config(config, base_dir=Path("fixture"))
        plan = job.compile_orders(output_dir="output")
        shift = next(
            order for order in plan.orders if order.order_id == "operation.shift"
        )

        self.assertEqual(shift.parameters["count"], 2)
        self.assertNotIn("ratio", shift.parameters)

    def test_scene_instances_compile_from_yaml(self) -> None:
        payload = _config_payload()
        instance_paths = (
            "/World/Panel/StepRepeats/SR_0_0",
            "/World/Panel/StepRepeats/SR_0_1",
        )
        payload["scene"]["root"] = "/World/Panel"
        payload["scene"]["instances"] = list(instance_paths)
        payload["scene"].pop("components", None)

        job = build_job_from_config(
            JobConfig.model_validate(payload),
            base_dir=Path("fixture"),
        )
        scene = next(
            order
            for order in job.compile_orders(output_dir="output").orders
            if order.order_id == "scene.load"
        )

        self.assertEqual(scene.parameters["instance_paths"], instance_paths)

    def test_operation_source_cannot_override_scene_metadata_source(self) -> None:
        payload = _config_payload()
        payload["operations"]["solder_joints"]["source"] = "usd"

        with self.assertRaisesRegex(
            ValueError,
            "solder_joints source must match.*'components'",
        ):
            build_job_from_config(
                JobConfig.model_validate(payload),
                base_dir=Path("fixture"),
            )

    def test_tombstone_source_cannot_override_scene_metadata_source(self) -> None:
        payload = _config_payload()
        payload["operations"] = {
            "tombstone": {
                "ratio": 0.25,
                "angle": [70, 90],
                "eligibility_source": "usd",
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            "tombstone eligibility_source must match.*'components'",
        ):
            build_job_from_config(
                JobConfig.model_validate(payload),
                base_dir=Path("fixture"),
            )

    def test_execution_serialization_path_requires_usd_extension(self) -> None:
        payload = _config_payload()
        payload["execution"] = {"serialization_path": "generated/final.json"}

        with self.assertRaisesRegex(PydanticValidationError, "must end with"):
            JobConfig.model_validate(payload)

    def test_interpreter_authors_public_api_and_compiles_orders(self) -> None:
        config = JobConfig.model_validate(_config_payload())
        job = build_job_from_config(config, base_dir=Path("fixture"))
        plan = job.compile_orders(output_dir="output")
        by_id = {order.order_id: order for order in plan.orders}

        self.assertEqual(by_id["capture.renderer"].parameters["resolution"], (320, 180))
        self.assertEqual(
            by_id["capture.camera"].parameters["rotation_range"],
            {"x": (0, 0), "y": (-2, 2), "z": -90},
        )
        self.assertEqual(
            by_id["appearance.lighting"].parameters["layers"]["Red"],
            {
                "intensity": (6500, 6500),
                "color_r": (1, 1),
                "color_g": (0, 0),
                "color_b": (0, 0),
            },
        )
        self.assertEqual(
            by_id["solder_joints.generate"].parameters["component_types"],
            ("THT_8",),
        )
        self.assertFalse(by_id["capture.outputs"].parameters["bounding_box_3d"])
        self.assertEqual(
            by_id["capture.visibility"].parameters["hide"],
            ("solder",),
        )
        self.assertTrue(
            by_id["capture.outputs"].parameters["rename_to_grid_index"],
        )
        declaration_ids = [declaration.order_id for declaration in job._declarations]
        self.assertLess(
            declaration_ids.index("operation.solder_joints"),
            declaration_ids.index("annotations.components"),
        )

    def test_interpreter_preserves_nested_component_filters(self) -> None:
        """Lower exact scene and operation filters without legacy inference."""
        payload = _config_payload()
        payload["scene"]["components"] = {
            "component_filter": {
                "side": ["Top"],
                "process_classes": ["SMT"],
            },
        }
        payload["operations"]["shift"] = {
            "ratio": 0.25,
            "translate": 0.2,
            "rotate_z": 15,
            "component_filter": {"package_type": ["0603"]},
        }

        job = build_job_from_config(JobConfig.model_validate(payload))
        by_id = {order.order_id: order for order in job.compile_orders().orders}

        self.assertEqual(
            dict(by_id["scene.components.select"].parameters["component_filter"]),
            {"side": ("Top",), "process_classes": ("SMT",)},
        )
        self.assertEqual(
            dict(by_id["operation.shift"].parameters["component_filter"]),
            {"package_type": ("0603",)},
        )

    def test_component_filter_rejects_legacy_selector_in_same_block(self) -> None:
        """Reject ambiguous normalized and overloaded scene selectors."""
        payload = _config_payload()
        payload["scene"]["components"]["component_filter"] = {
            "side": ["Top"],
        }

        with self.assertRaisesRegex(PydanticValidationError, "mutually exclusive"):
            JobConfig.model_validate(payload)

    def test_component_filter_rejects_nested_legacy_selectors(self) -> None:
        """Reject mixed selector forms in operation and annotation blocks."""
        cases = (
            (
                "solder_joints",
                {
                    "solver": "procedural",
                    "component_types": ["SMT"],
                    "component_filter": {"side": ["Top"]},
                },
            ),
            (
                "pin_variation",
                [
                    {
                        "components": ["J1"],
                        "component_filter": {"side": ["Top"]},
                    },
                ],
            ),
            (
                "hole_plane",
                {
                    "side": "bottom",
                    "components": ["J1"],
                    "component_filter": {"side": ["Top"]},
                },
            ),
        )
        for name, value in cases:
            with self.subTest(name=name):
                payload = _config_payload()
                payload["operations"][name] = value
                with self.assertRaisesRegex(
                    PydanticValidationError,
                    "component_filter",
                ):
                    JobConfig.model_validate(payload)

        payload = _config_payload()
        payload["annotations"]["geometry"] = {
            "pads": "pad",
            "components": ["J1"],
            "component_filter": {"side": ["Top"]},
        }
        with self.assertRaisesRegex(PydanticValidationError, "component_filter"):
            JobConfig.model_validate(payload)

    def test_legacy_scene_pool_allows_nested_normalized_solder_filter(self) -> None:
        """Keep parent legacy selection separate from a normalized child filter."""
        payload = _config_payload()
        payload["operations"]["solder_joints"] = {
            "solver": "procedural",
            "component_filter": {"side": ["Top"]},
        }

        job = build_job_from_config(JobConfig.model_validate(payload))
        by_id = {order.order_id: order for order in job.compile_orders().orders}

        self.assertIsNone(by_id["solder_joints.generate"].parameters["component_types"])
        self.assertEqual(
            dict(by_id["solder_joints.generate"].parameters["component_filter"]),
            {"side": ("Top",)},
        )

    def test_copper_hide_compiles_to_visibility_order(self) -> None:
        payload = _config_payload()
        payload["capture"]["visibility"] = {
            "hide": ["solder", "copper"],
            "show": ["hole", "pin_tip"],
        }
        config = JobConfig.model_validate(payload)
        job = build_job_from_config(config, base_dir=Path("fixture"))
        plan = job.compile_orders(output_dir="output")
        by_id = {order.order_id: order for order in plan.orders}

        self.assertEqual(
            set(by_id["capture.visibility"].parameters["hide"]),
            {"solder", "copper"},
        )
        self.assertEqual(
            set(by_id["capture.visibility"].parameters["show"]),
            {"hole", "pin_tip"},
        )

    def test_load_job_resolves_asset_paths_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "job.yaml"
            config_path.write_text(yaml.safe_dump(_config_payload()), encoding="utf-8")

            _, job = load_job(config_path)
            plan = job.compile_orders(output_dir="output")
            scene = next(
                order for order in plan.orders if order.order_id == "scene.load"
            )

            self.assertEqual(
                Path(scene.parameters["usd_path"]),
                config_path.parent / "board.usdc",
            )

    def test_load_job_reads_s3_config_and_resolves_relative_paths(self) -> None:
        client = _ConfigS3Client(_config_payload())

        config, job = load_job(
            "s3://input-bucket/jobs/board/job.yaml",
            storage_client=client,
        )
        plan = job.compile_orders(output_dir=config.output_dir)
        by_id = {order.order_id: order for order in plan.orders}

        self.assertEqual(client.downloads, [("input-bucket", "jobs/board/job.yaml")])
        self.assertEqual(
            by_id["scene.load"].parameters["usd_path"],
            "s3://input-bucket/jobs/board/board.usdc",
        )
        self.assertEqual(
            by_id["scene.sidecar.components"].parameters["path"],
            "s3://input-bucket/jobs/board/board.components.json",
        )

    def test_s3_config_records_the_hash_of_the_bytes_loaded(self) -> None:
        # A remote config used to leave config_sha256 None while the provenance
        # contract said every load_job() config is pinned by its bytes.
        payload = _config_payload()
        client = _ConfigS3Client(payload)

        _, job = load_job("s3://input-bucket/jobs/job.yaml", storage_client=client)

        self.assertEqual(job.source["config_path"], "s3://input-bucket/jobs/job.yaml")
        self.assertEqual(
            job.source["config_sha256"],
            hashlib.sha256(client.contents).hexdigest(),
        )

    def test_compiled_plan_carries_the_run_output_root(self) -> None:
        # Provenance must never enumerate artifacts outside the run; the plan
        # is what reaches the Isaac child, so the boundary travels on it.
        payload = _config_payload()
        client = _ConfigS3Client(payload)

        _, job = load_job("s3://input-bucket/jobs/job.yaml", storage_client=client)
        plan = job.compile_orders(output_dir="output")

        self.assertEqual(plan.metadata["output_root"], "output")
        capture_dirs = [
            order.parameters["output_dir"]
            for order in plan.orders
            if order.order_type == "capture.render"
        ]
        self.assertTrue(capture_dirs)
        for directory in capture_dirs:
            self.assertTrue(str(directory).startswith("output"))

    def test_s3_config_preserves_explicit_remote_asset_uris(self) -> None:
        payload = _config_payload()
        payload["scene"]["usd"] = "s3://assets-bucket/boards/board.usdc"
        client = _ConfigS3Client(payload)

        _, job = load_job(
            "s3://input-bucket/jobs/job.yaml",
            storage_client=client,
        )
        plan = job.compile_orders(output_dir="output")
        scene = next(order for order in plan.orders if order.order_id == "scene.load")

        self.assertEqual(
            scene.parameters["usd_path"],
            "s3://assets-bucket/boards/board.usdc",
        )

    def test_unknown_fields_are_rejected(self) -> None:
        payload = _config_payload()
        payload["unexpected"] = True

        with self.assertRaises(PydanticValidationError):
            JobConfig.model_validate(payload)

    def test_non_mapping_document_uses_pydantic_validation_error(self) -> None:
        for payload in ("not-a-mapping", ["not", "a", "mapping"]):
            with (
                self.subTest(payload=payload),
                self.assertRaisesRegex(
                    PydanticValidationError,
                    "JobConfig must be a mapping",
                ),
            ):
                JobConfig.model_validate(payload)

    def test_pin_tip_yaml_compiles_to_domain_operation(self) -> None:
        payload = _config_payload()
        payload["operations"]["pin_tip"] = {
            "side": "bottom",
            "class_name": "pin_tip",
            "offset_mm": 0.002,
            "components": ["X205"],
        }

        config = JobConfig.model_validate(payload)
        job = build_job_from_config(config)
        order = next(
            order
            for order in job.compile_orders(output_dir=config.output_dir).orders
            if order.order_id == "operation.pin_tip"
        )

        self.assertEqual(order.order_type, "operation.pin_tip.generate")
        self.assertEqual(order.parameters["components"], ("X205",))

    def test_light_layer_rejects_fixed_color_with_channel_ranges(self) -> None:
        payload = _config_payload()
        payload["appearance"]["lighting"]["layers"]["Red"]["color_r"] = [0.9, 1.0]

        with self.assertRaisesRegex(
            PydanticValidationError,
            "color cannot be combined with color_r",
        ):
            JobConfig.model_validate(payload)

    def test_api_version_requires_supported_semver(self) -> None:
        payload = _config_payload()
        payload["api_version"] = "1.0"

        with self.assertRaisesRegex(
            PydanticValidationError,
            "unsupported CDF API version '1.0'.*this build accepts "
            f"'{CDF_API_VERSION}'",
        ):
            JobConfig.model_validate(payload)

    def test_generated_schema_identifies_api_version(self) -> None:
        schema = json.loads(dump_job_config_schema())

        self.assertEqual(
            schema["$id"],
            f"urn:cdf:job-config:api-version:{CDF_API_VERSION}",
        )
        self.assertEqual(
            schema["properties"]["api_version"]["const"],
            CDF_API_VERSION,
        )
        self.assertNotIn("capture", schema["required"])
        self.assertIn(
            {"type": "null"},
            schema["properties"]["capture"]["anyOf"],
        )
        self.assertIsNone(schema["properties"]["capture"]["default"])
        self.assertEqual(
            schema["properties"]["execution"],
            {"$ref": "#/$defs/ExecutionConfig"},
        )
        self.assertIn(
            "serialization_path",
            schema["$defs"]["ExecutionConfig"]["properties"],
        )

    def test_generated_schema_requires_a_registered_domain(self) -> None:
        with (
            mock.patch(
                "cdf.config.core.load_installed_domain_configs",
            ),
            mock.patch(
                "cdf.config.core.registered_domain_configs",
                return_value={},
            ),
            self.assertRaisesRegex(
                CDFError,
                "no CDF domain configuration plugins are registered",
            ),
        ):
            dump_job_config_schema()

    def test_schema_cli_reports_unknown_domain_without_traceback(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            result = config_main(
                ["--print-schema", "--domain", "not-a-real-domain"],
            )

        self.assertEqual(result, 2)
        self.assertIn("unknown domain 'not-a-real-domain'", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_generated_schema_namespaces_domain_model_names(self) -> None:
        model_a = create_model(
            "SharedConfig",
            __base__=ConfigModel,
            alpha=(str, ...),
        )
        model_b = create_model(
            "SharedConfig",
            __base__=ConfigModel,
            beta=(int, ...),
        )

        def adapter(model):
            return SimpleNamespace(
                scene_model=model,
                appearance_model=model,
                annotations_model=model,
                operations_model=model,
                capture_model=model,
                postprocess_model=model,
            )

        with mock.patch(
            "cdf.config.core.registered_domain_configs",
            return_value={
                "domain-a": adapter(model_a),
                "domain-b": adapter(model_b),
            },
        ):
            schema = JobConfig.model_json_schema()

        definitions = schema["$defs"]
        self.assertIn(
            "alpha",
            definitions["domain-a.SharedConfig"]["properties"],
        )
        self.assertIn(
            "beta",
            definitions["domain-b.SharedConfig"]["properties"],
        )
        self.assertEqual(
            [reference["$ref"] for reference in schema["properties"]["scene"]["oneOf"]],
            [
                "#/$defs/domain-a.SharedConfig",
                "#/$defs/domain-b.SharedConfig",
            ],
        )

    def test_config_models_have_class_descriptions(self) -> None:
        undocumented = sorted(
            model.__name__
            for model in ConfigModel.__subclasses__()
            if not (model.__dict__.get("__doc__") or "").strip()
        )

        self.assertEqual(undocumented, [])

    def test_checked_in_examples_validate_and_compile(self) -> None:
        examples_dir = (
            Path(__file__).resolve().parents[4] / "pcb_aoi" / "config" / "examples"
        )
        examples = sorted(examples_dir.glob("*.yaml"))

        # A hard count has to be bumped for every example added, and it is not
        # what this test is for -- the loop below compiles each one, which is.
        # Keep a floor so an empty or mis-globbed directory still fails loudly.
        self.assertGreaterEqual(len(examples), 11)
        from pcb_aoi.solder_generation import registered_solver_ids

        available_solvers = set(registered_solver_ids(available_only=True))
        for path in examples:
            with self.subTest(example=path.name):
                # An example may select a solder solver that is environment-gated:
                # `surface_evolver` registers only where its geometry deps import
                # and the Evolver binary is built. Skip such an example where the
                # solver is absent (e.g. the CI numpy-only image) rather than fail
                # -- selecting an unregistered solver raises at load time by design.
                document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                operations = document.get("operations") or {}
                solder = operations.get("solder_joints") or {}
                solver = solder.get("solver")
                if solver and solver not in available_solvers:
                    self.skipTest(f"solder solver {solver!r} not registered")
                config, job = load_job(path)
                plan = job.compile_orders(output_dir=config.output_dir)
                self.assertGreater(len(plan.orders), 0)

    def test_api_version_comes_from_version_markdown(self) -> None:
        version_file = Path(__file__).resolve().parents[4] / "cdf/VERSION.md"

        self.assertIn(
            f"Current API version: `{CDF_API_VERSION}`",
            version_file.read_text(encoding="utf-8"),
        )

    def test_cli_errors_identify_active_api_version(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            return_code = config_main([])

        self.assertEqual(return_code, 2)
        self.assertIn(f"error [CDF API {CDF_API_VERSION}]", stderr.getvalue())

    def test_cli_forwards_capture_free_final_stage_serialization(self) -> None:
        config = JobConfig.model_validate(
            {
                "api_version": CDF_API_VERSION,
                "domain": "pcb-aoi",
                "scene": {"usd": "board.usda", "root": "/Board"},
                "execution": {"serialization_path": "generated/final.usda"},
            },
        )
        job = mock.Mock()
        job.execute.return_value = JobResult(success=True)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "cdf.config.cli.load_job",
                return_value=(config, job),
            ),
            redirect_stdout(io.StringIO()),
        ):
            config_path = Path(directory) / "job.yaml"
            return_code = config_main([str(config_path), "--execute"])

        self.assertEqual(return_code, 0)
        job.execute.assert_called_once_with(
            output_dir=None,
            serialization_path=str(
                config_path.parent / "generated" / "final.usda",
            ),
            windowed=False,
        )

    def test_cli_version_reports_package_and_api_contract(self) -> None:
        stdout = io.StringIO()

        with (
            mock.patch(
                "cdf.config.cli.distribution_version",
                return_value="3.4.5",
            ),
            redirect_stdout(stdout),
        ):
            return_code = config_main(["--version"])

        self.assertEqual(return_code, 0)
        self.assertEqual(
            stdout.getvalue().splitlines(),
            [
                "cdf-simulation package: 3.4.5",
                f"CDF authoring API:      {CDF_API_VERSION} ({CDF_API_STATUS})",
            ],
        )

    def test_cli_version_falls_back_to_source_tree_package_version(self) -> None:
        stdout = io.StringIO()

        with (
            mock.patch(
                "cdf.config.cli.distribution_version",
                side_effect=PackageNotFoundError,
            ),
            redirect_stdout(stdout),
        ):
            return_code = config_main(["--version"])

        self.assertEqual(return_code, 0)
        self.assertIn("cdf-simulation package: 1.2.0", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
