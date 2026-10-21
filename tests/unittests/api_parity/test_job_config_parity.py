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

"""Parity tests for YAML and Python CDF authoring surfaces."""

from __future__ import annotations

import inspect
import json
import unittest
from collections.abc import Callable
from inspect import Parameter
from pathlib import Path
from typing import TypeVar

import yaml

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi domain
from cdf import (
    CDF_API_VERSION,
    JobConfig,
    CDFJob,
    build_job_from_config,
    load_job,
)
from cdf.execution.job_order_types import JobOrderPlan
from cdf.sections import (
    AnnotationsSection,
    AppearanceSection,
    PostprocessSection,
    SceneSection,
)
from pcb_aoi.config.annotations import AnnotationsConfig, AnnotationsGeometryConfig
from pcb_aoi.config.appearance import (
    AppearanceConfig,
    LightingConfig,
    LightLayerConfig,
    MaterialOverrideConfig,
    ModusLightingConfig,
)
from pcb_aoi.config.capture import (
    CameraConfig,
    CaptureConfig,
    LimitsConfig,
    OutputsConfig,
    RendererConfig,
    RenderPassConfig,
    RoiSelectorConfig,
    RotationConfig,
    ScanGridConfig,
    VisibilityConfig,
)
from pcb_aoi.config.operations import (
    BrightConfig,
    DryContactConfig,
    ExcessConfig,
    HolePlaneConfig,
    InsufficientConfig,
    InterruptedConfig,
    MissingConfig,
    MissingSolderConfig,
    OperationsConfig,
    PinTipConfig,
    PinVariationGroupConfig,
    ReversePolarityConfig,
    ShiftConfig,
    SideflipConfig,
    SolderJointsConfig,
    TombstoneConfig,
)
from pcb_aoi.config.postprocess import (
    CloseOpeningsConfig,
    CropConfig,
    PostprocessConfig,
    StitchConfig,
)
from pcb_aoi.config.scene import ComponentsConfig, SceneConfig
from pcb_aoi.domain import DOMAIN
from pcb_aoi.sections import PcbAoiCaptureSection, PcbAoiOperationsSection
from pcb_aoi.solder_generation import registered_solver_ids
from tests.functional.test_smt_modus_sideflip import (
    build_job as build_smt_modus_sideflip,
)
from tests.functional.test_smt_shift import build_job as build_smt_shift
from tests.functional.test_smt_tombstone import build_job as build_smt_tombstone
from tests.functional.test_tht import build_job as build_tht
from tests.functional.smt_scenario import BOARD_USD as _SMT_BOARD_USD
from tests.functional.test_tht import BOARD_USD as _THT_BOARD_USD

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FUNCTIONAL_SMOKE_CONFIG = _REPO_ROOT / "tests" / "functional" / "job_config_smoke.yaml"
_EXAMPLES_DIR = _REPO_ROOT / "pcb_aoi" / "config" / "examples"
_SNAPSHOT_DIR = Path(__file__).resolve().parent
_PYTHON_SURFACE_SNAPSHOT = _SNAPSHOT_DIR / "python_api_surface.json"
_YAML_SURFACE_SNAPSHOT = _SNAPSHOT_DIR / "yaml_api_surface.json"

# A user-visible Python operation belongs here only when YAML parity is
# intentionally deferred. Keep the reason specific and point at the public
# documentation that describes the limitation. An empty mapping means every
# registered PCB operation must have a versioned JobConfig representation.
_OPERATIONS_WITHOUT_YAML: dict[str, str] = {}

# add_* methods that record ENTRIES of a registered aggregate job operation
# rather than standalone DOMAIN.operations. Solder-quality defects are
# per-type blocks that the compiler folds into ONE solder_defects.apply
# order (pose-defect format, aggregate execution) -- the production model
# is deliberate, so the parity rule must understand the mapping instead of
# demanding a domain registration per adder. Value = the owning job
# operation, asserted registered so this table cannot rot.
_JOB_OPERATION_ENTRY_ADDERS: dict[str, str] = {
    "insufficient": "solder_defects",
    "excess": "solder_defects",
    "bright": "solder_defects",
    "interrupted": "solder_defects",
    "missing_solder": "solder_defects",
    "dry_contact": "solder_defects",
}
_TestMethod = TypeVar("_TestMethod", bound=Callable[..., object])


def _covers_operations(
    *operation_names: str,
) -> Callable[[_TestMethod], _TestMethod]:
    """Mark a plan-parity test as covering named operation surfaces."""

    def decorate(test_method: _TestMethod) -> _TestMethod:
        test_method.yaml_parity_operations = frozenset(operation_names)
        return test_method

    return decorate


# ---------------------------------------------------------------------------
# API surface snapshot infrastructure
# ---------------------------------------------------------------------------

# Entire methods excluded from the Python surface snapshot because they are
# execution-path helpers with no YAML authoring equivalent. Reason strings
# are required so exclusions remain reviewable.
_PYTHON_ONLY_METHODS: dict[str, dict[str, str]] = {
    "capture": {
        "render": "execution trigger driven by output_dir or execution config, not a YAML capture field",
        "render_async": "execution trigger driven by output_dir or execution config, not a YAML capture field",
        "render_pass": "context-manager infrastructure for named render-pass authoring",
    },
    "operations": {
        "get_solder_joint_solvers": "introspection helper that returns available solver names, not an authoring call",
    },
}

# Individual parameters excluded from otherwise-YAML-exposed methods because
# they only make sense in a programmatic or execution context.
_PYTHON_ONLY_PARAMS: dict[str, dict[str, dict[str, str]]] = {
    "scene": {
        "configure": {
            "archive_member": "resolved from a ZIP asset at execution time; not user-authored in YAML",
        },
    },
    "annotations": {
        "label_components": {
            "source": "inferred from the scene metadata selection; not expressible in YAML",
        },
    },
    "capture": {
        "scan_grid": {
            "_derived": "internal-only: marks a grid computed by fov() so a "
            "directly authored one can be told apart; YAML expresses the "
            "same conflict through JobConfig's scan_grid/fov exclusion",
        },
    },
    "postprocess": {
        "crop": {
            "source": "selects the post-hoc RenderResult execution path; YAML always uses the declarative path",
            "dry_run": "post-hoc execution flag with no YAML equivalent",
            "runner": "post-hoc execution injection with no YAML equivalent",
        },
    },
}

_SECTION_CLASSES: dict[str, type] = {
    "scene": SceneSection,
    "appearance": AppearanceSection,
    "operations": PcbAoiOperationsSection,
    "annotations": AnnotationsSection,
    # The domain's capture section, as with operations above: pcb-aoi adds a
    # way of stating a capture in its own terms, and the YAML surface it is
    # compared against is the one a pcb-aoi job writes.
    "capture": PcbAoiCaptureSection,
    "postprocess": PostprocessSection,
}

_YAML_SECTION_MODELS: dict[str, list[type]] = {
    "scene": [SceneConfig, ComponentsConfig],
    "appearance": [
        AppearanceConfig,
        LightingConfig,
        LightLayerConfig,
        MaterialOverrideConfig,
        ModusLightingConfig,
    ],
    "operations": [
        OperationsConfig,
        ShiftConfig,
        TombstoneConfig,
        SideflipConfig,
        ReversePolarityConfig,
        MissingConfig,
        SolderJointsConfig,
        HolePlaneConfig,
        PinTipConfig,
        InsufficientConfig,
        ExcessConfig,
        MissingSolderConfig,
        BrightConfig,
        DryContactConfig,
        InterruptedConfig,
        PinVariationGroupConfig,
    ],
    "annotations": [AnnotationsConfig, AnnotationsGeometryConfig],
    "capture": [
        CaptureConfig,
        RenderPassConfig,
        RendererConfig,
        CameraConfig,
        RotationConfig,
        LimitsConfig,
        RoiSelectorConfig,
        OutputsConfig,
        ScanGridConfig,
        VisibilityConfig,
    ],
    "postprocess": [
        PostprocessConfig,
        CropConfig,
        StitchConfig,
        CloseOpeningsConfig,
    ],
}


def _generate_python_surface() -> dict[str, dict[str, list[str]]]:
    """Return the current Python authoring API surface keyed by section and method."""
    surface: dict[str, dict[str, list[str]]] = {}
    for section, cls in _SECTION_CLASSES.items():
        only_methods = _PYTHON_ONLY_METHODS.get(section, {})
        only_params = _PYTHON_ONLY_PARAMS.get(section, {})
        methods: dict[str, list[str]] = {}
        for name, method in sorted(
            inspect.getmembers(cls, predicate=inspect.isfunction)
        ):
            if name.startswith("_") or name in only_methods:
                continue
            sig = inspect.signature(method)
            excluded = only_params.get(name, {})
            params: list[str] = []
            for pname, param in sig.parameters.items():
                if pname == "self" or pname in excluded:
                    continue
                if param.kind == Parameter.VAR_KEYWORD:
                    params.append(f"**{pname}")
                elif param.kind == Parameter.VAR_POSITIONAL:
                    params.append(f"*{pname}")
                else:
                    params.append(pname)
            methods[name] = params
        surface[section] = methods
    return surface


def _generate_yaml_surface() -> dict[str, dict[str, list[str]]]:
    """Return the current YAML config model field surface keyed by section and model."""
    surface: dict[str, dict[str, list[str]]] = {}
    for section, models in _YAML_SECTION_MODELS.items():
        surface[section] = {
            model.__name__: list(model.model_fields)
            for model in sorted(models, key=lambda m: m.__name__)
        }
    return surface


def _intent(plan: JobOrderPlan) -> JobOrderPlan:
    """Return the plan with provenance stripped, leaving only executable intent.

    ``load_job()`` stamps ``metadata["source"]`` (the YAML's path and sha256)
    so a render can name the file it came from. That is a fact about how the
    plan was authored, not about what it does; a plan built in Python has none
    and is not thereby a different plan. Parity is a claim about intent, so the
    comparison drops it. Everything else in ``metadata`` (``api_version``,
    ``domain``, ``random_seed``) stays and must match.
    """
    metadata = {key: value for key, value in plan.metadata.items() if key != "source"}
    return plan.model_copy(update={"metadata": metadata})


def _minimal_capture() -> dict[str, object]:
    """Return the smallest valid YAML capture configuration."""
    return {
        "renderer": {"resolution": [64, 48]},
        "camera": {"rotation": {}},
        "scan_grid": [1, 1],
    }


def _configure_minimal_capture(job: CDFJob, *, image_format: str = "png") -> None:
    """Author the Python equivalent of `_minimal_capture`."""
    job.capture.output_start(0)
    job.capture.renderer(
        mode="pathtracing",
        resolution=(64, 48),
        spp=1,
        total_spp=32,
        max_bounces=4,
    )
    job.capture.camera(side="top", rotation_range={})
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.limits(triggers=1, max_images=-1)
    job.capture.outputs(
        rgb=True,
        image_format=image_format,
        semantic_types=None,
        frame_padding=4,
    )


class JobConfigParityTest(unittest.TestCase):
    """Verify new YAML configs compile identically to equivalent Python calls."""

    def test_every_registered_operation_has_yaml_and_semantic_parity(self) -> None:
        """Keep domain, model, schema, and behavioral parity coverage aligned.

        This is the coverage guard around the exact-plan tests below. Without
        it, adding a Python operation but forgetting its YAML surface adds no
        failing parity case because the test suite does not know that the new
        operation exists.
        """
        registered = set(DOMAIN.operations) | set(DOMAIN.job_operations)
        yaml_fields = set(OperationsConfig.model_fields)
        exceptions = set(_OPERATIONS_WITHOUT_YAML)

        # Derive the public Python API surface directly from the section class so
        # that a method added without a matching domain registration is caught here
        # rather than silently passing the registered-vs-yaml guard below.
        section_methods = {
            name
            for name, _ in inspect.getmembers(
                PcbAoiOperationsSection, predicate=inspect.isfunction
            )
            if not name.startswith("_")
        }
        # add_* methods follow the "add_{operation_name}" convention -- except
        # entry adders, which feed an aggregate job operation (see
        # _JOB_OPERATION_ENTRY_ADDERS) and are reconciled separately below.
        api_operations = {
            name[4:] for name in section_methods if name.startswith("add_")
        } - set(_JOB_OPERATION_ENTRY_ADDERS)
        for entry, owner in _JOB_OPERATION_ENTRY_ADDERS.items():
            self.assertIn(
                owner,
                DOMAIN.job_operations,
                f"entry adder add_{entry} names owner {owner!r}, which is not "
                "a registered job operation -- the escape hatch has rotted",
            )
            self.assertIn(
                f"add_{entry}",
                section_methods,
                f"_JOB_OPERATION_ENTRY_ADDERS lists {entry!r} but the section "
                "has no such adder -- remove the stale entry",
            )
        # Job operations are exposed as same-named methods (e.g. solder_joints).
        api_job_operations = section_methods & set(DOMAIN.job_operations)

        self.assertEqual(
            api_operations,
            set(DOMAIN.operations),
            "PcbAoiOperationsSection add_* methods must match DOMAIN.operations — "
            "add the missing entry to the domain frozenset or remove the stale method",
        )
        aggregate_owners = set(_JOB_OPERATION_ENTRY_ADDERS.values())
        self.assertEqual(
            api_job_operations | aggregate_owners,
            set(DOMAIN.job_operations) | aggregate_owners,
            "PcbAoiOperationsSection must have a same-named method for every "
            "DOMAIN.job_operations entry (aggregates are exposed through "
            "their entry adders instead)",
        )

        undocumented = {
            name
            for name, reason in _OPERATIONS_WITHOUT_YAML.items()
            if not reason.strip()
        }
        self.assertEqual(
            undocumented,
            set(),
            "operations without YAML need a documented limitation",
        )
        self.assertEqual(
            yaml_fields & exceptions,
            set(),
            "an operation cannot have YAML and a no-YAML exception",
        )
        # An aggregate job operation is expressed in YAML through its per-entry
        # blocks, so swap the owner for its entries before reconciling.
        self.assertEqual(
            (registered - aggregate_owners) | set(_JOB_OPERATION_ENTRY_ADDERS),
            yaml_fields | exceptions,
            "every registered PCB operation and job-operation entry adder "
            "must have a JobConfig field or an explicit documented exception",
        )

        schema = JobConfig.model_json_schema()
        schema_fields = set(
            schema["$defs"]["pcb-aoi.OperationsConfig"]["properties"],
        )
        self.assertEqual(
            schema_fields,
            yaml_fields,
            "generated JobConfig schema must expose every operation model field",
        )

        covered = set()
        for attribute_name in dir(type(self)):
            test_method = getattr(type(self), attribute_name)
            covered.update(
                getattr(test_method, "yaml_parity_operations", ()),
            )
        self.assertEqual(
            covered,
            yaml_fields,
            "every YAML operation needs an exact JobOrderPlan parity test",
        )

    def assert_plan_parity(
        self,
        payload: dict[str, object],
        python_job: CDFJob,
        *,
        base_dir: Path,
        output_dir: Path,
    ) -> None:
        """Assert exact normalized `JobOrderPlan` equality."""
        config = JobConfig.model_validate(payload)
        yaml_job = build_job_from_config(config, base_dir=base_dir)

        self.assertEqual(
            _intent(yaml_job.compile_orders(output_dir=output_dir)),
            _intent(python_job.compile_orders(output_dir=output_dir)),
        )

    def test_minimal_config_defaults_match_python_defaults(self) -> None:
        base_dir = Path("fixtures/minimal").resolve()
        output_dir = Path("output/minimal")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi", random_seed=0)
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    def test_scan_grid_object_form_matches_python_scan_grid(self) -> None:
        """The structured scan_grid exposes every public scan_grid field."""
        base_dir = Path("fixtures/minimal").resolve()
        output_dir = Path("output/minimal")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "capture": {
                **_minimal_capture(),
                "scan_grid": {
                    "x_count": 2,
                    "y_count": 3,
                    "scan_area_mm": [120.0, 80.0],
                    "centered_on": "panel",
                    "image_size_px": [2834, 1889],
                },
            },
        }
        python_job = CDFJob(domain="pcb-aoi", random_seed=0)
        python_job.scene.configure(usd_path=base_dir / "board.usdc", root_path="/Board")
        python_job.capture.output_start(0)
        python_job.capture.renderer(
            mode="pathtracing",
            resolution=(64, 48),
            spp=1,
            total_spp=32,
            max_bounces=4,
        )
        python_job.capture.camera(side="top", rotation_range={})
        python_job.capture.scan_grid(
            x_count=2,
            y_count=3,
            scan_area_mm=(120.0, 80.0),
            centered_on="panel",
            image_size_px=(2834, 1889),
        )
        python_job.capture.limits(triggers=1, max_images=-1)
        python_job.capture.outputs(
            rgb=True,
            image_format="png",
            semantic_types=None,
            frame_padding=4,
        )
        self.assert_plan_parity(
            payload, python_job, base_dir=base_dir, output_dir=output_dir
        )

    def test_scan_grid_object_form_placed_by_center_matches_python(self) -> None:
        """``scan_center_mm`` in the object form compiles like the Python call."""
        base_dir = Path("fixtures/minimal").resolve()
        output_dir = Path("output/minimal")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "capture": {
                **_minimal_capture(),
                "scan_grid": {
                    "x_count": 2,
                    "y_count": 1,
                    "scan_area_mm": [120.0, 40.0],
                    "scan_center_mm": [200.0, 72.0],
                    "image_size_px": [5669, 1889],
                },
            },
        }
        python_job = CDFJob(domain="pcb-aoi", random_seed=0)
        python_job.scene.configure(usd_path=base_dir / "board.usdc", root_path="/Board")
        python_job.capture.output_start(0)
        python_job.capture.renderer(
            mode="pathtracing",
            resolution=(64, 48),
            spp=1,
            total_spp=32,
            max_bounces=4,
        )
        python_job.capture.camera(side="top", rotation_range={})
        python_job.capture.scan_grid(
            x_count=2,
            y_count=1,
            scan_area_mm=(120.0, 40.0),
            scan_center_mm=(200.0, 72.0),
            image_size_px=(5669, 1889),
        )
        python_job.capture.limits(triggers=1, max_images=-1)
        python_job.capture.outputs(
            rgb=True,
            image_format="png",
            semantic_types=None,
            frame_padding=4,
        )
        self.assert_plan_parity(
            payload, python_job, base_dir=base_dir, output_dir=output_dir
        )

    def test_scan_grid_object_form_refuses_what_scan_grid_refuses(self) -> None:
        """A center needs a rectangle and excludes a label, as in Python."""
        for grid in (
            {"x_count": 1, "y_count": 1, "scan_center_mm": [1.0, 2.0]},
            {
                "x_count": 1,
                "y_count": 1,
                "scan_area_mm": [5.0, 5.0],
                "scan_center_mm": [1.0, 2.0],
                "centered_on": "panel",
            },
        ):
            with (
                self.subTest(grid=grid),
                self.assertRaisesRegex(ValueError, "scan_center_mm"),
            ):
                JobConfig.model_validate(
                    {
                        "api_version": CDF_API_VERSION,
                        "domain": "pcb-aoi",
                        "scene": {"usd": "board.usdc", "root": "/Board"},
                        "capture": {**_minimal_capture(), "scan_grid": grid},
                    }
                )

    def test_scan_grid_object_form_with_counts_only_matches_shorthand(self) -> None:
        """``{x_count, y_count}`` and ``[x_count, y_count]`` compile identically."""
        base_dir = Path("fixtures/minimal").resolve()
        output_dir = Path("output/minimal")
        plans = [
            _intent(
                build_job_from_config(
                    JobConfig.model_validate(
                        {
                            "api_version": CDF_API_VERSION,
                            "domain": "pcb-aoi",
                            "scene": {"usd": "board.usdc", "root": "/Board"},
                            "capture": {**_minimal_capture(), "scan_grid": grid},
                        }
                    ),
                    base_dir=base_dir,
                ).compile_orders(output_dir=output_dir)
            )
            for grid in ([2, 3], {"x_count": 2, "y_count": 3})
        ]
        self.assertEqual(plans[0], plans[1])

    def test_image_format_is_accepted_and_rejected_alike_on_both_surfaces(
        self,
    ) -> None:
        base_dir = Path("fixtures/minimal").resolve()
        output_dir = Path("output/minimal")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "capture": {**_minimal_capture(), "outputs": {"image_format": "bmp"}},
        }
        python_job = CDFJob(domain="pcb-aoi", random_seed=0)
        python_job.scene.configure(usd_path=base_dir / "board.usdc", root_path="/Board")
        _configure_minimal_capture(python_job, image_format="bmp")

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

        # A format neither surface can produce fails at declaration on both,
        # naming the accepted values, instead of exiting 0 with no rgb frames.
        payload["capture"]["outputs"]["image_format"] = "tiff"
        with self.assertRaisesRegex(ValueError, "'png'.*'bmp'") as yaml_error:
            JobConfig.model_validate(payload)
        with self.assertRaisesRegex(ValueError, "png, jpeg, jpg, bmp") as python_error:
            CDFJob(domain="pcb-aoi").capture.outputs(image_format="tiff")
        self.assertIn("tiff", str(yaml_error.exception))
        self.assertIn("tiff", str(python_error.exception))

    def test_shared_board_instances_match_python_api(self) -> None:
        base_dir = Path("fixtures/panel").resolve()
        output_dir = Path("output/panel")
        instance_paths = (
            "/World/Panel/StepRepeats/SR_0_0",
            "/World/Panel/StepRepeats/SR_0_1",
        )
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {
                "usd": "panel.usda",
                "root": "/World/Panel",
                "instances": list(instance_paths),
            },
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "panel.usda",
            root_path="/World/Panel",
        )
        python_job.scene.instances(prim_paths=instance_paths)
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    def test_embedded_usd_component_selection_matches_python_api(self) -> None:
        base_dir = Path("fixtures/usd-components").resolve()
        output_dir = Path("output/usd-components")
        component_types = ("THT_A", "THT_B")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {
                "usd": "board.usda",
                "root": "/World/Board",
                "components": {"types": list(component_types)},
            },
            "operations": {
                "solder_joints": {
                    "solver": "procedural",
                    "component_types": list(component_types),
                },
            },
            "annotations": {
                "components": True,
                "component_classes": {"connector": ["J1"]},
            },
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usda",
            root_path="/World/Board",
        )
        python_job.scene.select_components(
            source="usd",
            types=component_types,
        )
        python_job.operations.solder_joints(
            solver="procedural",
            component_types=component_types,
            source="usd",
        )
        python_job.annotations.label_components(
            source="usd",
            classes={"connector": ["J1"]},
        )
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    def test_component_filters_match_python_api(self) -> None:
        """Exercise normalized filters on every non-simulated authoring boundary."""
        base_dir = Path("fixtures/component-filters").resolve()
        output_dir = Path("output/component-filters")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {
                "usd": "board.usda",
                "root": "/World/Board",
                "components": {
                    "component_filter": {
                        "side": ["Top"],
                        "process_classes": ["SMT"],
                    },
                },
            },
            "operations": {
                "shift": {
                    "count": 1,
                    "translate": 0.2,
                    "rotate_z": 5.0,
                    "component_filter": {"package_type": ["0603"]},
                },
                "tombstone": {
                    "count": 1,
                    "angle": [70.0, 90.0],
                    "component_filter": {"component_type": ["capacitor"]},
                },
                "sideflip": {
                    "count": 1,
                    "angle": [60.0, 80.0],
                    "component_filter": {"footprint_id": ["RES_0603"]},
                },
                "reverse_polarity": {
                    "count": 1,
                    "component_filter": {"refdes": ["D1"]},
                },
                "pin_variation": [
                    {
                        "component_filter": {"process_class": ["through_hole"]},
                        "tilt": [0.0, 5.0],
                    },
                ],
                "solder_joints": {
                    "solver": "procedural",
                    "component_filter": {"process_classes": ["SMT"]},
                },
                "hole_plane": {
                    "side": "bottom",
                    "component_filter": {"package_type": ["TO92"]},
                },
                "pin_tip": {
                    "side": "bottom",
                    "component_filter": {"refdes": ["J1"]},
                },
            },
            "annotations": {
                "geometry": {
                    "pads": "pad",
                    "component_filter": {"component_type": ["resistor"]},
                },
            },
            "capture": {
                **_minimal_capture(),
                "roi_selection": [
                    {
                        "component_filter": {"footprint_id": ["RES_0603"]},
                        "count": 2,
                    },
                ],
            },
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usda",
            root_path="/World/Board",
        )
        python_job.scene.select_components(
            source="usd",
            component_filter={
                "side": ["Top"],
                "process_classes": ["SMT"],
            },
        )
        python_job.operations.add_shift(
            count=1,
            translate_range=0.2,
            rotate_z_range=5.0,
            component_filter={"package_type": ["0603"]},
        )
        python_job.operations.add_tombstone(
            count=1,
            angle_range=(70.0, 90.0),
            component_filter={"component_type": ["capacitor"]},
            eligibility_source="usd",
        )
        python_job.operations.add_sideflip(
            count=1,
            angle_range=(60.0, 80.0),
            component_filter={"footprint_id": ["RES_0603"]},
        )
        python_job.operations.add_reverse_polarity(
            count=1,
            component_filter={"refdes": ["D1"]},
        )
        python_job.operations.pin_variation(
            component_filter={"process_class": ["through_hole"]},
            tilt_range=(0.0, 5.0),
        )
        python_job.operations.solder_joints(
            solver="procedural",
            component_filter={"process_classes": ["SMT"]},
            source="usd",
            solver_options={},
            output_root_path=None,
            serialization=None,
            serialization_path=None,
        )
        python_job.operations.hole_plane(
            side="bottom",
            component_filter={"package_type": ["TO92"]},
        )
        python_job.operations.pin_tip(
            side="bottom",
            component_filter={"refdes": ["J1"]},
        )
        python_job.annotations.label_geometry(
            pads="pad",
            component_filter={"component_type": ["resistor"]},
        )
        python_job.capture.output_start(0)
        python_job.capture.renderer(
            mode="pathtracing",
            resolution=(64, 48),
            spp=1,
            total_spp=32,
            max_bounces=4,
        )
        python_job.capture.camera(side="top", rotation_range={})
        python_job.capture.scan_grid(x_count=1, y_count=1)
        python_job.capture.roi_selection(
            component_filter={"footprint_id": ["RES_0603"]},
            count=2,
        )
        python_job.capture.limits(triggers=1, max_images=-1)
        python_job.capture.outputs(
            rgb=True,
            image_format="png",
            semantic_types=None,
            frame_padding=4,
        )

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    def test_final_stage_serialization_matches_python_execution_api(self) -> None:
        base_dir = Path("fixtures/serialization").resolve()
        serialization_path = base_dir / "generated/final_stage.usda"
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "execution": {
                "serialization_path": "generated/final_stage.usda",
            },
        }

        config = JobConfig.model_validate(payload)
        yaml_job = build_job_from_config(config, base_dir=base_dir)
        python_job = CDFJob(domain="pcb-aoi", random_seed=0)
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )

        self.assertEqual(
            _intent(yaml_job.compile_orders(serialization_path=serialization_path)),
            _intent(python_job.compile_orders(serialization_path=serialization_path)),
        )
        self.assertEqual(
            config.execution.serialization_path,
            "generated/final_stage.usda",
        )

    def test_named_render_passes_match_python_api(self) -> None:
        base_dir = Path("fixtures/render-passes").resolve()
        output_dir = Path("output/render-passes")
        pass_config = {
            "renderer": {"resolution": [64, 48]},
            "camera": {"side": "top", "rotation": {}},
            "scan_grid": [1, 1],
        }
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "capture": {
                "passes": {
                    "full_board": pass_config,
                    "component_rois": {
                        **pass_config,
                        "roi_selection": [
                            {
                                "refdes": ["J1", "J2"],
                                "padding": 0.25,
                            },
                            {
                                "pins": [["J1", "1"], ["J1", "2"]],
                                "padding": 0.1,
                            },
                            {
                                "component_types": ["CAPACITOR"],
                                "count": 2,
                            },
                            {
                                "process_classes": ["through_hole"],
                                "focus": "pin",
                            },
                            {
                                "all_components": True,
                                "focus": "pin",
                            },
                            {
                                "defects": ["shift", "tombstone"],
                                "padding": 0.75,
                            },
                        ],
                    },
                },
            },
        }

        python_job = CDFJob(domain="pcb-aoi", random_seed=0)
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        for name, targets in (("full_board", None), ("component_rois", ("J1", "J2"))):
            with python_job.capture.render_pass(name) as render_pass:
                render_pass.output_start(0)
                render_pass.renderer(
                    mode="pathtracing",
                    resolution=(64, 48),
                    spp=1,
                    total_spp=32,
                    max_bounces=4,
                )
                render_pass.camera(side="top", rotation_range={})
                render_pass.scan_grid(x_count=1, y_count=1)
                if targets is not None:
                    render_pass.roi_selection(refdes=targets, padding=0.25)
                    render_pass.roi_selection(
                        pins=(("J1", "1"), ("J1", "2")),
                        padding=0.1,
                    )
                    render_pass.roi_selection(
                        component_types=("CAPACITOR",),
                        count=2,
                    )
                    render_pass.roi_selection(
                        process_classes=("through_hole",),
                        focus="pin",
                    )
                    render_pass.roi_selection(
                        all_components=True,
                        focus="pin",
                    )
                    render_pass.roi_selection(
                        defects=("shift", "tombstone"),
                        padding=0.75,
                    )
                render_pass.limits(triggers=1, max_images=-1)
                render_pass.outputs(
                    rgb=True,
                    image_format="png",
                    semantic_types=None,
                    frame_padding=4,
                )

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    @_covers_operations(
        "shift",
        "tombstone",
        "sideflip",
        "reverse_polarity",
        "solder_joints",
    )
    def test_complete_transform_job_matches_python_api(self) -> None:
        base_dir = Path("fixtures/complete").resolve()
        output_dir = Path("output/complete")
        component_types = ["PASSIVE_SMD_0805", "IC_SMD"]
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "random_seed": 17,
            "scene": {
                "usd": "board.usdc",
                "root": "/Board",
                "camera": "/World/Camera",
                "camera_xform": "/World/CameraRig",
                "ring_light": "/World/RingLight",
                "horizontal_aperture": 6.5,
                "components": {
                    "sidecar": "board.components.json",
                    "types": component_types,
                },
            },
            "appearance": {
                "lighting": {
                    "mode": "rgb_light",
                    "preserve_scene_light_color": True,
                    "exposure": [-0.25, 0.5],
                    "cone_angle": 120,
                    "cone_softness": [0.75, 1.0],
                    "dome_fill": {"intensity": [500, 750]},
                    "layers": {
                        "Red": {
                            "intensity": 6500,
                            "color_r": [0.95, 1.0],
                            "color_g": [0.0, 0.05],
                            "color_b": [0.0, 0.05],
                        },
                    },
                },
                "materials": {
                    "carrier": {
                        "material_path": "/World/Looks/Carrier",
                        "diffuse_texture": "C:/textures/carrier_albedo.png",
                        "roughness_texture": "C:/textures/carrier_roughness.png",
                        "normal_texture": "C:/textures/carrier_normal.png",
                        "diffuse_color": [0.2, 0.2, 0.2],
                        "diffuse_tint": [1.0, 0.3, 0.2],
                        "albedo_brightness": 1.2,
                        "opacity": 0.5,
                        "roughness": 0.55,
                        "roughness_texture_influence": 0.8,
                        "metallic": 0.0,
                        "specular_level": 0.35,
                        "normal_strength": 0.6,
                        "normal_format": "opengl",
                        "texture_scale": [0.006, 0.006],
                        "texture_rotation": 0.0,
                        "project_uvw": True,
                        "world_space_projection": True,
                    },
                },
                "vantablack_components": True,
            },
            "operations": {
                "shift": {"ratio": 0.1, "translate": 0.2, "rotate_z": 15},
                "tombstone": {
                    "ratio": 0.2,
                    "angle": [70, 90],
                    "component_types": ["PASSIVE_SMD_0805"],
                    "auto_filter": False,
                    "eligibility_source": "components",
                },
                "sideflip": {"ratio": 0.3, "angle": [60, 80]},
                "reverse_polarity": {
                    "ratio": 0.4,
                    "component_types": ["IC_SMD"],
                },
                "solder_joints": {
                    "solver": "procedural",
                    "component_types": ["PASSIVE_SMD_0805"],
                    "source": "components",
                    "options": {
                        "components": ["R1", "/Board/Components/Top/C1"],
                        "height_ratio": 0.35,
                        "solid": True,
                        "tht_solder_bath": True,
                    },
                    "output_root_path": "/Board/GeneratedSolder",
                    "serialization_path": "generated/solder.usda",
                },
            },
            "annotations": {
                "components": True,
                "component_classes": {"passive": ["R1", "C1"]},
                "operation_labels": "defect",
            },
            "capture": {
                "output_start": 3,
                "renderer": {
                    "mode": "pathtracing",
                    "resolution": [320, 180],
                    "spp": 2,
                    "total_spp": 8,
                    "max_bounces": 6,
                },
                "camera": {
                    "side": "bottom",
                    "rotation": {"x": [-1, 1], "y": 2, "z": -90},
                },
                "scan_grid": [2, 3],
                "roi_selection": [
                    {
                        "refdes": ["J1", "J2"],
                        "padding": 0.25,
                    },
                ],
                "limits": {"triggers": 2, "max_images": 5},
                "motion_blur": {
                    "probability": 0.25,
                    "alpha_range": [0.1, 0.2],
                },
                "outputs": {
                    "annotators": [
                        "rgb",
                        "bounding_box_2d_tight",
                        "bounding_box_2d_loose",
                        "bounding_box_3d",
                        "semantic_segmentation",
                        "instance_id_segmentation",
                        "distance_to_camera",
                    ],
                    "image_format": "jpg",
                    "semantic_types": ["class", "defect"],
                    "colorize": [
                        "semantic_segmentation",
                        "instance_id_segmentation",
                    ],
                    "frame_padding": 6,
                },
            },
            "postprocess": {
                "crop": {
                    "method": "bbox_tight",
                    "types": ["rgb", "semantic_segmentation"],
                    "offset": 12,
                    "min_coverage": 0.35,
                    "class_filter": ["shift", "tombstone"],
                    "xform_depth": 8,
                    "output_subdir": "closeups",
                    "mode": "paired",
                    "input_subdir": "defects",
                    "reference_subdir": "golden",
                },
                "stitch": {
                    "source_pass": "default",
                    "image_type": "rgb",
                    "output_subdir": "stitched",
                    "output_filename": "board.png",
                },
            },
        }

        python_job = CDFJob(domain="pcb-aoi", random_seed=17)
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
            camera_path="/World/Camera",
            camera_xform_path="/World/CameraRig",
            ring_light_root="/World/RingLight",
            horizontal_aperture=6.5,
        )
        python_job.scene.attach_sidecar(
            name="components",
            kind="pcb.components",
            path=base_dir / "board.components.json",
        )
        python_job.scene.select_components(types=component_types)
        python_job.appearance.lighting(
            mode="rgb_light",
            preserve_scene_light_color=True,
            exposure_range=(-0.25, 0.5),
            cone_angle_range=(120, 120),
            cone_softness_range=(0.75, 1.0),
            dome_fill={"intensity": (500, 750)},
            layers={
                "Red": {
                    "intensity": (6500, 6500),
                    "color_r": (0.95, 1.0),
                    "color_g": (0.0, 0.05),
                    "color_b": (0.0, 0.05),
                },
            },
        )
        python_job.appearance.material_override(
            "carrier",
            material_path="/World/Looks/Carrier",
            diffuse_texture="C:/textures/carrier_albedo.png",
            roughness_texture="C:/textures/carrier_roughness.png",
            normal_texture="C:/textures/carrier_normal.png",
            diffuse_color=(0.2, 0.2, 0.2),
            diffuse_tint=(1.0, 0.3, 0.2),
            albedo_brightness=1.2,
            opacity=0.5,
            roughness=0.55,
            roughness_texture_influence=0.8,
            metallic=0.0,
            specular_level=0.35,
            normal_strength=0.6,
            normal_format="opengl",
            texture_scale=(0.006, 0.006),
            texture_rotation=0.0,
            project_uvw=True,
            world_space_projection=True,
        )
        python_job.appearance.components(vantablack=True)
        python_job.operations.add_shift(
            ratio=0.1,
            translate_range=0.2,
            rotate_z_range=15,
        )
        python_job.operations.add_tombstone(
            ratio=0.2,
            angle_range=(70, 90),
            component_types=["PASSIVE_SMD_0805"],
            auto_filter=False,
            eligibility_source="components",
        )
        python_job.operations.add_sideflip(ratio=0.3, angle_range=(60, 80))
        python_job.operations.add_reverse_polarity(
            ratio=0.4,
            component_types=["IC_SMD"],
        )
        python_job.operations.solder_joints(
            solver="procedural",
            component_types=["PASSIVE_SMD_0805"],
            source="components",
            solver_options={
                "components": ["R1", "/Board/Components/Top/C1"],
                "height_ratio": 0.35,
                "solid": True,
                "tht_solder_bath": True,
            },
            output_root_path="/Board/GeneratedSolder",
            serialization_path=str(base_dir / "generated/solder.usda"),
        )
        python_job.annotations.label_components(
            source="components",
            classes={"passive": ["R1", "C1"]},
        )
        python_job.annotations.enable_operation_labels(namespace="defect")
        python_job.capture.output_start(3)
        python_job.capture.renderer(
            mode="pathtracing",
            resolution=(320, 180),
            spp=2,
            total_spp=8,
            max_bounces=6,
        )
        python_job.capture.camera(
            side="bottom",
            rotation_range={"x": (-1, 1), "y": (2, 2), "z": -90},
        )
        python_job.capture.scan_grid(x_count=2, y_count=3)
        python_job.capture.roi_selection(refdes=("J1", "J2"), padding=0.25)
        python_job.capture.limits(triggers=2, max_images=5)
        python_job.capture.augmentation(
            motion_blur={"probability": 0.25, "alpha_range": (0.1, 0.2)},
        )
        python_job.capture.outputs(
            rgb=True,
            image_format="jpg",
            bounding_box_2d_tight=True,
            bounding_box_2d_loose=True,
            bounding_box_3d=True,
            semantic_segmentation=True,
            colorize_semantic_segmentation=True,
            instance_id_segmentation=True,
            colorize_instance_id_segmentation=True,
            semantic_types=["class", "defect"],
            frame_padding=6,
            distance_to_camera=True,
        )
        python_job.postprocess.crop(
            method="bbox_tight",
            types=["rgb", "semantic_segmentation"],
            offset=12,
            min_coverage=0.35,
            class_filter=["shift", "tombstone"],
            xform_depth=8,
            output_subdir="closeups",
            mode="paired",
            input_subdir="defects",
            reference_subdir="golden",
        )
        python_job.postprocess.stitch(
            source_pass="default",
            image_type="rgb",
            output_subdir="stitched",
            output_filename="board.png",
        )

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    def test_implicit_component_sources_match_python_api(self) -> None:
        base_dir = Path("fixtures/implicit-sources").resolve()
        output_dir = Path("output/implicit-sources")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {
                "usd": "board.usdc",
                "root": "/Board",
                "components": {"sidecar": "board.components.json"},
            },
            "operations": {
                "tombstone": {"ratio": 0.25, "angle": [70, 90]},
                "solder_joints": {"solver": "procedural"},
            },
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        python_job.scene.attach_sidecar(
            name="components",
            kind="pcb.components",
            path=base_dir / "board.components.json",
        )
        python_job.scene.select_components(source="components")
        python_job.operations.add_tombstone(
            ratio=0.25,
            angle_range=(70, 90),
            component_types=None,
            auto_filter=True,
            eligibility_source="components",
        )
        python_job.operations.solder_joints(
            solver="procedural",
            component_types=None,
            source="components",
            solver_options={},
            output_root_path=None,
            serialization="none",
            serialization_path=None,
        )
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    @_covers_operations("missing")
    def test_missing_job_matches_python_api(self) -> None:
        base_dir = Path("fixtures/missing").resolve()
        output_dir = Path("output/missing")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {
                "usd": "board.usdc",
                "root": "/Board",
                "components": {
                    "sidecar": "board.components.json",
                    "types": ["PASSIVE_SMD_0805"],
                },
            },
            "operations": {
                "missing": {
                    "ratio": 0.5,
                    "component_filter": {"refdes": ["U1"]},
                },
            },
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        python_job.scene.attach_sidecar(
            name="components",
            kind="pcb.components",
            path=base_dir / "board.components.json",
        )
        python_job.scene.select_components(types=["PASSIVE_SMD_0805"])
        python_job.operations.add_missing(
            ratio=0.5,
            component_filter={"refdes": ["U1"]},
        )
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    @_covers_operations("pin_variation")
    def test_pin_variation_job_matches_python_api(self) -> None:
        """Both surfaces must reach the same plan, not merely the same spec.

        Two groups rather than one, because the YAML is a sequence and the
        Python API accumulates repeated calls: a single group would not
        exercise the part where the two shapes could diverge. The selectors
        differ between the groups on purpose -- `package_types` and
        `components` are mutually exclusive, so both need covering.
        """
        base_dir = Path("fixtures/pin_variation").resolve()
        output_dir = Path("output/pin_variation")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "operations": {
                "pin_variation": [
                    {
                        "package_types": ["TO92"],
                        "tilt": [0.0, 15.0],
                        "height": [0.0, 1.5],
                    },
                    {
                        "components": ["C1001", "C1002"],
                        "tilt": [0.0, 4.0],
                        "cut_angle": [0.0, 60.0],
                    },
                ],
            },
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        python_job.operations.pin_variation(
            package_types=("TO92",),
            tilt_range=(0.0, 15.0),
            height_variation=(0.0, 1.5),
        )
        python_job.operations.pin_variation(
            components=("C1001", "C1002"),
            tilt_range=(0.0, 4.0),
            cut_angle=(0.0, 60.0),
        )
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    @_covers_operations("hole_plane")
    def test_hole_plane_job_matches_python_api(self) -> None:
        """Both authoring surfaces must compile to the same hole_plane.generate order."""
        base_dir = Path("fixtures/hole_plane").resolve()
        output_dir = Path("output/hole_plane")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "operations": {
                "hole_plane": {
                    "class_name": "hole",
                    "side": "bottom",
                    "z_offset_mm": 0.1,
                },
            },
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        python_job.operations.hole_plane(
            class_name="hole",
            side="bottom",
            z_offset_mm=0.1,
        )
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    @_covers_operations("pin_tip")
    def test_pin_tip_job_matches_python_api(self) -> None:
        """Both surfaces must compile the same annotation-proxy operation."""
        base_dir = Path("fixtures/pin_tip").resolve()
        output_dir = Path("output/pin_tip")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "operations": {
                "pin_tip": {
                    "side": "bottom",
                    "class_name": "pin_tip",
                    "offset_mm": 0.003,
                    "components": ["X204", "X205"],
                },
            },
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        python_job.operations.pin_tip(
            side="bottom",
            class_name="pin_tip",
            offset_mm=0.003,
            components=("X204", "X205"),
        )
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    def test_hole_plane_with_components_matches_python_api(self) -> None:
        """Both surfaces must produce the same plan when a components filter is set."""
        base_dir = Path("fixtures/hole_plane").resolve()
        output_dir = Path("output/hole_plane")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "operations": {
                "hole_plane": {
                    "side": "bottom",
                    "components": ["C1", "C2"],
                },
            },
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        python_job.operations.hole_plane(
            side="bottom",
            components=["C1", "C2"],
        )
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    def test_geometry_annotations_job_matches_python_api(self) -> None:
        """Both surfaces must compile to the same annotations.geometry.apply order."""
        base_dir = Path("fixtures/hole_plane").resolve()
        output_dir = Path("output/hole_plane")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "annotations": {
                "geometry": {
                    "solder": "solder",
                    "pads": "pad",
                    "components": ["C1", "C2"],
                },
            },
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        python_job.annotations.label_geometry(
            solder="solder",
            pads="pad",
            components=["C1", "C2"],
        )
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    def test_geometry_annotations_barrels_matches_python_api(self) -> None:
        """Barrel class must round-trip through YAML and Python to the same plan."""
        base_dir = Path("fixtures/hole_plane").resolve()
        output_dir = Path("output/hole_plane")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "annotations": {
                "geometry": {
                    "pads": "pad",
                    "barrels": "pad",
                    "components": ["C1", "C2"],
                },
            },
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        python_job.annotations.label_geometry(
            pads="pad",
            barrels="pad",
            components=["C1", "C2"],
        )
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    def test_geometry_annotations_package_meshes_match_python_api(self) -> None:
        """Package body and marking must reach the same plan from either surface.

        The API-surface snapshot proves only that the field names exist. It
        cannot see whether ``apply_annotations()`` forwards their values, so a
        typo that dropped ``marking`` on one surface would still pass it.
        """
        base_dir = Path("fixtures/hole_plane").resolve()
        output_dir = Path("output/hole_plane")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "annotations": {
                "geometry": {
                    "body": "package",
                    "marking": "polarity",
                    "components": ["C1", "C2"],
                },
            },
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        python_job.annotations.label_geometry(
            body="package",
            marking="polarity",
            components=["C1", "C2"],
        )
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    def test_close_openings_job_matches_python_api(self) -> None:
        """Both surfaces must reach the same plan, closure included.

        Two closures rather than one, because the YAML accepts a sequence and
        the Python API accumulates repeated calls -- the shapes could diverge
        only where there is more than one. They read the same stitched mask and
        differ in the class they close, which is the realistic case: one board
        has holes and pads whose rims both need recovering.
        """
        base_dir = Path("fixtures/close_openings").resolve()
        output_dir = Path("output/close_openings")
        stitch = {
            "image_type": "semantic_segmentation",
            "output_subdir": "stitched",
            "output_filename": "mask.png",
        }
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {"usd": "board.usdc", "root": "/Board"},
            "capture": _minimal_capture(),
            "postprocess": {
                "stitch": stitch,
                "close_openings": [
                    {
                        "class_name": "hole",
                        "output_subdir": "stitched",
                        "output_filename": "mask.png",
                    },
                    {
                        "class_name": "pad",
                        "output_subdir": "stitched",
                        "output_filename": "mask.png",
                        "max_diameter_px": 24,
                        "min_circularity": 0.7,
                    },
                ],
            },
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        _configure_minimal_capture(python_job)
        python_job.postprocess.stitch(**stitch)
        python_job.postprocess.close_openings(
            class_name="hole",
            output_subdir="stitched",
            output_filename="mask.png",
        )
        python_job.postprocess.close_openings(
            class_name="pad",
            output_subdir="stitched",
            output_filename="mask.png",
            max_diameter_px=24,
            min_circularity=0.7,
        )

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    @_covers_operations(
        "insufficient",
        "excess",
        "bright",
        "interrupted",
        "missing_solder",
        "dry_contact",
    )
    @unittest.skipUnless(
        "surface_evolver" in registered_solver_ids(),
        "solder defects require the simulated solver, which registers only "
        "where its geometry deps import (not in the starved CI images); "
        "the @_covers_operations attribute still satisfies the coverage "
        "guard when skipped",
    )
    def test_solder_defect_job_matches_python_api(self) -> None:
        """The four per-type YAML blocks compile to the add_* declarations.

        Solder-quality defects are ENTRY ADDERS of the solder_defects
        aggregate: each block records one declaration and the compiler folds
        them into a single solder_defects.apply order. The YAML and Python
        surfaces must produce that plan identically.
        """
        base_dir = Path("fixtures/solder_defects").resolve()
        output_dir = Path("output/solder_defects")
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "scene": {
                "usd": "board.usdc",
                "root": "/Board",
                "components": {
                    "sidecar": "board.components.json",
                    "types": ["PASSIVE_SMD_0805"],
                },
            },
            "operations": {
                "solder_joints": {
                    "solver": "surface_evolver",
                    "source": "components",
                },
                "insufficient": {
                    "ratio": 0.05,
                    "volume_ratio": [0.3, 0.7],
                    "lead_climb": [0.25, 0.5],
                    "fullness": [0.4, 0.7],
                    "component_filter": {"refdes": ["R1"]},
                },
                "excess": {
                    "ratio": 0.02,
                    "volume_ratio": 2.0,
                    "lead_climb": [1.05, 1.35],
                    "fullness": 1.15,
                    "component_filter": {"refdes": ["R2"]},
                },
                "bright": {
                    "ratio": 0.03,
                    "lead_climb": [0.6, 0.9],
                    "fullness": [0.35, 0.6],
                    "component_filter": {"refdes": ["J1"]},
                },
                "interrupted": {
                    "ratio": 0.03,
                    "pad_exposure": [0.2, 0.5],
                    "component_filter": {"refdes": ["R3"]},
                },
                "missing_solder": {
                    "ratio": 0.02,
                    "component_filter": {"refdes": ["R4"]},
                },
                "dry_contact": {
                    "ratio": 1.0,
                    "gap_fraction": [0.25, 0.5],
                    "component_filter": {"package_type": ["TO254"]},
                },
            },
            "capture": _minimal_capture(),
        }

        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(
            usd_path=base_dir / "board.usdc",
            root_path="/Board",
        )
        python_job.scene.attach_sidecar(
            name="components",
            kind="pcb.components",
            path=base_dir / "board.components.json",
        )
        python_job.scene.select_components(types=["PASSIVE_SMD_0805"])
        python_job.operations.solder_joints(
            solver="surface_evolver",
            source="components",
            # the YAML adapter forwards the scene selection types
            component_types=["PASSIVE_SMD_0805"],
        )
        python_job.operations.add_insufficient(
            ratio=0.05,
            volume_ratio=[0.3, 0.7],
            lead_climb=[0.25, 0.5],
            fullness=[0.4, 0.7],
            component_filter={"refdes": ["R1"]},
        )
        python_job.operations.add_excess(
            ratio=0.02,
            volume_ratio=2.0,
            lead_climb=[1.05, 1.35],
            fullness=1.15,
            component_filter={"refdes": ["R2"]},
        )
        python_job.operations.add_bright(
            ratio=0.03,
            lead_climb=[0.6, 0.9],
            fullness=[0.35, 0.6],
            component_filter={"refdes": ["J1"]},
        )
        python_job.operations.add_interrupted(
            ratio=0.03,
            pad_exposure=[0.2, 0.5],
            component_filter={"refdes": ["R3"]},
        )
        python_job.operations.add_missing_solder(
            ratio=0.02,
            component_filter={"refdes": ["R4"]},
        )
        python_job.operations.add_dry_contact(
            ratio=1.0,
            gap_fraction=[0.25, 0.5],
            component_filter={"package_type": ["TO254"]},
        )
        _configure_minimal_capture(python_job)

        self.assert_plan_parity(
            payload,
            python_job,
            base_dir=base_dir,
            output_dir=output_dir,
        )

    def test_functional_smoke_config_matches_python_api(self) -> None:
        config, yaml_job = load_job(_FUNCTIONAL_SMOKE_CONFIG)
        output_dir = Path("output/functional-smoke")
        board = (
            _FUNCTIONAL_SMOKE_CONFIG.parent
            / "resources"
            / "ARTIX_PINHEADER_J13_coupon.usdc"
        )

        python_job = CDFJob(domain="pcb-aoi", random_seed=11)
        python_job.scene.configure(
            usd_path=board,
            root_path="/Board",
        )
        python_job.appearance.lighting(
            mode="white_light",
            preserve_scene_light_color=False,
            exposure_range=(0.0, 0.0),
            cone_angle_range=(120.0, 120.0),
            cone_softness_range=(1.0, 1.0),
            dome={
                "intensity": (1000.0, 1000.0),
                "visible": True,
            },
        )
        python_job.capture.output_start(0)
        python_job.capture.renderer(
            mode="pathtracing",
            resolution=(64, 64),
            spp=1,
            total_spp=1,
            max_bounces=2,
        )
        python_job.capture.camera(
            side="bottom",
            rotation_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": -90.0},
        )
        python_job.capture.scan_grid(x_count=1, y_count=1)
        python_job.capture.limits(triggers=1, max_images=1)
        python_job.capture.outputs(
            rgb=True,
            image_format="png",
            semantic_types=None,
            frame_padding=4,
        )

        self.assertEqual(config.api_version, CDF_API_VERSION)
        self.assertEqual(
            _intent(yaml_job.compile_orders(output_dir=output_dir)),
            _intent(python_job.compile_orders(output_dir=output_dir)),
        )

    def test_functional_python_scripts_match_yaml_examples(self) -> None:
        # The public release ships no board assets, so the examples carry a
        # placeholder board path the user replaces. Everything else must equal
        # the functional script: compare with the script's board path put in
        # place of the placeholder.
        cases = (
            ("smt_tombstone.yaml", build_smt_tombstone, _SMT_BOARD_USD),
            ("smt_shift.yaml", build_smt_shift, _SMT_BOARD_USD),
            ("smt_modus_sideflip.yaml", build_smt_modus_sideflip, _SMT_BOARD_USD),
            ("tht.yaml", build_tht, _THT_BOARD_USD),
        )

        for filename, build_python_job, board_usd in cases:
            with self.subTest(example=filename):
                path = _EXAMPLES_DIR / filename
                config, _ = load_job(path)
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                payload["scene"]["usd"] = str(board_usd)
                python_job = build_python_job(
                    resolution=(640, 360),
                    total_spp=4,
                )[0]

                self.assert_plan_parity(
                    payload,
                    python_job,
                    base_dir=path.parent,
                    output_dir=Path(config.output_dir),
                )

    def test_api_surface_snapshot(self) -> None:
        """Catch kwarg-level drift across all six authoring sections.

        Compares live Python method signatures and YAML Pydantic model fields
        against committed snapshot files. Any change to either surface — an
        added, removed, or renamed parameter or field — fails this test.

        To update the snapshots after a deliberate API change, run:
            python tests/unittests/api_parity/test_job_config_parity.py --update-snapshot
        """
        _UPDATE_CMD = "  python tests/unittests/api_parity/test_job_config_parity.py --update-snapshot"
        if not _PYTHON_SURFACE_SNAPSHOT.exists() or not _YAML_SURFACE_SNAPSHOT.exists():
            self.fail("Snapshot files are missing. Generate them with:\n" + _UPDATE_CMD)

        self.assertEqual(
            _generate_python_surface(),
            json.loads(_PYTHON_SURFACE_SNAPSHOT.read_text(encoding="utf-8")),
            f"Python API surface has changed — update {_PYTHON_SURFACE_SNAPSHOT.name} with:\n"
            + _UPDATE_CMD,
        )
        self.assertEqual(
            _generate_yaml_surface(),
            json.loads(_YAML_SURFACE_SNAPSHOT.read_text(encoding="utf-8")),
            f"YAML API surface has changed — update {_YAML_SURFACE_SNAPSHOT.name} with:\n"
            + _UPDATE_CMD,
        )


if __name__ == "__main__":
    import sys

    parser = __import__("argparse").ArgumentParser(add_help=False)
    parser.add_argument("--update-snapshot", action="store_true")
    args, remaining = parser.parse_known_args()

    if args.update_snapshot:
        _PYTHON_SURFACE_SNAPSHOT.write_text(
            json.dumps(_generate_python_surface(), indent=2) + "\n",
            encoding="utf-8",
        )
        _YAML_SURFACE_SNAPSHOT.write_text(
            json.dumps(_generate_yaml_surface(), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Updated {_PYTHON_SURFACE_SNAPSHOT}")
        print(f"Updated {_YAML_SURFACE_SNAPSHOT}")
    else:
        sys.argv = [sys.argv[0]] + remaining
        unittest.main()
