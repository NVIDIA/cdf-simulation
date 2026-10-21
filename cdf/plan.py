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

"""Job-plan compilation for the declarative execution model.

A Python job and its YAML counterpart resolve to the same normalized job plan.
``JobPlan.flow`` contains execution configuration, while ``JobPlan.pcba``
contains board-scoped configuration. The two dictionaries preserve that
strict boundary for compatibility with configuration-file jobs.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from scripts import file_utils

from .errors import CDFError, ValidationError
from .results import CropResult, RenderResult

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CROP_DIR = _REPO_ROOT / "scripts" / "crop"


def _join_output_location(root: str, child: str) -> str:
    """Join a child path onto a local or remote output location.

    Args:
        root: Base output directory or URI.
        child: Relative output subdirectory.

    Returns:
        Joined local path or remote URI.

    """
    if file_utils.is_remote(root):
        return "{}/{}".format(root.rstrip("/"), child.strip("/"))
    return str(Path(root) / child)


def _run_crop_config(
    cfg: dict[str, object],
    runner: Callable[[dict[str, object]], object] | None = None,
) -> None:
    """Execute a pipeline-shaped crop configuration.

    Args:
        cfg: Crop configuration using a local output directory.
        runner: Optional injected crop runner.

    """
    if runner is not None:
        runner(cfg)
        return
    if str(_CROP_DIR) not in sys.path:
        sys.path.insert(0, str(_CROP_DIR))
    from pipeline_adapter import run_pipeline_crop

    run_pipeline_crop(cfg)


@dataclass
class JobPlan:
    """Normalized declarative representation of the requested work."""

    pipeline_type: str
    flow: dict = field(default_factory=dict)
    pcba: dict = field(default_factory=dict)


def _contains_component_filter(value: object) -> bool:
    """Return whether a nested declaration contains a normalized filter."""
    if isinstance(value, dict):
        return (
            "component_filter" in value and value["component_filter"] is not None
        ) or any(_contains_component_filter(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_component_filter(item) for item in value)
    return False


def compile_plan(job, output_dir=None) -> JobPlan:
    """Validate the job description and normalize it into a :class:`JobPlan`."""
    if any(
        _contains_component_filter(declaration.parameters)
        for declaration in job._declarations
    ):
        raise ValidationError(
            "component_filter is supported by JobOrder execution only; "
            "use compile_orders() or execute() instead of legacy compile()",
        )
    scene, appearance = job.scene, job.appearance
    ops, annotations, capture = job.operations, job.annotations, job.capture

    problems = []
    if scene.config is None:
        problems.append("scene.configure(...) was never called")
    if scene.instance_paths is not None:
        problems.append(
            "scene.instances(...) requires the JobOrder execution path "
            "(compile_orders/execute); the legacy YAML renderer does not "
            "support shared board instances"
        )
    if capture.renderer_spec is None:
        problems.append("capture.renderer(...) was never called")
    if capture.scan_grid_spec is None:
        problems.append("capture.scan_grid(...) was never called")
    if capture.outputs_spec is None:
        problems.append("capture.outputs(...) was never called")
    if capture.camera_spec is None:
        # the pipeline reads CFG["camera_rotation"] unconditionally — a
        # missing .camera(...) call must fail here, not as a KeyError
        # inside Kit after a full (possibly GPU-expensive) launch
        problems.append("capture.camera(...) was never called")
    solder_spec = getattr(ops, "solder_joints_spec", None)
    if solder_spec:
        problems.append(
            "operations.solder_joints(solver={!r}) requires the JobOrder execution "
            "path (compile_orders/render/render_async); the legacy YAML renderer does "
            "not run implicit solder-generation orders".format(solder_spec["solver"])
        )

    tomb = ops.ops.get("tombstone")
    if tomb:
        source = tomb.get("eligibility_source")
        if source is not None and source != "usd":
            sidecar = scene.sidecars.get(source)
            if sidecar is None or sidecar["kind"] != "pcb.components":
                problems.append(
                    f"tombstone eligibility_source {source!r} needs a pcb.components "
                    "sidecar attached via scene.attach_sidecar"
                )
    try:
        component_types = _selected_component_types(scene)
    except ValidationError as exc:
        problems.append(str(exc))
        component_types = None
    if problems:
        raise ValidationError("invalid job description:\n- " + "\n- ".join(problems))

    defect_ops = {k: v for k, v in ops.ops.items() if k != "missing"}
    if "missing" in ops.ops:
        pipeline_type = "missing"
    elif defect_ops:
        pipeline_type = "defect"
    else:
        pipeline_type = "good"

    flow = {"pipeline_type": pipeline_type}

    # The whole lighting description -- selector, policy and payload -- lives
    # in the `lighting:` block; nothing about lighting sits at the flow's top
    # level any more.
    lighting = appearance.lighting_spec or {}
    if appearance.components_spec is not None:
        flow["vantablack_components"] = appearance.components_spec["vantablack"]

    for sidecar in scene.sidecars.values():
        key = {"pcb.components": "components_json", "pcb.sim_manifest": "sim_manifest"}[
            sidecar["kind"]
        ]
        flow[key] = sidecar["path"]

    flow["seed"] = capture.start
    flow["random_seed"] = job.random_seed
    flow["max_image_count"] = capture.limits_spec["max_images"]
    flow["num_triggers"] = capture.limits_spec["triggers"]
    if output_dir is not None:
        flow["output"] = str(output_dir)
    flow["resolution"] = capture.renderer_spec["resolution"]
    flow["pathtracing"] = {
        "spp": capture.renderer_spec["spp"],
        "total_spp": capture.renderer_spec["total_spp"],
        "max_bounces": capture.renderer_spec["max_bounces"],
    }
    flow["scan_grid"] = dict(capture.scan_grid_spec)

    if defect_ops:
        # the pipeline indexes all four blocks; undeclared ops are emitted
        # disabled, matching the reference configs
        defects = {}
        for name in ("shift", "tombstone", "sideflip", "reverse_polarity"):
            params = defect_ops.get(name)
            if params is None:
                defects[name] = {"enabled": False}
                continue
            block = {"enabled": True}
            block.update({k: v for k, v in params.items() if k != "eligibility_source"})
            defects[name] = block
        defects.update(
            {
                k: dict(v, enabled=True)
                for k, v in defect_ops.items()
                if k not in defects
            }
        )
        flow["defects"] = defects
    if "missing" in ops.ops:
        flow["missing"] = dict(ops.ops["missing"])

    if lighting:
        flow["lighting"] = dict(lighting)

    if capture.camera_spec is not None:
        flow["camera_rotation"] = dict(capture.camera_spec)
        flow["camera_side"] = capture.camera_side
    augmentation = dict(capture.augmentation_spec or {})
    if "motion_blur" not in augmentation:
        # required schema: the pipeline reads augmentation.motion_blur
        # unconditionally; probability 0 disables it
        augmentation["motion_blur"] = {
            "probability": 0.0,
            "alpha_range": [0.0, 0.0],
            "kernel_choices": [3],
        }
    flow["augmentation"] = augmentation

    writer = dict(capture.outputs_spec)
    if writer.pop("rename_to_grid_index", False):
        flow["rename_to_grid_index"] = True
    if "semantic_types" not in writer:
        # class labels for plain captures; the operation namespace when
        # operation labels were requested (defect flows)
        namespace = (annotations.operation_labels or {}).get("namespace")
        writer["semantic_types"] = [namespace] if namespace else ["class"]
        # keep frame_padding last, matching the flow-yaml convention
        writer["frame_padding"] = writer.pop("frame_padding")
    flow["writer"] = writer

    if job.postprocess.declared_crop is not None:
        flow["crop"] = dict(job.postprocess.declared_crop)

    pcba = {"scene": scene.config["usd_path"]}
    if scene.config["archive_member"] is not None:
        pcba["archive_member"] = scene.config["archive_member"]
    for src, dst in (
        ("camera_path", "camera_path"),
        ("camera_xform_path", "camera_xform_path"),
        ("ring_light_root", "ring_light_root"),
    ):
        if scene.config[src]:
            pcba[dst] = scene.config[src]
    pcba["pcba_root"] = scene.config["root_path"]
    if scene.config["horizontal_aperture"] is not None:
        pcba["horizontal_aperture"] = scene.config["horizontal_aperture"]
    pcba["component_types"] = component_types
    classes = (annotations.component_labels or {}).get("classes")
    if classes:
        pcba["component_classes"] = classes

    return JobPlan(pipeline_type=pipeline_type, flow=flow, pcba=pcba)


def _selected_component_types(scene):
    """Resolve scene.select_components into the pcba component_types list.

    Explicit types win; otherwise the metadata catalog normalized from a
    readable ``pcb.components`` sidecar supplies the board's package names.
    There is no keyword fallback. The legacy renderer resolves ``ALL`` against
    a registry travelling with the board's own pcba target, and the target
    this module generates has none -- emitting ``ALL`` here compiled cleanly
    and then failed at pipeline startup looking for ``cdf_job/components.yaml``.

    Raises:
        ValidationError: If neither explicit types nor a readable sidecar
            yield the board's component types. The message says which.

    """
    selection = scene.selection or {}
    if selection.get("types"):
        return list(selection["types"])
    direction = (
        "the legacy YAML renderer needs this board's concrete component types "
        "and has no registry to expand a keyword against: pass "
        "scene.select_components(types=[...]), or attach a readable "
        "pcb.components sidecar and select it with source=<name>"
    )
    if not selection:
        raise ValidationError(
            f"scene.select_components(...) was never called; {direction}"
        )
    source = selection.get("source")
    sidecar = scene.sidecars.get(source)
    if sidecar is None or sidecar["kind"] != "pcb.components":
        raise ValidationError(
            f"scene.select_components(source={source!r}) names no pcb.components "
            f"sidecar; {direction}"
        )
    sidecar_path = sidecar["path"]
    if not file_utils.is_file(sidecar_path):
        raise ValidationError(
            f"scene.select_components(source={source!r}): sidecar {sidecar_path} "
            f"is not readable; {direction}"
        )
    from pcb_aoi.board_metadata import SidecarMetadataProvider

    try:
        with file_utils.open(sidecar_path, encoding="utf-8") as handle:
            document = handle.read()
        catalog = SidecarMetadataProvider().from_json(
            document,
            source_identifier=sidecar_path,
        )
    except (OSError, CDFError) as exc:
        raise ValidationError(
            f"scene.select_components(source={source!r}): sidecar {sidecar_path} "
            f"could not be read ({exc}); {direction}"
        ) from exc
    names = sorted(
        {prototype.name for prototype in catalog.prototypes if prototype.name},
    )
    if not names:
        raise ValidationError(
            f"scene.select_components(source={source!r}): sidecar {sidecar_path} "
            f"names no component types; {direction}"
        )
    return names


def execute_crop(source, crop_block, *, dry_run=False, runner=None):
    """Run the in-process crop against an existing capture (post-hoc).

    Delegates to ``scripts/crop/pipeline_adapter.run_pipeline_crop`` so path
    resolution matches the ``--stage crop`` pipeline behavior exactly. The
    ``runner`` hook receives the pipeline-shaped cfg dict (for tests).
    """
    if not isinstance(source, RenderResult):
        raise ValidationError(
            "postprocess.crop: source must be a RenderResult for an existing capture"
        )
    cfg = {
        "pipeline_type": source.pipeline_type,
        "output": source.output_dir,
        "crop": dict(crop_block),
    }
    result = CropResult(
        output_dir=_join_output_location(
            source.output_dir,
            crop_block["output_subdir"],
        ),
        config=cfg,
    )
    if dry_run:
        return result

    if file_utils.is_remote(source.output_dir):
        source_prefix = source.output_dir.rstrip("/") + "/"
        with (
            file_utils.localize_tree(source_prefix) as localized_output,
            file_utils.output_tree(source.output_dir) as staged_output,
        ):
            shutil.copytree(localized_output, staged_output, dirs_exist_ok=True)
            execution_cfg = dict(cfg, output=staged_output)
            _run_crop_config(execution_cfg, runner=runner)
    else:
        _run_crop_config(cfg, runner=runner)
    result.executed = True
    return result
