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

"""Adapt the first JobOrder handlers to existing Isaac SDG APIs."""

from __future__ import annotations

import importlib
import json
import random
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from pcb_aoi.execution.board_instances import (
    build_component_semantic_projection,
    mark_instanced_stage_mutation,
    resolve_board_instance_cohort,
)
from scripts import file_utils

from ...errors import ValidationError
from .execution_context import IsaacExecutionContext, LoadedJsonDocument
from .job_order_parameters import (
    AugmentationParameters,
    CameraParameters,
    CaptureLimitParameters,
    CaptureOutputParameters,
    CaptureRenderParameters,
    CaptureVisibilityParameters,
    CaptureWriterParameters,
    ComponentAnnotationParameters,
    ComponentAppearanceParameters,
    ComponentSelectionParameters,
    GeometryAnnotationParameters,
    HolePlaneParameters,
    LightingApplyParameters,
    LightingParameters,
    MaterialOverrideParameters,
    MissingParameters,
    OperationLabelParameters,
    PostprocessCloseOpeningsParameters,
    PostprocessCropParameters,
    PostprocessFormatRoiOutputParameters,
    PostprocessGenerateJointRoisParameters,
    PostprocessStitchParameters,
    RendererParameters,
    ReversePolarityParameters,
    RoiSelectionParameters,
    ScanGridParameters,
    SceneLoadParameters,
    ShiftParameters,
    SidecarLoadParameters,
    SideflipParameters,
    SolderDefectParameters,
    SolderJointParameters,
    StageSerializationParameters,
    TombstoneParameters,
)

if TYPE_CHECKING:
    from pcb_aoi.board_metadata import BoardMetadataCatalog, MetadataCapability

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STANDALONE_DIR = _REPO_ROOT / "scripts" / "sdg" / "standalone"


def _standalone_module(name: str):
    """Import an existing standalone SDG module only inside Isaac Sim.

    Args:
        name: Top-level module name within the standalone script directory.

    Returns:
        Imported standalone SDG module.

    """
    path = str(_STANDALONE_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    importlib.invalidate_caches()
    return importlib.import_module(name)


def _load_usd_board_metadata(
    stage: object,
    board_root: str,
    source_identifier: str,
):
    """Load embedded board metadata from a composed USD stage.

    Keep the USD schema import inside the Isaac execution boundary so ordinary
    Python API compilation does not require ``pxr``.

    Args:
        stage: Open composed USD stage.
        board_root: Absolute board-root prim path.
        source_identifier: Localized source path used for provenance.

    Returns:
        Fully loaded source-neutral board metadata catalog.

    """
    from pcb_aoi.board_metadata import BoardMetadataRequest
    from pcb_aoi.board_metadata.usd_schema import UsdSchemaMetadataProvider

    return UsdSchemaMetadataProvider().load(
        BoardMetadataRequest(
            board_root=board_root,
            source_identifier=source_identifier,
            stage=stage,
        ),
    )


def _board_metadata(
    context: IsaacExecutionContext,
    source: str | None,
    required_capabilities: Iterable[MetadataCapability] = (),
) -> BoardMetadataCatalog:
    """Resolve the single metadata catalog selected for this job."""
    required = frozenset(required_capabilities)
    sidecar_sources = tuple(name for name in context.board_metadata if name != "usd")
    selected_source = context.state.get("board_metadata.source")
    if selected_source is not None and not isinstance(selected_source, str):
        raise ValidationError("selected board metadata source is invalid")

    anchored_source = source if source in sidecar_sources else None
    if anchored_source is None and sidecar_sources:
        anchored_source = sidecar_sources[0]
    if anchored_source is None:
        anchored_source = "usd" if "usd" in context.board_metadata else source
    if anchored_source is None:
        raise ValidationError("no board metadata catalog is loaded")
    if source is not None and source != anchored_source:
        if sidecar_sources:
            raise ValidationError(
                f"attached component sidecar selects metadata source "
                f"{anchored_source!r}; cannot use {source!r}",
            )
        raise ValidationError(
            f"board metadata source is {anchored_source!r}; cannot use {source!r}",
        )
    if selected_source is not None and selected_source != anchored_source:
        raise ValidationError(
            f"board metadata source is already {selected_source!r}; "
            f"cannot switch to {anchored_source!r} within one job",
        )

    catalog = context.board_metadata.get(anchored_source)
    if catalog is None:
        raise ValidationError(
            f"board metadata source {anchored_source!r} is not loaded",
        )
    missing = required - catalog.capabilities
    if missing:
        names = ", ".join(sorted(capability.value for capability in missing))
        raise ValidationError(
            f"board metadata source {anchored_source!r} is missing capabilities: "
            f"{names}",
        )
    context.state["board_metadata.source"] = anchored_source
    return catalog


def _defect_targeting(parameters: object) -> dict[str, float | int]:
    """Return the selected count or ratio from validated defect parameters."""
    count = getattr(parameters, "count", None)
    if count is not None:
        return {"count": int(count)}
    ratio = getattr(parameters, "ratio", None)
    if ratio is None:
        raise ValidationError("defect targeting is missing count and ratio")
    return {"ratio": float(ratio)}


def _project_geometry_class(context, source_paths, class_name, refdes):
    """Add one geometry class to an instanced cohort's semantic projection.

    With a board-instance cohort the semantic image is projected from instance
    ids through catalog component records, so labels authored on prims are not
    what the writer reads. Projection resolves a prim through its nearest
    labeled ancestor, so a record for the geometry itself takes precedence over
    the component record that encloses it.

    Args:
        context: Mutable execution context holding the projection.
        source_paths: Prototype-source prim paths to classify.
        class_name: Semantic class to project.
        refdes: Reference designator recorded on the projected entries.

    Returns:
        Number of projected records added.

    """
    from pcb_aoi.execution.board_instances import (
        ComponentProjectionRecord,
        InstanceSemanticProjection,
    )

    projection = context.instance_semantic_projection
    if projection is None:
        cohort = context.board_instance_cohort
        if cohort is None:
            return 0
        projection = InstanceSemanticProjection(cohort=cohort, components=())
    board_root = projection.cohort.prototype_source_path.rstrip("/")
    prefix = f"{board_root}/"
    added = []
    for path in source_paths:
        if not path.startswith(prefix):
            continue
        added.append(
            ComponentProjectionRecord(
                source_path=path,
                member_paths=(path,),
                refdes=refdes,
                class_name=class_name,
            ),
        )
    if not added:
        return 0
    context.instance_semantic_projection = InstanceSemanticProjection(
        cohort=projection.cohort,
        components=projection.components + tuple(added),
    )
    return len(added)


def _label_geometry_prims(stage, common, prim_paths, label):
    """Apply one semantic class to each prim in ``prim_paths``.

    Args:
        stage: Open USD stage.
        common: Loaded standalone ``common`` module.
        prim_paths: Absolute prim paths to label.
        label: Semantic class to author.

    Returns:
        Number of distinct prims labeled.

    """
    from pxr import Sdf

    # USD 0.25.11 forbids CreateAttribute on instance proxy prims.  Un-instance
    # any component roots whose sub-geometry appears in prim_paths so those
    # children become real, writable prims.  The caller's
    # mark_instanced_stage_mutation signal ensures the stage is reopened as a
    # flattened layer before rendering, which picks up the authored semantics.
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsInstanceProxy():
            continue
        ancestor = prim.GetParent()
        while ancestor and ancestor.IsValid() and not ancestor.IsInstance():
            ancestor = ancestor.GetParent()
        if ancestor and ancestor.IsValid():
            ancestor.SetInstanceable(False)

    authored: set[str] = set()
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or path in authored:
            continue
        prim.CreateAttribute(
            "semantic:Semantics:class:semanticData",
            Sdf.ValueTypeNames.String,
        ).Set(label)
        prim.CreateAttribute(
            "semantic:Semantics:class:semanticType",
            Sdf.ValueTypeNames.String,
        ).Set("class")
        authored.add(path)

    if authored:
        common.refresh_authored_semantics(stage, sorted(authored))
    return len(authored)


def sample_material_variation(
    random_seed: int,
    material_path: str,
    variation: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Draw one appearance sample for a material override.

    Declared ranges are resolved at execution rather than in the authoring layer,
    so a compiled plan stays a description of intent. The stream is keyed by the
    job seed and the material path, which makes a rerun of the same plan
    reproduce the same surface while two materials in one job vary independently.

    Args:
        random_seed: Job-level random seed.
        material_path: Absolute material prim path being overridden.
        variation: Mapping of control name to inclusive ``(low, high)`` bounds.

    Returns:
        Sampled value per declared control; empty when nothing was declared.

    Example:

    .. code-block:: python

        sample_material_variation(0, "/World/Looks/Carrier", {"roughness": (0.2, 0.6)})

    """
    if not variation:
        return {}
    stream = random.Random(f"{random_seed}:{material_path}")
    return {
        control: stream.uniform(float(low), float(high))
        for control, (low, high) in sorted(variation.items())
    }


_DEFAULT_RENDER_ORDER_ID = "capture.render"


def _is_render_order_id(order_id: str) -> bool:
    """Return whether ``order_id`` names a ``capture.render`` JobOrder.

    The compiler names the default pass ``capture.render`` and a named pass
    ``capture.<pass>.render``; every other capture order id
    (``capture.renderer``, ``capture.<pass>.camera``, ...) is configuration.
    """
    if order_id == _DEFAULT_RENDER_ORDER_ID:
        return True
    parts = order_id.split(".")
    return len(parts) == 3 and parts[0] == "capture" and parts[2] == "render"


def _resolve_crop_capture_output(
    context: IsaacExecutionContext,
) -> tuple[str, dict[str, object]]:
    """Return the render order id and output a crop should read from.

    ``postprocess.crop`` carries no ``source_pass``; the compiler makes it
    depend on every render pass in the plan. The legacy default pass wins when
    it rendered, a single named pass is used as-is, and several named passes
    without a default are refused rather than guessed at.

    Args:
        context: Execution context holding completed order outputs.

    Returns:
        The selected render order id and its structured output.

    Raises:
        ValidationError: If no render pass completed, or if the choice between
            several named passes is ambiguous.

    """
    render_outputs = {
        order_id: output
        for order_id, output in context.outputs.items()
        if _is_render_order_id(order_id) and isinstance(output, dict)
    }
    if not render_outputs:
        raise ValidationError(
            "postprocess.crop requires a successful capture.render output",
        )
    if _DEFAULT_RENDER_ORDER_ID in render_outputs:
        return _DEFAULT_RENDER_ORDER_ID, render_outputs[_DEFAULT_RENDER_ORDER_ID]
    if len(render_outputs) == 1:
        return next(iter(render_outputs.items()))
    raise ValidationError(
        "postprocess.crop cannot choose between several named capture passes "
        f"{sorted(render_outputs)}; declare a default capture block or crop a "
        "job with one render pass",
    )


class IsaacSdgRuntime:
    """Concrete adapter for reusable functions in ``scripts/sdg/standalone``."""

    @staticmethod
    def _store_configuration(
        context: IsaacExecutionContext,
        key: str,
        parameters: BaseModel,
    ) -> dict[str, object]:
        """Retain validated intent until its runtime lifecycle consumes it.

        Args:
            context: Mutable execution context.
            key: Stable state slot for the configuration.
            parameters: Validated Pydantic parameter model.

        Returns:
            Stored configuration values.

        """
        values = parameters.model_dump(mode="python")
        context.state[key] = values
        return values

    async def load_scene(
        self,
        context: IsaacExecutionContext,
        parameters: SceneLoadParameters,
    ) -> dict[str, object]:
        """Open a USD stage and store it in the execution context.

        Args:
            context: Mutable execution context.
            parameters: Validated scene-load parameters.

        Returns:
            Scene identity and validated board-root information.

        Raises:
            RuntimeError: If Isaac cannot open the stage or find its board root.

        """
        import omni.kit.app
        import omni.usd

        common = _standalone_module("common")
        app = context.app or omni.kit.app.get_app()
        local_usd_path = context.localize_scene(
            parameters.usd_path,
            parameters.archive_member,
        )
        await common.open_usd_stage(app, local_usd_path)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac opened the USD path but returned no stage")
        root = stage.GetPrimAtPath(parameters.root_path)
        if not root or not root.IsValid():
            raise RuntimeError(
                f"scene root {parameters.root_path!r} does not exist in the opened stage",
            )

        context.app = app
        context.stage = stage
        context.scene = parameters.model_dump(mode="python")
        context.scene["usd_path"] = local_usd_path
        context.state["storage.scene_uri"] = parameters.usd_path
        context.board_instance_cohort = None
        context.instance_semantic_projection = None
        cohort = None
        board_root_path = parameters.root_path
        if parameters.instance_paths is not None:
            cohort = resolve_board_instance_cohort(
                stage,
                parameters.root_path,
                parameters.instance_paths,
            )
            context.board_instance_cohort = cohort
            context.scene.update(cohort.as_mapping())
            board_root_path = cohort.prototype_source_path
        context.scene["board_root_path"] = board_root_path
        catalog = _load_usd_board_metadata(
            stage,
            board_root_path,
            local_usd_path,
        )
        context.board_metadata["usd"] = catalog
        output = {
            "usd_path": parameters.usd_path,
            "root_path": parameters.root_path,
            "board_root_path": board_root_path,
            "metadata_components": len(catalog.components),
        }
        if cohort is not None:
            output.update(cohort.as_mapping())
        return output

    def load_sidecar(
        self,
        context: IsaacExecutionContext,
        parameters: SidecarLoadParameters,
    ) -> dict[str, object]:
        """Decode and retain a JSON sidecar for dependent handlers.

        Args:
            context: Mutable execution context.
            parameters: Validated sidecar parameters.

        Returns:
            Sidecar identity and top-level record count.

        Raises:
            ValidationError: If the sidecar is missing or malformed.

        """
        context.require_stage()
        if parameters.kind == "pcb.components" and context.board_instance_cohort:
            raise ValidationError(
                "scene.instances requires embedded USD component metadata; "
                "pcb.components sidecars cannot describe a shared prototype cohort",
            )
        if not file_utils.is_file(parameters.path):
            raise ValidationError(f"sidecar file does not exist: {parameters.path}")
        try:
            with file_utils.open(parameters.path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(
                f"could not read sidecar {parameters.path}: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise ValidationError(
                f"sidecar root must be a JSON object: {parameters.path}",
            )

        catalog = None
        if parameters.kind == "pcb.components":
            from pcb_aoi.board_metadata import (
                BoardMetadataRequest,
                SidecarMetadataProvider,
            )

            # The board root comes from the STAGE, not from the document:
            # scene.load may have adopted a session that referenced the board
            # in under /World, and the sidecar's own board_prim_path then
            # describes a stage nobody has open.
            catalog = SidecarMetadataProvider().load(
                BoardMetadataRequest(
                    board_root=context.scene.get("board_root_path") or None,
                    source_identifier=parameters.path,
                    sidecar_data=data,
                ),
            )

        if catalog is not None:
            selected_source = context.state.get("board_metadata.source")
            if selected_source not in (None, parameters.name):
                raise ValidationError(
                    f"board metadata source is already {selected_source!r}; "
                    f"cannot attach {parameters.name!r}",
                )
            existing_sidecars = tuple(
                name for name in context.board_metadata if name != "usd"
            )
            if existing_sidecars and parameters.name not in existing_sidecars:
                raise ValidationError(
                    "only one pcb.components sidecar may be attached per job",
                )
            context.board_metadata.clear()
            context.board_metadata[parameters.name] = catalog
            context.state["board_metadata.source"] = parameters.name
        else:
            context.json_documents[parameters.name] = LoadedJsonDocument(
                name=parameters.name,
                kind=parameters.kind,
                path=parameters.path,
                data=data,
            )
        return {
            "name": parameters.name,
            "kind": parameters.kind,
            "path": parameters.path,
            "records": len(data),
            "metadata_components": len(catalog.components) if catalog else 0,
        }

    def select_components(
        self,
        context: IsaacExecutionContext,
        parameters: ComponentSelectionParameters,
    ) -> dict[str, object]:
        """Build the component population consumed by PCB operations.

        Args:
            context: Mutable execution context.
            parameters: Validated component-selection parameters.

        Returns:
            Selected family names and component prim paths.

        Raises:
            ValidationError: If component types cannot be resolved or no
                matching component prims exist.

        """
        stage = context.require_stage()
        from pcb_aoi.board_metadata import MetadataCapability

        source = parameters.source
        catalog = _board_metadata(
            context,
            source,
            (MetadataCapability.COMPONENTS,),
        )

        component_types = tuple(parameters.types or ())
        if parameters.component_filter is not None:
            from pcb_aoi.board_metadata import select_components

            if catalog is None:
                raise ValidationError(
                    "component_filter requires board component metadata",
                )
            selected = select_components(
                catalog,
                parameters.component_filter,
                board_root=str(
                    context.scene.get("board_root_path")
                    or context.scene.get("root_path")
                    or ""
                ),
                where="scene.components.component_filter",
            )
            component_pool = tuple(component.prim_path for component in selected)
            context.component_types = ()
            context.component_pool = component_pool
            context.component_selection_resolved = True
            return {
                "component_filter": parameters.component_filter.to_transport(),
                "component_paths": list(component_pool),
                "count": len(component_pool),
            }
        if not component_types and catalog is not None:
            component_types = tuple(
                sorted(
                    {
                        component.package_name
                        for component in catalog.components
                        if component.package_name
                    },
                ),
            )
        if not component_types:
            raise ValidationError(
                "component selection needs explicit types or board metadata",
            )

        root_path = context.scene.get("board_root_path") or context.scene.get(
            "root_path",
        )
        if not isinstance(root_path, str):
            raise ValidationError("scene.load did not provide board_root_path")
        if catalog is not None:
            component_pool = tuple(
                dict.fromkeys(
                    component.prim_path
                    for component_type in component_types
                    for component in catalog.components_by_type(component_type)
                ),
            )
        else:
            common = _standalone_module("common")
            component_pool = tuple(
                common.build_component_pool(stage, root_path, list(component_types)),
            )
        if not component_pool:
            raise ValidationError(
                f"no components found under {root_path!r} for types {list(component_types)}",
            )

        context.component_types = component_types
        context.component_pool = component_pool
        context.component_selection_resolved = True
        return {
            "component_types": list(component_types),
            "component_paths": list(component_pool),
            "count": len(component_pool),
        }

    def configure_lighting(
        self,
        context: IsaacExecutionContext,
        parameters: LightingParameters,
    ) -> dict[str, object]:
        """Retain lighting intent for scene/capture initialization."""
        return self._store_configuration(context, "appearance.lighting", parameters)

    def apply_lighting(
        self,
        context: IsaacExecutionContext,
        parameters: LightingApplyParameters,
    ) -> dict[str, object]:
        """Apply the configured static light rig to the prepared stage.

        Args:
            context: Mutable execution context containing the open stage.
            parameters: Capture side used to orient the light rig.

        Returns:
            Authored light-rig identity and placement.

        Example:

        .. code-block:: python

            result = runtime.apply_lighting(  # doctest: +SKIP
                context,
                LightingApplyParameters(camera_side="top"),
            )

        """
        from .capture_execution import _apply_lighting_appearance

        return _apply_lighting_appearance(context, parameters.camera_side)

    def configure_component_appearance(
        self,
        context: IsaacExecutionContext,
        parameters: ComponentAppearanceParameters,
    ) -> dict[str, object]:
        """Retain component-material intent for scene initialization."""
        return self._store_configuration(
            context,
            "appearance.components",
            parameters,
        )

    def apply_material_override(
        self,
        context: IsaacExecutionContext,
        parameters: MaterialOverrideParameters,
    ) -> dict[str, object]:
        """Apply an OmniPBR texture override to an existing USD material."""
        from pxr import Gf, Sdf, Usd, UsdShade

        stage = context.require_stage()

        # Declared variation ranges are resolved here rather than in the authoring
        # layer so the plan stays a description. The stream is seeded from the job
        # seed and the material path, so a rerun of the same plan reproduces the
        # same surface and two materials in one job vary independently.
        sampled = sample_material_variation(
            int(context.plan_metadata.get("random_seed", 0)),
            parameters.material_path,
            parameters.variation,
        )

        def control(name: str, declared: float) -> float:
            """Return the sampled value for a control, else its declared constant."""
            return sampled.get(name, declared)

        texture_scale = tuple(
            value * sampled.get("texture_scale", 1.0)
            for value in parameters.texture_scale
        )
        material_prim = stage.GetPrimAtPath(parameters.material_path)
        if not material_prim.IsValid() or not material_prim.IsA(UsdShade.Material):
            raise ValidationError(
                f"material does not exist at {parameters.material_path!r}",
            )

        shader_path = f"{parameters.material_path}/CDFOmniPBR"
        resolved_textures: dict[str, str] = {}
        with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
            material = UsdShade.Material(material_prim)
            shader = UsdShade.Shader.Define(stage, shader_path)
            shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

            def set_input(name: str, value: object, value_type: object) -> None:
                shader.CreateInput(name, value_type).Set(value)

            set_input(
                "diffuse_color_constant",
                Gf.Vec3f(*parameters.diffuse_color),
                Sdf.ValueTypeNames.Color3f,
            )
            set_input(
                "diffuse_tint",
                Gf.Vec3f(*parameters.diffuse_tint),
                Sdf.ValueTypeNames.Color3f,
            )
            set_input(
                "albedo_brightness",
                control("albedo_brightness", parameters.albedo_brightness),
                Sdf.ValueTypeNames.Float,
            )
            set_input(
                "enable_opacity",
                parameters.opacity < 1.0,
                Sdf.ValueTypeNames.Bool,
            )
            set_input(
                "opacity_constant",
                parameters.opacity,
                Sdf.ValueTypeNames.Float,
            )
            set_input(
                "reflection_roughness_constant",
                control("roughness", parameters.roughness),
                Sdf.ValueTypeNames.Float,
            )
            set_input(
                "reflection_roughness_texture_influence",
                parameters.roughness_texture_influence,
                Sdf.ValueTypeNames.Float,
            )
            set_input(
                "metallic_constant",
                control("metallic", parameters.metallic),
                Sdf.ValueTypeNames.Float,
            )
            set_input(
                "specular_level",
                control("specular_level", parameters.specular_level),
                Sdf.ValueTypeNames.Float,
            )
            set_input(
                "bump_factor",
                control("normal_strength", parameters.normal_strength),
                Sdf.ValueTypeNames.Float,
            )
            set_input("project_uvw", parameters.project_uvw, Sdf.ValueTypeNames.Bool)
            set_input(
                "world_or_object",
                parameters.world_space_projection,
                Sdf.ValueTypeNames.Bool,
            )
            set_input(
                "texture_scale",
                Gf.Vec2f(*texture_scale),
                Sdf.ValueTypeNames.Float2,
            )
            set_input(
                "texture_rotate",
                control("texture_rotation", parameters.texture_rotation),
                Sdf.ValueTypeNames.Float,
            )
            set_input(
                "flip_tangent_v",
                parameters.normal_format == "directx",
                Sdf.ValueTypeNames.Bool,
            )
            for input_name, texture_path in (
                ("diffuse_texture", parameters.diffuse_texture),
                ("reflectionroughness_texture", parameters.roughness_texture),
                ("normalmap_texture", parameters.normal_texture),
            ):
                if texture_path is not None:
                    # a scene shipped as a ZIP carries its textures with it, and
                    # the extraction directory is only known now
                    localized = context.resolve_scene_asset(texture_path)
                    resolved_textures[input_name] = localized
                    set_input(
                        input_name,
                        Sdf.AssetPath(str(Path(localized).expanduser().resolve())),
                        Sdf.ValueTypeNames.Asset,
                    )
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(),
                "out",
            )
        result = {
            "material_path": parameters.material_path,
            "shader_path": shader_path,
            "project_uvw": parameters.project_uvw,
        }
        if resolved_textures:
            result["textures"] = dict(resolved_textures)
        if sampled:
            # report the draw so a dataset can be traced back to its appearance
            result["sampled_variation"] = {
                control_name: round(value, 6)
                for control_name, value in sorted(sampled.items())
            }
        return result

    def configure_operation_labels(
        self,
        context: IsaacExecutionContext,
        parameters: OperationLabelParameters,
    ) -> dict[str, object]:
        """Retain the semantic namespace used by operation handlers."""
        return self._store_configuration(
            context,
            "annotations.operations",
            parameters,
        )

    def configure_capture_output(
        self,
        context: IsaacExecutionContext,
        parameters: CaptureOutputParameters,
    ) -> dict[str, object]:
        """Retain the first output frame index."""
        return self._store_configuration(context, "capture.output", parameters)

    def apply_solder_defects(
        self,
        context: IsaacExecutionContext,
        parameters: SolderDefectParameters,
    ) -> dict[str, object]:
        """Select components for solder defects and record the intent.

        Selection mirrors the structural operations: each defect type
        independently selects ``ratio`` of the (remaining) eligible pool --
        the scene's selected component pool, or the sidecar instances of its
        ``component_types`` families -- non-overlapping across types and with
        the pose operations. Every produced record carries the type's
        observables (volume ratios are sampled per component HERE, with the
        job seed; gap_fraction ranges pass through for per-joint sampling in
        the engine, alongside a per-record seed). Selected prims get the
        ``defect=<type>`` semantic, exactly like pose operations, so
        ``operation_labels`` exports the class. The run manifest observer also
        retains the component identity and sampled defect observables.

        Args:
            context: Mutable execution context.
            parameters: Validated solder-defect entries plus sidecar source.

        Returns:
            Applied record count and selected component paths.

        Raises:
            ValidationError: If a pool cannot be resolved.

        """
        stage = context.require_stage()
        catalog = _board_metadata(context, parameters.source)

        import numpy as np

        rng = np.random.default_rng(
            int(context.plan_metadata.get("random_seed", 0)),
        )
        structural_used = {
            record["path"]
            for record in (context.state.get("operation.structural.records") or ())
            if isinstance(record, dict) and record.get("path")
        }
        used = set(structural_used)
        records: list[dict[str, object]] = []
        solder_parent = tuple(
            str(path)
            for path in context.state.get("solder_joints.component_pool") or ()
        )
        if not solder_parent:
            raise ValidationError(
                "solder defects require a non-empty solder-joint discovery pool",
            )
        for entry in parameters.entries:
            if entry.component_filter is not None:
                from pcb_aoi.board_metadata import select_components

                matched_components = select_components(
                    catalog,
                    entry.component_filter,
                    parent_paths=solder_parent,
                    where=f"operations.{entry.defect}.component_filter",
                )
                pool = sorted(
                    component.prim_path
                    for component in matched_components
                    if component.prim_path not in used
                )
            elif entry.component_types is not None:
                # Resolve through the catalog's canonical index -- type,
                # family, OR package name -- the same vocabulary
                # scene.components.types uses. Matching record.family alone
                # silently selected NOTHING on embedded-USD boards, whose
                # records carry component_type but no family.
                matched: set[str] = set()
                for value in entry.component_types:
                    matched.update(
                        record.prim_path
                        for record in catalog.components_by_type(str(value))
                    )
                if not matched:
                    # A typo or stale family name would otherwise return a
                    # successful zero-record operation -- the silent-drop
                    # failure mode this resolver was rebuilt to eliminate.
                    raise ValidationError(
                        f"operations.{entry.defect}: component_types "
                        f"{list(entry.component_types)!r} matched no "
                        "component (type, family, and package names are "
                        "accepted)",
                    )
                pool = sorted(
                    path
                    for path in matched
                    if path in solder_parent and path not in used
                )
            else:
                # Same pool the pose operations draw from: the scene's
                # selected components.
                pool = sorted(
                    str(path) for path in solder_parent if str(path) not in used
                )
            count = min(len(pool), max(1, int(len(pool) * float(entry.ratio))))
            paths = (
                [str(p) for p in rng.choice(pool, size=count, replace=False)]
                if pool
                else []
            )
            for path in paths:
                used.add(path)
                record: dict[str, object] = {
                    "path": path,
                    "defect": entry.defect,
                    # Deterministic per (job seed, component): the engine's
                    # per-joint draws (dry-contact azimuth/gap) key off this.
                    "seed": int(rng.integers(0, 2**31 - 1)),
                }
                if entry.volume_ratio is not None:
                    # Sampled per component, per the proposal: the record
                    # carries the collapsed scalar so the observable is
                    # exactly one number per component.
                    value = entry.volume_ratio
                    if isinstance(value, (list, tuple)):
                        value = float(rng.uniform(value[0], value[1]))
                    record["volume_ratio"] = float(value)
                if entry.gap_fraction is not None:
                    # Per-JOINT observable: the range passes through whole.
                    value = entry.gap_fraction
                    record["gap_fraction"] = (
                        list(value)
                        if isinstance(value, (list, tuple))
                        else float(value)
                    )
                # THT shape parameters: per-JOINT, so ranges pass through
                # whole and the engine samples against the record seed.
                for shape_name in ("lead_climb", "fullness", "pad_exposure"):
                    value = getattr(entry, shape_name, None)
                    if value is not None:
                        record[shape_name] = (
                            list(value)
                            if isinstance(value, (list, tuple))
                            else float(value)
                        )
                records.append(record)

        defect_ops = _standalone_module("defect_ops")
        for record in records:
            prim = stage.GetPrimAtPath(str(record["path"]))
            if prim and prim.IsValid():
                defect_ops.apply_defect_semantic(prim, str(record["defect"]))

        existing = list(context.state.get("operation.solder_defects.records") or ())
        existing.extend(records)
        context.state["operation.solder_defects.records"] = existing
        context.manifest.observe_defects(records, context.entity_identity)
        return {
            "count": len(records),
            "component_paths": [str(record["path"]) for record in records],
        }

    def discover_solder_joints(
        self,
        context: IsaacExecutionContext,
        parameters: SolderJointParameters,
    ) -> dict[str, object]:
        """Discover reusable prototype/side solder work from board metadata.

        Args:
            context: Mutable execution context containing board metadata.
            parameters: Validated solder-joint discovery parameters.

        Returns:
            Solver identity, candidate count, and serialized candidates.

        Raises:
            ValidationError: If the requested metadata source is unavailable.

        """
        from pcb_aoi.solder_generation import (
            discover_solder_components,
            resolve_solver,
        )

        solver = resolve_solver(parameters.solver)
        catalog = _board_metadata(context, parameters.source)
        parent_paths = context.component_parent_paths()
        if parameters.component_filter is not None:
            from pcb_aoi.board_metadata import select_components

            selected = select_components(
                catalog,
                parameters.component_filter,
                parent_paths=parent_paths,
                where="operations.solder_joints.component_filter",
            )
            component_paths = tuple(component.prim_path for component in selected)
            component_types = None
        else:
            component_paths = parent_paths
            component_types = parameters.component_types

        solver_components = parameters.solver_options.get("components")
        if solver_components is not None:
            if not isinstance(solver_components, (list, tuple)) or not all(
                isinstance(identifier, str) and identifier
                for identifier in solver_components
            ):
                raise ValidationError(
                    "operations.solder_joints.options.components must be a list "
                    "of non-empty refdes or component paths",
                )
            if solver_components:
                allowed_identifiers = set(solver_components)
                matched_paths = tuple(
                    component.prim_path
                    for component in catalog.components
                    if (
                        component_paths is None
                        or component.prim_path in component_paths
                    )
                    and (
                        component.refdes in allowed_identifiers
                        or component.prim_path in allowed_identifiers
                    )
                )
                matched_identifiers = {
                    identifier
                    for component in catalog.components
                    if component.prim_path in matched_paths
                    for identifier in (component.refdes, component.prim_path)
                    if identifier in allowed_identifiers
                }
                unmatched = sorted(allowed_identifiers - matched_identifiers)
                if unmatched:
                    formatted = ", ".join(repr(value) for value in unmatched)
                    raise ValidationError(
                        "operations.solder_joints.options.components entries "
                        f"matched no component in the effective pool: {formatted}",
                    )
                component_paths = matched_paths
        candidates = discover_solder_components(
            catalog,
            component_types,
            component_paths=component_paths,
            require_sample_paste=solver.requires_source_paste,
        )
        serialized = [candidate.to_dict() for candidate in candidates]
        context.state["solder_joints.candidates"] = candidates
        context.state["solder_joints.component_pool"] = tuple(
            dict.fromkeys(
                path for candidate in candidates for path in candidate.instance_paths
            ),
        )
        context.state["solder_joints.solver"] = solver.solver_id
        context.state["solder_joints.solver_type"] = solver.solver_type.value
        return {
            "solver": solver.solver_id,
            "solver_type": solver.solver_type.value,
            "requires_source_paste": solver.requires_source_paste,
            "count": len(serialized),
            "candidates": serialized,
        }

    def collect_solder_joint_inputs(
        self,
        context: IsaacExecutionContext,
        parameters: SolderJointParameters,
    ) -> dict[str, object]:
        """Collect solver-neutral inputs for the selected generation delegate.

        Structural defects have already been applied when this handler runs.
        The selected delegate receives the active stage, metadata catalog,
        discovered candidates, and accumulated defect records. It owns
        backend-specific conditioning and work partitioning.

        Args:
            context: Mutable execution context containing discovered candidates.
            parameters: Validated solder-joint collection parameters.

        Returns:
            Transport-safe summary of the collected generation context.

        Raises:
            ValidationError: If discovery or scene inputs are unavailable.

        """
        from pcb_aoi.solder_generation import (
            SolderComponentCandidate,
            SolderGenerationExtension,
            SolderGenerationRequest,
            SolderSerializationMode,
            resolve_solver,
        )

        solver = resolve_solver(parameters.solver)
        stage = context.require_stage()
        catalog = _board_metadata(context, parameters.source)
        candidates = context.state.get("solder_joints.candidates")
        if not isinstance(candidates, tuple) or not all(
            isinstance(candidate, SolderComponentCandidate) for candidate in candidates
        ):
            raise ValidationError(
                "solder-joint collection requires discovery to complete first",
            )
        source_usd_path = context.scene.get("usd_path")
        root_path = context.scene.get("board_root_path") or context.scene.get(
            "root_path",
        )
        if not isinstance(source_usd_path, str) or not isinstance(root_path, str):
            raise ValidationError(
                "solder-joint collection requires scene USD and board-root paths",
            )
        defect_records = tuple(
            dict(record)
            for source_key in (
                "operation.structural.records",
                # Solder-level defects (solder_defects.apply) ride the same
                # record contract: path + defect + observables, no transform.
                "operation.solder_defects.records",
            )
            for record in (context.state.get(source_key) or ())
            if isinstance(record, dict)
        )
        output_root_path = parameters.output_root_path or "{}/SolderJoints".format(
            root_path.rstrip("/")
        )
        root_prefix = "{}/".format(root_path.rstrip("/"))
        if output_root_path == root_path.rstrip("/") or not output_root_path.startswith(
            root_prefix
        ):
            raise ValidationError(
                f"solder output root {output_root_path!r} must be a descendant of PCBA root {root_path!r}",
            )
        generation_request = SolderGenerationRequest(
            request_id=f"{output_root_path}@{parameters.solver}",
            solver_id=parameters.solver,
            source_usd_path=source_usd_path,
            root_path=root_path,
            output_root_path=output_root_path,
            stage=stage,
            board_metadata=catalog,
            candidates=candidates,
            defect_records=defect_records,
            solver_options=dict(parameters.solver_options),
            serialization=SolderSerializationMode(parameters.serialization),
            serialization_path=parameters.serialization_path,
        )
        extension = SolderGenerationExtension()
        context.state["solder_joints.extension"] = extension
        context.state["solder_joints.generation_request"] = generation_request
        summary = generation_request.summary()
        summary["solver_type"] = solver.solver_type.value
        return summary

    def generate_solder_joints(
        self,
        context: IsaacExecutionContext,
        parameters: SolderJointParameters,
    ) -> dict[str, object]:
        """Invoke the registered solder-generation solver.

        Args:
            context: Mutable execution context containing collected inputs.
            parameters: Validated solder-joint generation parameters.

        Returns:
            Generated solder artifact and placement metadata.

        Raises:
            ValidationError: If input collection has not completed.

        """
        from pcb_aoi.solder_generation import (
            SolderGenerationExtension,
            SolderGenerationRequest,
        )

        extension = context.state.get("solder_joints.extension")
        generation_request = context.state.get(
            "solder_joints.generation_request",
        )
        if not isinstance(extension, SolderGenerationExtension) or not isinstance(
            generation_request,
            SolderGenerationRequest,
        ):
            raise ValidationError(
                "solder-joint generation requires input collection first",
            )
        if generation_request.solver_id != parameters.solver:
            raise ValidationError(
                "solder-joint generation solver does not match collection",
            )
        solver_request = extension.prepare(generation_request)
        result = extension.generate(generation_request, solver_request)
        context.state["solder_joints.solver_request"] = solver_request
        context.state["solder_joints.solver_result"] = result
        return {
            "request_id": result.request_id,
            "solver": result.solver_id,
            "artifact_count": len(result.artifacts),
            "coverage": [
                {
                    "work_item_id": item.work_item_id,
                    "status": item.status.value,
                    "fallback_from": item.fallback_from,
                }
                for item in result.coverage
            ],
            "warnings": list(result.warnings),
        }

    def attach_solder_joints(
        self,
        context: IsaacExecutionContext,
        parameters: SolderJointParameters,
    ) -> dict[str, object]:
        """Attach solver output through the solder-generation extension.

        Standalone boards register copied solder semantic attributes with
        Replicator immediately. Registered board-instance cohorts deliberately
        skip that registration because touching semantics below their shared
        prototype can invalidate authored materials in the active renderer. The
        instance-safe capture path derives component masks from instance IDs.

        Args:
            context: Mutable execution context containing generated artifacts.
            parameters: Validated solder-joint attachment parameters.

        Returns:
            Attached USD layer, coverage, and reversible placement metadata.

        Raises:
            ValidationError: If generation did not complete first.

        """
        from pcb_aoi.solder_generation import (
            SolderGenerationExtension,
            SolderGenerationRequest,
            SolderSolverRequest,
            SolderSolverResult,
        )

        extension = context.state.get("solder_joints.extension")
        generation_request = context.state.get(
            "solder_joints.generation_request",
        )
        solver_request = context.state.get("solder_joints.solver_request")
        solver_result = context.state.get("solder_joints.solver_result")
        if not (
            isinstance(extension, SolderGenerationExtension)
            and isinstance(generation_request, SolderGenerationRequest)
            and isinstance(solver_request, SolderSolverRequest)
            and isinstance(solver_result, SolderSolverResult)
        ):
            raise ValidationError(
                "solder-joint attachment requires generation to complete first",
            )
        if generation_request.solver_id != parameters.solver:
            raise ValidationError(
                "solder-joint attachment solver does not match collection",
            )
        manifest = extension.attach(
            generation_request,
            solver_request,
            solver_result,
        )
        if context.board_instance_cohort is None:
            common = _standalone_module("common")
            common.refresh_authored_semantics(
                generation_request.stage,
                list(manifest.generated_prim_paths),
            )
        context.state["solder_joints.placement_manifest"] = manifest
        summary = manifest.summary()
        solder_class = context.state.get("annotations.geometry.solder")
        if solder_class:
            # The joints exist only now, so annotations.geometry.apply deferred
            # its solder class to here. Label the output root and the joints
            # themselves: inheritance covers descendant meshes, and the explicit
            # prims keep the label independent of how the solver nests them.
            paths = [generation_request.output_root_path]
            paths.extend(manifest.generated_prim_paths)
            summary["semantic_class"] = solder_class
            summary["semantic_prims"] = _label_geometry_prims(
                generation_request.stage,
                _standalone_module("common"),
                paths,
                solder_class,
            )
            summary["semantic_projected"] = _project_geometry_class(
                context,
                paths,
                solder_class,
                "solder",
            )
        mark_instanced_stage_mutation(context, "solder_joints.attach")
        return summary

    def generate_hole_plane(
        self,
        context: IsaacExecutionContext,
        parameters: HolePlaneParameters,
    ) -> dict[str, object]:
        """Create hole-reveal plane(s) on the active stage.

        Without a component filter, creates one board-sized quad mesh whose XY
        footprint covers ``board_substrate`` prims; substrate geometry occludes
        it everywhere except at through-hole voids.  With ``parameters.components``
        set, creates one small plane per selected-component pad prim instead,
        sized from the pad's world-space geometry.

        Generated prims carry ``pcba:mesh:role = "board_hole"`` and are hidden
        by default. Use ``capture.visibility(show=["hole"])`` to reveal them
        temporarily in annotation passes.

        Args:
            context: Mutable execution context with the loaded board stage.
            parameters: Validated hole-plane generation parameters.

        Returns:
            Dictionary with ``prim_path``, ``class_name``, ``side``, and
            ``z_offset_mm`` describing the created geometry.

        Raises:
            ValidationError: If the board metadata lacks ``substrate_z_range``
                or no ``board_substrate`` prims are found.

        """
        from pcb_aoi.board_metadata import MetadataCapability
        from pcb_aoi.geometry_labels import SEGMENTATION_PROXIES_SCOPE
        from pcb_aoi.hole_plane import create_hole_plane, create_hole_plane_at_position
        from pcb_aoi.refdes_selection import select_refdes

        stage = context.require_stage()
        catalog = _board_metadata(context, source=None)
        if not catalog.boards or catalog.boards[0].substrate_z_range is None:
            raise ValidationError(
                "hole_plane.generate requires board substrate z-range metadata; "
                "ensure the board USD contains board_substrate meshes with "
                "pcba:mesh:role attributes"
            )
        board = catalog.boards[0]
        board_root = str(
            context.scene.get("board_root_path") or context.scene.get("root_path")
        )
        segmentation_proxies_path = (
            f"{board_root.rstrip('/')}/{SEGMENTATION_PROXIES_SCOPE}"
        )
        sides = ("bottom", "top") if parameters.side == "both" else (parameters.side,)
        requested_components = parameters.components
        parent_paths = context.component_parent_paths()
        if parameters.component_filter is not None:
            from pcb_aoi.board_metadata import select_components

            selected_components = select_components(
                catalog,
                parameters.component_filter,
                parent_paths=parent_paths,
                where="operations.hole_plane.component_filter",
            )
            requested_components = tuple(
                component.refdes for component in selected_components
            )
        elif requested_components is None:
            has_parent_selection = parent_paths is not None
            parent_paths = set(parent_paths or ())
            catalog_paths = {component.prim_path for component in catalog.components}
            if has_parent_selection and parent_paths != catalog_paths:
                requested_components = tuple(
                    component.refdes
                    for component in catalog.components
                    if component.prim_path in parent_paths
                )

        _pads_meta_available = {
            MetadataCapability.PADS,
            MetadataCapability.COMPONENTS,
        }.issubset(catalog.capabilities)
        if requested_components == ():
            message = (
                "hole_plane components is empty; explicit refdes filters never "
                "fall back to a board-wide plane"
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            context.state["hole_plane.prim_path"] = []
            return {
                "prim_path": [],
                "class_name": parameters.class_name,
                "side": parameters.side,
                "z_offset_mm": parameters.z_offset_mm,
                "requested_components": [],
                "matched_components": [],
                "unmatched_components": [],
                "ineligible_components": [],
                "failed_prim_paths": [],
                "warnings": [message],
            }
        if requested_components is not None and not _pads_meta_available:
            message = (
                "hole_plane components matched no eligible pad geometry because the "
                "board does not expose PADS and COMPONENTS metadata; explicit refdes "
                "filters never fall back to a board-wide plane"
            )
            warnings.warn(
                message,
                RuntimeWarning,
                stacklevel=2,
            )
            context.state["hole_plane.prim_path"] = []
            return {
                "prim_path": [],
                "class_name": parameters.class_name,
                "side": parameters.side,
                "z_offset_mm": parameters.z_offset_mm,
                "requested_components": list(requested_components),
                "matched_components": [],
                "unmatched_components": list(requested_components),
                "ineligible_components": [],
                "failed_prim_paths": [],
                "warnings": [message],
            }
        if requested_components is not None and _pads_meta_available:
            # Per-component path: one small plane per selected-component pad prim.
            pads_catalog = _board_metadata(context, None, (MetadataCapability.PADS,))
            comp_catalog = _board_metadata(
                context, None, (MetadataCapability.COMPONENTS,)
            )
            selection = select_refdes(
                comp_catalog.components,
                requested_components,
                key=lambda component: component.refdes,
            )
            allowed_comp_paths = frozenset(
                component.prim_path for component in selection.selected
            )
            selected_pads = [
                p for p in pads_catalog.pads if p.component_path in allowed_comp_paths
            ]
            eligible_paths = frozenset(pad.component_path for pad in selected_pads)
            eligible_refdes = frozenset(
                component.refdes
                for component in selection.selected
                if component.prim_path in eligible_paths
            )
            ineligible = tuple(
                refdes for refdes in selection.matched if refdes not in eligible_refdes
            )
            diagnostics = []
            if selection.unmatched:
                diagnostics.append(
                    "hole_plane components were not found in board metadata: "
                    f"{list(selection.unmatched)}"
                )
            if ineligible:
                diagnostics.append(
                    "hole_plane components have no eligible pad geometry: "
                    f"{list(ineligible)}"
                )
            for message in diagnostics:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            all_prim_paths: list[str] = []
            failed_prim_paths: list[str] = []
            for s in sides:
                suffix = f"_{s}" if parameters.side == "both" else ""
                parent = (
                    f"{parameters.output_prim_path}{suffix}"
                    if parameters.output_prim_path
                    else f"{segmentation_proxies_path}/HolePlane{suffix}"
                )
                for i, pad in enumerate(selected_pads):
                    created = create_hole_plane_at_position(
                        stage,
                        f"{parent}/pad_{i}",
                        pad.prim_path,
                        board.substrate_z_range,
                        side=s,
                        z_offset_mm=parameters.z_offset_mm,
                        meters_per_unit=board.meters_per_unit,
                        class_name=parameters.class_name,
                    )
                    if created is not None:
                        _standalone_module("common").refresh_authored_semantics(
                            stage, [created]
                        )
                        all_prim_paths.append(created)
                    else:
                        failed_prim_paths.append(pad.prim_path)
            if failed_prim_paths:
                message = (
                    "hole_plane could not generate proxies for "
                    f"{len(failed_prim_paths)} selected pad(s): {failed_prim_paths}"
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                diagnostics.append(message)
            context.state["hole_plane.prim_path"] = all_prim_paths
            return {
                "prim_path": all_prim_paths,
                "class_name": parameters.class_name,
                "side": parameters.side,
                "z_offset_mm": parameters.z_offset_mm,
                "requested_components": list(selection.requested),
                "matched_components": [
                    refdes for refdes in selection.matched if refdes in eligible_refdes
                ],
                "unmatched_components": list(selection.unmatched),
                "ineligible_components": list(ineligible),
                "failed_prim_paths": failed_prim_paths,
                "warnings": diagnostics,
            }

        # Board-wide path: one large plane covering the entire XY footprint.
        prim_paths = []
        for s in sides:
            suffix = f"_{s}" if parameters.side == "both" else ""
            path = (
                f"{parameters.output_prim_path}{suffix}"
                if parameters.output_prim_path
                else f"{segmentation_proxies_path}/HolePlane{suffix}"
            )
            p = create_hole_plane(
                stage,
                board_root,
                path,
                board.substrate_z_range,
                side=s,
                z_offset_mm=parameters.z_offset_mm,
                meters_per_unit=board.meters_per_unit,
                class_name=parameters.class_name,
            )
            _standalone_module("common").refresh_authored_semantics(stage, [p])
            prim_paths.append(p)
        result_paths = prim_paths if parameters.side == "both" else prim_paths[0]
        context.state["hole_plane.prim_path"] = result_paths
        return {
            "prim_path": result_paths,
            "class_name": parameters.class_name,
            "side": parameters.side,
            "z_offset_mm": parameters.z_offset_mm,
        }

    def serialize_stage(
        self,
        context: IsaacExecutionContext,
        parameters: StageSerializationParameters,
    ) -> dict[str, object]:
        """Export the final composed stage before capture begins.

        Args:
            context: Mutable execution context containing the prepared stage.
            parameters: Validated final-stage serialization destination.

        Returns:
            Published serialization path.

        Raises:
            RuntimeError: If USD cannot flatten or export the composed stage.

        """
        stage = context.require_stage()
        destination = parameters.serialization_path
        if file_utils.is_remote(destination):
            output_root, filename = destination.rsplit("/", 1)
            local_path = Path(context.output_tree(output_root)) / filename
        else:
            local_path = Path(destination).expanduser().resolve()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        flattened_stage = stage.Flatten()
        if not flattened_stage.Export(str(local_path)):
            raise RuntimeError(
                f"could not export flattened USD stage to {destination}",
            )
        context.state["stage.serialization_local_path"] = str(local_path)
        return {"serialization_path": destination}

    def configure_renderer(
        self,
        context: IsaacExecutionContext,
        parameters: RendererParameters,
    ) -> dict[str, object]:
        """Retain renderer intent for render-product initialization."""
        return self._store_configuration(context, "capture.renderer", parameters)

    def configure_camera(
        self,
        context: IsaacExecutionContext,
        parameters: CameraParameters,
    ) -> dict[str, object]:
        """Retain camera randomization intent for capture."""
        return self._store_configuration(context, "capture.camera", parameters)

    def configure_scan_grid(
        self,
        context: IsaacExecutionContext,
        parameters: ScanGridParameters,
    ) -> dict[str, object]:
        """Retain scan-grid intent for capture."""
        return self._store_configuration(context, "capture.scan_grid", parameters)

    def configure_roi_selection(
        self,
        context: IsaacExecutionContext,
        parameters: RoiSelectionParameters,
    ) -> dict[str, object]:
        """Retain stacked component-or-pin selection for focused capture.

        Args:
            context: Mutable Isaac execution context.
            parameters: Validated ROI selection configuration.

        Returns:
            Stored ROI configuration.

        """
        return self._store_configuration(
            context,
            "capture.roi_selection",
            parameters,
        )

    def configure_capture_limits(
        self,
        context: IsaacExecutionContext,
        parameters: CaptureLimitParameters,
    ) -> dict[str, object]:
        """Retain trigger and image limits for capture."""
        return self._store_configuration(context, "capture.limits", parameters)

    def configure_capture_visibility(
        self,
        context: IsaacExecutionContext,
        parameters: CaptureVisibilityParameters,
    ) -> dict[str, object]:
        """Retain geometry visibility intent for one capture pass."""
        return self._store_configuration(
            context,
            "capture.visibility",
            parameters,
        )

    def configure_writer(
        self,
        context: IsaacExecutionContext,
        parameters: CaptureWriterParameters,
    ) -> dict[str, object]:
        """Retain writer and annotator intent for capture."""
        return self._store_configuration(context, "capture.outputs", parameters)

    def configure_augmentation(
        self,
        context: IsaacExecutionContext,
        parameters: AugmentationParameters,
    ) -> dict[str, object]:
        """Retain post-render augmentation intent for capture."""
        return self._store_configuration(
            context,
            "capture.augmentation",
            parameters,
        )

    async def render_capture(
        self,
        context: IsaacExecutionContext,
        parameters: CaptureRenderParameters,
    ) -> dict[str, object]:
        """Render the configured scan directly through Isaac SDG APIs.

        Args:
            context: Mutable execution context containing prepared stage state.
            parameters: Validated capture destination.

        Returns:
            Structured capture artifact and frame information.

        """
        from .capture_execution import execute_capture

        local_output = context.output_tree(parameters.output_dir)
        pass_id = context.active_order_id or "capture.render"
        context.manifest.begin_capture_pass(
            pass_id,
            output_uri=str(parameters.output_dir),
            local_root=local_output,
        )
        local_parameters = parameters.model_copy(
            update={"output_dir": local_output},
        )
        try:
            output = await execute_capture(context, local_parameters)
        finally:
            for key in tuple(context.state):
                if key.startswith("capture."):
                    context.state.pop(key)
        output["output_dir"] = parameters.output_dir
        output["dataset_id"] = context.manifest_view.dataset_id
        output["dataset_index"] = context.join_output_location(
            str(parameters.output_dir),
            "dataset_index.json",
        )
        trigger_directories = output.get("trigger_directories")
        if isinstance(trigger_directories, list):
            output["trigger_directories"] = [
                context.published_output_path(parameters.output_dir, path)
                for path in trigger_directories
            ]
        context.state["storage.capture_output_uri"] = parameters.output_dir
        context.state["storage.capture_local_output"] = local_output
        return output

    def crop_capture(
        self,
        context: IsaacExecutionContext,
        parameters: PostprocessCropParameters,
    ) -> dict[str, object]:
        """Crop captured artifacts through the existing crop facade.

        Args:
            context: Mutable execution context containing capture output.
            parameters: Validated crop intent.

        Returns:
            Crop output directory and execution status.

        Raises:
            ValidationError: If capture has not completed successfully, or if
                several named passes rendered and none of them is the default.

        """
        capture_order_id, capture_output = _resolve_crop_capture_output(context)
        try:
            manifest_pass = context.manifest_view.capture_pass(capture_order_id)
            output_dir = manifest_pass.get("output_uri")
            trigger_metadata = context.manifest_view.trigger_metadata(capture_order_id)
        except KeyError:
            output_dir = capture_output.get("output_dir")
            trigger_metadata = capture_output.get("trigger_metadata")
        pipeline_type = capture_output.get("pipeline_type")
        if not isinstance(output_dir, str) or not isinstance(pipeline_type, str):
            raise ValidationError(
                f"{capture_order_id} output is missing output_dir or pipeline_type",
            )

        crop_dir = _REPO_ROOT / "scripts" / "crop"
        crop_dir_text = str(crop_dir)
        if crop_dir_text not in sys.path:
            sys.path.insert(0, crop_dir_text)
        importlib.invalidate_caches()
        pipeline_adapter = importlib.import_module("pipeline_adapter")
        crop = parameters.model_dump(mode="python", exclude_none=True)
        crop["enabled"] = True
        local_output = context.local_output_tree(output_dir)
        trigger_indices = (
            sorted(
                {
                    int(metadata["trigger_idx"])
                    for metadata in trigger_metadata
                    if isinstance(metadata, dict) and "trigger_idx" in metadata
                }
            )
            if isinstance(trigger_metadata, list)
            else []
        )
        if not trigger_indices:
            trigger_indices = [0]
        derived_artifacts = []
        for trigger_index in trigger_indices:
            trigger_crop = dict(crop)
            artifact_path = parameters.output_subdir
            if len(trigger_indices) > 1:
                artifact_path = (
                    f"{parameters.output_subdir}/trigger_{trigger_index:04d}"
                )
                trigger_crop["output_subdir"] = artifact_path
            config = {
                "pipeline_type": pipeline_type,
                "output": local_output,
                "trigger_index": trigger_index,
                "crop": trigger_crop,
            }
            pipeline_adapter.run_pipeline_crop(config)
            derived_artifacts.append(
                {
                    "kind": "crop_collection",
                    "path": artifact_path,
                    "method": parameters.method,
                    "trigger_idx": trigger_index,
                }
            )
        try:
            context.manifest.record_derived_artifacts(
                capture_order_id,
                derived_artifacts,
            )
        except KeyError:
            pass
        return {
            "output_dir": context.join_output_location(
                output_dir,
                parameters.output_subdir,
            ),
            "executed": True,
        }

    def close_capture_openings(
        self,
        context: IsaacExecutionContext,
        parameters: PostprocessCloseOpeningsParameters,
    ) -> dict[str, object]:
        """Close ring-shaped labels in one capture pass into their openings.

        Args:
            context: Mutable execution context containing capture output.
            parameters: Validated closure intent.

        Returns:
            Per-trigger counts of openings closed and pixels added.

        Raises:
            ValidationError: If the selected capture pass did not succeed.

        """
        capture_order_id = (
            "capture.render"
            if parameters.source_pass == "default"
            else f"capture.{parameters.source_pass}.render"
        )
        capture_output = context.outputs.get(capture_order_id)
        if not isinstance(capture_output, dict):
            raise ValidationError(
                f"postprocess.close_openings requires successful {capture_order_id}",
            )
        try:
            manifest_pass = context.manifest_view.capture_pass(capture_order_id)
            output_dir = manifest_pass.get("output_uri")
            trigger_metadata = context.manifest_view.trigger_metadata(capture_order_id)
        except KeyError:
            output_dir = capture_output.get("output_dir")
            trigger_metadata = capture_output.get("trigger_metadata")
        if not isinstance(output_dir, str) or not isinstance(trigger_metadata, list):
            raise ValidationError(
                f"{capture_order_id} output is missing output_dir or trigger_metadata",
            )

        from ...postprocess.opening_closure import close_openings

        local_output = Path(context.local_output_tree(output_dir))
        destination = local_output / parameters.output_subdir
        masks = sorted(
            destination.glob(f"{Path(parameters.output_filename).stem}*.png")
        )
        if not masks:
            raise ValidationError(
                f"postprocess.close_openings found no {parameters.output_filename} "
                f"in {destination}; declare it after the stitch that writes it",
            )
        reports = [
            close_openings(
                mask,
                class_name=parameters.class_name,
                max_diameter_px=parameters.max_diameter_px,
                min_circularity=parameters.min_circularity,
            )
            for mask in masks
        ]
        try:
            context.manifest.record_derived_artifacts(
                capture_order_id,
                [
                    {
                        "kind": "closed_semantic_mask",
                        "path": mask.relative_to(local_output).as_posix(),
                    }
                    for mask in masks
                ],
            )
        except KeyError:
            pass
        return {
            "class": parameters.class_name,
            "masks": [str(mask) for mask in masks],
            "openings_closed": sum(int(r["openings_closed"]) for r in reports),
            "pixels_added": sum(int(r["pixels_added"]) for r in reports),
        }

    def stitch_capture(
        self,
        context: IsaacExecutionContext,
        parameters: PostprocessStitchParameters,
    ) -> dict[str, object]:
        """Stitch one capture pass using its canonical pixel layout.

        Args:
            context: Mutable execution context containing capture output.
            parameters: Validated stitch intent.

        Returns:
            Stitched image paths, canvas sizes, and tile counts.

        Raises:
            ValidationError: If the selected capture pass did not succeed.

        """
        capture_order_id = (
            "capture.render"
            if parameters.source_pass == "default"
            else f"capture.{parameters.source_pass}.render"
        )
        capture_output = context.outputs.get(capture_order_id)
        if not isinstance(capture_output, dict):
            raise ValidationError(
                f"postprocess.stitch requires successful {capture_order_id}",
            )
        try:
            manifest_pass = context.manifest_view.capture_pass(capture_order_id)
            output_dir = manifest_pass.get("output_uri")
            trigger_metadata = context.manifest_view.trigger_metadata(capture_order_id)
        except KeyError:
            output_dir = capture_output.get("output_dir")
            trigger_metadata = capture_output.get("trigger_metadata")
        if not isinstance(output_dir, str) or not isinstance(trigger_metadata, list):
            raise ValidationError(
                f"{capture_order_id} output is missing output_dir or trigger_metadata",
            )

        from ...postprocess.image_stitching import stitch_capture_grid

        local_output = context.local_output_tree(output_dir)
        artifacts = stitch_capture_grid(
            local_output,
            trigger_metadata,
            image_type=parameters.image_type,
            output_subdir=parameters.output_subdir,
            output_filename=parameters.output_filename,
            blend=parameters.blend,
            write_labels=parameters.write_labels,
            recolor_tiles=parameters.recolor_tiles,
            color_map=parameters.color_map,
        )
        published = []
        derived_artifacts = []
        for artifact in artifacts:
            published_artifact = {
                **artifact,
                "path": context.published_output_path(
                    output_dir,
                    str(artifact["path"]),
                ),
            }
            labels_path = artifact.get("labels_path")
            if isinstance(labels_path, str):
                published_artifact["labels_path"] = context.published_output_path(
                    output_dir,
                    labels_path,
                )
            published.append(published_artifact)
            derived_artifacts.append(
                {
                    "kind": f"stitched_{parameters.image_type}",
                    "path": Path(str(artifact["path"]))
                    .relative_to(Path(local_output))
                    .as_posix(),
                    "trigger_idx": artifact["trigger_idx"],
                    "canvas_size_px": artifact["canvas_size_px"],
                    "tile_count": artifact["tile_count"],
                }
            )
            if isinstance(labels_path, str):
                derived_artifacts.append(
                    {
                        "kind": f"stitched_{parameters.image_type}_labels",
                        "path": Path(labels_path)
                        .relative_to(Path(local_output))
                        .as_posix(),
                        "trigger_idx": artifact["trigger_idx"],
                    }
                )
        try:
            context.manifest.record_derived_artifacts(
                capture_order_id,
                derived_artifacts,
            )
        except KeyError:
            pass
        return {
            "source_pass": parameters.source_pass,
            "image_type": parameters.image_type,
            "artifacts": published,
            "executed": True,
        }

    def format_roi_output(
        self,
        context: IsaacExecutionContext,
        parameters: PostprocessFormatRoiOutputParameters,
    ) -> dict[str, object]:
        """Format capture outputs to the AOI ROI output contract.

        Args:
            context: Mutable execution context containing capture outputs.
            parameters: Validated formatting intent.

        Returns:
            Trigger count, total patch count, and acceptance status.

        Raises:
            ValidationError: If either capture pass did not succeed or
                RGB/ROI patch counts or dimensions mismatch.

        """
        rgb_order_id = f"capture.{parameters.rgb_pass}.render"
        roi_order_id = f"capture.{parameters.roi_pass}.render"
        rgb_output = context.outputs.get(rgb_order_id)
        roi_output = context.outputs.get(roi_order_id)
        if not isinstance(rgb_output, dict):
            raise ValidationError(
                f"postprocess.format_roi_output requires successful {rgb_order_id}",
            )
        if not isinstance(roi_output, dict):
            raise ValidationError(
                f"postprocess.format_roi_output requires successful {roi_order_id}",
            )
        try:
            rgb_pass = context.manifest_view.capture_pass(rgb_order_id)
            roi_pass = context.manifest_view.capture_pass(roi_order_id)
            rgb_dir = rgb_pass.get("output_uri")
            roi_dir = roi_pass.get("output_uri")
            trigger_metadata_rgb = context.manifest_view.trigger_metadata(rgb_order_id)
            trigger_metadata_roi = context.manifest_view.trigger_metadata(roi_order_id)
        except KeyError:
            rgb_dir = rgb_output.get("output_dir")
            roi_dir = roi_output.get("output_dir")
            trigger_metadata_rgb = rgb_output.get("trigger_metadata")
            trigger_metadata_roi = roi_output.get("trigger_metadata")
        if not (
            isinstance(rgb_dir, str)
            and isinstance(roi_dir, str)
            and isinstance(trigger_metadata_rgb, list)
            and isinstance(trigger_metadata_roi, list)
        ):
            raise ValidationError(
                "capture outputs are missing output_dir or trigger_metadata",
            )

        from pcb_aoi.postprocess.roi_output import format_aoi_roi_output

        result = format_aoi_roi_output(
            context.local_output_tree(rgb_dir),
            context.local_output_tree(roi_dir),
            trigger_metadata_rgb,
            trigger_metadata_roi,
        )
        path_mapping = result.pop("path_mapping", {})
        if isinstance(path_mapping, dict):
            try:
                context.manifest.remap_artifacts(roi_order_id, path_mapping)
            except KeyError:
                pass
        derived = result.get("artifacts")
        if isinstance(derived, list):
            try:
                context.manifest.record_derived_artifacts(roi_order_id, derived)
            except KeyError:
                pass
        for pass_id in (rgb_order_id, roi_order_id):
            try:
                context.manifest.rewrite_trigger_documents(pass_id)
            except KeyError:
                pass
        return result

    def generate_joint_rois(
        self,
        context: IsaacExecutionContext,
        parameters: PostprocessGenerateJointRoisParameters,
    ) -> dict[str, object]:
        """Generate ``joint_rois.json`` from tight bounding box outputs.

        Args:
            context: Mutable execution context containing capture output.
            parameters: Validated generation intent.

        Returns:
            Total joint count and execution status.

        Raises:
            ValidationError: If the ROI capture pass did not succeed or
                bounding box files cannot be read.

        """
        rgb_order_id = f"capture.{parameters.rgb_pass}.render"
        roi_order_id = f"capture.{parameters.roi_pass}.render"
        rgb_output = context.outputs.get(rgb_order_id)
        roi_output = context.outputs.get(roi_order_id)
        if not isinstance(rgb_output, dict):
            raise ValidationError(
                f"postprocess.generate_joint_rois requires successful {rgb_order_id}",
            )
        if not isinstance(roi_output, dict):
            raise ValidationError(
                f"postprocess.generate_joint_rois requires successful {roi_order_id}",
            )
        try:
            rgb_pass = context.manifest_view.capture_pass(rgb_order_id)
            roi_pass = context.manifest_view.capture_pass(roi_order_id)
            rgb_dir = rgb_pass.get("output_uri")
            roi_dir = roi_pass.get("output_uri")
            trigger_metadata_roi = context.manifest_view.trigger_metadata(roi_order_id)
        except KeyError:
            rgb_dir = rgb_output.get("output_dir")
            roi_dir = roi_output.get("output_dir")
            trigger_metadata_roi = roi_output.get("trigger_metadata")
        if not (
            isinstance(rgb_dir, str)
            and isinstance(roi_dir, str)
            and isinstance(trigger_metadata_roi, list)
        ):
            raise ValidationError(
                "capture outputs are missing output_dir or ROI trigger_metadata",
            )

        from pcb_aoi.postprocess.joint_rois import (
            _component_metadata_by_ref,
            build_pad_joint_keys,
            generate_joint_rois,
        )

        instance_label_by_segment: dict[str, str] = {}
        cohort = context.board_instance_cohort
        if cohort is not None:
            for index, path in enumerate(cohort.instance_paths):
                segment = path.rstrip("/").rsplit("/", 1)[-1]
                instance_label_by_segment[segment] = str(index)

        component_metadata: dict[str, dict[str, str | None]] = {}
        pad_joint_keys: dict[str, tuple[str, str]] = {}
        if context.board_metadata:
            catalog = _board_metadata(context, source=None)
            component_metadata = _component_metadata_by_ref(catalog.components)
            # Stated pad/hole -> component pin associations widen each joint's
            # box to its pad. The catalog is read from the cohort's prototype
            # source, so its paths are re-rooted under every placement.
            pad_joint_keys = build_pad_joint_keys(
                (*catalog.pads, *catalog.holes),
                source_root=cohort.prototype_source_path if cohort else None,
                instance_paths=cohort.instance_paths if cohort else (),
            )

        result = generate_joint_rois(
            context.local_output_tree(roi_dir),
            trigger_metadata_roi,
            rgb_pass_name=parameters.rgb_pass,
            roi_pass_name=parameters.roi_pass,
            output_roots_by_pass={
                parameters.rgb_pass: context.local_output_tree(rgb_dir),
                parameters.roi_pass: context.local_output_tree(roi_dir),
            },
            instance_label_by_segment=instance_label_by_segment,
            component_metadata=component_metadata,
            pad_joint_keys=pad_joint_keys,
        )
        pin_only = result.get("pin_only_joint_count")
        if isinstance(pin_only, int) and pin_only:
            warnings.warn(
                f"postprocess.generate_joint_rois: {pin_only} of "
                f"{result.get('joint_count')} joint(s) have no associated pad box; "
                "their joint_bbox_xyxy equals bbox_xyxy (joint_bbox_source "
                "'pin_only'). Label pads in the ROI pass "
                "(annotations.geometry.pads) and check the board's stated "
                "pad-to-pin associations.",
                RuntimeWarning,
                stacklevel=2,
            )
        artifacts = result.get("artifacts")
        if isinstance(artifacts, list):
            artifacts_by_order: dict[str, list[dict[str, object]]] = {}
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                capture_pass = artifact.get("capture_pass")
                if not isinstance(capture_pass, str):
                    continue
                artifact_record = {
                    key: value
                    for key, value in artifact.items()
                    if key != "capture_pass"
                }
                artifacts_by_order.setdefault(
                    f"capture.{capture_pass}.render",
                    [],
                ).append(artifact_record)
            for pass_id, pass_artifacts in artifacts_by_order.items():
                try:
                    context.manifest.record_derived_artifacts(
                        pass_id,
                        pass_artifacts,
                    )
                except KeyError:
                    pass
        return result

    def apply_component_annotations(
        self,
        context: IsaacExecutionContext,
        parameters: ComponentAnnotationParameters,
    ) -> dict[str, object]:
        """Apply baseline component semantics through the existing SDG API.

        Args:
            context: Mutable execution context.
            parameters: Validated component-annotation intent.

        Returns:
            Number of component prims labeled.

        Raises:
            ValidationError: If component selection has not run.

        """
        stage = context.require_stage()
        if not context.component_pool and not context.component_types:
            raise ValidationError(
                "component annotations require scene.components.select first",
            )
        root_path = context.scene.get("board_root_path") or context.scene.get(
            "root_path",
        )
        if not isinstance(root_path, str):
            raise ValidationError("scene.load did not provide board_root_path")
        cohort = context.board_instance_cohort
        if cohort is not None:
            from pcb_aoi.board_metadata import MetadataCapability

            catalog = _board_metadata(
                context,
                parameters.source,
                (MetadataCapability.COMPONENTS,),
            )
            projection = build_component_semantic_projection(
                cohort,
                catalog,
                context.component_pool,
                parameters.classes,
            )
            context.instance_semantic_projection = projection
            return {
                "count": len(projection.components) * len(cohort.instance_paths),
                "source_count": len(projection.components),
                "instance_count": len(cohort.instance_paths),
                "mode": "instance_id_projection",
            }
        common = _standalone_module("common")
        classes = (
            {name: list(refdes) for name, refdes in parameters.classes.items()}
            if parameters.classes
            else None
        )
        count = common.apply_semantics(
            stage,
            root_path,
            list(context.component_types),
            classes,
            component_paths=list(context.component_pool),
        )
        return {"count": int(count)}

    def apply_geometry_annotations(
        self,
        context: IsaacExecutionContext,
        parameters: GeometryAnnotationParameters,
    ) -> dict[str, object]:
        """Label board geometry with semantic classes for segmentation.

        Component semantics label a whole component scope, so a pin and the
        solder around it collapse into one class. Both are labeled here at the
        geometry that carries them: the solder operation's own output root, and
        the terminal prims the metadata catalog resolves. Replicator semantics
        are inherited, so labeling the solder root covers every joint beneath it
        without enumerating them.  Barrel prims (``pcba:mesh:role = "board_barrel"``)
        are resolved by stage traversal rather than the catalog.

        When ``parameters.components`` or ``parameters.component_filter`` is
        set, catalog-resolved targets (pads, terminals, holes) and barrel
        traversal are restricted to prims whose component association matches
        the selection. Otherwise the scene component pool is the scope, and a
        record whose owning component the catalog could not resolve is kept:
        a component scope cannot exclude geometry that belongs to no component,
        and boards routinely carry such holes. Solder is not filtered because
        it is resolved at attach time via a separate mechanism.

        Args:
            context: Mutable execution context.
            parameters: Validated geometry-annotation intent.

        Returns:
            Prim counts labeled per geometry target.

        Raises:
            ValidationError: If a requested target has no geometry to label.

        """
        stage = context.require_stage()
        common = _standalone_module("common")
        result: dict[str, object] = {}

        if parameters.solder is not None:
            # Annotation orders are compiled ahead of the operations that build
            # geometry, so solder does not exist yet. The class is recorded for
            # the attach step, which already refreshes authored semantics over
            # the prims it just created.
            context.state["annotations.geometry.solder"] = parameters.solder
            result["solder"] = "pending_solder_attachment"

        # which catalog records back a target is domain vocabulary
        from pcb_aoi.board_metadata import MetadataCapability
        from pcb_aoi.geometry_labels import CATALOG_TARGETS, catalog_target

        # Resolve the allowed component prim paths when a filter is active.
        comp_catalog = _board_metadata(
            context,
            None,
            (MetadataCapability.COMPONENTS,),
        )
        parent_paths = context.component_parent_paths()
        allowed_comp_paths = None if parent_paths is None else frozenset(parent_paths)
        # An explicit selector names the components whose geometry is wanted,
        # so a record with no resolved owner cannot satisfy it. The implicit
        # scene scope only narrows the board to the selected components; a
        # record the catalog could not attribute to any component (mechanical
        # holes, holes whose owner association is missing) is not excluded by
        # it, or a plain ``label_geometry(holes=...)`` would label nothing on
        # boards whose holes carry no component association.
        explicit_scope = parameters.component_filter is not None or bool(
            parameters.components
        )
        if parameters.component_filter is not None:
            from pcb_aoi.board_metadata import select_components

            allowed_comp_paths = frozenset(
                component.prim_path
                for component in select_components(
                    comp_catalog,
                    parameters.component_filter,
                    parent_paths=parent_paths,
                    where="annotations.geometry.component_filter",
                )
            )
        elif parameters.components:
            refdes_set = frozenset(parameters.components)
            allowed_comp_paths = frozenset(
                component.prim_path
                for component in comp_catalog.components
                if (
                    allowed_comp_paths is None
                    or component.prim_path in allowed_comp_paths
                )
                and component.refdes in refdes_set
            )

        for target in sorted(CATALOG_TARGETS):
            class_name = getattr(parameters, target, None)
            if class_name is None:
                continue
            capability, attribute = catalog_target(target)
            catalog = _board_metadata(context, None, (capability,))
            all_records = getattr(catalog, attribute)
            if allowed_comp_paths is not None:
                records = [
                    r
                    for r in all_records
                    if r.component_path in allowed_comp_paths
                    or (r.component_path is None and not explicit_scope)
                ]
            else:
                records = list(all_records)
            paths = sorted({record.prim_path for record in records})
            if not paths:
                filter_note = (
                    " within the effective component filter"
                    if allowed_comp_paths is not None
                    else ""
                )
                raise ValidationError(
                    f"geometry annotations for {target} found no {target} "
                    f"records in the board metadata{filter_note}",
                )
            labeled_count = _label_geometry_prims(
                stage,
                common,
                paths,
                class_name,
            )
            result[target] = labeled_count
            if labeled_count:
                mark_instanced_stage_mutation(context, "annotations.geometry")
            result[f"{target}_records"] = len(paths)
            result[f"{target}_projected"] = _project_geometry_class(
                context,
                paths,
                class_name,
                target,
            )

        # Label role-resolved geometry by stage traversal. Which targets exist,
        # what role token their prims carry, how the owning component is
        # reached and whether they are prototype geometry is domain vocabulary,
        # so it arrives through the registered geometry-role hook and this
        # runtime consumes targets it does not have to recognise.
        from cdf.execution.isaac.domain_adapters import (
            OWNERSHIP_ANCESTOR,
            geometry_roles,
        )

        role_specs = geometry_roles()
        requested_roles = {
            target: getattr(parameters, target, None)
            for target in role_specs
            if getattr(parameters, target, None) is not None
        }
        if requested_roles:
            from pcb_aoi.board_metadata.usd_schema.stage_traversal import (
                iter_prim_subtree,
            )
            from pcb_aoi.board_metadata.usd_schema_metadata_keys import (
                MESH_ROLE_ATTR,
                USD_SCHEMA_METADATA_KEYS,
            )

            comp_rel_names = USD_SCHEMA_METADATA_KEYS[
                "board_feature.component_relationship"
            ]
            board_root = str(
                context.scene.get("board_root_path") or context.scene.get("root_path")
            )
            board_root_prim = stage.GetPrimAtPath(board_root)
            role_to_target = {
                role_specs[target].role: target for target in requested_roles
            }
            catalog_component_paths = frozenset(
                component.prim_path for component in comp_catalog.components
            )
            paths_by_target: dict[str, list[str]] = {
                target: [] for target in requested_roles
            }
            for prim in iter_prim_subtree(board_root_prim):
                attr = prim.GetAttribute(MESH_ROLE_ATTR)
                role = attr.Get() if attr else None
                target = role_to_target.get(role)
                if target is None:
                    continue
                if allowed_comp_paths is not None:
                    comp_path = None
                    if role_specs[target].ownership == OWNERSHIP_ANCESTOR:
                        # The prim hangs under its component; the nearest
                        # ancestor that is a component is the owner.
                        ancestor = prim.GetParent()
                        while ancestor and ancestor.IsValid():
                            if str(ancestor.GetPath()) in catalog_component_paths:
                                comp_path = str(ancestor.GetPath())
                                break
                            ancestor = ancestor.GetParent()
                    else:
                        for rel_name in comp_rel_names:
                            rel = prim.GetRelationship(rel_name)
                            targets = rel.GetTargets() if rel else []
                            if targets:
                                comp_path = str(targets[0])
                                break
                    # Same rule as the catalog records above: an unowned prim
                    # is only excluded by an explicit selector.
                    if comp_path is None:
                        if explicit_scope:
                            continue
                    elif comp_path not in allowed_comp_paths:
                        continue
                paths_by_target[target].append(str(prim.GetPath()))

            for target, class_name in requested_roles.items():
                paths = paths_by_target[target]
                if not paths:
                    continue
                labeled = _label_geometry_prims(stage, common, paths, class_name)
                if labeled:
                    mark_instanced_stage_mutation(context, "annotations.geometry")
                # Same contract as the catalog branch above: result[target] is
                # how many prims were labelled, and the number of prims the
                # resolution found goes to "<target>_records" beside it.
                result[target] = labeled
                result[f"{target}_records"] = len(paths)
                if role_specs[target].projected:
                    # Prototype geometry, so an instanced cohort reads the
                    # class through the projection, like terminals.
                    result[f"{target}_projected"] = _project_geometry_class(
                        context,
                        paths,
                        class_name,
                        target,
                    )

        return result

    @staticmethod
    def _require_structural_operation_state(
        context: IsaacExecutionContext,
        operation_name: str,
    ) -> None:
        """Require component selection and the supported semantic namespace.

        Args:
            context: Mutable execution context.
            operation_name: Structural operation being validated.

        Raises:
            ValidationError: If required component or annotation state is absent.

        """
        context.require_stage()
        if not context.component_pool:
            raise ValidationError(
                f"{operation_name} requires scene.components.select first",
            )
        labels = context.state.get("annotations.operations") or {}
        namespace = labels.get("namespace", "defect")
        if namespace != "defect":
            raise ValidationError(
                "existing structural operation APIs support operation-label "
                f"namespace 'defect', got {namespace!r}",
            )

    @staticmethod
    def _operation_board_metadata(
        context: IsaacExecutionContext,
        source: str | None = None,
    ):
        """Return source-neutral metadata used by structural defect geometry.

        Args:
            context: Mutable execution context.
            source: Explicit metadata name, or None to use embedded USD metadata.

        Returns:
            Loaded board metadata catalog.

        Raises:
            ValidationError: If an explicit metadata source is not loaded.

        """
        if source is None and not context.board_metadata:
            return None
        return _board_metadata(context, source)

    @staticmethod
    def _effective_component_pool(
        context: IsaacExecutionContext,
        component_filter: object | None,
        *,
        source: str | None = None,
        where: str,
    ) -> tuple[str, ...]:
        """Intersect an operation filter with the scene component pool.

        Args:
            context: Mutable execution context.
            component_filter: Optional normalized child filter.
            source: Metadata source used to evaluate the filter.
            where: Configuration location used in validation messages.

        Returns:
            Effective component paths in stable catalog order.

        """
        parent = context.component_parent_paths()
        if component_filter is None:
            return tuple(parent or ())
        from pcb_aoi.board_metadata import MetadataCapability, select_components

        catalog = _board_metadata(
            context,
            source,
            (MetadataCapability.COMPONENTS,),
        )
        selected = select_components(
            catalog,
            component_filter,
            parent_paths=parent,
            where=where,
        )
        return tuple(component.prim_path for component in selected)

    def _apply_transform_operation(
        self,
        context: IsaacExecutionContext,
        operation_name: str,
        defect_parameters: dict[str, object],
        *,
        eligibility_source: str | None = None,
        component_filter: object | None = None,
    ) -> dict[str, object]:
        """Apply one transform operation and aggregate its defect records.

        Args:
            context: Mutable execution context.
            operation_name: Existing ``defect_ops`` operation name.
            defect_parameters: Normalized operation configuration.
            eligibility_source: Optional metadata source used for eligibility.
            component_filter: Optional child filter intersected with the scene pool.

        Returns:
            Applied defect count and selected component paths.

        """
        self._require_structural_operation_state(context, operation_name)
        if context.state.get("operation.missing.paths"):
            raise ValidationError(
                "missing cannot be combined with transform structural operations",
            )

        stage = context.require_stage()
        board_metadata = self._operation_board_metadata(
            context,
            eligibility_source,
        )
        sim_manifest = next(
            (
                dict(document.data)
                for document in context.json_documents.values()
                if document.kind == "pcb.sim_manifest"
            ),
            None,
        )

        import numpy as np

        defect_ops = _standalone_module("defect_ops")
        defect_context = defect_ops.build_defect_context(
            stage,
            sim_manifest,
            board_metadata,
        )
        existing_records = list(
            context.state.get("operation.structural.records") or (),
        )
        used_paths = {record["path"] for record in existing_records}
        operation_index = int(context.state.get("operation.structural.count", 0))
        np.random.seed(
            int(context.plan_metadata.get("random_seed", 0)) + operation_index,
        )
        component_pool = self._effective_component_pool(
            context,
            component_filter,
            source=eligibility_source,
            where=f"operations.{operation_name}.component_filter",
        )
        records = defect_ops.prepare_defects(
            list(component_pool),
            {operation_name: defect_parameters},
            defect_context,
            excluded_paths=used_paths,
            apply_semantics=context.instance_semantic_projection is None,
        )
        # prepare_defects moves the components with Replicator's pose_ops,
        # which writes Fabric (the renderer's view) but never the USD stage.
        # Every USD reader downstream -- solder-generation collect, attach
        # placement, stage Flatten/serialization -- would silently see the
        # NOMINAL poses. Author the stored final transforms onto the stage as
        # well; Kit's USD->Fabric sync keeps the rendered view identical.
        # Focused ROI crops still frame each part where the board places it,
        # so measure that first: once authored, the placed pose is gone.
        from .capture_execution import _record_placed_bounds

        _record_placed_bounds(context, stage, records)
        self._author_defect_transforms(stage, records)
        if not records and component_pool:
            # Never a legitimate outcome: the caller asked for an operation
            # over a non-empty pool and got no operation. Say WHY before
            # blaming the board -- an empty eligible pool is the common case
            # and reads nothing like a metadata mismatch.
            raise ValidationError(
                self._describe_empty_transform_result(
                    context,
                    component_pool,
                    defect_ops,
                    defect_context,
                    operation_name,
                    defect_parameters,
                    used_paths,
                ),
            )
        existing_records.extend(records)
        context.state["operation.structural.records"] = existing_records
        context.state["operation.structural.count"] = operation_index + 1
        context.state[f"operation.{operation_name}.records"] = records
        context.manifest.observe_defects(records, context.entity_identity)
        return {
            "count": len(records),
            "requested": defect_ops.describe_targeting(defect_parameters),
            "component_paths": [record["path"] for record in records],
        }

    @staticmethod
    def _describe_empty_transform_result(
        context: IsaacExecutionContext,
        component_pool: object,
        defect_ops: object,
        defect_context: object,
        operation_name: str,
        defect_parameters: dict[str, object],
        used_paths: set[str],
    ) -> str:
        """Explain why a transform operation over a non-empty pool applied nothing.

        ``prepare_defects`` skips an operation whose eligible pool is empty
        with an info log and returns no records, which used to surface as
        "prim paths do not resolve" -- a diagnosis that sends the user off to
        check the board while the real cause is in the config. Three causes
        are told apart, most common first:

        1. Earlier structural operations already consumed every component
           the operation could have used (seed-driven ``shift``/``sideflip``
           draw first and can take the one component a later ``tombstone``
           or ``reverse_polarity`` pinned by ``component_types``).
        2. ``component_types`` matches no component in the pool at all.
        3. Components were selected but their prims do not resolve on this
           stage -- the original, metadata-mismatch diagnosis.

        Args:
            context: Execution context carrying the scene root.
            component_pool: Component paths the operation drew from (after filters).
            defect_ops: Loaded standalone ``defect_ops`` module.
            defect_context: Shared defect metadata from ``build_defect_context``.
            operation_name: Structural operation that applied nothing.
            defect_parameters: Normalized operation configuration.
            used_paths: Component paths already assigned by earlier operations.

        Returns:
            Human-readable failure reason for a ``ValidationError``.

        """
        pool = [str(path) for path in component_pool]
        prefix = (
            f"{operation_name} selected from {len(pool)} components but "
            "applied none of them. "
        )
        remaining = [path for path in pool if path not in used_paths]
        if not remaining:
            return prefix + (
                "Every component in the pool was already assigned by an "
                "earlier structural operation "
                f"({sorted(used_paths)}); structural operations cannot share "
                "a component. Enlarge scene.components.types or drop an "
                "operation."
            )
        eligible_ignoring_used = defect_ops._eligible_components(
            operation_name,
            defect_parameters,
            pool,
            set(),
            defect_context,
            log=False,
        )
        eligible_ignoring_used = [str(path) for path in eligible_ignoring_used]
        component_types = defect_parameters.get("component_types")
        if eligible_ignoring_used and all(
            path in used_paths for path in eligible_ignoring_used
        ):
            # component_types is optional: naming an empty selector here would
            # send the user to a knob they never set.
            taken = (
                f"Its component_types {list(component_types)!r} match only "
                f"components that earlier structural operations already took "
                f"({sorted(eligible_ignoring_used)})."
                if component_types
                else (
                    "Every component it is eligible for was already taken by "
                    "an earlier structural operation "
                    f"({sorted(eligible_ignoring_used)})."
                )
            )
            return (
                prefix
                + taken
                + (
                    " Seed-driven operations (shift, sideflip) draw before pinned "
                    "ones and can take a pinned component; pick a different "
                    "target, change random_seed, or reorder scene.components.types."
                )
            )
        if component_types and not eligible_ignoring_used:
            return prefix + (
                f"Its component_types {list(component_types)!r} match no "
                f"component in the pool ({pool}). A prim path must match "
                "exactly; any other token is matched as a substring of the "
                "component prim path."
            )
        return prefix + (
            "Their prim paths do not resolve under "
            f"{context.scene.get('board_root_path') or context.scene.get('root_path')!r} "
            "on this stage; "
            "component metadata that describes a different board (or a "
            "different export of it) is the usual cause."
        )

    @staticmethod
    def _author_defect_transforms(
        stage: object,
        records: list[dict[str, object]],
    ) -> None:
        """Write applied defect transforms onto the USD stage.

        Args:
            stage: Active USD stage the defects were applied to.
            records: Applied-defect records from ``defect_ops.prepare_defects``,
                each carrying the component path and a ``<defect>_final_flat``
                row-major local transform.

        """
        to_author = []
        for record in records:
            final = record.get(f"{record.get('defect')}_final_flat")
            if isinstance(final, (list, tuple)) and len(final) == 16:
                to_author.append((str(record["path"]), final))
        if not to_author:
            return
        # Imported only when there is something to author: real runs always
        # reach here (defect_ops records carry the final transform) and run
        # inside Isaac where pxr exists -- and if it somehow does not, this
        # still fails loud rather than silently skipping the write-back.
        from pxr import Gf, UsdGeom

        for path, final in to_author:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            # A single matrix op mirrors the shape pose_ops writes to Fabric
            # (and defect_ops' own restore path), replacing whatever op stack
            # the component was authored with.
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(
                Gf.Matrix4d(*(float(v) for v in final)),
            )

    def apply_shift(
        self,
        context: IsaacExecutionContext,
        parameters: ShiftParameters,
    ) -> dict[str, object]:
        """Apply placement shifts through the existing defect API.

        Args:
            context: Mutable execution context.
            parameters: Validated shift intent.

        Returns:
            Applied shift count and selected component paths.

        """
        return self._apply_transform_operation(
            context,
            "shift",
            {
                "enabled": True,
                **_defect_targeting(parameters),
                "translate_range": parameters.translate_range,
                "rotate_z_range": parameters.rotate_z_range,
                "variant_index": parameters.variant_index,
            },
            component_filter=parameters.component_filter,
        )

    def apply_tombstone(
        self,
        context: IsaacExecutionContext,
        parameters: TombstoneParameters,
    ) -> dict[str, object]:
        """Select and apply tombstones through the existing defect API.

        Args:
            context: Mutable execution context.
            parameters: Validated tombstone intent.

        Returns:
            Applied tombstone count and selected component paths.

        Raises:
            ValidationError: If required component metadata is absent.

        """
        angle_min, angle_max = parameters.angle_range
        return self._apply_transform_operation(
            context,
            "tombstone",
            {
                "enabled": True,
                **_defect_targeting(parameters),
                "angle_min": angle_min,
                "angle_max": angle_max,
                "component_types": (
                    list(parameters.component_types)
                    if parameters.component_types
                    else None
                ),
                "auto_filter": parameters.auto_filter,
            },
            eligibility_source=parameters.eligibility_source,
            component_filter=parameters.component_filter,
        )

    def apply_sideflip(
        self,
        context: IsaacExecutionContext,
        parameters: SideflipParameters,
    ) -> dict[str, object]:
        """Apply sideflips through the existing defect API.

        Args:
            context: Mutable execution context.
            parameters: Validated sideflip intent.

        Returns:
            Applied sideflip count and selected component paths.

        """
        angle_min, angle_max = parameters.angle_range
        return self._apply_transform_operation(
            context,
            "sideflip",
            {
                "enabled": True,
                **_defect_targeting(parameters),
                "angle_min": angle_min,
                "angle_max": angle_max,
                "variant_index": parameters.variant_index,
                "rotate_z_jitter_deg": parameters.rotate_z_jitter_deg,
                "slide_range_mm": parameters.slide_range_mm,
            },
            component_filter=parameters.component_filter,
        )

    def apply_reverse_polarity(
        self,
        context: IsaacExecutionContext,
        parameters: ReversePolarityParameters,
    ) -> dict[str, object]:
        """Apply reverse polarity through the existing defect API.

        Args:
            context: Mutable execution context.
            parameters: Validated reverse-polarity intent.

        Returns:
            Applied reverse-polarity count and selected component paths.

        """
        return self._apply_transform_operation(
            context,
            "reverse_polarity",
            {
                "enabled": True,
                **_defect_targeting(parameters),
                "component_types": (
                    list(parameters.component_types)
                    if parameters.component_types
                    else None
                ),
            },
            component_filter=parameters.component_filter,
        )

    def apply_missing(
        self,
        context: IsaacExecutionContext,
        parameters: MissingParameters,
    ) -> dict[str, object]:
        """Select and hide missing components through the existing API.

        Args:
            context: Mutable execution context.
            parameters: Validated missing-component intent.

        Returns:
            Selected missing-component count and paths.

        Raises:
            ValidationError: If transform defects already exist or no candidates remain.

        """
        self._require_structural_operation_state(context, "missing")
        if context.state.get("operation.structural.records"):
            raise ValidationError(
                "missing cannot be combined with transform structural operations",
            )
        if context.state.get("operation.missing.paths"):
            raise ValidationError("missing operation was already applied")

        import numpy as np

        missing_ops = _standalone_module("missing_ops")
        np.random.seed(int(context.plan_metadata.get("random_seed", 0)))
        available = list(
            self._effective_component_pool(
                context,
                parameters.component_filter,
                where="operations.missing.component_filter",
            ),
        )
        if not available:
            raise ValidationError("missing has no available components")
        paths = missing_ops.select_missing_components(
            available,
            _defect_targeting(parameters),
        )
        stage = context.require_stage()
        # Which hide to use is derived, not asked for. There are two, and they
        # are not interchangeable:
        #
        #   MakeInvisible  keeps the prim, so a semantic label authored on it
        #                  survives -- but the Fabric Scene Delegate beauty pass
        #                  does not reliably pick the toggle up mid-run, so the
        #                  part can still appear in RGB.
        #   SetActive(False)  prunes the subtree from USD and Fabric both, so it
        #                  is genuinely gone from RGB, depth and segmentation --
        #                  and takes any label authored on the prim with it.
        #
        # The choice therefore follows where the labels live, which the context
        # already knows. With a catalog projection the labels do not come from
        # the prim, so it is safe to prune and we get the hide that survives FSD.
        # Without one we author defect=missing on the prim and have to keep it.
        if context.instance_semantic_projection is None:
            missing_ops.apply_missing_semantics(stage, paths)
            missing_ops.hide_components(stage, paths)
        else:
            missing_ops.deactivate_components(stage, paths)
        context.state["operation.missing.paths"] = list(paths)
        context.manifest.observe_defects(
            ({"path": path, "defect": "missing"} for path in paths),
            context.entity_identity,
        )
        return {
            "count": len(paths),
            "component_paths": list(paths),
        }
