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


"""Own validated shared-PCB instance state inside Isaac execution.

The public API explicitly declares a complete board-placement cohort through
``scene.instances(...)``. This module validates that declaration against the
open USD stage and owns every runtime policy that depends on it: prototype
source resolution, source-to-instance path expansion, component projection
records, and the switch from native semantic capture to instance-ID
projection.

It does not discover panels on behalf of the public API. Without a validated
``BoardInstanceCohort``, callers must retain ordinary single-board behavior.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cdf.errors import ValidationError

if TYPE_CHECKING:
    from cdf.execution.isaac.execution_context import IsaacExecutionContext
    from pcb_aoi.board_metadata import BoardMetadataCatalog


def _reference_items(list_operation: object) -> tuple[object, ...]:
    """Return authored reference items from an Sdf list operation."""
    return tuple(
        item
        for field_name in (
            "explicitItems",
            "prependedItems",
            "addedItems",
            "appendedItems",
        )
        for item in getattr(list_operation, field_name, ())
    )


def _rebase_authored_path(
    authored_path: str,
    spec_path: object,
    composed_path: object,
) -> str:
    """Translate a layer-authored prim path into composed stage namespace.

    An internal reference is authored against the namespace of the layer
    that holds it. That layer is not always mounted at the same place it
    was authored for: a batch run normally opens the board as the root layer,
    while a parent stage may reference it under ``/World/Board_x`` and shift
    every prim in it down one level. The
    authored target still names the pre-shift path, so reading it verbatim
    asks the stage for a prim that only exists in the other namespace.

    The shift is recovered from the prim itself: its authored spec path and
    its composed path share a suffix, and the prefixes that differ ARE the
    reference's namespace mapping.

    Args:
        authored_path: Prim path as written in the authoring layer.
        spec_path: Path of the spec that authored it, in that layer.
        composed_path: Path the owning prim composed to on the stage.

    Returns:
        ``authored_path`` moved into the composed namespace, unchanged when
        the two namespaces coincide.

    """
    spec_parts = str(spec_path).strip("/").split("/")
    composed_parts = str(composed_path).strip("/").split("/")
    shared = 0
    while (
        shared < len(spec_parts)
        and shared < len(composed_parts)
        and spec_parts[-1 - shared] == composed_parts[-1 - shared]
    ):
        shared += 1
    old_prefix = "/" + "/".join(spec_parts[: len(spec_parts) - shared])
    new_prefix = "/" + "/".join(composed_parts[: len(composed_parts) - shared])
    if old_prefix == new_prefix:
        return authored_path
    if authored_path == old_prefix:
        return new_prefix
    if old_prefix != "/" and not authored_path.startswith(old_prefix + "/"):
        # Outside the shifted subtree, so the mapping says nothing about it.
        # Returning it unchanged keeps the caller's existence check as the
        # thing that reports the problem, with the path actually authored.
        return authored_path
    suffix = authored_path[len(old_prefix) :].lstrip("/")
    return new_prefix.rstrip("/") + "/" + suffix if suffix else new_prefix


def _internal_reference_source(prim: object) -> str | None:
    """Return the first stage-local prim referenced by an instance."""
    for prim_spec in prim.GetPrimStack():
        for reference in _reference_items(prim_spec.referenceList):
            asset_path = str(getattr(reference, "assetPath", "") or "")
            prim_path = str(getattr(reference, "primPath", "") or "")
            if not asset_path and prim_path.startswith("/"):
                return _rebase_authored_path(
                    prim_path,
                    prim_spec.path,
                    prim.GetPath(),
                )
    return None


@dataclass(frozen=True)
class BoardInstanceTarget:
    """One composed target expanded from a prototype-source path.

    Args:
        instance_index: Stable index within the registered cohort.
        instance_path: Composed prim path of the board placement.
        source_path: Prim path below the stage-local prototype source.
        composed_path: Source-relative prim path below the board placement.

    """

    instance_index: int
    #: Stable index within the registered cohort.
    instance_path: str
    #: Composed prim path of the board placement.
    source_path: str
    #: Prim path below the stage-local prototype source.
    composed_path: str
    #: Source-relative prim path below the board placement.


@dataclass(frozen=True)
class BoardInstanceCohort:
    """Validated complete cohort sharing one authorable USD prototype.

    Args:
        instance_paths: Ordered composed board-placement paths.
        representative_instance_path: Placement used for shared calculations.
        prototype_path: USD prototype path shared by every placement.
        prototype_source_path: Stage-local authorable source of the prototype.

    """

    instance_paths: tuple[str, ...]
    #: Ordered composed board-placement paths.
    representative_instance_path: str
    #: Placement used for shared calculations.
    prototype_path: str
    #: USD prototype path shared by every placement.
    prototype_source_path: str
    #: Stage-local authorable source of the prototype.

    def as_mapping(self) -> dict[str, object]:
        """Return transport-safe cohort details for execution output.

        Returns:
            Mapping containing the resolved cohort paths.

        """
        return {
            "instance_paths": self.instance_paths,
            "representative_instance_path": self.representative_instance_path,
            "prototype_path": self.prototype_path,
            "prototype_source_path": self.prototype_source_path,
        }

    def expand_source_path(self, source_path: str) -> tuple[BoardInstanceTarget, ...]:
        """Expand one prototype-source path across every board placement.

        Args:
            source_path: Absolute prim path rooted at the prototype source.

        Returns:
            Ordered composed target for every registered board placement.

        Raises:
            ValidationError: If the source path lies outside the prototype
                source.

        """
        root = self.prototype_source_path.rstrip("/")
        root_prefix = f"{root}/"
        if source_path == root:
            relative_path = ""
        elif source_path.startswith(root_prefix):
            relative_path = source_path.removeprefix(root)
        else:
            raise ValidationError(
                f"instance source path {source_path!r} is outside prototype "
                f"source {self.prototype_source_path!r}",
            )
        return tuple(
            BoardInstanceTarget(
                instance_index=index,
                instance_path=instance_path,
                source_path=source_path,
                composed_path=f"{instance_path.rstrip('/')}{relative_path}",
            )
            for index, instance_path in enumerate(self.instance_paths)
        )


_INSTANCED_STAGE_MUTATIONS_KEY = "capture.instanced_stage_mutations"


def mark_instanced_stage_mutation(
    context: IsaacExecutionContext,
    reason: str,
) -> None:
    """Record a shared-prototype mutation that capture must refresh.

    Args:
        context: Active Isaac execution context.
        reason: Stable identifier for the mutation source.

    Raises:
        ValidationError: If the mutation reason is empty.

    """
    if not reason:
        raise ValidationError("instanced stage mutation reason must not be empty")
    reasons = tuple(context.state.get(_INSTANCED_STAGE_MUTATIONS_KEY) or ())
    context.state[_INSTANCED_STAGE_MUTATIONS_KEY] = tuple(
        dict.fromkeys((*reasons, reason)),
    )


def _export_deinstanced_capture_stage(
    stage: object,
    cohort: BoardInstanceCohort,
    local_path: str,
) -> None:
    """Export a flattened snapshot with editable composed board descendants."""
    instance_roots = []
    try:
        for instance_path in cohort.instance_paths:
            prim = stage.GetPrimAtPath(instance_path)
            authored = prim.HasAuthoredMetadata("instanceable")
            value = prim.GetMetadata("instanceable")
            instance_roots.append((prim, authored, value))
            prim.SetInstanceable(False)
        flattened_stage = stage.Flatten()
        if not flattened_stage.Export(local_path):
            raise RuntimeError(
                "could not export temporary flattened panel stage for capture",
            )
    finally:
        for prim, authored, value in reversed(instance_roots):
            if authored:
                prim.SetInstanceable(bool(value))
            else:
                prim.ClearInstanceable()


async def refresh_instanced_capture_stage(
    context: IsaacExecutionContext,
    open_stage: Callable[[object, str], Awaitable[None]],
) -> object:
    """Reopen a flattened stage after prototype or instance-sub-geometry mutations.

    Live Hydra state can become stale when geometry or semantic data is
    authored below a shared board prototype (panel jobs) or below a component
    instance that was de-instanced at annotation time (single-board jobs).
    Reopening the composed result establishes a clean render index so Hydra
    builds its scene graph from concrete prims rather than instance proxies.
    Catalog and cohort identities remain valid because ROI expansion targets
    the unchanged composed instance paths.  Unmodified stages pass through
    without reopening.

    Args:
        context: Active Isaac execution context.
        open_stage: Asynchronous Kit stage-opening callback.

    Returns:
        Existing stage when no refresh is needed, otherwise the reopened stage.

    Example:

    .. code-block:: python

        stage = await refresh_instanced_capture_stage(  # doctest: +SKIP
            context,
            common.open_usd_stage,
        )

    """
    stage = context.require_stage()
    reasons = tuple(context.state.get(_INSTANCED_STAGE_MUTATIONS_KEY) or ())
    if not reasons:
        return stage

    temporary_directory = context._temporary_directory(
        prefix="cdf-annotated-capture-",
    )
    local_path = str(Path(temporary_directory) / "flattened-stage.usdc")

    if context.board_instance_cohort is not None:
        # Panel job: de-instance cohort roots before flattening so composed
        # descendants are concrete (not prototype references) in the flat USDC.
        _export_deinstanced_capture_stage(
            stage,
            context.board_instance_cohort,
            local_path,
        )
    else:
        # Single-board job: the session layer already carries de-instancing
        # overrides authored by _label_geometry_prims.  A plain flatten bakes
        # those overrides into the USDC so Hydra opens the stage with concrete
        # component sub-prims (e.g. pin prims) and discovers their semantic
        # labels rather than seeing stale instance-proxy placeholders.
        flattened_stage = stage.Flatten()
        if not flattened_stage.Export(local_path):
            raise RuntimeError(
                "could not export temporary flattened annotated stage for capture",
            )

    import omni.kit.app
    import omni.usd

    app = context.app or omni.kit.app.get_app()
    await open_stage(app, local_path)
    refreshed_stage = omni.usd.get_context().get_stage()
    if refreshed_stage is None:
        raise RuntimeError("Isaac reopened the annotated stage but returned no stage")
    if context.board_instance_cohort is not None:
        for instance_path in context.board_instance_cohort.instance_paths:
            prim = refreshed_stage.GetPrimAtPath(instance_path)
            if not prim or not prim.IsValid():
                raise RuntimeError(
                    f"refreshed panel stage is missing board instance {instance_path!r}",
                )
    context.app = app
    context.stage = refreshed_stage
    context.state.pop(_INSTANCED_STAGE_MUTATIONS_KEY, None)
    context.state["capture.instanced_stage_refresh_reasons"] = reasons
    context.state["capture.instanced_stage_path"] = local_path
    return refreshed_stage


@dataclass(frozen=True)
class ComponentProjectionRecord:
    """Catalog-derived component membership for semantic projection.

    Args:
        source_path: Component prim below the prototype source.
        member_paths: Component, package, contact, and associated pad paths.
        refdes: Component reference designator.
        class_name: Component class emitted into semantic labels.

    """

    source_path: str
    #: Component prim below the prototype source.
    member_paths: tuple[str, ...]
    #: Component, package, contact, and associated pad paths.
    refdes: str
    #: Component reference designator.
    class_name: str
    #: Component class emitted into semantic labels.

    def as_mapping(self, *, defect: str | None = None) -> dict[str, object]:
        """Return one post-processing component record.

        Args:
            defect: Optional structural or missing-component label.

        Returns:
            Mapping accepted by semantic projection post-processing.

        """
        semantics = {"class": self.class_name}
        if defect is not None:
            semantics["defect"] = defect
        return {
            "source_path": self.source_path,
            "member_paths": self.member_paths,
            "refdes": self.refdes,
            "semantics": semantics,
        }


@dataclass(frozen=True)
class InstanceSemanticProjection:
    """Typed component projection state for one board-instance cohort.

    Args:
        cohort: Validated board-placement cohort.
        components: Catalog-derived component membership records.

    """

    cohort: BoardInstanceCohort
    #: Validated board-placement cohort.
    components: tuple[ComponentProjectionRecord, ...]
    #: Catalog-derived component membership records.


def resolve_board_instance_cohort(
    stage: object,
    scene_root_path: str,
    instance_paths: tuple[str, ...],
) -> BoardInstanceCohort:
    """Validate a complete shared-prototype cohort and resolve its source.

    Args:
        stage: Open composed USD stage.
        scene_root_path: Root containing every registered board placement.
        instance_paths: Ordered board-instance paths supplied by the caller.

    Returns:
        Validated cohort with prototype, source, and representative paths.

    Raises:
        ValidationError: If paths do not form one complete, authorable cohort.

    """
    from pxr import Usd

    prototype_paths = []
    source_paths = []
    root_prefix = f"{scene_root_path.rstrip('/')}/"
    for path in instance_paths:
        if path != scene_root_path and not path.startswith(root_prefix):
            raise ValidationError(
                f"scene instance {path!r} is outside scene root {scene_root_path!r}",
            )
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValidationError(f"scene instance {path!r} does not exist")
        if not prim.IsInstance():
            raise ValidationError(f"scene instance {path!r} is not a USD instance")
        prototype = prim.GetPrototype()
        if not prototype or not prototype.IsValid():
            raise ValidationError(
                f"scene instance {path!r} has no usable USD prototype",
            )
        source_path = _internal_reference_source(prim)
        if source_path is None:
            raise ValidationError(
                f"scene instance {path!r} has no stage-local prototype source; "
                "external-reference cohorts are not yet authorable",
            )
        source = stage.GetPrimAtPath(source_path)
        if not source or not source.IsValid():
            raise ValidationError(
                f"scene instance {path!r} references missing source {source_path!r}",
            )
        prototype_paths.append(str(prototype.GetPath()))
        source_paths.append(source_path)

    if len(set(prototype_paths)) != 1:
        details = ", ".join(
            f"{path} -> {prototype}"
            for path, prototype in zip(instance_paths, prototype_paths, strict=True)
        )
        raise ValidationError(
            f"scene instances do not share one USD prototype: {details}",
        )
    if len(set(source_paths)) != 1:
        details = ", ".join(
            f"{path} -> {source}"
            for path, source in zip(instance_paths, source_paths, strict=True)
        )
        raise ValidationError(
            f"scene instances do not share one prototype source: {details}",
        )

    prototype_path = prototype_paths[0]
    discovered_paths = tuple(
        str(prim.GetPath())
        for prim in Usd.PrimRange(stage.GetPseudoRoot())
        if prim.IsInstance()
        and prim.GetPrototype()
        and str(prim.GetPrototype().GetPath()) == prototype_path
    )
    supplied = set(instance_paths)
    omitted = tuple(path for path in discovered_paths if path not in supplied)
    if omitted:
        raise ValidationError(
            "scene.instances must register every stage instance of the shared "
            f"prototype; omitted paths: {list(omitted)!r}",
        )
    return BoardInstanceCohort(
        instance_paths=instance_paths,
        representative_instance_path=instance_paths[0],
        prototype_path=prototype_path,
        prototype_source_path=source_paths[0],
    )


def build_component_semantic_projection(
    cohort: BoardInstanceCohort,
    catalog: BoardMetadataCatalog,
    component_paths: Iterable[str],
    classes: Mapping[str, tuple[str, ...]] | None,
) -> InstanceSemanticProjection:
    """Build typed component projection records from board metadata.

    Args:
        cohort: Validated board-placement cohort.
        catalog: Metadata catalog selected for the job.
        component_paths: Prototype-source components selected for annotation.
        classes: Optional class-to-refdes mapping from the public declaration.

    Returns:
        Component membership and classes bound to the validated cohort.

    Raises:
        ValidationError: If a selected component is absent from the catalog.

    """
    components_by_path = {
        component.prim_path: component for component in catalog.components
    }
    pad_paths_by_component: dict[str, list[str]] = {}
    for pad in catalog.pads:
        if pad.component_path is not None:
            pad_paths_by_component.setdefault(pad.component_path, []).append(
                pad.prim_path,
            )
    refdes_classes = {
        str(refdes): str(class_name)
        for class_name, refdes_values in (classes or {}).items()
        for refdes in refdes_values
    }
    records = []
    for component_path in component_paths:
        component = components_by_path.get(component_path)
        if component is None:
            raise ValidationError(
                f"selected component is absent from board metadata: {component_path}",
            )
        records.append(
            ComponentProjectionRecord(
                source_path=component.prim_path,
                member_paths=tuple(
                    dict.fromkeys(
                        (
                            component.prim_path,
                            *component.body_prim_paths,
                            *component.contact_prim_paths,
                            *pad_paths_by_component.get(component.prim_path, ()),
                        ),
                    ),
                ),
                refdes=component.refdes,
                class_name=refdes_classes.get(component.refdes, "capacitor"),
            ),
        )
    return InstanceSemanticProjection(cohort=cohort, components=tuple(records))


def prepare_instance_semantic_capture(
    projection: InstanceSemanticProjection | None,
    writer_config: dict[str, object],
    *,
    structural_records: Iterable[object] = (),
    missing_paths: Iterable[object] = (),
) -> dict[str, object] | None:
    """Prepare instance-safe semantic projection for one capture pass.

    Invoke this internal gate while building every capture-pass configuration.
    Projection is not a public capture argument and this function does not
    discover instances by examining the USD stage. Earlier runtime orders
    create ``projection`` only when the caller explicitly registered a
    validated cohort with ``scene.instances(...)`` and requested component
    annotations. Without that state, or when the current pass does not request
    semantic segmentation, return ``None`` and preserve native semantic output.

    When projection is active, add structural and missing-component defect
    labels and configure instance-ID capture as the projection source. Disable
    native semantic capture so it cannot author through an instance proxy onto
    the shared prototype. Automatically enabled instance-ID artifacts are
    marked for removal after projection.

    Args:
        projection: Typed component projection state, if instance-safe
            annotation is active.
        writer_config: Mutable BasicWriter configuration for the capture pass.
        structural_records: Applied structural-defect result records.
        missing_paths: Source component paths hidden by missing defects.

    Returns:
        Projection configuration for post-capture processing, or ``None`` when
        the pass must use its normal writer configuration.

    Raises:
        ValidationError: If the pass requests an unsupported semantic
            namespace.

    """
    if projection is None or not writer_config.get("semantic_segmentation"):
        return None
    semantic_types = tuple(writer_config.get("semantic_types") or ("class",))
    unsupported = set(semantic_types) - {"class", "defect"}
    if unsupported:
        raise ValidationError(
            "instance semantic projection supports only class and defect "
            f"semantic types, got {sorted(unsupported)!r}",
        )

    defects_by_path = {}
    for record in structural_records:
        if not isinstance(record, Mapping):
            continue
        source_path = str(record.get("path") or "")
        defect = record.get("defect")
        if source_path and defect is not None:
            defects_by_path[source_path] = str(defect)
    for component_path in missing_paths:
        defects_by_path[str(component_path)] = "missing"

    requested_instance_ids = bool(writer_config.get("instance_id_segmentation"))
    if not requested_instance_ids:
        writer_config["instance_id_segmentation"] = True
        writer_config["colorize_instance_id_segmentation"] = True
    colorized_semantics = bool(
        writer_config.get("colorize_semantic_segmentation"),
    )
    writer_config["semantic_segmentation"] = False
    writer_config["colorize_semantic_segmentation"] = False
    return {
        "board_root": projection.cohort.prototype_source_path,
        "instance_paths": projection.cohort.instance_paths,
        "components": tuple(
            component.as_mapping(defect=defects_by_path.get(component.source_path))
            for component in projection.components
        ),
        "semantic_types": semantic_types,
        "colorized": colorized_semantics,
        "remove_instance_artifacts": not requested_instance_ids,
    }
