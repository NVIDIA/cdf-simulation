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

"""Own mutable state shared by JobOrder handlers inside Isaac Sim."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import TYPE_CHECKING

from scripts import file_utils

from ...errors import ValidationError
from ..dataset_manifest import DatasetManifestObserver, EntityIdentity, ManifestView

#: Scheme addressing a companion asset relative to the loaded scene, whether the
#: scene is a plain USD or was extracted from a ZIP.
SCENE_URI_SCHEME = "scene://"

if TYPE_CHECKING:
    from pcb_aoi.board_metadata import BoardMetadataCatalog
    from pcb_aoi.execution.board_instances import (
        BoardInstanceCohort,
        InstanceSemanticProjection,
    )

    from ..job_order_types import JobOrderPlan


@dataclass(frozen=True)
class LoadedJsonDocument:
    """Decoded legacy JSON document available to dependent handlers.

    Args:
        name: Name referenced by JobOrder parameters.
        kind: Semantic document kind.
        path: Resolved source path.
        data: Decoded JSON object.

    """

    name: str
    kind: str
    path: str
    data: Mapping[str, object]


@dataclass
class IsaacExecutionContext:
    """Mutable state for one in-process Isaac JobOrder execution.

    Args:
        runtime: Runtime adapter implementing concrete Isaac/SDG operations.
        app: Optional already-resolved Kit application object.
        stage: Optional already-open USD stage.
        board_instance_cohort: Validated shared-board placement state.
        instance_semantic_projection: Catalog-derived instance annotation state.

    """

    runtime: object | None = None
    app: object | None = None
    stage: object | None = None
    scene: dict[str, object] = field(default_factory=dict)
    json_documents: dict[str, LoadedJsonDocument] = field(default_factory=dict)
    #: Non-component legacy documents that have no catalog representation.
    board_metadata: dict[str, BoardMetadataCatalog] = field(default_factory=dict)
    #: Catalog snapshots keyed by attached metadata-source name.
    component_types: tuple[str, ...] = ()
    component_pool: tuple[str, ...] = ()
    component_selection_resolved: bool = False
    #: Distinguishes an explicitly resolved empty pool from legacy direct
    #: handler calls that did not execute component selection.
    plan_metadata: dict[str, object] = field(default_factory=dict)
    manifest: DatasetManifestObserver = field(default_factory=DatasetManifestObserver)
    #: Run-scoped source of truth for dataset and postprocess metadata.
    outputs: dict[str, object] = field(default_factory=dict)
    state: dict[str, object] = field(default_factory=dict)
    board_instance_cohort: BoardInstanceCohort | None = None
    instance_semantic_projection: InstanceSemanticProjection | None = None
    _storage_stack: contextlib.ExitStack | None = field(
        default=None,
        init=False,
        repr=False,
    )
    #: Active executor-owned stack managing localized storage resources.
    _active_order_id: str | None = field(
        default=None,
        init=False,
        repr=False,
    )

    #: JobOrder currently invoking the runtime adapter.
    _scene_root: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    #: Directory holding the loaded scene: a ZIP's extraction root, or the
    #: directory containing a plain USD.
    _localized_trees: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    #: Map from input locations to their plan-scoped local paths.
    _output_trees: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    #: Map from output locations to their plan-scoped staging directories.

    def __post_init__(self) -> None:
        """Create the default runtime adapter without importing Isaac."""
        if self.runtime is None:
            from .job_order_runtime import IsaacSdgRuntime

            self.runtime = IsaacSdgRuntime()

    def component_parent_paths(self) -> tuple[str, ...] | None:
        """Return the effective component parent, or None when unrestricted.

        Returns:
            Explicit component paths, an explicitly resolved empty tuple, or
            None for legacy direct handler calls with no selection state.

        """
        if self.component_pool or self.component_selection_resolved:
            return tuple(str(path) for path in self.component_pool)
        return None

    def prepare(self, plan: JobOrderPlan) -> None:
        """Receive immutable plan metadata before the first handler runs.

        Args:
            plan: Validated plan selected for execution.

        """
        metadata = plan.model_dump(mode="json")["metadata"]
        if not isinstance(metadata, dict):
            raise ValidationError("JobOrderPlan metadata must serialize to an object")
        self.plan_metadata = metadata
        self.manifest.reset(metadata)
        # Static provenance -- code, versions, scene, config, runtime -- is
        # knowable before the first handler runs, so record it now: a run that
        # dies half-way still leaves its trigger manifests attributable. The
        # executor replaces it with the full record (orders, artifacts,
        # success) when the plan finishes. Never allowed to fail the run.
        try:
            from ...provenance import static_record

            self.manifest.record_provenance(static_record(plan))
        except Exception as error:  # noqa: BLE001 -- provenance must not fail a render
            import warnings

            warnings.warn(
                f"static provenance was not recorded: {type(error).__name__}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    @contextlib.contextmanager
    def storage_scope(self) -> Iterator[IsaacExecutionContext]:
        """Own localized storage resources for one complete job plan.

        Yields:
            This execution context while its storage resources are active.

        Raises:
            ValidationError: If the context is already executing a plan.

        Example:

        .. code-block:: python

            with context.storage_scope():
                print(context.localize_tree("board.usd"))

        """
        if self._storage_stack is not None:
            raise ValidationError("execution context storage scope is already active")
        self._localized_trees.clear()
        self._output_trees.clear()
        try:
            with contextlib.ExitStack() as stack:
                self._storage_stack = stack
                yield self
        finally:
            self._storage_stack = None
            self._localized_trees.clear()
            self._output_trees.clear()

    def localize_tree(self, uri: str | os.PathLike[str]) -> str:
        """Return a plan-scoped local path for an input tree.

        Args:
            uri: Local path or supported remote input URI.

        Returns:
            Local path valid until the current storage scope exits.

        Raises:
            ValidationError: If remote storage is requested outside execution.

        Example:

        .. code-block:: python

            with context.storage_scope():
                print(context.localize_tree("board.usd"))

        """
        uri_text = os.fspath(uri)
        if uri_text in self._localized_trees:
            return self._localized_trees[uri_text]
        if self._storage_stack is None:
            if file_utils.is_remote(uri_text):
                raise ValidationError(
                    "remote input requires an executor-managed storage scope",
                )
            return uri_text
        localized = self._storage_stack.enter_context(
            file_utils.localize_tree(uri_text),
        )
        self._localized_trees[uri_text] = localized
        return localized

    def _temporary_directory(self, *, prefix: str) -> str:
        """Return an executor-owned temporary directory for this plan."""
        if self._storage_stack is None:
            raise ValidationError(
                "temporary execution storage requires an active storage scope",
            )
        return self._storage_stack.enter_context(
            tempfile.TemporaryDirectory(prefix=prefix),
        )

    def localize_scene(
        self,
        uri: str | os.PathLike[str],
        archive_member: str | None = None,
    ) -> str:
        """Localize a scene path, extracting and selecting a USD from ZIP input.

        ZIP archives may contain companion files at any depth, but candidate
        USD scenes must be regular files at the archive root. A sole candidate
        is selected automatically; multiple candidates require an explicit
        ``archive_member``.

        Args:
            uri: Local path or supported remote scene URI.
            archive_member: Root-level USD filename to select from ZIP input.

        Returns:
            Local USD path valid until the current storage scope exits.

        Raises:
            ValidationError: If ZIP input is used outside a storage scope, no
                root-level USD exists, or scene selection is ambiguous or
                invalid.

        Example:

        .. code-block:: python

            with context.storage_scope():
                print(context.localize_scene("boards.zip", "panel.usda"))

        """
        localized = self.localize_tree(uri)
        if Path(localized).suffix.lower() != ".zip":
            self._scene_root = str(Path(localized).parent)
            if archive_member is not None:
                raise ValidationError(
                    "archive_member is only valid when usd_path identifies a ZIP file",
                )
            return localized
        if self._storage_stack is None:
            raise ValidationError(
                "ZIP input requires an executor-managed storage scope",
            )

        extracted = self._storage_stack.enter_context(
            file_utils.extract_zip(localized),
        )
        root = Path(extracted)
        candidates = sorted(
            path.name
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}
        )
        if not candidates:
            raise ValidationError(
                "ZIP archive contains no root-level USD files (.usd, .usda, or .usdc)",
            )
        if archive_member is None:
            if len(candidates) != 1:
                raise ValidationError(
                    "ZIP archive contains multiple root-level USD files; specify "
                    "archive_member from: {}".format(", ".join(candidates)),
                )
            archive_member = candidates[0]
        if PurePath(archive_member).name != archive_member or "/" in archive_member:
            raise ValidationError("archive_member must be a root-level filename")
        if archive_member not in candidates:
            raise ValidationError(
                "archive_member {!r} is not a root-level USD file; available: {}".format(
                    archive_member, ", ".join(candidates)
                ),
            )
        self._scene_root = str(root)
        return str(root / archive_member)

    def resolve_scene_asset(self, path: str | os.PathLike[str]) -> str:
        """Resolve a companion-asset reference against the loaded scene.

        A scene may ship its textures beside the USD, and when that scene is a
        ZIP the directory they land in is chosen at load time, so a job cannot
        name it in advance. Two forms address them:

        * ``scene://textures/albedo.png`` is explicit -- always relative to the
          scene, and an error if it is not there.
        * ``textures/albedo.png`` is tried as given first, so a path that exists
          relative to the working directory keeps winning, and only falls back
          to the scene when it does not.

        Absolute and rooted paths are returned untouched.

        Args:
            path: Asset path, optionally scene-relative.

        Returns:
            An absolute path to the asset.

        Raises:
            ValidationError: If a scene-relative reference escapes the scene
                directory, or names a file that is not there.

        Example:

        .. code-block:: python

            context.resolve_scene_asset("scene://textures/albedo.png")

        """
        text = str(path)
        if text.startswith(SCENE_URI_SCHEME):
            return self._resolve_within_scene(
                text[len(SCENE_URI_SCHEME) :],
                reference=text,
                explicit=True,
            )
        candidate = Path(text).expanduser()
        # a rooted path is not relative even where it carries no drive letter,
        # so "/assets/albedo.png" is never looked for inside the scene
        if candidate.is_absolute() or candidate.root:
            return str(candidate)
        if candidate.exists():
            return str(candidate.resolve())
        return self._resolve_within_scene(text, reference=text, explicit=False)

    def _resolve_within_scene(
        self,
        relative: str,
        *,
        reference: str,
        explicit: bool,
    ) -> str:
        """Resolve one relative asset path inside the scene directory.

        Args:
            relative: Path below the scene directory.
            reference: Original reference, used in error messages.
            explicit: Whether the caller asked for the scene explicitly.

        Returns:
            Absolute path to the asset.

        Raises:
            ValidationError: If no scene is loaded, the path escapes the scene
                directory, or the asset is not present.

        """
        if self._scene_root is None:
            raise ValidationError(
                f"cannot resolve {reference!r} against the scene: "
                "scene.configure(usd_path=...) has not run",
            )
        root = Path(self._scene_root).resolve()
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValidationError(
                f"scene-relative reference {reference!r} escapes the scene "
                f"directory {str(root)!r}",
            )
        if not resolved.is_file():
            available = sorted(
                str(item.relative_to(root))
                for item in root.rglob("*")
                if item.is_file()
            )
            listing = ", ".join(available[:20]) or "(none)"
            if explicit:
                raise ValidationError(
                    f"scene has no asset {relative!r}; available: {listing}",
                )
            raise ValidationError(
                f"asset {reference!r} is not on disk and the scene does not "
                f"carry it either; the scene holds: {listing}",
            )
        return str(resolved)

    def output_tree(self, uri: str | os.PathLike[str]) -> str:
        """Return a plan-scoped local directory for an output tree.

        Args:
            uri: Local output path or supported remote output URI.

        Returns:
            Local directory shared by capture and postprocessing.

        Raises:
            ValidationError: If remote storage is requested outside execution.

        Example:

        .. code-block:: python

            with context.storage_scope():
                print(context.output_tree("output"))

        """
        uri_text = os.fspath(uri)
        if uri_text in self._output_trees:
            return self._output_trees[uri_text]
        if self._storage_stack is None:
            if file_utils.is_remote(uri_text):
                raise ValidationError(
                    "remote output requires an executor-managed storage scope",
                )
            return uri_text
        localized = self._storage_stack.enter_context(
            file_utils.output_tree(uri_text),
        )
        self._output_trees[uri_text] = localized
        return localized

    def local_output_tree(self, uri: str | os.PathLike[str]) -> str:
        """Return the staged directory associated with an output location.

        Args:
            uri: Original local or remote output location.

        Returns:
            Staged directory when registered, otherwise the original path.

        Example:

        .. code-block:: python

            print(context.local_output_tree("output"))

        """
        uri_text = os.fspath(uri)
        return self._output_trees.get(uri_text, uri_text)

    def published_output_path(
        self,
        output_uri: str,
        local_path: str | os.PathLike[str],
    ) -> str:
        """Translate a staged artifact path to its published location.

        Args:
            output_uri: Original local or remote output root.
            local_path: Artifact path beneath the staged output root.

        Returns:
            Local artifact path or corresponding remote URI.

        Raises:
            ValidationError: If the artifact is outside the staged output tree.

        Example:

        .. code-block:: python

            print(
                context.published_output_path(
                    "output",
                    "output/frame.png",
                )
            )

        """
        local_root = self.local_output_tree(output_uri)
        relative = os.path.relpath(os.fspath(local_path), local_root)
        if relative == ".":
            return output_uri
        if relative == ".." or relative.startswith(".." + os.sep):
            raise ValidationError(
                f"capture artifact is outside the staged output tree: {local_path}",
            )
        return self.join_output_location(
            output_uri,
            relative.replace(os.sep, "/"),
        )

    @staticmethod
    def join_output_location(root: str, child: str) -> str:
        """Join a child path onto a local or remote output location.

        Args:
            root: Base output directory or URI.
            child: Relative output subdirectory.

        Returns:
            Joined local path or remote URI.

        Example:

        .. code-block:: python

            print(context.join_output_location("output", "cropped"))

        """
        if file_utils.is_remote(root):
            return "{}/{}".format(root.rstrip("/"), child.strip("/"))
        return os.path.join(root, child)

    def require_stage(self) -> object:
        """Return the open USD stage required by stage-dependent handlers.

        Returns:
            Open USD stage.

        Raises:
            ValidationError: If ``scene.load`` has not established a stage.

        """
        if self.stage is None:
            raise ValidationError(
                "Isaac execution context has no stage; run scene.load first",
            )
        return self.stage

    def entity_identity(self, path: str) -> EntityIdentity:
        """Resolve a USD path to available component identity metadata.

        Args:
            path: Component or descendant USD prim path.

        Returns:
            Entity identity enriched from loaded board metadata when possible.

        Example:

        .. code-block:: python

            print(context.entity_identity("/Board/Components/R1")["refdes"])
            R1

        """
        for catalog in self.board_metadata.values():
            for component in catalog.components:
                candidate_paths = (component.prim_path,)
                if self.board_instance_cohort is not None:
                    try:
                        candidate_paths = tuple(
                            target.composed_path
                            for target in self.board_instance_cohort.expand_source_path(
                                component.prim_path
                            )
                        )
                    except ValidationError:
                        candidate_paths = (component.prim_path,)
                for candidate in candidate_paths:
                    if path == candidate or path.startswith(f"{candidate}/"):
                        identity: EntityIdentity = {
                            "prim_path": candidate,
                            "refdes": component.refdes,
                            "board_root": component.board_root,
                        }
                        if component.component_type:
                            identity["component_type"] = component.component_type
                        return identity
        for component_path in sorted(
            (str(item) for item in self.component_pool),
            key=len,
            reverse=True,
        ):
            if path == component_path or path.startswith(f"{component_path}/"):
                return {
                    "prim_path": component_path,
                    "refdes": component_path.rstrip("/").rsplit("/", 1)[-1],
                }
        parts = path.rstrip("/").split("/")
        if "Components" in parts:
            index = len(parts) - 1 - parts[::-1].index("Components")
            if index + 1 < len(parts):
                return {
                    "prim_path": "/".join(parts[: index + 2]),
                    "refdes": parts[index + 1],
                }
        return {"prim_path": path}

    @property
    def output_view(self) -> Mapping[str, object]:
        """Return a read-only view of accumulated handler outputs.

        Returns:
            Mapping from JobOrder IDs to handler outputs.

        """
        return MappingProxyType(self.outputs)

    @property
    def active_order_id(self) -> str | None:
        """Return the JobOrder currently invoking the runtime adapter.

        Returns:
            Active JobOrder identifier, or None outside a handler invocation.

        Example:

        .. code-block:: python

            print(context.active_order_id)
            None

        """
        return self._active_order_id

    @property
    def manifest_view(self) -> ManifestView:
        """Return a read-only snapshot of dataset observations.

        Returns:
            Manifest query facade detached from mutable observer state.

        Example:

        .. code-block:: python

            print(context.manifest_view.dataset_id)  # doctest: +SKIP

        """
        return self.manifest.view()
