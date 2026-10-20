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

"""Collect and serialize one CDF dataset manifest."""

from __future__ import annotations

import copy
import json
import numbers
import uuid
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import TypedDict

from ..errors import ValidationError

SCHEMA_NAME = "cdf.dataset_manifest"
SCHEMA_VERSION = "1.0.0"


class EntityIdentity(TypedDict, total=False):
    """Normalized identity for a rendered scene entity."""

    prim_path: str
    refdes: str
    board_root: str
    component_type: str


class ManifestDefect(TypedDict, total=False):
    """Normalized defect associated with one manifest entity."""

    defect_id: str
    type: str
    defect: str
    entity_id: str
    usd_path: str
    path: str
    refdes: str | None
    severity: dict[str, object]
    visible_in_roles: list[str]


class ManifestArtifact(TypedDict, total=False):
    """File or collection produced by capture or postprocessing."""

    artifact_id: str
    frame_id: str
    kind: str
    path: str
    role: str
    trigger_idx: int


class ManifestCameraCalibration(TypedDict, total=False):
    """Calibration of the capture camera at the moment one frame rendered.

    Enough to turn a pixel of that frame back into a board coordinate: an
    intrinsic matrix in camera-frame millimetres and the extrinsic pair. A
    frame whose calibration could not be computed carries ``unavailable``
    instead of the geometry, so the reason survives into the dataset rather
    than the field silently going missing.
    """

    projection: str | None
    camera_path: str
    resolution_px: list[int]
    horizontal_aperture_mm: float | None
    vertical_aperture_mm: float | None
    mm_per_px: list[float] | None
    intrinsic_matrix: list[list[float]] | None
    intrinsic_convention: str
    world_to_camera: list[list[float]]
    camera_to_world: list[list[float]]
    matrix_convention: str
    position: list[float]
    view_direction: list[float]
    up_direction: list[float]
    stage_meters_per_unit: float
    unavailable: str


class ManifestROISelection(TypedDict, total=False):
    """Focused component or physical-pin selection recorded for one frame."""

    refdes: str
    component_path: str
    component_type: str | None
    package_name: str | None
    package_type: str | None
    footprint_id: str | None
    side: str | None
    process_class: str | None
    process_classes: list[str]
    pin_number: str | None
    pin_path: str
    pad_paths: tuple[str, ...]
    solder_joint_path: str | None
    selector_index: int
    selector_type: str
    horizontal_aperture: float
    vertical_aperture: float
    instance_index: int
    instance_path: str
    source_component_path: str
    source_pin_path: str


class ManifestCamera(TypedDict, total=False):
    """Camera state recorded with one frame."""

    position: list[float]
    roi_selection: ManifestROISelection
    calibration: ManifestCameraCalibration


class ManifestFrame(TypedDict, total=False):
    """Canonical rendered frame and its dataset relationships."""

    frame_id: str
    frame: int
    trigger: int
    role: str
    pair_id: str
    position: list[float]
    camera: ManifestCamera
    grid: dict[str, int]
    roi_selection: ManifestROISelection
    defect_ids: list[str]
    defect_association: str
    artifacts: dict[str, list[str]]


class ManifestView:
    """Read-only query facade over a run-scoped manifest snapshot.

    Args:
        snapshot: Deep-copied observer state.

    Example:

    .. code-block:: python

        view = observer.view()
        print(view.capture_pass("capture.render")["name"])
        default

    """

    def __init__(self, snapshot: Mapping[str, object]) -> None:
        """Initialize the view from an isolated observer snapshot."""
        self._snapshot = copy.deepcopy(dict(snapshot))

    @property
    def dataset_id(self) -> str:
        """Return the unique execution dataset identifier.

        Returns:
            Unique identifier for the observed execution.

        Example:

        .. code-block:: python

            print(view.dataset_id)  # doctest: +SKIP

        """
        return str(self._snapshot["dataset_id"])

    def capture_pass(self, pass_id: str) -> dict[str, object]:
        """Return one capture pass as an isolated copy.

        Args:
            pass_id: Capture render JobOrder identifier.

        Returns:
            Capture-pass manifest.

        Raises:
            KeyError: If the pass has not been observed.

        Example:

        .. code-block:: python

            print(view.capture_pass("capture.render")["output_uri"])
            output

        """
        passes = self._snapshot["passes"]
        if not isinstance(passes, dict):
            raise KeyError(pass_id)
        return copy.deepcopy(passes[pass_id])

    def trigger_metadata(self, pass_id: str) -> list[dict[str, object]]:
        """Return schema-rich per-trigger documents for a capture pass.

        Args:
            pass_id: Capture render JobOrder identifier.

        Returns:
            Ordered trigger metadata documents.

        Example:

        .. code-block:: python

            print(len(view.trigger_metadata("capture.render")))
            1

        """
        capture_pass = self.capture_pass(pass_id)
        triggers = capture_pass.get("triggers")
        if not isinstance(triggers, list):
            return []
        return [copy.deepcopy(item["document"]) for item in triggers]

    def defects(self) -> list[ManifestDefect]:
        """Return every normalized defect in the dataset.

        Returns:
            Defect records with entity identity and sampled severity.

        Example:

        .. code-block:: python

            print(view.defects())  # doctest: +SKIP

        """
        defects = self._snapshot.get("defects")
        return copy.deepcopy(defects) if isinstance(defects, list) else []

    def frames(
        self,
        pass_id: str,
        trigger_index: int | None = None,
    ) -> list[ManifestFrame]:
        """Return frames for a pass, optionally restricted to one trigger.

        Args:
            pass_id: Capture render JobOrder identifier.
            trigger_index: Trigger index to select, or None for every trigger.

        Returns:
            Canonical frame records in capture order.

        Example:

        .. code-block:: python

            print(len(view.frames("capture.render")))
            1

        """
        documents = self.trigger_metadata(pass_id)
        if trigger_index is not None:
            documents = [
                document
                for document in documents
                if int(document["trigger_idx"]) == trigger_index
            ]
        return [
            copy.deepcopy(frame)
            for document in documents
            for frame in document.get("frames", [])
        ]

    def artifacts(
        self,
        pass_id: str,
        *,
        kind: str | None = None,
    ) -> list[ManifestArtifact]:
        """Return captured artifacts for a pass, optionally filtered by kind.

        Args:
            pass_id: Capture render JobOrder identifier.
            kind: Artifact kind to select, or None for every kind.

        Returns:
            Captured artifact records in trigger order.

        Example:

        .. code-block:: python

            print(view.artifacts("capture.render", kind="rgb"))  # doctest: +SKIP

        """
        records = [
            copy.deepcopy(artifact)
            for document in self.trigger_metadata(pass_id)
            for artifact in document.get("artifacts", [])
        ]
        if kind is None:
            return records
        return [artifact for artifact in records if artifact.get("kind") == kind]


class ManifestSerializer:
    """Schema-aware writer for trigger manifests and dataset indexes.

    Example:

    .. code-block:: python

        serializer = ManifestSerializer()
        serializer.write_json(Path("metadata.json"), {"schema_version": "1.0.0"})

    """

    def write_json(self, path: Path, document: Mapping[str, object]) -> None:
        """Write one JSON manifest atomically enough for local capture output.

        Args:
            path: Destination JSON path.
            document: JSON-compatible manifest document.

        Example:

        .. code-block:: python

            serializer.write_json(Path("metadata.json"), {"frames": []})

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def read_json(self, path: Path) -> dict[str, object]:
        """Load and validate a compatible persisted manifest document.

        Args:
            path: Per-trigger manifest or dataset-index path.

        Returns:
            Parsed manifest document.

        Raises:
            ValidationError: If the document has an unsupported schema or
                schema major version.

        Example:

        .. code-block:: python

            document = serializer.read_json(Path("dataset_index.json"))

        """
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValidationError(
                f"cannot read dataset manifest {path}: {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise ValidationError(f"dataset manifest {path} must contain an object")

        schema_name = document.get("schema_name")
        if schema_name not in {SCHEMA_NAME, f"{SCHEMA_NAME}.index"}:
            raise ValidationError(
                f"dataset manifest {path} has unsupported schema {schema_name!r}"
            )
        version = str(document.get("schema_version") or "")
        try:
            major_version = int(version.split(".", 1)[0])
        except ValueError as exc:
            raise ValidationError(
                f"dataset manifest {path} has invalid schema version {version!r}"
            ) from exc
        supported_major = int(SCHEMA_VERSION.split(".", 1)[0])
        if major_version != supported_major:
            raise ValidationError(
                f"dataset manifest {path} uses unsupported schema version {version!r}"
            )
        return document

    def enrich_semantic_labels(
        self,
        paths: Iterable[Path],
        defects: Iterable[Mapping[str, object]],
    ) -> None:
        """Associate defect-class label entries with defects and refdes values.

        Args:
            paths: Semantic label JSON files to update.
            defects: Normalized defects available to the rendered pass.

        Example:

        .. code-block:: python

            serializer.enrich_semantic_labels([], [])

        """
        by_type: dict[str, list[dict[str, object]]] = {}
        for defect in defects:
            defect_type = str(defect.get("type") or "")
            if not defect_type:
                continue
            association = {
                key: defect[key]
                for key in ("defect_id", "entity_id", "refdes", "usd_path")
                if defect.get(key) is not None
            }
            by_type.setdefault(defect_type, []).append(association)
        if not by_type:
            return

        for path in paths:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(document, dict):
                continue
            changed = False
            for entry in document.values():
                if not isinstance(entry, dict):
                    continue
                associations = by_type.get(str(entry.get("class") or ""))
                if associations:
                    entry["defects"] = associations
                    changed = True
            if changed:
                self.write_json(path, document)


class DatasetManifestObserver:
    """Run-scoped collector for capture, defect, and postprocess metadata.

    Producers submit facts to this object while consumers query immutable
    ``ManifestView`` snapshots. Only ``ManifestSerializer`` writes manifest
    JSON, preventing postprocessors from replacing richer capture metadata.

    Args:
        serializer: Optional manifest serializer implementation.

    Example:

    .. code-block:: python

        observer = DatasetManifestObserver()
        observer.reset({"random_seed": 7})
        print(observer.view().dataset_id)  # doctest: +SKIP

    """

    def __init__(self, serializer: ManifestSerializer | None = None) -> None:
        """Initialize an empty observation scope."""
        self.serializer = serializer or ManifestSerializer()
        self.reset({})

    @classmethod
    def from_index(
        cls,
        index_path: str | Path,
        *,
        pass_roots: Mapping[str, str | Path] | None = None,
        serializer: ManifestSerializer | None = None,
    ) -> DatasetManifestObserver:
        """Hydrate an observer from a persisted dataset index.

        Args:
            index_path: Dataset index written by the observer.
            pass_roots: Optional local root override keyed by capture pass ID.
                Supply this for remote outputs after staging is unavailable.
            serializer: Optional manifest serializer implementation.

        Returns:
            Observer populated with persisted dataset and trigger records.

        Raises:
            ValidationError: If roots cannot be resolved or documents disagree.

        Example:

        .. code-block:: python

            observer = DatasetManifestObserver.from_index(
                "output/dataset_index.json"
            )

        """
        index_file = Path(index_path)
        manifest_serializer = serializer or ManifestSerializer()
        index = manifest_serializer.read_json(index_file)
        observer = cls(manifest_serializer)
        observer._dataset_id = str(index.get("dataset_id") or "")
        if not observer._dataset_id:
            raise ValidationError(f"dataset index {index_file} has no dataset_id")
        plan_metadata = index.get("plan_metadata")
        observer._plan_metadata = (
            copy.deepcopy(plan_metadata) if isinstance(plan_metadata, dict) else {}
        )
        # index_document() persists the provenance record, so hydration has to
        # take it back: without this a caller that hydrates and then rewrites
        # the index replaces a real record with null.
        provenance = index.get("provenance")
        observer._provenance = (
            copy.deepcopy(provenance) if isinstance(provenance, dict) else None
        )
        entities = index.get("entities")
        defects = index.get("defects")
        if not isinstance(entities, list) or not isinstance(defects, list):
            raise ValidationError(
                f"dataset index {index_file} has invalid entity or defect indexes"
            )
        observer._entities = {
            str(entity["entity_id"]): copy.deepcopy(entity)
            for entity in entities
            if isinstance(entity, dict) and entity.get("entity_id")
        }
        observer._defects = {
            str(defect["defect_id"]): copy.deepcopy(defect)
            for defect in defects
            if isinstance(defect, dict) and defect.get("defect_id")
        }
        observer._passes = {}

        root_overrides = dict(pass_roots or {})
        passes = index.get("passes")
        if not isinstance(passes, list):
            raise ValidationError(f"dataset index {index_file} has no pass list")
        if len(passes) == 1 and isinstance(passes[0], dict):
            pass_id = str(passes[0].get("pass_id") or "")
            if pass_id:
                root_overrides.setdefault(pass_id, index_file.parent)
        for persisted_pass in passes:
            observer._hydrate_pass(
                index_file,
                persisted_pass,
                root_overrides,
            )
        return observer

    def _hydrate_pass(
        self,
        index_file: Path,
        persisted_pass: object,
        root_overrides: Mapping[str, str | Path],
    ) -> None:
        """Hydrate one capture pass referenced by a persisted index."""
        if not isinstance(persisted_pass, dict):
            raise ValidationError(
                f"dataset index {index_file} contains an invalid pass"
            )
        pass_id = str(persisted_pass.get("pass_id") or "")
        output_uri = str(persisted_pass.get("output_uri") or "")
        if not pass_id or not output_uri:
            raise ValidationError(
                f"dataset index {index_file} contains an incomplete pass"
            )
        root = root_overrides.get(pass_id)
        if root is None:
            persisted_root = persisted_pass.get("local_root")
            if persisted_root:
                root = Path(str(persisted_root))
        if root is None:
            if "://" in output_uri:
                raise ValidationError(
                    f"{pass_id} uses remote output {output_uri!r}; "
                    "provide pass_roots for hydration"
                )
            output_path = Path(output_uri)
            root = (
                index_file.parent
                if index_file.parent.name == output_path.name
                else output_path
            )
        trigger_records = self._hydrate_triggers(
            pass_id,
            Path(root),
            persisted_pass.get("triggers"),
        )
        derived = persisted_pass.get("derived_artifacts")
        self._passes[pass_id] = {
            "pass_id": pass_id,
            "name": str(persisted_pass.get("name") or pass_id),
            "output_uri": output_uri,
            "local_root": str(root),
            "triggers": trigger_records,
            "derived_artifacts": copy.deepcopy(
                derived if isinstance(derived, list) else []
            ),
        }

    def _hydrate_triggers(
        self,
        pass_id: str,
        root: Path,
        persisted_triggers: object,
    ) -> list[dict[str, object]]:
        """Load the trigger documents for one hydrated capture pass."""
        if not isinstance(persisted_triggers, list):
            raise ValidationError(f"{pass_id} has no trigger list")
        records = []
        for trigger in persisted_triggers:
            if not isinstance(trigger, dict):
                raise ValidationError(f"{pass_id} contains an invalid trigger")
            metadata_path = str(trigger.get("metadata_path") or "")
            trigger_index = trigger.get("trigger_idx")
            if not metadata_path or not isinstance(trigger_index, int):
                raise ValidationError(f"{pass_id} contains an incomplete trigger")
            document = self.serializer.read_json(root / metadata_path)
            if document.get("dataset_id") != self._dataset_id:
                raise ValidationError(
                    f"{pass_id} trigger {trigger_index} belongs to a different dataset"
                )
            self._validate_document(document)
            records.append(
                {
                    "trigger_idx": trigger_index,
                    "metadata_path": metadata_path,
                    "document": document,
                }
            )
        return records

    def reset(self, plan_metadata: Mapping[str, object]) -> None:
        """Start a new dataset observation scope.

        Args:
            plan_metadata: Serialized metadata from the active JobOrder plan.

        Example:

        .. code-block:: python

            observer.reset({"random_seed": 7})

        """
        self._dataset_id = str(uuid.uuid4())
        self._plan_metadata = copy.deepcopy(dict(plan_metadata))
        self._passes: dict[str, dict[str, object]] = {}
        self._entities: dict[str, dict[str, object]] = {}
        self._defects: dict[str, dict[str, object]] = {}
        self._provenance: dict[str, object] | None = None

    def record_provenance(self, record: Mapping[str, object]) -> None:
        """Attach the run's provenance record and republish the index.

        Called twice per run: once from ``IsaacExecutionContext.prepare()`` with
        the static facts (code, versions, scene, config, runtime), so a run
        that dies mid-way still has them, and once from the executor when it
        finishes, with orders, artifacts and ``success`` added. The later call
        replaces the earlier one. Every registered pass root gets its
        ``dataset_index.json`` rewritten so the record sits with the data.

        Trigger manifests already on disk are deliberately left alone: the
        They are not overwritten, and a trigger's compact summary is a snapshot
        of when it was written. The identity fields of that summary (code,
        versions, config,
        board) do not change between the two calls, so a trigger and the index
        never disagree about what produced the data; ``phase`` and
        ``unverified_count`` may differ, and ``dataset_index.json`` is the
        authoritative final record.

        Args:
            record: Provenance record from :func:`cdf.provenance.build`.

        Example:

        .. code-block:: python

            observer.record_provenance({"fingerprint_version": "1.1"})
            print(observer.index_document()["provenance"]["fingerprint_version"])
            1.1

        """
        self._provenance = copy.deepcopy(dict(record))
        if self._passes:
            self.serialize_index()

    def provenance_summary(self) -> dict[str, object] | None:
        """Return the compact provenance carried by every trigger manifest.

        Returns:
            The summary, or ``None`` when no provenance has been recorded.

        Example:

        .. code-block:: python

            print(DatasetManifestObserver().provenance_summary())
            None

        """
        if self._provenance is None:
            return None
        from ..provenance.fingerprint import summary

        return summary(self._provenance)

    def begin_capture_pass(
        self,
        pass_id: str,
        *,
        output_uri: str,
        local_root: str | Path,
    ) -> None:
        """Register the output roots and stable name for one capture pass.

        Args:
            pass_id: Capture render JobOrder identifier.
            output_uri: Published output directory or URI.
            local_root: Writable output directory for the active execution.

        Example:

        .. code-block:: python

            observer.begin_capture_pass(
                "capture.render", output_uri="output", local_root="output"
            )

        """
        name = (
            "default"
            if pass_id == "capture.render"
            else pass_id.removeprefix("capture.").removesuffix(".render")
        )
        self._passes[pass_id] = {
            "pass_id": pass_id,
            "name": name,
            "output_uri": output_uri,
            "local_root": str(local_root),
            "triggers": [],
            "derived_artifacts": [],
        }

    def observe_defects(
        self,
        records: Iterable[Mapping[str, object]],
        identity_for_path: Callable[[str], EntityIdentity],
    ) -> list[str]:
        """Normalize defects and retain their sampled severity.

        Args:
            records: Applied defect records from runtime producers.
            identity_for_path: Resolver for component metadata by USD path.

        Returns:
            Stable defect identifiers in input order.

        Example:

        .. code-block:: python

            ids = observer.observe_defects(
                [{"path": "/Board/Components/R1", "defect": "missing"}],
                lambda path: {"prim_path": path, "refdes": "R1"},
            )
            print(len(ids))
            1

        """
        identifiers = []
        for record in records:
            path = str(record.get("path") or "")
            defect_type = str(record.get("defect") or "")
            if not path or not defect_type:
                continue
            identity = identity_for_path(path)
            entity_path = str(identity.get("prim_path") or path)
            entity_id = self._stable_id("entity", entity_path)
            entity = {
                "entity_id": entity_id,
                "usd_path": entity_path,
                **{
                    key: value
                    for key, value in identity.items()
                    if key != "prim_path" and value is not None
                },
            }
            self._entities[entity_id] = entity
            defect_id = self._stable_id(
                "defect",
                path,
                defect_type,
            )
            severity = self._severity(record, defect_type)
            defect = {
                "defect_id": defect_id,
                "type": defect_type,
                "defect": defect_type,
                "entity_id": entity_id,
                "usd_path": path,
                "path": path,
                "refdes": identity.get("refdes"),
                "severity": severity,
            }
            if defect_type == "missing":
                defect["visible_in_roles"] = ["defective"]
            self._defects[defect_id] = defect
            identifiers.append(defect_id)
        return identifiers

    def record_trigger(
        self,
        pass_id: str,
        trigger_index: int,
        legacy_metadata: Mapping[str, object],
        frames: Iterable[Mapping[str, object]],
        artifacts: Iterable[Mapping[str, object]],
    ) -> dict[str, object]:
        """Record and serialize one completed capture trigger.

        Args:
            pass_id: Capture render JobOrder identifier.
            trigger_index: Zero-based trigger index.
            legacy_metadata: Compatibility fields produced by capture.
            frames: Canonical frame records for this trigger.
            artifacts: Artifacts emitted for these frames.

        Returns:
            Schema-rich trigger manifest written to ``metadata.json``.

        Raises:
            KeyError: If the capture pass was not registered.

        Example:

        .. code-block:: python

            document = observer.record_trigger(
                "capture.render", 0, {"trigger_idx": 0}, [], []
            )
            print(document["schema_version"])
            1.0.0

        """
        capture_pass = self._passes[pass_id]
        frame_records = copy.deepcopy(list(frames))
        self._normalize_frame_cameras(frame_records)
        artifact_records = copy.deepcopy(list(artifacts))
        defect_records = list(self._defects.values())
        document = {
            **copy.deepcopy(dict(legacy_metadata)),
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "dataset_id": self._dataset_id,
            "pass": {
                "pass_id": pass_id,
                "name": capture_pass["name"],
                "output_uri": capture_pass["output_uri"],
            },
            "trigger": {
                "trigger_id": self._stable_id(
                    "trigger", self._dataset_id, pass_id, str(trigger_index)
                ),
                "index": trigger_index,
            },
            "entities": copy.deepcopy(list(self._entities.values())),
            "defects": copy.deepcopy(defect_records),
            "frames": frame_records,
            "artifacts": artifact_records,
        }
        provenance_summary = self.provenance_summary()
        if provenance_summary is not None:
            document["provenance"] = provenance_summary
        self._validate_document(document)
        trigger_record = {
            "trigger_idx": trigger_index,
            "metadata_path": f"trigger_{trigger_index:04d}/metadata.json",
            "document": document,
        }
        triggers = capture_pass["triggers"]
        if not isinstance(triggers, list):
            raise KeyError(pass_id)
        triggers.append(trigger_record)
        local_root = Path(str(capture_pass["local_root"]))
        self.serializer.write_json(
            local_root / f"trigger_{trigger_index:04d}" / "metadata.json",
            document,
        )
        defects_by_id = {str(defect["defect_id"]): defect for defect in defect_records}
        frames_by_id = {str(frame["frame_id"]): frame for frame in frame_records}
        for artifact in artifact_records:
            if artifact.get("kind") != "semantic_labels":
                continue
            frame = frames_by_id.get(str(artifact.get("frame_id")), {})
            visible_defects = [
                defects_by_id[str(defect_id)]
                for defect_id in frame.get("defect_ids", [])
                if str(defect_id) in defects_by_id
            ]
            self.serializer.enrich_semantic_labels(
                [local_root / str(artifact["path"])],
                visible_defects,
            )
        self.serialize_index()
        return copy.deepcopy(document)

    def record_derived_artifacts(
        self,
        pass_id: str,
        artifacts: Iterable[Mapping[str, object]],
    ) -> None:
        """Attach postprocess outputs to their source capture pass.

        Args:
            pass_id: Source capture render JobOrder identifier.
            artifacts: Derived artifact descriptions.

        Example:

        .. code-block:: python

            observer.record_derived_artifacts(
                "capture.render", [{"kind": "stitched_rgb", "path": "panel.png"}]
            )

        """
        capture_pass = self._passes[pass_id]
        derived = capture_pass["derived_artifacts"]
        if not isinstance(derived, list):
            raise KeyError(pass_id)
        artifact_records = copy.deepcopy(list(artifacts))
        derived.extend(artifact_records)
        triggers = capture_pass["triggers"]
        if isinstance(triggers, list):
            for artifact in artifact_records:
                trigger_index = artifact.get("trigger_idx")
                if trigger_index is None:
                    continue
                for trigger in triggers:
                    if trigger["trigger_idx"] == trigger_index:
                        trigger["document"].setdefault(
                            "derived_artifacts",
                            [],
                        ).append(copy.deepcopy(artifact))
                        break
        local_root = Path(str(capture_pass["local_root"]))
        for artifact in artifact_records:
            if artifact.get("kind") != "semantic_labels":
                continue
            visible_ids = set()
            has_trigger_scope = False
            trigger_index = artifact.get("trigger_idx")
            if isinstance(triggers, list):
                for trigger in triggers:
                    if trigger["trigger_idx"] != trigger_index:
                        continue
                    has_trigger_scope = True
                    visible_ids.update(
                        str(defect_id)
                        for frame in trigger["document"].get("frames", [])
                        for defect_id in frame.get("defect_ids", [])
                    )
                    break
            visible_defects = [
                defect
                for defect_id, defect in self._defects.items()
                if not has_trigger_scope or defect_id in visible_ids
            ]
            self.serializer.enrich_semantic_labels(
                [local_root / str(artifact["path"])],
                visible_defects,
            )
        self.rewrite_trigger_documents(pass_id)

    def remap_artifacts(
        self,
        pass_id: str,
        path_mapping: Mapping[str, str],
    ) -> None:
        """Update frame and artifact paths after a formatter renames files.

        Args:
            pass_id: Source capture render JobOrder identifier.
            path_mapping: Old-to-new paths relative to the capture-pass root.

        Example:

        .. code-block:: python

            observer.remap_artifacts(
                "capture.render", {"old.png": "semantic_mask.png"}
            )

        """
        capture_pass = self._passes[pass_id]
        triggers = capture_pass["triggers"]
        if not isinstance(triggers, list):
            return
        for trigger in triggers:
            document = trigger["document"]
            for artifact in document.get("artifacts", []):
                old_path = artifact.get("path")
                if old_path in path_mapping:
                    artifact["path"] = path_mapping[old_path]
            for frame in document.get("frames", []):
                for paths in frame.get("artifacts", {}).values():
                    for index, old_path in enumerate(paths):
                        if old_path in path_mapping:
                            paths[index] = path_mapping[old_path]
        self.rewrite_trigger_documents(pass_id)

    def rewrite_trigger_documents(self, pass_id: str) -> None:
        """Rewrite a pass after a postprocessor changes artifact files.

        Args:
            pass_id: Capture render JobOrder identifier.

        Example:

        .. code-block:: python

            observer.rewrite_trigger_documents("capture.render")

        """
        capture_pass = self._passes[pass_id]
        local_root = Path(str(capture_pass["local_root"]))
        triggers = capture_pass["triggers"]
        if not isinstance(triggers, list):
            return
        for trigger in triggers:
            self.serializer.write_json(
                local_root / str(trigger["metadata_path"]),
                trigger["document"],
            )
        self.serialize_index()

    def serialize_index(self) -> None:
        """Write a dataset-wide index into every registered pass root.

        Example:

        .. code-block:: python

            observer.serialize_index()

        """
        index = self.index_document()
        for capture_pass in self._passes.values():
            local_root = Path(str(capture_pass["local_root"]))
            self.serializer.write_json(local_root / "dataset_index.json", index)

    def index_document(self) -> dict[str, object]:
        """Build the dataset-wide compatibility index.

        Returns:
            Schema document linking every capture pass and trigger manifest.

        Example:

        .. code-block:: python

            print(observer.index_document()["schema_name"])
            cdf.dataset_manifest.index

        """
        passes = []
        artifact_index = []
        for capture_pass in self._passes.values():
            triggers = capture_pass["triggers"]
            if isinstance(triggers, list):
                for trigger in triggers:
                    for artifact in trigger["document"].get("artifacts", []):
                        artifact_index.append(
                            {
                                **copy.deepcopy(artifact),
                                "pass_id": capture_pass["pass_id"],
                                "trigger_idx": trigger["trigger_idx"],
                            }
                        )
            for artifact in capture_pass["derived_artifacts"]:
                indexed_artifact = {
                    **copy.deepcopy(artifact),
                    "pass_id": capture_pass["pass_id"],
                }
                indexed_artifact.setdefault(
                    "artifact_id",
                    self._stable_id(
                        "artifact",
                        str(capture_pass["pass_id"]),
                        str(artifact.get("kind") or ""),
                        str(artifact.get("path") or ""),
                    ),
                )
                artifact_index.append(indexed_artifact)
            passes.append(
                {
                    "pass_id": capture_pass["pass_id"],
                    "name": capture_pass["name"],
                    "output_uri": capture_pass["output_uri"],
                    "local_root": capture_pass["local_root"],
                    "triggers": [
                        {
                            "trigger_idx": trigger["trigger_idx"],
                            "metadata_path": trigger["metadata_path"],
                            "frame_count": len(trigger["document"].get("frames", [])),
                        }
                        for trigger in triggers
                    ]
                    if isinstance(triggers, list)
                    else [],
                    "derived_artifacts": copy.deepcopy(
                        capture_pass["derived_artifacts"]
                    ),
                }
            )
        return {
            "schema_name": f"{SCHEMA_NAME}.index",
            "schema_version": SCHEMA_VERSION,
            "dataset_id": self._dataset_id,
            "plan_metadata": copy.deepcopy(self._plan_metadata),
            "provenance": copy.deepcopy(self._provenance),
            "entities": copy.deepcopy(list(self._entities.values())),
            "defects": copy.deepcopy(list(self._defects.values())),
            "artifacts": artifact_index,
            "passes": passes,
        }

    def view(self) -> ManifestView:
        """Return an immutable-by-copy query view of current observations.

        Returns:
            Read-only manifest query facade.

        Example:

        .. code-block:: python

            print(observer.view().dataset_id)  # doctest: +SKIP

        """
        return ManifestView(
            {
                "dataset_id": self._dataset_id,
                "plan_metadata": self._plan_metadata,
                "entities": list(self._entities.values()),
                "defects": list(self._defects.values()),
                "passes": self._passes,
            }
        )

    @staticmethod
    def _stable_id(namespace: str, *parts: str) -> str:
        """Return a stable UUID-backed identifier for normalized values."""
        value = "\x1f".join((namespace, *parts))
        return f"{namespace}:{uuid.uuid5(uuid.NAMESPACE_URL, value)}"

    @staticmethod
    def _severity(
        record: Mapping[str, object],
        defect_type: str,
    ) -> dict[str, object]:
        """Return sampled severity fields from one producer record."""
        parameters = record.get(f"{defect_type}_params")
        if isinstance(parameters, Mapping):
            return copy.deepcopy(dict(parameters))
        ignored = {
            "path",
            "defect",
            "original_flat",
            f"{defect_type}_final_flat",
        }
        return {
            key: copy.deepcopy(value)
            for key, value in record.items()
            if key not in ignored
        }

    @staticmethod
    def _normalize_calibration(calibration: Mapping[str, object]) -> dict[str, object]:
        """Coerce one calibration record to plain JSON types.

        The values arrive from USD, so they are as likely to be ``Gf`` scalars
        or numpy floats as Python ones. They serialize by accident today and
        raise on some other board tomorrow, which is the kind of failure that
        only shows up once a dataset is already half written.

        Args:
            calibration: Raw calibration mapping recorded by the capture.

        Returns:
            The same fields, with every number a Python ``float`` or ``int``.

        """

        def _number(value: object) -> object:
            if isinstance(value, bool) or value is None:
                return value
            if isinstance(value, numbers.Integral):
                return int(value)
            if isinstance(value, numbers.Real):
                return float(value)
            return value

        def _coerce(value: object) -> object:
            if isinstance(value, Mapping):
                return {str(k): _coerce(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_coerce(item) for item in value]
            return _number(value)

        return {str(key): _coerce(value) for key, value in calibration.items()}

    @classmethod
    def _normalize_frame_cameras(cls, frames: Iterable[object]) -> None:
        """Normalize every frame's camera calibration in place before writing.

        Runs ahead of ``_validate_document``, so it must tolerate frames that
        are not records at all and leave complaining about them to validation.
        """
        for frame in frames:
            if not isinstance(frame, MutableMapping):
                continue
            camera = frame.get("camera")
            if not isinstance(camera, MutableMapping):
                continue
            calibration = camera.get("calibration")
            if isinstance(calibration, Mapping):
                camera["calibration"] = cls._normalize_calibration(calibration)

    @staticmethod
    def _validate_calibration(
        frame_id: object, calibration: Mapping[str, object]
    ) -> None:
        """Reject a calibration that cannot mean what it claims.

        Args:
            frame_id: Identifier of the frame carrying the calibration.
            calibration: Normalized calibration record.

        Raises:
            ValidationError: If a matrix has the wrong shape or a field that
                must be numeric is not.

        """
        if calibration.get("unavailable") is not None:
            # An explicit reason is a valid record; there is no geometry to check.
            return
        shapes = {
            "intrinsic_matrix": (3, 3),
            "world_to_camera": (4, 4),
            "camera_to_world": (4, 4),
        }
        for field, (rows, cols) in shapes.items():
            matrix = calibration.get(field)
            if matrix is None:
                continue
            valid = (
                isinstance(matrix, list)
                and len(matrix) == rows
                and all(
                    isinstance(row, list)
                    and len(row) == cols
                    and all(isinstance(value, (int, float)) for value in row)
                    for row in matrix
                )
            )
            if not valid:
                raise ValidationError(
                    f"frame {frame_id} camera calibration {field} must be a "
                    f"{rows}x{cols} numeric matrix"
                )
        for field, length in (("resolution_px", 2), ("mm_per_px", 2)):
            value = calibration.get(field)
            if value is None:
                continue
            if not (
                isinstance(value, list)
                and len(value) == length
                and all(isinstance(item, (int, float)) for item in value)
            ):
                raise ValidationError(
                    f"frame {frame_id} camera calibration {field} must be "
                    f"{length} numbers"
                )

    @staticmethod
    def _validate_document(document: Mapping[str, object]) -> None:
        """Validate frame, artifact, and defect references before writing."""
        defects = document.get("defects")
        entities = document.get("entities")
        frames = document.get("frames")
        artifacts = document.get("artifacts")
        if not all(
            isinstance(value, list) for value in (entities, defects, frames, artifacts)
        ):
            raise ValidationError("dataset manifest collections must be lists")
        collections = {
            "entities": entities,
            "defects": defects,
            "frames": frames,
            "artifacts": artifacts,
        }
        for name, collection in collections.items():
            if not all(isinstance(item, Mapping) for item in collection):
                raise ValidationError(
                    f"dataset manifest {name} entries must be objects"
                )
        entity_ids = {item.get("entity_id") for item in entities}
        defect_ids = {item.get("defect_id") for item in defects}
        frame_ids = {item.get("frame_id") for item in frames}
        if (
            None in entity_ids
            or None in defect_ids
            or None in frame_ids
            or len(frame_ids) != len(frames)
        ):
            raise ValidationError(
                "dataset manifest identifiers must be present and unique"
            )
        for defect in defects:
            if defect.get("entity_id") not in entity_ids:
                raise ValidationError(
                    f"defect {defect['defect_id']} references an unknown entity"
                )
        for frame in frames:
            unknown = set(frame.get("defect_ids", ())) - defect_ids
            if unknown:
                raise ValidationError(
                    f"frame {frame['frame_id']} references unknown defects {sorted(unknown)}"
                )
            camera = frame.get("camera")
            calibration = (
                camera.get("calibration") if isinstance(camera, Mapping) else None
            )
            if isinstance(calibration, Mapping):
                DatasetManifestObserver._validate_calibration(
                    frame["frame_id"], calibration
                )
        for artifact in artifacts:
            frame_id = artifact.get("frame_id")
            if frame_id is not None and frame_id not in frame_ids:
                raise ValidationError(
                    f"artifact {artifact.get('path')} references unknown frame {frame_id}"
                )
