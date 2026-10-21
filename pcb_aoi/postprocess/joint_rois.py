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

"""Generate ``joint_rois.json`` from per-tile tight bounding box outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from cdf.errors import ValidationError
from pcb_aoi.board_metadata import ComponentRecord

_NP_PATTERN = re.compile(r"^bounding_box_2d_tight_x(\d+)_y(\d+)\.npy$")
_UNAVAILABLE_METADATA_VALUES = frozenset({"not_yet_available"})


def _stitched_output_name(trigger_index: int, trigger_count: int) -> str:
    """Return a collision-safe stitched companion filename."""
    if trigger_count == 1:
        return "joint_rois.json"
    return f"joint_rois_trigger_{trigger_index:04d}.json"


def _metadata_value(value: str | None) -> str | None:
    """Return a usable metadata value or None for an unavailable placeholder."""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in _UNAVAILABLE_METADATA_VALUES:
        return None
    return normalized


def _component_metadata_by_ref(
    components: Iterable[ComponentRecord],
) -> dict[str, dict[str, str | None]]:
    """Build unambiguous component metadata keyed by reference designator.

    ``component_type`` prefers the catalog's authored semantic type, then its
    family, package type, footprint identity, and package name. The parallel
    ``component_type_source`` field identifies which catalog field supplied the
    value. Conflicting records for the same refdes are omitted instead of
    attaching metadata from the wrong board.

    Args:
        components: Component records from the selected board metadata catalog.

    Returns:
        Metadata keyed by refdes, excluding ambiguous identities.

    """
    metadata_by_ref: dict[str, dict[str, str | None]] = {}
    ambiguous_refs: set[str] = set()
    for component in components:
        candidates = (
            ("component_type", component.component_type),
            ("family", component.family),
            ("package_type", component.package_type),
            ("footprint_id", component.footprint_id),
            ("package_name", component.package_name),
        )
        component_type = None
        component_type_source = None
        for source, value in candidates:
            component_type = _metadata_value(value)
            if component_type is not None:
                component_type_source = source
                break
        metadata = {
            "component_type": component_type,
            "component_type_source": component_type_source,
            "package_type": _metadata_value(component.package_type),
            "package_name": _metadata_value(component.package_name),
            "footprint_id": _metadata_value(component.footprint_id),
            "process_class": _metadata_value(component.process_class),
        }
        existing = metadata_by_ref.get(component.refdes)
        if existing is not None and existing != metadata:
            ambiguous_refs.add(component.refdes)
            metadata_by_ref.pop(component.refdes, None)
        elif component.refdes not in ambiguous_refs:
            metadata_by_ref[component.refdes] = metadata
    return metadata_by_ref


def _lead_id(lead_name: str) -> str:
    """Return the trailing numeric lead identifier, or the whole name."""
    lead_match = re.search(r"(\d+)$", lead_name)
    return lead_match.group(1) if lead_match else lead_name


def _component_lead(parts: list[str], start: int) -> tuple[str, str] | None:
    """Return (component_ref, lead_id) from a ``Components/<ref>/<lead>`` run."""
    try:
        components_idx = parts.index("Components", start)
    except ValueError:
        return None
    # Require at least: Components / <ref> / <lead>.
    if len(parts) < components_idx + 3:
        return None
    return parts[components_idx + 1], _lead_id(parts[components_idx + 2])


def _board_instance_segment(prim_path: str) -> str | None:
    """Return the ``StepRepeats/<instance>`` segment of a prim path, if any."""
    parts = prim_path.split("/")
    try:
        step_idx = parts.index("StepRepeats")
    except ValueError:
        return None
    if len(parts) < step_idx + 2:
        return None
    return parts[step_idx + 1]


def _parse_component_key(prim_path: str) -> tuple[str, str, str] | None:
    """Derive (board_instance, component_ref, lead_id) from a component prim path.

    Expects USD paths with a ``StepRepeats/<instance>/.../Components/<ref>/<lead>``
    segment. Optional intermediate scopes accommodate annotation proxies such
    as ``SegmentationProxies/PinTips/Components``. Returns None for paths that do
    not match this structure (e.g., standalone layer pads).
    """
    parts = prim_path.split("/")
    try:
        step_idx = parts.index("StepRepeats")
    except ValueError:
        return None
    component_lead = _component_lead(parts, step_idx + 2)
    if component_lead is None:
        return None
    return parts[step_idx + 1], component_lead[0], component_lead[1]


def build_pad_joint_keys(
    pads: Iterable[object],
    *,
    source_root: str | None = None,
    instance_paths: Iterable[str] = (),
) -> dict[str, tuple[str, str]]:
    """Map board-feature prim paths to the joint they belong to.

    Each record must expose ``prim_path``, ``component_path`` and
    ``terminal_number``; ``contact_prim_path`` is used when present. These are
    the exporter-stated pad associations of a board metadata catalog
    (``PadRecord`` and ``HoleRecord``); nothing is inferred from XY proximity.
    The lead identifier is taken from the associated contact prim name when the
    catalog resolved one, so it matches the ``<lead>`` segment that
    :func:`generate_joint_rois` parses from the lead path, and otherwise from
    the stated terminal number. Records without a component or terminal are
    skipped. A prim path claimed by two different joints is dropped rather
    than assigned to either.

    When the catalog was read from a shared board prototype, pass its
    ``source_root`` and the composed ``instance_paths``: every record below
    ``source_root`` is re-rooted beneath each instance, which is where the
    capture pass reports it.

    Args:
        pads: Pad or hole records from the selected board metadata catalog.
        source_root: Board root the catalog paths are relative to, if any.
        instance_paths: Composed board-instance paths sharing that source.

    Returns:
        Absolute prim path mapped to ``(component_ref, lead_id)``.

    Example:

    .. code-block:: python

        from pcb_aoi.postprocess.joint_rois import build_pad_joint_keys

        keys = build_pad_joint_keys(catalog.pads)

    """
    keys: dict[str, tuple[str, str]] = {}
    conflicting: set[str] = set()
    instances = tuple(str(path).rstrip("/") for path in instance_paths)
    root = source_root.rstrip("/") if source_root else None
    for record in pads:
        component_path = getattr(record, "component_path", None)
        if not component_path:
            continue
        contact_path = getattr(record, "contact_prim_path", None)
        if contact_path:
            lead_id = _lead_id(str(contact_path).rsplit("/", 1)[-1])
        else:
            terminal_number = getattr(record, "terminal_number", None)
            if terminal_number is None or not str(terminal_number).strip():
                continue
            lead_id = _lead_id(str(terminal_number).strip())
        joint = (str(component_path).rstrip("/").rsplit("/", 1)[-1], lead_id)
        prim_path = str(record.prim_path)
        if root and instances and prim_path.startswith(f"{root}/"):
            suffix = prim_path[len(root) :]
            targets = tuple(f"{instance}{suffix}" for instance in instances)
        else:
            targets = (prim_path,)
        for target in targets:
            if target in conflicting:
                continue
            existing = keys.get(target)
            if existing is not None and existing != joint:
                conflicting.add(target)
                keys.pop(target)
            else:
                keys[target] = joint
    return keys


def _union_box(
    boxes: dict[str, list[float]],
    key: str,
    box: list[float],
) -> None:
    """Grow ``boxes[key]`` to include ``box`` (``[x1, y1, x2, y2]``)."""
    prev = boxes.get(key)
    if prev is None:
        boxes[key] = list(box)
    else:
        boxes[key] = [
            min(prev[0], box[0]),
            min(prev[1], box[1]),
            max(prev[2], box[2]),
            max(prev[3], box[3]),
        ]


def generate_joint_rois(
    roi_root: str | Path,
    trigger_metadata_roi: list[dict[str, object]],
    *,
    rgb_pass_name: str = "full_board",
    roi_pass_name: str = "full_board_roi",
    output_roots_by_pass: Mapping[str, str | Path] | None = None,
    instance_label_by_segment: dict[str, str] | None = None,
    component_metadata: Mapping[str, Mapping[str, str | None]] | None = None,
    pad_joint_keys: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, object]:
    """Generate ``joint_rois.json`` from per-tile tight bounding box outputs.

    Reads ``bounding_box_2d_tight_x*_y*.npy`` and the parallel
    ``bounding_box_2d_tight_prim_paths_x*_y*.json`` files, transforms
    each entry from per-tile pixel coordinates to full-board canvas
    coordinates using the tile ``dst_rect_xywh_px`` from the pixel layout,
    unions bboxes that span multiple tiles for the same component, and
    writes ``joint_rois.json`` to each trigger directory for every participating
    capture pass. A companion copy is written beside each stitched image as
    ``stitched/joint_rois.json`` for a single trigger or
    ``stitched/joint_rois_trigger_XXXX.json`` for multiple triggers. Component
    metadata fields remain present with null values when their source does not
    provide a value.

    Joints are defined by entries whose prim path contains a
    ``StepRepeats/<instance>/.../Components/<ref>/<lead>`` segment; their union
    is ``bbox_xyxy`` (the pin). Layer pads and other board features never
    create a joint. A board-feature entry listed in ``pad_joint_keys`` is
    attributed to its stated joint in the same board instance, and
    ``joint_bbox_xyxy`` is the union of the pin box and those boxes. Such
    entries appear only when the ROI pass labels them (for example
    ``annotations.geometry.pads``). A joint with no associated pad box keeps
    ``joint_bbox_xyxy == bbox_xyxy`` and reports ``joint_bbox_source`` as
    ``"pin_only"``.

    **Capture-pass requirement**: the ROI capture pass must be configured with
    ``rename_to_grid_index: true`` in its ``outputs`` block.  Both
    ``bounding_box_2d_tight_x*_y*.npy`` and the parallel
    ``bounding_box_2d_tight_prim_paths_x*_y*.json`` files are located by the
    ``x<col>_y<row>`` suffix that setting produces, which also provides the
    tile indices used to look up ``dst_rect_xywh_px`` from the pixel layout.

    Args:
        roi_root: Local root of the ROI capture pass.
        trigger_metadata_roi: Per-trigger metadata from the ROI capture order.
        rgb_pass_name: Name written to ``source_pass`` in the output document.
        roi_pass_name: Name of the ROI capture pass used to derive bounding boxes.
        output_roots_by_pass: Optional capture-pass names mapped to local output
            roots. The manifest is written beneath every listed pass. The ROI
            pass root is always included.
        instance_label_by_segment: Optional mapping from USD step-repeat path
            segment (e.g. ``"SR_0_0"``) to a stable cohort label (e.g. ``"0"``).
            When provided, ``board_instance`` in the output uses the label;
            otherwise the raw path segment is used.
        component_metadata: Optional component metadata keyed by refdes. Output
            fields remain present with null values when a refdes is unavailable.
        pad_joint_keys: Optional absolute board-feature prim path mapped to
            ``(component_ref, lead_id)``, normally built with
            :func:`build_pad_joint_keys` from the catalog's stated pad
            associations. Without it every joint is ``"pin_only"``.

    Returns:
        Mapping containing the joint count, the number of joints whose
        ``joint_bbox_xyxy`` is the pin box alone, generated artifacts, and
        execution status.

    Raises:
        ValidationError: If a ``.npy`` file cannot be read.

    Example:

    .. code-block:: python

        from pcb_aoi.postprocess.joint_rois import generate_joint_rois

        result = generate_joint_rois("/tmp/full_board_roi", [])
        print(result["joint_count"])
        # 0

    """
    import numpy as np

    roi_root = Path(roi_root)
    output_roots = {
        str(pass_name): Path(output_root)
        for pass_name, output_root in (output_roots_by_pass or {}).items()
    }
    output_roots[roi_pass_name] = roi_root
    total_joints = 0
    total_pin_only = 0
    artifacts: list[dict[str, object]] = []
    trigger_count = len(trigger_metadata_roi)

    for meta in trigger_metadata_roi:
        trigger_idx = int(meta["trigger_idx"])
        trigger_dir = roi_root / f"trigger_{trigger_idx:04d}"
        tiles: list[dict] = meta.get("pixel_layout", {}).get("tiles", [])
        tile_by_xy = {(int(t["x_idx"]), int(t["y_idx"])): t for t in tiles}

        # Map each `joint_key` to `[x1, y1, x2, y2]` in full-board canvas coordinates.
        joint_bboxes: dict[str, list[float]] = {}
        joint_prim_paths: dict[str, set[str]] = {}
        # Associated pad boxes, keyed like the joints they belong to.
        pad_bboxes: dict[str, list[float]] = {}
        pad_prim_paths: dict[str, set[str]] = {}

        for npy_path in sorted(trigger_dir.iterdir()):
            match = _NP_PATTERN.fullmatch(npy_path.name)
            if not match:
                continue
            x_idx, y_idx = int(match.group(1)), int(match.group(2))
            tile = tile_by_xy.get((x_idx, y_idx))
            if tile is None:
                continue
            rect = tile["dst_rect_xywh_px"]
            tile_left, tile_top = float(rect[0]), float(rect[1])

            paths_file = trigger_dir / (
                f"bounding_box_2d_tight_prim_paths_x{x_idx}_y{y_idx}.json"
            )
            if not paths_file.is_file():
                continue
            prim_paths: list[str] = json.loads(paths_file.read_text(encoding="utf-8"))

            try:
                bboxes = np.load(str(npy_path), allow_pickle=False)
            except Exception as exc:
                raise ValidationError(
                    f"could not read {npy_path.name}: {exc}",
                ) from exc

            for bbox, prim_path in zip(bboxes, prim_paths):
                box = [
                    tile_left + float(bbox["x_min"]),
                    tile_top + float(bbox["y_min"]),
                    tile_left + float(bbox["x_max"]),
                    tile_top + float(bbox["y_max"]),
                ]
                parsed = _parse_component_key(prim_path)
                if parsed is None:
                    pad_joint = (pad_joint_keys or {}).get(prim_path)
                    if pad_joint is None:
                        continue
                    instance_segment = _board_instance_segment(prim_path)
                    if instance_segment is None:
                        continue
                    component_ref, lead_id = pad_joint
                    target = pad_bboxes
                    target_paths = pad_prim_paths
                else:
                    instance_segment, component_ref, lead_id = parsed
                    target = joint_bboxes
                    target_paths = joint_prim_paths
                if instance_label_by_segment:
                    board_instance = instance_label_by_segment.get(
                        instance_segment, instance_segment
                    )
                else:
                    board_instance = instance_segment
                joint_key = f"{board_instance}/{component_ref}/{lead_id}"
                target_paths.setdefault(joint_key, set()).add(prim_path)
                _union_box(target, joint_key, box)

        items = []
        for joint_key, (x1, y1, x2, y2) in sorted(joint_bboxes.items()):
            joint_box = {joint_key: [x1, y1, x2, y2]}
            has_pads = joint_key in pad_bboxes
            if has_pads:
                _union_box(joint_box, joint_key, pad_bboxes[joint_key])
            else:
                total_pin_only += 1
            key_parts = joint_key.split("/", 2)
            component_ref = key_parts[1] if len(key_parts) > 1 else joint_key
            pin_number = key_parts[2] if len(key_parts) > 2 else None
            metadata = (component_metadata or {}).get(component_ref, {})
            items.append(
                {
                    "joint_id": joint_key,
                    "board_instance": key_parts[0],
                    "component_ref": component_ref,
                    "pin_number": pin_number,
                    "component_type": metadata.get("component_type"),
                    "component_type_source": metadata.get("component_type_source"),
                    "package_type": metadata.get("package_type"),
                    "package_name": metadata.get("package_name"),
                    "footprint_id": metadata.get("footprint_id"),
                    "process_class": metadata.get("process_class"),
                    "source_prim_paths": sorted(joint_prim_paths[joint_key]),
                    "bbox_xyxy": [round(x1), round(y1), round(x2), round(y2)],
                    "pad_prim_paths": sorted(pad_prim_paths.get(joint_key, ())),
                    "joint_bbox_xyxy": [round(value) for value in joint_box[joint_key]],
                    "joint_bbox_source": "pin_and_pads" if has_pads else "pin_only",
                },
            )

        output = {
            "schema_version": "1.0",
            "coordinate_system": "pixel_top_left_xyxy_half_open",
            "source_pass": rgb_pass_name,
            "source_trigger": f"trigger_{trigger_idx:04d}",
            "items": items,
        }
        serialized_output = json.dumps(output, indent=2) + "\n"
        stitched_name = _stitched_output_name(trigger_idx, trigger_count)
        for pass_name, output_root in output_roots.items():
            output_trigger_dir = output_root / f"trigger_{trigger_idx:04d}"
            output_trigger_dir.mkdir(parents=True, exist_ok=True)
            (output_trigger_dir / "joint_rois.json").write_text(
                serialized_output,
                encoding="utf-8",
            )
            stitched_dir = output_root / "stitched"
            stitched_dir.mkdir(parents=True, exist_ok=True)
            (stitched_dir / stitched_name).write_text(
                serialized_output,
                encoding="utf-8",
            )
            artifacts.append(
                {
                    "capture_pass": pass_name,
                    "kind": "joint_rois",
                    "path": f"trigger_{trigger_idx:04d}/joint_rois.json",
                    "trigger_idx": trigger_idx,
                    "joint_count": len(items),
                }
            )
            artifacts.append(
                {
                    "capture_pass": pass_name,
                    "kind": "joint_rois_stitched",
                    "path": f"stitched/{stitched_name}",
                    "trigger_idx": trigger_idx,
                    "joint_count": len(items),
                }
            )
        total_joints += len(items)

    return {
        "joint_count": total_joints,
        "pin_only_joint_count": total_pin_only,
        "artifacts": artifacts,
        "executed": True,
    }
