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

"""Build a DAFT dataset from Cosmos-3 pair records.

The packaging itself: how a pair record becomes DAFT contextual blocks, which raw
files it references, and what the manifest says. Kept free of CLI concerns so the
same functions can be driven from a ``job.postprocess`` step as easily as from
``daft_pack.py``.

Pure standard library -- no Isaac, no Pillow, no numpy.
"""

from __future__ import annotations

import json
import shutil
import sys
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

_DAFT_SCHEMA_VERSION = "daft_cosmos3_v0.1"
_PASSES = ("clean", "defect")


# ── pure structure builders (dict -> dict; no IO) ────────────────────────────

def _views(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("views") or {}


def _view_ids(record: dict[str, Any]) -> list[str]:
    ids = record.get("view_ids")
    if ids:
        return list(ids)
    return list(_views(record).keys())


def scene_context(record: dict[str, Any]) -> dict[str, Any]:
    """Scene-level context: board plane, 3D box, seg legend (``scene.json``)."""
    return {
        "schema_version": _DAFT_SCHEMA_VERSION,
        "scene_id": record.get("scene_id"),
        "source_schema_version": record.get("schema_version"),
        "capture_mode": record.get("capture_mode"),
        "view_ids": _view_ids(record),
        "board_plane_global": record.get("board_plane_global"),
        "bbox_3d_global": record.get("bbox_3d_global"),
        "segmentation": record.get("segmentation"),
    }


def calibration_context(record: dict[str, Any]) -> dict[str, Any]:
    """One calibration record per view: intrinsics K + extrinsics.

    Clean and defect passes of a view share one camera pose (that is what makes
    the pair pixel-aligned), so a single record per view is correct.
    """
    out: dict[str, Any] = {}
    for view_id in _view_ids(record):
        vrec = _views(record).get(view_id, {})
        calib = vrec.get("calibration") or {}
        out[view_id] = {
            "K": calib.get("K"),
            "world_to_camera": calib.get("world_to_camera"),
            "camera_to_world": calib.get("camera_to_world"),
            "resolution_px": vrec.get("resolution_px"),
        }
    return {"views": out}


def _raw_uri(view_id: str, pass_name: str, kind: str) -> str:
    ext = "png" if kind == "rgb" else "exr"
    return f"raw/{kind}/{view_id}/{pass_name}.{ext}"


def images_context(record: dict[str, Any]) -> dict[str, Any]:
    """Per view/pass image + depth URIs, re-homed onto the DAFT tree."""
    out: dict[str, Any] = {}
    for view_id in _view_ids(record):
        vrec = _views(record).get(view_id, {})
        passes: dict[str, Any] = {}
        for pass_name in _PASSES:
            prec = vrec.get(pass_name) or {}
            # Each output is gated on its OWN source uri, matching
            # raw_file_plan exactly. Deciding both from image_uri meant a record
            # with an image but no depth still advertised a depth file that was
            # never copied, and a record with depth but no image copied a file
            # the manifest then said was absent. images.json has to describe the
            # files that are actually there.
            passes[pass_name] = {
                "rgb": (_raw_uri(view_id, pass_name, "rgb")
                        if prec.get("image_uri") else None),
                "depth": (_raw_uri(view_id, pass_name, "depth")
                          if prec.get("depth_uri") else None),
            }
        out[view_id] = {
            "depth_format": vrec.get("depth_format"),
            "depth_units": vrec.get("depth_units"),
            "depth_definition": vrec.get("depth_definition"),
            "passes": passes,
        }
    return {"views": out}


def defects_context(record: dict[str, Any]) -> dict[str, Any]:
    """Defect block + per-view bbox_old / bbox_new / union + occlusion."""
    defect = dict(record.get("defect") or {})
    per_view: dict[str, Any] = {}
    for view_id in _view_ids(record):
        vrec = _views(record).get(view_id, {})
        per_view[view_id] = {
            "bbox_old_xyxy": vrec.get("bbox_old_xyxy"),
            "bbox_new_xyxy": vrec.get("bbox_new_xyxy"),
            "bbox_union_xyxy": vrec.get("bbox_union_xyxy"),
            "occlusion": vrec.get("occlusion"),
        }
    return {"defect": defect, "views": per_view}


def generation_context(record: dict[str, Any]) -> dict[str, Any]:
    """Generation-parameter passthrough (``generation.json``)."""
    return {"generation": record.get("generation")}


def build_contextual(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """All five contextual JSON payloads for one scene."""
    return {
        "scene": scene_context(record),
        "calibration": calibration_context(record),
        "images": images_context(record),
        "defects": defects_context(record),
        "generation": generation_context(record),
    }


def _contained(root: Path, *parts: str) -> Path:
    """Join ``parts`` under ``root`` and refuse to leave it.

    Every path segment here comes out of a capture record, and a record is data
    rather than trusted input -- a ``scene_id`` or ``image_uri`` carrying ``..``
    or an absolute path would otherwise let packing read outside the raw dataset
    or write outside the DAFT root. Resolved without touching the filesystem so
    the check does not depend on what happens to exist yet.

    Args:
        root: Directory the result must stay inside.
        parts: Record-derived path segments.

    Returns:
        The joined path.

    Raises:
        ValueError: If a segment is absolute or escapes ``root``.
    """
    root = Path(root)
    for part in parts:
        if PurePosixPath(part).is_absolute() or Path(part).is_absolute():
            raise ValueError(f"path from record is absolute, refusing: {part!r}")
    candidate = root.joinpath(*parts)
    resolved_root = Path(os.path.normpath(root.absolute()))
    resolved = Path(os.path.normpath(candidate.absolute()))
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(
            f"path from record escapes {root}: {'/'.join(parts)!r}",
        )
    return candidate


def raw_file_plan(record: dict[str, Any]) -> list[tuple[str, str]]:
    """``[(source_uri, dst_relpath), ...]`` for every RGB + EXR to copy.

    ``source_uri`` is the record's own uri (relative to the raw dataset root);
    ``dst_relpath`` is relative to ``scenes/<scene_id>/``.  Passes without a
    source uri are skipped (nothing to copy).
    """
    plan: list[tuple[str, str]] = []
    for view_id in _view_ids(record):
        vrec = _views(record).get(view_id, {})
        for pass_name in _PASSES:
            prec = vrec.get(pass_name) or {}
            img = prec.get("image_uri")
            dep = prec.get("depth_uri")
            if img:
                plan.append((img, _raw_uri(view_id, pass_name, "rgb")))
            if dep:
                plan.append((dep, _raw_uri(view_id, pass_name, "depth")))
    return plan


def manifest(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Dataset manifest indexing every packed scene."""
    scenes = []
    for record in records:
        defect = record.get("defect") or {}
        scenes.append({
            "scene_id": record.get("scene_id"),
            "dir": f"scenes/{record.get('scene_id')}",
            "capture_mode": record.get("capture_mode"),
            "defect_type": defect.get("defect_type"),
            "defect_class_id": defect.get("defect_class_id"),
            "has_defect": defect.get("has_defect"),
            "view_ids": _view_ids(record),
        })
    return {
        "schema_version": _DAFT_SCHEMA_VERSION,
        "generator": "daft_pack.py",
        "scene_count": len(scenes),
        "scenes": scenes,
    }


# ── file IO (copy callable injectable for testing) ───────────────────────────

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def pack(
    record: dict[str, Any],
    src_root: Path,
    dst_root: Path,
    *,
    copy: Callable[[Path, Path], Any] = shutil.copy2,
    strict: bool = False,
) -> Path:
    """Materialize one scene's DAFT tree; returns the scene directory.

    Args:
        record: A ``*.pair.json`` raw record dict.
        src_root: Root of the raw capture output (uris are relative to it).
        dst_root: DAFT dataset root to write ``scenes/<id>/`` under.
        copy: ``copy(src_path, dst_path)`` callable (injectable for tests).
        strict: Raise if a planned source file is missing (default: skip).

    Returns:
        The scene directory ``dst_root / "scenes" / <scene_id>``.
    """
    src_root = Path(src_root)
    scene_id = str(record.get("scene_id"))
    scene_dir = _contained(Path(dst_root) / "scenes", scene_id)

    for src_uri, dst_rel in raw_file_plan(record):
        src_path = _contained(src_root, src_uri)
        dst_path = _contained(scene_dir, dst_rel)
        if not src_path.exists():
            if strict:
                raise FileNotFoundError(src_path)
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        copy(src_path, dst_path)

    for name, payload in build_contextual(record).items():
        _write_json(_contained(scene_dir / "contextual", f"{name}.json"), payload)
    return scene_dir


def load_records(src_root: Path) -> list[dict[str, Any]]:
    """Load and sort every ``*.pair.json`` raw record under ``src_root``."""
    src_root = Path(src_root)
    records = []
    for path in sorted(src_root.glob("*.pair.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def pack_dir(
    src_root: Path,
    dst_root: Path,
    *,
    copy: Callable[[Path, Path], Any] = shutil.copy2,
    strict: bool = False,
) -> dict[str, Any]:
    """Pack every ``*.pair.json`` under ``src_root`` and write the manifest."""
    records = load_records(src_root)
    # Two records sharing a scene_id would pack into the same directory, the
    # second silently replacing the first, while the manifest listed both and
    # pointed them at one place. Better to refuse than to ship a dataset that
    # quietly lost a scene.
    seen: set[str] = set()
    for record in records:
        scene_id = str(record.get("scene_id"))
        if scene_id in seen:
            raise ValueError(
                f"duplicate scene_id {scene_id!r} in {src_root}: packing would "
                f"overwrite the scene already written",
            )
        seen.add(scene_id)
    for record in records:
        pack(record, src_root, dst_root, copy=copy, strict=strict)
    mani = manifest(records)
    _write_json(Path(dst_root) / "manifest.json", mani)
    return mani
