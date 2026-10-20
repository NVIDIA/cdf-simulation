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

"""Execute capture JobOrders through reusable Isaac SDG APIs."""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import numbers
import os
import random
import re
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from pcb_aoi.execution.board_instances import (
    prepare_instance_semantic_capture,
    refresh_instanced_capture_stage,
)
from pcb_aoi.geometry import scan_grid as scan_grid_geometry

from ...errors import ValidationError
from ...image_formats import WRITER_NATIVE_IMAGE_FORMATS
from ..phase_timer import PhaseTimer
from . import guard_band
from .camera_calibration import orthographic_calibration

if TYPE_CHECKING:
    from .execution_context import IsaacExecutionContext
    from .job_order_parameters import CaptureRenderParameters


_REPO_ROOT = Path(__file__).resolve().parents[3]
_STANDALONE_DIR = _REPO_ROOT / "scripts" / "sdg" / "standalone"
_EDITOR_DIR = _REPO_ROOT / "scripts" / "sdg" / "editor"

_GRID_SUFFIX = re.compile(r"_x(?P<x>\d+)_y(?P<y>\d+)$")
_FRAME_SUFFIX = re.compile(r"_(?P<frame>\d+)$")
_WARMUP_MIN_STEPS = 5
# A cold RTX pipeline compile measured about 65 s; this only bounds a renderer
# that never starts delivering frames.
_WARMUP_TIMEOUT_S = 900.0


def _artifact_kind(path: Path) -> str:
    """Classify a Replicator artifact from its filename."""
    name = path.stem
    for suffix, kind in (
        ("semantic_segmentation_labels", "semantic_labels"),
        ("semantic_segmentation", "semantic_mask"),
        ("instance_id_segmentation", "instance_mask"),
        ("bounding_box_2d_tight_prim_paths", "bounding_box_2d_tight_paths"),
        ("bounding_box_2d_tight", "bounding_box_2d_tight"),
        ("bounding_box_2d_loose_prim_paths", "bounding_box_2d_loose_paths"),
        ("bounding_box_2d_loose", "bounding_box_2d_loose"),
        ("bounding_box_3d", "bounding_box_3d"),
        ("rgb", "rgb"),
    ):
        if name.startswith(suffix):
            return kind
    return name.split("_", 1)[0]


def _frame_calibration(
    stage: object,
    camera_path: str,
    resolution: Sequence[int],
    crop_px: int = 0,
) -> dict[str, object]:
    """Calibrate the capture camera for one frame without endangering the render.

    A frame that rendered is worth more than its calibration record, so a
    failure here is reported in the record and as a warning rather than raised.

    Args:
        stage: Open USD stage holding the capture camera.
        camera_path: Absolute prim path of the capture camera.
        resolution: Render resolution as ``(width, height)`` in pixels.
        crop_px: Guard band cropped from every side of the rendered frame; the
            calibration describes the delivered tile.

    Returns:
        The calibration mapping, or ``{"unavailable": reason}`` when it could
        not be computed.

    """
    try:
        return orthographic_calibration(
            stage,
            camera_path,
            resolution,
            crop_px=crop_px,
        )
    except Exception as exc:  # noqa: BLE001 - the render must not die for a sidecar
        warnings.warn(
            f"camera calibration unavailable for {camera_path!r}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def _write_pass_calibration(
    output_root: Path,
    pass_dir: str,
    frames: list[dict[str, object]],
    serializer: object,
    role: str,
    trigger_index: int,
) -> dict[str, object] | None:
    """Write the per-frame camera calibration of one pass to ``calibration.json``.

    The same records travel inside the dataset manifest, but a reader that only
    has the image folder should still be able to turn a pixel back into a board
    coordinate, so the calibration is also written beside the frames. It goes
    out through the manifest's own serializer and comes back as a
    :class:`~cdf.execution.dataset_manifest.ManifestArtifact`, because
    ``_discover_frame_artifacts`` only recognises frame-numbered filenames and
    would never find this one -- an unregistered file is invisible to
    ``ManifestView.artifacts()`` and therefore not part of the dataset.

    Args:
        output_root: Dataset output root that artifact paths are relative to.
        pass_dir: Directory the writer emitted this pass's frames into.
        frames: Canonical frame records carrying ``camera.calibration``.
        serializer: Manifest serializer that writes the sidecar.
        role: Capture role these frames belong to.
        trigger_index: Zero-based index of the trigger being captured.

    Returns:
        The artifact record describing the written file, or ``None`` when no
        frame had a calibration or the write failed.

    """
    entries = []
    for frame in frames:
        camera = frame.get("camera")
        calibration = camera.get("calibration") if isinstance(camera, Mapping) else None
        if not calibration:
            continue
        entries.append(
            {
                "frame": frame.get("frame"),
                "frame_id": frame.get("frame_id"),
                "role": frame.get("role"),
                **calibration,
            },
        )
    if not entries:
        return None
    path = Path(pass_dir) / "calibration.json"
    try:
        serializer.write_json(
            path,
            {
                "schema": "cdf.camera_calibration",
                "schema_version": "1.0.0",
                "frames": entries,
            },
        )
    except (OSError, TypeError, ValueError) as exc:
        # Same rule as _frame_calibration above: the frames are already on
        # disk and the same records travel in the manifest, so a sidecar that
        # cannot be written is a warning, not the end of the pass.
        warnings.warn(
            f"camera calibration sidecar unavailable for {pass_dir!r}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    relative = path.relative_to(output_root).as_posix()
    # Pass-level, not frame-level: one file covers every frame of the pass, so
    # it carries no frame_id. _validate_document allows that and only checks
    # frame_id when one is present.
    return {
        "artifact_id": (
            f"artifact:{role}:trigger:{trigger_index:04d}:camera_calibration"
        ),
        "kind": "camera_calibration",
        "path": relative,
        "role": role,
        "trigger_idx": trigger_index,
    }


def _discover_frame_artifacts(
    output_root: Path,
    pass_dir: Path,
    frame_records: list[dict[str, object]],
    frame_to_grid: list[dict[str, int]],
) -> list[dict[str, object]]:
    """Map files emitted by Replicator to canonical frame records."""
    grid_by_frame = {
        int(item["frame"]): (int(item["x_idx"]), int(item["y_idx"]))
        for item in frame_to_grid
    }
    artifacts = []
    for path in sorted(item for item in pass_dir.iterdir() if item.is_file()):
        if path.name in {"metadata.json", "dataset_index.json"}:
            continue
        stem = path.stem
        grid_match = _GRID_SUFFIX.search(stem)
        frame_match = _FRAME_SUFFIX.search(stem)
        frame_record = None
        for candidate in frame_records:
            frame_number = int(candidate["frame"])
            grid = grid_by_frame.get(frame_number)
            if grid_match and grid == (
                int(grid_match.group("x")),
                int(grid_match.group("y")),
            ):
                frame_record = candidate
                break
            if frame_match and frame_number == int(frame_match.group("frame")):
                frame_record = candidate
                break
        if frame_record is None:
            continue
        relative = path.relative_to(output_root).as_posix()
        kind = _artifact_kind(path)
        artifact = {
            "artifact_id": f"artifact:{frame_record['frame_id']}:{kind}:{path.suffix}",
            "frame_id": frame_record["frame_id"],
            "kind": kind,
            "path": relative,
            "role": frame_record.get("role", "capture"),
        }
        artifacts.append(artifact)
        frame_artifacts = frame_record.setdefault("artifacts", {})
        assert isinstance(frame_artifacts, dict)
        frame_artifacts.setdefault(kind, []).append(relative)
    return artifacts


def _require_frame_artifacts(
    frame_records: list[dict[str, object]],
    pass_dir: Path,
) -> None:
    """Fail when the writer left a rendered frame without any output file.

    Frame records are created when a frame is stepped, not when the writer
    writes it, so an empty artifact map is the only sign the writer skipped
    that frame. Without this check the pass reports the frame as captured.

    Args:
        frame_records: Frame records of one pass, after artifact discovery.
        pass_dir: Directory the pass writer wrote into.

    Raises:
        RuntimeError: If any frame has no discovered artifact.

    """
    unwritten = [
        str(frame["frame_id"]) for frame in frame_records if not frame["artifacts"]
    ]
    if unwritten:
        raise RuntimeError(
            f"writer produced no output for {len(unwritten)} frame(s) in "
            f"{pass_dir}: {', '.join(unwritten)}",
        )


def _convert_frame(source: Path, destination: Path, image_module: object) -> bool:
    """Re-encode one frame, dropping alpha; ``False`` while the PNG is incomplete.

    Only decoding the source is retried: a PNG the writer is still flushing
    fails to open or decodes as truncated, and the caller polls and comes back
    for it. Encoding and writing the destination are never retried -- a
    permission error or a full disk does not heal by waiting -- so those
    propagate and the capture fails instead of finishing without the frames
    it was asked for.
    """
    try:
        with image_module.open(source) as image:
            image.load()
            frame = image.convert("RGB")
    except (OSError, SyntaxError):
        return False
    frame.save(destination)
    source.unlink()
    return True


def convert_rgb_outputs(
    pass_dir: str | Path,
    target_format: str,
    *,
    frame_padding: int = 4,
    frames: Sequence[int] = (),
    flush_timeout_s: float = 15.0,
    poll_interval_s: float = 0.05,
) -> int:
    """Re-encode one pass's ``rgb_*.png`` frames as ``rgb_*.<target_format>``.

    BasicWriter encodes png and jpeg itself; every other accepted format is
    written natively as PNG and converted here, once the pass is flushed. Only
    rgb frames change: masks, bounding boxes and label sidecars keep the
    writer's own formats. Alpha is dropped, because the rgb annotator yields
    RGBA and formats such as BMP carry three channels.

    The writer flushes asynchronously, so, like ``rename_outputs_to_grid_index``,
    this polls: each expected frame's PNG must exist and decode completely
    before it is converted. Only the frames named in ``frames`` are touched;
    an rgb PNG left in the directory by an earlier run is not this capture's
    output and stays as it is. A frame that never surfaces complete within
    ``flush_timeout_s`` fails the capture, because the dataset would otherwise
    finish without an artifact it promised.

    Args:
        pass_dir: Directory the writer emitted this pass's frames into.
        target_format: Extension to produce, for example ``"bmp"``.
        frame_padding: Zero padding of the writer's frame numbers.
        frames: Frame numbers written this pass; conversion waits for these.
        flush_timeout_s: Longest wait for the writer to surface every frame.
        poll_interval_s: Pause between passes over the directory.

    Returns:
        The number of frames converted.

    Raises:
        RuntimeError: If an expected frame has not surfaced complete by the
            deadline.
        OSError: If a converted frame cannot be encoded or written.

    """
    from PIL import Image

    directory = Path(pass_dir)
    pending = {f"rgb_{int(frame):0{frame_padding}d}.png" for frame in frames}
    converted = 0
    deadline = time.monotonic() + max(0.0, float(flush_timeout_s))
    while True:
        for name in sorted(pending):
            source = directory / name
            if source.exists() and _convert_frame(
                source,
                source.with_suffix(f".{target_format}"),
                Image,
            ):
                converted += 1
                pending.discard(name)
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(max(0.001, float(poll_interval_s)))
    if pending:
        raise RuntimeError(
            f"convert_rgb_outputs: {len(pending)} rgb frame(s) never surfaced "
            f"complete in {directory} within {flush_timeout_s:.1f}s: "
            f"{sorted(pending)}",
        )
    return converted


def _refine_frame_defect_associations(
    output_root: Path,
    frame_records: list[dict[str, object]],
    defects_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    """Use emitted prim-path annotations to verify visible frame defects."""
    for frame in frame_records:
        frame["defect_association"] = "scene_active"
        frame_artifacts = frame.get("artifacts")
        if not isinstance(frame_artifacts, dict):
            continue
        path_files = frame_artifacts.get("bounding_box_2d_tight_paths")
        if not isinstance(path_files, list) or not path_files:
            continue
        visible_paths: set[str] = set()
        annotation_available = False
        for relative_path in path_files:
            try:
                values = json.loads(
                    (output_root / str(relative_path)).read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            if isinstance(values, list):
                annotation_available = True
                visible_paths.update(str(value) for value in values)
        if not annotation_available:
            continue
        verified_ids = []
        for defect_id in frame.get("defect_ids", []):
            defect = defects_by_id.get(str(defect_id))
            if defect is None:
                continue
            if defect.get("type") == "missing":
                verified_ids.append(defect_id)
                continue
            defect_path = str(defect.get("usd_path") or "")
            if any(
                visible == defect_path
                or visible.startswith(f"{defect_path}/")
                or defect_path.startswith(f"{visible}/")
                for visible in visible_paths
            ):
                verified_ids.append(defect_id)
        frame["defect_ids"] = verified_ids
        frame["defect_association"] = "bounding_box_2d_tight"


def _script_module(directory: Path, name: str) -> ModuleType:
    """Import an existing SDG module from a repository script directory.

    Args:
        directory: Directory containing the requested module.
        name: Top-level module name to import.

    Returns:
        Imported module.

    """
    path = str(directory)
    if path not in sys.path:
        sys.path.insert(0, path)
    importlib.invalidate_caches()
    return importlib.import_module(name)


def _required_state(
    context: IsaacExecutionContext,
    key: str,
) -> Mapping[str, object]:
    """Return one required configuration block from the execution context.

    Args:
        context: Prepared Isaac execution context.
        key: State key populated by a configuration handler.

    Returns:
        Stored configuration mapping.

    Raises:
        ValidationError: If the required configuration has not executed.

    """
    value = context.state.get(key)
    if not isinstance(value, Mapping):
        raise ValidationError(f"capture.render requires {key} first")
    return value


def _scanned_bounds_xy(scan_grid) -> tuple[float, float, float, float] | None:
    """World XY rectangle the cells of this grid actually cover.

    The union of the cell footprints, not the board: a capture that states a
    field of view scans less than the stage holds, and the difference is the
    whole point of stating one.

    Returns ``None`` when the grid was given as explicit camera positions with
    no footprint, which is the legacy form and carries nothing to intersect.
    """
    try:
        x_centers = [float(value) for value in scan_grid["x_centers"]]
        y_centers = [float(value) for value in scan_grid["y_centers"]]
        half_x = float(scan_grid["footprint_x"]) / 2.0
        half_y = float(scan_grid["footprint_y"]) / 2.0
    except (KeyError, TypeError, ValueError):
        return None
    if not x_centers or not y_centers or half_x <= 0 or half_y <= 0:
        return None
    return (
        min(x_centers) - half_x,
        min(y_centers) - half_y,
        max(x_centers) + half_x,
        max(y_centers) + half_y,
    )


def _board_instances_in_frame(context, stage=None, scan_grid=None) -> list[str] | None:
    """Prim paths of the board placements this capture covers.

    "Covers", not "exists": a partial field of view over a step-repeated panel
    images some of the placements and not others, and a sidecar that lists all
    of them tells a consumer the stitched image contains boards it does not.
    Each instance's world bounds are intersected with the scanned rectangle.

    Returns ``None`` rather than an empty list when there is no cohort, so a
    single-board scene is distinguishable from a panel whose instances could not
    be resolved -- an empty list would read as "we looked and found none".
    Falls back to the full cohort when the scan bounds or the stage are
    unavailable: listing every placement is the answer that was right before a
    field of view could be stated, and it is better than dropping the field.
    """
    cohort = getattr(context, "board_instance_cohort", None)
    paths = getattr(cohort, "instance_paths", None) if cohort is not None else None
    if not paths:
        return None
    all_paths = [str(path) for path in paths]
    bounds = _scanned_bounds_xy(scan_grid) if scan_grid is not None else None
    if stage is None or bounds is None:
        return all_paths

    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=True,
    )
    min_x, min_y, max_x, max_y = bounds
    covered = []
    for path in all_paths:
        instance_bounds = _combined_prim_bounds(stage, cache, (path,))
        if instance_bounds is None:
            # A placement we cannot measure is reported rather than dropped:
            # the alternative silently shrinks the list on a stage whose
            # extents are missing.
            covered.append(path)
            continue
        instance_min, instance_max = instance_bounds
        if (
            instance_min[0] <= max_x
            and instance_max[0] >= min_x
            and instance_min[1] <= max_y
            and instance_max[1] >= min_y
        ):
            covered.append(path)
    return covered


def _stage_provenance(stage) -> dict | None:
    """Creator and timestamp recorded in the root layer, when it has them.

    ``source_usd`` is a filesystem path, which is machine-specific and says
    nothing about which export produced the file. USD carries ``creator`` and a
    modification time on the layer; both are optional, so anything missing is
    simply absent from the mapping rather than reported as null.
    """
    try:
        layer = stage.GetRootLayer()
    except Exception:  # noqa: BLE001 - provenance is never worth failing a capture
        return None
    out: dict[str, object] = {}
    try:
        creator = (layer.customLayerData or {}).get("creator")
        if creator:
            out["creator"] = str(creator)
    except Exception:  # noqa: BLE001
        creator = None
    for attr, key in (("comment", "comment"), ("version", "version")):
        try:
            value = getattr(layer, attr, None)
            if value:
                out[key] = str(value)
        except Exception:  # noqa: BLE001
            value = None
    try:
        identifier = layer.realPath or layer.identifier
        if identifier:
            import os as _os

            if _os.path.exists(identifier):
                import datetime as _dt

                out["modified_utc"] = (
                    _dt.datetime.fromtimestamp(
                        _os.path.getmtime(identifier),
                        tz=_dt.UTC,
                    )
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
    except Exception:  # noqa: BLE001
        return out or None
    return out or None


async def _refresh_instanced_capture_stage(
    context: IsaacExecutionContext,
) -> object:
    """Delegate instance-aware stage refresh through the board-instance module."""
    common = _script_module(_STANDALONE_DIR, "common")
    return await refresh_instanced_capture_stage(
        context,
        common.open_usd_stage,
    )


def build_capture_config(
    context: IsaacExecutionContext,
    output_dir: str,
) -> dict[str, object]:
    """Build the runtime configuration from typed JobOrder state.

    This mapping is internal to the handler and is never serialized or treated
    as a public schema. It adapts semantic JobOrder contracts to the existing
    lower-level SDG functions one call at a time.

    Args:
        context: Prepared Isaac execution context.
        output_dir: Directory that receives capture artifacts.

    Returns:
        Runtime values consumed by existing SDG capability functions.

    Raises:
        ValidationError: If required capture or scene intent is missing.

    Example:

    .. code-block:: python

        config = build_capture_config(context, "/tmp/capture")  # doctest: +SKIP
        print(config["output"])  # doctest: +SKIP

    """
    renderer = _required_state(context, "capture.renderer")
    camera = _required_state(context, "capture.camera")
    scan_grid = _required_state(context, "capture.scan_grid")
    writer = _required_state(context, "capture.outputs")
    limits = context.state.get("capture.limits") or {
        "triggers": 1,
        "max_images": -1,
    }
    output = context.state.get("capture.output") or {"start": 0}
    lighting = context.state.get("appearance.lighting") or {
        "mode": "usd_light",
        "preserve_scene_light_color": False,
    }
    augmentation = context.state.get("capture.augmentation") or {}
    visibility = context.state.get("capture.visibility") or {
        "hide": (),
        "show": (),
    }
    roi_selection = context.state.get("capture.roi_selection")

    scene = context.scene
    root_path = scene.get("root_path")
    if not isinstance(root_path, str):
        raise ValidationError("capture.render requires scene.load first")

    camera_path = scene.get("camera_path") or "/World/camera_light/Camera"
    camera_xform_path = scene.get("camera_xform_path") or "/World/camera_light"
    ring_light_root = (
        scene.get("ring_light_root") or "/World/camera_light/aoi_ring_light"
    )
    rotation = dict(camera["rotation_range"])

    motion_blur = augmentation.get("motion_blur")
    if motion_blur is None:
        motion_blur = {
            "probability": 0.0,
            "alpha_range": (0.0, 0.0),
            "kernel_choices": (3,),
        }

    writer_config = dict(writer)
    rename_to_grid_index = bool(
        writer_config.pop("rename_to_grid_index", False),
    )
    manifest_visibility_annotations = any(
        context.state.get(key)
        for key in (
            "operation.structural.records",
            "operation.solder_defects.records",
            "operation.missing.paths",
        )
    )
    if manifest_visibility_annotations:
        writer_config["bounding_box_2d_tight"] = True
    # The writer only encodes png/jpeg itself (anything else dies silently in
    # its io thread), so a converted format renders as PNG and is re-encoded
    # by convert_rgb_outputs once the pass has flushed. With rgb disabled
    # there is nothing to re-encode: conversion would only wait out its
    # deadline for frames nobody asked for.
    image_format = str(writer_config.pop("image_format"))
    writer_native = image_format in WRITER_NATIVE_IMAGE_FORMATS
    convert_rgb_to = (
        image_format
        if not writer_native and bool(writer_config.get("rgb", True))
        else None
    )
    writer_config["image_output_format"] = image_format if writer_native else "png"
    semantic_projection = None
    # The reopened stage contains ordinary composed descendants with their
    # authored semantics, so native capture avoids lossy colorized instance IDs.
    cohort = context.board_instance_cohort
    stage = context.stage
    deinstanced_capture = bool(
        cohort is not None
        and stage is not None
        and all(
            not stage.GetPrimAtPath(path).IsInstance() for path in cohort.instance_paths
        )
    )
    if not deinstanced_capture:
        semantic_projection = prepare_instance_semantic_capture(
            context.instance_semantic_projection,
            writer_config,
            structural_records=context.state.get("operation.structural.records") or (),
            missing_paths=context.state.get("operation.missing.paths") or (),
        )

    config = {
        "output": str(Path(output_dir).expanduser().resolve()),
        "pcba_root": root_path,
        "camera_path": str(camera_path),
        "camera_xform_path": str(camera_xform_path),
        "ring_light_root": str(ring_light_root),
        "modus_light_root": str(
            scene.get("modus_light_root") or "/World/_aoi_modus_light"
        ),
        "modus_bounds_paths": _modus_bounds_paths(context),
        "horizontal_aperture": scene.get("horizontal_aperture"),
        "component_types": list(context.component_types),
        "resolution": tuple(renderer["resolution"]),
        "render_mode": renderer["mode"],
        "realtime": {"subframes": renderer.get("realtime_subframes", 1)},
        "pathtracing": {
            "spp": renderer["spp"],
            "total_spp": renderer["total_spp"],
            "max_bounces": renderer["max_bounces"],
        },
        "scan_grid": {
            "x_num": scan_grid["x_count"],
            "y_num": scan_grid["y_count"],
        },
        "scan_area_mm": scan_grid.get("scan_area_mm"),
        "scan_area_centered_on": scan_grid.get("centered_on"),
        "scan_center_mm": scan_grid.get("scan_center_mm"),
        "scan_image_size_px": scan_grid.get("image_size_px"),
        "guard_px": int(scan_grid.get("guard_px") or 0),
        "meters_per_unit_override": scene.get("meters_per_unit"),
        "camera_side": camera.get("side", "top"),
        "camera_rotation": {
            "x_range": _axis_range(rotation.get("x", 0.0)),
            "y_range": _axis_range(rotation.get("y", 0.0)),
            "z_fixed": _fixed_axis(rotation.get("z", 0.0)),
        },
        "num_triggers": int(limits["triggers"]),
        "max_image_count": int(limits["max_images"]),
        "image_seed": int(output["start"]),
        "random_seed": int(context.plan_metadata.get("random_seed", 0)),
        "lighting": _lighting_config(lighting),
        "vantablack_components": bool(
            (context.state.get("appearance.components") or {}).get(
                "vantablack",
                False,
            ),
        ),
        "augmentation": {"motion_blur": dict(motion_blur)},
        "rename_to_grid_index": rename_to_grid_index,
        "convert_rgb_to": convert_rgb_to,
        "hidden_geometry": list(visibility.get("hide") or ()),
        "shown_geometry": list(visibility.get("show") or ()),
        "writer": writer_config,
        "manifest_visibility_annotations": manifest_visibility_annotations,
    }
    if semantic_projection is not None:
        config["semantic_projection"] = semantic_projection
    # The guard band is cropped from every per-tile output; refuse an output
    # that cannot be, rather than write one describing a frame never delivered.
    if config["guard_px"]:
        refused = guard_band.unsupported_outputs(writer_config)
        if refused:
            raise ValidationError(
                f"capture.scan_grid guard_px={config['guard_px']} cannot crop "
                f"{', '.join(refused)}; render them with guard_px 0 (fov guard_px: 0)"
            )
    config["render_resolution"] = guard_band.render_resolution(
        config["resolution"],
        config["guard_px"],
    )
    if isinstance(roi_selection, Mapping):
        config["roi_selection"] = dict(roi_selection)
    return config


def _axis_range(value: object) -> tuple[float, float]:
    """Normalize a fixed or ranged camera axis.

    Args:
        value: Fixed numeric value or two-item range.

    Returns:
        Inclusive numeric range.

    Raises:
        ValidationError: If the value is not numeric or a two-item range.

    """
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric, numeric
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError) as exc:
            raise ValidationError("camera axis range must contain numbers") from exc
    raise ValidationError("camera axis must be a number or two-item range")


def _fixed_axis(value: object) -> float:
    """Normalize a fixed camera axis.

    Args:
        value: Fixed numeric value.

    Returns:
        Numeric fixed-axis value.

    Raises:
        ValidationError: If the value is not numeric.

    """
    if not isinstance(value, (int, float)):
        raise ValidationError("camera z axis must be a fixed number")
    return float(value)


def _lighting_config(lighting: Mapping[str, object]) -> dict[str, object]:
    """Normalize optional lighting values for existing SDG helpers.

    Args:
        lighting: Validated lighting JobOrder state.

    Returns:
        Lighting mapping with defaults required by ``common``.

    """
    # ``LightingParameters`` is a fixed-shape model, so model_dump() emits every
    # optional rig payload -- `layers`, `white_light`, `modus` -- as None for the
    # rigs this job did not select. `randomize_lighting` tests membership rather
    # than truth ("white_light" not in light_cfg), so a None-valued key would
    # pass the guard and then fail on subscript. Drop them here instead.
    config = {key: value for key, value in lighting.items() if value is not None}
    config.setdefault("exposure_range", (0.0, 0.0))
    config.setdefault("cone_angle_range", (120.0, 120.0))
    config.setdefault("cone_softness_range", (1.0, 1.0))
    return config


def _ensure_camera(
    stage: object,
    config: dict[str, object],
) -> object | None:
    """Create or configure the orthographic camera used by capture.

    Args:
        stage: Open USD stage.
        config: Internal capture configuration.

    Returns:
        Anonymous authored rig layer when one was required, otherwise None.

    """
    from pxr import Sdf, Usd, UsdGeom

    camera_path = config["camera_path"]
    camera_xform_path = config["camera_xform_path"]
    rig_layer = None
    camera = UsdGeom.Camera.Get(stage, camera_path)
    if not camera.GetPrim().IsValid():
        rig_layer = Sdf.Layer.CreateAnonymous("cdf_job_camera_rig.usda")
        stage.GetRootLayer().subLayerPaths.insert(0, rig_layer.identifier)
        with Usd.EditContext(stage, Usd.EditTarget(rig_layer)):
            xform = UsdGeom.Xform.Define(stage, camera_xform_path)
            if not any(
                op.GetOpName() == "xformOp:translate"
                for op in xform.GetOrderedXformOps()
            ):
                xform.AddTranslateOp()
            camera = UsdGeom.Camera.Define(stage, camera_path)

    aperture = config.get("horizontal_aperture")
    if aperture is not None:
        camera.GetHorizontalApertureAttr().Set(float(aperture))
    camera.GetProjectionAttr().Set("orthographic")
    return rig_layer


def _set_roi_apertures(
    camera_schema: object,
    roi_spec: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    """Point the camera at one focused ROI, widened by the pass's guard band.

    The recorded ROI apertures describe the delivered frame; the camera renders
    ``guard_px`` more pixels on every side at the same pitch.

    Args:
        camera_schema: ``UsdGeom.Camera`` of the capture camera.
        roi_spec: Focused ROI with delivered ``horizontal_aperture`` and
            ``vertical_aperture`` in raw USD units.
        config: Internal capture configuration.

    """
    guard = int(config.get("guard_px") or 0)
    width, height = (int(value) for value in config["resolution"])
    camera_schema.GetHorizontalApertureAttr().Set(
        guard_band.render_aperture(roi_spec["horizontal_aperture"], width, guard),
    )
    camera_schema.GetVerticalApertureAttr().Set(
        guard_band.render_aperture(roi_spec["vertical_aperture"], height, guard),
    )


#: Context-state key mapping each defect-moved ROI target to its world bounds
#: where the board places it, as ``(min_xyz, max_xyz)``.
_PLACED_BOUNDS_KEY = "operation.structural.placed_bounds"


def _record_placed_bounds(
    context: IsaacExecutionContext,
    stage: object,
    records: Sequence[Mapping[str, object]],
) -> None:
    """Remember where each defect-moved component sits before its defect.

    A focused ROI crop frames a component where the board places it, as an
    AOI does: framed from the defective pose, a tombstone crop zooms onto the
    standing part and leaves its bare pad out. Bounds cannot be recovered from
    ``original_flat`` after the fact, because the bounding-box cache aligns
    every bound to world axes in the CURRENT pose, so a rotated part's bound
    stays inflated however it is re-posed. Call this before the defect
    transforms are authored onto the stage.

    Args:
        context: Mutable execution context receiving the placed bounds.
        stage: Live USD stage the defects were applied to.
        records: Applied-defect records from ``defect_ops.prepare_defects``,
            each carrying the component path and its ``original_flat``
            row-major local transform.

    Raises:
        ValidationError: If a record lacks its placed transform, or a placed
            component has no bounds.

    """
    if not records:
        return
    from pxr import Gf, Usd, UsdGeom

    placed = []
    for record in records:
        prim = stage.GetPrimAtPath(str(record["path"]))
        if not prim or not prim.IsValid():
            continue
        original = record.get("original_flat")
        if not isinstance(original, (list, tuple)) or len(original) != 16:
            raise ValidationError(
                f"defect record for {record['path']!r} has no placed transform",
            )
        placed.append((prim, Gf.Matrix4d(*(float(value) for value in original))))
    for prim, matrix in placed:
        xformable = UsdGeom.Xformable(prim)
        # pose_ops moves components in Fabric only, so the stage still holds
        # the placed pose; without Replicator, defect_ops writes the stage
        # itself and the placed pose has to be put back before measuring.
        if not Gf.IsClose(xformable.GetLocalTransformation(), matrix, 1e-9):
            xformable.MakeMatrixXform().Set(matrix)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=True,
    )
    cohort = context.board_instance_cohort
    placed_bounds = context.state.setdefault(_PLACED_BOUNDS_KEY, {})
    for prim, _matrix in placed:
        source_path = str(prim.GetPath())
        target_paths = (
            tuple(
                target.composed_path
                for target in cohort.expand_source_path(source_path)
            )
            if cohort is not None
            else (source_path,)
        )
        for target_path in target_paths:
            bounds = _combined_prim_bounds(stage, cache, (target_path,))
            if bounds is None:
                raise ValidationError(
                    f"defect-moved component {target_path!r} has no placed bounds",
                )
            placed_bounds.setdefault(target_path, bounds)


def _roi_selection_specs(
    context: IsaacExecutionContext,
    stage: object,
    config: Mapping[str, object],
    camera_z: float,
) -> tuple[dict[str, object], ...]:
    """Resolve component and pin targets into camera positions and apertures."""
    request = config.get("roi_selection")
    if not isinstance(request, Mapping):
        return ()
    from pxr import Usd, UsdGeom

    source = str(context.state.get("board_metadata.source") or "usd")
    catalog = context.board_metadata.get(source)
    if catalog is None:
        raise ValidationError("focused ROI capture requires board metadata")
    root_path = str(config["pcba_root"])
    metadata_root_path = str(context.scene.get("board_root_path") or root_path)
    resolution = tuple(config["resolution"])
    aspect = float(resolution[1]) / float(resolution[0])
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=True,
    )
    selections = request.get("selections")
    if not isinstance(selections, (list, tuple)) or not selections:
        raise ValidationError("ROI selection requires at least one selector")

    components_by_path = {
        component.prim_path: component
        for component in catalog.components
        if component.board_root == metadata_root_path
    }
    selected = _resolve_roi_selection(
        context,
        catalog,
        metadata_root_path,
        selections,
        components_by_path,
        random_seed=int(config.get("random_seed", 0)),
    )

    specs = []
    cohort = context.board_instance_cohort
    placed_bounds = context.state.get(_PLACED_BOUNDS_KEY) or {}
    for component, selector_index, selector_type, selector in selected:
        refdes = component.refdes
        target_paths = (
            cohort.expand_source_path(component.prim_path)
            if cohort is not None
            else (component.prim_path,)
        )
        for target in target_paths:
            component_path = target.composed_path if cohort is not None else str(target)
            placed = placed_bounds.get(component_path)
            if placed is not None:
                # A defect moved this part: frame it where the board places it,
                # so the crop matches the clean one and keeps any bare pad.
                low, high = placed
                center = tuple((low[axis] + high[axis]) * 0.5 for axis in range(3))
                size = tuple(high[axis] - low[axis] for axis in range(3))
            else:
                prim = stage.GetPrimAtPath(component_path)
                bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if bounds.IsEmpty():
                    raise ValidationError(
                        f"component ROI target {component_path!r} has no bounds",
                    )
                center = bounds.GetMidpoint()
                size = bounds.GetSize()
            width, height = _roi_apertures(
                float(size[0]),
                float(size[1]),
                padding=float(selector.get("padding", 0.5)),
                aspect=aspect,
                camera_z_rotation=float(config["camera_rotation"]["z_fixed"]),
            )
            if width <= 0.0:
                raise ValidationError(
                    f"component ROI target {component_path!r} has no planar extent",
                )
            spec = {
                "refdes": str(refdes),
                "component_path": component_path,
                **_component_roi_metadata(component),
                "selector_index": selector_index,
                "selector_type": selector_type,
                "position": (float(center[0]), float(center[1]), camera_z),
                "target": (float(center[0]), float(center[1]), float(center[2])),
                "horizontal_aperture": width * 10.0,
                "vertical_aperture": height * 10.0,
            }
            if cohort is not None:
                spec.update(
                    {
                        "instance_index": target.instance_index,
                        "instance_path": target.instance_path,
                        "source_component_path": component.prim_path,
                    },
                )
            specs.append(spec)
    specs.extend(
        _pin_roi_selection_specs(
            context,
            stage,
            catalog,
            metadata_root_path,
            selections,
            components_by_path,
            cache,
            camera_z,
            aspect=aspect,
            camera_side=str(config.get("camera_side") or "top"),
            camera_z_rotation=float(config["camera_rotation"]["z_fixed"]),
            random_seed=int(config.get("random_seed", 0)),
        ),
    )
    specs.sort(key=lambda spec: int(spec["selector_index"]))
    return tuple(specs)


def _pin_roi_selection_specs(
    context: IsaacExecutionContext,
    stage: object,
    catalog: object,
    root_path: str,
    selections: list[object] | tuple[object, ...],
    components_by_path: Mapping[str, object],
    cache: object,
    camera_z: float,
    *,
    aspect: float,
    camera_side: str,
    camera_z_rotation: float,
    random_seed: int,
) -> tuple[dict[str, object], ...]:
    """Resolve pin identities into pin, pad, and solder-joint camera bounds."""
    cohort = context.board_instance_cohort
    resolved = _resolve_pin_roi_selection(
        context,
        catalog,
        root_path,
        selections,
        components_by_path,
        random_seed=random_seed,
    )
    preflight_warnings = _pin_roi_preflight_warnings(
        catalog,
        resolved,
        camera_side,
    )
    context.state["capture.roi_selection.warnings"] = preflight_warnings
    for message in preflight_warnings:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    manifest = context.state.get("solder_joints.placement_manifest")
    generated_paths = tuple(getattr(manifest, "generated_prim_paths", ()) or ())
    specs = []
    for component, terminal, selector_index, selector in resolved:
        terminal_targets = (
            cohort.expand_source_path(terminal.prim_path)
            if cohort is not None
            else (terminal.prim_path,)
        )
        matching_pads = _terminal_pads(
            catalog,
            component,
            terminal,
            camera_side,
        )
        for terminal_target in terminal_targets:
            instance_index = (
                terminal_target.instance_index if cohort is not None else None
            )
            terminal_path = (
                terminal_target.composed_path
                if cohort is not None
                else str(terminal_target)
            )
            candidate_pad_paths = tuple(
                path
                for pad in matching_pads
                for path in _composed_source_paths(
                    cohort,
                    pad.prim_path,
                    instance_index,
                )
            )
            nearest_pad_path = _nearest_prim_path(
                stage,
                cache,
                candidate_pad_paths,
                terminal_path,
            )
            pad_paths = (nearest_pad_path,) if nearest_pad_path is not None else ()
            association_paths = pad_paths or (terminal_path,)
            association_bounds = _combined_prim_bounds(
                stage,
                cache,
                association_paths,
            )
            if association_bounds is None:
                raise ValidationError(
                    f"pin ROI target {terminal_path!r} has no bounds",
                )

            solder_path = _nearest_generated_solder_path(
                stage,
                cache,
                generated_paths,
                cohort,
                instance_index,
                association_bounds,
            )
            focus_paths = pad_paths or (
                (solder_path,) if solder_path is not None else (terminal_path,)
            )
            focus_bounds = _combined_prim_bounds(stage, cache, focus_paths)
            assert focus_bounds is not None

            minimum, maximum = focus_bounds
            center = tuple(
                (minimum[index] + maximum[index]) * 0.5 for index in range(3)
            )
            size = tuple(maximum[index] - minimum[index] for index in range(3))
            width, height = _roi_apertures(
                size[0],
                size[1],
                padding=float(selector.get("padding", 0.5)),
                aspect=aspect,
                camera_z_rotation=camera_z_rotation,
            )
            if width <= 0.0:
                raise ValidationError(
                    f"pin ROI target {terminal_path!r} has no planar extent",
                )
            spec = {
                "refdes": component.refdes,
                **_component_roi_metadata(component),
                "pin_number": terminal.number,
                "pin_path": terminal_path,
                "pad_paths": pad_paths,
                "solder_joint_path": solder_path,
                "component_path": (
                    _composed_source_paths(
                        cohort,
                        component.prim_path,
                        instance_index,
                    )[0]
                ),
                "selector_index": selector_index,
                "selector_type": "pins",
                "position": (center[0], center[1], camera_z),
                "target": tuple(float(value) for value in center),
                "horizontal_aperture": width * 10.0,
                "vertical_aperture": height * 10.0,
            }
            if cohort is not None:
                spec.update(
                    {
                        "instance_index": terminal_target.instance_index,
                        "instance_path": terminal_target.instance_path,
                        "source_component_path": component.prim_path,
                        "source_pin_path": terminal.prim_path,
                    },
                )
            specs.append(spec)
    return tuple(specs)


def _component_roi_metadata(component: object) -> dict[str, object]:
    """Return source-neutral component metadata for a focused ROI record."""
    process_classes = getattr(component, "process_classes", ()) or ()
    return {
        "component_type": getattr(component, "component_type", None),
        "package_name": getattr(component, "package_name", None),
        "package_type": getattr(component, "package_type", None),
        "footprint_id": getattr(component, "footprint_id", None),
        "side": getattr(component, "side", None),
        "process_class": getattr(component, "process_class", None),
        "process_classes": list(process_classes),
    }


def _resolve_pin_roi_selection(
    context: IsaacExecutionContext,
    catalog: object,
    root_path: str,
    selections: list[object] | tuple[object, ...],
    components_by_path: Mapping[str, object],
    *,
    random_seed: int,
) -> tuple[tuple[object, object, int, Mapping[str, object]], ...]:
    """Resolve and deduplicate exact or component-expanded pin selectors."""
    resolved = []
    selected_paths = set()
    for selector_index, selector in enumerate(selections):
        if not isinstance(selector, Mapping):
            raise ValidationError("ROI selection entries must be mappings")
        pins = selector.get("pins")
        if isinstance(pins, (list, tuple)):
            targets = _resolve_explicit_pin_targets(
                catalog,
                root_path,
                pins,
                selector_index,
                selector,
            )
            parent = context.component_parent_paths()
            outside_parent = (
                []
                if parent is None
                else [
                    component.refdes
                    for component, _terminal, _selector_index, _selector in targets
                    if component.prim_path not in parent
                ]
            )
            if outside_parent:
                raise ValidationError(
                    "ROI pin targets are outside the scene component filter: "
                    + ", ".join(repr(refdes) for refdes in outside_parent),
                )
        elif selector.get("focus") == "pin":
            selector_type, candidates = _roi_selector_candidates(
                context,
                catalog,
                root_path,
                selector,
                components_by_path,
            )
            if not candidates:
                raise ValidationError(
                    f"ROI {selector_type} selector found no eligible components",
                )
            candidates = _sample_roi_component_candidates(
                candidates,
                selector_type,
                selector_index,
                selector,
                selected_paths=set(),
                random_seed=random_seed,
            )
            targets = tuple(
                (component, terminal, selector_index, selector)
                for component in candidates
                for terminal in _pin_selector_terminals(
                    catalog,
                    component,
                    selector_type,
                    selector,
                )
            )
            if not targets:
                raise ValidationError(
                    f"ROI {selector_type} selector found no cataloged pins",
                )
        else:
            continue
        for target in targets:
            terminal = target[1]
            if terminal.prim_path in selected_paths:
                continue
            resolved.append(target)
            selected_paths.add(terminal.prim_path)
    return tuple(resolved)


def _pin_roi_preflight_warnings(
    catalog: object,
    resolved: tuple[tuple[object, object, int, Mapping[str, object]], ...],
    camera_side: str,
) -> tuple[str, ...]:
    """Report duplicate logical pin numbers before geometric disambiguation."""
    selected_by_component: dict[str, dict[str, tuple[object, object]]] = {}
    for component, terminal, *_ in resolved:
        if not terminal.number:
            continue
        selected_by_component.setdefault(component.prim_path, {})[
            terminal.prim_path
        ] = (component, terminal)

    messages = []
    for component_path, selected in selected_by_component.items():
        component = next(iter(selected.values()))[0]
        for pin_number, duplicate_terminals in catalog.duplicate_terminal_number_groups(
            component_path,
        ):
            targets = tuple(
                selected[terminal.prim_path]
                for terminal in duplicate_terminals
                if terminal.prim_path in selected
            )
            if len(targets) < 2:
                continue
            candidate_counts = tuple(
                len(_terminal_pads(catalog, component, terminal, camera_side))
                for component, terminal in targets
            )
            requires_geometry = any(count > 1 for count in candidate_counts)
            prim_paths = ", ".join(terminal.prim_path for _, terminal in targets)
            if requires_geometry:
                resolution = (
                    "pad associations are ambiguous; CDF will perform a "
                    "world-space geometric scan"
                )
            elif all(count == 1 for count in candidate_counts):
                resolution = (
                    "unique contact-to-pad associations avoid geometric disambiguation"
                )
            else:
                resolution = (
                    "no ambiguous pad candidate set requires a geometric scan; "
                    "missing associations fall back to pin geometry"
                )
            messages.append(
                f"component {component.refdes!r} has {len(targets)} pin prims with "
                f"pcba:pin:number {pin_number!r} ({prim_paths}); {resolution}",
            )
    return tuple(messages)


def _resolve_explicit_pin_targets(
    catalog: object,
    root_path: str,
    pins: list[object] | tuple[object, ...],
    selector_index: int,
    selector: Mapping[str, object],
) -> tuple[tuple[object, object, int, Mapping[str, object]], ...]:
    """Resolve explicit component-pin identities without deduplication."""
    resolved = []
    for identity in pins:
        if not isinstance(identity, (list, tuple)) or len(identity) != 2:
            raise ValidationError("ROI pins must contain (refdes, pin_number) pairs")
        refdes, pin_number = (str(value) for value in identity)
        component = catalog.component(board_root=root_path, refdes=refdes)
        if component is None:
            raise ValidationError(f"pin ROI component {refdes!r} was not found")
        terminals = tuple(
            terminal
            for terminal in catalog.terminals_for_component(component)
            if terminal.number == pin_number
        )
        if not terminals:
            raise ValidationError(f"pin ROI target {refdes}.{pin_number} was not found")
        resolved.extend(
            (component, terminal, selector_index, selector) for terminal in terminals
        )
    return tuple(resolved)


def _pin_selector_terminals(
    catalog: object,
    component: object,
    selector_type: str,
    selector: Mapping[str, object],
) -> tuple[object, ...]:
    """Return cataloged terminals eligible for one pin-focused selector."""
    terminals = tuple(catalog.terminals_for_component(component))
    if selector_type != "process_classes":
        return terminals

    requested = {
        str(value).casefold() for value in selector.get("process_classes") or ()
    }
    classified = tuple(
        terminal
        for terminal in terminals
        if terminal.process_class
        and str(terminal.process_class).casefold() in requested
    )
    if classified:
        return classified

    process_classes = {
        str(value).casefold()
        for value in getattr(component, "process_classes", ())
        if value
    }
    aggregate = str(getattr(component, "process_class", "") or "").casefold()
    unambiguous = aggregate not in {"", "mixed"} or len(process_classes) == 1
    component_classes = process_classes | ({aggregate} if aggregate else set())
    return terminals if unambiguous and component_classes & requested else ()


def _terminal_pads(
    catalog: object,
    component: object,
    terminal: object,
    camera_side: str,
) -> tuple[object, ...]:
    """Return camera-side pads associated with one physical terminal."""
    pads = tuple(
        pad
        for pad in catalog.pads_for_component(component)
        if pad.side is None or str(pad.side).casefold() == camera_side.casefold()
    )
    direct = tuple(pad for pad in pads if pad.contact_prim_path == terminal.prim_path)
    if direct:
        return direct
    return tuple(pad for pad in pads if pad.terminal_number == terminal.number)


def _composed_source_paths(
    cohort: object | None,
    source_path: str,
    instance_index: int | None,
) -> tuple[str, ...]:
    """Return one source path in standalone or selected-instance space."""
    if cohort is None:
        return (source_path,)
    return tuple(
        target.composed_path
        for target in cohort.expand_source_path(source_path)
        if target.instance_index == instance_index
    )


def _combined_prim_bounds(
    stage: object,
    cache: object,
    prim_paths: tuple[str, ...],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return aligned world bounds spanning valid prim paths."""
    ranges = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not bounds.IsEmpty():
            ranges.append(bounds)
    if not ranges:
        return None
    return (
        tuple(
            min(float(item.GetMin()[index]) for item in ranges) for index in range(3)
        ),
        tuple(
            max(float(item.GetMax()[index]) for item in ranges) for index in range(3)
        ),
    )


def _capture_reference_plane_z(
    context: IsaacExecutionContext,
    stage: object,
    scan_grid: Mapping[str, object],
    *,
    top_side: bool,
) -> tuple[float | None, str | None]:
    """Return the imaged face's world Z and what it was measured against.

    Prefers the metadata catalog's measured ``board.substrate_z_range``, which
    is the PCB surface itself. Without it, prefers the world bounds of one
    representative board instance (the
    board itself, e.g. one step-repeat placement on a panel) over the
    capture root's bounds, so a root scoped to a carrier/panel fixture
    -- wider than the board it holds -- does not leak fixture thickness
    into the reported reference plane. Falls back to the capture root's
    bounds (already computed for the scan grid) when there is no board-
    instance cohort, e.g. a single-board scene.

    Args:
        context: Prepared Isaac execution context.
        stage: Open USD stage.
        scan_grid: Resolved ``config["scan_grid"]``; may carry ``bbox_min``/
            ``bbox_max`` from :func:`common.resolve_scan_grid_from_stage`.
        top_side: Whether the imaged face is the bbox's top (``True``) or
            bottom (``False``) in world Z.

    Returns:
        ``(world_z, source)`` where ``source`` is ``"substrate_z_range"``,
        ``"board_instance"`` or ``"capture_root"``, or ``(None, None)`` when no
        source is available.

    """
    # Best source first: the metadata catalog measured the SUBSTRATE, so its
    # z range is the PCB surface. Every bbox route below is an axis-aligned box
    # over whatever geometry the scope contains, which means the "board plane"
    # can land on the tallest component or the longest protruding pin instead.
    metadata = getattr(context, "board_metadata", None)
    board = getattr(metadata, "board", None) if metadata is not None else None
    z_range = getattr(board, "substrate_z_range", None) if board is not None else None
    # Structural check, not truthiness: a test double or a partially populated
    # catalog can put any object on this attribute, and a wrong reference plane
    # is silent -- the sidecar simply places the image at the wrong height.
    if (
        isinstance(z_range, (tuple, list))
        and len(z_range) == 2
        and all(
            isinstance(v, numbers.Real) and not isinstance(v, bool) for v in z_range
        )
    ):
        lo, hi = float(z_range[0]), float(z_range[1])
        return (max(lo, hi) if top_side else min(lo, hi)), "substrate_z_range"

    cohort = context.board_instance_cohort
    if cohort is not None:
        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            ["default", "render", "proxy"],
            useExtentsHint=True,
        )
        instance_bounds = _combined_prim_bounds(
            stage,
            cache,
            (cohort.representative_instance_path,),
        )
        if instance_bounds is not None:
            minimum, maximum = instance_bounds
            return (maximum[2] if top_side else minimum[2]), "board_instance"

    bbox_bound = scan_grid.get("bbox_max" if top_side else "bbox_min")
    if isinstance(bbox_bound, (list, tuple)) and len(bbox_bound) == 3:
        return float(bbox_bound[2]), "capture_root"
    return None, None


def _nearest_generated_solder_path(
    stage: object,
    cache: object,
    generated_paths: tuple[str, ...],
    cohort: object | None,
    instance_index: int | None,
    focus_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> str | None:
    """Return the generated solder mesh nearest one pin-and-pad center."""
    from pxr import UsdGeom

    focus_center = tuple(
        (focus_bounds[0][index] + focus_bounds[1][index]) * 0.5 for index in range(2)
    )
    focus_extent = max(
        focus_bounds[1][index] - focus_bounds[0][index] for index in range(2)
    )
    maximum_distance = max(focus_extent * 2.0, 1e-6) ** 2
    candidates = []
    for source_path in generated_paths:
        try:
            composed_paths = _composed_source_paths(
                cohort,
                source_path,
                instance_index,
            )
        except ValidationError:
            composed_paths = (source_path,) if cohort is None else ()
        for prim_path in composed_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
                continue
            bounds = _combined_prim_bounds(stage, cache, (prim_path,))
            if bounds is None:
                continue
            center = tuple(
                (bounds[0][index] + bounds[1][index]) * 0.5 for index in range(2)
            )
            distance = sum(
                (center[index] - focus_center[index]) ** 2 for index in range(2)
            )
            if distance <= maximum_distance:
                candidates.append((distance, prim_path))
    return min(candidates)[1] if candidates else None


def _nearest_prim_path(
    stage: object,
    cache: object,
    candidates: tuple[str, ...],
    target_path: str,
) -> str | None:
    """Return the candidate prim nearest a target prim in the XY plane."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    target_bounds = _combined_prim_bounds(stage, cache, (target_path,))
    if target_bounds is None:
        return candidates[0] if candidates else None
    target_center = tuple(
        (target_bounds[0][index] + target_bounds[1][index]) * 0.5 for index in range(2)
    )
    distances = []
    for candidate in candidates:
        bounds = _combined_prim_bounds(stage, cache, (candidate,))
        if bounds is None:
            continue
        center = tuple(
            (bounds[0][index] + bounds[1][index]) * 0.5 for index in range(2)
        )
        distance = sum(
            (center[index] - target_center[index]) ** 2 for index in range(2)
        )
        distances.append((distance, candidate))
    return min(distances)[1] if distances else None


def _aim_along_view(
    camera: Sequence[float],
    target: Sequence[float],
    view_direction: Sequence[float],
) -> tuple[float, float, float]:
    """Move a camera within its height until its line of sight passes through a target.

    A focused ROI crop is framed around the target's position on the board. A
    camera straight above the target only sees it when it looks straight down;
    tilted by a pass's ``camera.rotation`` it looks ``height * tan(tilt)`` past
    it. Sliding the camera back along its own line of sight, at the same
    height, puts the target back in the centre of the image. An untilted
    camera is returned unchanged, so a top-down capture is bit-identical.

    Args:
        camera: Current camera position in world units.
        target: Point that must sit at the image centre.
        view_direction: Camera line of sight in world space (any length).

    Returns:
        The camera position at ``camera``'s height that looks at ``target``.

    Raises:
        ValidationError: If the camera looks parallel to the board or away
            from the target, so no camera at this height can see it.

    """
    dx, dy, dz = (float(value) for value in view_direction)
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return (float(camera[0]), float(camera[1]), float(camera[2]))
    height = float(camera[2])
    if abs(dz) < 1e-6:
        raise ValidationError(
            "focused ROI capture: the camera looks parallel to the board, so no "
            "camera at its height can frame the ROI",
        )
    distance = (float(target[2]) - height) / dz
    if distance <= 0.0:
        raise ValidationError(
            "focused ROI capture: the camera looks away from the board "
            f"(view direction {dx:.3f}, {dy:.3f}, {dz:.3f})",
        )
    return (
        float(target[0]) - distance * dx,
        float(target[1]) - distance * dy,
        height,
    )


def _aimed_roi_translate(
    stage: object,
    camera_path: str,
    translate: Sequence[float],
    target: Sequence[float],
) -> tuple[float, float, float]:
    """Return the camera-rig translate that aims the camera at a focused ROI target.

    Reads the orientation the camera actually has on the stage, so it must run
    after the pass's rotation has been applied. The correction is applied as a
    world-space offset to ``translate``, the same convention the scan positions
    use.

    Args:
        stage: Open USD stage holding the capture camera.
        camera_path: Absolute prim path of the capture camera.
        translate: Translate currently set on the camera rig.
        target: Point that must sit at the image centre.

    Returns:
        The translate to set on the camera rig.

    """
    from pxr import Gf, Usd, UsdGeom

    camera_to_world = UsdGeom.Xformable(
        stage.GetPrimAtPath(camera_path),
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    camera = camera_to_world.ExtractTranslation()
    view = camera_to_world.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
    aimed = _aim_along_view(tuple(camera), target, tuple(view))
    return tuple(
        float(value) + (aimed[index] - float(camera[index]))
        for index, value in enumerate(translate)
    )


def _trigger_camera_rotation(
    rotation: Mapping[str, object],
    random_seed: int,
    trigger_index: int,
) -> tuple[float, float, float]:
    """Draw one camera rotation that holds for every frame of a focused-ROI trigger.

    Replicator is reseeded and re-draws a ``rep.distribution`` rotation at
    every written frame, after the frame's scene state is set, so a crop camera
    aimed from the stage would be aimed with the previous frame's tilt. A
    focused-ROI pass therefore draws the tilt itself, once per trigger, from
    its own stream (not ``numpy``, whose draws feed lighting and
    augmentation). A fixed axis is returned exactly.

    Args:
        rotation: The capture's ``camera_rotation`` config.
        random_seed: Plan random seed.
        trigger_index: Index of the trigger.

    Returns:
        ``(x, y, z)`` Euler rotation in degrees, before the bottom-side offset.

    """
    rng = random.Random(f"roi-camera-rotation:{int(random_seed)}:{int(trigger_index)}")

    def draw(low: float, high: float) -> float:
        low, high = float(low), float(high)
        return low if low == high else rng.uniform(low, high)

    return (
        draw(*rotation["x_range"]),
        draw(*rotation["y_range"]),
        float(rotation["z_fixed"]),
    )


def _roi_apertures(
    size_x: float,
    size_y: float,
    *,
    padding: float,
    aspect: float,
    camera_z_rotation: float,
) -> tuple[float, float]:
    """Return rotation-aware orthographic apertures in scene units."""
    radians = math.radians(camera_z_rotation)
    cosine = abs(math.cos(radians))
    sine = abs(math.sin(radians))
    projected_width = cosine * size_x + sine * size_y
    projected_height = sine * size_x + cosine * size_y
    scale = 1.0 + padding
    horizontal = max(
        projected_width * scale,
        projected_height * scale / aspect,
    )
    return horizontal, horizontal * aspect


def _resolve_roi_selection(
    context: IsaacExecutionContext,
    catalog: object,
    root_path: str,
    selections: list[object] | tuple[object, ...],
    components_by_path: Mapping[str, object],
    *,
    random_seed: int,
) -> tuple[tuple[object, int, str, Mapping[str, object]], ...]:
    """Resolve, sample, and deduplicate stacked ROI selectors."""
    selected = []
    selected_paths = set()
    for selector_index, selector in enumerate(selections):
        if not isinstance(selector, Mapping):
            raise ValidationError("ROI selection entries must be mappings")
        if selector.get("pins") is not None or selector.get("focus") == "pin":
            continue
        selector_type, candidates = _roi_selector_candidates(
            context,
            catalog,
            root_path,
            selector,
            components_by_path,
        )
        if not candidates:
            raise ValidationError(
                f"ROI {selector_type} selector found no eligible components",
            )
        candidates = _sample_roi_component_candidates(
            candidates,
            selector_type,
            selector_index,
            selector,
            selected_paths=selected_paths,
            random_seed=random_seed,
        )
        if not candidates:
            continue
        for component in candidates:
            selected.append((component, selector_index, selector_type, selector))
            selected_paths.add(component.prim_path)
    return tuple(selected)


def _sample_roi_component_candidates(
    candidates: list[object],
    selector_type: str,
    selector_index: int,
    selector: Mapping[str, object],
    *,
    selected_paths: set[str],
    random_seed: int,
) -> list[object]:
    """Deduplicate and deterministically sample component candidates."""
    candidates = [
        component
        for component in candidates
        if component.prim_path not in selected_paths
    ]
    count = selector.get("count")
    if count is None and selector_type == "component_types":
        count = 1
    if count is None:
        return candidates

    count = int(count)
    if len(candidates) < count:
        raise ValidationError(
            f"ROI {selector_type} selector requested {count} components "
            f"but only {len(candidates)} eligible components remain",
        )
    return random.Random(random_seed + selector_index).sample(candidates, count)


def _roi_selector_candidates(
    context: IsaacExecutionContext,
    catalog: object,
    root_path: str,
    selector: Mapping[str, object],
    components_by_path: Mapping[str, object],
) -> tuple[str, list[object]]:
    """Resolve one ROI selector into an ordered component candidate list."""
    parent = context.component_parent_paths()
    parent_paths = (
        {
            component.prim_path
            for component in catalog.components
            if component.board_root == root_path
        }
        if parent is None
        else set(parent)
    )
    component_filter = selector.get("component_filter")
    if isinstance(component_filter, Mapping):
        from pcb_aoi.board_metadata import select_components

        components = list(
            select_components(
                catalog,
                component_filter,
                parent_paths=parent_paths,
                board_root=root_path,
                where="capture.roi_selection.component_filter",
            ),
        )
        return "component_filter", components

    refdes = selector.get("refdes")
    if isinstance(refdes, (list, tuple)):
        components = []
        for value in refdes:
            component = catalog.component(board_root=root_path, refdes=str(value))
            if component is None:
                raise ValidationError(f"ROI target {value!r} was not found")
            if component.prim_path not in parent_paths:
                raise ValidationError(
                    f"ROI target {value!r} is outside the scene component filter",
                )
            components.append(component)
        return "refdes", components

    component_types = selector.get("component_types")
    if isinstance(component_types, (list, tuple)):
        matching_paths = {
            component.prim_path
            for value in component_types
            for component in catalog.components_by_type(str(value))
            if component.board_root == root_path and component.prim_path in parent_paths
        }
        components = [
            component
            for component in catalog.components
            if component.prim_path in matching_paths
        ]
        return "component_types", components

    process_classes = selector.get("process_classes")
    if isinstance(process_classes, (list, tuple)):
        requested = {str(value).casefold() for value in process_classes}
        components = [
            component
            for component in catalog.components
            if component.board_root == root_path
            and component.prim_path in parent_paths
            and requested
            & {
                str(value).casefold()
                for value in (
                    *getattr(component, "process_classes", ()),
                    getattr(component, "process_class", None),
                )
                if value
            }
        ]
        return "process_classes", components

    if selector.get("all_components") is True:
        return (
            "all_components",
            [
                component
                for component in catalog.components
                if component.board_root == root_path
                and component.prim_path in parent_paths
            ],
        )

    defects = selector.get("defects")
    if isinstance(defects, (list, tuple)):
        requested = {str(value) for value in defects}
        paths = [
            str(record["path"])
            for record in context.state.get("operation.structural.records") or ()
            if isinstance(record, Mapping) and str(record.get("defect")) in requested
        ]
        if "missing" in requested:
            paths.extend(
                str(path) for path in context.state.get("operation.missing.paths") or ()
            )
        components = [
            components_by_path[path]
            for path in paths
            if path in components_by_path and path in parent_paths
        ]
        return "defects", components

    raise ValidationError("ROI selection entry has no supported selector")


def _setup_lighting(
    stage: object,
    config: dict[str, object],
    rig_layer: object | None,
    common: object,
    *,
    rebuild: bool = False,
) -> None:
    """Apply scene-light policy and build an optional AOI ring rig.

    Args:
        stage: Open USD stage.
        config: Internal capture configuration.
        rig_layer: Authored layer created with the camera, if any.
        common: Existing standalone SDG capability module.
        rebuild: Whether to replace an already-authored static rig.

    """
    from pxr import Usd

    lighting = config["lighting"]
    mode = common.lighting_mode(config)
    if mode == "usd_light":
        return
    rig_root = None
    if mode == "rgb_light":
        rig_root = config["ring_light_root"]
    elif mode == "modus_light":
        rig_root = config["modus_light_root"]
    if rig_root is not None and not rebuild and stage.GetPrimAtPath(rig_root).IsValid():
        return
    common.disable_all_lights(stage)
    if mode not in ("rgb_light", "modus_light"):
        # white_light / dome_light build their single DomeLight per trigger in
        # randomize_lighting; there is no rig to author up front.
        return

    edit_context = (
        Usd.EditContext(stage, Usd.EditTarget(rig_layer)) if rig_layer else None
    )
    with edit_context or contextlib.nullcontext():
        if mode == "modus_light":
            _build_modus_rig(stage, config, lighting, common)
            return
        common.build_ring_lights(
            stage,
            config["ring_light_root"],
            camera_side=config["camera_side"],
            cone_angle=float(sum(lighting["cone_angle_range"]) / 2.0),
            cone_softness=float(sum(lighting["cone_softness_range"]) / 2.0),
        )


def _build_modus_rig(
    stage: object,
    config: dict[str, object],
    lighting: Mapping[str, object],
    common: object,
) -> None:
    """Build the Modus rig to mirror the standalone pipeline's setup.

    The Modus lights are world-fixed under ``modus_light_root`` rather than
    parented to the camera like the ring rig, so their placement is derived
    from the board bounds and they do not travel with the scan.

    Args:
        stage: Open USD stage.
        config: Internal capture configuration.
        lighting: Validated lighting block.
        common: Existing standalone SDG capability module.

    """
    exposure_range = lighting.get("exposure_range") or (0.0, 0.0)
    common.build_modus_lights(
        stage,
        config["modus_light_root"],
        config["pcba_root"],
        camera_side=config["camera_side"],
        modus=lighting["modus"] or {},
        exposure=float(sum(exposure_range) / 2.0),
        bounds_paths=config.get("modus_bounds_paths"),
    )


def _modus_bounds_paths(context: IsaacExecutionContext) -> tuple[str, ...]:
    """Return the prims whose union of world bounds sizes the Modus rig.

    The Modus rig is world-fixed and authored once per capture pass, like the
    line-scan illuminator it models: a bar that spans the whole product passing
    under it, not one camera field of view. On a step-repeated panel that
    product is every registered board placement. ``board_root_path`` there is
    the shared prototype source: often an abstract ``class`` prim with no
    world bounds, and even a concrete source is one board, not the panel.

    Args:
        context: Execution context after ``scene.load``.

    Returns:
        Board-instance paths for a panel, otherwise the board root alone.

    Raises:
        ValidationError: If the scene has not been loaded.

    """
    cohort = context.board_instance_cohort
    if cohort is not None:
        return tuple(str(path) for path in cohort.instance_paths)
    root_path = context.scene.get("board_root_path") or context.scene.get("root_path")
    if not isinstance(root_path, str):
        raise ValidationError("Modus light sizing requires scene.load first")
    return (root_path,)


def _clear_prim_specs(stage: object, prim_path: str) -> None:
    """Delete a prim's specs, and their subtrees, from every layer of the stage.

    Args:
        stage: Open USD stage.
        prim_path: Absolute path of a pipeline-owned prim.

    Raises:
        ValidationError: If a layer holding the prim refuses the edit.

    """
    from pxr import Sdf

    path = Sdf.Path(prim_path)
    for layer in stage.GetLayerStack(includeSessionLayers=True):
        if not layer.GetPrimAtPath(path):
            continue
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(path, Sdf.Path.emptyPath)
        if not layer.Apply(edit):
            raise ValidationError(
                f"cannot clear {prim_path!r} from layer {layer.identifier!r}",
            )


def _apply_lighting_appearance(
    context: IsaacExecutionContext,
    camera_side: str,
) -> dict[str, object]:
    """Apply the configured static lighting rig before stage serialization.

    Args:
        context: Mutable execution context containing the open stage.
        camera_side: Board side the rig illuminates.

    Returns:
        Authored rig mode, root path, and camera side.

    Raises:
        ValidationError: If scene or lighting configuration is unavailable.

    """
    from pxr import Sdf

    stage = context.require_stage()
    lighting = _required_state(context, "appearance.lighting")
    mode = str(lighting["mode"])
    if mode not in {"rgb_light", "modus_light"}:
        raise ValidationError(
            f"appearance.lighting.apply does not author mode {mode!r}",
        )
    root_path = context.scene.get("board_root_path") or context.scene.get("root_path")
    if not isinstance(root_path, str):
        raise ValidationError("appearance.lighting.apply requires scene.load first")

    config = {
        "pcba_root": root_path,
        "camera_side": camera_side,
        "ring_light_root": str(
            context.scene.get("ring_light_root") or "/World/camera_light/aoi_ring_light"
        ),
        "modus_light_root": str(
            context.scene.get("modus_light_root") or "/World/_aoi_modus_light"
        ),
        "modus_bounds_paths": _modus_bounds_paths(context),
        "lighting": _lighting_config(lighting),
    }
    common = _script_module(_STANDALONE_DIR, "common")
    rig_root = (
        config["modus_light_root"]
        if mode == "modus_light"
        else config["ring_light_root"]
    )
    # Each pass re-applies the rig, so the previous pass's rig has to go first:
    # removing prims only edits the current edit target, and a capture that
    # reopens a flattened stage bakes the old rig into the new root layer. Left
    # in any layer, its lights (and their xform ops) stay composed under the
    # rebuilt rig.
    _clear_prim_specs(stage, rig_root)
    sub_layer_paths = stage.GetRootLayer().subLayerPaths
    rig_layers = context.state.setdefault("appearance.lighting.rig_layers", [])
    for stale_layer in rig_layers:
        if stale_layer.identifier in sub_layer_paths:
            sub_layer_paths.remove(stale_layer.identifier)
    rig_layers.clear()
    rig_layer = Sdf.Layer.CreateAnonymous("cdf_job_lighting_rig.usda")
    sub_layer_paths.insert(0, rig_layer.identifier)
    rig_layers.append(rig_layer)
    _setup_lighting(
        stage,
        config,
        rig_layer,
        common,
        rebuild=True,
    )
    result = {
        "mode": mode,
        "root_path": rig_root,
        "camera_side": camera_side,
    }
    context.state["appearance.lighting.rig"] = result
    return result


def _setup_component_appearance(
    context: IsaacExecutionContext,
    stage: object,
    config: dict[str, object],
) -> dict[str, object]:
    """Apply the existing component material setup when it supports the stage.

    Args:
        context: Mutable Isaac execution context.
        stage: Open USD stage.
        config: Internal capture configuration.

    Returns:
        Material setup status suitable for capture metadata.

    """
    previous_pipeline_mode = os.environ.get("FILLET_PIPELINE_MODE")
    os.environ["FILLET_PIPELINE_MODE"] = "1"
    try:
        material_module = _script_module(
            _EDITOR_DIR,
            "component_material_override",
        )
        material_state = material_module._State()
        ready, information = material_state.setup(
            stage,
            component_types=config["component_types"],
            vantablack_body=config["vantablack_components"],
        )
    except Exception as exc:  # noqa: BLE001 - preserve legacy optional behavior
        return {"applied": False, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        if previous_pipeline_mode is None:
            os.environ.pop("FILLET_PIPELINE_MODE", None)
        else:
            os.environ["FILLET_PIPELINE_MODE"] = previous_pipeline_mode

    if ready:
        context.state["runtime.component_material"] = material_state
        return {"applied": True, "details": information}
    return {"applied": False, "reason": str(information)}


async def _write_settled_frame(
    rep: object,
    writer: object,
    *,
    move: Callable[[], None],
    reseed: Callable[[], None],
    rt_subframes: int,
    timer: object,
    armed: bool = False,
) -> None:
    """Move the camera, let the renderer catch up, then write exactly one frame.

    A step issued right after the camera moves can hand the writer an image
    rendered at the previous pose (seen on every cold start and in 5 of 6 warm
    runs at ``rt_subframes`` 4, so a scan grid repeated a cell). The first step
    here is never written; the writer, attached with ``trigger=None``, is
    scheduled only for the second. Both steps see the same seed, so the
    written frame draws exactly what a single step drew before.

    ``armed`` is the first cell of a pass whose writer Replicator already
    scheduled at attach (see :func:`_writer_armed_at_attach`): its next step is
    written whatever we do. The caller applied this cell's state and ran its
    settle step before attaching the writer, so here it is reseeded and
    stepped once, without ``move`` and without ``schedule_write``.

    Args:
        rep: Imported Replicator module.
        writer: Writer attached with ``trigger=None``.
        move: Applies this frame's camera pose and scene state.
        reseed: Seeds Replicator for this output frame.
        rt_subframes: Subframes per step, from the renderer configuration.
        timer: Phase timer with a ``phase(name)`` context manager.
        armed: The writer is already scheduled for the next step.

    """
    if armed:
        reseed()
    else:
        move()
        reseed()
        with timer.phase("frame.settle"):
            await rep.orchestrator.step_async(
                rt_subframes=rt_subframes,
                delta_time=0.0,
            )
        reseed()
        writer.schedule_write()
    with timer.phase("frame.step"):
        await rep.orchestrator.step_async(rt_subframes=rt_subframes, delta_time=0.0)


def _writer_armed_at_attach(rep: object) -> bool:
    """Whether a writer attached now is written on the next step unasked.

    omni.replicator.core 1.13.36, ``OgnScheduleWriterInternalState.__init__``:
    ``is_triggered = get_status() in [Status.STARTED, Status.STEPPED]`` ("to
    capture first frame"). The first orchestrator capture after attach is then
    scheduled even with ``trigger=None``; after the warm-up the status is
    ``STEPPED``, so it always is here. Mirror that rule rather than assume it.
    """
    status = rep.orchestrator.get_status()
    return status in (rep.orchestrator.Status.STARTED, rep.orchestrator.Status.STEPPED)


def _require_one_write_per_cell(written: object, expected: int, pass_dir: str) -> None:
    """Fail when the writer did not write exactly one frame per camera position.

    An extra or missing write shifts every later file onto the wrong grid cell,
    and nothing downstream can tell. ``written`` is the writer's ``num_written``
    read before detach; anything but an int is not checked.
    """
    if isinstance(written, int) and written != expected:
        raise RuntimeError(
            f"writer wrote {written} frame(s) for {expected} camera position(s) "
            f"in {pass_dir}; output files would not map to grid cells",
        )


async def _warmup(
    render_product: object,
    rep: object,
    *,
    timeout_s: float = _WARMUP_TIMEOUT_S,
) -> None:
    """Step the renderer until it delivers frames, before writer attachment.

    With a cold RTX shader cache, steps return without rendering until the
    pipeline has compiled, and a writer attached during that window receives
    no frame and writes nothing. A fixed step count therefore does not warm
    the renderer; this steps until the ``rgb`` annotator returns image data.

    Args:
        render_product: Replicator render product used for capture.
        rep: Imported Replicator module.
        timeout_s: Seconds after which a step that still returns no frame
            gives up. A frame returned by a step that ran past the deadline
            is accepted, since one step can block for the whole compile.

    Raises:
        RuntimeError: If no frame has arrived once ``timeout_s`` has elapsed.

    """
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    render_product_path = render_product.path
    annotator.attach(render_product_path)
    try:
        deadline = time.monotonic() + timeout_s
        steps = 0
        while True:
            await rep.orchestrator.step_async(rt_subframes=4, delta_time=0.0)
            steps += 1
            frame = annotator.get_data()
            if steps >= _WARMUP_MIN_STEPS and getattr(frame, "size", 0):
                return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"renderer delivered no frame for {render_product_path} "
                    f"after {steps} warm-up steps in {timeout_s:.0f}s",
                )
    finally:
        annotator.detach(render_product_path)


def _positions_within_image_budget(
    positions: Sequence[tuple[float, float, float]],
    max_image_count: int,
    outputs_written: int,
) -> list[tuple[float, float, float]]:
    """Trim capture positions so the pass stays within ``max_images``.

    ``max_images`` counts capture positions (image sets), not writer frames.
    A ``missing`` pipeline renders every position twice (an aligned
    ``reference``/``defective`` pair) and that pair is one image set, so
    ``max_images: 1`` yields exactly one pair.

    Args:
        positions: Candidate camera positions for this trigger.
        max_image_count: Configured ``max_images``; ``-1`` means unlimited.
        outputs_written: Positions already rendered by earlier triggers.

    Returns:
        The leading positions that fit the remaining budget.

    """
    if max_image_count < 0:
        return list(positions)
    remaining = int(max_image_count) - int(outputs_written)
    if remaining <= 0:
        return []
    return list(positions[:remaining])


def _zero_frame_message(
    pipeline_type: str,
    config: Mapping[str, object],
    *,
    roi_spec_count: int,
) -> str:
    """Describe why a capture pass produced no frames."""
    scan_grid = config.get("scan_grid") or {}
    return (
        "capture pass rendered no frames "
        f"(pipeline_type={pipeline_type}, triggers={config.get('num_triggers')}, "
        f"max_images={config.get('max_image_count')}, "
        f"output.start={config.get('image_seed')}, "
        f"roi_targets={roi_spec_count}, "
        f"scan_grid={scan_grid.get('x_num')}x{scan_grid.get('y_num')}); "
        "raise capture.limits.max_images, lower capture.output.start, or "
        "check that the ROI selection resolves to at least one component"
    )


async def execute_capture(
    context: IsaacExecutionContext,
    parameters: CaptureRenderParameters,
) -> dict[str, object]:
    """Render a configured JobOrder scan inside the current Isaac process.

    Args:
        context: Prepared execution context with an open stage.
        parameters: Validated capture destination.

    Returns:
        Structured output directory, frame, trigger, pixel-layout, metadata,
        and material information. ``trigger_metadata`` contains the same
        per-trigger dictionaries written to ``metadata.json``.

    Raises:
        ValidationError: If required configuration or component state is absent.

    """
    import omni.replicator.core as rep
    from pxr import Gf

    timer = PhaseTimer()
    with timer.phase("stage.refresh"):
        stage = await _refresh_instanced_capture_stage(context)
    config = build_capture_config(context, parameters.output_dir)
    guard_px = int(config["guard_px"])
    output_root = config["output"]
    Path(output_root).mkdir(parents=True, exist_ok=True)
    pass_id = context.active_order_id or "capture.render"
    try:
        context.manifest_view.capture_pass(pass_id)
    except KeyError:
        context.manifest.begin_capture_pass(
            pass_id,
            output_uri=str(parameters.output_dir),
            local_root=str(output_root),
        )

    with timer.phase("scripts.import"):
        common = _script_module(_STANDALONE_DIR, "common")
        defect_ops = _script_module(_STANDALONE_DIR, "defect_ops")
    defect_records = list(context.state.get("operation.structural.records") or ())
    solder_defect_records = list(
        context.state.get("operation.solder_defects.records") or ()
    )
    missing_paths = list(context.state.get("operation.missing.paths") or ())
    all_defect_records = [*defect_records, *solder_defect_records]
    all_defect_records.extend(
        {"path": path, "defect": "missing"} for path in missing_paths
    )
    defect_ids = context.manifest.observe_defects(
        all_defect_records,
        context.entity_identity,
    )
    defects_by_id = {
        str(defect["defect_id"]): defect for defect in context.manifest_view.defects()
    }
    missing_ops = (
        _script_module(_STANDALONE_DIR, "missing_ops") if missing_paths else None
    )
    if missing_paths:
        pipeline_type = "missing"
    elif all_defect_records:
        pipeline_type = "defect"
    else:
        pipeline_type = "good"
    with timer.phase("renderer.configure"):
        rt_subframes = common.configure_pathtracing(config)
    with timer.phase("rig.camera"):
        rig_layer = _ensure_camera(stage, config)
    with timer.phase("rig.lighting"):
        _setup_lighting(stage, config, rig_layer, common)
    with timer.phase("appearance.components"):
        material = _setup_component_appearance(context, stage, config)

    with timer.phase("render_product.create"):
        # Larger than the delivered tile by the guard band, which the writer
        # crops off again; with no guard band the two are the same.
        render_product = rep.create.render_product(
            config["camera_path"],
            config["render_resolution"],
        )
    visibility_snapshots = ()
    hidden_geometry = tuple(config.get("hidden_geometry") or ())
    shown_geometry = tuple(config.get("shown_geometry") or ())
    active_writers = []
    frames = []
    trigger_directories = []
    trigger_metadata = []
    pixel_layouts = []
    semantic_projection_artifacts = []
    roi_warnings = ()
    try:
        with timer.phase("visibility.show"):
            if shown_geometry:
                from pcb_aoi.capture_visibility import _show_board_roles

                board_root = str(config["pcba_root"])
                visibility_snapshots = _show_board_roles(
                    stage,
                    board_root,
                    set(shown_geometry),
                )
        with timer.phase("visibility.hide"):
            if hidden_geometry:
                hidden_set = set(hidden_geometry)
                solder_targets = hidden_set & {"solder"}
                role_targets = hidden_set - {"solder"}
                if solder_targets:
                    manifest = context.state.get("solder_joints.placement_manifest")
                    if manifest is None:
                        raise ValidationError(
                            "capture visibility cannot hide solder before solder attachment",
                        )
                    from pcb_aoi.capture_visibility import _hide_solder

                    solder_snapshots = _hide_solder(
                        stage,
                        manifest,
                        context.board_instance_cohort,
                    )
                    visibility_snapshots = visibility_snapshots + solder_snapshots
                if role_targets:
                    from pcb_aoi.capture_visibility import _hide_board_roles

                    board_root = str(config["pcba_root"])
                    role_snapshots = _hide_board_roles(stage, board_root, role_targets)
                    visibility_snapshots = visibility_snapshots + role_snapshots
        # The first warm-up pays for RTX pipeline and MDL shader compilation,
        # which a persistent session pays once; the per-pass one below does not.
        with timer.phase("warmup.initial"):
            await _warmup(render_product, rep)
        with timer.phase("scan_grid.resolve"):
            common.resolve_scan_grid_from_stage(stage, config)
            positions_full = common.build_scan_positions(config["scan_grid"])
        if missing_ops is not None and config.get("roi_selection"):
            missing_ops.restore_components(stage, missing_paths)
        try:
            with timer.phase("roi.select"):
                roi_specs = _roi_selection_specs(
                    context,
                    stage,
                    config,
                    float(positions_full[0][2]),
                )
            roi_warnings = tuple(
                context.state.get("capture.roi_selection.warnings") or (),
            )
        finally:
            if missing_ops is not None and config.get("roi_selection"):
                missing_ops.hide_components(stage, missing_paths)
        camera_schema = None
        if roi_specs:
            from pxr import UsdGeom

            camera_schema = UsdGeom.Camera.Get(stage, config["camera_path"])
        translate_op = common.find_translate_op(
            stage,
            config["camera_xform_path"],
        )
        camera = rep.get.prim_at_path(config["camera_path"])
        rotation = config["camera_rotation"]
        camera_x_offset = 180.0 if config["camera_side"] == "bottom" else 0.0
        image_seed = config["image_seed"]
        frame_cursor = 0
        output_cursor = 0

        for trigger_index in range(config["num_triggers"]):
            trigger_dir = os.path.join(
                output_root,
                f"trigger_{trigger_index:04d}",
            )
            os.makedirs(trigger_dir, exist_ok=True)
            trigger_directories.append(trigger_dir)

            common.seed_trigger_numpy(config["random_seed"], trigger_index)
            light_metadata = common.randomize_lighting(stage, config)
            common.set_replicator_seed_for_output_frame(
                config["random_seed"] + trigger_index,
            )
            if roi_specs:
                # Each crop camera is aimed from its settled tilt (see
                # _aimed_roi_translate), so the tilt must hold for the trigger.
                roi_rotation = _trigger_camera_rotation(
                    rotation,
                    config["random_seed"],
                    trigger_index,
                )
                pose_rotation = (
                    roi_rotation[0] + camera_x_offset,
                    roi_rotation[1],
                    roi_rotation[2],
                )
            else:
                pose_rotation = rep.distribution.uniform(
                    (
                        rotation["x_range"][0] + camera_x_offset,
                        rotation["y_range"][0],
                        rotation["z_fixed"],
                    ),
                    (
                        rotation["x_range"][1] + camera_x_offset,
                        rotation["y_range"][1],
                        rotation["z_fixed"],
                    ),
                )
            with camera:
                rep.modify.pose(rotation=pose_rotation)
            augmentation_metadata = common.setup_augmentation(
                render_product,
                config,
            )
            if roi_specs:
                selected_roi_specs = list(roi_specs[image_seed:])
                positions = [spec["position"] for spec in selected_roi_specs]
            else:
                selected_roi_specs = []
                positions = common.scan_positions_for_trigger(
                    positions_full,
                    trigger_index,
                    image_seed,
                )
            positions = _positions_within_image_budget(
                positions,
                config["max_image_count"],
                output_cursor,
            )
            if selected_roi_specs:
                selected_roi_specs = selected_roi_specs[: len(positions)]
            if not positions:
                if (
                    config["max_image_count"] >= 0
                    and output_cursor >= config["max_image_count"]
                ):
                    break
                continue

            frame_to_grid = []
            trigger_frames: list[dict[str, object]] = []
            trigger_artifacts: list[dict[str, object]] = []
            pass_specs = [(None, trigger_dir, False)]
            if missing_paths:
                pass_specs = [
                    ("reference", os.path.join(trigger_dir, "reference"), False),
                    ("defective", os.path.join(trigger_dir, "defective"), True),
                ]
            base_frame_cursor = frame_cursor
            for role, pass_dir, hide_missing in pass_specs:
                os.makedirs(pass_dir, exist_ok=True)
                if missing_ops is not None:
                    if hide_missing:
                        missing_ops.hide_components(stage, missing_paths)
                    else:
                        missing_ops.restore_components(stage, missing_paths)

                first_x, first_y, first_z = positions[0]
                first_roi_spec = selected_roi_specs[0] if selected_roi_specs else None
                if first_roi_spec is not None:
                    assert camera_schema is not None
                    _set_roi_apertures(camera_schema, first_roi_spec, config)
                translate_op.Set(Gf.Vec3d(first_x, first_y, first_z))
                with timer.phase("warmup.pass"):
                    await _warmup(render_product, rep)

                # Where each focused-ROI cell's camera rig actually stood, once
                # aimed; the frame records it so it agrees with the calibration.
                aimed_translates: dict[int, tuple[float, float, float]] = {}

                def cell_state(
                    position_index,
                    positions=positions,
                    selected_roi_specs=selected_roi_specs,
                    base_frame_cursor=base_frame_cursor,
                    aimed_translates=aimed_translates,
                ):
                    """Callables applying one cell's scene state and seed."""
                    x, y, z = positions[position_index]
                    roi_spec = (
                        selected_roi_specs[position_index]
                        if selected_roi_specs
                        else None
                    )
                    frame_number = image_seed + base_frame_cursor + position_index

                    def apply_cell():
                        if roi_spec is not None:
                            assert camera_schema is not None
                            _set_roi_apertures(camera_schema, roi_spec, config)
                        translate_op.Set(Gf.Vec3d(x, y, z))
                        if roi_spec is not None:
                            aimed = _aimed_roi_translate(
                                stage,
                                config["camera_path"],
                                (x, y, z),
                                roi_spec["target"],
                            )
                            aimed_translates[position_index] = aimed
                            translate_op.Set(Gf.Vec3d(*aimed))
                        if defect_records:
                            defect_ops.reapply_defects(defect_records)

                    def seed_cell():
                        common.set_replicator_seed_for_output_frame(frame_number)

                    return apply_cell, seed_cell

                # Cell 0's full state, settled before any writer exists so this
                # step cannot be written. An armed writer then writes cell 0
                # from one step (see _writer_armed_at_attach).
                first_move, first_reseed = cell_state(0)
                first_move()
                first_reseed()
                with timer.phase("frame.settle"):
                    await rep.orchestrator.step_async(
                        rt_subframes=rt_subframes,
                        delta_time=0.0,
                    )

                with timer.phase("writer.attach"):
                    writer = rep.WriterRegistry.get("BasicWriter")
                    writer.initialize(output_dir=pass_dir, **config["writer"])
                    if guard_px:
                        # In memory, before encoding: no re-encode, and every
                        # annotator is cropped from the same payload.
                        guard_band.install_writer_crop(
                            writer,
                            guard_px,
                            config["render_resolution"],
                        )
                    common.set_writer_start_frame_index(
                        writer,
                        image_seed + base_frame_cursor,
                    )
                    # Manual trigger: only the step after schedule_write() is
                    # written, so each cell's settle step stays out of the output.
                    armed = _writer_armed_at_attach(rep)
                    writer.attach([render_product], trigger=None)
                    active_writers.append(writer)

                pass_frame_to_grid = []
                role_frames: list[dict[str, object]] = []
                try:
                    for position_index, (x, y, z) in enumerate(positions):
                        roi_spec = (
                            selected_roi_specs[position_index]
                            if selected_roi_specs
                            else None
                        )
                        frame_number = image_seed + base_frame_cursor + position_index
                        move, reseed = cell_state(position_index)
                        await _write_settled_frame(
                            rep,
                            writer,
                            move=move,
                            reseed=reseed,
                            rt_subframes=rt_subframes,
                            timer=timer,
                            armed=armed and position_index == 0,
                        )
                        camera_position = aimed_translates.get(
                            position_index,
                            (x, y, z),
                        )
                        grid_index = common.grid_index_for_position(
                            x,
                            y,
                            config["scan_grid"],
                        )
                        if grid_index is not None:
                            pass_frame_to_grid.append(
                                {
                                    "frame": frame_number,
                                    "x_idx": grid_index[0],
                                    "y_idx": grid_index[1],
                                },
                            )
                        frame = {
                            "frame_id": (
                                f"{pass_id}:trigger:{trigger_index:04d}:"
                                f"frame:{frame_number:04d}:{role or 'capture'}"
                            ),
                            "frame": frame_number,
                            "trigger": trigger_index,
                            "role": role or "capture",
                            "position": list(camera_position),
                            "camera": {"position": list(camera_position)},
                            "defect_ids": [
                                defect_id
                                for defect_id, record in zip(
                                    defect_ids,
                                    all_defect_records,
                                )
                                if not (
                                    record.get("defect") == "missing"
                                    and role == "reference"
                                )
                            ],
                            "artifacts": {},
                        }
                        if role is not None:
                            frame["pair_id"] = (
                                f"{pass_id}:trigger:{trigger_index:04d}:"
                                f"pair:{frame_number:04d}"
                            )
                        if grid_index is not None:
                            frame["grid"] = {
                                "x_idx": grid_index[0],
                                "y_idx": grid_index[1],
                            }
                        if roi_spec is not None:
                            roi_selection = {
                                key: value
                                for key, value in roi_spec.items()
                                if key not in ("position", "target")
                            }
                            frame["roi_selection"] = roi_selection
                            frame["camera"]["roi_selection"] = roi_selection
                        # The camera is where this frame was rendered from, so
                        # this is the one moment its calibration is knowable.
                        frame["camera"]["calibration"] = _frame_calibration(
                            stage,
                            config["camera_path"],
                            config["render_resolution"],
                            crop_px=guard_px,
                        )
                        frames.append(frame)
                        role_frames.append(frame)
                        trigger_frames.append(frame)
                finally:
                    # Split deliberately: these two are the same "flush" to a
                    # caller and completely different costs to fix.
                    with timer.phase("orchestrator.wait"):
                        await rep.orchestrator.wait_until_complete_async()
                    pass_written = getattr(writer, "num_written", None)
                    with timer.phase("writer.detach"):
                        writer.detach()
                        active_writers.remove(writer)
                _require_one_write_per_cell(pass_written, len(positions), pass_dir)
                if guard_px:
                    guard_band.raise_writer_crop_errors(writer, pass_dir)
                calibration_artifact = _write_pass_calibration(
                    Path(output_root),
                    pass_dir,
                    role_frames,
                    context.manifest.serializer,
                    role or "capture",
                    trigger_index,
                )
                if calibration_artifact is not None:
                    trigger_artifacts.append(calibration_artifact)
                semantic_projection = config.get("semantic_projection")
                if isinstance(semantic_projection, Mapping):
                    from pcb_aoi.postprocess.semantic_projection import (
                        project_instance_semantics,
                    )

                    with timer.phase("semantic.projection"):
                        projected = project_instance_semantics(
                            pass_dir,
                            semantic_projection,
                            semantic_types=tuple(
                                semantic_projection["semantic_types"],
                            ),
                            colorized=bool(semantic_projection["colorized"]),
                            remove_instance_artifacts=bool(
                                semantic_projection["remove_instance_artifacts"],
                            ),
                        )
                    semantic_projection_artifacts.extend(
                        {
                            **artifact,
                            "role": role,
                            "trigger": trigger_index,
                        }
                        for artifact in projected
                    )
                # Before the grid rename and before artifact discovery, so both
                # see the final extension.
                if config.get("convert_rgb_to"):
                    with timer.phase("rgb.convert"):
                        convert_rgb_outputs(
                            pass_dir,
                            config["convert_rgb_to"],
                            frame_padding=int(config["writer"]["frame_padding"]),
                            frames=[int(frame["frame"]) for frame in role_frames],
                        )
                if config["rename_to_grid_index"]:
                    common.rename_outputs_to_grid_index(
                        pass_dir,
                        [
                            (
                                item["frame"],
                                item["x_idx"],
                                item["y_idx"],
                            )
                            for item in pass_frame_to_grid
                        ],
                        frame_padding=int(config["writer"]["frame_padding"]),
                    )
                trigger_artifacts.extend(
                    _discover_frame_artifacts(
                        Path(output_root),
                        Path(pass_dir),
                        role_frames,
                        pass_frame_to_grid,
                    )
                )
                _require_frame_artifacts(role_frames, Path(pass_dir))
                if guard_px:
                    # The tiles are what a consumer validates against
                    # dst_rect, so check the files, not the crop's intent.
                    guard_band.require_delivered_size(
                        [
                            Path(output_root) / path
                            for frame in role_frames
                            for paths in frame["artifacts"].values()
                            for path in paths
                        ],
                        config["resolution"],
                    )
                _refine_frame_defect_associations(
                    Path(output_root),
                    role_frames,
                    defects_by_id,
                )
                if not frame_to_grid:
                    frame_to_grid = pass_frame_to_grid

            frame_cursor += len(positions)
            output_cursor += len(positions)
            if roi_specs:
                pixel_layout = {
                    "supported": False,
                    "reason": "focused ROI frames use independent camera apertures",
                    "camera_side": config.get("camera_side", "top"),
                    "canvas_size_px": list(config["resolution"]),
                }
            else:
                pixel_layout = scan_grid_geometry.compute_pixel_layout(
                    config["scan_grid"],
                    config["resolution"],
                    config["camera_rotation"],
                    frame_to_grid,
                    camera_side=config.get("camera_side", "top"),
                )
                if pixel_layout.get("supported") and config.get("scan_image_size_px"):
                    # The size the capture stated; the stitch crops the mosaic
                    # of whole-pixel cells back to it.
                    pixel_layout = {
                        **pixel_layout,
                        "image_size_px": [
                            int(value) for value in config["scan_image_size_px"]
                        ],
                    }
                if pixel_layout.get("supported"):
                    top_side = config.get("camera_side", "top") == "top"
                    reference_z, reference_source = _capture_reference_plane_z(
                        context,
                        stage,
                        config["scan_grid"],
                        top_side=top_side,
                    )
                    # Only the auto-derived scan grid (resolve_scan_grid_from_stage)
                    # carries a board bbox at all; the legacy explicit
                    # x_start/x_end form does not, so there is no reference
                    # plane to declare in that case.
                    if reference_z is not None:
                        from pxr import UsdGeom

                        pixel_layout = {
                            **pixel_layout,
                            "world_reference_z": reference_z,
                            "reference_plane_source": reference_source,
                            # The override when there is one: this field is
                            # what a consumer multiplies our world coordinates
                            # by, so it has to be the scale the capture was
                            # actually framed with. Carrying the stage's own
                            # declaration here while framing by another would
                            # hand out a mapping that is wrong by their ratio.
                            "stage_meters_per_unit": float(
                                config.get("meters_per_unit_override")
                                or UsdGeom.GetStageMetersPerUnit(stage),
                            ),
                            "stage_meters_per_unit_source": (
                                "job_override"
                                if config.get("meters_per_unit_override")
                                else "stage_declaration"
                            ),
                            "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                            "source_usd": context.scene.get("usd_path"),
                            # Which board placements the frame actually contains.
                            # Asked for so a consumer does not have to guess
                            # which step-repeat a stitched panel image shows.
                            "board_instances": _board_instances_in_frame(
                                context,
                                stage,
                                config["scan_grid"],
                            ),
                            # Which reading of "the board" the scan rectangle
                            # was centred on. Carried, not interpreted -- see
                            # ScanGridParameters.centered_on.
                            "scan_area_centered_on": config.get(
                                "scan_area_centered_on",
                            ),
                            # World XY (mm) the rectangle was placed on when the
                            # field of view was stated as corners; None when it
                            # was centered on the board.
                            "scan_center_mm": config.get("scan_center_mm"),
                            # A path names a file, not a build. Two exports a week
                            # apart share a path and are different stages, so the
                            # stage's own provenance travels with the mapping.
                            "source_usd_provenance": _stage_provenance(stage),
                        }
            pixel_layouts.append(
                {
                    "trigger_idx": trigger_index,
                    **pixel_layout,
                },
            )

            legacy_metadata = {
                "trigger_idx": trigger_index,
                "num_positions": len(positions),
                "lighting": light_metadata,
                "augmentation": augmentation_metadata,
                "material": material,
                "defects": [dict(record) for record in all_defect_records],
                "missing_components": missing_paths,
                "frame_to_grid": frame_to_grid,
                "roi_selections": [
                    {
                        "frame": image_seed + base_frame_cursor + index,
                        **{
                            key: value
                            for key, value in spec.items()
                            if key not in ("position", "target")
                        },
                    }
                    for index, spec in enumerate(selected_roi_specs)
                ],
                "warnings": list(roi_warnings),
                "hidden_geometry": list(hidden_geometry),
                "shown_geometry": list(shown_geometry),
                "manifest_visibility_annotations": bool(
                    config["manifest_visibility_annotations"]
                ),
                "pixel_layout": pixel_layout,
                # Pixels rendered beyond every tile edge and cropped before the
                # tiles were written; the tiles and pixel_layout above are the
                # delivered size either way.
                "guard_px": guard_px,
                "job_order_plan": dict(context.plan_metadata),
            }
            metadata = context.manifest.record_trigger(
                pass_id,
                trigger_index,
                legacy_metadata,
                trigger_frames,
                trigger_artifacts,
            )
            trigger_metadata.append(metadata)
            if (
                config["max_image_count"] >= 0
                and output_cursor >= config["max_image_count"]
            ):
                break
    finally:
        try:
            for writer in active_writers:
                writer.detach()
            render_product.destroy()
        finally:
            try:
                if visibility_snapshots:
                    from pcb_aoi.capture_visibility import _restore_visibility

                    _restore_visibility(stage, visibility_snapshots)
            finally:
                if defect_records:
                    defect_ops.restore_defects(defect_records)
                if missing_ops is not None:
                    missing_ops.restore_components(stage, missing_paths)

    if not frames:
        raise ValidationError(
            _zero_frame_message(
                pipeline_type,
                config,
                roi_spec_count=len(roi_specs),
            ),
        )
    return {
        "output_dir": output_root,
        "pipeline_type": pipeline_type,
        "frame_count": len(frames),
        "frames": frames,
        "trigger_directories": trigger_directories,
        "trigger_metadata": trigger_metadata,
        "pixel_layouts": pixel_layouts,
        "material": material,
        "semantic_projection": semantic_projection_artifacts,
        "hidden_geometry": list(hidden_geometry),
        "shown_geometry": list(shown_geometry),
        "warnings": list(roi_warnings),
        "phases": timer.report(),
    }
