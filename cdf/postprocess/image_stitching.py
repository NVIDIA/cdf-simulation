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

"""Stitch orthographic CDF scan-grid frames into canonical images."""

from __future__ import annotations

import math
import re
import warnings
from itertools import pairwise
from pathlib import Path

from ..errors import ValidationError


def _frame_image_index(trigger_dir: Path, image_type: str) -> dict[int, Path]:
    """Index one trigger's frame images by BasicWriter frame number."""
    pattern = re.compile(rf"^{re.escape(image_type)}_(\d+)\.[^.]+$")
    images = {}
    for path in trigger_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None:
            images[int(match.group(1))] = path
    return images


def _grid_image_index(
    trigger_dir: Path,
    image_type: str,
) -> dict[tuple[int, int], Path]:
    """Index one trigger's grid-renamed images by X and Y tile indices."""
    pattern = re.compile(
        rf"^{re.escape(image_type)}_x(\d+)_y(\d+)\.[^.]+$",
    )
    images = {}
    for path in trigger_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None:
            images[(int(match.group(1)), int(match.group(2)))] = path
    return images


def _trigger_output_name(
    output_filename: str,
    trigger_index: int,
    trigger_count: int,
) -> str:
    """Return a stable output filename for one trigger."""
    if trigger_count == 1:
        return output_filename
    path = Path(output_filename)
    return f"{path.stem}_trigger_{trigger_index:04d}{path.suffix}"


BLEND_MODES = ("none", "feather", "labels")


def _tile_overlaps(tiles: list[dict[str, object]]) -> tuple[int, int]:
    """Return the (x, y) tile overlap in pixels implied by a pixel layout.

    ``compute_pixel_layout`` places tiles from geometry alone and deliberately
    leaves the overlap policy to the caller, so the amount is recovered here from
    the smallest gap between neighbouring destination origins.

    Args:
        tiles: Pixel-layout tile entries.

    Returns:
        Overlap along x and y, clamped at zero when tiles abut or leave a gap.

    """
    first = tiles[0]["dst_rect_xywh_px"]
    width, height = float(first[2]), float(first[3])
    lefts = sorted({round(float(t["dst_rect_xywh_px"][0])) for t in tiles})
    tops = sorted({round(float(t["dst_rect_xywh_px"][1])) for t in tiles})
    step_x = min((b - a for a, b in pairwise(lefts)), default=width)
    step_y = min((b - a for a, b in pairwise(tops)), default=height)
    return max(0, int(width - step_x)), max(0, int(height - step_y))


def _frame_positions(metadata: dict[str, object]) -> dict[int, object]:
    """Return the recorded camera position per frame from the trigger metadata."""
    positions = {}
    for record in metadata.get("frames") or ():
        if not isinstance(record, dict) or "frame" not in record:
            continue
        camera = record.get("camera")
        position = camera.get("position") if isinstance(camera, dict) else None
        positions[int(record["frame"])] = position or record.get("position")
    return positions


def _reject_repeated_cell(
    seen: dict[bytes, tuple[int, object]],
    tile,
    frame: int,
    tile_spec: dict[str, object],
    trigger_index: int,
    positions: dict[int, object],
    semantics: bytes = b"",
) -> None:
    """Refuse a mosaic in which one rendered frame stands in for another cell.

    ``semantics`` is the tile's canonical label table for segmentation images.
    A pixel identifier only means something through that table, so two tiles
    repeat each other only when both the pixels and their meaning match; equal
    identifiers naming different classes in different frames are legitimate.

    Uniformity is judged per colour channel with alpha excluded: a flat coloured
    background has zero spread in each channel, but not across channels, and a
    constant alpha would otherwise count as variation.

    A stale renderer frame repeats an earlier cell byte for byte while the
    camera has moved on, and nothing downstream notices. Two cells of one grid
    can only match exactly when there is nothing in them -- a flat background
    renders the same everywhere -- so identical cells are an error unless the
    content is near-uniform.
    """
    import hashlib

    import numpy as np

    samples = np.asarray(tile)
    hasher = hashlib.blake2b(samples.tobytes(), digest_size=16)
    hasher.update(semantics)
    digest = hasher.digest()
    cell = (
        tile_spec.get("x_idx"),
        tile_spec.get("y_idx"),
        tile_spec.get("dst_rect_xywh_px"),
    )
    first = seen.get(digest)
    if first is None:
        seen[digest] = (frame, cell)
        return
    first_frame, first_cell = first
    if first_cell == cell or _max_channel_std(samples) < _UNIFORM_TILE_STD:
        return

    def where(frame_number, grid_cell):
        position = positions.get(frame_number)
        placed = f"cell (x={grid_cell[0]}, y={grid_cell[1]}) at {grid_cell[2]}"
        return f"frame {frame_number} [{placed}" + (
            f", camera {position}]" if position is not None else "]"
        )

    raise ValidationError(
        f"trigger {trigger_index}: {where(frame, cell)} is pixel-identical to "
        f"{where(first_frame, first_cell)}; the renderer delivered a stale frame "
        "for a different cell, so the stitched board would repeat part of it",
    )


def _tile_semantics(image_type: str, source_path: Path) -> bytes:
    """Return the tile's label sidecar in canonical form, or empty bytes without one."""
    import json

    suffix = source_path.stem.removeprefix(f"{image_type}_")
    labels_path = source_path.with_name(f"{image_type}_labels_{suffix}.json")
    if not labels_path.is_file():
        return b""
    raw = labels_path.read_bytes()
    try:
        return json.dumps(json.loads(raw), sort_keys=True).encode("utf-8")
    except ValueError:
        return raw


#: Below this pixel standard deviation (8-bit) a tile is treated as empty
#: background, which may legitimately repeat across cells.
_UNIFORM_TILE_STD = 1.0


def _max_channel_std(samples) -> float:
    """Largest per-channel standard deviation over the colour channels (alpha excluded)."""
    import numpy as np

    pixels = np.asarray(samples, dtype=np.float64)
    if pixels.ndim == 3 and pixels.shape[-1] in (2, 4):
        pixels = pixels[..., :-1]
    if pixels.ndim == 3:
        return float(pixels.reshape(-1, pixels.shape[-1]).std(axis=0).max())
    return float(pixels.std())


def _feather_ramp(length: int, feather: int):
    """Return a smoothstep ramp that rises over ``feather`` pixels at each end."""
    import numpy as np

    weights = np.ones(length, np.float32)
    width = min(feather, length // 2)
    if width > 0:
        edge = np.linspace(0.0, 1.0, width + 2, dtype=np.float32)[1:-1]
        edge = edge * edge * (3.0 - 2.0 * edge)
        weights[:width] = edge
        weights[-width:] = edge[::-1]
    return weights


def _parse_label_identifier(value: object) -> int | tuple[int, int, int, int]:
    """Parse a numeric or colorized segmentation label identifier."""
    text = str(value).strip()
    if not text.startswith("("):
        try:
            return int(text)
        except ValueError as error:
            raise ValidationError(
                f"invalid segmentation label identifier {value!r}",
            ) from error
    if not text.endswith(")"):
        raise ValidationError(f"invalid segmentation color identifier {value!r}")
    try:
        channels = tuple(int(part.strip()) for part in text[1:-1].split(","))
    except ValueError as error:
        raise ValidationError(
            f"invalid segmentation color identifier {value!r}",
        ) from error
    if len(channels) != 4 or any(channel < 0 or channel > 255 for channel in channels):
        raise ValidationError(
            f"segmentation color identifier must contain four bytes: {value!r}",
        )
    return channels


def _colorized_label_identifiers(keys, color_map=None):
    """Assign stable RGBA identifiers to one unified semantic table.

    Args:
        keys: Ordered class-entry tuples from the merged tile label tables.
        color_map: Optional mapping of class name to an RGBA 4-tuple.  When
            a class name appears in ``color_map``, its specified color is used
            instead of the hash-derived assignment.  Collision avoidance still
            applies to hash-derived colors; user-supplied colors are used as-is
            and must not duplicate each other or the reserved BACKGROUND and
            UNLABELLED identifiers.

    Raises:
        ValidationError: If ``color_map`` contains an invalid class name, RGBA
            identifier, or duplicate color assignment.

    """
    from .semantic_colors import semantic_color

    identifiers = {}
    used = set()
    if color_map is not None and not isinstance(color_map, dict):
        raise ValidationError("color_map must be a dict")
    for class_name, rgba in (color_map or {}).items():
        if not isinstance(class_name, str):
            raise ValidationError("color_map keys must be strings")
        if (
            not isinstance(rgba, (list, tuple))
            or len(rgba) != 4
            or not all(
                isinstance(channel, int) and 0 <= channel <= 255 for channel in rgba
            )
        ):
            raise ValidationError(
                f"color_map[{class_name!r}] must be an RGBA 4-tuple of "
                "integers in [0, 255]",
            )
    normalized_map = {name: tuple(rgba) for name, rgba in (color_map or {}).items()}
    reserved_colors = {
        (0, 0, 0, 0): "BACKGROUND",
        (0, 0, 0, 255): "UNLABELLED",
    }
    explicit_colors = dict(reserved_colors)
    for class_name, color in normalized_map.items():
        if class_name in ("BACKGROUND", "UNLABELLED"):
            continue
        existing_class = explicit_colors.get(color)
        if existing_class is not None:
            raise ValidationError(
                f"color_map assigns RGBA {color!r} to both "
                f"{existing_class!r} and {class_name!r}",
            )
        explicit_colors[color] = class_name

    used.update(explicit_colors)
    for key in keys:
        class_name = dict(key).get("class")
        if class_name == "BACKGROUND":
            color = (0, 0, 0, 0)
        elif class_name == "UNLABELLED":
            color = (0, 0, 0, 255)
        elif class_name in normalized_map:
            color = normalized_map[class_name]
        else:
            color = semantic_color(dict(key))
            while color in used:
                packed = (color[0] | color[1] << 8 | color[2] << 16) + 1
                color = (
                    packed & 255,
                    packed >> 8 & 255,
                    packed >> 16 & 255,
                    color[3],
                )
        identifiers[key] = color
        used.add(color)
    return identifiers


def _tile_label_table(
    image_type: str,
    frame_sources: dict[int, Path],
    color_map=None,
):
    """Build one class table shared by every tile of a segmentation pass.

    Each frame's identifiers are assigned independently, so identifier 2 can be
    a different class in two neighbouring tiles. Stitching the raw images would
    silently mix them; the tables are unified here instead. BACKGROUND keeps
    identifier 0 so that uncovered canvas and rendered background agree.

    Args:
        image_type: Segmentation image prefix.
        frame_sources: Selected numeric or grid-renamed image per frame.
        color_map: Optional mapping of class names to RGBA identifiers.

    Returns:
        Tuple of the per-frame identifier remaps and the unified class table.

    Raises:
        ValidationError: If a frame's label sidecar is missing or unreadable.

    """
    import json

    per_frame = {}
    prefix = f"{image_type}_"
    for frame, source_path in frame_sources.items():
        suffix = source_path.stem.removeprefix(prefix)
        path = source_path.with_name(f"{image_type}_labels_{suffix}.json")
        if not path.is_file():
            raise ValidationError(
                f"label-aware stitching requires {path.name} in {source_path.parent}",
            )
        try:
            labels = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValidationError(
                f"could not read segmentation labels {path}: {error}",
            ) from error
        per_frame[frame] = {
            _parse_label_identifier(identifier): tuple(
                sorted((str(name), str(value)) for name, value in semantics.items())
            )
            for identifier, semantics in labels.items()
        }

    identifier_types = {
        tuple if isinstance(identifier, tuple) else int
        for labels in per_frame.values()
        for identifier in labels
    }
    if len(identifier_types) > 1:
        raise ValidationError(
            "label-aware stitching cannot mix numeric and colorized identifiers",
        )

    def _rank(key):
        """Order BACKGROUND first, then UNLABELLED, then classes by name."""
        values = dict(key)
        name = values.get("class", "")
        return (0 if name == "BACKGROUND" else 1 if name == "UNLABELLED" else 2, key)

    ordered_keys = sorted(
        {key for keys in per_frame.values() for key in keys.values()},
        key=_rank,
    )
    if tuple in identifier_types:
        unified = _colorized_label_identifiers(ordered_keys, color_map=color_map)
    else:
        unified = {key: identifier for identifier, key in enumerate(ordered_keys)}
    remaps = {
        frame: {source: unified[key] for source, key in keys.items()}
        for frame, keys in per_frame.items()
    }
    table = {str(identifier): dict(key) for key, identifier in unified.items()}
    return remaps, table


def _passthrough_label_table(
    image_type: str,
    frame_sources: dict[int, Path],
) -> dict[str, object]:
    """Merge per-tile label sidecars without remapping pixel identifiers.

    Unlike ``_tile_label_table``, this function keeps the original identifiers
    (integer or RGBA) from each tile.  When two tiles use different identifiers
    for the same class, a ``RuntimeWarning`` is emitted and the stitch proceeds;
    use ``blend='labels'`` to unify colors and suppress that warning.

    Args:
        image_type: Segmentation image prefix.
        frame_sources: Selected numeric or grid-renamed image per frame.

    Returns:
        Merged label table keyed by the original per-tile identifiers.

    Raises:
        ValidationError: If a frame's label sidecar is missing, unreadable, or
            two tiles assign the same identifier to different classes.

    """
    import json

    merged: dict[str, object] = {}
    prefix = f"{image_type}_"
    for source_path in frame_sources.values():
        suffix = source_path.stem.removeprefix(prefix)
        path = source_path.with_name(f"{image_type}_labels_{suffix}.json")
        if not path.is_file():
            raise ValidationError(
                f"write_labels requires {path.name} in {source_path.parent}",
            )
        try:
            labels = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValidationError(
                f"could not read segmentation labels {path}: {error}",
            ) from error
        for key, entry in labels.items():
            existing = merged.get(key)
            if existing is not None and existing != entry:
                raise ValidationError(
                    f"conflicting segmentation label for {key!r} between tiles: "
                    f"{existing!r} vs {entry!r}",
                )
            merged[key] = entry
    seen_semantics: dict[object, str] = {}
    for key, entry in merged.items():
        canonical = json.dumps(entry, sort_keys=True)
        prior_key = seen_semantics.get(canonical)
        if prior_key is not None:
            warnings.warn(
                f"tiles use inconsistent color assignments: identifiers "
                f"{prior_key!r} and {key!r} both map to {entry!r}; "
                "the stitched image will contain the same class in multiple "
                "colors — use blend='labels' to unify colors across tiles",
                RuntimeWarning,
                stacklevel=4,
            )
        else:
            seen_semantics[canonical] = key
    return merged


def stitch_capture_grid(
    capture_root: str | Path,
    trigger_metadata: list[dict[str, object]],
    *,
    image_type: str = "rgb",
    output_subdir: str = "stitched",
    output_filename: str = "rgb.png",
    blend: str = "none",
    write_labels: bool = False,
    recolor_tiles: bool = False,
    color_map: dict | None = None,
) -> tuple[dict[str, object], ...]:
    """Stitch captured scan-grid frames using their canonical pixel layouts.

    Args:
        capture_root: Local root directory produced by one capture pass.
        trigger_metadata: Per-trigger metadata returned by capture execution.
        image_type: BasicWriter image prefix to stitch.
        output_subdir: Directory created beneath the capture root.
        output_filename: Filename used for a single-trigger result.
        blend: ``none`` pastes each tile, so wherever the scan step equals the
            camera footprint the tiles butt-join and the boundary stays visible.
            ``feather`` cross-fades over the overlap the layout implies, which
            requires an aperture wider than the cell pitch.
        write_labels: When ``True`` and ``blend`` is not ``labels``, merge the
            per-tile label sidecars and write a ``*_labels.json`` sidecar
            alongside the stitched image.  The original pixel identifiers are
            preserved; no color remapping occurs.  A ``RuntimeWarning`` is
            emitted when tiles use different identifiers for the same class;
            use ``blend='labels'`` in that case.
        recolor_tiles: When ``True`` with ``blend='labels'``, rewrite each
            source tile and its label sidecar with the unified identifiers used
            by the stitched output.
        color_map: Optional mapping of class name to an RGBA 4-tuple
            ``(R, G, B, A)`` with values in ``[0, 255]``.  Only applies when
            ``blend='labels'``.  Named classes will use the supplied color
            instead of the hash-derived assignment.  BACKGROUND and UNLABELLED
            are always ``(0,0,0,0)`` and ``(0,0,0,255)`` respectively and
            cannot be overridden.

    Returns:
        One artifact dictionary per trigger with local path and canvas size.
        When the trigger's pixel layout carries world-space placement (see
        ``scan_grid_geometry.compute_pixel_layout``), an
        ``image_*_to_usd.json`` sidecar is written beside the stitched image
        with the four USD corner coordinates and an affine pixel-to-USD
        transform, and the artifact carries ``image_to_usd_path``.

    Raises:
        ValidationError: If layout metadata or a required frame is unavailable,
            ``write_labels`` is combined with ``blend='labels'``, a per-tile
            label sidecar is missing or unreadable, or two tiles assign the
            same identifier to different classes.

    Example:

    .. code-block:: python

        artifacts = stitch_capture_grid(
            "output/full_board",
            trigger_metadata,
            output_filename="panel.png",
        )
        print(artifacts[0]["path"])

    """
    try:
        from PIL import Image
    except ImportError as error:
        raise ValidationError(
            "postprocess stitching requires Pillow",
        ) from error
    if recolor_tiles and blend != "labels":
        raise ValidationError("recolor_tiles requires blend='labels'")

    root = Path(capture_root)
    destination = root / output_subdir
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = []
    trigger_count = len(trigger_metadata)
    for metadata in trigger_metadata:
        trigger_index = int(metadata["trigger_idx"])
        layout = metadata.get("pixel_layout")
        if not isinstance(layout, dict) or not layout.get("supported"):
            reason = layout.get("reason") if isinstance(layout, dict) else None
            raise ValidationError(
                "postprocess stitching requires a supported pixel layout"
                + (f": {reason}" if reason else ""),
            )
        canvas_size = layout.get("canvas_size_px")
        tiles = layout.get("tiles")
        if not (
            isinstance(canvas_size, list)
            and len(canvas_size) == 2
            and isinstance(tiles, list)
            and tiles
        ):
            raise ValidationError("pixel layout is missing canvas or tile data")

        if blend not in BLEND_MODES:
            raise ValidationError(
                f"stitch blend must be one of {BLEND_MODES} (got {blend!r})",
            )
        if write_labels and blend == "labels":
            raise ValidationError(
                "write_labels may not be combined with blend='labels' "
                "(labels mode already writes a remapped label sidecar)",
            )
        trigger_dir = root / f"trigger_{trigger_index:04d}"
        frame_images = _frame_image_index(trigger_dir, image_type)
        grid_images = _grid_image_index(trigger_dir, image_type)
        frame_sources = {}
        for tile_spec in tiles:
            if not isinstance(tile_spec, dict):
                raise ValidationError("pixel layout contains an invalid tile")
            frame = int(tile_spec["frame"])
            source_path = frame_images.get(frame)
            if source_path is None:
                x_index = tile_spec.get("x_idx")
                y_index = tile_spec.get("y_idx")
                if x_index is not None and y_index is not None:
                    source_path = grid_images.get((int(x_index), int(y_index)))
            if source_path is None:
                raise ValidationError(
                    f"missing {image_type} image for frame {frame} in {trigger_dir}",
                )
            frame_sources[frame] = source_path
        label_remaps = None
        label_table = None
        if blend == "labels":
            label_remaps, label_table = _tile_label_table(
                image_type,
                frame_sources,
                color_map=color_map,
            )
        passthrough_table = None
        if write_labels:
            passthrough_table = _passthrough_label_table(image_type, frame_sources)
        canvas = None
        accumulator = None
        weights = None
        if blend == "feather":
            import numpy as np

            overlap_x, overlap_y = _tile_overlaps(tiles)
            ramp_x = None
            ramp_y = None

        seen_cells: dict[bytes, tuple[int, object]] = {}
        frame_positions = _frame_positions(metadata)
        for tile_spec in tiles:
            frame = int(tile_spec["frame"])
            source_path = frame_sources[frame]
            rectangle = tile_spec.get("dst_rect_xywh_px")
            if not isinstance(rectangle, list) or len(rectangle) != 4:
                raise ValidationError(f"frame {frame} has no destination rectangle")
            left, top, width, height = (round(float(v)) for v in rectangle)
            with Image.open(source_path) as source:
                tile = source.copy()
            _reject_repeated_cell(
                seen_cells,
                tile,
                frame,
                tile_spec,
                trigger_index,
                frame_positions,
                _tile_semantics(image_type, source_path),
            )

            if blend == "labels":
                import json

                import numpy as np

                remap = label_remaps[frame]
                samples = np.asarray(tile).copy()
                remapped = np.zeros_like(samples)
                for source, target in remap.items():
                    if isinstance(source, tuple):
                        remapped[np.all(samples == source, axis=-1)] = target
                    else:
                        remapped[samples == source] = target
                tile = Image.fromarray(remapped, tile.mode)
                if recolor_tiles:
                    tile.save(source_path)
                    prefix = f"{image_type}_"
                    suffix = source_path.stem.removeprefix(prefix)
                    labels_path = source_path.with_name(
                        f"{image_type}_labels_{suffix}.json",
                    )
                    tile_table = {
                        str(target): label_table[str(target)]
                        for target in set(remap.values())
                    }
                    labels_path.write_text(
                        json.dumps(tile_table, indent=2, sort_keys=True) + chr(10),
                        encoding="utf-8",
                    )

            if tile.size != (width, height):
                # Class identifiers must not be interpolated into new classes.
                resampling = (
                    Image.Resampling.NEAREST
                    if blend == "labels"
                    else Image.Resampling.LANCZOS
                )
                tile = tile.resize((width, height), resampling)

            if blend == "labels":
                if canvas is None:
                    canvas = Image.new(
                        tile.mode,
                        (int(canvas_size[0]), int(canvas_size[1])),
                    )
                canvas.paste(tile, (left, top))
                continue

            if blend == "none":
                if canvas is None:
                    canvas = Image.new(
                        tile.mode,
                        (int(canvas_size[0]), int(canvas_size[1])),
                    )
                canvas.paste(tile, (left, top))
                continue

            samples = np.asarray(tile, dtype=np.float32)
            if samples.ndim == 2:
                samples = samples[:, :, None]
            if accumulator is None:
                accumulator = np.zeros(
                    (int(canvas_size[1]), int(canvas_size[0]), samples.shape[2]),
                    np.float32,
                )
                weights = np.zeros(
                    (int(canvas_size[1]), int(canvas_size[0])),
                    np.float32,
                )
                mode = tile.mode
                ramp_y = _feather_ramp(height, overlap_y)
                ramp_x = _feather_ramp(width, overlap_x)
            tile_weight = np.outer(ramp_y, ramp_x)
            right = min(left + width, accumulator.shape[1])
            bottom = min(top + height, accumulator.shape[0])
            span_x, span_y = right - left, bottom - top
            accumulator[top:bottom, left:right] += (
                samples[:span_y, :span_x] * tile_weight[:span_y, :span_x, None]
            )
            weights[top:bottom, left:right] += tile_weight[:span_y, :span_x]

        if blend == "feather":
            if accumulator is None:
                raise ValidationError("pixel layout contains no stitchable tiles")
            covered = weights > 1e-6
            resolved = np.zeros_like(accumulator)
            resolved[covered] = accumulator[covered] / weights[covered][:, None]
            resolved = np.clip(resolved, 0.0, 255.0).astype("uint8")
            if resolved.shape[2] == 1:
                resolved = resolved[:, :, 0]
            canvas = Image.fromarray(resolved, mode)

        if canvas is None:
            raise ValidationError("pixel layout contains no stitchable tiles")
        canvas, canvas_size, crop_offset_px = _crop_to_image_size(
            canvas, layout.get("image_size_px")
        )
        filename = _trigger_output_name(
            output_filename,
            trigger_index,
            trigger_count,
        )
        output_path = destination / filename
        canvas.save(output_path)
        labels_path = None
        if blend == "labels":
            import json

            labels_path = output_path.with_name(f"{output_path.stem}_labels.json")
            labels_path.write_text(
                json.dumps(label_table, indent=2, sort_keys=True) + chr(10),
                encoding="utf-8",
            )
        elif passthrough_table is not None:
            import json

            labels_path = output_path.with_name(f"{output_path.stem}_labels.json")
            labels_path.write_text(
                json.dumps(passthrough_table, indent=2, sort_keys=True) + chr(10),
                encoding="utf-8",
            )
        image_to_usd_path = _write_image_to_usd_sidecar(
            output_path, layout, canvas_size, crop_offset_px=crop_offset_px
        )
        artifacts.append(
            {
                "trigger_idx": trigger_index,
                "path": str(output_path),
                **({"labels_path": str(labels_path)} if labels_path else {}),
                **(
                    {"image_to_usd_path": str(image_to_usd_path)}
                    if image_to_usd_path
                    else {}
                ),
                "canvas_size_px": [int(canvas_size[0]), int(canvas_size[1])],
                "tile_count": len(tiles),
            },
        )
    return tuple(artifacts)


def _crop_to_image_size(canvas, image_size_px):
    """Crop a stitched mosaic to the size the capture stated, centred.

    Cells are rendered at one whole-pixel resolution, so a mosaic of them can be
    larger than the image a scanner would produce from the same rectangle -- by
    under one pixel per cell along an axis. The surplus is split between the two
    edges (the extra pixel, if odd, on the right/bottom), so the image stays
    centred on the requested rectangle to within half a pixel.

    Returns:
        ``(canvas, [width, height], (left, top))`` -- the left/top crop offset is
        what the pixel-to-USD affine must be moved by.

    Raises:
        ValidationError: If the stated size is larger than the mosaic, which
            would mean the cells do not cover the image they were derived from.

    """
    width, height = canvas.size
    if not image_size_px:
        return canvas, [width, height], (0, 0)
    target_w, target_h = int(image_size_px[0]), int(image_size_px[1])
    if target_w > width or target_h > height:
        raise ValidationError(
            f"stitched mosaic {width}x{height} is smaller than the stated image "
            f"size {target_w}x{target_h}; the cells do not cover it",
        )
    if (target_w, target_h) == (width, height):
        return canvas, [width, height], (0, 0)
    left = (width - target_w) // 2
    top = (height - target_h) // 2
    cropped = canvas.crop((left, top, left + target_w, top + target_h))
    return cropped, [target_w, target_h], (left, top)


def _fov_from_layout(layout, canvas_size) -> dict | None:
    """State the capture the way a scanner is specified.

    The mapping above answers "where is this pixel"; this answers "what
    rectangle of board did we scan, and at what resolution" -- the form the
    customer's AOI is configured in (a scan area in millimetres at a dpi). It is
    derived from the same layout, so the two cannot disagree.

    The pitch comes from the affine's own per-pixel world vectors rather than
    from the world-axis footprints. Those two disagree the moment the camera
    carries a fixed yaw of +-90 deg: the world-X extent then spans the canvas's
    HEIGHT, and dividing it by the canvas width reports a scan area that is
    right in magnitude and wrong in axis -- 750 x 480 dpi from a capture that is
    600 dpi square. Reading the pitch off the affine is rotation-free, and keeps
    this block and the mapping above answering with one set of numbers.

    Returns ``None`` when the stage scale is unknown, since without it a pixel
    pitch cannot be put in millimetres and a guessed dpi is worse than none.
    """
    mpu = layout.get("stage_meters_per_unit")
    mapping = layout.get("pixel_to_world_xy") or {}
    u_per_px = mapping.get("u_per_px_xy")
    v_per_px = mapping.get("v_per_px_xy")
    if not mpu or not u_per_px or not v_per_px:
        return None
    try:
        width_px, height_px = int(canvas_size[0]), int(canvas_size[1])
        mm_per_su = float(mpu) * 1000.0
        # Magnitude of one pixel step along each image axis, in world space.
        mm_per_px_u = math.hypot(float(u_per_px[0]), float(u_per_px[1])) * mm_per_su
        mm_per_px_v = math.hypot(float(v_per_px[0]), float(v_per_px[1])) * mm_per_su
    except (TypeError, ValueError):
        return None
    if width_px <= 0 or height_px <= 0 or mm_per_px_u <= 0 or mm_per_px_v <= 0:
        return None
    area_u_mm = mm_per_px_u * width_px
    area_v_mm = mm_per_px_v * height_px
    grid = {}
    if layout.get("x_num") and layout.get("y_num"):
        grid = {"x_num": int(layout["x_num"]), "y_num": int(layout["y_num"])}
    centered_on = layout.get("scan_area_centered_on")
    center_mm = layout.get("scan_center_mm")
    return {
        # What the capture said the rectangle was centred on. Not derivable
        # from the layout -- a rectangle stated by its size sits on the bbox
        # midpoint whichever reading was meant -- so it is carried from the
        # declaration, and is ``None`` when the capture did not say.
        "centered_on": str(centered_on) if centered_on else None,
        # Where a rectangle stated by its corners was placed, world XY in
        # millimetres; ``None`` when it was centered on the board. The affine
        # already carries the placement; this says it was requested.
        "center_mm": (
            [float(center_mm[0]), float(center_mm[1])] if center_mm else None
        ),
        # Along the image axes, which is how a scanner states it: the width of
        # the picture first. Not world X and Y -- see the docstring.
        "scan_area_mm": [area_u_mm, area_v_mm],
        "dpi": [25.4 / mm_per_px_u, 25.4 / mm_per_px_v],
        "mm_per_px": [mm_per_px_u, mm_per_px_v],
        "axes": "image",
        "grid": grid,
        # Relative, not absolute: the pitch is recovered through the affine,
        # so a genuinely square 600 dpi capture lands a few parts in ten
        # million apart. An exact comparison calls that anisotropic.
        "square_pixels": abs(mm_per_px_u - mm_per_px_v)
        <= 1e-6 * max(mm_per_px_u, mm_per_px_v),
    }


def _write_image_to_usd_sidecar(
    output_path: Path,
    layout: dict[str, object],
    canvas_size: list,
    *,
    crop_offset_px: tuple[int, int] = (0, 0),
) -> Path | None:
    """Write a pixel-to-USD sidecar beside a stitched image, when derivable.

    ``pixel_to_world_xy`` and ``world_reference_z`` are populated by
    :func:`scan_grid_geometry.compute_pixel_layout` and capture execution
    respectively (see ``cdf/execution/isaac/capture_execution.py``); older
    metadata without them silently skips the sidecar.

    ``reference_plane_z`` is the bounding-box face (top or bottom, per
    ``camera_side``) of one representative board instance when the capture
    has a board-instance cohort (``reference_plane_source ==
    "board_instance"``) -- the board itself, not a surrounding carrier/panel
    fixture. Single-board scenes without a cohort fall back to the capture
    root's own bbox face (``reference_plane_source == "capture_root"``),
    which is only as precise as that root's scope.

    Args:
        output_path: Path the stitched image was saved to.
        layout: This trigger's ``pixel_layout`` metadata.
        canvas_size: ``[width, height]`` of the stitched canvas in pixels.
        crop_offset_px: ``(left, top)`` of the delivered image inside the
            mosaic, when the stitch cropped it to a stated image size.

    Returns:
        Path to the written sidecar, or ``None`` when the layout lacks the
        world-space fields required to derive it.

    """
    affine = layout.get("pixel_to_world_xy")
    reference_z = layout.get("world_reference_z")
    if not isinstance(affine, dict) or reference_z is None:
        return None

    import json

    origin = affine["origin_xy"]
    u_per_px = affine["u_per_px_xy"]
    v_per_px = affine["v_per_px_xy"]

    # The affine comes from the UNROUNDED tile layout, but the stitcher rounds
    # each destination rectangle before pasting. With fractional tile offsets the
    # two disagree by up to half a pixel, and the sidecar would then describe a
    # placement the image does not have. Re-anchor on the rounding the stitcher
    # actually applied to the first tile, so the transform describes the pixels
    # that were written.
    tiles = layout.get("tiles")
    if isinstance(tiles, list) and tiles:
        first = tiles[0].get("dst_rect_xywh_px")
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            shift_px = (
                round(float(first[0])) - float(first[0]),
                round(float(first[1])) - float(first[1]),
            )
            origin = [
                origin[0] + shift_px[0] * u_per_px[0] + shift_px[1] * v_per_px[0],
                origin[1] + shift_px[0] * u_per_px[1] + shift_px[1] * v_per_px[1],
            ]
    # A cropped image starts crop_offset_px into the mosaic the affine describes.
    crop_left, crop_top = (float(crop_offset_px[0]), float(crop_offset_px[1]))
    if crop_left or crop_top:
        origin = [
            origin[0] + crop_left * u_per_px[0] + crop_top * v_per_px[0],
            origin[1] + crop_left * u_per_px[1] + crop_top * v_per_px[1],
        ]
    z = float(reference_z)
    width_px, height_px = float(canvas_size[0]), float(canvas_size[1])

    def _corner(px: float, py: float) -> dict[str, list[float]]:
        return {
            "pixel_xy": [px, py],
            "usd_xyz": [
                origin[0] + px * u_per_px[0] + py * v_per_px[0],
                origin[1] + px * u_per_px[1] + py * v_per_px[1],
                z,
            ],
        }

    sidecar = {
        # This file is an external contract, so it says what it is rather than
        # leaving a consumer to infer it from the field names.
        "schema_version": "pcb_aoi.image_to_usd/1.0.0",
        # Coordinates are USD STAGE (world) space, in stage units -- not metres,
        # not millimetres. stage_meters_per_unit below converts.
        "coordinate_space": "usd_stage_world",
        "units": "usd_stage_units",
        # The scanner vendor's proposed example uses a lower-left [0, 0] origin. This file
        # does NOT: it is raster-style, top-left. The difference is a row flip
        # and is called out here because getting it wrong is silent.
        # Pixel (0, 0) is the image's top-left corner, +x right, +y down --
        # the same raster convention PNG/PIL/numpy use. A consumer assuming
        # a bottom-left, y-up origin must flip the row index first.
        "pixel_coordinate_system": "top-left-xy-pixels",
        # Pixel-edge, not pixel-center: integer coordinates address the grid
        # lines between samples (column 0 is the left edge of the first
        # pixel), matching dst_rect_xywh_px / the affine below. The center
        # of pixel (col, row) is at (col + 0.5, row + 0.5).
        "pixel_indexing_convention": "pixel-edge",
        "image_size_px": [int(canvas_size[0]), int(canvas_size[1])],
        # compute_pixel_layout only ever reports "supported" for an
        # orthographic camera (see scan_grid_geometry.compute_pixel_layout),
        # so this is currently always "orthographic" -- stated explicitly
        # rather than left implicit, since a perspective camera would need
        # camera parameters instead of a flat reference plane.
        "projection": "orthographic",
        "camera_side": layout.get("camera_side"),
        "reference_plane_source": layout.get("reference_plane_source"),
        "stage_meters_per_unit": layout.get("stage_meters_per_unit"),
        # Whether that scale is the stage's own or the job's. A consumer that
        # reads the stage back and gets a different number needs to know which
        # of the two this mapping was built on.
        "stage_meters_per_unit_source": layout.get("stage_meters_per_unit_source"),
        "stage_up_axis": layout.get("stage_up_axis"),
        "source_usd": layout.get("source_usd"),
        # A path names a file, not a build; two exports a week apart share it.
        "source_usd_provenance": layout.get("source_usd_provenance"),
        # Which board placements this frame contains, so a consumer reading a
        # stitched panel does not have to work out which step-repeat it is
        # looking at. Absent (not empty) when the scene carries a single board.
        "instances_in_frame": layout.get("board_instances"),
        "reference_plane_z": z,
        # reference_plane_z is a MEASURED PLANE, and which measurement is stated
        # by reference_plane_source: "substrate_z_range" is the PCB surface from
        # the metadata catalogue; the bbox routes are an axis-aligned minimum
        # over whatever the scope contains, so under a tall part or a long pin
        # they track that part and are BOM-dependent. Read the source before
        # treating this as a board face.
        # When the affine and the corners disagree -- they are derived from the
        # same layout and should not -- the AFFINE is authoritative. The corners
        # are a convenience evaluation of it at the four extremes.
        "authoritative_mapping": "affine_pixel_to_usd",
        "corners_usd": {
            "top_left": _corner(0.0, 0.0),
            "top_right": _corner(width_px, 0.0),
            "bottom_left": _corner(0.0, height_px),
            "bottom_right": _corner(width_px, height_px),
        },
        "affine_pixel_to_usd": {
            "origin_xy": list(origin),
            # u/v read as texture coordinates to anyone who meets them cold.
            # These are world displacement per pixel step, so they say so. The
            # u_per_px_xy / v_per_px_xy spellings are kept as aliases because
            # two sample packs already shipped carrying them.
            "world_per_px_x": list(u_per_px),
            "world_per_px_y": list(v_per_px),
            "u_per_px_xy": list(u_per_px),
            "v_per_px_xy": list(v_per_px),
        },
    }
    fov = _fov_from_layout(layout, canvas_size)
    if fov is not None:
        sidecar["fov"] = fov
    # Named with the prefix first, not the image stem first. A consumer's file
    # filter is the thing that has to find this: PPX selects Modus AOI output
    # with `*/augmentation_restitch/artifact/image`, and a sidecar called
    # `<stem>_image_to_usd.json` falls outside a filter that its own image
    # matches. Asked for by the PPX side; the contents are unchanged.
    sidecar_path = output_path.with_name(f"image_{output_path.stem}_to_usd.json")
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    return sidecar_path
