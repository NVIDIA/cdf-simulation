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

"""Render each scan cell with a guard band and crop it before anything is written.

The RTX renderer's outermost pixel column and row of a frame are not
trustworthy: measured on every scan tile, the right column comes out at
0.74-0.89x the interior brightness (the column before it 1.04-1.09x) and the
bottom row at about 0.90x. Cells of a field-of-view grid abut with zero
overlap, so every seam of the stitched image carries that defect as a dark
line.

The fix is to render ``guard_px`` extra pixels beyond every edge -- the camera
aperture grown by exactly that many pixel pitches, the cell centre unchanged --
and crop them off again *in the writer*, before the per-tile outputs are
encoded. The tiles on disk, the pixel layout and the calibration then have the
shape they always had, which matters because consumers outside this repository
stitch the raw tiles themselves from ``pixel_layout`` and validate tile size
against it.

Every annotator the writer receives is cropped the same way: images and
per-pixel arrays are sliced, 2D boxes are shifted by ``-guard_px``, clipped to
the delivered tile and dropped when they fall wholly in the band. Outputs that
cannot be cropped to an equivalent of the unguarded frame (3D boxes, occlusion,
point clouds, skeletons, camera parameters) are refused up front rather than
written to describe a frame that was never delivered.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

#: Writer annotators whose payload is one value per pixel, ``(height, width,
#: ...)``; cropped by slicing. ``Aug``-prefixed annotators are augmented colour.
PIXEL_ANNOTATORS = frozenset(
    {
        "rgb",
        "normals",
        "distance_to_camera",
        "distance_to_image_plane",
        "semantic_segmentation",
        "instance_id_segmentation",
        "instance_segmentation",
        "motion_vectors",
    }
)

#: Writer annotators whose payload is a list of 2D pixel boxes; shifted,
#: clipped and filtered.
BOX_ANNOTATORS = frozenset({"bounding_box_2d_tight", "bounding_box_2d_loose"})

#: Entries the writer receives but never writes; left as they are.
PASS_THROUGH_ANNOTATORS = frozenset({"camera", "resolution"})

#: BasicWriter outputs with no faithful crop: they describe the whole rendered
#: frame (or the camera that rendered it) in terms a pixel crop cannot restate.
UNSUPPORTED_OUTPUTS = frozenset(
    {
        "bounding_box_3d",
        "occlusion",
        "pointcloud",
        "skeleton_data",
        "camera_params",
    }
)

#: Per-box arrays that travel beside a box payload and must be filtered with it.
_BOX_PARALLEL_KEYS = ("primPaths", "bboxIds")

#: Image files whose pixel size is checked against the delivered tile.
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp"})


class GuardBandError(RuntimeError):
    """A per-tile output that cannot be cropped to the delivered tile."""


def render_resolution(resolution: Sequence[int], guard_px: int) -> tuple[int, int]:
    """Return the size rendered for a delivered ``resolution`` with a guard band.

    Args:
        resolution: Delivered tile ``(width, height)`` in pixels.
        guard_px: Pixels rendered beyond every edge.

    Returns:
        ``(width + 2 * guard_px, height + 2 * guard_px)``.

    Example:

    .. code-block:: python

        print(render_resolution((4060, 3248), 8))
        (4076, 3264)

    """
    return (
        int(resolution[0]) + 2 * int(guard_px),
        int(resolution[1]) + 2 * int(guard_px),
    )


def render_aperture(aperture: float, delivered_px: int, guard_px: int) -> float:
    """Widen an orthographic aperture by ``2 * guard_px`` of its own pixel pitches.

    The pitch is ``aperture / delivered_px``; keeping it while adding pixels on
    both sides keeps the delivered tile's footprint, centre and scale exact.

    Args:
        aperture: Aperture spanning ``delivered_px`` pixels, in any unit.
        delivered_px: Pixels the delivered tile has along this axis.
        guard_px: Pixels rendered beyond each edge.

    Returns:
        The aperture to render, in the unit of ``aperture``.

    Raises:
        ValueError: If ``delivered_px`` is not positive.

    Example:

    .. code-block:: python

        print(render_aperture(100.0, 1000, 10))
        102.0

    """
    if int(delivered_px) <= 0:
        raise ValueError(f"delivered_px must be positive, got {delivered_px!r}")
    if not guard_px:
        return float(aperture)
    return float(aperture) * (int(delivered_px) + 2 * int(guard_px)) / int(delivered_px)


def unsupported_outputs(writer_config: Mapping[str, object]) -> list[str]:
    """Name the enabled writer outputs a guard band cannot crop.

    Args:
        writer_config: BasicWriter keyword arguments for the pass.

    Returns:
        Sorted names of enabled outputs in :data:`UNSUPPORTED_OUTPUTS`.

    Example:

    .. code-block:: python

        print(unsupported_outputs({"rgb": True, "bounding_box_3d": True}))
        ['bounding_box_3d']

    """
    return sorted(name for name in UNSUPPORTED_OUTPUTS if writer_config.get(name))


def crop_pixels(
    array: np.ndarray, guard_px: int, render_size: Sequence[int]
) -> np.ndarray:
    """Crop ``guard_px`` from every side of one per-pixel annotator array.

    Args:
        array: Annotator data shaped ``(height, width, ...)``.
        guard_px: Pixels to remove from each edge.
        render_size: Rendered ``(width, height)`` the array must have.

    Returns:
        A contiguous copy of the delivered region.

    Raises:
        GuardBandError: If ``array`` is not an array of the rendered size.

    Example:

    .. code-block:: python

        tile = crop_pixels(np.zeros((20, 30, 4), np.uint8), 2, (30, 20))
        print(tile.shape)
        (16, 26, 4)

    """
    width, height = int(render_size[0]), int(render_size[1])
    if not isinstance(array, np.ndarray) or array.ndim < 2:
        raise GuardBandError(
            f"expected a (height, width, ...) array, got {type(array).__name__}"
        )
    if tuple(array.shape[:2]) != (height, width):
        raise GuardBandError(
            f"annotator array is {array.shape[1]}x{array.shape[0]}, "
            f"rendered frame is {width}x{height}"
        )
    if not guard_px:
        return array
    return np.ascontiguousarray(
        array[guard_px : height - guard_px, guard_px : width - guard_px]
    )


def crop_boxes(
    boxes: np.ndarray,
    guard_px: int,
    render_size: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Move 2D pixel boxes from the rendered frame into the delivered tile.

    Box corners are inclusive pixel indices (Replicator's convention: a box
    centred in a 1280-pixel frame spans ``301..978``). Each box is shifted by
    ``-guard_px``, clipped to ``[0, size - 1]`` -- the bound the renderer
    itself clips to at a frame edge -- and dropped when no part of it lies
    inside the delivered tile.

    Args:
        boxes: Structured array with ``x_min``, ``y_min``, ``x_max`` and
            ``y_max`` fields.
        guard_px: Pixels removed from each edge.
        render_size: Rendered ``(width, height)``.

    Returns:
        The kept boxes, and the boolean mask over the input that selected
        them, for filtering arrays that run parallel to the boxes.

    Raises:
        GuardBandError: If ``boxes`` lacks the corner fields.

    Example:

    .. code-block:: python

        dtype = [("x_min", "<i4"), ("y_min", "<i4"), ("x_max", "<i4"), ("y_max", "<i4")]
        kept, mask = crop_boxes(np.array([(0, 0, 3, 3), (10, 10, 12, 12)], dtype), 2, (20, 20))
        print(kept.tolist(), mask.tolist())
        [(0, 0, 1, 1), (8, 8, 10, 10)] [True, True]

    """
    fields = ("x_min", "y_min", "x_max", "y_max")
    names = getattr(getattr(boxes, "dtype", None), "names", None) or ()
    if not all(name in names for name in fields):
        raise GuardBandError(f"box payload lacks {fields}, has fields {names}")
    delivered_w = int(render_size[0]) - 2 * int(guard_px)
    delivered_h = int(render_size[1]) - 2 * int(guard_px)
    shifted = boxes.copy()
    for name in fields:
        shifted[name] = shifted[name] - guard_px
    keep = (
        (shifted["x_max"] >= 0)
        & (shifted["y_max"] >= 0)
        & (shifted["x_min"] <= delivered_w - 1)
        & (shifted["y_min"] <= delivered_h - 1)
    )
    kept = shifted[keep]
    kept["x_min"] = np.clip(kept["x_min"], 0, delivered_w - 1)
    kept["x_max"] = np.clip(kept["x_max"], 0, delivered_w - 1)
    kept["y_min"] = np.clip(kept["y_min"], 0, delivered_h - 1)
    kept["y_max"] = np.clip(kept["y_max"], 0, delivered_h - 1)
    return kept, keep


def _crop_box_payload(
    payload: Mapping[str, object],
    guard_px: int,
    render_size: Sequence[int],
) -> dict[str, object]:
    """Crop one box annotator payload and every array that runs parallel to it."""
    boxes = payload.get("data")
    if not isinstance(boxes, np.ndarray):
        raise GuardBandError(
            f"box payload data is {type(boxes).__name__}, not an array"
        )
    kept, keep = crop_boxes(boxes, guard_px, render_size)
    cropped = {**payload, "data": kept}
    for key in _BOX_PARALLEL_KEYS:
        values = payload.get(key)
        if values is None:
            continue
        if len(values) != len(boxes):
            raise GuardBandError(
                f"box payload {key!r} has {len(values)} entries for {len(boxes)} boxes"
            )
        if isinstance(values, np.ndarray):
            cropped[key] = values[keep]
        else:
            cropped[key] = [value for value, kept_box in zip(values, keep) if kept_box]
    return cropped


def crop_writer_payload(
    data: Mapping[str, object],
    guard_px: int,
    render_size: Sequence[int],
) -> dict[str, object]:
    """Crop every annotator in one writer ``write()`` payload to the delivered tile.

    The payload is not modified; Replicator caches it for re-delivery.

    Args:
        data: The mapping Replicator passes to ``Writer.write``, with
            ``data["annotators"][name][render_product]`` per annotator.
        guard_px: Pixels removed from each edge.
        render_size: Rendered ``(width, height)``.

    Returns:
        A payload of the same structure describing the delivered tile.

    Raises:
        GuardBandError: If an annotator has no faithful crop or its data does
            not have the rendered size.

    """
    annotators = data.get("annotators")
    if not isinstance(annotators, Mapping):
        raise GuardBandError("writer payload carries no annotators mapping")
    cropped_annotators: dict[str, object] = {}
    for name, per_product in annotators.items():
        base = name.removesuffix("_fast")
        if base in PASS_THROUGH_ANNOTATORS:
            cropped_annotators[name] = per_product
            continue
        if base in PIXEL_ANNOTATORS or base.startswith("Aug"):
            kind = "pixels"
        elif any(base.startswith(box) for box in BOX_ANNOTATORS):
            kind = "boxes"
        else:
            raise GuardBandError(
                f"annotator {name!r} has no guard-band crop; render it with guard_px=0"
            )
        cropped_products = {}
        for product, payload in per_product.items():
            if not isinstance(payload, Mapping):
                raise GuardBandError(f"annotator {name!r} payload is not a mapping")
            if kind == "pixels":
                cropped_products[product] = {
                    **payload,
                    "data": crop_pixels(payload.get("data"), guard_px, render_size),
                }
            else:
                cropped_products[product] = _crop_box_payload(
                    payload, guard_px, render_size
                )
        cropped_annotators[name] = cropped_products
    return {**data, "annotators": cropped_annotators}


def install_writer_crop(
    writer: object, guard_px: int, render_size: Sequence[int]
) -> None:
    """Make an initialized writer crop every payload before it encodes it.

    Wraps the instance's own ``write`` rather than subclassing: BasicWriter
    records its class name in the ``metadata.txt`` it writes beside the frames,
    so a subclass would change an output this feature promises to leave alone.
    Replicator calls ``self.write(data)``, which finds the instance attribute.

    A payload that cannot be cropped is not written at all -- an exception
    raised inside a writer node is logged by Replicator and otherwise lost, so
    it is recorded on ``writer.guard_band_errors`` for the capture to raise once
    the pass has flushed (see :func:`raise_writer_crop_errors`).

    Args:
        writer: Writer instance, after ``initialize``.
        guard_px: Pixels removed from each edge.
        render_size: Rendered ``(width, height)``.

    """
    original: Callable[[Mapping[str, object]], object] = writer.write
    errors: list[str] = []

    def write(data):
        try:
            cropped = crop_writer_payload(data, guard_px, render_size)
        except GuardBandError as exc:
            errors.append(str(exc))
            return None
        return original(cropped)

    writer.guard_band_errors = errors
    writer.write = write


def raise_writer_crop_errors(writer: object, pass_dir: str | Path) -> None:
    """Fail the pass if its writer refused any payload.

    Args:
        writer: Writer prepared by :func:`install_writer_crop`.
        pass_dir: Directory the pass wrote into, for the message.

    Raises:
        GuardBandError: If any payload could not be cropped.

    """
    errors = getattr(writer, "guard_band_errors", None)
    if isinstance(errors, list) and errors:
        raise GuardBandError(
            f"guard band: {len(errors)} frame(s) in {pass_dir} were not written: "
            + "; ".join(sorted(set(errors)))
        )


def require_delivered_size(
    paths: Sequence[str | Path],
    resolution: Sequence[int],
) -> int:
    """Check every image and per-pixel array output has the delivered tile size.

    A crop that silently failed would hand a consumer tiles larger than the
    ``dst_rect`` it validates them against; this reads each file's header and
    refuses the pass instead.

    Args:
        paths: Output files of one pass.
        resolution: Delivered ``(width, height)``.

    Returns:
        The number of files checked.

    Raises:
        GuardBandError: If a checked file has another size.

    """
    from PIL import Image

    width, height = int(resolution[0]), int(resolution[1])
    checked = 0
    for raw in paths:
        path = Path(raw)
        suffix = path.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            with Image.open(path) as image:
                size = tuple(image.size)
        elif suffix == ".npy" and any(
            path.name.startswith(name) for name in PIXEL_ANNOTATORS
        ):
            shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
            size = (shape[1], shape[0]) if len(shape) >= 2 else tuple(shape)
        else:
            continue
        checked += 1
        if size != (width, height):
            raise GuardBandError(
                f"{path.name} is {size[0]}x{size[1]} px, the delivered tile is "
                f"{width}x{height}; the guard band was not cropped"
            )
    return checked
