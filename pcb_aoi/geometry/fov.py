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
"""Resolve an AOI field of view into a scan grid, apertures and a resolution.

An AOI user specifies a capture the way a scanner is specified -- *a rectangle
of board in millimetres, at a scanning resolution in dpi*. Everything this
pipeline consumes is expressed the way a USD camera is specified -- apertures in
scene units, a pixel resolution, and a cell count. This module is the one place
that converts between the two, so neither side computes pixel counts
independently and drifts from the other.

Common AOI line-scanner settings, used as defaults here:

* 600 dpi by default.
* The scan area is set to a rectangle that contains the product -- that
  rectangle is the field of view.
* A typical sensor's physical resolution is 1200 dpi.
* Common image resolutions are 600 dpi or 1200 dpi --
  25.4 mm / 600 px = 0.042333 mm/px and 25.4 mm / 1200 px = 0.021167 mm/px.
  1200 is rarely used.
* An importer may assume a fixed 600 dpi, which is why an image's size and its
  pixel pitch have to be stated together: a frame of the right pixel count at
  the wrong pitch lands at the wrong scale.

The scanner can equally be described by its pixel pitch, so the resolution may
be given as ``dpi`` or as ``mm_per_px``; the two are the same quantity and only
one is accepted at a time.

Orthographic only, which is what makes this arithmetic exact: an orthographic
camera's world footprint **is** its aperture, independent of camera distance, so
millimetres and pixels relate through one scale factor with no projection term.
A perspective capture would need the camera height as well and is rejected
rather than approximated.

Assumptions
-----------
Recorded here, and echoed into every result under ``assumptions``, because they
depend on the scanner setup and a silent guess in a coordinate contract is
expensive. Each is a keyword argument, so a different setup changes a call site,
not this module.

1. **The board is centred in the scan area** unless the caller says
   otherwise. For a multi-up panel it is unspecified whether "the board" means the
   panel or one placement; ``centre_xy_su`` takes the centre explicitly so the
   caller decides, and ``centered_on`` records which was meant.
2. **Pixels are square.** Not an assumption about the scanner so much as a
   fact about the renderer: an orthographic RTX camera derives its vertical
   extent from the horizontal aperture and the image aspect and ignores the
   vertical aperture, so a non-square pitch cannot be rendered -- asking for
   one delivered the horizontal pitch on both axes, a sawtooth inside every
   tile, a step at every row seam, and a calibration claiming ``fx != fy``.
   AOI line scanners image at one pitch on both axes as well. So a ``dpi`` or
   ``mm_per_px`` pair must be equal (to a part in a million) or it is
   refused, and an ``image_size_px`` is delivered at **one** pitch: the coarser
   of ``area / px`` on the two axes, so the requested rectangle stays fully
   covered and the other axis covers more than was asked, centred on the
   requested rectangle. ``scan_area_mm`` stays the request;
   ``scan_area_mm_effective`` is what the image covers.
3. **The sample pitch is primary, and the image size is computed once.** A scan
   area is rarely an exact pixel multiple. Rounding the pixel count and then
   dividing it back into the requested millimetres makes the X and Y pitches
   differ by a part in ten thousand -- non-square pixels from an isotropic
   request. Instead the pitch is fixed at ``25.4 / dpi`` and the pixel count is
   ``area / pitch`` rounded once per axis by :func:`image_pixels` -- the rule the
   scanner applies, so an image of the size the scanner would produce is the
   one delivered. What is delivered is reported as ``scan_area_mm_effective``.
4. **Tiling splits the area, not the resolution, and never changes the size.**
   When the canvas exceeds ``max_tile_px`` the area is divided into equal cells,
   each rendered at its own resolution. Cells are whole pixels, so the rendered
   mosaic (``mosaic_size_px``) can exceed the image by under one cell-pixel per
   cell; the stitched canvas is cropped back to ``image_size_px``, keeping the
   requested centre. Cell counts are chosen as the smallest that fit.
5. **Each cell renders a guard band that is cropped before the tile is
   written.** The renderer's outermost row and column of every frame are not
   trustworthy (measured: right column 0.74-0.89x, bottom row ~0.90x of the
   interior), and butt-joined cells put those edges on every seam. So each cell
   is rendered ``guard_px`` larger on every side, with its aperture widened by
   exactly ``2 * guard_px`` pixel pitches per axis and its centre unchanged,
   and the extra pixels are cropped from every per-tile output before it lands
   on disk. Tiles, pixel layout and calibration describe the delivered tile, so
   nothing downstream can see the guard; ``guard_px = 0`` renders the tile
   as-is.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

MM_PER_INCH = 25.4

#: Typical default scanning resolution.
DEFAULT_DPI = 600.0

#: Pixels rendered beyond every edge of a cell and cropped before the tile is
#: written (assumption 5). Several times the measured one-pixel edge defect, and
#: small next to a 4096-pixel cell.
DEFAULT_GUARD_PX = 8

#: A typical upper bound for a flatbed scanner sensor. Not enforced: a
#: simulated capture can sample finer, and a request above it is reported so
#: the reader knows a physical scanner would have to resample to reach it.
SENSOR_DPI = 1200.0

#: Common scanner image resolutions. A request outside this set renders fine
#: here but is unusual for a physical scanner, which is reported rather than
#: refused.
SUPPORTED_DPI = (600.0, 1200.0)

#: Relative tolerance for "this dpi is one the machine offers": a pitch given
#: as 0.042333 mm/px is 600 dpi, not 600.005 dpi.
_DPI_MATCH_TOLERANCE = 1e-4

#: Relative tolerance under which a ``dpi`` or ``mm_per_px`` pair is one value
#: written twice: [600, 600.0000001] is isotropic, [600, 600.1] is not.
_ISOTROPIC_TOLERANCE = 1e-6

#: Relative tolerance on rendered pixel squareness. The renderer keeps the
#: horizontal pitch on both axes, so a vertical pitch off by this fraction
#: drifts by ``tolerance * height`` pixels across one frame -- 0.14 px over a
#: 1406 px tile, below anything measurable; the bug this guards against was
#: 5e-3.
SQUARE_PIXEL_TOLERANCE = 1e-4

#: What ``centered_on`` is allowed to say. Mirrors the YAML config's
#: ``Literal["panel", "instance", "unspecified"]`` so a direct call to
#: :func:`resolve_fov` cannot record a label no sidecar consumer recognizes.
VALID_CENTERED_ON = frozenset({"panel", "instance", "unspecified"})


class FovError(ValueError):
    """A field of view that cannot be resolved into a capture."""


def _snap(count: float) -> float:
    """``count`` snapped to an integer when it is one up to float noise.

    25.4 mm at 600 dpi is 600.0000000000001 (or 599.9999999999999) in binary
    floating point; rounding that up or down as written is out by a pixel on
    every request that divides exactly.
    """
    nearest = round(count)
    if abs(count - nearest) <= 1e-9 * max(1.0, abs(count)):
        return float(nearest)
    return count


#: How an area that is not a whole number of pixels becomes one: down. A scanner
#: acquires whole pixels inside its field of view, so a partial edge pixel is
#: not part of the image; down also never delivers pixels outside the requested
#: rectangle, and the shortfall is under one pixel (100 mm at 600 dpi is
#: 2362 px).
_IMAGE_PIXEL_RULE = math.floor


def image_pixels(count: float) -> int:
    """The scanner's pixel count for ``count`` (``area / pitch``).

    Raises:
        FovError: The dimension is smaller than one pixel. Rounding it up to one
            would deliver a pixel larger than the requested area.

    """
    pixels = int(_IMAGE_PIXEL_RULE(_snap(count)))
    if pixels < 1:
        raise FovError(
            f"fov dimension is {count:.4g} pixel(s) at this resolution, less than one; "
            "state a larger area or a finer resolution"
        )
    return pixels


def _ceil_pixels(count: float) -> int:
    """Pixels needed to cover ``count`` of them, at least one."""
    return max(1, math.ceil(_snap(count)))


def _pair(value, name: str) -> tuple[float, float]:
    """Accept a scalar or a 2-sequence and return a pair of floats."""
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    if isinstance(value, Sequence) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise FovError("%s must be a number or a pair, got %r" % (name, value))


def isotropic_resolution(value, name: str) -> float:
    """The one resolution a ``dpi`` or ``mm_per_px`` statement makes.

    A scalar is returned as given. A pair is accepted only when its two values
    agree to :data:`_ISOTROPIC_TOLERANCE`, and is then folded to their mean.

    Raises:
        FovError: A non-positive value, or a pair asking for non-square pixels
            (assumption 2). The YAML ``FovConfig`` validator calls this too, so
            a payload and a Python call are refused with the same message.

    """
    value_x, value_y = _pair(value, name)
    if value_x <= 0 or value_y <= 0:
        raise FovError(f"{name} must be positive, got {value!r}")
    if abs(value_x - value_y) > _ISOTROPIC_TOLERANCE * max(value_x, value_y):
        raise FovError(
            f"{name} {value_x:g} x {value_y:g} asks for non-square pixels, which "
            "cannot be rendered: an orthographic RTX camera takes its vertical "
            "extent from the horizontal aperture and the image aspect, so the "
            "image would come out at the horizontal pitch on both axes with "
            "seams between tiles. Give one value, e.g. "
            f"{name}: {value_x:g}"
        )
    return 0.5 * (value_x + value_y)


def check_square_pixels(
    horizontal_aperture_su: float,
    vertical_aperture_su: float,
    resolution_px: Sequence[int],
    *,
    stage_meters_per_unit: float | None = None,
    tolerance: float = SQUARE_PIXEL_TOLERANCE,
) -> None:
    """Refuse an orthographic frame whose apertures and resolution disagree.

    Apertures in scene units, resolution ``(width, height)`` in the camera's
    own axes. The renderer keeps ``horizontal / width`` on both axes and
    ignores the vertical aperture, so when ``vertical / height`` differs the
    frame covers a different strip of the board than the scan grid placed it
    over, and every tile row lands offset from the next. :func:`resolve_fov`
    never asks for that; a hand-authored ``scan_grid`` with ``scan_area_mm``
    beside an authored ``renderer.resolution`` can.

    Raises:
        FovError: The two pitches differ by more than ``tolerance``, relative.

    """
    width, height = int(resolution_px[0]), int(resolution_px[1])
    if width <= 0 or height <= 0 or horizontal_aperture_su <= 0:
        return
    pitch_h = float(horizontal_aperture_su) / width
    pitch_v = float(vertical_aperture_su) / height
    if abs(pitch_h - pitch_v) <= tolerance * pitch_h:
        return
    unit = "su"
    if stage_meters_per_unit:
        pitch_h = su_to_mm(pitch_h, stage_meters_per_unit)
        pitch_v = su_to_mm(pitch_v, stage_meters_per_unit)
        unit = "mm"
    raise FovError(
        f"non-square pixels: the horizontal aperture gives {pitch_h:.6f} {unit}/px "
        f"across {width} px, the vertical aperture {pitch_v:.6f} {unit}/px down "
        f"{height} px. An orthographic RTX camera ignores the vertical aperture "
        "and renders the horizontal pitch on both axes, so this frame would not "
        "cover the cell the scan grid placed it on and the tiles would not meet. "
        "Make scan_area_mm / grid counts and renderer.resolution agree, or "
        "state the capture with capture.fov"
    )


def _corner_pair(corners) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two ``(x, y)`` points from two ``(x, y)`` or ``(x, y, z)`` corners."""
    if not isinstance(corners, Sequence) or len(corners) != 2:
        raise FovError("corners_mm must be two points, got %r" % (corners,))
    points = []
    for corner in corners:
        if not isinstance(corner, Sequence) or len(corner) not in (2, 3):
            raise FovError(
                "each corner in corners_mm must be (x, y) or (x, y, z), got %r"
                % (corner,)
            )
        try:
            points.append((float(corner[0]), float(corner[1])))
        except (TypeError, ValueError) as error:
            raise FovError(
                f"each corner in corners_mm must contain numeric x and y, got {corner!r}"
            ) from error
    return points[0], points[1]


def mm_to_su(mm: float, stage_meters_per_unit: float) -> float:
    """Millimetres to scene units.

    ``stage_meters_per_unit`` is the stage's own metadata: 0.01 for centimetre
    stages, 0.001 for millimetre stages. Do not infer it from geometry -- some
    exports declare a value that disagrees with the numbers in the file, and the
    declared value is what every other consumer of the stage will use.
    """
    if stage_meters_per_unit <= 0:
        raise FovError("stage_meters_per_unit must be > 0, got %r" % stage_meters_per_unit)
    return (mm / 1000.0) / stage_meters_per_unit


def su_to_mm(su: float, stage_meters_per_unit: float) -> float:
    """Scene units to millimetres."""
    if stage_meters_per_unit <= 0:
        raise FovError("stage_meters_per_unit must be > 0, got %r" % stage_meters_per_unit)
    return su * stage_meters_per_unit * 1000.0


def scan_region_su(
    scan_area_mm,
    *,
    projection: str,
    stage_meters_per_unit: float,
) -> tuple[float, float]:
    """The stated scan rectangle in scene units, or a refusal.

    The conversion happens at render time, against the stage's own scale,
    because only the stage knows it: a constant guessed upstream is out by a
    factor of ten on a millimetre stage, which is the difference between
    framing the product and framing a tenth of it.

    Raises:
        FovError: For a non-orthographic projection, or a non-positive
            dimension. Refused rather than ignored -- the resolution and the
            cell counts were already derived from this rectangle, so carrying
            on would frame the whole board at the pixel count of a part of it
            and report the part.

    """
    if projection != "orthographic":
        raise FovError(
            "scan_area_mm requires an orthographic camera, got %r; a "
            "perspective capture would need the camera height and a "
            "projection term" % projection
        )
    area_x, area_y = _pair(scan_area_mm, "scan_area_mm")
    if area_x <= 0 or area_y <= 0:
        raise FovError("scan_area_mm must be positive, got %r" % (scan_area_mm,))
    return (
        mm_to_su(area_x, stage_meters_per_unit),
        mm_to_su(area_y, stage_meters_per_unit),
    )


def _against_the_machine(dpi_x: float, dpi_y: float) -> dict:
    """Say how a delivered resolution relates to the scanner's own numbers.

    Reported, never enforced. The simulated capture is exact at any pitch; the
    reader of a dataset needs to know whether the real machine could have made
    the same image, and at which of its two settings.
    """
    return {
        "sensor_dpi": SENSOR_DPI,
        "supported_dpi": list(SUPPORTED_DPI),
        "within_sensor": max(dpi_x, dpi_y) <= SENSOR_DPI * (1.0 + _DPI_MATCH_TOLERANCE),
        # One setting has to cover both axes: a machine that offers 600 and
        # 1200 dpi cannot deliver 600 across and 1200 down in one pass, so an
        # axis-by-axis match would call a mixed pitch supported.
        "supported": any(
            all(
                abs(value - option) <= _DPI_MATCH_TOLERANCE * option
                for value in (dpi_x, dpi_y)
            )
            for option in SUPPORTED_DPI
        ),
    }


def resolve_fov(
    *,
    stage_meters_per_unit: float,
    scan_area_mm: Sequence[float] | None = None,
    corners_mm: Sequence[Sequence[float]] | None = None,
    dpi: float | Sequence[float] | None = None,
    mm_per_px: float | Sequence[float] | None = None,
    image_size_px: Sequence[int] | None = None,
    centre_xy_su: Sequence[float] = (0.0, 0.0),
    centered_on: str = "unspecified",
    max_tile_px: Sequence[int] = (4096, 4096),
    projection: str = "orthographic",
    guard_px: int = 0,
) -> dict:
    """Turn a scan rectangle into camera apertures, a cell count and a resolution.

    Exactly two of the three capture quantities are independent: area,
    resolution and pixel count. Supply the area and one statement of the
    resolution -- ``dpi``, ``mm_per_px`` or ``image_size_px``.

    The area is either a size (``scan_area_mm``, centered on the board) or two
    opposite corners (``corners_mm``), which is how a machine test program
    states its field of view: the lower-left and upper-right points of the
    region in world coordinates. Corners also place the rectangle, so the
    result carries its center in millimetres (``center_mm``).

    Args:
        stage_meters_per_unit: Stage metadata, metres per scene unit.
        scan_area_mm: ``(length_x, breadth_y)`` of board to capture, centered on
            the board.
        corners_mm: Two opposite corners of the region to capture, each
            ``(x, y)`` or ``(x, y, z)`` in world millimetres; ``z`` is ignored,
            the capture being orthographic from above or below. Their order
            does not matter. Exactly one of ``scan_area_mm`` and ``corners_mm``.
        dpi: Scanning resolution; a scalar for isotropic, or ``(dpi_x, dpi_y)``.
        mm_per_px: The same resolution as a pixel pitch -- the sensor's x,y
            resolution as a machine spec sheet states it (0.042333 for 600
            dpi). A scalar or ``(pitch_x, pitch_y)``.
        image_size_px: ``(width, height)`` of the stitched canvas. Give this
            instead of a resolution to pin the output size and let dpi follow.
        centre_xy_su: World XY the rectangle is centred on. Assumption 1.
            Ignored with ``corners_mm``, which place the rectangle themselves.
        centered_on: What that centre refers to -- ``"panel"``, ``"instance"``,
            or ``"unspecified"``. Recorded, never inferred. Must stay
            ``"unspecified"`` with ``corners_mm``.
        max_tile_px: Largest single rendered frame, guard band included. Larger
            canvases tile.
        projection: Only ``"orthographic"`` is exact here; anything else raises.
        guard_px: Pixels rendered beyond every edge of each cell and cropped
            before the tile is written (assumption 5). ``0`` renders the
            delivered tile directly.

    Returns:
        A mapping carrying both the scanner-facing quantities and the
        pipeline-facing ones, plus ``assumptions``.

    Raises:
        FovError: On a non-orthographic projection, a missing or
            over-determined pair of inputs, a non-positive dimension, or a
            guard band that is negative or leaves no room in ``max_tile_px``.

    """
    if projection != "orthographic":
        raise FovError(
            "field of view is exact only for an orthographic capture; "
            "%r would need the camera height and a projection term" % projection
        )
    center_mm = None
    corners_requested = None
    if (scan_area_mm is None) == (corners_mm is None):
        raise FovError(
            "give exactly one of scan_area_mm or corners_mm; got %s"
            % ("both" if corners_mm is not None else "neither")
        )
    if corners_mm is not None:
        (x0, y0), (x1, y1) = _corner_pair(corners_mm)
        if x0 == x1 or y0 == y1:
            raise FovError(
                "corners_mm must span a positive area, got %r" % (corners_mm,)
            )
        if centered_on != "unspecified":
            raise FovError(
                "centered_on describes a rectangle centered on the board; "
                "corners_mm place it themselves, got centered_on=%r" % centered_on
            )
        corners_requested = [[min(x0, x1), min(y0, y1)], [max(x0, x1), max(y0, y1)]]
        scan_area_mm = (abs(x1 - x0), abs(y1 - y0))
        center_mm = [0.5 * (x0 + x1), 0.5 * (y0 + y1)]
        centre_xy_su = (
            mm_to_su(center_mm[0], stage_meters_per_unit),
            mm_to_su(center_mm[1], stage_meters_per_unit),
        )
    if centered_on not in VALID_CENTERED_ON:
        raise FovError(
            "centered_on must be one of %s, got %r"
            % (sorted(VALID_CENTERED_ON), centered_on)
        )

    area_x, area_y = _pair(scan_area_mm, "scan_area_mm")
    if area_x <= 0 or area_y <= 0:
        raise FovError("scan_area_mm must be positive, got %r" % (scan_area_mm,))

    stated = [
        name
        for name, value in (
            ("dpi", dpi),
            ("mm_per_px", mm_per_px),
            ("image_size_px", image_size_px),
        )
        if value is not None
    ]
    if len(stated) != 1:
        raise FovError(
            "give exactly one of dpi, mm_per_px or image_size_px alongside "
            "scan_area_mm (they are not independent); got %s"
            % (", ".join(stated) or "none")
        )

    # A pitch is a dpi written the other way up; fold it in here so the rest of
    # the arithmetic has one resolution input.
    # A pitch is a dpi written the other way up; fold it in here so the rest of
    # the arithmetic has one resolution input. Both are one number (assumption
    # 2): a pair that disagrees is refused rather than rendered wrong.
    mm_per_px_requested = None
    dpi_requested = None
    if mm_per_px is not None:
        mm_per_px_requested = list(_pair(mm_per_px, "mm_per_px"))
        dpi = MM_PER_INCH / isotropic_resolution(mm_per_px, "mm_per_px")
        dpi_requested = [MM_PER_INCH / value for value in mm_per_px_requested]
    elif dpi is not None:
        dpi_requested = list(_pair(dpi, "dpi"))
        dpi = isotropic_resolution(dpi, "dpi")

    # Assumption 3, restated precisely: the SAMPLE PITCH is the primary quantity,
    # not the rectangle. Deriving the effective area by dividing a rounded pixel
    # count back into the requested millimetres makes the X and Y pitches differ
    # by a part in ten thousand -- non-square pixels from an isotropic request,
    # which is a silently wrong contract. Fixing the pitch and rounding the
    # pixel count once keeps dpi exactly as asked.
    grown_axis = None
    if dpi is not None:
        pitch = MM_PER_INCH / dpi
        # Once, by the scanner's rule. The cells below never change this number;
        # they only decide how it is rendered.
        px_w = image_pixels(area_x / pitch)
        px_h = image_pixels(area_y / pitch)
    else:
        px_w_f, px_h_f = _pair(image_size_px, "image_size_px")
        px_w, px_h = int(px_w_f), int(px_h_f)
        if px_w <= 0 or px_h <= 0:
            raise FovError("image_size_px must be positive, got %r" % (image_size_px,))
        # Assumption 2: one pitch for both axes, the coarser of the two the
        # pixel count implies, so the whole requested rectangle is in the
        # image. The other axis then covers more than was asked; the pixel
        # count stays exactly as pinned.
        pitch_for_x = area_x / px_w
        pitch_for_y = area_y / px_h
        pitch = max(pitch_for_x, pitch_for_y)
        if abs(pitch_for_x - pitch_for_y) > _ISOTROPIC_TOLERANCE * pitch:
            grown_axis = "y" if pitch_for_x > pitch_for_y else "x"
    pitch_x = pitch_y = pitch

    eff_dpi_x = MM_PER_INCH / pitch_x
    eff_dpi_y = MM_PER_INCH / pitch_y

    max_w_f, max_h_f = _pair(max_tile_px, "max_tile_px")
    max_w, max_h = int(max_w_f), int(max_h_f)
    if max_w <= 0 or max_h <= 0:
        raise FovError("max_tile_px must be positive, got %r" % (max_tile_px,))
    if isinstance(guard_px, bool) or not isinstance(guard_px, int) or guard_px < 0:
        raise FovError(f"guard_px must be a non-negative integer, got {guard_px!r}")
    # Assumption 5: max_tile_px bounds what is rendered, so the guard band comes
    # out of it -- a 4096-pixel limit holds 4080-pixel cells at an 8-pixel guard.
    usable_w, usable_h = max_w - 2 * guard_px, max_h - 2 * guard_px
    if usable_w < 1 or usable_h < 1:
        raise FovError(
            f"guard_px={guard_px} leaves no room for a cell inside max_tile_px "
            f"{(max_w, max_h)!r}"
        )
    x_num = max(1, math.ceil(px_w / usable_w))
    y_num = max(1, math.ceil(px_h / usable_h))

    # Assumption 4: the area divides, the resolution does not. Cells are equal
    # and whole pixels (the renderer has one resolution per pass), so the mosaic
    # may exceed the image by under one pixel per cell along an axis; the stitch
    # crops it back to exactly px_w x px_h.
    tile_w = _ceil_pixels(px_w / x_num)
    tile_h = _ceil_pixels(px_h / y_num)
    mosaic_w, mosaic_h = x_num * tile_w, y_num * tile_h

    # The pitch is fixed, so an area is a pixel count times the pitch: what the
    # delivered image covers, and what the render has to span before the crop.
    eff_area_x = px_w * pitch_x
    eff_area_y = px_h * pitch_y
    rendered_area_x = mosaic_w * pitch_x
    rendered_area_y = mosaic_h * pitch_y

    # Orthographic: one cell's world footprint IS its aperture.
    hap_su = mm_to_su(rendered_area_x / x_num, stage_meters_per_unit)
    vap_su = mm_to_su(rendered_area_y / y_num, stage_meters_per_unit)
    # Assumption 5: what the camera renders before the crop. The same pitch, so
    # the aperture grows by exactly 2 * guard_px pitches; the cell centres and
    # the step between them do not move.
    render_tile_w, render_tile_h = tile_w + 2 * guard_px, tile_h + 2 * guard_px
    render_hap_su = mm_to_su(render_tile_w * pitch_x, stage_meters_per_unit)
    render_vap_su = mm_to_su(render_tile_h * pitch_y, stage_meters_per_unit)

    cx, cy = (float(centre_xy_su[0]), float(centre_xy_su[1]))
    half_x = mm_to_su(eff_area_x, stage_meters_per_unit) / 2.0
    half_y = mm_to_su(eff_area_y, stage_meters_per_unit) / 2.0

    if dpi is not None:
        pitch_note = (
            "pixel counts computed once as area / pitch rounded "
            f"{_IMAGE_PIXEL_RULE.__name__} (whole pixels inside the field of "
            "view); scan_area_mm_effective is what the image covers"
        )
    elif grown_axis is None:
        pitch_note = (
            "pitch is scan_area_mm / image_size_px, the same on both axes; "
            "scan_area_mm_effective is what the image covers"
        )
    else:
        # The request cannot be met at square pixels; say by how much, on the
        # record, rather than leave it to be discovered in the image.
        axis = 0 if grown_axis == "x" else 1
        requested = (area_x, area_y)[axis]
        effective = (eff_area_x, eff_area_y)[axis]
        pitch_note = (
            f"image_size_px {px_w} x {px_h} is not the aspect of scan_area_mm "
            f"{area_x:.3f} x {area_y:.3f} mm; at one pitch (the coarser, so the "
            f"whole rectangle is imaged) world {grown_axis} covers "
            f"{effective:.3f} mm instead of the requested {requested:.3f}, "
            f"{0.5 * (effective - requested):.3f} mm each side of it, centered on "
            "the requested rectangle; scan_area_mm_effective is what the image "
            "covers"
        )

    over = []
    machine = _against_the_machine(eff_dpi_x, eff_dpi_y)
    if not machine["within_sensor"]:
        over.append(
            "requested %.1f x %.1f dpi exceeds a typical scanner sensor's %.0f dpi; "
            "simulated capture is not limited by it, a physical scanner would "
            "have to resample" % (eff_dpi_x, eff_dpi_y, SENSOR_DPI)
        )
    if not machine["supported"]:
        over.append(
            "%.1f x %.1f dpi is not a common scanner resolution (%s); "
            "simulated capture is not limited by it"
            % (eff_dpi_x, eff_dpi_y, " or ".join("%.0f" % d for d in SUPPORTED_DPI))
        )

    return {
        "schema_version": 1,
        "projection": projection,
        # --- as requested -------------------------------------------------
        "scan_area_mm": [area_x, area_y],
        "corners_mm": corners_requested,
        "center_mm": center_mm,
        "dpi_requested": dpi_requested,
        "mm_per_px_requested": mm_per_px_requested,
        "image_size_px_requested": (
            [int(image_size_px[0]), int(image_size_px[1])]
            if image_size_px is not None
            else None
        ),
        # --- as delivered -------------------------------------------------
        "scan_area_mm_effective": [eff_area_x, eff_area_y],
        "dpi_effective": [eff_dpi_x, eff_dpi_y],
        "image_size_px": [px_w, px_h],
        "tile_size_px": [tile_w, tile_h],
        # What the cells render before the stitch crops to image_size_px. Equal
        # to it whenever the image divides into whole cells.
        "mosaic_size_px": [mosaic_w, mosaic_h],
        "scan_area_mm_rendered": [rendered_area_x, rendered_area_y],
        "mm_per_px": [pitch_x, pitch_y],
        # Rendered beyond every cell edge and cropped before the tile is
        # written; tile_size_px is what is delivered, render_tile_size_px what
        # the camera renders.
        "guard_px": guard_px,
        "render_tile_size_px": [render_tile_w, render_tile_h],
        # --- what the capture consumes ------------------------------------
        "x_num": x_num,
        "y_num": y_num,
        "horizontal_aperture_su": hap_su,
        "vertical_aperture_su": vap_su,
        "render_horizontal_aperture_su": render_hap_su,
        "render_vertical_aperture_su": render_vap_su,
        "centre_xy_su": [cx, cy],
        "bounds_su": {
            "min_xy": [cx - half_x, cy - half_y],
            "max_xy": [cx + half_x, cy + half_y],
        },
        "stage_meters_per_unit": stage_meters_per_unit,
        # --- how this compares to the real scanner ------------------------
        "machine": machine,
        # --- what we had to decide for them -------------------------------
        "assumptions": [
            (
                "scan area placed by corners_mm, centered on (%.3f, %.3f) mm; it "
                "is the whole pixels about that center (rounded down, within half a pixel per edge)"
                % (center_mm[0], center_mm[1])
            )
            if center_mm is not None
            else "board centered in the scan area; center taken from centered_on=%s"
            % centered_on,
            (
                f"square pixels: one pitch of {pitch:.6f} mm/px on both axes (an "
                "orthographic render cannot deliver two)"
            ),
            pitch_note,
            "tiling splits the area into %d x %d equal cells, each at its own "
            "resolution; the stitched mosaic is cropped to image_size_px"
            % (x_num, y_num),
            (
                f"each cell renders a {guard_px} px guard band on every side, "
                "cropped before the tile is written; tiles are tile_size_px"
            )
            if guard_px
            else "no guard band; each cell renders exactly tile_size_px",
        ]
        + over,
        "centered_on": centered_on,
    }


def aperture_to_usd(aperture_su: float) -> float:
    """Scene-unit aperture to the raw USD attribute value.

    USD stores ``horizontalAperture`` / ``verticalAperture`` / ``focalLength`` in
    **tenths of a scene unit** -- the old film-format convention, carried
    forward. Anyone who writes a scene-unit number straight onto the camera is
    out by a factor of ten, and the render still succeeds, just framed wrong.
    """
    return aperture_su * 10.0
