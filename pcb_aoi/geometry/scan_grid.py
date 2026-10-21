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

"""
Pure-Python helpers for computing the camera scan-grid that covers a board's
3D bounding box. No Omniverse / USD dependencies — unit-testable on the host.

Scene assumptions
-----------------
* World up-axis is **+Z**. The "top" of the board (the side the camera looks
  at) faces **+Z**; the camera sits above the board at some larger Z and
  looks down ``-Z``.
* The board's bounding box is **axis-aligned with world X/Y/Z** — i.e. the
  board has been authored / oriented so its long edges lie along world X
  and Y. A board rotated about Z in world space would have an AABB larger
  than the board itself, and the cells produced here would scan that
  oversized AABB rather than the board; rotate the board prim flat first.
* The grid cells walk along **world X / Y**. The camera may be yawed about
  world Z (``camera_z_rotation_deg``) — this only swaps which camera-local
  aperture binds which world axis; the grid stays world-aligned.

Aperture units
--------------
USD's ``GfCamera`` stores ``horizontalAperture`` / ``verticalAperture`` /
``focalLength`` in **tenths of scene units** (the older "film-format"
convention). Divide those raw USD values by 10 before passing them as
``*_su`` arguments here.
"""

from __future__ import annotations

import math


def auto_complete_grid_nums(
    bbox_size_x: float,
    bbox_size_y: float,
    *,
    horizontal_aperture_su: float,
    vertical_aperture_su: float,
    x_num: int | None,
    y_num: int | None,
    camera_z_rotation_deg: float = 0.0,
) -> tuple[int, int]:
    """Resolve missing grid dimensions to aspect-matched cells.

    Given the bbox size and camera apertures, when exactly one of
    ``x_num``/``y_num`` is ``None`` this returns a value for the other so
    that each cell's world-axis aspect (cell_x : cell_y) matches the
    camera's footprint aspect (eff_x : eff_y), which minimises overlap and
    makes each frame fill its image area.

    With camera yaw a multiple of ±90°, ``eff_x`` is whichever aperture
    binds world X (vertical aperture for ±90° yaw, horizontal otherwise);
    ``eff_y`` is the other aperture. Non-cardinal yaws raise
    ``NotImplementedError`` from :func:`compute_scan_grid_geometry` and so
    are not relevant here.

    Returns ``(x_num, y_num)`` rounded up to integers ``>= 1``. If both are
    given, returns them unchanged. If both are ``None`` raises
    ``ValueError``.
    """
    if x_num is None and y_num is None:
        raise ValueError("Must provide at least one of x_num or y_num")
    if x_num is not None and x_num < 1:
        raise ValueError(f"x_num must be >= 1 (got {x_num})")
    if y_num is not None and y_num < 1:
        raise ValueError(f"y_num must be >= 1 (got {y_num})")
    if bbox_size_x <= 0 or bbox_size_y <= 0:
        raise ValueError(f"bbox sizes must be > 0 (got {bbox_size_x}, {bbox_size_y})")
    if horizontal_aperture_su <= 0 or vertical_aperture_su <= 0:
        raise ValueError(
            f"apertures must be > 0 (got hap={horizontal_aperture_su}, vap={vertical_aperture_su})"
        )
    if x_num is not None and y_num is not None:
        return int(x_num), int(y_num)

    swap = int(round(((float(camera_z_rotation_deg) % 360.0) + 360.0) % 360.0 / 90.0)) % 2 == 1
    eff_x = float(vertical_aperture_su) if swap else float(horizontal_aperture_su)
    eff_y = float(horizontal_aperture_su) if swap else float(vertical_aperture_su)

    if y_num is None:
        y_num = max(1, math.ceil(int(x_num) * bbox_size_y * eff_x / (bbox_size_x * eff_y)))
    else:  # x_num is None
        x_num = max(1, math.ceil(int(y_num) * bbox_size_x * eff_y / (bbox_size_y * eff_x)))
    return int(x_num), int(y_num)


_SPAN_NAME = {True: "bbox", False: "the requested scan region"}


def compute_scan_grid_geometry(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    *,
    projection: str,
    horizontal_aperture_su: float,
    vertical_aperture_su: float,
    focal_length_su: float | None,
    x_num: int | None = None,
    y_num: int | None = None,
    camera_z_rotation_deg: float = 0.0,
    z_min_camera_su: float | None = None,
    scan_region_su: tuple[float, float] | None = None,
    scan_region_center_su: tuple[float, float] | None = None,
    epsilon: float = 1e-6,
) -> dict:
    """Compute camera ``(x, y, z)`` cells whose union FOV covers an AABB.

    Inputs
    ------
    bbox_min, bbox_max : tuple of 3 floats
        World-space AABB of the board (scene units). The bbox must be
        **axis-aligned with world X/Y/Z**; rotate the board prim flat
        before computing this. ``bbox_max[2]`` is taken as the board top.
    projection : str
        ``"orthographic"`` or ``"perspective"`` (alias ``"pinhole"``).
    horizontal_aperture_su, vertical_aperture_su : float (scene units)
        Camera apertures in **camera-local** axes (image-x and image-y).
        Convert from raw USD values by dividing by 10.
    focal_length_su : float or None (scene units)
        Required for ``"perspective"``; ignored for ``"orthographic"``.
    x_num, y_num : int or None
        Number of grid cells along **world** X / Y. ``x_num * y_num`` images
        in total. Either may be ``None`` (but not both): the missing
        dimension is auto-filled by :func:`auto_complete_grid_nums` so
        cells are aspect-matched to the camera footprint (typically the
        smallest count that still covers the bbox without gaps in that
        axis).
    camera_z_rotation_deg : float, default 0
        Camera yaw around world Z (degrees). After rotation, image-x in
        world is ``(cos θ, sin θ, 0)`` and image-y is ``(-sin θ, cos θ, 0)``.
        Cells stay world-axis-aligned; rotation only swaps which aperture
        binds which world axis. **Only cardinal multiples of 90°** are
        accepted — for non-cardinal yaws the rotated footprint rectangle's
        corners fall short of its world-AABB corners, so an axis-aligned
        grid would leave bbox corners uncovered. Raises
        ``NotImplementedError`` otherwise.
    z_min_camera_su : float or None
        Optional safety floor on the camera's Z. The chosen Z is
        ``max(auto, z_min_camera_su)``.
    scan_region_su : tuple of 2 floats or None
        World XY extent to tile -- centered on the bbox midpoint unless
        ``scan_region_center_su`` places it -- when that is
        deliberately **not** the whole board — an AOI field of view scans the
        smallest rectangle containing the product, so the board outside it is
        meant to be missed. Given this, coverage is checked against the region
        rather than the bbox and the cell centres span the region. ``None``
        keeps the default: tile the board and refuse to leave a gap.
    scan_region_center_su : tuple of 2 floats or None
        World XY center of ``scan_region_su`` when the region is placed by the
        caller rather than centered on the board -- a field of view stated as
        two corners from the machine's test program. ``None`` keeps the bbox
        midpoint. Only meaningful with ``scan_region_su``.
    epsilon : float
        Numerical slack for aperture-vs-bbox coverage checks.

    Z derivation
    ------------
    * **perspective** — ``z = board_top + height_required`` where
      ``height_required`` is the smallest height whose footprint covers
      ``bbox_size_x / x_num`` (and the same for Y, after rotation swap).
      So ``x_num = y_num = 1`` auto-zooms out until one frame captures the
      whole board; larger counts zoom in proportionally.
    * **orthographic** — footprint = aperture (distance-independent). ``z``
      only needs to clear the board geometry: ``board_top + bbox_size_z``
      (with a small floor when the bbox is degenerate).

    Outputs
    -------
    dict with keys

    * ``x_start``, ``x_end`` (float, scene units, world X) — first and last
      camera-X positions. ``x_start >= x_end`` (descending convention).
    * ``y_start``, ``y_end`` (float, scene units, world Y) — same.
    * ``x_step``, ``y_step`` (float, scene units) — uniform step between
      adjacent centres along each axis. ``0`` when ``x_num == 1`` /
      ``y_num == 1``; the centre is then the bbox midpoint along that axis.
    * ``z`` (float, scene units, world Z) — common camera height for the
      whole grid.
    * ``x_num``, ``y_num`` (int) — final cell counts (after auto-fill).
    * ``footprint_x``, ``footprint_y`` (float, scene units) — cell
      footprint sizes along **world** X / Y (after camera yaw is applied).
      Not the same as ``horizontal_aperture_su``/``vertical_aperture_su``
      when yaw is ±90°.
    * ``x_centers``, ``y_centers`` (list of float, descending) — explicit
      camera-X / camera-Y centres in iteration order.
    * ``projection`` (str) — normalised projection name (``"perspective"``
      or ``"orthographic"``).

    Iteration order matches :func:`build_scan_positions`: outer loop walks
    Y from ``y_centers[0]`` (largest world Y, "top") down to
    ``y_centers[-1]``; inner loop walks X from ``x_centers[0]`` (largest
    world X, "right") down to ``x_centers[-1]``. So frame 0 of a trigger
    is always cell ``(x_idx=0, y_idx=0)`` — the world top-right cell.

    Raises
    ------
    ValueError
        Apertures non-positive, bbox degenerate, ``x_num``/``y_num`` < 1,
        unknown projection, missing ``focal_length_su`` for perspective,
        aperture-and-num combination cannot cover the bbox in some axis,
        or both ``x_num`` and ``y_num`` are ``None``.
    NotImplementedError
        ``camera_z_rotation_deg`` is not a multiple of 90°.
    """
    if scan_region_center_su is not None and scan_region_su is None:
        raise ValueError("scan_region_center_su places scan_region_su, which is not set")
    bbox_size_x = float(bbox_max[0]) - float(bbox_min[0])
    bbox_size_y = float(bbox_max[1]) - float(bbox_min[1])
    # What the grid has to span. Normally the board; for a stated field of view
    # a smaller rectangle centred on it, and then falling short of the board is
    # the intent rather than a gap to refuse.
    if scan_region_su is not None:
        span_x, span_y = (float(scan_region_su[0]), float(scan_region_su[1]))
        if span_x <= 0 or span_y <= 0:
            raise ValueError(f"scan_region_su must be positive (got {scan_region_su})")
    else:
        span_x, span_y = (bbox_size_x, bbox_size_y)
    # Auto-fill from what this grid has to span, which is the scan region when
    # one is stated. Filling from the bbox instead sizes the missing count for
    # the whole board and then tiles a smaller rectangle with it: the aspect it
    # matched is not the aspect it covers.
    x_num, y_num = auto_complete_grid_nums(
        span_x,
        span_y,
        horizontal_aperture_su=horizontal_aperture_su,
        vertical_aperture_su=vertical_aperture_su,
        x_num=x_num,
        y_num=y_num,
        camera_z_rotation_deg=camera_z_rotation_deg,
    )
    if x_num < 1 or y_num < 1:
        raise ValueError(f"x_num and y_num must be >= 1 (got {x_num}, {y_num})")
    if horizontal_aperture_su <= 0 or vertical_aperture_su <= 0:
        raise ValueError(
            f"apertures must be > 0 (got hap={horizontal_aperture_su}, vap={vertical_aperture_su})"
        )

    # Camera yaw around world Z. Map the (camera-local) horizontal/vertical
    # apertures onto world X/Y. For a camera looking down -Z with yaw θ,
    # image-x → (cos θ, sin θ, 0), image-y → (-sin θ, cos θ, 0). The footprint
    # rectangle is axis-aligned in world only when |sin θ * cos θ| == 0,
    # i.e. θ is a multiple of 90°. Non-cardinal yaws make the rotated
    # rectangle's corners fall short of its world-AABB corners, leaving bbox
    # corners uncoverable by an axis-aligned grid — refuse instead of
    # silently producing gaps.
    theta = ((float(camera_z_rotation_deg) % 360.0) + 360.0) % 360.0
    snapped = round(theta / 90.0) * 90.0
    if abs(theta - snapped) > 1e-6:
        raise NotImplementedError(
            f"camera_z_rotation_deg={camera_z_rotation_deg} is not a multiple "
            f"of 90°. Only cardinal yaws (0, ±90, 180) are supported because "
            f"non-cardinal rotated rectangles cannot tile an axis-aligned bbox."
        )
    swap_aperture_xy = int(round(snapped / 90.0)) % 2 == 1
    if swap_aperture_xy:
        eff_hap = float(vertical_aperture_su)  # world-X aperture
        eff_vap = float(horizontal_aperture_su)  # world-Y aperture
    else:
        eff_hap = float(horizontal_aperture_su)
        eff_vap = float(vertical_aperture_su)

    bbox_size_z = float(bbox_max[2]) - float(bbox_min[2])
    board_top_z = float(bbox_max[2])

    proj = projection.lower()
    if proj == "pinhole":
        proj = "perspective"

    if proj == "orthographic":
        footprint_x = eff_hap
        footprint_y = eff_vap
        if x_num * footprint_x + epsilon < span_x:
            raise ValueError(
                f"Orthographic FOV cannot cover {_SPAN_NAME[scan_region_su is None]} "
                f"along X with x_num={x_num}: x_num * footprint_x "
                f"({x_num * footprint_x:.4f}) < {span_x:.4f}. "
                f"Increase aperture or x_num."
            )
        if y_num * footprint_y + epsilon < span_y:
            raise ValueError(
                f"Orthographic FOV cannot cover {_SPAN_NAME[scan_region_su is None]} "
                f"along Y with y_num={y_num}: y_num * footprint_y "
                f"({y_num * footprint_y:.4f}) < {span_y:.4f}. "
                f"Increase aperture or y_num."
            )
        # z just needs to clear the board geometry; lift by one board thickness
        # (or a small epsilon if the bbox is thin / degenerate).
        z_auto = board_top_z + max(bbox_size_z, max(epsilon, 1e-3))
        z = float(max(z_auto, z_min_camera_su)) if z_min_camera_su is not None else z_auto
    elif proj == "perspective":
        if focal_length_su is None or focal_length_su <= 0:
            raise ValueError(f"focal_length_su must be > 0 for perspective (got {focal_length_su})")
        # Smallest height above board_top whose footprint covers one cell.
        h_for_x = span_x * focal_length_su / (eff_hap * x_num)
        h_for_y = span_y * focal_length_su / (eff_vap * y_num)
        height_required = max(h_for_x, h_for_y, 0.0)
        z_auto = board_top_z + height_required
        z = float(max(z_auto, z_min_camera_su)) if z_min_camera_su is not None else z_auto
        height = z - board_top_z
        if height <= 0:
            raise ValueError(
                f"camera z ({z}) must be above board_top_z ({board_top_z}) for perspective"
            )
        footprint_x = height * eff_hap / float(focal_length_su)
        footprint_y = height * eff_vap / float(focal_length_su)
    else:
        raise ValueError(
            f"unknown projection {projection!r}; expected 'orthographic' or 'perspective'"
        )

    if scan_region_center_su is not None:
        cx, cy = float(scan_region_center_su[0]), float(scan_region_center_su[1])
    else:
        cx = 0.5 * (float(bbox_min[0]) + float(bbox_max[0]))
        cy = 0.5 * (float(bbox_min[1]) + float(bbox_max[1]))

    # When the footprint along an axis already covers the bbox, one cell
    # is sufficient. Earlier the formula ``(bbox - footprint) / (num-1)``
    # produced a *negative* step in that case, which (a) flipped iteration
    # direction and (b) placed the first center outside the bbox so the
    # camera scanned empty space (GeForce20 / flattern3 trigger). Honour
    # the user's requested ``num`` by collapsing all centers in the
    # over-covered axis to the bbox midpoint with step=0.
    if x_num == 1 or footprint_x >= span_x:
        x_centers = [cx] * x_num
        x_step = 0.0
    else:
        x_step = (span_x - footprint_x) / (x_num - 1)
        x_high = cx + (span_x - footprint_x) / 2.0
        x_centers = [x_high - i * x_step for i in range(x_num)]

    if y_num == 1 or footprint_y >= span_y:
        y_centers = [cy] * y_num
        y_step = 0.0
    else:
        y_step = (span_y - footprint_y) / (y_num - 1)
        y_high = cy + (span_y - footprint_y) / 2.0
        y_centers = [y_high - i * y_step for i in range(y_num)]

    return {
        "x_start": float(x_centers[0]),
        "x_end": float(x_centers[-1]),
        "y_start": float(y_centers[0]),
        "y_end": float(y_centers[-1]),
        "x_step": float(x_step),
        "y_step": float(y_step),
        "z": float(z),
        "x_num": len(x_centers),
        "y_num": len(y_centers),
        "footprint_x": float(footprint_x),
        "footprint_y": float(footprint_y),
        "projection": proj,
        "x_centers": [float(v) for v in x_centers],
        "y_centers": [float(v) for v in y_centers],
        "span_x": float(span_x),
        "span_y": float(span_y),
    }


def compute_pixel_layout(
    scan_grid: dict,
    resolution: list[int] | tuple[int, int],
    camera_rotation: dict,
    frame_to_grid: list[tuple[int, int, int]] | list[dict],
    *,
    camera_side: str = "top",
    epsilon: float = 1e-6,
) -> dict:
    """Derive canonical pixel-space tile placements for an orthographic grid.

    The returned canvas uses a top-left origin, +X right, and +Y down. Tile
    origins remain floating-point so subpixel scan steps are not rounded
    independently; only ``canvas_size_px`` is rounded up to integer pixels.
    Placement describes geometry only and intentionally does not prescribe an
    overlap blend policy.

    This helper is deliberately conservative. A single destination rectangle
    is exact only for an orthographic camera with no X/Y tilt and a cardinal
    yaw. Unsupported configurations return ``{"supported": false, "reason":
    ...}`` instead of raising so metadata generation cannot invalidate an
    otherwise successful render.

    A supported result also carries ``pixel_to_world_xy``, an affine map from
    canvas pixels to world-space X/Y (scene units, no Z): ``world_xy =
    origin_xy + px * u_per_px_xy + py * v_per_px_xy``. Callers combine this
    with a declared board reference Z-plane and the stage's meters-per-unit
    (both scene/stage-level facts, not computed here) to place a stitched
    image in USD space.
    """

    def unsupported(reason: str) -> dict:
        return {"supported": False, "reason": reason}

    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        return unsupported("resolution must contain width and height")
    try:
        width, height = int(resolution[0]), int(resolution[1])
    except (TypeError, ValueError):
        return unsupported("resolution width and height must be numeric")
    if width <= 0 or height <= 0:
        return unsupported("resolution width and height must be positive")

    projection = str(scan_grid.get("projection", "")).lower()
    if projection != "orthographic":
        return unsupported(
            "pixel layout currently requires an orthographic camera "
            f"(got {projection or 'unknown'})"
        )

    for axis in ("x", "y"):
        bounds = camera_rotation.get(f"{axis}_range", (0.0, 0.0))
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            return unsupported(f"camera {axis}_range must contain two values")
        try:
            has_tilt = any(abs(float(value)) > epsilon for value in bounds)
        except (TypeError, ValueError):
            return unsupported(f"camera {axis}_range values must be numeric")
        if has_tilt:
            return unsupported(
                "pixel layout currently requires zero camera X/Y tilt"
            )

    try:
        yaw_deg = float(camera_rotation.get("z_fixed", 0.0))
    except (TypeError, ValueError):
        return unsupported("camera z_fixed must be numeric")
    theta_deg = yaw_deg % 360.0
    snapped_deg = (round(theta_deg / 90.0) * 90.0) % 360.0
    angular_error = abs(theta_deg - snapped_deg)
    angular_error = min(angular_error, 360.0 - angular_error)
    if angular_error > epsilon:
        return unsupported(
            "pixel layout currently requires camera yaw to be a multiple of 90 degrees"
        )

    x_centers = scan_grid.get("x_centers")
    y_centers = scan_grid.get("y_centers")
    if not isinstance(x_centers, list) or not isinstance(y_centers, list):
        return unsupported("scan grid must contain x_centers and y_centers")
    if not x_centers or not y_centers:
        return unsupported("scan grid centers must not be empty")
    try:
        x_centers = [float(value) for value in x_centers]
        y_centers = [float(value) for value in y_centers]
        footprint_x = float(scan_grid.get("footprint_x", 0.0))
        footprint_y = float(scan_grid.get("footprint_y", 0.0))
    except (TypeError, ValueError):
        return unsupported("scan grid centers and footprints must be numeric")
    if footprint_x <= 0.0 or footprint_y <= 0.0:
        return unsupported("scan grid footprints must be positive")

    swap_axes = int(round(snapped_deg / 90.0)) % 2 == 1
    image_x_footprint = footprint_y if swap_axes else footprint_x
    image_y_footprint = footprint_x if swap_axes else footprint_y
    image_x_scale = width / image_x_footprint
    image_y_scale = height / image_y_footprint

    theta = math.radians(snapped_deg)
    image_right_world = (math.cos(theta), math.sin(theta))
    image_up_world = (-math.sin(theta), math.cos(theta))
    if camera_side not in {"top", "bottom"}:
        return unsupported(
            f"camera_side must be 'top' or 'bottom' (got {camera_side!r})"
        )
    # The bottom camera is rolled 180 degrees around its local X axis. Its
    # image-right vector is unchanged, but image-up/image-down are reversed.
    if camera_side == "bottom":
        image_up_world = (-image_up_world[0], -image_up_world[1])
    image_down_world = (-image_up_world[0], -image_up_world[1])

    def clean(value: float) -> float:
        if abs(value) <= epsilon:
            return 0.0
        nearest = round(value)
        if abs(value - nearest) <= epsilon:
            return float(nearest)
        return float(value)

    center_pixels: dict[tuple[int, int], tuple[float, float]] = {}
    for y_idx, world_y in enumerate(y_centers):
        for x_idx, world_x in enumerate(x_centers):
            image_x = (
                world_x * image_right_world[0]
                + world_y * image_right_world[1]
            ) * image_x_scale
            image_y = (
                world_x * image_down_world[0]
                + world_y * image_down_world[1]
            ) * image_y_scale
            center_pixels[(x_idx, y_idx)] = (image_x, image_y)

    minimum_x = min(value[0] for value in center_pixels.values())
    minimum_y = min(value[1] for value in center_pixels.values())
    origins = {
        key: (clean(value[0] - minimum_x), clean(value[1] - minimum_y))
        for key, value in center_pixels.items()
    }

    canvas_width = max(origin[0] + width for origin in origins.values())
    canvas_height = max(origin[1] + height for origin in origins.values())
    canvas_width = clean(canvas_width)
    canvas_height = clean(canvas_height)

    # World-space affine for the canvas, anchored on the (x_idx=0, y_idx=0)
    # cell: `image_right_world` / `image_down_world` are unit vectors, so
    # dividing by the pixel scale gives the world delta per pixel step along
    # each image axis. `origins[(0, 0)]` is that cell's *center* projected
    # into canvas pixels, so its true top-left world corner is offset by
    # half a footprint along -right/-down before the per-pixel deltas are
    # applied back from its known canvas origin to pixel (0, 0).
    u_per_px = (
        image_right_world[0] / image_x_scale,
        image_right_world[1] / image_x_scale,
    )
    v_per_px = (
        image_down_world[0] / image_y_scale,
        image_down_world[1] / image_y_scale,
    )
    anchor_center_world = (x_centers[0], y_centers[0])
    # footprint_x / footprint_y are spans along the WORLD axes, but they are
    # being walked along the IMAGE axes. At z_fixed 90 or 270 the image right
    # direction follows world Y, so multiplying it by footprint_x shifted the
    # anchor -- pixel_to_world_xy came out wrong and the sidecar placed the
    # stitched image at the wrong USD coordinates.
    #
    # Projecting the world-axis spans onto each image direction is exact at every
    # cardinal rotation and is the same value as before at 0 degrees.
    footprint_along_right = (abs(image_right_world[0]) * footprint_x
                             + abs(image_right_world[1]) * footprint_y)
    footprint_along_down = (abs(image_down_world[0]) * footprint_x
                            + abs(image_down_world[1]) * footprint_y)
    anchor_corner_world = (
        anchor_center_world[0]
        - 0.5 * footprint_along_right * image_right_world[0]
        - 0.5 * footprint_along_down * image_down_world[0],
        anchor_center_world[1]
        - 0.5 * footprint_along_right * image_right_world[1]
        - 0.5 * footprint_along_down * image_down_world[1],
    )
    anchor_left, anchor_top = origins[(0, 0)]
    origin_world = (
        anchor_corner_world[0] - anchor_left * u_per_px[0] - anchor_top * v_per_px[0],
        anchor_corner_world[1] - anchor_left * u_per_px[1] - anchor_top * v_per_px[1],
    )
    pixel_to_world_xy = {
        "origin_xy": [origin_world[0], origin_world[1]],
        "u_per_px_xy": [u_per_px[0], u_per_px[1]],
        "v_per_px_xy": [v_per_px[0], v_per_px[1]],
    }

    tiles: list[dict] = []
    for item in frame_to_grid:
        try:
            if isinstance(item, dict):
                frame = int(item["frame"])
                x_idx = int(item["x_idx"])
                y_idx = int(item["y_idx"])
            else:
                frame, x_idx, y_idx = (int(value) for value in item)
            left, top = origins[(x_idx, y_idx)]
        except (KeyError, TypeError, ValueError):
            return unsupported("frame_to_grid contains an invalid entry")

        tiles.append(
            {
                "frame": frame,
                "x_idx": x_idx,
                "y_idx": y_idx,
                "dst_rect_xywh_px": [
                    left,
                    top,
                    float(width),
                    float(height),
                ],
            }
        )

    return {
        "supported": True,
        "coordinate_system": "top-left-xywh-pixels",
        "camera_side": camera_side,
        "canvas_size_px": [math.ceil(canvas_width), math.ceil(canvas_height)],
        "tiles": tiles,
        "pixel_to_world_xy": pixel_to_world_xy,
        # The scanned rectangle, in the grid's own terms. The affine above says
        # where a pixel is; these say how much board the capture covered, which
        # is what lets the sidecar restate it as a scan area and a dpi. Without
        # them that block cannot be derived and comes out null.
        "footprint_x": float(footprint_x),
        "footprint_y": float(footprint_y),
        "x_num": len(x_centers),
        "y_num": len(y_centers),
    }


def grid_covers_bbox(geom: dict, bbox_min, bbox_max, epsilon: float = 1e-6) -> bool:
    """Check that the union of x_num*y_num footprints covers the bbox in X and Y."""
    fx = geom["footprint_x"]
    fy = geom["footprint_y"]
    x_min = min(geom["x_centers"]) - fx / 2.0
    x_max = max(geom["x_centers"]) + fx / 2.0
    y_min = min(geom["y_centers"]) - fy / 2.0
    y_max = max(geom["y_centers"]) + fy / 2.0
    return (
        x_min <= bbox_min[0] + epsilon
        and x_max + epsilon >= bbox_max[0]
        and y_min <= bbox_min[1] + epsilon
        and y_max + epsilon >= bbox_max[1]
    )
