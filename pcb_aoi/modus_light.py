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

"""Compute the Modus linear AOI lighting geometry.

No Omniverse / USD dependencies are needed for geometry tests. The USD authoring
wrapper lives in ``common.build_modus_lights``.

Concept
-------
The existing AOI **ring light** is a 3D dome (hemisphere) of DiskLights.
**Modus light** is a *2D* dome: a single row of RECT lights sitting on an arc
in the world **XZ plane** (at ``Y = board-center-Y``), all aiming at the board
centre, for a top-down camera looking down ``-Z``. It mimics a linear/tunnel
RGB(W) AOI illuminator.

Scene assumptions
-----------------
* World up-axis is **+Z**; the board lies in the XY plane. Top inspection
  lights the maximum Z face; bottom inspection lights the minimum Z face.
* Six RectLights in a row along world **X**: 3 left (``-X``) + 3 right (``+X``)
  of the camera, symmetric. The camera occupies the centre slot, so the seven
  slots left->right read ``blue . red . white . [CAMERA] . white . red . blue``
  and the colour order **centre -> outward** is ``white -> red -> blue``.
* Each RectLight is **vertical**: its long side (height) runs along world **Y**,
  its short side (width) along world **X**, so it reads as a vertical bar. The
  height defaults to ``2 x`` the board's Y extent.
* Each light aims at the board centre — its emission normal (UsdLux RectLight
  emits toward local ``-Z``) points from the light's ``(X, Z)`` position back to
  the centre; because every light shares ``Y = board-centre-Y`` the aim
  direction has no Y component, so the long/height axis stays exactly world Y.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Colour order centre -> outward. slot_from_centre 1 = innermost (nearest the
# camera), 3 = outermost. Blue is pinned at 45 deg elevation per the spec.
COLOR_ORDER = ("white", "red", "blue")

DEFAULT_ELEVATIONS = {"white": 75.0, "red": 60.0, "blue": 45.0}

# Saturated RGB(W) matching the aoi_rgb ring recipe (white brightest).
DEFAULT_COLORS = {
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.02, 0.01),
    "blue": (0.03, 0.05, 0.98),
}

# Per-colour intensities (white brightest). Render-tuned defaults; every value
# here is a config/API param.
DEFAULT_INTENSITIES = {"white": 9000.0, "red": 7000.0, "blue": 6000.0}

# Arc radius as a multiple of the board's larger XY extent, so the lights clear
# the board regardless of board size.
DEFAULT_ARC_RADIUS_SCALE = 1.5

# Rect height as a multiple of the board's Y extent (spec: 2x).
DEFAULT_HEIGHT_SCALE = 2.0

# Rect width as a fraction of the board's X extent (also the nominal camera-gap
# width — see the ``even-X vs arc`` design note in the module doc / report).
DEFAULT_WIDTH_FRAC = 0.12


@dataclass(frozen=True)
class ModusLight:
    """Resolved world-space RectLight in the Modus rig.

    Args:
        name: Stable light name.
        color_group: White, red, or blue light group.
        side: Positive one for world +X or negative one for world -X.
        slot_from_center: One-based light slot measured from the camera.
        position: World-space light position.
        width: RectLight extent along its local X axis.
        height: RectLight extent along its local Y axis.
        elevation_deg: Elevation above the inspected board surface.
        color: Linear RGB light color.
        intensity: RectLight intensity.
        aim_point: World-space point illuminated by the light.
        normal: Unit emission direction from the light toward the aim point.
        x_axis: Local positive X axis expressed in world space.
        y_axis: Local positive Y axis expressed in world space.
        z_axis: Local positive Z axis expressed in world space.

    Example:

    .. code-block:: python

        light = compute_modus_layout(
            board_center=(0.0, 0.0, 0.0),
            board_top_z=1.0,
            board_x_extent=100.0,
            board_y_extent=100.0,
        )[0]
        print(light.color_group)
        # white

    """

    name: str
    #: Stable light name, such as ``White_R`` or ``Blue_L``.

    color_group: str
    #: White, red, or blue group name.

    side: int
    #: Positive one for world +X or negative one for world -X.

    slot_from_center: int
    #: One-based light slot measured outward from the camera.

    position: tuple[float, float, float]
    #: World-space light position.

    width: float
    #: RectLight extent along its local X axis.

    height: float
    #: RectLight extent along its local Y axis.

    elevation_deg: float
    #: Elevation above the inspected board surface.

    color: tuple[float, float, float]
    #: Linear RGB light color.

    intensity: float
    #: RectLight intensity.

    aim_point: tuple[float, float, float]
    #: World-space point illuminated by the light.

    normal: tuple[float, float, float]
    #: Unit emission direction from the light toward the aim point.

    x_axis: tuple[float, float, float]
    #: Local positive X axis expressed in world space.

    y_axis: tuple[float, float, float]
    #: Local positive Y axis expressed in world space.

    z_axis: tuple[float, float, float]
    #: Local positive Z axis expressed in world space.


def _normalize(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def compute_modus_layout(
    *,
    board_center: Sequence[float],
    board_top_z: float,
    board_x_extent: float,
    board_y_extent: float,
    elevations: Mapping[str, float] | None = None,
    arc_radius_scale: float = DEFAULT_ARC_RADIUS_SCALE,
    width: float | None = None,
    width_frac: float = DEFAULT_WIDTH_FRAC,
    height_scale: float = DEFAULT_HEIGHT_SCALE,
    colors: Mapping[str, Sequence[float]] | None = None,
    intensities: Mapping[str, float] | None = None,
    camera_side: str = "top",
) -> list[ModusLight]:
    """Resolve the six Modus RectLights in world space.

    Args:
        board_center: World-space center of the board bounds.
        board_top_z: World Z coordinate of the board's maximum-Z face.
        board_x_extent: Board bbox size along world X.
        board_y_extent: Board bbox size along world Y.
        elevations: Elevation in degrees keyed by color group.
        arc_radius_scale: Arc radius scale relative to the largest board extent.
        width: Absolute RectLight width. When None, it is
            ``width_frac * board_x_extent``.
        width_frac: Width as a fraction of the board's X extent.
        height_scale: RectLight height scale relative to the board's Y extent.
        colors: RGB values keyed by color group.
        intensities: Light intensity keyed by color group.
        camera_side: Inspected side. Bottom inspection mirrors the arc below
            the bbox minimum Z, derived from its center and maximum Z.

    Returns:
        Six lights ordered white right/left, red right/left, and blue
        right/left, symmetric in X about the board center.

    Raises:
        ValueError: If the camera side is invalid or a light has no aim direction.

    Example:

    .. code-block:: python

        lights = compute_modus_layout(
            board_center=(0.0, 0.0, 0.0),
            board_top_z=1.0,
            board_x_extent=100.0,
            board_y_extent=100.0,
            camera_side="bottom",
        )
        print(len(lights))
        # 6

    """
    cx, cy, cz = (
        float(board_center[0]),
        float(board_center[1]),
        float(board_center[2]),
    )
    if camera_side not in {"top", "bottom"}:
        raise ValueError(f"Unsupported camera_side: {camera_side!r}")
    top_z = float(board_top_z)
    surface_z = top_z if camera_side == "top" else 2.0 * cz - top_z
    z_sign = 1.0 if camera_side == "top" else -1.0
    bx = abs(float(board_x_extent))
    by = abs(float(board_y_extent))

    elevations = {**DEFAULT_ELEVATIONS, **(elevations or {})}
    colors = {**DEFAULT_COLORS, **(colors or {})}
    intensities = {**DEFAULT_INTENSITIES, **(intensities or {})}

    radius = float(arc_radius_scale) * max(bx, by)
    height = float(height_scale) * by
    rect_width = float(width) if width is not None else float(width_frac) * bx

    # The arc and aim share the inspected face reference, preserving the
    # requested elevation independently of the board's thickness.
    aim = (cx, cy, surface_z)
    lights = []
    for slot, group in enumerate(COLOR_ORDER, start=1):
        elev = float(elevations[group])
        elev_rad = math.radians(elev)
        dx = radius * math.cos(elev_rad)
        dz = radius * math.sin(elev_rad)
        z = surface_z + z_sign * dz
        for side in (1, -1):
            x = cx + side * dx
            pos = (x, cy, z)
            to_aim = (aim[0] - pos[0], aim[1] - pos[1], aim[2] - pos[2])
            normal = _normalize(to_aim)  # emission dir (local -Z)
            z_axis = (-normal[0], -normal[1], -normal[2])  # local +Z
            y_axis = (0.0, 1.0, 0.0)  # long/height axis = world Y
            x_axis = _normalize(_cross(y_axis, z_axis))  # local +X
            lights.append(
                ModusLight(
                    name=f"{group.capitalize()}_{'R' if side > 0 else 'L'}",
                    color_group=group,
                    side=side,
                    slot_from_center=slot,
                    position=pos,
                    width=rect_width,
                    height=height,
                    elevation_deg=elev,
                    color=tuple(float(c) for c in colors[group]),
                    intensity=float(intensities[group]),
                    aim_point=aim,
                    normal=normal,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    z_axis=z_axis,
                )
            )
    return lights


def modus_bounds_from_prims(
    stage: object,
    prim_paths: Sequence[str],
) -> tuple[tuple[float, float, float], float, float, float]:
    """Return the union world bounds that size the Modus rig.

    The rig spans what the camera images: one board, or every placement of a
    step-repeated panel. Each prim must exist and have non-empty world
    bounds; an abstract ``class`` prototype, for example, composes no world
    bounds, and a silently skipped prim would shrink the rig. ``pxr`` is
    imported lazily so the layout math above stays importable without USD.

    Args:
        stage: Open USD stage.
        prim_paths: Prims whose aligned world bounds are unioned.

    Returns:
        Center, top Z, X extent, and Y extent of the union bounds, in the
        argument order :func:`compute_modus_layout` expects.

    Raises:
        RuntimeError: If no path is given, a prim is missing, or a prim has
            empty world bounds.

    Example:

    .. code-block:: python

        center, top_z, x_extent, y_extent = modus_bounds_from_prims(  # doctest: +SKIP
            stage,
            ("/World/Panel/SR_0_0", "/World/Panel/SR_0_1"),
        )

    """
    from pxr import Usd, UsdGeom

    paths = tuple(str(path) for path in prim_paths)
    if not paths:
        raise RuntimeError("Modus light: no prims given to size the rig")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=True,
    )
    lo = [math.inf, math.inf, math.inf]
    hi = [-math.inf, -math.inf, -math.inf]
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Modus light: pcba_root not found: {path}")
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise RuntimeError(f"Modus light: world bbox empty for {path}")
        for axis in range(3):
            lo[axis] = min(lo[axis], float(bounds.GetMin()[axis]))
            hi[axis] = max(hi[axis], float(bounds.GetMax()[axis]))
    center = (
        (lo[0] + hi[0]) / 2.0,
        (lo[1] + hi[1]) / 2.0,
        (lo[2] + hi[2]) / 2.0,
    )
    return center, hi[2], hi[0] - lo[0], hi[1] - lo[1]
