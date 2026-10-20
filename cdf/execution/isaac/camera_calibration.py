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

"""Per-frame calibration of the orthographic capture camera.

A capture frame is only reusable as spatial ground truth if the reader can
turn a pixel back into a board coordinate. The capture loop knows everything
needed for that at the moment it steps the renderer -- the camera prim's world
transform and its apertures -- but until now it only recorded the camera
position. This module turns that state into the two matrices a consumer
expects: an intrinsic matrix that maps camera-frame millimetres to pixels, and
the extrinsic world-to-camera transform.

The capture camera is orthographic, so the intrinsic matrix has no focal
length: pixels per millimetre along each axis and the image centre. USD
authors camera apertures in tenths of the stage's linear unit; the values here
are converted to millimetres through ``metersPerUnit`` so the numbers mean the
same thing on every board regardless of how its stage was authored.
"""

from __future__ import annotations

from collections.abc import Sequence

#: USD camera apertures are authored in tenths of the stage's linear unit.
_USD_APERTURE_SCALE = 0.1


def _matrix_rows(matrix: object) -> list[list[float]]:
    """Return a ``Gf.Matrix4d`` as four row lists of floats."""
    return [[float(matrix[r][c]) for c in range(4)] for r in range(4)]


def orthographic_calibration(
    stage: object,
    camera_path: str,
    resolution: Sequence[int],
    *,
    crop_px: int = 0,
) -> dict[str, object]:
    """Describe the orthographic capture camera as it stands on ``stage``.

    Reads the camera's authored apertures and its composed world transform, so
    it must be called after the frame's position and apertures have been set
    and before the next frame moves them.

    Args:
        stage: Open USD stage holding the capture camera.
        camera_path: Absolute prim path of the ``UsdGeom.Camera``.
        resolution: Render resolution as ``(width, height)`` in pixels.
        crop_px: Pixels cropped from every side of the rendered frame before it
            is delivered (a capture guard band). The result then describes the
            delivered frame: ``crop_px`` fewer pixels per edge and an aperture
            smaller by the same number of pixel pitches, about the same centre,
            so the camera pose is unchanged.

    Returns:
        A JSON-serialisable mapping with ``projection``, apertures and
        pixel pitch in millimetres, the ``intrinsic_matrix`` (3x3, camera-frame
        mm to pixels, image origin top-left, +y down), and the extrinsic
        ``world_to_camera`` / ``camera_to_world`` matrices (4x4, row-major,
        stage units, row-vector convention ``world = local @ camera_to_world``).

    Raises:
        ValueError: If ``camera_path`` is not a valid camera prim, or
            ``crop_px`` leaves no pixels.

    Example:

    .. code-block:: python

        calibration = orthographic_calibration(stage, "/World/camera_light/Camera", (900, 900))
        print(calibration["mm_per_px"])  # doctest: +SKIP
        [0.00311, 0.00311]

    """
    from pxr import Usd, UsdGeom

    camera = UsdGeom.Camera.Get(stage, camera_path)
    prim = camera.GetPrim()
    if not prim.IsValid():
        raise ValueError(f"camera calibration: {camera_path!r} is not a camera prim")

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 0.01)
    unit_to_mm = meters_per_unit * 1000.0
    width_px, height_px = int(resolution[0]), int(resolution[1])

    # UsdGeom.Camera has schema fallbacks (35 mm film) for both apertures, so
    # Get() never returns None; only an authored value describes this capture.
    h_attr = camera.GetHorizontalApertureAttr()
    v_attr = camera.GetVerticalApertureAttr()
    aperture_h_mm = (
        float(h_attr.Get()) * _USD_APERTURE_SCALE * unit_to_mm
        if h_attr.HasAuthoredValue()
        else None
    )
    aperture_v_mm = (
        float(v_attr.Get()) * _USD_APERTURE_SCALE * unit_to_mm
        if v_attr.HasAuthoredValue()
        else None
    )
    if aperture_v_mm is None and aperture_h_mm is not None and width_px:
        # The renderer derives the vertical extent from the render aspect when
        # only the horizontal aperture is authored; mirror that here.
        aperture_v_mm = aperture_h_mm * height_px / width_px
    if crop_px:
        # Orthographic, so an aperture is linear in pixels: the delivered frame
        # keeps the rendered pitch and loses crop_px pitches per edge.
        delivered_w, delivered_h = width_px - 2 * crop_px, height_px - 2 * crop_px
        if delivered_w < 1 or delivered_h < 1:
            raise ValueError(
                f"camera calibration: crop_px={crop_px} leaves no pixels of a "
                f"{width_px}x{height_px} frame"
            )
        if aperture_h_mm is not None:
            aperture_h_mm = aperture_h_mm * delivered_w / width_px
        if aperture_v_mm is not None:
            aperture_v_mm = aperture_v_mm * delivered_h / height_px
        width_px, height_px = delivered_w, delivered_h

    projection = camera.GetProjectionAttr().Get()
    xform = UsdGeom.Xformable(prim)
    camera_to_world = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_to_camera = camera_to_world.GetInverse()
    position = camera_to_world.ExtractTranslation()
    # USD cameras look down their local -Z with +Y up.
    view_direction = camera_to_world.TransformDir((0.0, 0.0, -1.0))
    up_direction = camera_to_world.TransformDir((0.0, 1.0, 0.0))

    mm_per_px = None
    intrinsic = None
    if aperture_h_mm and aperture_v_mm and width_px and height_px:
        px_per_mm_x = width_px / aperture_h_mm
        px_per_mm_y = height_px / aperture_v_mm
        mm_per_px = [round(1.0 / px_per_mm_x, 6), round(1.0 / px_per_mm_y, 6)]
        # Camera frame: +x right, +y up. Image frame: +y down, origin top-left.
        intrinsic = [
            [px_per_mm_x, 0.0, width_px / 2.0],
            [0.0, -px_per_mm_y, height_px / 2.0],
            [0.0, 0.0, 1.0],
        ]

    return {
        "projection": str(projection) if projection is not None else None,
        "camera_path": str(camera_path),
        "resolution_px": [width_px, height_px],
        "horizontal_aperture_mm": aperture_h_mm,
        "vertical_aperture_mm": aperture_v_mm,
        "mm_per_px": mm_per_px,
        "intrinsic_matrix": intrinsic,
        "intrinsic_convention": (
            "pixel = K @ [x_cam_mm, y_cam_mm, 1]; orthographic, so no depth term; "
            "image origin top-left, +y down"
        ),
        "world_to_camera": _matrix_rows(world_to_camera),
        "camera_to_world": _matrix_rows(camera_to_world),
        "matrix_convention": "row-major, row-vector: world = local @ camera_to_world",
        "position": [float(v) for v in position],
        "view_direction": [float(v) for v in view_direction],
        "up_direction": [float(v) for v in up_direction],
        "stage_meters_per_unit": meters_per_unit,
    }
