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

"""Per-frame calibration of the orthographic capture camera."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from cdf.execution.dataset_manifest import ManifestSerializer
from cdf.execution.isaac.camera_calibration import orthographic_calibration
from cdf.execution.isaac.capture_execution import _write_pass_calibration

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:  # pragma: no cover - exercised by the usd lane only
    Usd = None


def _stage_with_camera(*, position, aperture_h, aperture_v=None, meters_per_unit=0.001):
    """Return an in-memory millimetre stage with one orthographic camera."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    xform = UsdGeom.Xform.Define(stage, "/World/camera_light")
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    camera = UsdGeom.Camera.Define(stage, "/World/camera_light/Camera")
    camera.GetProjectionAttr().Set("orthographic")
    # USD apertures are tenths of the stage unit: 90 mm of board -> 900
    camera.GetHorizontalApertureAttr().Set(aperture_h)
    if aperture_v is not None:
        camera.GetVerticalApertureAttr().Set(aperture_v)
    return stage


@unittest.skipIf(Usd is None, "pxr not available")
class OrthographicCalibrationTest(unittest.TestCase):
    """The calibration must let a reader turn a pixel back into millimetres."""

    def test_pixel_pitch_comes_from_aperture_and_resolution(self):
        """A 90 mm aperture over 900 px is 0.1 mm per pixel, in millimetres."""
        stage = _stage_with_camera(
            position=(10.0, 20.0, 50.0), aperture_h=900.0, aperture_v=900.0
        )
        cal = orthographic_calibration(stage, "/World/camera_light/Camera", (900, 900))
        self.assertEqual(cal["projection"], "orthographic")
        self.assertAlmostEqual(cal["horizontal_aperture_mm"], 90.0)
        self.assertAlmostEqual(cal["vertical_aperture_mm"], 90.0)
        self.assertEqual(cal["mm_per_px"], [0.1, 0.1])
        self.assertEqual(cal["resolution_px"], [900, 900])

    def test_intrinsic_maps_camera_millimetres_to_pixels(self):
        """The camera centre lands on the image centre and +y_cam goes up the image."""
        stage = _stage_with_camera(
            position=(0.0, 0.0, 50.0), aperture_h=900.0, aperture_v=900.0
        )
        cal = orthographic_calibration(stage, "/World/camera_light/Camera", (900, 900))
        k = cal["intrinsic_matrix"]
        centre = [k[0][0] * 0 + k[0][2], k[1][1] * 0 + k[1][2]]
        self.assertEqual(centre, [450.0, 450.0])
        # 45 mm to the right and 45 mm up in camera frame -> right edge, top row
        px = k[0][0] * 45.0 + k[0][2]
        py = k[1][1] * 45.0 + k[1][2]
        self.assertAlmostEqual(px, 900.0)
        self.assertAlmostEqual(py, 0.0)

    def test_extrinsics_invert_each_other_and_carry_the_position(self):
        """world_to_camera undoes camera_to_world; position is the camera translation."""
        stage = _stage_with_camera(position=(12.5, -3.0, 40.0), aperture_h=200.0)
        cal = orthographic_calibration(stage, "/World/camera_light/Camera", (400, 300))
        self.assertEqual(cal["position"], [12.5, -3.0, 40.0])
        c2w = Gf.Matrix4d(*[v for row in cal["camera_to_world"] for v in row])
        w2c = Gf.Matrix4d(*[v for row in cal["world_to_camera"] for v in row])
        product = c2w * w2c
        for r in range(4):
            for c in range(4):
                self.assertAlmostEqual(product[r][c], 1.0 if r == c else 0.0, places=9)
        # a world point at the camera position maps to the camera origin
        origin = w2c.Transform(Gf.Vec3d(12.5, -3.0, 40.0))
        self.assertAlmostEqual(origin[0], 0.0)
        self.assertAlmostEqual(origin[1], 0.0)
        self.assertAlmostEqual(origin[2], 0.0)
        # the camera looks down its local -Z
        self.assertEqual([round(v, 6) for v in cal["view_direction"]], [0.0, 0.0, -1.0])

    def test_vertical_aperture_falls_back_to_the_render_aspect(self):
        """Only a horizontal aperture authored: the vertical one follows the aspect."""
        stage = _stage_with_camera(position=(0.0, 0.0, 10.0), aperture_h=400.0)
        cal = orthographic_calibration(stage, "/World/camera_light/Camera", (400, 300))
        self.assertAlmostEqual(cal["horizontal_aperture_mm"], 40.0)
        self.assertAlmostEqual(cal["vertical_aperture_mm"], 30.0)
        self.assertEqual(cal["mm_per_px"], [0.1, 0.1])

    def test_stage_units_are_normalised_to_millimetres(self):
        """A centimetre stage reports the same millimetre pitch as a millimetre one."""
        stage = _stage_with_camera(
            position=(0.0, 0.0, 5.0),
            aperture_h=90.0,
            aperture_v=90.0,
            meters_per_unit=0.01,
        )
        cal = orthographic_calibration(stage, "/World/camera_light/Camera", (900, 900))
        self.assertAlmostEqual(cal["horizontal_aperture_mm"], 90.0)
        self.assertEqual(cal["mm_per_px"], [0.1, 0.1])
        self.assertEqual(cal["stage_meters_per_unit"], 0.01)

    def test_missing_camera_is_an_error(self):
        """A wrong camera path is reported, not silently calibrated as identity."""
        stage = Usd.Stage.CreateInMemory()
        with self.assertRaises(ValueError):
            orthographic_calibration(stage, "/World/nope", (10, 10))


class WritePassCalibrationTest(unittest.TestCase):
    """The calibration is also written beside the frames."""

    def test_writes_one_entry_per_calibrated_frame(self):
        """Frames without a calibration are skipped; the rest land in calibration.json."""
        frames = [
            {
                "frame": 0,
                "frame_id": "p:0",
                "role": "capture",
                "camera": {"calibration": {"mm_per_px": [0.1, 0.1]}},
            },
            {
                "frame": 1,
                "frame_id": "p:1",
                "role": "capture",
                "camera": {"position": [0, 0, 1]},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            artifact = _write_pass_calibration(
                Path(tmp),
                tmp,
                frames,
                ManifestSerializer(),
                "capture",
                0,
            )
            # The sidecar is only part of the dataset if it comes back as a
            # manifest artifact: its name carries no frame number, so
            # _discover_frame_artifacts would never find it on disk.
            self.assertEqual(artifact["kind"], "camera_calibration")
            self.assertEqual(artifact["path"], "calibration.json")
            self.assertEqual(artifact["role"], "capture")
            self.assertEqual(artifact["trigger_idx"], 0)
            self.assertNotIn("frame_id", artifact)
            with open(
                os.path.join(tmp, "calibration.json"), encoding="utf-8"
            ) as handle:
                doc = json.load(handle)
        self.assertEqual(doc["schema"], "cdf.camera_calibration")
        self.assertEqual([e["frame"] for e in doc["frames"]], [0])
        self.assertEqual(doc["frames"][0]["mm_per_px"], [0.1, 0.1])

    def test_nothing_to_write_returns_none(self):
        """No calibrated frame, no file."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                _write_pass_calibration(
                    Path(tmp),
                    tmp,
                    [{"frame": 0, "camera": {}}],
                    ManifestSerializer(),
                    "capture",
                    0,
                )
            )
            self.assertFalse(os.path.exists(os.path.join(tmp, "calibration.json")))


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(Usd is None, "pxr not available")
class TiltedRoiAimTest(unittest.TestCase):
    """A tilted focused-ROI camera must still look through its part."""

    def _aim(self, rotation):
        from cdf.execution.isaac.capture_execution import _aimed_roi_translate

        stage = _stage_with_camera(position=(28.115, 110.535, 27.79), aperture_h=50.0)
        camera = UsdGeom.Xformable(stage.GetPrimAtPath("/World/camera_light/Camera"))
        camera.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        target = (28.115, 110.535, 1.6)
        translate = _aimed_roi_translate(
            stage, "/World/camera_light/Camera", (28.115, 110.535, 27.79), target
        )
        UsdGeom.Xformable(
            stage.GetPrimAtPath("/World/camera_light")
        ).GetOrderedXformOps()[0].Set(Gf.Vec3d(*translate))
        cal = orthographic_calibration(stage, "/World/camera_light/Camera", (64, 64))
        return cal, target

    def test_line_of_sight_passes_through_the_target(self):
        for rotation in ((45.0, 0.0, 0.0), (0.0, 45.0, 0.0), (20.0, -30.0, -90.0)):
            with self.subTest(rotation=rotation):
                cal, target = self._aim(rotation)
                position, view = cal["position"], cal["view_direction"]
                self.assertAlmostEqual(position[2], 27.79, places=6)
                distance = (target[2] - position[2]) / view[2]
                for axis in range(3):
                    self.assertAlmostEqual(
                        position[axis] + distance * view[axis], target[axis], places=6
                    )

    def test_top_down_camera_is_not_moved(self):
        cal, _target = self._aim((0.0, 0.0, -90.0))
        self.assertEqual(cal["position"], [28.115, 110.535, 27.79])
