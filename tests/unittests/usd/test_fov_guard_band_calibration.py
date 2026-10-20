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

"""Calibration of a guard-banded render describes the delivered tile.

Needs pxr, so it lives in the usd lane; the rest of the guard-band tests are in
``tests/unittests/core/declarative_api/test_fov_guard_band.py``.
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from cdf.execution.isaac import guard_band

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:  # pragma: no cover - exercised by the usd lane only
    Usd = None


@unittest.skipIf(Usd is None, "pxr not available")
class CalibrationCropTest(unittest.TestCase):
    """The calibration of a cropped render is the calibration of the tile."""

    def test_guarded_render_calibrates_as_the_delivered_tile(self):
        def calibrate(aperture, resolution, crop):
            from cdf.execution.isaac.camera_calibration import (
                orthographic_calibration,
            )

            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageMetersPerUnit(stage, 0.001)
            xform = UsdGeom.Xform.Define(stage, "/World/camera_light")
            xform.AddTranslateOp().Set(Gf.Vec3d(5.0, 6.0, 50.0))
            camera = UsdGeom.Camera.Define(stage, "/World/camera_light/Camera")
            camera.GetProjectionAttr().Set("orthographic")
            camera.GetHorizontalApertureAttr().Set(aperture[0])
            camera.GetVerticalApertureAttr().Set(aperture[1])
            return orthographic_calibration(
                stage, "/World/camera_light/Camera", resolution, crop_px=crop
            )

        plain = calibrate((800.0, 600.0), (800, 600), 0)
        guarded = calibrate(
            (
                guard_band.render_aperture(800.0, 800, 8),
                guard_band.render_aperture(600.0, 600, 8),
            ),
            (816, 616),
            8,
        )
        self.assertEqual(sorted(plain), sorted(guarded))
        self.assertEqual(guarded["resolution_px"], [800, 600])
        self.assertEqual(guarded["mm_per_px"], plain["mm_per_px"])
        self.assertAlmostEqual(guarded["horizontal_aperture_mm"], 80.0, places=5)
        self.assertAlmostEqual(guarded["vertical_aperture_mm"], 60.0, places=5)
        self.assertTrue(
            np.allclose(
                guarded["intrinsic_matrix"], plain["intrinsic_matrix"], atol=1e-6
            )
        )
        self.assertEqual(guarded["camera_to_world"], plain["camera_to_world"])
        self.assertEqual(json.dumps(sorted(guarded)), json.dumps(sorted(plain)))


if __name__ == "__main__":
    unittest.main()
