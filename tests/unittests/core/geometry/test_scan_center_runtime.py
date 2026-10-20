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

"""``scan_center_mm`` on the runtime path: stage scale conversion and sidecar.

The geometry layer is tested stage-free elsewhere. This covers the adapter in
``scripts/sdg/standalone/common.py`` that converts the requested center with the
stage's (or the job's overriding) metersPerUnit before tiling, and the stitch
that carries the requested center into the image-to-USD sidecar.
"""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from PIL import Image

from cdf.postprocess.image_stitching import stitch_capture_grid
from pcb_aoi.geometry.fov import FovError, check_square_pixels, mm_to_su
from pcb_aoi.geometry.fov import scan_region_su as fov_scan_region_su
from pcb_aoi.geometry.scan_grid import (
    auto_complete_grid_nums,
    compute_scan_grid_geometry,
)

_COMMON = (
    Path(__file__).resolve().parents[4] / "scripts" / "sdg" / "standalone" / "common.py"
)


def _resolve_scan_grid_from_stage(namespace: dict[str, Any]):
    """Compile only ``resolve_scan_grid_from_stage``; the module needs Kit to import."""
    source = ast.parse(_COMMON.read_text(encoding="utf-8"))
    function = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_scan_grid_from_stage"
    )
    exec(  # noqa: S102 -- a fixed, repository-owned function
        compile(ast.Module(body=[function], type_ignores=[]), str(_COMMON), "exec"),
        namespace,
    )
    return namespace["resolve_scan_grid_from_stage"]


def _fake_usd(declared_meters_per_unit: float):
    """UsdGeom / Usd stand-ins: an orthographic camera and a 200 x 100 su board."""
    camera = mock.Mock()
    camera.GetProjectionAttr.return_value.Get.return_value = "orthographic"
    camera.GetHorizontalApertureAttr.return_value.Get.return_value = 100.0
    camera.GetVerticalApertureAttr.return_value.Get.return_value = 100.0
    camera.GetFocalLengthAttr.return_value.Get.return_value = None
    bound = mock.Mock()
    bound.IsEmpty.return_value = False
    bound.GetMin.return_value = (0.0, 0.0, 0.0)
    bound.GetMax.return_value = (200.0, 100.0, 1.0)
    usd_geom = mock.Mock()
    usd_geom.Camera.return_value = camera
    usd_geom.BBoxCache.return_value.ComputeWorldBound.return_value.GetRange.return_value = bound
    usd_geom.GetStageMetersPerUnit.return_value = declared_meters_per_unit
    return usd_geom, mock.Mock()


class ScanCenterStageScaleTest(unittest.TestCase):
    """The requested center reaches the tiler in the stage's own units."""

    def _run(self, declared, **overrides):
        usd_geom, usd = _fake_usd(declared)
        geometry = mock.Mock(side_effect=compute_scan_grid_geometry)
        resolve = _resolve_scan_grid_from_stage(
            {
                "Any": Any,
                "UsdGeom": usd_geom,
                "Usd": usd,
                "mm_to_su": mm_to_su,
                "check_square_pixels": check_square_pixels,
                "fov_scan_region_su": fov_scan_region_su,
                "auto_complete_grid_nums": auto_complete_grid_nums,
                "compute_scan_grid_geometry": geometry,
            }
        )
        stage = mock.Mock()
        stage.GetPrimAtPath.return_value.IsValid.return_value = True
        cfg = {
            "camera_path": "/Camera",
            "pcba_root": "/Board",
            "scan_grid": {"x_num": 1, "y_num": 1},
            "scan_area_mm": [30.0, 20.0],
            "scan_center_mm": [155.0, 72.0],
            "camera_rotation": {"z_fixed": 0.0},
            **overrides,
        }
        with mock.patch("builtins.print"):
            resolve(stage, cfg)
        geometry.assert_called_once()
        return geometry.call_args.kwargs["scan_region_center_su"]

    def test_a_millimetre_stage_keeps_the_millimetres(self):
        center = self._run(0.001)
        self.assertAlmostEqual(center[0], 155.0)
        self.assertAlmostEqual(center[1], 72.0)

    def test_a_centimetre_stage_divides_by_ten(self):
        center = self._run(0.01)
        self.assertAlmostEqual(center[0], 15.5)
        self.assertAlmostEqual(center[1], 7.2)

    def test_the_job_override_wins_over_the_stage_declaration(self):
        for key in ("meters_per_unit_override", "meters_per_unit"):
            with self.subTest(key=key):
                center = self._run(0.01, **{key: 0.001})
                self.assertAlmostEqual(center[0], 155.0)
                self.assertAlmostEqual(center[1], 72.0)

    def test_no_center_leaves_the_tiler_on_the_board(self):
        self.assertIsNone(self._run(0.001, scan_center_mm=None))


class SquarePixelGuardRuntimeTest(unittest.TestCase):
    """A hand-authored grid whose apertures and resolution disagree is refused.

    ``capture.fov`` never produces one; ``scan_grid`` with ``scan_area_mm``
    beside an authored ``renderer.resolution`` can, and the renderer would
    keep the horizontal pitch on both axes and seam every tile row
    (SDGW-1359).
    """

    _run = ScanCenterStageScaleTest._run

    def test_a_non_square_authored_frame_is_refused(self):
        # Hand-authored regression values: 0.041540 mm/px across world X,
        # 0.041316 down world Y.
        for yaw, resolution in ((0.0, [1125, 1406]), (-90.0, [1406, 1125])):
            with (
                self.subTest(yaw=yaw),
                self.assertRaisesRegex(FovError, "non-square pixels"),
            ):
                self._run(
                    0.001,
                    scan_grid={"x_num": 6, "y_num": 6},
                    scan_area_mm=[280.398, 348.54],
                    scan_center_mm=None,
                    resolution=resolution,
                    camera_rotation={"z_fixed": yaw},
                )

    def test_a_square_authored_frame_is_silent(self):
        for yaw, resolution in ((0.0, [1125, 1406]), (-90.0, [1406, 1125])):
            with self.subTest(yaw=yaw):
                self._run(
                    0.001,
                    scan_grid={"x_num": 6, "y_num": 6},
                    scan_area_mm=[280.398, 280.398 / 6750 * 8436],
                    scan_center_mm=None,
                    resolution=resolution,
                    camera_rotation={"z_fixed": yaw},
                )


class ScanCenterSidecarTest(unittest.TestCase):
    """The stitched sidecar records the center the capture asked for."""

    def test_the_requested_center_reaches_the_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trigger = root / "trigger_0000"
            trigger.mkdir()
            Image.new("RGB", (2, 2), (255, 0, 0)).save(trigger / "rgb_0000.png")
            metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "supported": True,
                        "canvas_size_px": [2, 2],
                        "camera_side": "top",
                        "world_reference_z": 1.0,
                        "stage_meters_per_unit": 0.001,
                        "stage_up_axis": "Z",
                        "x_num": 1,
                        "y_num": 1,
                        "scan_center_mm": [155.0, 72.0],
                        "pixel_to_world_xy": {
                            "origin_xy": [154.0, 73.0],
                            "u_per_px_xy": [1.0, 0.0],
                            "v_per_px_xy": [0.0, -1.0],
                        },
                        "tiles": [{"frame": 0, "dst_rect_xywh_px": [0, 0, 2, 2]}],
                    },
                },
            ]
            artifacts = stitch_capture_grid(root, metadata, output_filename="board.png")
            sidecar = json.loads(
                Path(artifacts[0]["image_to_usd_path"]).read_text(encoding="utf-8"),
            )
        self.assertEqual(sidecar["fov"]["center_mm"], [155.0, 72.0])
        self.assertIsNone(sidecar["fov"]["centered_on"])


if __name__ == "__main__":
    unittest.main()
