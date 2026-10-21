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

"""Kit-free tests for scan-grid pixel layout metadata."""

from __future__ import annotations

import unittest

from pcb_aoi.geometry.scan_grid import compute_pixel_layout


class PixelLayoutTest(unittest.TestCase):
    """Verify world-grid to canonical-canvas conversion."""

    def test_six_by_eight_layout(self):
        scan_grid = {
            "projection": "orthographic",
            "footprint_x": 23.333333333333332,
            "footprint_y": 41.48148148148148,
            "x_centers": [
                17.333333333333336,
                -6.0,
                -29.333333333333336,
                -52.666666666666664,
                -76.0,
                -99.33333333333334,
            ],
            "y_centers": [
                14.004258191144025,
                1.850898904144449,
                -10.302460382855127,
                -22.455819669854705,
                -34.60917895685428,
                -46.762538243853854,
                -58.915897530853435,
                -71.06925681785302,
            ],
        }
        frame_to_grid = [
            (0, 0, 0),
            (1, 1, 0),
            (6, 0, 1),
            (47, 5, 7),
        ]

        layout = compute_pixel_layout(
            scan_grid,
            [1920, 1080],
            {"x_range": [0, 0], "y_range": [0, 0], "z_fixed": -90},
            frame_to_grid,
        )

        self.assertTrue(layout["supported"])
        self.assertEqual(
            set(layout),
            {
                "supported",
                "coordinate_system",
                "camera_side",
                "canvas_size_px",
                "tiles",
                "pixel_to_world_xy",
                "footprint_x",
                "footprint_y",
                "x_num",
                "y_num",
            },
        )
        self.assertEqual(layout["coordinate_system"], "top-left-xywh-pixels")
        self.assertEqual(layout["camera_side"], "top")
        self.assertEqual(layout["canvas_size_px"], [5858, 6480])
        self.assertEqual(
            layout["tiles"][0]["dst_rect_xywh_px"],
            [0.0, 0.0, 1920.0, 1080.0],
        )
        self.assertEqual(
            layout["tiles"][1]["dst_rect_xywh_px"],
            [0.0, 1080.0, 1920.0, 1080.0],
        )
        self.assertAlmostEqual(
            layout["tiles"][2]["dst_rect_xywh_px"][0],
            562.5269155696947,
        )
        self.assertEqual(
            layout["tiles"][2]["dst_rect_xywh_px"][1:],
            [0.0, 1920.0, 1080.0],
        )
        self.assertAlmostEqual(
            layout["tiles"][3]["dst_rect_xywh_px"][0],
            3937.6884089878627,
        )
        self.assertEqual(
            layout["tiles"][3]["dst_rect_xywh_px"][1:],
            [5400.0, 1920.0, 1080.0],
        )

    def test_partial_capture_uses_full_grid_canvas(self):
        scan_grid = {
            "projection": "orthographic",
            "footprint_x": 40.0,
            "footprint_y": 20.0,
            "x_centers": [20.0, 10.0, 0.0],
            "y_centers": [8.0, 0.0],
        }

        layout = compute_pixel_layout(
            scan_grid,
            [400, 200],
            {"x_range": [0, 0], "y_range": [0, 0], "z_fixed": 0},
            [{"frame": 9, "x_idx": 1, "y_idx": 1}],
        )

        self.assertEqual(layout["canvas_size_px"], [600, 280])
        self.assertEqual(
            layout["tiles"],
            [
                {
                    "frame": 9,
                    "x_idx": 1,
                    "y_idx": 1,
                    "dst_rect_xywh_px": [100.0, 80.0, 400.0, 200.0],
                }
            ],
        )

    def test_bottom_camera_reverses_only_image_vertical_axis(self):
        scan_grid = {
            "projection": "orthographic",
            "footprint_x": 20.0,
            "footprint_y": 20.0,
            "x_centers": [20.0, 10.0],
            "y_centers": [20.0, 10.0],
        }
        frames = [(0, 0, 0), (1, 1, 0), (2, 0, 1)]

        top = compute_pixel_layout(
            scan_grid,
            [200, 200],
            {"x_range": [0, 0], "y_range": [0, 0], "z_fixed": -90},
            frames,
        )
        bottom = compute_pixel_layout(
            scan_grid,
            [200, 200],
            {"x_range": [0, 0], "y_range": [0, 0], "z_fixed": -90},
            frames,
            camera_side="bottom",
        )

        self.assertEqual(bottom["camera_side"], "bottom")
        self.assertEqual(top["tiles"][0]["dst_rect_xywh_px"][:2], [0.0, 0.0])
        self.assertEqual(bottom["tiles"][0]["dst_rect_xywh_px"][:2], [0.0, 100.0])
        self.assertEqual(bottom["tiles"][1]["dst_rect_xywh_px"][:2], [0.0, 0.0])
        self.assertEqual(bottom["tiles"][2]["dst_rect_xywh_px"][:2], [100.0, 100.0])

    def test_pixel_to_world_affine_recovers_tile_corners(self):
        """The affine must map each tile's true footprint corners, not centers."""
        scan_grid = {
            "projection": "orthographic",
            "footprint_x": 40.0,
            "footprint_y": 20.0,
            "x_centers": [20.0, -20.0],
            "y_centers": [0.0],
        }

        layout = compute_pixel_layout(
            scan_grid,
            [400, 200],
            {"x_range": [0, 0], "y_range": [0, 0], "z_fixed": 0},
            [(0, 0, 0), (1, 1, 0)],
        )

        self.assertEqual(
            layout["pixel_to_world_xy"],
            {
                "origin_xy": [-40.0, 10.0],
                "u_per_px_xy": [0.1, 0.0],
                "v_per_px_xy": [0.0, -0.1],
            },
        )

        # Tile 1 (world x in [-40, 0]) occupies canvas [0, 400); its true
        # world top-left corner is (-40, 10), not its center (-20, 0).
        origin = layout["pixel_to_world_xy"]["origin_xy"]
        u = layout["pixel_to_world_xy"]["u_per_px_xy"]
        v = layout["pixel_to_world_xy"]["v_per_px_xy"]

        def world_at(px, py):
            return (origin[0] + px * u[0] + py * v[0], origin[1] + px * u[1] + py * v[1])

        self.assertEqual(world_at(0, 0), (-40.0, 10.0))
        self.assertEqual(world_at(400, 200), (0.0, -10.0))
        # Canvas midline at pixel 400 is the world x=0 seam between tiles.
        self.assertEqual(world_at(400, 0), (0.0, 10.0))

    def test_unsupported_camera_geometry_reports_reason(self):
        scan_grid = {
            "projection": "orthographic",
            "footprint_x": 40.0,
            "footprint_y": 20.0,
            "x_centers": [0.0],
            "y_centers": [0.0],
        }

        tilted = compute_pixel_layout(
            scan_grid,
            [400, 200],
            {"x_range": [1, 1], "y_range": [0, 0], "z_fixed": 0},
            [(0, 0, 0)],
        )
        rotated = compute_pixel_layout(
            scan_grid,
            [400, 200],
            {"x_range": [0, 0], "y_range": [0, 0], "z_fixed": 45},
            [(0, 0, 0)],
        )
        perspective = compute_pixel_layout(
            {**scan_grid, "projection": "perspective"},
            [400, 200],
            {"x_range": [0, 0], "y_range": [0, 0], "z_fixed": 0},
            [(0, 0, 0)],
        )

        self.assertFalse(tilted["supported"])
        self.assertIn("zero camera X/Y tilt", tilted["reason"])
        self.assertFalse(rotated["supported"])
        self.assertIn("multiple of 90", rotated["reason"])
        self.assertFalse(perspective["supported"])
        self.assertIn("orthographic camera", perspective["reason"])


if __name__ == "__main__":
    unittest.main()
