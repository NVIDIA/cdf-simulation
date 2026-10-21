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

"""Field-of-view resolution: millimetres and dpi in, camera quantities out."""

import unittest
from typing import ClassVar

from cdf.postprocess.image_stitching import _fov_from_layout
from cdf.execution.isaac.capture_execution import _scanned_bounds_xy
from pcb_aoi.geometry.fov import (
    DEFAULT_DPI,
    SENSOR_DPI,
    SUPPORTED_DPI,
    FovError,
    aperture_to_usd,
    check_square_pixels,
    mm_to_su,
    resolve_fov,
    scan_region_su,
    su_to_mm,
)
from pcb_aoi.geometry.scan_grid import (
    compute_pixel_layout,
    compute_scan_grid_geometry,
)

CM_STAGE = 0.01
MM_STAGE = 0.001

_BBOX_MIN = (0.0, 0.0, 0.0)
_BBOX_MAX = (183.874, 120.0, 1.6)

#: A capture at a quarter turn: two cells across world X, three
#: along world Y, so the camera's horizontal aperture covers world Y.
_QUARTER_TURN = {
    "horizontal_aperture_su": 50.0,
    "vertical_aperture_su": 60.03,
    "x_num": 2,
    "y_num": 3,
    "camera_z_rotation_deg": -90.0,
}
_REGION = (120.06, 150.0)


def _geometry(**kwargs):
    """Orthographic grid over the reference board."""
    return compute_scan_grid_geometry(
        _BBOX_MIN,
        _BBOX_MAX,
        projection="orthographic",
        focal_length_su=None,
        **kwargs,
    )


class ResolveFovTest(unittest.TestCase):
    """The arithmetic that turns a scan rectangle into camera quantities."""

    def test_scanner_default_600dpi_gives_square_pixels(self):
        """An isotropic dpi request must not produce a rectangular pitch.

        This is the whole reason pitch is primary: rounding a pixel count and
        dividing it back into the requested millimetres yields pitches that
        differ in the fifth decimal, which is a non-square pixel nobody asked
        for.
        """
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[300.0, 200.0],
            dpi=DEFAULT_DPI,
        )
        px_x, px_y = r["mm_per_px"]
        self.assertAlmostEqual(px_x, px_y, delta=1e-15)
        for value in r["dpi_effective"]:
            self.assertAlmostEqual(value, 600.0, places=9)

    def test_delivered_area_is_the_scanner_pixel_count_times_the_pitch(self):
        """The image is area / pitch rounded once, so it covers within a pixel."""
        for area in ([300.0, 200.0], [137.3, 91.7], [480.0, 360.0]):
            with self.subTest(area=area):
                r = resolve_fov(
                    stage_meters_per_unit=CM_STAGE,
                    scan_area_mm=area,
                    dpi=DEFAULT_DPI,
                )
                got = r["scan_area_mm_effective"]
                for axis in (0, 1):
                    pitch = r["mm_per_px"][axis]
                    self.assertAlmostEqual(
                        got[axis], r["image_size_px"][axis] * pitch, places=9
                    )
                    # Rounded down: never more than the request, and short of
                    # it by less than one pixel.
                    self.assertLessEqual(got[axis], area[axis] + 1e-9)
                    self.assertLess(area[axis] - got[axis], pitch)

    def test_apertures_reconstruct_the_rendered_area(self):
        """Orthographic: aperture IS the per-cell footprint, so the grid closes."""
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[300.0, 200.0],
            dpi=DEFAULT_DPI,
        )
        total_x = su_to_mm(r["horizontal_aperture_su"] * r["x_num"], CM_STAGE)
        total_y = su_to_mm(r["vertical_aperture_su"] * r["y_num"], CM_STAGE)
        self.assertAlmostEqual(total_x, r["scan_area_mm_rendered"][0], places=9)
        self.assertAlmostEqual(total_y, r["scan_area_mm_rendered"][1], places=9)

    def test_the_mosaic_covers_the_image_and_is_cropped_by_under_a_pixel_per_cell(self):
        """Cells are equal whole pixels; the stitch crops the surplus off."""
        for area, dpi in (([300.0, 200.0], 600.0), ([100.0, 200.0], 1200.0)):
            with self.subTest(area=area, dpi=dpi):
                r = resolve_fov(
                    stage_meters_per_unit=CM_STAGE, scan_area_mm=area, dpi=dpi
                )
                nums = (r["x_num"], r["y_num"])
                for axis in (0, 1):
                    mosaic = r["mosaic_size_px"][axis]
                    image = r["image_size_px"][axis]
                    self.assertEqual(mosaic, nums[axis] * r["tile_size_px"][axis])
                    self.assertGreaterEqual(mosaic, image)
                    self.assertLessEqual(mosaic - image, nums[axis] - 1)

    def test_no_tiling_when_the_canvas_fits(self):
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[50.0, 40.0],
            dpi=DEFAULT_DPI,
        )
        self.assertEqual((r["x_num"], r["y_num"]), (1, 1))

    def test_pixel_count_route_is_the_inverse(self):
        """Asking for a pixel canvas lands on the dpi that canvas implies."""
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[300.0, 300.0],
            image_size_px=[8000, 8000],
        )
        for value in r["dpi_effective"]:
            self.assertAlmostEqual(value, 8000 * 25.4 / 300.0, places=9)

    def test_an_anisotropic_dpi_is_refused(self):
        """The renderer cannot produce non-square pixels, so it is not accepted.

        An orthographic RTX camera takes its vertical extent from the
        horizontal aperture and the image aspect: a pair rendered the X pitch
        on both axes and seamed every tile row.
        """
        with self.assertRaisesRegex(FovError, "non-square pixels"):
            resolve_fov(
                stage_meters_per_unit=CM_STAGE,
                scan_area_mm=[300.0, 200.0],
                dpi=[600.0, 300.0],
            )

    def test_an_equal_dpi_pair_is_one_dpi(self):
        for pair in ([600.0, 600.0], [600.0, 600.0 * (1 + 1e-9)]):
            with self.subTest(pair=pair):
                r = resolve_fov(
                    stage_meters_per_unit=CM_STAGE,
                    scan_area_mm=[100.0, 200.0],
                    dpi=pair,
                )
                self.assertEqual(r["image_size_px"], [2362, 4724])
                self.assertEqual(r["mm_per_px"][0], r["mm_per_px"][1])

    def test_centre_places_the_bounds(self):
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[100.0, 50.0],
            dpi=DEFAULT_DPI,
            centre_xy_su=[3.0, -2.0],
            centered_on="instance",
        )
        lo = r["bounds_su"]["min_xy"]
        hi = r["bounds_su"]["max_xy"]
        self.assertAlmostEqual((lo[0] + hi[0]) / 2, 3.0, places=9)
        self.assertAlmostEqual((lo[1] + hi[1]) / 2, -2.0, places=9)
        self.assertEqual(r["centered_on"], "instance")
        self.assertTrue(any("centered_on=instance" in a for a in r["assumptions"]))

    def test_perspective_is_refused(self):
        with self.assertRaisesRegex(FovError, "orthographic"):
            resolve_fov(
                stage_meters_per_unit=CM_STAGE,
                scan_area_mm=[10.0, 10.0],
                dpi=DEFAULT_DPI,
                projection="perspective",
            )

    def test_over_and_under_determined_inputs_are_refused(self):
        with self.assertRaisesRegex(FovError, "exactly one"):
            resolve_fov(
                stage_meters_per_unit=CM_STAGE,
                scan_area_mm=[10.0, 10.0],
                dpi=DEFAULT_DPI,
                image_size_px=[100, 100],
            )
        with self.assertRaisesRegex(FovError, "exactly one"):
            resolve_fov(stage_meters_per_unit=CM_STAGE, scan_area_mm=[10.0, 10.0])

    def test_non_positive_inputs_are_refused(self):
        for bad in (0.0, -1.0):
            with self.subTest(bad=bad):
                with self.assertRaises(FovError):
                    resolve_fov(
                        stage_meters_per_unit=CM_STAGE,
                        scan_area_mm=[bad, 10.0],
                        dpi=DEFAULT_DPI,
                    )
                with self.assertRaises(FovError):
                    resolve_fov(
                        stage_meters_per_unit=bad,
                        scan_area_mm=[10.0, 10.0],
                        dpi=DEFAULT_DPI,
                    )

    def test_an_unrecognized_centered_on_is_refused(self):
        with self.assertRaisesRegex(FovError, "centered_on"):
            resolve_fov(
                stage_meters_per_unit=CM_STAGE,
                scan_area_mm=[10.0, 10.0],
                dpi=DEFAULT_DPI,
                centered_on="banana",
            )

    def test_a_malformed_image_size_px_is_refused_not_ignored(self):
        for image_size_px in ([100], [100, 100, 100]):
            with self.subTest(image_size_px=image_size_px):
                with self.assertRaisesRegex(FovError, "image_size_px"):
                    resolve_fov(
                        stage_meters_per_unit=CM_STAGE,
                        scan_area_mm=[10.0, 10.0],
                        image_size_px=image_size_px,
                    )

    def test_a_malformed_max_tile_px_is_refused_not_ignored(self):
        for max_tile_px in ([4096], [4096, 4096, 4096]):
            with self.subTest(max_tile_px=max_tile_px):
                with self.assertRaisesRegex(FovError, "max_tile_px"):
                    resolve_fov(
                        stage_meters_per_unit=CM_STAGE,
                        scan_area_mm=[10.0, 10.0],
                        dpi=DEFAULT_DPI,
                        max_tile_px=max_tile_px,
                    )

    def test_usd_aperture_is_tenths_of_a_scene_unit(self):
        """The film-format convention.

        Writing scene units straight onto the camera is out by ten and still
        renders, just framed wrong.
        """
        self.assertAlmostEqual(aperture_to_usd(15.0), 150.0, places=9)

    def test_mm_su_round_trip(self):
        for mpu in (CM_STAGE, MM_STAGE):
            with self.subTest(mpu=mpu):
                self.assertAlmostEqual(
                    su_to_mm(mm_to_su(37.5, mpu), mpu),
                    37.5,
                    places=9,
                )


class ScanRegionTest(unittest.TestCase):
    """Tiling a rectangle deliberately smaller than the board.

    ``compute_scan_grid_geometry`` otherwise asserts that the cells cover the
    board, which is right for every capture that existed before a field of view
    could be stated, and exactly wrong for one that can: an AOI scans the
    smallest rectangle containing the product and is meant to miss the rest.
    """

    def test_a_region_smaller_than_the_board_is_allowed(self):
        geometry = _geometry(**_QUARTER_TURN, scan_region_su=_REGION)
        # 120.06 mm spans two thirds of a 183.874 mm board.
        self.assertAlmostEqual(geometry["span_x"], 120.06, places=9)
        self.assertAlmostEqual(geometry["footprint_x"], 60.03, places=9)
        self.assertAlmostEqual(geometry["x_step"], 60.03, places=9)

    def test_the_same_request_is_refused_without_a_region(self):
        """The board-covering assertion still holds for everything else."""
        with self.assertRaisesRegex(ValueError, "bbox"):
            _geometry(**_QUARTER_TURN)

    def test_cells_tile_the_region_without_gap_or_overlap(self):
        """Steps equal footprints, so the union is exactly the region."""
        geometry = _geometry(**_QUARTER_TURN, scan_region_su=_REGION)
        self.assertAlmostEqual(
            geometry["x_num"] * geometry["footprint_x"],
            geometry["span_x"],
            places=9,
        )
        self.assertAlmostEqual(
            geometry["y_num"] * geometry["footprint_y"],
            geometry["span_y"],
            places=9,
        )

    def test_the_region_is_centered_on_the_board(self):
        geometry = _geometry(**_QUARTER_TURN, scan_region_su=_REGION)
        centres = geometry["x_centers"]
        board_centre_x = 0.5 * (_BBOX_MIN[0] + _BBOX_MAX[0])
        self.assertAlmostEqual(
            0.5 * (centres[0] + centres[-1]),
            board_centre_x,
            places=9,
        )

    def test_a_non_positive_region_is_refused(self):
        with self.assertRaises(ValueError):
            _geometry(
                horizontal_aperture_su=50.0,
                vertical_aperture_su=60.03,
                x_num=1,
                y_num=1,
                scan_region_su=(0.0, 150.0),
            )


class SidecarFovTest(unittest.TestCase):
    """The sidecar's scanner-terms block, read off the affine.

    Deriving it from the world-axis footprints reports the right magnitudes on
    the wrong axes as soon as the camera is yawed a quarter turn: 750 x 480 dpi
    from a capture that is 600 dpi square.
    """

    def _layout(self, z_rotation):
        """A pixel layout for the reference capture at one camera yaw."""
        resolved = resolve_fov(
            stage_meters_per_unit=MM_STAGE,
            scan_area_mm=[120.0, 150.0],
            dpi=600,
            max_tile_px=[1600, 1600],
        )
        swap = round(z_rotation / 90.0) % 2 == 1
        tile_w, tile_h = resolved["tile_size_px"]
        world_x = resolved["scan_area_mm_rendered"][0] / resolved["x_num"]
        world_y = resolved["scan_area_mm_rendered"][1] / resolved["y_num"]
        geometry = _geometry(
            horizontal_aperture_su=world_y if swap else world_x,
            vertical_aperture_su=world_x if swap else world_y,
            x_num=resolved["x_num"],
            y_num=resolved["y_num"],
            camera_z_rotation_deg=z_rotation,
            scan_region_su=tuple(resolved["scan_area_mm_rendered"]),
        )
        layout = compute_pixel_layout(
            geometry,
            (tile_h, tile_w) if swap else (tile_w, tile_h),
            {"z_fixed": z_rotation},
            [
                (index, index % resolved["x_num"], index // resolved["x_num"])
                for index in range(resolved["x_num"] * resolved["y_num"])
            ],
        )
        return {**layout, "stage_meters_per_unit": MM_STAGE}

    def test_dpi_is_isotropic_at_every_cardinal_yaw(self):
        for z_rotation in (0.0, -90.0, 180.0, 90.0):
            with self.subTest(z_rotation=z_rotation):
                layout = self._layout(z_rotation)
                fov = _fov_from_layout(layout, layout["canvas_size_px"])
                self.assertIsNotNone(fov)
                self.assertTrue(fov["square_pixels"], fov["dpi"])
                for value in fov["dpi"]:
                    self.assertAlmostEqual(value, 600.0, delta=0.1)

    def test_scan_area_follows_the_canvas_orientation(self):
        """Stated along the image axes, so width first -- and it turns.

        Tolerance is one pixel per cell (0.0423 mm each at 600 dpi, over a
        2 x 3 grid): every cell is a whole number of pixels and they are equal,
        so a pixel count that does not divide by the cell count is covered by
        rounding each cell up rather than by making one of them different.
        """
        upright = self._layout(0.0)
        turned = self._layout(-90.0)
        tolerance = 25.4 / 600.0 * 3
        self.assertAlmostEqual(
            _fov_from_layout(upright, upright["canvas_size_px"])["scan_area_mm"][0],
            120.06,
            delta=tolerance,
        )
        self.assertAlmostEqual(
            _fov_from_layout(turned, turned["canvas_size_px"])["scan_area_mm"][0],
            150.0,
            delta=tolerance,
        )

    def test_no_stage_scale_means_no_block_rather_than_a_guess(self):
        layout = self._layout(0.0)
        layout.pop("stage_meters_per_unit")
        self.assertIsNone(_fov_from_layout(layout, layout["canvas_size_px"]))


class CoverageRoundingTest(unittest.TestCase):
    """Pixel counts follow the scanner rule, and exact requests stay exact."""

    def test_an_exact_request_is_not_grown_by_binary_arithmetic(self):
        """25.4 mm at 600 dpi is 600 pixels, not 601.

        ``area / pitch`` is 600.0000000000001 in binary floating point, so a
        plain ceil charges a pixel for every request that divides exactly --
        and the delivered area then disagrees with the requested one on the
        cases that should match perfectly.
        """
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[25.4, 25.4],
            dpi=600,
        )
        self.assertEqual(r["image_size_px"], [600, 600])
        for delivered, requested in zip(r["scan_area_mm_effective"], (25.4, 25.4)):
            self.assertAlmostEqual(delivered, requested, places=9)

    def test_an_inexact_request_rounds_by_the_scanner_rule(self):
        """2834.6 px is 2834: the size the scanner states for 120 mm at 600 dpi."""
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[120.0, 120.0],
            dpi=600,
        )
        self.assertEqual(r["image_size_px"], [2834, 2834])
        self.assertLessEqual(r["scan_area_mm_effective"][0], 120.0)


class ScannerImageSizeTest(unittest.TestCase):
    """A worked example of the scanner image-size rule.

    FoV corners (10, 10) and (20, 30) cm: a 100 x 200 mm rectangle. The scanner
    software states the image as width = mm / 25.4 * dpi, so at 600 dpi it is
    2362 x 4724, and at a 40 um x,y resolution 2500 x 5000.
    """

    def test_600_dpi_matches_the_note(self):
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE, scan_area_mm=[100.0, 200.0], dpi=600
        )
        self.assertEqual(r["image_size_px"], [2362, 4724])

    def test_tiling_does_not_change_the_size(self):
        """4724 rows split into two cells used to deliver 4726."""
        for max_tile_px in ([4096, 4096], [1000, 1000], [999, 777]):
            with self.subTest(max_tile_px=max_tile_px):
                r = resolve_fov(
                    stage_meters_per_unit=CM_STAGE,
                    scan_area_mm=[100.0, 200.0],
                    dpi=600,
                    max_tile_px=max_tile_px,
                )
                self.assertEqual(r["image_size_px"], [2362, 4724])

    def test_a_40_micrometre_resolution_matches_the_note(self):
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE, scan_area_mm=[100.0, 200.0], mm_per_px=0.04
        )
        self.assertEqual(r["image_size_px"], [2500, 5000])

    def test_1200_dpi_is_0_021167_mm_per_pixel(self):
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE, scan_area_mm=[100.0, 200.0], dpi=1200
        )
        for pitch in r["mm_per_px"]:
            self.assertAlmostEqual(pitch, 0.021167, places=6)
        self.assertEqual(r["image_size_px"], [4724, 9448])

    def test_a_subpixel_dimension_is_refused(self):
        """0.01 mm at 600 dpi is 0.24 px; one 0.042 mm pixel would exceed it."""
        for area in ([0.01, 10.0], [10.0, 0.01]):
            with (
                self.subTest(scan_area_mm=area),
                self.assertRaisesRegex(FovError, "less than one"),
            ):
                resolve_fov(stage_meters_per_unit=CM_STAGE, scan_area_mm=area, dpi=600)

    def test_exactly_one_pixel_is_one_pixel(self):
        one_pixel_mm = 25.4 / 600
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[one_pixel_mm, one_pixel_mm],
            dpi=600,
        )
        self.assertEqual(r["image_size_px"], [1, 1])


class PinnedCanvasTest(unittest.TestCase):
    """A pinned pixel count that does not divide into whole equal cells."""

    def test_a_pinned_size_is_delivered_even_when_cells_do_not_divide_it(self):
        """They pinned the number on purpose: the mosaic is cropped back to it."""
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[300.0, 200.0],
            image_size_px=[4097, 1000],
            max_tile_px=[2048, 2048],
        )
        self.assertEqual(r["image_size_px_requested"], [4097, 1000])
        self.assertEqual(r["image_size_px"], [4097, 1000])
        self.assertEqual(r["mosaic_size_px"], [4098, 1000])

    def test_a_divisible_pinned_size_is_delivered_exactly(self):
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[300.0, 200.0],
            image_size_px=[4096, 1000],
            max_tile_px=[2048, 2048],
        )
        self.assertEqual(r["image_size_px"], [4096, 1000])
        self.assertFalse(
            [note for note in r["assumptions"] if "not divisible" in note],
        )


class SquarePixelTest(unittest.TestCase):
    """An image_size_px whose aspect is not the area's is delivered square.

    Example: 280.398 x 348.54 mm pinned at 6750 x 8436 px asks for
    0.041540 mm/px across and 0.041316 down. A renderer that uses 0.041316 on
    both image axes leaves a 6 px step at every row seam and a 3 px sawtooth
    inside each tile, while calibration.json claims fx != fy.
    """

    _REGRESSION_CASE: ClassVar[dict[str, list[float]]] = {
        "scan_area_mm": [280.398, 348.54],
        "image_size_px": [6750, 8436],
        "max_tile_px": [1125, 1406],
    }

    def test_regression_request_gets_one_pitch_and_keeps_its_pixel_count(self):
        r = resolve_fov(stage_meters_per_unit=MM_STAGE, **self._REGRESSION_CASE)
        pitch = 280.398 / 6750
        self.assertAlmostEqual(pitch, 0.041540, places=6)
        self.assertEqual(r["mm_per_px"], [pitch, pitch])
        self.assertEqual(r["dpi_effective"][0], r["dpi_effective"][1])
        self.assertEqual(r["image_size_px"], [6750, 8436])
        self.assertEqual(r["image_size_px_requested"], [6750, 8436])
        self.assertEqual((r["x_num"], r["y_num"]), (6, 6))
        self.assertEqual(r["tile_size_px"], [1125, 1406])

    def test_the_requested_rectangle_stays_covered_and_the_growth_is_recorded(self):
        r = resolve_fov(stage_meters_per_unit=MM_STAGE, **self._REGRESSION_CASE)
        self.assertEqual(r["scan_area_mm"], [280.398, 348.54])
        eff_x, eff_y = r["scan_area_mm_effective"]
        self.assertAlmostEqual(eff_x, 280.398, places=9)
        self.assertAlmostEqual(eff_y, 8436 * 280.398 / 6750, places=9)
        self.assertAlmostEqual(eff_y, 350.435, places=3)
        self.assertEqual(r["scan_area_mm_rendered"], r["scan_area_mm_effective"])
        note = " ".join(r["assumptions"])
        self.assertIn("world y covers 350.435 mm", note)
        self.assertIn("requested 348.540", note)

    def test_the_apertures_match_the_frame_aspect(self):
        """What the render-time guard checks, met by construction."""
        r = resolve_fov(stage_meters_per_unit=MM_STAGE, **self._REGRESSION_CASE)
        check_square_pixels(
            r["horizontal_aperture_su"], r["vertical_aperture_su"], r["tile_size_px"]
        )
        self.assertAlmostEqual(
            r["horizontal_aperture_su"] / r["tile_size_px"][0],
            r["vertical_aperture_su"] / r["tile_size_px"][1],
            places=12,
        )

    def test_the_grown_axis_stays_centred_on_the_request(self):
        r = resolve_fov(
            stage_meters_per_unit=MM_STAGE,
            centre_xy_su=[10.0, -20.0],
            centered_on="panel",
            **self._REGRESSION_CASE,
        )
        low, high = r["bounds_su"]["min_xy"], r["bounds_su"]["max_xy"]
        self.assertAlmostEqual((low[1] + high[1]) / 2.0, -20.0, places=9)
        self.assertLess(low[1], -20.0 - 348.54 / 2.0)
        self.assertGreater(high[1], -20.0 + 348.54 / 2.0)

    def test_corners_with_a_pinned_canvas_grow_about_the_corners(self):
        r = resolve_fov(
            stage_meters_per_unit=MM_STAGE,
            corners_mm=[(100.0, 100.0), (200.0, 300.0)],
            image_size_px=[1000, 1000],
            max_tile_px=(100000, 100000),
        )
        self.assertEqual(r["image_size_px"], [1000, 1000])
        self.assertEqual(r["mm_per_px"], [0.2, 0.2])
        self.assertEqual(r["scan_area_mm_effective"], [200.0, 200.0])
        self.assertEqual(r["center_mm"], [150.0, 200.0])
        low, high = r["bounds_su"]["min_xy"], r["bounds_su"]["max_xy"]
        self.assertEqual((low, high), ([50.0, 100.0], [250.0, 300.0]))
        self.assertIn("world x covers 200.000", " ".join(r["assumptions"]))

    def test_a_canvas_with_the_areas_aspect_grows_nothing(self):
        r = resolve_fov(
            stage_meters_per_unit=MM_STAGE,
            scan_area_mm=[127.0, 254.0],
            image_size_px=[3000, 6000],
        )
        self.assertEqual(r["scan_area_mm_effective"], [127.0, 254.0])
        self.assertNotIn("instead of the requested", " ".join(r["assumptions"]))


class SquarePixelGuardTest(unittest.TestCase):
    """The render-time check where apertures meet a resolution."""

    def test_square_frames_pass(self):
        for hap, vap, res in (
            (280.398 / 6, 280.398 / 6750 * 1406, (1125, 1406)),
            (10.0, 10.0, (100, 100)),
            (10.0, 10.0 * (1 + 5e-5), (100, 100)),
        ):
            with self.subTest(hap=hap, vap=vap, res=res):
                check_square_pixels(hap, vap, res)

    def test_regression_authored_frame_is_refused_naming_both_pitches(self):
        """The apertures the pre-fix runtime set for the regression capture."""
        hap = 280.398 / 6
        vap = 348.54 / 6
        with self.assertRaisesRegex(
            FovError, r"0\.041540 mm/px across 1125 px.*0\.041316 mm/px down 1406"
        ):
            check_square_pixels(hap, vap, (1125, 1406), stage_meters_per_unit=MM_STAGE)


class ScanRegionRuleTest(unittest.TestCase):
    """The stated rectangle reaches scene units, or the capture is refused."""

    def test_a_perspective_capture_is_refused_rather_than_ignored(self):
        """Ignoring it succeeds with different coverage than was asked for.

        By the time the stage is open, the resolution and the cell counts have
        already been derived from this rectangle. Dropping it there frames the
        whole board at the pixel count of a part of it -- and then reports the
        part.
        """
        with self.assertRaises(FovError):
            scan_region_su(
                [120.0, 150.0],
                projection="perspective",
                stage_meters_per_unit=MM_STAGE,
            )

    def test_the_rectangle_converts_against_the_stage_scale(self):
        self.assertEqual(
            scan_region_su(
                [120.0, 150.0],
                projection="orthographic",
                stage_meters_per_unit=MM_STAGE,
            ),
            (120.0, 150.0),
        )

    def test_a_centimetre_stage_is_ten_times_smaller_in_units(self):
        region = scan_region_su(
            [120.0, 150.0],
            projection="orthographic",
            stage_meters_per_unit=CM_STAGE,
        )
        self.assertAlmostEqual(region[0], 12.0, places=9)
        self.assertAlmostEqual(region[1], 15.0, places=9)


class GridCountFromRegionTest(unittest.TestCase):
    """An auto-filled count is matched to the rectangle that is tiled."""

    def test_the_missing_count_follows_the_scan_region_not_the_board(self):
        """Aspect-matching to the board while tiling a smaller rectangle sizes
        the count for a shape the capture does not have."""
        # A long board, 40 x 200, tiled two cells across by square cells: ten
        # rows to cover it. A field of view over its middle 40 x 40 needs two.
        board = compute_scan_grid_geometry(
            (-20.0, -100.0, 0.0),
            (20.0, 100.0, 1.0),
            projection="orthographic",
            focal_length_su=None,
            horizontal_aperture_su=20.0,
            vertical_aperture_su=20.0,
            x_num=2,
            y_num=None,
        )
        region = compute_scan_grid_geometry(
            (-20.0, -100.0, 0.0),
            (20.0, 100.0, 1.0),
            projection="orthographic",
            focal_length_su=None,
            horizontal_aperture_su=20.0,
            vertical_aperture_su=20.0,
            x_num=2,
            y_num=None,
            scan_region_su=(40.0, 40.0),
        )
        self.assertEqual(board["y_num"], 10)
        self.assertEqual(region["y_num"], 2)


class ScannedBoundsTest(unittest.TestCase):
    """Which board placements a capture covers is a question about its cells."""

    def _grid(self, **overrides):
        grid = {
            "x_centers": [10.0, -10.0],
            "y_centers": [0.0],
            "footprint_x": 20.0,
            "footprint_y": 30.0,
        }
        grid.update(overrides)
        return grid

    def test_the_union_of_the_cells_is_the_scanned_rectangle(self):
        self.assertEqual(
            _scanned_bounds_xy(self._grid()),
            (-20.0, -15.0, 20.0, 15.0),
        )

    def test_a_grid_without_a_footprint_answers_nothing(self):
        """The legacy explicit-positions form carries nothing to intersect, and
        guessing a footprint would drop placements that are in fact imaged."""
        self.assertIsNone(_scanned_bounds_xy({"x_start": 1.0, "x_end": -1.0}))
        self.assertIsNone(_scanned_bounds_xy(self._grid(footprint_x=0.0)))


class PixelPitchInputTest(unittest.TestCase):
    """The resolution may be stated as the sensor's pitch instead of a dpi.

    Scanner specifications quote both -- 600 dpi and 0.04233 mm/px -- and a
    payload that carries the pitch must land on exactly the frame the dpi
    would, or the two spellings of one machine disagree about the image size.
    """

    def test_the_vendor_pitch_is_the_vendor_dpi(self):
        for dpi, pitch in ((600.0, 25.4 / 600.0), (1200.0, 25.4 / 1200.0)):
            with self.subTest(dpi=dpi):
                by_dpi = resolve_fov(
                    stage_meters_per_unit=CM_STAGE,
                    scan_area_mm=[120.0, 150.0],
                    dpi=dpi,
                )
                by_pitch = resolve_fov(
                    stage_meters_per_unit=CM_STAGE,
                    scan_area_mm=[120.0, 150.0],
                    mm_per_px=pitch,
                )
                self.assertEqual(by_pitch["image_size_px"], by_dpi["image_size_px"])
                self.assertEqual(by_pitch["tile_size_px"], by_dpi["tile_size_px"])
                for got, want in zip(
                    by_pitch["dpi_effective"], by_dpi["dpi_effective"], strict=True
                ):
                    self.assertAlmostEqual(got, want, places=9)
                self.assertEqual(by_pitch["mm_per_px_requested"], [pitch, pitch])
                self.assertIsNone(by_dpi["mm_per_px_requested"])

    def test_the_vendors_rounded_pitch_still_reads_as_600_dpi(self):
        """0.04233 mm/px as quoted in scanner specifications is 600 dpi, not 600.05."""
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[120.0, 150.0],
            mm_per_px=0.04233,
        )
        self.assertTrue(r["machine"]["supported"], r["dpi_effective"])

    def test_a_mixed_axis_pitch_is_refused(self):
        """600 dpi across and 1200 down: no scanner setting, and no render."""
        for pitch in ([25.4 / 600.0, 25.4 / 1200.0], [0.05, 0.1]):
            with (
                self.subTest(pitch=pitch),
                self.assertRaisesRegex(FovError, "mm_per_px .* non-square pixels"),
            ):
                resolve_fov(
                    stage_meters_per_unit=CM_STAGE,
                    scan_area_mm=[120.0, 150.0],
                    mm_per_px=pitch,
                )

    def test_an_equal_pitch_pair_is_one_pitch(self):
        r = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[120.0, 150.0],
            mm_per_px=[25.4 / 600.0, 25.4 / 600.0],
        )
        self.assertTrue(r["machine"]["supported"])
        self.assertEqual(r["mm_per_px_requested"], [25.4 / 600.0, 25.4 / 600.0])

    def test_a_pitch_beside_a_dpi_or_a_canvas_is_refused(self):
        for extra in ({"dpi": DEFAULT_DPI}, {"image_size_px": [100, 100]}):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(FovError, "exactly one"):
                    resolve_fov(
                        stage_meters_per_unit=CM_STAGE,
                        scan_area_mm=[10.0, 10.0],
                        mm_per_px=0.05,
                        **extra,
                    )

    def test_a_non_positive_pitch_is_refused(self):
        for bad in (0.0, -0.05):
            with self.subTest(bad=bad), self.assertRaisesRegex(FovError, "mm_per_px"):
                resolve_fov(
                    stage_meters_per_unit=CM_STAGE,
                    scan_area_mm=[10.0, 10.0],
                    mm_per_px=bad,
                )


class MachineComparisonTest(unittest.TestCase):
    """How the delivered resolution relates to the real scanner: said, not enforced."""

    def _resolve(self, **resolution):
        return resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[120.0, 150.0],
            **resolution,
        )

    def test_the_two_machine_settings_are_reported_as_supported(self):
        for dpi in SUPPORTED_DPI:
            with self.subTest(dpi=dpi):
                machine = self._resolve(dpi=dpi)["machine"]
                self.assertTrue(machine["supported"])
                self.assertTrue(machine["within_sensor"])
                self.assertEqual(machine["sensor_dpi"], SENSOR_DPI)
                self.assertEqual(machine["supported_dpi"], list(SUPPORTED_DPI))

    def test_a_resolution_the_machine_does_not_offer_is_reported_not_refused(self):
        r = self._resolve(dpi=300.0)
        self.assertFalse(r["machine"]["supported"])
        self.assertTrue(r["machine"]["within_sensor"])
        self.assertTrue(any("not a common scanner resolution" in a for a in r["assumptions"]))
        self.assertFalse(any("physical" in a for a in r["assumptions"]))

    def test_a_resolution_beyond_the_sensor_is_reported_not_refused(self):
        r = self._resolve(dpi=2400.0)
        self.assertFalse(r["machine"]["within_sensor"])
        self.assertFalse(r["machine"]["supported"])
        self.assertTrue(any("sensor's 1200 dpi" in a for a in r["assumptions"]))

    def test_a_pinned_canvas_is_compared_too(self):
        """The dpi a canvas implies is what the machine would have to deliver."""
        exact = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            scan_area_mm=[127.0, 254.0],
            image_size_px=[3000, 6000],
        )
        self.assertTrue(exact["machine"]["supported"], exact["dpi_effective"])
        odd = self._resolve(image_size_px=[1000, 1000])
        self.assertFalse(odd["machine"]["supported"])


class SidecarCentredOnTest(unittest.TestCase):
    """The label is carried, because it cannot be derived."""

    def _layout(self):
        return {
            "stage_meters_per_unit": MM_STAGE,
            "x_num": 1,
            "y_num": 1,
            "canvas_size_px": [100, 200],
            "pixel_to_world_xy": {
                "u_per_px_xy": [0.05, 0.0],
                "v_per_px_xy": [0.0, -0.05],
            },
        }

    def test_the_capture_says_what_the_rectangle_was_centred_on(self):
        layout = {**self._layout(), "scan_area_centered_on": "instance"}
        fov = _fov_from_layout(layout, layout["canvas_size_px"])
        self.assertEqual(fov["centered_on"], "instance")

    def test_a_capture_that_did_not_say_reports_nothing_rather_than_a_guess(self):
        layout = self._layout()
        fov = _fov_from_layout(layout, layout["canvas_size_px"])
        self.assertIsNone(fov["centered_on"])


class FovCornersTest(unittest.TestCase):
    """A field of view stated as two corners, the way a test program exports it."""

    def test_the_scanner_example_sizes_the_image(self):
        """(10, 10) to (20, 30) cm at 600 dpi, and at a 40 micrometre pitch.

        Rounded down, once: 100 mm at 600 dpi is 2362.2 px -> 2362, the size
        a line scanner reports for the same rectangle.
        """
        corners = [(100.0, 100.0, 0.0), (200.0, 300.0, 0.0)]
        by_dpi = resolve_fov(
            stage_meters_per_unit=MM_STAGE,
            corners_mm=corners,
            dpi=600,
            max_tile_px=(100000, 100000),
        )
        self.assertEqual(by_dpi["image_size_px"], [2362, 4724])
        self.assertEqual(by_dpi["scan_area_mm"], [100.0, 200.0])
        by_pitch = resolve_fov(
            stage_meters_per_unit=MM_STAGE,
            corners_mm=corners,
            mm_per_px=0.04,
            max_tile_px=(100000, 100000),
        )
        self.assertEqual(by_pitch["image_size_px"], [2500, 5000])

    def test_corners_place_the_rectangle(self):
        """The center is the corners' midpoint, not the board's."""
        resolved = resolve_fov(
            stage_meters_per_unit=CM_STAGE,
            corners_mm=[(100.0, 100.0), (200.0, 300.0)],
            dpi=600,
        )
        self.assertEqual(resolved["center_mm"], [150.0, 200.0])
        self.assertEqual(resolved["corners_mm"], [[100.0, 100.0], [200.0, 300.0]])
        self.assertEqual(resolved["centre_xy_su"], [15.0, 20.0])
        low = resolved["bounds_su"]["min_xy"]
        high = resolved["bounds_su"]["max_xy"]
        # The image is a whole number of pixels rounded down, centred on the
        # corners' midpoint, so each edge sits within half a pixel of its corner.
        half_pixel_su = 25.4 / 600 / 2 / 10.0
        self.assertAlmostEqual(low[0], 10.0, delta=half_pixel_su)
        self.assertAlmostEqual(high[1], 30.0, delta=half_pixel_su)
        self.assertAlmostEqual((low[0] + high[0]) / 2.0, 15.0)

    def test_corner_order_and_z_do_not_matter(self):
        forward = resolve_fov(
            stage_meters_per_unit=MM_STAGE,
            corners_mm=[(100.0, 100.0, 0.0), (200.0, 300.0, 5.0)],
            dpi=600,
        )
        backward = resolve_fov(
            stage_meters_per_unit=MM_STAGE,
            corners_mm=[(200.0, 300.0), (100.0, 100.0)],
            dpi=600,
        )
        for key in ("corners_mm", "center_mm", "image_size_px", "scan_area_mm"):
            self.assertEqual(forward[key], backward[key], key)

    def test_malformed_corners_are_refused(self):
        square = [(0.0, 0.0), (1.0, 1.0)]
        for kwargs in (
            {"corners_mm": [(0.0, 0.0), (0.0, 5.0)], "dpi": 600},
            {"corners_mm": [(0.0, 0.0)], "dpi": 600},
            {"corners_mm": [(0.0,), (1.0, 1.0)], "dpi": 600},
            {"corners_mm": [("x", 0.0), (1.0, 1.0)], "dpi": 600},
            {"corners_mm": [(0.0, None), (1.0, 1.0)], "dpi": 600},
            {"corners_mm": square, "scan_area_mm": (1.0, 1.0), "dpi": 600},
            {"dpi": 600},
            {"corners_mm": square, "dpi": 600, "centered_on": "panel"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(FovError):
                resolve_fov(stage_meters_per_unit=MM_STAGE, **kwargs)

    def test_a_placed_region_is_tiled_about_its_own_centre(self):
        """The grid follows the stated center, not the board midpoint."""
        centered = _geometry(**_QUARTER_TURN, scan_region_su=_REGION)
        placed = _geometry(
            **_QUARTER_TURN,
            scan_region_su=_REGION,
            scan_region_center_su=(50.0, 70.0),
        )
        mean_x = sum(placed["x_centers"]) / len(placed["x_centers"])
        mean_y = sum(placed["y_centers"]) / len(placed["y_centers"])
        self.assertAlmostEqual(mean_x, 50.0)
        self.assertAlmostEqual(mean_y, 70.0)
        self.assertEqual(placed["x_step"], centered["x_step"])
        self.assertEqual(placed["footprint_x"], centered["footprint_x"])

    def test_a_centre_without_a_region_is_refused(self):
        with self.assertRaisesRegex(ValueError, "scan_region_center_su places"):
            _geometry(**_QUARTER_TURN, scan_region_center_su=(50.0, 70.0))


if __name__ == "__main__":
    unittest.main()
