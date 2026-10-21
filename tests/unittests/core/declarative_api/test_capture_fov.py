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

"""What a ``capture.fov`` section has to reach for the scan rectangle to hold.

Deriving the resolution and the cell counts is not enough. Two other things
have to land, and each was silently missing in an earlier run that produced
correctly-sized tiles at the wrong world scale:

* the millimetres have to reach the render, where the stage's own scale turns
  them into an aperture, and where the scan grid's orthographic auto-fit --
  which only ever grows an aperture, up to ``bbox / count`` -- has to be
  switched off;
* the frame has to turn with the camera's fixed yaw, since the rectangle is
  stated in world axes and the frame is in camera axes.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi domain
from cdf import JobConfig, build_job_from_config
from cdf.config import CDF_API_VERSION
from pcb_aoi.geometry.fov import resolve_fov

_SCAN_AREA_MM = [120.0, 150.0]
_DPI = 600
_MAX_TILE_PX = [1600, 1600]


def _payload(*, z_rotation: float) -> dict[str, object]:
    """A minimal job whose capture is stated as a scan rectangle."""
    return {
        "api_version": CDF_API_VERSION,
        "domain": "pcb-aoi",
        "random_seed": 7,
        "output_dir": "output",
        "scene": {
            "usd": "board.usdc",
            "root": "/Board",
            "camera": "/World/Camera",
        },
        "capture": {
            "renderer": {"total_spp": 2},
            "camera": {"side": "top", "rotation": {"z": z_rotation}},
            "fov": {
                "scan_area_mm": _SCAN_AREA_MM,
                "dpi": _DPI,
                "max_tile_px": _MAX_TILE_PX,
                "centered_on": "panel",
            },
            "limits": {"max_images": 1},
            "outputs": {"annotators": ["rgb"]},
        },
    }


def _orders(z_rotation: float) -> dict[str, dict]:
    job = build_job_from_config(
        JobConfig.model_validate(_payload(z_rotation=z_rotation)),
        base_dir=Path("fixture"),
    )
    return {
        order.order_id: dict(order.parameters)
        for order in job.compile_orders(output_dir="output").orders
    }


def _resolved() -> dict:
    return resolve_fov(
        stage_meters_per_unit=0.01,
        scan_area_mm=_SCAN_AREA_MM,
        dpi=_DPI,
        max_tile_px=_MAX_TILE_PX,
        centered_on="panel",
    )


class CaptureFovTest(unittest.TestCase):
    """The three things a fov has to reach beyond the pixel count."""

    def test_scan_grid_carries_the_millimetres(self) -> None:
        """The rectangle has to reach the render, in millimetres.

        Not as an aperture: converting needs the stage's metersPerUnit, and
        guessing it upstream put a 120 mm capture out by a factor of ten. Its
        presence is also what switches off the grid's auto-fit, which only ever
        grows an aperture and so refuses exactly the shrink a field of view is.
        """
        resolved = _resolved()
        scan_grid = next(
            params
            for order_id, params in _orders(z_rotation=0.0).items()
            if order_id.endswith("scan_grid")
        )
        self.assertEqual(
            list(scan_grid["scan_area_mm"]),
            list(resolved["scan_area_mm_rendered"]),
        )

    def test_scan_grid_carries_the_image_size_the_stitch_crops_to(self) -> None:
        """The cells render a mosaic; the stitched image is the stated size.

        Stated in image axes, so it turns with the camera like the frame does.
        """
        resolved = _resolved()
        width, height = resolved["image_size_px"]
        for z_rotation, expected in ((0.0, [width, height]), (-90.0, [height, width])):
            with self.subTest(z_rotation=z_rotation):
                scan_grid = next(
                    params
                    for order_id, params in _orders(z_rotation).items()
                    if order_id.endswith("scan_grid")
                )
                self.assertEqual(list(scan_grid["image_size_px"]), expected)

    def test_scan_grid_spans_the_board_without_a_fov(self) -> None:
        """A plain scan grid keeps the behaviour every existing job relies on."""
        payload = _payload(z_rotation=0.0)
        capture = payload["capture"]
        assert isinstance(capture, dict)
        capture.pop("fov")
        capture["scan_grid"] = [1, 1]
        capture["renderer"] = {"resolution": [320, 180], "total_spp": 2}
        job = build_job_from_config(
            JobConfig.model_validate(payload),
            base_dir=Path("fixture"),
        )
        scan_grid = next(
            dict(order.parameters)
            for order in job.compile_orders(output_dir="output").orders
            if order.order_id.endswith("scan_grid")
        )
        self.assertIsNone(scan_grid["scan_area_mm"])
        self.assertNotIn("image_size_px", scan_grid)

    def test_unrotated_camera_keeps_the_frame_as_derived(self) -> None:
        resolved = _resolved()
        orders = _orders(z_rotation=0.0)
        self.assertIsNone(orders["scene.load"]["horizontal_aperture"])
        renderer = next(
            params
            for order_id, params in orders.items()
            if order_id.endswith("renderer")
        )
        self.assertEqual(
            list(renderer["resolution"]),
            list(resolved["tile_size_px"]),
        )

    def test_quarter_turn_turns_the_frame(self) -> None:
        """At -90 deg the camera's width lies along world Y, so the frame turns.

        The aperture turns to match at render time, from the same millimetres;
        the grid counts stay in world axes, which the next test pins down.
        """
        resolved = _resolved()
        orders = _orders(z_rotation=-90.0)
        renderer = next(
            params
            for order_id, params in orders.items()
            if order_id.endswith("renderer")
        )
        tile_w, tile_h = resolved["tile_size_px"]
        self.assertEqual(list(renderer["resolution"]), [tile_h, tile_w])

    def test_the_grid_counts_stay_in_world_axes(self) -> None:
        """The grid is world-axis-aligned, so the yaw must not turn it."""
        resolved = _resolved()
        for z_rotation in (0.0, -90.0):
            with self.subTest(z_rotation=z_rotation):
                scan_grid = next(
                    params
                    for order_id, params in _orders(z_rotation).items()
                    if order_id.endswith("scan_grid")
                )
                self.assertEqual(scan_grid["x_count"], resolved["x_num"])
                self.assertEqual(scan_grid["y_count"], resolved["y_num"])


class _StubJob:
    """Enough of a job to record declarations from one detached section."""

    def __init__(self):
        self.declarations = []

    def _record_declaration(self, order_id, order_type, parameters, **kwargs):
        self.declarations.append((order_id, order_type, dict(parameters)))


def _pass_section():
    """One pcb-aoi render pass, detached from a real job."""
    from pcb_aoi.sections import PcbAoiCapturePassSection

    return PcbAoiCapturePassSection(_StubJob())


_RENDERER = {"mode": "pathtracing", "spp": 1, "total_spp": 2, "max_bounces": 1}


class CaptureFovOwnershipTest(unittest.TestCase):
    """Where a field of view lives, and what the core has to know about it."""

    def test_the_domain_contributes_the_capture_section(self) -> None:
        """A pcb-aoi job gets a capture section that can state a scan area."""
        from cdf import CDFJob
        from pcb_aoi.sections import PcbAoiCaptureSection

        job = CDFJob(domain="pcb-aoi")
        self.assertIsInstance(job.capture, PcbAoiCaptureSection)
        self.assertTrue(callable(job.capture.fov))

    def test_named_passes_speak_the_same_vocabulary(self) -> None:
        """``render_pass`` must not offer less than the default pass."""
        from cdf import CDFJob

        job = CDFJob(domain="pcb-aoi")
        self.assertTrue(callable(job.capture.render_pass("top").fov))

    def test_core_capture_does_not_know_what_a_fov_is(self) -> None:
        """The core section stays domain-neutral: no fov, and it says so.

        The explicit refusal matters because the alternative is an
        AttributeError from inside configuration interpretation, which reads as
        a defect in CDF rather than as a job asking a domain for something
        that domain does not offer.
        """
        from cdf.sections import CapturePassSection
        from pcb_aoi.config.capture import RenderPassConfig, _apply_render_pass

        self.assertFalse(hasattr(CapturePassSection, "fov"))
        config = RenderPassConfig.model_validate(
            {
                "renderer": {"total_spp": 2},
                "camera": {"side": "top", "rotation": {"z": 0.0}},
                "fov": {"scan_area_mm": _SCAN_AREA_MM, "dpi": _DPI},
            },
        )
        with self.assertRaisesRegex(ValueError, "not supported by this job"):
            _apply_render_pass(CapturePassSection(_StubJob()), config)


class CaptureFovCallOrderTest(unittest.TestCase):
    """The public calls commute, and refuse to be told the same thing twice."""

    def _stated(self, section):
        section.fov(scan_area_mm=_SCAN_AREA_MM, dpi=_DPI, max_tile_px=_MAX_TILE_PX)
        return section

    def test_a_yaw_declared_after_the_fov_still_turns_the_frame(self) -> None:
        """Order-independence is not a nicety here: the failure is invisible.

        Deriving the frame once, at ``fov()`` time, reads whatever yaw happened
        to be declared already -- zero, when the camera came second -- and a
        capture framed square to the wrong axis looks right in every number the
        job records.
        """
        camera_first = _pass_section()
        camera_first.camera(side="top", rotation_range={"z": -90})
        camera_first.renderer(**_RENDERER)
        self._stated(camera_first)

        fov_first = self._stated(_pass_section())
        fov_first.renderer(**_RENDERER)
        fov_first.camera(side="top", rotation_range={"z": -90})

        self.assertEqual(
            fov_first.renderer_spec["resolution"],
            camera_first.renderer_spec["resolution"],
        )
        self.assertEqual(fov_first.scan_grid_spec, camera_first.scan_grid_spec)

    def test_a_renderer_declared_after_the_fov_keeps_the_derived_size(self) -> None:
        section = self._stated(_pass_section())
        section.camera(side="top", rotation_range={"z": -90})
        derived = list(section.fov_spec["frame_size_px"])
        section.renderer(**_RENDERER)
        self.assertEqual(section.renderer_spec["resolution"], derived)

    def test_a_resolution_beside_a_fov_is_refused_either_way_round(self) -> None:
        """Both state the frame size; which one wins is only visible later."""
        from cdf.errors import ValidationError

        resolution_first = _pass_section()
        resolution_first.renderer(resolution=[320, 180], **_RENDERER)
        with self.assertRaisesRegex(ValidationError, "already states"):
            self._stated(resolution_first)

        fov_first = self._stated(_pass_section())
        with self.assertRaisesRegex(ValidationError, "already derives"):
            fov_first.renderer(resolution=[320, 180], **_RENDERER)

    def test_a_scan_grid_beside_a_fov_is_refused_either_way_round(self) -> None:
        """Both derive the cell counts; whichever call comes second wins silently."""
        from cdf.errors import ValidationError

        grid_first = _pass_section()
        grid_first.scan_grid(x_count=1, y_count=1)
        with self.assertRaisesRegex(ValidationError, "already states"):
            self._stated(grid_first)

        fov_first = self._stated(_pass_section())
        with self.assertRaisesRegex(ValidationError, "already derives"):
            fov_first.scan_grid(x_count=1, y_count=1)

    def test_a_pass_with_neither_is_refused_before_it_renders(self) -> None:
        from cdf.errors import ValidationError

        section = _pass_section()
        section.renderer(**_RENDERER)
        section.camera(side="top", rotation_range={"z": 0.0})
        section.scan_grid(x_count=1, y_count=1)
        section.outputs_spec = {}
        with self.assertRaisesRegex(ValidationError, "no resolution"):
            section._validate_required_configuration()


class CaptureFovDeclarativeContractTest(unittest.TestCase):
    """What the YAML layer refuses, and when it refuses it."""

    def test_a_fov_beside_a_scan_grid_is_refused(self) -> None:
        """Both derive the cell counts; the fov wins, invisibly but for a count."""
        payload = _payload(z_rotation=0.0)
        payload["capture"]["scan_grid"] = [2, 2]
        with self.assertRaisesRegex(Exception, "scan_grid or fov, not both"):
            JobConfig.model_validate(payload)

    def test_an_over_or_under_determined_fov_fails_at_parse_time(self) -> None:
        """Area, dpi and pixel count are three names for two quantities.

        Failing here rather than where the arithmetic runs keeps invalid
        declarative input inside the normal validation boundary, and makes the
        generated schema describe the contract that is actually enforced.
        """
        from pcb_aoi.config.capture import FovConfig

        for fov in (
            {"scan_area_mm": _SCAN_AREA_MM},
            {"scan_area_mm": _SCAN_AREA_MM, "dpi": _DPI, "image_size_px": [10, 10]},
        ):
            with self.subTest(fov=fov), self.assertRaises(PydanticValidationError):
                FovConfig.model_validate(fov)

    def test_a_dimension_that_cannot_be_scanned_is_refused(self) -> None:
        from pcb_aoi.config.capture import FovConfig

        for fov in (
            {"scan_area_mm": [0.0, 150.0], "dpi": _DPI},
            {"scan_area_mm": [-120.0, 150.0], "dpi": _DPI},
            {"scan_area_mm": _SCAN_AREA_MM, "dpi": 0},
            {"scan_area_mm": _SCAN_AREA_MM, "mm_per_px": 0},
            {"scan_area_mm": _SCAN_AREA_MM, "image_size_px": [0, 10]},
        ):
            with self.subTest(fov=fov), self.assertRaises(PydanticValidationError):
                FovConfig.model_validate(fov)

    def test_a_pitch_is_a_third_spelling_of_the_same_resolution(self) -> None:
        """A scanner spec quotes 600 dpi and 0.04233 mm/px for one machine setting.

        A payload carrying either must compile to the same orders, or the
        image size depends on which spelling the payload author picked.
        """
        by_dpi = _orders(z_rotation=-90.0)
        payload = _payload(z_rotation=-90.0)
        fov = payload["capture"]["fov"]
        del fov["dpi"]
        fov["mm_per_px"] = 25.4 / _DPI
        job = build_job_from_config(
            JobConfig.model_validate(payload), base_dir=Path("fixture")
        )
        by_pitch = {
            order.order_id: dict(order.parameters)
            for order in job.compile_orders(output_dir="output").orders
        }
        self.assertEqual(by_pitch, by_dpi)

    def test_a_pitch_beside_a_dpi_fails_at_parse_time(self) -> None:
        from pcb_aoi.config.capture import FovConfig

        with self.assertRaisesRegex(Exception, "exactly one"):
            FovConfig.model_validate(
                {"scan_area_mm": _SCAN_AREA_MM, "dpi": _DPI, "mm_per_px": 0.04233}
            )

    def test_what_the_rectangle_is_centred_on_reaches_the_job_order(self) -> None:
        """Recorded rather than inferred -- and worthless if it stops here."""
        scan_grid = next(
            params
            for order_id, params in _orders(z_rotation=0.0).items()
            if order_id.endswith("scan_grid")
        )
        self.assertEqual(scan_grid["centered_on"], "panel")

    def test_a_label_without_a_rectangle_is_refused(self) -> None:
        """A grid that spans the board is centred on the board by construction."""
        from cdf.errors import ValidationError

        section = _pass_section()
        with self.assertRaisesRegex(ValidationError, "centered_on describes"):
            section.scan_grid(x_count=1, y_count=1, centered_on="panel")


class CaptureFovFluentValidationTest(unittest.TestCase):
    """The Python API rejects the same malformed input the YAML layer does.

    ``fov()`` hands everything to :func:`resolve_fov`, which is the one place
    both layers converge -- so validating here closes the gap for a caller
    who never goes through :class:`FovConfig` at all.
    """

    def test_an_unrecognized_centered_on_is_refused(self) -> None:
        from cdf.errors import ValidationError

        section = _pass_section()
        with self.assertRaisesRegex(ValidationError, "centered_on"):
            section.fov(
                scan_area_mm=_SCAN_AREA_MM,
                dpi=_DPI,
                centered_on="banana",
            )

    def test_a_pitch_lands_on_the_same_frame_as_its_dpi(self) -> None:
        by_dpi = _pass_section()
        by_dpi.camera(side="top", rotation_range={"z": -90})
        by_dpi.fov(scan_area_mm=_SCAN_AREA_MM, dpi=_DPI, max_tile_px=_MAX_TILE_PX)
        by_pitch = _pass_section()
        by_pitch.camera(side="top", rotation_range={"z": -90})
        by_pitch.fov(
            scan_area_mm=_SCAN_AREA_MM,
            mm_per_px=25.4 / _DPI,
            max_tile_px=_MAX_TILE_PX,
        )
        self.assertEqual(
            by_pitch.fov_spec["frame_size_px"], by_dpi.fov_spec["frame_size_px"]
        )
        self.assertEqual(by_pitch.scan_grid_spec, by_dpi.scan_grid_spec)

    def test_a_pitch_beside_a_dpi_is_refused(self) -> None:
        from cdf.errors import ValidationError

        with self.assertRaisesRegex(ValidationError, "exactly one"):
            _pass_section().fov(scan_area_mm=_SCAN_AREA_MM, dpi=_DPI, mm_per_px=0.04233)

    def test_a_malformed_image_size_px_is_refused_not_ignored(self) -> None:
        from cdf.errors import ValidationError

        for image_size_px in ([100], [100, 200, 300]):
            with self.subTest(image_size_px=image_size_px):
                section = _pass_section()
                with self.assertRaisesRegex(ValidationError, "image_size_px"):
                    section.fov(
                        scan_area_mm=_SCAN_AREA_MM,
                        image_size_px=image_size_px,
                    )

    def test_a_malformed_max_tile_px_is_refused_not_ignored(self) -> None:
        from cdf.errors import ValidationError

        for max_tile_px in ([4096], [4096, 4096, 4096]):
            with self.subTest(max_tile_px=max_tile_px):
                section = _pass_section()
                with self.assertRaisesRegex(ValidationError, "max_tile_px"):
                    section.fov(
                        scan_area_mm=_SCAN_AREA_MM,
                        dpi=_DPI,
                        max_tile_px=max_tile_px,
                    )


_CORNERS_MM = [[100.0, 100.0, 0.0], [220.0, 250.0, 0.0]]


def _corner_orders() -> dict[str, dict]:
    payload = _payload(z_rotation=0.0)
    payload["capture"]["fov"] = {
        "corners_mm": _CORNERS_MM,
        "dpi": _DPI,
        "max_tile_px": _MAX_TILE_PX,
    }
    job = build_job_from_config(
        JobConfig.model_validate(payload), base_dir=Path("fixture")
    )
    return {
        order.order_id: dict(order.parameters)
        for order in job.compile_orders(output_dir="output").orders
    }


def _by_suffix(orders: dict[str, dict], suffix: str) -> dict:
    return next(
        params for order_id, params in orders.items() if order_id.endswith(suffix)
    )


class CaptureFovCornersTest(unittest.TestCase):
    """A field of view stated as two corners is sized AND placed by them."""

    def test_the_corner_midpoint_reaches_the_scan_grid(self) -> None:
        scan_grid = _by_suffix(_corner_orders(), "scan_grid")
        self.assertEqual(list(scan_grid["scan_center_mm"]), [160.0, 175.0])
        self.assertIsNone(scan_grid["centered_on"])

    def test_corners_size_the_capture_like_the_equivalent_area(self) -> None:
        """Same size, same pixels and cells: only the placement differs."""
        by_corners = _corner_orders()
        by_area = _orders(z_rotation=0.0)
        self.assertEqual(
            _by_suffix(by_corners, "renderer"), _by_suffix(by_area, "renderer")
        )
        grid_c = _by_suffix(by_corners, "scan_grid")
        grid_a = _by_suffix(by_area, "scan_grid")
        self.assertEqual(grid_c["scan_area_mm"], grid_a["scan_area_mm"])
        self.assertEqual(
            (grid_c["x_count"], grid_c["y_count"]),
            (grid_a["x_count"], grid_a["y_count"]),
        )

    def test_the_yaml_layer_refuses_what_resolve_fov_refuses(self) -> None:
        from pcb_aoi.config.capture import FovConfig

        for fov in (
            {"corners_mm": _CORNERS_MM, "scan_area_mm": _SCAN_AREA_MM, "dpi": _DPI},
            {"corners_mm": [[0.0, 0.0], [0.0, 5.0]], "dpi": _DPI},
            {"corners_mm": _CORNERS_MM, "dpi": _DPI, "centered_on": "panel"},
            {"corners_mm": [[0.0, 0.0]], "dpi": _DPI},
            {"dpi": _DPI},
        ):
            with self.subTest(fov=fov), self.assertRaises(PydanticValidationError):
                FovConfig.model_validate(fov)

    def test_a_centre_needs_a_rectangle_and_excludes_a_label(self) -> None:
        from cdf.errors import ValidationError

        section = _pass_section()
        with self.assertRaisesRegex(ValidationError, "scan_center_mm places"):
            section.scan_grid(x_count=1, y_count=1, scan_center_mm=[1.0, 2.0])
        with self.assertRaisesRegex(ValidationError, "give one"):
            section.scan_grid(
                x_count=1,
                y_count=1,
                scan_area_mm=[5.0, 5.0],
                scan_center_mm=[1.0, 2.0],
                centered_on="panel",
            )

    def test_a_malformed_centre_is_a_validation_error(self) -> None:
        from cdf.errors import ValidationError

        for center in ([1.0], [1.0, 2.0, 3.0], ["a", 2.0], [True, 2.0], 5.0):
            with (
                self.subTest(center=center),
                self.assertRaisesRegex(ValidationError, "scan_center_mm must be"),
            ):
                _pass_section().scan_grid(
                    x_count=1,
                    y_count=1,
                    scan_area_mm=[5.0, 5.0],
                    scan_center_mm=center,
                )

    def test_typed_scan_grid_refuses_a_centre_with_a_label(self) -> None:
        from cdf.execution.isaac.job_order_parameters import ScanGridParameters

        with self.assertRaises(PydanticValidationError):
            ScanGridParameters.model_validate(
                {
                    "x_count": 1,
                    "y_count": 1,
                    "scan_area_mm": [5.0, 5.0],
                    "scan_center_mm": [1.0, 2.0],
                    "centered_on": "panel",
                }
            )
        ScanGridParameters.model_validate(
            {
                "x_count": 1,
                "y_count": 1,
                "scan_area_mm": [5.0, 5.0],
                "scan_center_mm": [1.0, 2.0],
            }
        )


class ScanGridImageSizeValidationTest(unittest.TestCase):
    """A malformed image_size_px is a ValidationError, not an IndexError."""

    def test_malformed_image_size_is_refused(self) -> None:
        from cdf.errors import ValidationError

        for size in (
            [4096],
            [1, 2, 3],
            [0, 10],
            [-1, 10],
            [1.5, 10],
            [True, 10],
            "64x64",
        ):
            with (
                self.subTest(image_size_px=size),
                self.assertRaisesRegex(ValidationError, "image_size_px must be"),
            ):
                _pass_section().scan_grid(x_count=1, y_count=1, image_size_px=size)

    def test_a_positive_pair_is_accepted(self) -> None:
        _pass_section().scan_grid(x_count=1, y_count=1, image_size_px=(64, 48))


class CaptureFovSquarePixelTest(unittest.TestCase):
    """Non-square pixels are refused, or never asked for.

    An orthographic RTX camera takes its vertical extent from the horizontal
    aperture and the image aspect, so a capture that needs two pitches renders
    one and seams every tile row. The YAML layer and the Python API refuse a
    pitch pair the same way, and a pinned canvas gets one pitch.
    """

    _PAIRS = ({"dpi": [600, 300]}, {"mm_per_px": [25.4 / 600, 25.4 / 1200]})

    def test_a_resolution_pair_is_refused_by_the_python_api(self) -> None:
        from cdf.errors import ValidationError

        for resolution in self._PAIRS:
            with (
                self.subTest(resolution=resolution),
                self.assertRaisesRegex(ValidationError, "non-square pixels"),
            ):
                _pass_section().fov(scan_area_mm=_SCAN_AREA_MM, **resolution)

    def test_a_resolution_pair_is_refused_at_parse_time(self) -> None:
        from pcb_aoi.config.capture import FovConfig

        for resolution in self._PAIRS:
            with (
                self.subTest(resolution=resolution),
                self.assertRaisesRegex(PydanticValidationError, "non-square pixels"),
            ):
                FovConfig.model_validate({"scan_area_mm": _SCAN_AREA_MM, **resolution})
            payload = _payload(z_rotation=0.0)
            del payload["capture"]["fov"]["dpi"]
            payload["capture"]["fov"].update(resolution)
            with self.assertRaisesRegex(PydanticValidationError, "non-square pixels"):
                JobConfig.model_validate(payload)

    def test_an_equal_pair_compiles_like_the_scalar(self) -> None:
        payload = _payload(z_rotation=-90.0)
        payload["capture"]["fov"]["dpi"] = [_DPI, _DPI]
        job = build_job_from_config(
            JobConfig.model_validate(payload), base_dir=Path("fixture")
        )
        by_pair = {
            order.order_id: dict(order.parameters)
            for order in job.compile_orders(output_dir="output").orders
        }
        self.assertEqual(by_pair, _orders(z_rotation=-90.0))

    def test_regression_pinned_canvas_reaches_the_grid_at_one_pitch(self) -> None:
        """6750 x 8436 px kept; the grid spans the grown, centred rectangle."""
        payload = _payload(z_rotation=-90.0)
        payload["capture"]["camera"]["side"] = "bottom"
        payload["capture"]["fov"] = {
            "scan_area_mm": [280.398, 348.54],
            "image_size_px": [6750, 8436],
            "max_tile_px": [1125, 1406],
            # Reference 6 x 6 tiling: with the default guard band the guarded
            # 1125 x 1406 tile no longer fits max_tile_px and the grid grows to
            # 7 x 7. This test is about the pitch, so it pins the band off.
            "guard_px": 0,
        }
        job = build_job_from_config(
            JobConfig.model_validate(payload), base_dir=Path("fixture")
        )
        orders = {
            order.order_id: dict(order.parameters)
            for order in job.compile_orders(output_dir="output").orders
        }
        grid = _by_suffix(orders, "scan_grid")
        renderer = _by_suffix(orders, "renderer")
        pitch = 280.398 / 6750
        self.assertEqual((grid["x_count"], grid["y_count"]), (6, 6))
        self.assertAlmostEqual(grid["scan_area_mm"][0], 280.398, places=9)
        self.assertAlmostEqual(grid["scan_area_mm"][1], 8436 * pitch, places=9)
        # Image axes: the quarter turn puts the canvas width along world Y.
        self.assertEqual(list(grid["image_size_px"]), [8436, 6750])
        self.assertEqual(list(renderer["resolution"]), [1406, 1125])
        # One cell's footprint over its frame is the same pitch both ways.
        self.assertAlmostEqual(
            grid["scan_area_mm"][1] / 6 / renderer["resolution"][0],
            grid["scan_area_mm"][0] / 6 / renderer["resolution"][1],
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
