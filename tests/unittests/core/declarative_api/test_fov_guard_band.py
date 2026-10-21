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

"""The fov guard band: render wider, crop before writing, change no contract.

Every RTX scan tile has an unreliable outermost column and row, and fov cells
abut with no overlap, so each seam carried them. Each cell now renders
``guard_px`` extra pixels per edge at the same pitch and the writer crops them
before encoding. These tests pin the two halves of that promise: the delivered
tile is exactly the unguarded tile (geometry, pixels, boxes, calibration), and
nothing a downstream consumer reads changes shape.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

import numpy as np
from PIL import Image
from pydantic import ValidationError as PydanticValidationError

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi domain
from cdf import (
    IsaacExecutionContext,
    IsaacSdgRuntime,
    JobConfig,
    build_job_from_config,
)
from cdf.config import CDF_API_VERSION
from cdf.errors import ValidationError
from cdf.execution.isaac import guard_band
from cdf.execution.isaac.capture_execution import (
    build_capture_config,
    execute_capture,
)
from cdf.execution.isaac.job_order_parameters import (
    CaptureRenderParameters,
    ScanGridParameters,
)
from pcb_aoi.geometry.fov import DEFAULT_GUARD_PX, FovError, resolve_fov
from pcb_aoi.geometry.scan_grid import compute_pixel_layout, compute_scan_grid_geometry

#: A large golden-board scan: corners 130.0688,20.7406 .. 405.0662,364.445 mm at 600 dpi,
#: a 2 x 2 grid of tiles 3248 x 4060 px in world axes (its camera is yawed
#: -90 deg, which is why the files on disk are 4060 x 3248).
_DIG_CORNERS = [[130.0688, 20.7406], [405.0662, 364.445]]
_BOX_DTYPE = [
    ("semanticId", "<u4"),
    ("x_min", "<i4"),
    ("y_min", "<i4"),
    ("x_max", "<i4"),
    ("y_max", "<i4"),
    ("occlusionRatio", "<f4"),
]


def _dig(guard_px: int) -> dict:
    return resolve_fov(
        stage_meters_per_unit=0.001,
        corners_mm=_DIG_CORNERS,
        dpi=600,
        guard_px=guard_px,
    )


class ResolveFovGuardTest(unittest.TestCase):
    """The guard band is added to the render, never to what is delivered."""

    def test_delivered_quantities_do_not_change(self):
        plain, guarded = _dig(0), _dig(8)
        for key in (
            "image_size_px",
            "tile_size_px",
            "mosaic_size_px",
            "x_num",
            "y_num",
            "horizontal_aperture_su",
            "vertical_aperture_su",
            "scan_area_mm_rendered",
            "mm_per_px",
            "centre_xy_su",
            "bounds_su",
        ):
            with self.subTest(key=key):
                self.assertEqual(plain[key], guarded[key])
        self.assertEqual(guarded["tile_size_px"], [3248, 4060])
        self.assertEqual(guarded["x_num"], 2)
        self.assertEqual(guarded["y_num"], 2)

    def test_render_grows_by_two_guards_at_the_same_pitch(self):
        resolved = _dig(8)
        self.assertEqual(resolved["guard_px"], 8)
        self.assertEqual(resolved["render_tile_size_px"], [3264, 4076])
        pitch_x, pitch_y = resolved["mm_per_px"]
        # Millimetre stage: scene units are millimetres.
        self.assertAlmostEqual(
            resolved["render_horizontal_aperture_su"]
            - resolved["horizontal_aperture_su"],
            16 * pitch_x,
            places=9,
        )
        self.assertAlmostEqual(
            resolved["render_vertical_aperture_su"] - resolved["vertical_aperture_su"],
            16 * pitch_y,
            places=9,
        )
        self.assertAlmostEqual(
            resolved["render_horizontal_aperture_su"] / 3264,
            resolved["horizontal_aperture_su"] / 3248,
            places=12,
        )

    def test_zero_guard_renders_the_tile(self):
        resolved = _dig(0)
        self.assertEqual(resolved["render_tile_size_px"], resolved["tile_size_px"])
        self.assertEqual(
            resolved["render_horizontal_aperture_su"],
            resolved["horizontal_aperture_su"],
        )

    def test_max_tile_px_bounds_the_guarded_render(self):
        """A tile that fits only without its guard band takes one more cell."""
        common = {
            "stage_meters_per_unit": 0.001,
            "scan_area_mm": [25.4 * 2000 / 600, 25.4 * 1000 / 600],
            "dpi": 600,
            "max_tile_px": [1000, 1000],
        }
        self.assertEqual(resolve_fov(**common, guard_px=0)["x_num"], 2)
        guarded = resolve_fov(**common, guard_px=8)
        self.assertEqual(guarded["x_num"], 3)
        self.assertLessEqual(max(guarded["render_tile_size_px"]), 1000)

    def test_invalid_guard_is_refused(self):
        for guard in (-1, 1.5, True, 600):
            with self.subTest(guard=guard), self.assertRaises(FovError):
                resolve_fov(
                    stage_meters_per_unit=0.001,
                    scan_area_mm=[10.0, 10.0],
                    dpi=600,
                    max_tile_px=[1000, 1000],
                    guard_px=guard,
                )

    def test_guarded_cell_samples_the_unguarded_pixels(self):
        """Pixel centres of the cropped render are the unguarded centres, exactly.

        Same pitch, same cell centre: the crop is the tile, not an approximation.
        """
        footprint, pixels, guard, centre = 171.873, 4060, 8, 336.3
        wide = guard_band.render_aperture(footprint, pixels, guard)
        wide_pixels = pixels + 2 * guard
        guarded = [
            centre - wide / 2 + (index + 0.5) * wide / wide_pixels
            for index in range(guard, guard + pixels)
        ]
        plain = [
            centre - footprint / 2 + (index + 0.5) * footprint / pixels
            for index in range(pixels)
        ]
        self.assertTrue(np.allclose(guarded, plain, rtol=0, atol=1e-9))


class GridLayoutUnchangedTest(unittest.TestCase):
    """The grid is computed from the delivered footprint, so nothing downstream moves."""

    def test_centres_step_and_pixel_layout_do_not_depend_on_the_guard(self):
        resolved = _dig(8)
        tile = resolved["tile_size_px"]
        geometry = compute_scan_grid_geometry(
            (0.0, 0.0, 0.0),
            (500.0, 400.0, 1.0),
            projection="orthographic",
            horizontal_aperture_su=resolved["horizontal_aperture_su"],
            vertical_aperture_su=resolved["vertical_aperture_su"],
            focal_length_su=None,
            x_num=2,
            y_num=2,
            scan_region_su=resolved["scan_area_mm_rendered"],
            scan_region_center_su=resolved["center_mm"],
        )
        layout = compute_pixel_layout(
            geometry,
            tile,
            {"x_range": (0, 0), "y_range": (0, 0), "z_fixed": 0.0},
            [(0, 0, 0), (1, 1, 0), (2, 0, 1), (3, 1, 1)],
        )
        self.assertEqual(layout["canvas_size_px"], [6496, 8120])
        self.assertEqual(
            [tile_layout["dst_rect_xywh_px"] for tile_layout in layout["tiles"]],
            [
                [3248.0, 0.0, 3248.0, 4060.0],
                [0.0, 0.0, 3248.0, 4060.0],
                [3248.0, 4060.0, 3248.0, 4060.0],
                [0.0, 4060.0, 3248.0, 4060.0],
            ],
        )


class CropPayloadTest(unittest.TestCase):
    """Every annotator is cropped from the same payload, consistently."""

    render = (12, 10)

    def _frame(self, channels=None, dtype=np.uint8):
        height, width = self.render[1], self.render[0]
        shape = (height, width) if channels is None else (height, width, channels)
        return np.arange(np.prod(shape)).reshape(shape).astype(dtype)

    def test_pixel_annotators_lose_the_band_on_every_side(self):
        cases = {
            "rgb": self._frame(4),
            "semantic_segmentation": self._frame(None, np.uint32),
            "instance_id_segmentation": self._frame(4),
            "distance_to_image_plane": self._frame(None, np.float32),
            "normals": self._frame(4, np.float32),
        }
        for name, frame in cases.items():
            with self.subTest(annotator=name):
                cropped = guard_band.crop_pixels(frame, 2, self.render)
                self.assertEqual(cropped.shape[:2], (6, 8))
                self.assertEqual(cropped.dtype, frame.dtype)
                self.assertTrue(np.array_equal(cropped, frame[2:8, 2:10]))
                self.assertTrue(cropped.flags["C_CONTIGUOUS"])

    def test_a_frame_of_the_wrong_size_is_refused(self):
        with self.assertRaises(guard_band.GuardBandError):
            guard_band.crop_pixels(np.zeros((8, 8, 4), np.uint8), 2, self.render)
        with self.assertRaises(guard_band.GuardBandError):
            guard_band.crop_pixels([[0]], 2, self.render)

    def test_boxes_shift_clip_and_drop(self):
        boxes = np.array(
            [
                (1, 4, 4, 6, 6, -1.0),  # inside: shifted by -2
                (2, 0, 0, 1, 1, -1.0),  # wholly in the top-left band: dropped
                (3, 1, 3, 5, 5, -1.0),  # straddles the left band: clipped to 0
                (4, 9, 7, 11, 9, -1.0),  # straddles the far corner: clipped to 7, 5
                (5, 10, 2, 11, 3, -1.0),  # wholly in the right band: dropped
                (6, 2, 2, 9, 7, -1.0),  # exactly the delivered tile
            ],
            dtype=_BOX_DTYPE,
        )
        kept, keep = guard_band.crop_boxes(boxes, 2, self.render)
        self.assertEqual(keep.tolist(), [True, False, True, True, False, True])
        self.assertEqual(
            [
                tuple(int(box[field]) for field in ("x_min", "y_min", "x_max", "y_max"))
                for box in kept
            ],
            [(2, 2, 4, 4), (0, 1, 3, 3), (7, 5, 7, 5), (0, 0, 7, 5)],
        )
        self.assertEqual(kept["semanticId"].tolist(), [1, 3, 4, 6])
        self.assertEqual(kept.dtype, boxes.dtype)

    def test_box_payload_filters_its_parallel_prim_paths(self):
        boxes = np.array(
            [(1, 4, 4, 6, 6, -1.0), (2, 0, 0, 1, 1, -1.0)], dtype=_BOX_DTYPE
        )
        data = {
            "trigger_outputs": {},
            "annotators": {
                "bounding_box_2d_tight_fast": {
                    "rp": {
                        "data": boxes,
                        "primPaths": ["/a", "/b"],
                        "bboxIds": np.array([7, 8]),
                        "idToLabels": {"1": {"class": "pin"}},
                    }
                },
            },
        }
        cropped = guard_band.crop_writer_payload(data, 2, self.render)
        payload = cropped["annotators"]["bounding_box_2d_tight_fast"]["rp"]
        self.assertEqual(payload["primPaths"], ["/a"])
        self.assertEqual(payload["bboxIds"].tolist(), [7])
        self.assertEqual(payload["idToLabels"], {"1": {"class": "pin"}})
        # The original payload is Replicator's cache; it must be untouched.
        self.assertEqual(
            len(data["annotators"]["bounding_box_2d_tight_fast"]["rp"]["data"]), 2
        )

    def test_parallel_arrays_that_disagree_are_refused(self):
        boxes = np.array([(1, 4, 4, 6, 6, -1.0)], dtype=_BOX_DTYPE)
        data = {
            "annotators": {
                "bounding_box_2d_loose": {"rp": {"data": boxes, "primPaths": []}},
            },
        }
        with self.assertRaises(guard_band.GuardBandError):
            guard_band.crop_writer_payload(data, 2, self.render)

    def test_whole_payload_crops_images_and_keeps_labels(self):
        rgb = self._frame(4)
        data = {
            "trigger_outputs": {"t": 1},
            "annotators": {
                "rgb": {"rp": {"data": rgb}},
                "semantic_segmentation": {
                    "rp": {
                        "data": self._frame(None, np.uint32),
                        "idToLabels": {"0": "x"},
                    }
                },
                "camera": {"rp": {"data": "camera"}},
            },
        }
        cropped = guard_band.crop_writer_payload(data, 2, self.render)
        self.assertEqual(cropped["trigger_outputs"], {"t": 1})
        self.assertEqual(cropped["annotators"]["rgb"]["rp"]["data"].shape, (6, 8, 4))
        semantic = cropped["annotators"]["semantic_segmentation"]["rp"]
        self.assertEqual(semantic["data"].shape, (6, 8))
        self.assertEqual(semantic["idToLabels"], {"0": "x"})
        self.assertEqual(cropped["annotators"]["camera"], data["annotators"]["camera"])
        self.assertEqual(data["annotators"]["rgb"]["rp"]["data"].shape, (10, 12, 4))

    def test_outputs_without_a_faithful_crop_are_refused(self):
        for name in (
            "bounding_box_3d",
            "camera_params",
            "occlusion",
            "pointcloud",
            "mystery",
        ):
            with (
                self.subTest(annotator=name),
                self.assertRaises(guard_band.GuardBandError),
            ):
                guard_band.crop_writer_payload(
                    {"annotators": {name: {"rp": {"data": np.zeros(3)}}}},
                    2,
                    self.render,
                )
        self.assertEqual(
            guard_band.unsupported_outputs(
                {"rgb": True, "bounding_box_3d": True, "camera_params": False}
            ),
            ["bounding_box_3d"],
        )

    def test_zero_guard_is_identity(self):
        frame = self._frame(4)
        self.assertIs(guard_band.crop_pixels(frame, 0, self.render), frame)
        boxes = np.array([(1, 0, 0, 11, 9, -1.0)], dtype=_BOX_DTYPE)
        kept, keep = guard_band.crop_boxes(boxes, 0, self.render)
        self.assertTrue(np.array_equal(kept, boxes))
        self.assertEqual(keep.tolist(), [True])
        self.assertEqual(guard_band.render_resolution((12, 10), 0), (12, 10))
        self.assertEqual(guard_band.render_aperture(0.1234567, 7, 0), 0.1234567)


class WriterCropTest(unittest.TestCase):
    """The crop sits in front of the writer's own encoder."""

    def test_installed_crop_reaches_the_encoder_and_records_refusals(self):
        written = []

        class _Writer:
            def write(self, data):
                written.append(data["annotators"]["rgb"]["rp"]["data"].shape)

        writer = _Writer()
        guard_band.install_writer_crop(writer, 3, (20, 16))
        writer.write({"annotators": {"rgb": {"rp": {"data": np.zeros((16, 20, 4))}}}})
        self.assertEqual(written, [(10, 14, 4)])
        guard_band.raise_writer_crop_errors(writer, "pass")

        writer.write({"annotators": {"rgb": {"rp": {"data": np.zeros((9, 9, 4))}}}})
        self.assertEqual(len(written), 1)  # refused, not written mis-sized
        with self.assertRaises(guard_band.GuardBandError):
            guard_band.raise_writer_crop_errors(writer, "pass")

    def test_delivered_size_check_reads_the_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGBA", (8, 6)).save(root / "rgb_x0_y0.png")
            np.save(root / "distance_to_image_plane_x0_y0.npy", np.zeros((6, 8)))
            (root / "bounding_box_2d_tight_x0_y0.npy").write_bytes(b"")
            paths = sorted(root.iterdir())
            self.assertEqual(guard_band.require_delivered_size(paths, (8, 6)), 2)
            Image.new("RGBA", (12, 10)).save(root / "rgb_x1_y0.png")
            with self.assertRaises(guard_band.GuardBandError):
                guard_band.require_delivered_size(sorted(root.iterdir()), (8, 6))


class GuardDeclarationTest(unittest.TestCase):
    """fov turns the guard band on; a plain scan grid keeps today's order."""

    def _orders(self, fov_extra=None, scan_grid=None):
        capture = {
            "renderer": {"total_spp": 2},
            "camera": {"side": "top", "rotation": {"z": -90.0}},
            "limits": {"max_images": 1},
            "outputs": {"annotators": ["rgb"]},
        }
        if scan_grid is None:
            capture["fov"] = {
                "scan_area_mm": [120.0, 150.0],
                "dpi": 600,
                **(fov_extra or {}),
            }
        else:
            capture["renderer"]["resolution"] = [320, 180]
            capture["scan_grid"] = scan_grid
        payload = {
            "api_version": CDF_API_VERSION,
            "domain": "pcb-aoi",
            "random_seed": 7,
            "output_dir": "output",
            "scene": {"usd": "board.usdc", "root": "/Board", "camera": "/World/Camera"},
            "capture": capture,
        }
        job = build_job_from_config(
            JobConfig.model_validate(payload), base_dir=Path("x")
        )
        return {
            order.order_id.rsplit(".", 1)[-1]: dict(order.parameters)
            for order in job.compile_orders(output_dir="output").orders
        }

    def test_fov_records_the_default_guard_and_the_delivered_resolution(self):
        orders = self._orders()
        resolved = resolve_fov(
            stage_meters_per_unit=0.001,
            scan_area_mm=[120.0, 150.0],
            dpi=600,
            guard_px=DEFAULT_GUARD_PX,
        )
        self.assertEqual(orders["scan_grid"]["guard_px"], DEFAULT_GUARD_PX)
        tile_w, tile_h = resolved["tile_size_px"]
        # Delivered, and turned with the -90 deg camera; the guard is not in it.
        self.assertEqual(list(orders["renderer"]["resolution"]), [tile_h, tile_w])

    def test_zero_guard_records_the_order_it_always_did(self):
        orders = self._orders({"guard_px": 0})
        self.assertNotIn("guard_px", orders["scan_grid"])

    def test_a_hand_authored_grid_has_no_guard_unless_asked(self):
        self.assertNotIn("guard_px", self._orders(scan_grid=[1, 1])["scan_grid"])
        orders = self._orders(
            scan_grid={
                "x_count": 2,
                "y_count": 1,
                "scan_area_mm": [20.0, 10.0],
                "guard_px": 4,
            }
        )
        self.assertEqual(orders["scan_grid"]["guard_px"], 4)

    def test_a_guard_without_a_scan_area_is_refused(self):
        with self.assertRaises(PydanticValidationError):
            self._orders(scan_grid={"x_count": 1, "y_count": 1, "guard_px": 4})
        with self.assertRaises(ValueError):
            ScanGridParameters(x_count=1, y_count=1, guard_px=4)
        ScanGridParameters(x_count=1, y_count=1, scan_area_mm=(1.0, 1.0), guard_px=4)


class RuntimeApertureTest(unittest.TestCase):
    """The runtime widens the camera, and only the camera."""

    def _resolve(self, guard_px, **overrides):
        from pcb_aoi.geometry.fov import check_square_pixels, mm_to_su
        from pcb_aoi.geometry.fov import scan_region_su as fov_scan_region_su
        from pcb_aoi.geometry.scan_grid import auto_complete_grid_nums
        from tests.unittests.core.geometry.test_scan_center_runtime import (
            _fake_usd,
            _resolve_scan_grid_from_stage,
        )

        usd_geom, usd = _fake_usd(0.001)
        geometry = mock.Mock(side_effect=compute_scan_grid_geometry)
        resolve = _resolve_scan_grid_from_stage(
            {
                "Any": object,
                "UsdGeom": usd_geom,
                "Usd": usd,
                "mm_to_su": mm_to_su,
                "check_square_pixels": check_square_pixels,
                "fov_scan_region_su": fov_scan_region_su,
                "auto_complete_grid_nums": auto_complete_grid_nums,
                "compute_scan_grid_geometry": geometry,
                "render_aperture": guard_band.render_aperture,
            }
        )
        stage = mock.Mock()
        stage.GetPrimAtPath.return_value.IsValid.return_value = True
        cfg = {
            "camera_path": "/Camera",
            "pcba_root": "/Board",
            "scan_grid": {"x_num": 2, "y_num": 1},
            # Square pixels: 30 mm over 100 px and 15 mm over 50 px per cell.
            "scan_area_mm": [60.0, 15.0],
            "resolution": [100, 50],
            "camera_rotation": {"z_fixed": 0.0},
            "guard_px": guard_px,
            **overrides,
        }
        with mock.patch("builtins.print"):
            geom = resolve(stage, cfg)
        camera = usd_geom.Camera.return_value
        return (
            geom,
            geometry.call_args.kwargs,
            camera.GetHorizontalApertureAttr.return_value.Set.call_args.args[0],
            camera.GetVerticalApertureAttr.return_value.Set.call_args.args[0],
        )

    def test_camera_renders_the_band_and_the_grid_does_not_see_it(self):
        plain, plain_kwargs, plain_h, plain_v = self._resolve(0)
        guarded, kwargs, render_h, render_v = self._resolve(5)
        # 30 mm over 100 px and 15 mm over 50 px, plus 5 px per side.
        self.assertAlmostEqual(plain_h, 300.0)
        self.assertAlmostEqual(plain_v, 150.0)
        self.assertAlmostEqual(render_h, 330.0)
        self.assertAlmostEqual(render_v, 180.0)
        self.assertEqual(kwargs["horizontal_aperture_su"], 30.0)
        self.assertEqual(kwargs["vertical_aperture_su"], 15.0)
        for key in ("x_centers", "y_centers", "x_step", "footprint_x", "footprint_y"):
            with self.subTest(key=key):
                self.assertEqual(plain[key], guarded[key])
        self.assertEqual(guarded["guard_px"], 5)
        self.assertEqual(plain_kwargs, kwargs)

    def test_a_guard_without_a_scan_area_is_refused(self):
        with self.assertRaises(ValueError):
            self._resolve(5, scan_area_mm=None)


def _context(scan_grid: dict, outputs: dict | None = None) -> IsaacExecutionContext:
    context = IsaacExecutionContext(
        runtime=IsaacSdgRuntime(),
        stage=object(),
        scene={"root_path": "/Board"},
        component_types=("PASSIVE_SMD_0805",),
        plan_metadata={"random_seed": 7},
    )
    context.state.update(
        {
            "capture.renderer": {
                "mode": "pathtracing",
                "resolution": (16, 12),
                "spp": 1,
                "total_spp": 2,
                "max_bounces": 1,
            },
            "capture.camera": {"rotation_range": {"x": (0, 0), "y": (0, 0), "z": 0}},
            "capture.scan_grid": scan_grid,
            "capture.outputs": outputs
            or {"rgb": True, "image_format": "png", "frame_padding": 4},
            "capture.limits": {"triggers": 1, "max_images": 1},
            "capture.output": {"start": 0},
        },
    )
    return context


class CaptureGuardTest(unittest.TestCase):
    """The render is guarded; the files, calibration and layout are not."""

    _GRID: ClassVar[dict[str, object]] = {
        "x_count": 1,
        "y_count": 1,
        "scan_area_mm": [1.0, 1.0],
        "guard_px": 2,
    }

    def test_config_renders_wider_and_refuses_uncroppable_outputs(self):
        config = build_capture_config(_context(self._GRID), "output")
        self.assertEqual(config["resolution"], (16, 12))
        self.assertEqual(config["render_resolution"], (20, 16))
        self.assertEqual(config["guard_px"], 2)
        plain = build_capture_config(_context({**self._GRID, "guard_px": 0}), "output")
        self.assertEqual(plain["render_resolution"], (16, 12))
        with self.assertRaises(ValidationError):
            build_capture_config(
                _context(
                    self._GRID,
                    {
                        "rgb": True,
                        "bounding_box_3d": True,
                        "image_format": "png",
                        "frame_padding": 4,
                    },
                ),
                "output",
            )

    def test_capture_writes_delivered_tiles_and_calibration(self):
        class _Camera:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        state = {}

        class _Writer:
            num_written = 0

            def initialize(self, **kwargs):
                state["dir"] = Path(kwargs["output_dir"])

            def attach(self, *_args, **_kwargs):
                pass

            def detach(self):
                pass

            def schedule_write(self):
                state["scheduled"] = True

            def write(self, data):
                frame = data["annotators"]["rgb"]["rp"]["data"]
                Image.fromarray(frame).save(
                    state["dir"] / f"rgb_{state['frame']:04d}.png"
                )
                self.num_written += 1

        writer = _Writer()

        async def _step(**_kwargs):
            if state.pop("scheduled", False):
                # What the render product delivers: the guarded frame.
                writer.write(
                    {
                        "annotators": {
                            "rgb": {"rp": {"data": np.zeros((16, 20, 4), np.uint8)}}
                        }
                    }
                )

        replicator = types.ModuleType("omni.replicator.core")
        replicator.create = mock.Mock()
        replicator.create.render_product.return_value = mock.Mock(path="rp")
        replicator.WriterRegistry = mock.Mock()
        replicator.WriterRegistry.get.return_value = writer
        replicator.get = mock.Mock()
        replicator.get.prim_at_path.return_value = _Camera()
        replicator.modify = mock.Mock()
        replicator.distribution = mock.Mock()
        replicator.orchestrator = mock.Mock()
        replicator.orchestrator.step_async = mock.AsyncMock(side_effect=_step)
        replicator.orchestrator.wait_until_complete_async = mock.AsyncMock()
        pxr = types.ModuleType("pxr")
        pxr.Gf = mock.Mock()

        common = mock.Mock()
        common.configure_pathtracing.return_value = 2
        common.build_scan_positions.return_value = [(1.0, 2.0, 3.0)]
        common.randomize_lighting.return_value = {}
        common.setup_augmentation.return_value = {"applied": False}
        common.scan_positions_for_trigger.side_effect = lambda positions, *_: positions
        common.grid_index_for_position.return_value = (0, 0)
        common.set_replicator_seed_for_output_frame.side_effect = lambda frame: (
            state.__setitem__("frame", int(frame))
        )
        layout = {"supported": True, "canvas_size_px": [16, 12], "tiles": []}
        geometry = mock.Mock()
        geometry.compute_pixel_layout.return_value = layout
        calibration = mock.Mock(return_value={"resolution_px": [16, 12]})

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(
                sys.modules,
                {
                    "omni": types.ModuleType("omni"),
                    "omni.replicator": types.ModuleType("omni.replicator"),
                    "omni.replicator.core": replicator,
                    "pxr": pxr,
                },
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._script_module",
                side_effect=lambda _directory, name: {
                    "common": common,
                    "defect_ops": mock.Mock(),
                }[name],
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution.scan_grid_geometry", geometry
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution.orthographic_calibration",
                calibration,
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._capture_reference_plane_z",
                return_value=(None, None),
            ),
            mock.patch("cdf.execution.isaac.capture_execution._ensure_camera"),
            mock.patch("cdf.execution.isaac.capture_execution._setup_lighting"),
            mock.patch(
                "cdf.execution.isaac.capture_execution._setup_component_appearance",
                return_value={"applied": False},
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._warmup", new=mock.AsyncMock()
            ),
        ):
            output = asyncio.run(
                execute_capture(
                    _context(self._GRID), CaptureRenderParameters(output_dir=directory)
                )
            )
            with Image.open(Path(directory) / "trigger_0000" / "rgb_0000.png") as tile:
                tile_size = tile.size

        self.assertEqual(tile_size, (16, 12))
        replicator.create.render_product.assert_called_once_with(
            "/World/camera_light/Camera", (20, 16)
        )
        self.assertEqual(calibration.call_args.args[2], (20, 16))
        self.assertEqual(calibration.call_args.kwargs, {"crop_px": 2})
        # The pixel layout is computed for the delivered tile.
        self.assertEqual(geometry.compute_pixel_layout.call_args.args[1], (16, 12))
        self.assertEqual(output["trigger_metadata"][0]["guard_px"], 2)
        self.assertEqual(output["trigger_metadata"][0]["pixel_layout"], layout)


if __name__ == "__main__":
    unittest.main()
