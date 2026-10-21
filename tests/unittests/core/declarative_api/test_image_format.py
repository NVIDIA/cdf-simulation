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

"""Kit-free tests for the ``image_format`` allow-list and bmp conversion."""

import asyncio
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image
from pydantic import ValidationError as PydanticValidationError

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi CDF domain
from cdf import (
    IsaacExecutionContext,
    IsaacSdgRuntime,
    CDFJob,
    ValidationError,
)
from cdf.execution.isaac.capture_execution import (
    _discover_frame_artifacts,
    build_capture_config,
    convert_rgb_outputs,
    execute_capture,
)
from cdf.execution.isaac.job_order_parameters import (
    CaptureRenderParameters,
    CaptureWriterParameters,
)
from cdf.image_formats import IMAGE_FORMATS, WRITER_NATIVE_IMAGE_FORMATS
from pcb_aoi.config.capture import OutputsConfig

_RGBA_PIXEL = (200, 30, 90, 128)


def _write_rgba_png(path: Path, size=(4, 3), pixel=_RGBA_PIXEL) -> None:
    Image.new("RGBA", size, pixel).save(path)


def _configure_pass(section, image_format: str, **outputs) -> None:
    section.renderer(
        mode="pathtracing",
        resolution=(64, 64),
        spp=1,
        total_spp=1,
        max_bounces=1,
    )
    section.camera(side="bottom", rotation_range={"z": -90})
    section.scan_grid(x_count=1, y_count=1)
    section.outputs(**{"rgb": True, "image_format": image_format, **outputs})


def _capture_state(image_format: str, *, rgb: bool = True) -> dict[str, object]:
    return {
        "capture.renderer": {
            "mode": "pathtracing",
            "resolution": (64, 64),
            "spp": 1,
            "total_spp": 2,
            "max_bounces": 1,
        },
        "capture.camera": {
            "rotation_range": {"x": (0, 0), "y": (0, 0), "z": -90},
        },
        "capture.scan_grid": {"x_count": 1, "y_count": 1},
        "capture.outputs": {
            "rgb": rgb,
            "image_format": image_format,
            "frame_padding": 4,
            "rename_to_grid_index": True,
        },
        "capture.limits": {"triggers": 1, "max_images": 1},
        "capture.output": {"start": 0},
    }


class ImageFormatValidationTest(unittest.TestCase):
    """Every authoring surface accepts the same formats and rejects the rest."""

    def test_every_accepted_format_passes_all_three_surfaces(self) -> None:
        for image_format in IMAGE_FORMATS:
            with self.subTest(image_format=image_format):
                self.assertEqual(
                    OutputsConfig(image_format=image_format).image_format,
                    image_format,
                )
                self.assertEqual(
                    CaptureWriterParameters(image_format=image_format).image_format,
                    image_format,
                )
                job = CDFJob(domain="pcb-aoi")
                job.capture.outputs(image_format=image_format)
                self.assertEqual(
                    job.capture.outputs_spec["image_output_format"],
                    image_format,
                )
        self.assertIn("bmp", IMAGE_FORMATS)
        self.assertNotIn("bmp", WRITER_NATIVE_IMAGE_FORMATS)

    def test_unknown_format_is_rejected_at_declaration_time(self) -> None:
        with self.assertRaisesRegex(PydanticValidationError, "png"):
            OutputsConfig(image_format="tiff")
        with self.assertRaisesRegex(PydanticValidationError, "png"):
            CaptureWriterParameters(image_format="tiff")
        job = CDFJob(domain="pcb-aoi")
        with self.assertRaisesRegex(
            ValidationError,
            "outputs.image_format must be one of png, jpeg, jpg, bmp, got 'tiff'",
        ):
            job.capture.outputs(image_format="tiff")
        with self.assertRaisesRegex(ValidationError, "got None"):
            job.capture.outputs(image_format=None)

    @staticmethod
    def _crop_job(image_format: str, *, types: list[str], rgb: bool = True):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usdc", root_path="/Board")
        job.capture.output_start(0)
        job.capture.limits(triggers=1, max_images=1)
        _configure_pass(
            job.capture,
            image_format,
            rgb=rgb,
            bounding_box_2d_tight=True,
            semantic_segmentation=True,
        )
        job.postprocess.crop(
            method="bbox_tight",
            types=types,
            offset=0,
            min_coverage=0.1,
            xform_depth=1,
            output_subdir="closeups",
        )
        return job

    def test_crop_of_the_rgb_frames_of_a_converted_pass_is_rejected(self) -> None:
        job = self._crop_job("bmp", types=["rgb", "semantic_segmentation"])

        with self.assertRaisesRegex(
            ValidationError,
            r"postprocess\.crop opens the rgb frames of capture pass 'default' "
            r"as the writer's png/jpeg, but that pass declares image_format 'bmp'",
        ):
            job.compile_orders(output_dir="output")

    def test_crop_of_only_the_native_artifacts_of_a_converted_pass_compiles(
        self,
    ) -> None:
        # Conversion changes the rgb frames and nothing else, so a crop whose
        # ``types`` excludes rgb never meets a .bmp.
        job = self._crop_job("bmp", types=["semantic_segmentation"])

        plan = job.compile_orders(output_dir="output")

        self.assertIn("postprocess.crop", [order.order_type for order in plan.orders])

    def test_crop_of_a_converted_pass_without_rgb_compiles(self) -> None:
        # With rgb disabled there is no converted artifact at all; the crop of
        # ``rgb`` fails for the same reason it would on a png pass, not here.
        job = self._crop_job("bmp", types=["rgb", "semantic_segmentation"], rgb=False)

        plan = job.compile_orders(output_dir="output")

        outputs = [
            order.parameters
            for order in plan.orders
            if order.order_type == "capture.outputs.configure"
        ]
        self.assertEqual(len(outputs), 1)
        self.assertFalse(outputs[0]["rgb"])
        self.assertEqual(outputs[0]["image_format"], "bmp")

    @staticmethod
    def _panel_job(full_board_format: str, **full_board_outputs):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="panel.usda", root_path="/World/Panel")
        with job.capture.render_pass("full_board") as render_pass:
            _configure_pass(render_pass, full_board_format, **full_board_outputs)
        with job.capture.render_pass("component_rois") as render_pass:
            _configure_pass(render_pass, "png")
        return job

    def test_stitch_of_a_converted_pass_is_rejected_but_other_passes_compile(
        self,
    ) -> None:
        job = self._panel_job("bmp")
        job.postprocess.stitch(source_pass="component_rois")

        plan = job.compile_orders(output_dir="output")
        outputs = {
            order.order_id: order.parameters["image_format"]
            for order in plan.orders
            if order.order_type == "capture.outputs.configure"
        }
        self.assertEqual(
            outputs,
            {
                "capture.full_board.outputs": "bmp",
                "capture.component_rois.outputs": "png",
            },
        )

        job.postprocess.stitch(source_pass="full_board")
        with self.assertRaisesRegex(
            ValidationError,
            r"postprocess\.stitch opens the rgb frames of capture pass "
            r"'full_board' .* declares image_format 'bmp'",
        ):
            job.compile_orders(output_dir="output")

    def test_semantic_stitch_and_close_openings_of_a_converted_pass_compile(
        self,
    ) -> None:
        # Semantic masks stay PNG on a bmp pass, and close_openings reads the
        # stitched mask that stitch writes, so neither touches a .bmp.
        job = self._panel_job("bmp", semantic_segmentation=True)
        job.postprocess.stitch(
            source_pass="full_board",
            image_type="semantic_segmentation",
            output_filename="mask.png",
            blend="labels",
        )
        job.postprocess.close_openings(
            source_pass="full_board",
            class_name="hole",
            output_filename="mask.png",
        )

        plan = job.compile_orders(output_dir="output")

        self.assertEqual(
            [
                order.order_type
                for order in plan.orders
                if order.order_type.startswith("postprocess.")
            ],
            ["postprocess.stitch", "postprocess.close_openings"],
        )

    def test_rgb_stitch_of_a_converted_pass_without_rgb_compiles(self) -> None:
        job = self._panel_job("bmp", rgb=False)
        job.postprocess.stitch(source_pass="full_board")

        plan = job.compile_orders(output_dir="output")

        self.assertIn("postprocess.stitch", [order.order_type for order in plan.orders])

    def test_roi_formatting_rejects_only_a_converted_rgb_pass(self) -> None:
        # format_roi_output opens rgb tiles from rgb_pass only; the ROI pass
        # contributes masks and labels, and generate_joint_rois reads bounding
        # boxes, so a bmp ROI pass is fine for both.
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="panel.usda", root_path="/World/Panel")
        with job.capture.render_pass("full_board") as render_pass:
            _configure_pass(render_pass, "png", rename_to_grid_index=True)
        with job.capture.render_pass("full_board_roi") as render_pass:
            _configure_pass(
                render_pass,
                "bmp",
                rename_to_grid_index=True,
                semantic_segmentation=True,
                bounding_box_2d_tight=True,
            )
        job.postprocess.format_roi_output()
        job.postprocess.generate_joint_rois()

        plan = job.compile_orders(output_dir="output")
        self.assertEqual(
            sorted(
                order.order_type
                for order in plan.orders
                if order.order_type.startswith("postprocess.")
            ),
            ["postprocess.format_roi_output", "postprocess.generate_joint_rois"],
        )

        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="panel.usda", root_path="/World/Panel")
        with job.capture.render_pass("full_board") as render_pass:
            _configure_pass(render_pass, "bmp", rename_to_grid_index=True)
        with job.capture.render_pass("full_board_roi") as render_pass:
            _configure_pass(
                render_pass,
                "png",
                rename_to_grid_index=True,
                semantic_segmentation=True,
            )
        job.postprocess.format_roi_output()
        with self.assertRaisesRegex(
            ValidationError,
            r"postprocess\.format_roi_output opens the rgb frames of capture "
            r"pass 'full_board'",
        ):
            job.compile_orders(output_dir="output")


class ConvertRgbOutputsTest(unittest.TestCase):
    """The post-write conversion touches rgb frames and nothing else."""

    def test_capture_config_renders_a_converted_format_as_png(self) -> None:
        for image_format, writer_format, convert_to in (
            ("png", "png", None),
            ("jpg", "jpg", None),
            ("bmp", "png", "bmp"),
        ):
            with self.subTest(image_format=image_format):
                context = IsaacExecutionContext(
                    runtime=IsaacSdgRuntime(),
                    stage=object(),
                    scene={"root_path": "/Board"},
                    component_types=(),
                    plan_metadata={"random_seed": 1},
                )
                context.state.update(_capture_state(image_format))

                config = build_capture_config(context, "output")

                self.assertEqual(config["writer"]["image_output_format"], writer_format)
                self.assertEqual(config["convert_rgb_to"], convert_to)
                self.assertNotIn("image_format", config["writer"])

    def test_bmp_without_rgb_schedules_no_conversion(self) -> None:
        # Nothing to re-encode: conversion would only wait out its deadline for
        # rgb frames nobody asked for and then fail the pass over them.
        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            stage=object(),
            scene={"root_path": "/Board"},
            component_types=(),
            plan_metadata={"random_seed": 1},
        )
        context.state.update(_capture_state("bmp", rgb=False))

        config = build_capture_config(context, "output")

        self.assertIsNone(config["convert_rgb_to"])
        self.assertFalse(config["writer"]["rgb"])
        self.assertEqual(config["writer"]["image_output_format"], "png")

    def test_rgb_frames_are_reencoded_without_alpha_and_others_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pass_dir = Path(temporary_directory) / "trigger_0000"
            pass_dir.mkdir()
            _write_rgba_png(pass_dir / "rgb_0000.png")
            _write_rgba_png(pass_dir / "rgb_0001.png", pixel=(1, 2, 3, 255))
            _write_rgba_png(pass_dir / "semantic_segmentation_0000.png")
            (pass_dir / "bounding_box_2d_tight_0000.npy").write_bytes(b"npy")
            (pass_dir / "rgb_0000_labels.json").write_text("{}")

            converted = convert_rgb_outputs(
                pass_dir,
                "bmp",
                frame_padding=4,
                frames=[0, 1],
            )

            self.assertEqual(converted, 2)
            self.assertEqual(
                sorted(path.name for path in pass_dir.iterdir()),
                [
                    "bounding_box_2d_tight_0000.npy",
                    "rgb_0000.bmp",
                    "rgb_0000_labels.json",
                    "rgb_0001.bmp",
                    "semantic_segmentation_0000.png",
                ],
            )
            with Image.open(pass_dir / "rgb_0000.bmp") as image:
                self.assertEqual(image.format, "BMP")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (4, 3))
                self.assertEqual(image.getpixel((0, 0)), _RGBA_PIXEL[:3])
            with Image.open(pass_dir / "rgb_0001.bmp") as image:
                self.assertEqual(image.getpixel((3, 2)), (1, 2, 3))
            with Image.open(pass_dir / "semantic_segmentation_0000.png") as image:
                self.assertEqual(image.mode, "RGBA")

    def test_only_this_captures_frames_are_converted(self) -> None:
        # Reusing an output directory must not convert, and delete, an rgb PNG
        # an earlier run left there.
        with tempfile.TemporaryDirectory() as temporary_directory:
            pass_dir = Path(temporary_directory)
            _write_rgba_png(pass_dir / "rgb_0000.png")
            _write_rgba_png(pass_dir / "rgb_0007.png", pixel=(9, 9, 9, 255))
            stale = (pass_dir / "rgb_0007.png").read_bytes()

            converted = convert_rgb_outputs(
                pass_dir, "bmp", frame_padding=4, frames=[0]
            )

            self.assertEqual(converted, 1)
            self.assertEqual(
                sorted(path.name for path in pass_dir.iterdir()),
                ["rgb_0000.bmp", "rgb_0007.png"],
            )
            self.assertEqual((pass_dir / "rgb_0007.png").read_bytes(), stale)

    def test_frames_that_never_surface_or_stay_truncated_fail_the_capture(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pass_dir = Path(temporary_directory)
            _write_rgba_png(pass_dir / "rgb_0000.png")
            complete = (pass_dir / "rgb_0000.png").read_bytes()
            (pass_dir / "rgb_0002.png").write_bytes(complete[: len(complete) // 2])

            with self.assertRaises(RuntimeError) as raised:
                convert_rgb_outputs(
                    pass_dir,
                    "bmp",
                    frame_padding=4,
                    frames=[0, 1, 2],
                    flush_timeout_s=0.2,
                    poll_interval_s=0.01,
                )

            # The complete frame was converted; the truncated one was retried
            # up to the deadline and left as the writer left it.
            self.assertTrue((pass_dir / "rgb_0000.bmp").exists())
            self.assertTrue((pass_dir / "rgb_0002.png").exists())
            self.assertFalse((pass_dir / "rgb_0002.bmp").exists())
            message = str(raised.exception)
            self.assertIn("rgb_0001.png", message)
            self.assertIn("rgb_0002.png", message)
            self.assertNotIn("rgb_0000.png", message)

    def test_a_truncated_frame_that_completes_later_is_converted(self) -> None:
        # The writer flushes asynchronously: a PNG can be half on disk when the
        # pass ends. Retrying the decode, not failing on it, is the point.
        with tempfile.TemporaryDirectory() as temporary_directory:
            pass_dir = Path(temporary_directory)
            _write_rgba_png(pass_dir / "complete.png", pixel=(5, 6, 7, 255))
            complete = (pass_dir / "complete.png").read_bytes()
            (pass_dir / "complete.png").unlink()
            source = pass_dir / "rgb_0000.png"
            source.write_bytes(complete[: len(complete) // 2])
            finish = threading.Timer(0.1, source.write_bytes, args=(complete,))
            finish.start()
            try:
                converted = convert_rgb_outputs(
                    pass_dir,
                    "bmp",
                    frame_padding=4,
                    frames=[0],
                    flush_timeout_s=5.0,
                    poll_interval_s=0.01,
                )
            finally:
                finish.join()

            self.assertEqual(converted, 1)
            self.assertFalse(source.exists())
            with Image.open(pass_dir / "rgb_0000.bmp") as image:
                self.assertEqual(image.getpixel((0, 0)), (5, 6, 7))

    def test_destination_write_failures_propagate_and_keep_the_source(self) -> None:
        # A permission error or a full disk is not a frame still flushing; it
        # must surface as the failure it is, not be retried into a warning.
        with tempfile.TemporaryDirectory() as temporary_directory:
            pass_dir = Path(temporary_directory)
            _write_rgba_png(pass_dir / "rgb_0000.png")

            with (
                mock.patch.object(
                    Image.Image,
                    "save",
                    side_effect=PermissionError(13, "Permission denied"),
                ),
                self.assertRaises(PermissionError),
            ):
                convert_rgb_outputs(
                    pass_dir,
                    "bmp",
                    frame_padding=4,
                    frames=[0],
                    flush_timeout_s=5.0,
                    poll_interval_s=0.01,
                )

            self.assertEqual(
                sorted(path.name for path in pass_dir.iterdir()),
                ["rgb_0000.png"],
            )

    def test_discovered_artifacts_record_the_bmp_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            pass_dir = output_root / "trigger_0000"
            pass_dir.mkdir()
            _write_rgba_png(pass_dir / "rgb_0000.png")
            convert_rgb_outputs(pass_dir, "bmp", frame_padding=4, frames=[0])
            frame = {"frame": 0, "frame_id": "capture:frame:0000", "role": "capture"}

            artifacts = _discover_frame_artifacts(output_root, pass_dir, [frame], [])

            self.assertEqual(
                artifacts,
                [
                    {
                        "artifact_id": "artifact:capture:frame:0000:rgb:.bmp",
                        "frame_id": "capture:frame:0000",
                        "kind": "rgb",
                        "path": "trigger_0000/rgb_0000.bmp",
                        "role": "capture",
                    },
                ],
            )
            self.assertEqual(frame["artifacts"], {"rgb": ["trigger_0000/rgb_0000.bmp"]})

    def test_capture_converts_after_flush_and_before_the_grid_rename(self) -> None:
        class _Camera:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        context = IsaacExecutionContext(
            runtime=IsaacSdgRuntime(),
            stage=object(),
            scene={"root_path": "/Board"},
            component_types=("PASSIVE_SMD_0805",),
            plan_metadata={"random_seed": 7},
        )
        context.state.update(_capture_state("bmp"))

        render_product = mock.Mock(path="render-product")
        writer = mock.Mock()
        replicator = types.ModuleType("omni.replicator.core")
        replicator.create = mock.Mock()
        replicator.create.render_product.return_value = render_product
        replicator.WriterRegistry = mock.Mock()
        replicator.WriterRegistry.get.return_value = writer
        replicator.get = mock.Mock()
        replicator.get.prim_at_path.return_value = _Camera()
        replicator.modify = mock.Mock()
        replicator.distribution = mock.Mock()
        replicator.distribution.uniform.return_value = "rotation"
        replicator.orchestrator = mock.Mock()
        replicator.orchestrator.step_async = mock.AsyncMock()
        replicator.orchestrator.wait_until_complete_async = mock.AsyncMock()

        pxr = types.ModuleType("pxr")
        pxr.Gf = mock.Mock()
        pxr.Gf.Vec3d.side_effect = lambda x, y, z: (x, y, z)

        common = mock.Mock()
        common.configure_pathtracing.return_value = 2
        common.build_scan_positions.return_value = [(1.0, 2.0, 3.0)]
        common.find_translate_op.return_value = mock.Mock()
        common.randomize_lighting.return_value = {"ring_light": False}
        common.setup_augmentation.return_value = {"applied": False}
        common.scan_positions_for_trigger.side_effect = (
            lambda positions, trigger, seed: positions
        )
        common.cap_positions_for_max_outputs.side_effect = (
            lambda positions, pipeline, maximum, cursor: positions[: maximum - cursor]
        )
        common.grid_index_for_position.return_value = (0, 0)
        scan_grid_geometry = mock.Mock()
        scan_grid_geometry.compute_pixel_layout.return_value = {"supported": False}

        # Stand in for the writer's per-step output; the capture requires it.
        async def _write_frame(**_kwargs):
            if writer.attach.call_args is None:
                return  # a step before the writer is attached writes nothing
            pass_dir = Path(writer.initialize.call_args.kwargs["output_dir"])
            (pass_dir / "rgb_0000.png").write_bytes(b"")

        replicator.orchestrator.step_async = mock.AsyncMock(side_effect=_write_frame)

        # One recorder for both post-flush steps, so their relative order is
        # observable: the rename must see .bmp names, so conversion goes first.
        sequence = mock.Mock()
        sequence.attach_mock(common.rename_outputs_to_grid_index, "rename")
        convert = sequence.convert

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
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
                side_effect=lambda directory, name: {
                    "common": common,
                    "defect_ops": mock.Mock(),
                }[name],
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution.scan_grid_geometry",
                scan_grid_geometry,
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution.convert_rgb_outputs",
                convert,
            ),
            mock.patch("cdf.execution.isaac.capture_execution._ensure_camera"),
            mock.patch("cdf.execution.isaac.capture_execution._setup_lighting"),
            mock.patch(
                "cdf.execution.isaac.capture_execution._setup_component_appearance",
                return_value={"applied": False},
            ),
            mock.patch(
                "cdf.execution.isaac.capture_execution._warmup",
                new=mock.AsyncMock(),
            ),
        ):
            output = asyncio.run(
                execute_capture(
                    context,
                    CaptureRenderParameters(output_dir=temporary_directory),
                ),
            )

        self.assertEqual(output["frame_count"], 1)
        writer_kwargs = writer.initialize.call_args.kwargs
        self.assertEqual(writer_kwargs["image_output_format"], "png")
        pass_dir = writer_kwargs["output_dir"]
        self.assertEqual(
            [call[0] for call in sequence.mock_calls],
            ["convert", "rename"],
        )
        convert.assert_called_once_with(
            pass_dir,
            "bmp",
            frame_padding=4,
            frames=[0],
        )


if __name__ == "__main__":
    unittest.main()
