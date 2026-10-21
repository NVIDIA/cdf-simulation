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

"""Tests for canonical scan-grid image stitching."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image

from cdf.errors import ValidationError
from cdf.postprocess.image_stitching import (
    _colorized_label_identifiers,
    stitch_capture_grid,
)


class ImageStitchingTest(unittest.TestCase):
    """Verify frame placement from capture pixel-layout metadata."""

    def test_stitches_frames_into_declared_canvas(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trigger = root / "trigger_0000"
            trigger.mkdir()
            Image.new("RGB", (2, 2), (255, 0, 0)).save(trigger / "rgb_0000.png")
            Image.new("RGB", (2, 2), (0, 255, 0)).save(trigger / "rgb_0001.png")
            metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "supported": True,
                        "canvas_size_px": [4, 2],
                        "tiles": [
                            {
                                "frame": 0,
                                "dst_rect_xywh_px": [0, 0, 2, 2],
                            },
                            {
                                "frame": 1,
                                "dst_rect_xywh_px": [2, 0, 2, 2],
                            },
                        ],
                    },
                },
            ]

            artifacts = stitch_capture_grid(
                root,
                metadata,
                output_filename="board.png",
            )

            with Image.open(artifacts[0]["path"]) as stitched:
                self.assertEqual(stitched.size, (4, 2))
                self.assertEqual(stitched.getpixel((0, 0)), (255, 0, 0))
                self.assertEqual(stitched.getpixel((3, 0)), (0, 255, 0))

    def test_stitches_grid_index_renamed_frames(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trigger = root / "trigger_0000"
            trigger.mkdir()
            Image.new("RGB", (2, 2), (255, 0, 0)).save(trigger / "rgb_x0_y0.png")
            Image.new("RGB", (2, 2), (0, 255, 0)).save(trigger / "rgb_x1_y0.png")
            metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "supported": True,
                        "canvas_size_px": [4, 2],
                        "tiles": [
                            {
                                "frame": 0,
                                "x_idx": 0,
                                "y_idx": 0,
                                "dst_rect_xywh_px": [0, 0, 2, 2],
                            },
                            {
                                "frame": 1,
                                "x_idx": 1,
                                "y_idx": 0,
                                "dst_rect_xywh_px": [2, 0, 2, 2],
                            },
                        ],
                    },
                },
            ]

            artifacts = stitch_capture_grid(
                root,
                metadata,
                output_filename="board.png",
            )

            with Image.open(artifacts[0]["path"]) as stitched:
                self.assertEqual(stitched.size, (4, 2))
                self.assertEqual(stitched.getpixel((0, 0)), (255, 0, 0))
                self.assertEqual(stitched.getpixel((3, 0)), (0, 255, 0))

    def test_feather_blend_crossfades_overlapping_tiles(self):
        """Overlapping tiles blend instead of one overwriting the other."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trigger = root / "trigger_0000"
            trigger.mkdir()
            Image.new("RGB", (8, 4), (0, 0, 0)).save(trigger / "rgb_0000.png")
            Image.new("RGB", (8, 4), (255, 255, 255)).save(trigger / "rgb_0001.png")
            # a 4 px overlap: tile 1 starts before tile 0 ends
            metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "supported": True,
                        "canvas_size_px": [12, 4],
                        "tiles": [
                            {"frame": 0, "dst_rect_xywh_px": [0, 0, 8, 4]},
                            {"frame": 1, "dst_rect_xywh_px": [4, 0, 8, 4]},
                        ],
                    },
                },
            ]

            pasted = stitch_capture_grid(
                root,
                metadata,
                output_filename="pasted.png",
                blend="none",
            )
            feathered = stitch_capture_grid(
                root,
                metadata,
                output_filename="feathered.png",
                blend="feather",
            )

            with Image.open(pasted[0]["path"]) as image:
                # the later tile simply wins the whole overlap
                self.assertEqual(image.getpixel((4, 2)), (255, 255, 255))
            with Image.open(feathered[0]["path"]) as image:
                self.assertEqual(image.size, (12, 4))
                # outside the overlap each tile still owns its pixels
                self.assertEqual(image.getpixel((0, 2)), (0, 0, 0))
                self.assertEqual(image.getpixel((11, 2)), (255, 255, 255))
                # inside it the value is a blend, and it rises left to right
                overlap = [image.getpixel((x, 2))[0] for x in range(4, 8)]
                self.assertTrue(all(0 < value < 255 for value in overlap), overlap)
                self.assertEqual(overlap, sorted(overlap), overlap)

    def test_writes_image_to_usd_sidecar_when_layout_has_world_placement(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trigger = root / "trigger_0000"
            trigger.mkdir()
            Image.new("RGB", (2, 2), (255, 0, 0)).save(trigger / "rgb_0000.png")
            Image.new("RGB", (2, 2), (0, 255, 0)).save(trigger / "rgb_0001.png")
            metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "supported": True,
                        "canvas_size_px": [4, 2],
                        "camera_side": "top",
                        "world_reference_z": 12.5,
                        "stage_meters_per_unit": 0.001,
                        "stage_up_axis": "Z",
                        "source_usd": "omniverse://example/board.usd",
                        "pixel_to_world_xy": {
                            "origin_xy": [-4.0, 1.0],
                            "u_per_px_xy": [1.0, 0.0],
                            "v_per_px_xy": [0.0, -1.0],
                        },
                        "tiles": [
                            {"frame": 0, "dst_rect_xywh_px": [0, 0, 2, 2]},
                            {"frame": 1, "dst_rect_xywh_px": [2, 0, 2, 2]},
                        ],
                    },
                },
            ]

            artifacts = stitch_capture_grid(
                root,
                metadata,
                output_filename="board.png",
            )

            self.assertIn("image_to_usd_path", artifacts[0])
            sidecar = json.loads(
                Path(artifacts[0]["image_to_usd_path"]).read_text(encoding="utf-8"),
            )
            self.assertEqual(sidecar["reference_plane_z"], 12.5)
            self.assertEqual(sidecar["stage_meters_per_unit"], 0.001)
            self.assertEqual(sidecar["stage_up_axis"], "Z")
            self.assertEqual(sidecar["source_usd"], "omniverse://example/board.usd")
            self.assertEqual(sidecar["projection"], "orthographic")
            self.assertEqual(sidecar["pixel_indexing_convention"], "pixel-edge")
            self.assertEqual(sidecar["pixel_coordinate_system"], "top-left-xy-pixels")
            corners = sidecar["corners_usd"]
            self.assertEqual(corners["top_left"]["pixel_xy"], [0.0, 0.0])
            self.assertEqual(corners["top_left"]["usd_xyz"], [-4.0, 1.0, 12.5])
            self.assertEqual(corners["top_right"]["pixel_xy"], [4.0, 0.0])
            self.assertEqual(corners["top_right"]["usd_xyz"], [0.0, 1.0, 12.5])
            self.assertEqual(corners["bottom_left"]["pixel_xy"], [0.0, 2.0])
            self.assertEqual(corners["bottom_left"]["usd_xyz"], [-4.0, -1.0, 12.5])
            self.assertEqual(corners["bottom_right"]["pixel_xy"], [4.0, 2.0])
            self.assertEqual(corners["bottom_right"]["usd_xyz"], [0.0, -1.0, 12.5])

    def test_sidecar_follows_the_rounding_the_stitcher_applied(self):
        """Fractional tile offsets must not desync the transform from the pixels.

        The layout places tiles in floating point; the stitcher rounds each
        destination rectangle before pasting. Deriving the affine from the
        unrounded layout described a placement the image does not have, by up
        to half a pixel. The sidecar re-anchors on the realized placement, so
        here a tile asking for x=0.4 lands at x=0 and the transform reports the
        world position of the pixel that was actually written.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trigger = root / "trigger_0000"
            trigger.mkdir()
            Image.new("RGB", (2, 2), (255, 0, 0)).save(trigger / "rgb_0000.png")
            Image.new("RGB", (2, 2), (0, 255, 0)).save(trigger / "rgb_0001.png")
            metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "supported": True,
                        "canvas_size_px": [4, 2],
                        "camera_side": "top",
                        "world_reference_z": 12.5,
                        "stage_meters_per_unit": 0.001,
                        "stage_up_axis": "Z",
                        "source_usd": "omniverse://example/board.usd",
                        "pixel_to_world_xy": {
                            "origin_xy": [-4.0, 1.0],
                            "u_per_px_xy": [1.0, 0.0],
                            "v_per_px_xy": [0.0, -1.0],
                        },
                        # 0.4 rounds to 0, so the realized origin is 0.4 px to
                        # the LEFT of what the unrounded layout claimed.
                        "tiles": [
                            {"frame": 0, "dst_rect_xywh_px": [0.4, 0.0, 2, 2]},
                            {"frame": 1, "dst_rect_xywh_px": [2.4, 0.0, 2, 2]},
                        ],
                    },
                },
            ]

            artifacts = stitch_capture_grid(
                root,
                metadata,
                output_filename="board.png",
            )
            sidecar = json.loads(
                Path(artifacts[0]["image_to_usd_path"]).read_text(encoding="utf-8"),
            )

            # u_per_px is (1, 0), so a -0.4 px shift moves the origin -0.4 in x.
            top_left = sidecar["corners_usd"]["top_left"]["usd_xyz"]
            self.assertAlmostEqual(top_left[0], -4.4, places=6)
            self.assertAlmostEqual(top_left[1], 1.0, places=6)
            # Scale is untouched: only the anchor moves.
            affine = sidecar["affine_pixel_to_usd"]
            self.assertEqual(affine["u_per_px_xy"], [1.0, 0.0])
            self.assertEqual(affine["v_per_px_xy"], [0.0, -1.0])

    def _cropped_capture(self, root, image_size_px):
        """Two 3x3 cells side by side: a 6x3 mosaic, columns tagged by value."""
        trigger = root / "trigger_0000"
        trigger.mkdir()
        for frame, offset in ((0, 0), (1, 3)):
            tile = np.zeros((3, 3, 3), np.uint8)
            for column in range(3):
                tile[:, column, 0] = 10 * (offset + column)
            Image.fromarray(tile, "RGB").save(trigger / f"rgb_{frame:04d}.png")
        return [
            {
                "trigger_idx": 0,
                "pixel_layout": {
                    "supported": True,
                    "canvas_size_px": [6, 3],
                    "image_size_px": image_size_px,
                    "camera_side": "top",
                    "world_reference_z": 0.0,
                    "stage_meters_per_unit": 0.001,
                    "pixel_to_world_xy": {
                        "origin_xy": [0.0, 0.0],
                        "u_per_px_xy": [0.5, 0.0],
                        "v_per_px_xy": [0.0, -0.5],
                    },
                    "x_num": 2,
                    "y_num": 1,
                    "tiles": [
                        {"frame": 0, "dst_rect_xywh_px": [0, 0, 3, 3]},
                        {"frame": 1, "dst_rect_xywh_px": [3, 0, 3, 3]},
                    ],
                },
            },
        ]

    def test_a_stated_image_size_crops_the_mosaic_centred(self):
        """Whole-pixel cells can overshoot the size a scanner states; the stitch
        crops the surplus, splitting it between the edges, and the sidecar
        describes the cropped image rather than the mosaic."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifacts = stitch_capture_grid(
                root, self._cropped_capture(root, [4, 2]), output_filename="board.png"
            )
            self.assertEqual(artifacts[0]["canvas_size_px"], [4, 2])
            with Image.open(artifacts[0]["path"]) as stitched:
                self.assertEqual(stitched.size, (4, 2))
                # One column off the left, one off the right; the surplus row
                # (odd) goes off the bottom.
                self.assertEqual(stitched.getpixel((0, 0))[0], 10)
                self.assertEqual(stitched.getpixel((3, 0))[0], 40)
            sidecar = json.loads(
                Path(artifacts[0]["image_to_usd_path"]).read_text(encoding="utf-8"),
            )
            self.assertEqual(sidecar["image_size_px"], [4, 2])
            # Pixel (0, 0) of the crop is mosaic pixel (1, 0): 0.5 world units in.
            self.assertEqual(
                sidecar["corners_usd"]["top_left"]["usd_xyz"], [0.5, 0.0, 0.0]
            )
            self.assertEqual(
                sidecar["corners_usd"]["bottom_right"]["usd_xyz"], [2.5, -1.0, 0.0]
            )
            self.assertAlmostEqual(sidecar["fov"]["scan_area_mm"][0], 4 * 0.5, places=9)

    def test_a_mosaic_smaller_than_the_stated_size_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValidationError, "smaller than the stated"):
                stitch_capture_grid(
                    root, self._cropped_capture(root, [7, 3]), output_filename="b.png"
                )

    def test_omits_image_to_usd_sidecar_without_world_placement(self):
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
                        "tiles": [{"frame": 0, "dst_rect_xywh_px": [0, 0, 2, 2]}],
                    },
                },
            ]

            artifacts = stitch_capture_grid(root, metadata, output_filename="board.png")

            self.assertNotIn("image_to_usd_path", artifacts[0])
            self.assertFalse(
                list(Path(artifacts[0]["path"]).parent.glob("image_*_to_usd.json")),
            )

    def test_rejects_unknown_blend_mode(self):
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
                        "tiles": [{"frame": 0, "dst_rect_xywh_px": [0, 0, 2, 2]}],
                    },
                },
            ]

            with self.assertRaisesRegex(ValidationError, "blend"):
                stitch_capture_grid(root, metadata, blend="crossfade")


class RepeatedCellTest(unittest.TestCase):
    """A stale frame standing in for another cell must not be stitched."""

    @staticmethod
    def _stitch(first, second, frames=None):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trigger = root / "trigger_0000"
            trigger.mkdir()
            Image.fromarray(first).save(trigger / "rgb_0000.png")
            Image.fromarray(second).save(trigger / "rgb_0001.png")
            height, width = first.shape[:2]
            metadata = {
                "trigger_idx": 0,
                "pixel_layout": {
                    "supported": True,
                    "canvas_size_px": [width, 2 * height],
                    "tiles": [
                        {
                            "frame": 0,
                            "x_idx": 0,
                            "y_idx": 0,
                            "dst_rect_xywh_px": [0, 0, width, height],
                        },
                        {
                            "frame": 1,
                            "x_idx": 0,
                            "y_idx": 1,
                            "dst_rect_xywh_px": [0, height, width, height],
                        },
                    ],
                },
            }
            if frames is not None:
                metadata["frames"] = frames
            return stitch_capture_grid(root, [metadata])

    def test_identical_textured_cells_are_rejected(self):
        textured = np.random.default_rng(1).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        frames = [
            {"frame": 0, "camera": {"position": [0.0, 77.5, 10.0]}},
            {"frame": 1, "camera": {"position": [0.0, -22.5, 10.0]}},
        ]
        with self.assertRaisesRegex(
            ValidationError, r"frame 1 .*-22\.5.* pixel-identical .*frame 0"
        ):
            self._stitch(textured, textured.copy(), frames)

    def test_identical_flat_background_cells_are_stitched(self):
        flat = np.full((8, 8, 3), 226, dtype=np.uint8)
        artifacts = self._stitch(flat, flat.copy())
        self.assertEqual(artifacts[0]["canvas_size_px"], [8, 16])

    def test_identical_flat_coloured_rgba_background_is_stitched(self):
        # Off-board cells of a large fov: one flat colour, constant alpha.
        flat = np.empty((8, 8, 4), dtype=np.uint8)
        flat[...] = (60, 30, 20, 255)
        artifacts = self._stitch(flat, flat.copy())
        self.assertEqual(artifacts[0]["canvas_size_px"], [8, 16])

    def test_identical_textured_rgba_cells_are_rejected(self):
        textured = np.random.default_rng(3).integers(0, 256, (8, 8, 4), dtype=np.uint8)
        textured[..., 3] = 255
        with self.assertRaisesRegex(ValidationError, "pixel-identical"):
            self._stitch(textured, textured.copy())

    def test_distinct_cells_are_stitched(self):
        rng = np.random.default_rng(2)
        first = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        second = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        artifacts = self._stitch(first, second)
        self.assertEqual(artifacts[0]["tile_count"], 2)

    @staticmethod
    def _stitch_segmentation(first_labels, second_labels, blend):
        """Stitch two pixel-identical masks whose ID 9 is textured enough to be checked."""
        mask = np.asarray([[0, 9], [9, 9]], np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trigger = root / "trigger_0000"
            trigger.mkdir()
            for frame, labels in ((0, first_labels), (1, second_labels)):
                Image.fromarray(mask, "L").save(
                    trigger / f"semantic_segmentation_{frame:04d}.png",
                )
                (trigger / f"semantic_segmentation_labels_{frame:04d}.json").write_text(
                    json.dumps(labels),
                    encoding="utf-8",
                )
            metadata = {
                "trigger_idx": 0,
                "pixel_layout": {
                    "supported": True,
                    "canvas_size_px": [4, 2],
                    "tiles": [
                        {"frame": 0, "dst_rect_xywh_px": [0, 0, 2, 2]},
                        {"frame": 1, "dst_rect_xywh_px": [2, 0, 2, 2]},
                    ],
                },
            }
            return stitch_capture_grid(
                root,
                [metadata],
                image_type="semantic_segmentation",
                output_filename="mask.png",
                blend=blend,
            )

    def test_identical_ids_with_different_classes_are_stitched(self):
        """ID 9 is solder in one frame and pin in the other: not a repeated cell."""
        for blend in ("labels", "none"):
            with self.subTest(blend=blend):
                artifacts = self._stitch_segmentation(
                    {"0": {"class": "BACKGROUND"}, "9": {"class": "solder"}},
                    {"0": {"class": "BACKGROUND"}, "9": {"class": "pin"}},
                    blend,
                )
                self.assertEqual(artifacts[0]["tile_count"], 2)

    def test_identical_masks_with_identical_classes_are_rejected(self):
        """Same pixels and same meaning in two cells is still a stale frame."""
        labels = {"0": {"class": "BACKGROUND"}, "9": {"class": "solder"}}
        with self.assertRaisesRegex(ValidationError, "pixel-identical"):
            self._stitch_segmentation(labels, dict(labels), "labels")


class LabelAwareStitchTest(unittest.TestCase):
    """Verify that segmentation tiles are unified into one class table."""

    def _write_tile(self, trigger_dir, frame, pixels, labels):
        """Write one segmentation tile and its per-frame label sidecar."""
        Image.fromarray(np.asarray(pixels, np.uint8), "L").save(
            trigger_dir / f"semantic_segmentation_{frame:04d}.png",
        )
        (trigger_dir / f"semantic_segmentation_labels_{frame:04d}.json").write_text(
            json.dumps(labels),
            encoding="utf-8",
        )

    def test_unifies_identifiers_that_differ_between_tiles(self):
        """Identifier 2 means solder in one tile and pin in the other."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trigger_dir = root / "trigger_0000"
            trigger_dir.mkdir()
            self._write_tile(
                trigger_dir,
                0,
                [[0, 2], [2, 2]],
                {
                    "0": {"class": "BACKGROUND"},
                    "2": {"class": "solder"},
                },
            )
            self._write_tile(
                trigger_dir,
                1,
                [[0, 2], [2, 2]],
                {
                    "0": {"class": "BACKGROUND"},
                    "2": {"class": "pin"},
                },
            )
            metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "supported": True,
                        "canvas_size_px": [4, 2],
                        "tiles": [
                            {"frame": 0, "dst_rect_xywh_px": [0, 0, 2, 2]},
                            {"frame": 1, "dst_rect_xywh_px": [2, 0, 2, 2]},
                        ],
                    },
                },
            ]

            artifacts = stitch_capture_grid(
                root,
                metadata,
                image_type="semantic_segmentation",
                output_filename="mask.png",
                blend="labels",
            )

            table = json.loads(
                Path(artifacts[0]["labels_path"]).read_text(encoding="utf-8"),
            )
            classes = {value["class"]: int(key) for key, value in table.items()}
            self.assertEqual(classes["BACKGROUND"], 0)
            self.assertEqual(len({classes["solder"], classes["pin"]}), 2)

            stitched = np.asarray(Image.open(artifacts[0]["path"]))
            self.assertEqual(int(stitched[1, 0]), classes["solder"])
            self.assertEqual(int(stitched[1, 3]), classes["pin"])

    def test_stitches_colorized_grid_tiles_with_label_sidecars(self):
        """Grid-renamed RGBA tiles retain a unified color label table."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trigger_dir = root / "trigger_0000"
            trigger_dir.mkdir()
            solder = (92, 141, 92, 255)
            pin = (184, 32, 148, 255)
            for x_index, color, class_name in (
                (0, solder, "solder"),
                (1, pin, "pin"),
            ):
                suffix = f"x{x_index}_y0"
                pixels = np.zeros((2, 2, 4), np.uint8)
                pixels[:, :] = color
                Image.fromarray(pixels, "RGBA").save(
                    trigger_dir / f"semantic_segmentation_{suffix}.png",
                )
                labels = {
                    "(0, 0, 0, 0)": {"class": "BACKGROUND"},
                    str(color): {"class": class_name},
                }
                (
                    trigger_dir / f"semantic_segmentation_labels_{suffix}.json"
                ).write_text(json.dumps(labels), encoding="utf-8")
            metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "supported": True,
                        "canvas_size_px": [4, 2],
                        "tiles": [
                            {
                                "frame": 0,
                                "x_idx": 0,
                                "y_idx": 0,
                                "dst_rect_xywh_px": [0, 0, 2, 2],
                            },
                            {
                                "frame": 1,
                                "x_idx": 1,
                                "y_idx": 0,
                                "dst_rect_xywh_px": [2, 0, 2, 2],
                            },
                        ],
                    },
                },
            ]

            artifacts = stitch_capture_grid(
                root,
                metadata,
                image_type="semantic_segmentation",
                output_filename="mask.png",
                blend="labels",
            )

            table = json.loads(
                Path(artifacts[0]["labels_path"]).read_text(encoding="utf-8"),
            )
            self.assertEqual(
                {values["class"] for values in table.values()},
                {"BACKGROUND", "solder", "pin"},
            )
            with Image.open(artifacts[0]["path"]) as stitched:
                self.assertEqual(stitched.mode, "RGBA")
                self.assertNotEqual(stitched.getpixel((0, 0)), (0, 0, 0, 0))
                self.assertNotEqual(
                    stitched.getpixel((0, 0)),
                    stitched.getpixel((3, 0)),
                )

    def test_recolors_source_tiles_with_explicit_label_colors(self):
        """Source tiles and sidecars use the stitched explicit color map."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trigger_dir = root / "trigger_0000"
            trigger_dir.mkdir()
            original_colors = ((92, 141, 92, 255), (184, 32, 148, 255))
            classes = ("solder", "pin")
            target_colors = {
                "solder": (255, 0, 0, 255),
                "pin": (0, 255, 0, 255),
            }
            for frame, (color, class_name) in enumerate(
                zip(original_colors, classes),
            ):
                pixels = np.zeros((2, 2, 4), np.uint8)
                pixels[:, :] = color
                Image.fromarray(pixels, "RGBA").save(
                    trigger_dir / f"semantic_segmentation_{frame:04d}.png",
                )
                labels = {
                    "(0, 0, 0, 0)": {"class": "BACKGROUND"},
                    str(color): {"class": class_name},
                }
                (
                    trigger_dir / f"semantic_segmentation_labels_{frame:04d}.json"
                ).write_text(json.dumps(labels), encoding="utf-8")
            metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "supported": True,
                        "canvas_size_px": [4, 2],
                        "tiles": [
                            {"frame": 0, "dst_rect_xywh_px": [0, 0, 2, 2]},
                            {"frame": 1, "dst_rect_xywh_px": [2, 0, 2, 2]},
                        ],
                    },
                },
            ]

            artifacts = stitch_capture_grid(
                root,
                metadata,
                image_type="semantic_segmentation",
                output_filename="mask.png",
                blend="labels",
                recolor_tiles=True,
                color_map=target_colors,
            )

            for frame, class_name in enumerate(classes):
                image_path = trigger_dir / f"semantic_segmentation_{frame:04d}.png"
                with Image.open(image_path) as tile:
                    self.assertEqual(tile.getpixel((0, 0)), target_colors[class_name])
                labels_path = (
                    trigger_dir / f"semantic_segmentation_labels_{frame:04d}.json"
                )
                labels = json.loads(labels_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    labels[str(target_colors[class_name])]["class"], class_name
                )

            with Image.open(artifacts[0]["path"]) as stitched:
                self.assertEqual(stitched.getpixel((0, 0)), target_colors["solder"])
                self.assertEqual(stitched.getpixel((3, 0)), target_colors["pin"])

    def test_recolor_tiles_requires_label_blending(self):
        """Tile recoloring is only valid for label-aware stitching."""
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValidationError, "requires blend='labels'"),
        ):
            stitch_capture_grid(
                directory,
                [],
                recolor_tiles=True,
            )

    def test_rejects_duplicate_explicit_label_colors(self):
        """Two classes may not share one user-supplied RGBA identifier."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trigger_dir = root / "trigger_0000"
            trigger_dir.mkdir()
            solder = (92, 141, 92, 255)
            pin = (184, 32, 148, 255)
            for frame, color, class_name in (
                (0, solder, "solder"),
                (1, pin, "pin"),
            ):
                pixels = np.zeros((2, 2, 4), np.uint8)
                pixels[:, :] = color
                Image.fromarray(pixels, "RGBA").save(
                    trigger_dir / f"semantic_segmentation_{frame:04d}.png",
                )
                labels = {
                    "(0, 0, 0, 0)": {"class": "BACKGROUND"},
                    str(color): {"class": class_name},
                }
                (
                    trigger_dir / f"semantic_segmentation_labels_{frame:04d}.json"
                ).write_text(json.dumps(labels), encoding="utf-8")
            metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "supported": True,
                        "canvas_size_px": [4, 2],
                        "tiles": [
                            {"frame": 0, "dst_rect_xywh_px": [0, 0, 2, 2]},
                            {"frame": 1, "dst_rect_xywh_px": [2, 0, 2, 2]},
                        ],
                    },
                },
            ]

            with self.assertRaisesRegex(ValidationError, "color_map assigns RGBA"):
                stitch_capture_grid(
                    root,
                    metadata,
                    image_type="semantic_segmentation",
                    blend="labels",
                    color_map={
                        "solder": (100, 120, 140, 255),
                        "pin": (100, 120, 140, 255),
                    },
                )

    def test_rejects_explicit_reserved_label_color(self):
        """A class may not use a reserved segmentation identifier."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trigger_dir = root / "trigger_0000"
            trigger_dir.mkdir()
            solder = (92, 141, 92, 255)
            pixels = np.zeros((2, 2, 4), np.uint8)
            pixels[:, :] = solder
            Image.fromarray(pixels, "RGBA").save(
                trigger_dir / "semantic_segmentation_0000.png",
            )
            labels = {
                "(0, 0, 0, 0)": {"class": "BACKGROUND"},
                str(solder): {"class": "solder"},
            }
            (trigger_dir / "semantic_segmentation_labels_0000.json").write_text(
                json.dumps(labels), encoding="utf-8"
            )
            metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "supported": True,
                        "canvas_size_px": [2, 2],
                        "tiles": [
                            {"frame": 0, "dst_rect_xywh_px": [0, 0, 2, 2]},
                        ],
                    },
                },
            ]

            with self.assertRaisesRegex(ValidationError, "BACKGROUND"):
                stitch_capture_grid(
                    root,
                    metadata,
                    image_type="semantic_segmentation",
                    blend="labels",
                    color_map={"solder": (0, 0, 0, 0)},
                )

    def test_rejects_invalid_explicit_label_colors(self):
        """Explicit identifiers must contain exactly four byte channels."""
        keys = ((("class", "solder"),),)
        invalid_maps = (
            ({"solder": (1, 2, 3)}, "RGBA 4-tuple"),
            ({"solder": (300, 0, 0, 255)}, "integers in \\[0, 255\\]"),
        )

        for color_map, message in invalid_maps:
            with (
                self.subTest(color_map=color_map),
                self.assertRaisesRegex(ValidationError, message),
            ):
                _colorized_label_identifiers(keys, color_map=color_map)

    def test_requires_label_sidecars(self):
        """A segmentation stitch without its label files is a hard error."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trigger_dir = root / "trigger_0000"
            trigger_dir.mkdir()
            Image.fromarray(np.zeros((2, 2), np.uint8), "L").save(
                trigger_dir / "semantic_segmentation_0000.png",
            )
            metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "supported": True,
                        "canvas_size_px": [2, 2],
                        "tiles": [{"frame": 0, "dst_rect_xywh_px": [0, 0, 2, 2]}],
                    },
                },
            ]

            with self.assertRaises(ValidationError):
                stitch_capture_grid(
                    root,
                    metadata,
                    image_type="semantic_segmentation",
                    blend="labels",
                )


@unittest.skipIf(Image is None, "Pillow not available")
class WriteLabelsTest(unittest.TestCase):
    """Verify write_labels passthrough label-sidecar behavior."""

    def _make_tile_pair(self, trigger_dir, solder_color, pin_color):
        """Write two consistent colorized tiles with their label sidecars."""
        for x_index, color, class_name in (
            (0, solder_color, "solder"),
            (1, pin_color, "pin"),
        ):
            suffix = f"x{x_index}_y0"
            pixels = np.zeros((2, 2, 4), np.uint8)
            pixels[:, :] = color
            Image.fromarray(pixels, "RGBA").save(
                trigger_dir / f"semantic_segmentation_{suffix}.png",
            )
            labels = {
                "(0, 0, 0, 0)": {"class": "BACKGROUND"},
                str(color): {"class": class_name},
            }
            (trigger_dir / f"semantic_segmentation_labels_{suffix}.json").write_text(
                json.dumps(labels), encoding="utf-8"
            )

    _METADATA: ClassVar[list[dict[str, object]]] = [
        {
            "trigger_idx": 0,
            "pixel_layout": {
                "supported": True,
                "canvas_size_px": [4, 2],
                "tiles": [
                    {
                        "frame": 0,
                        "x_idx": 0,
                        "y_idx": 0,
                        "dst_rect_xywh_px": [0, 0, 2, 2],
                    },
                    {
                        "frame": 1,
                        "x_idx": 1,
                        "y_idx": 0,
                        "dst_rect_xywh_px": [2, 0, 2, 2],
                    },
                ],
            },
        },
    ]

    def test_writes_passthrough_label_json(self):
        """write_labels emits a label sidecar with original tile identifiers."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trigger_dir = root / "trigger_0000"
            trigger_dir.mkdir()
            solder = (92, 141, 92, 255)
            pin = (184, 32, 148, 255)
            self._make_tile_pair(trigger_dir, solder, pin)

            artifacts = stitch_capture_grid(
                root,
                self._METADATA,
                image_type="semantic_segmentation",
                output_filename="mask.png",
                write_labels=True,
            )

            self.assertIn("labels_path", artifacts[0])
            table = json.loads(
                Path(artifacts[0]["labels_path"]).read_text(encoding="utf-8"),
            )
            self.assertEqual(
                {v["class"] for v in table.values()},
                {"BACKGROUND", "solder", "pin"},
            )
            self.assertIn(str(solder), table)
            self.assertIn(str(pin), table)
            with Image.open(artifacts[0]["path"]) as stitched:
                self.assertEqual(stitched.getpixel((0, 0)), solder)
                self.assertEqual(stitched.getpixel((2, 0)), pin)

    def test_requires_label_sidecars(self):
        """write_labels raises when a tile's sidecar is missing."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trigger_dir = root / "trigger_0000"
            trigger_dir.mkdir()
            Image.fromarray(np.zeros((2, 2, 4), np.uint8), "RGBA").save(
                trigger_dir / "semantic_segmentation_x0_y0.png",
            )
            Image.fromarray(np.zeros((2, 2, 4), np.uint8), "RGBA").save(
                trigger_dir / "semantic_segmentation_x1_y0.png",
            )

            with self.assertRaisesRegex(ValidationError, "write_labels requires"):
                stitch_capture_grid(
                    root,
                    self._METADATA,
                    image_type="semantic_segmentation",
                    write_labels=True,
                )

    def test_warns_on_inconsistent_tile_colors(self):
        """write_labels warns when two tiles use different colors for the same class."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trigger_dir = root / "trigger_0000"
            trigger_dir.mkdir()
            # Both tiles claim their own color maps to "solder"
            for x_index, color in ((0, (92, 141, 92, 255)), (1, (55, 100, 55, 255))):
                suffix = f"x{x_index}_y0"
                pixels = np.zeros((2, 2, 4), np.uint8)
                pixels[:, :] = color
                Image.fromarray(pixels, "RGBA").save(
                    trigger_dir / f"semantic_segmentation_{suffix}.png",
                )
                labels = {
                    "(0, 0, 0, 0)": {"class": "BACKGROUND"},
                    str(color): {"class": "solder"},
                }
                (
                    trigger_dir / f"semantic_segmentation_labels_{suffix}.json"
                ).write_text(json.dumps(labels), encoding="utf-8")

            with self.assertWarns(RuntimeWarning):
                stitch_capture_grid(
                    root,
                    self._METADATA,
                    image_type="semantic_segmentation",
                    output_filename="mask.png",
                    write_labels=True,
                )

    def test_rejects_write_labels_with_blend_labels(self):
        """write_labels=True combined with blend=labels is rejected at runtime."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trigger_0000").mkdir()

            with self.assertRaisesRegex(ValidationError, "write_labels"):
                stitch_capture_grid(
                    root,
                    self._METADATA,
                    image_type="semantic_segmentation",
                    blend="labels",
                    write_labels=True,
                )
