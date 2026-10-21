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

"""Tests for instance-ID-driven semantic segmentation projection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cdf.errors import ValidationError
from pcb_aoi.postprocess.semantic_projection import (
    _validate_source_ids,
    project_instance_semantics,
)


class SemanticProjectionTest(unittest.TestCase):
    """Verify catalog semantics are projected through composed prim paths."""

    def setUp(self) -> None:
        """Create a two-instance component projection."""
        self.projection = {
            "board_root": "/World/_class_Boards/Board",
            "instance_paths": ("/World/Panel/I0", "/World/Panel/I1"),
            "components": (
                {
                    "source_path": "/World/_class_Boards/Board/Components/R1",
                    "member_paths": (
                        "/World/_class_Boards/Board/Components/R1",
                        "/World/_class_Boards/Board/Layers/Bottom/Pad_R1_1",
                    ),
                    "semantics": {"class": "resistor"},
                },
                {
                    "source_path": "/World/_class_Boards/Board/Components/C1",
                    "semantics": {"class": "capacitor"},
                },
            ),
        }

    def test_projects_colorized_instance_paths_to_component_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            colors = np.asarray(
                [
                    [
                        (10, 20, 30, 255),
                        (40, 50, 60, 255),
                        (70, 80, 90, 255),
                        (0, 0, 0, 0),
                    ],
                ],
                dtype=np.uint8,
            )
            Image.fromarray(colors).save(
                root / "instance_id_segmentation_0000.png",
            )
            mapping = {
                "(10, 20, 30, 255)": ("/World/Panel/I0/Layers/Bottom/Pad_R1_1"),
                "(40, 50, 60, 255)": "/World/Panel/I1/Components/C1/contact",
                "(70, 80, 90, 255)": "/World/Panel/I0/Substrate",
            }
            (root / "instance_id_segmentation_mapping_0000.json").write_text(
                json.dumps(mapping),
                encoding="utf-8",
            )

            artifacts = project_instance_semantics(root, self.projection)

            self.assertEqual(artifacts[0]["semantic_groups"], 2)
            labels = json.loads(
                (root / "semantic_segmentation_labels_0000.json").read_text(
                    encoding="utf-8",
                ),
            )
            with Image.open(root / "semantic_segmentation_0000.png") as image:
                output_colors = list(image.convert("RGBA").get_flattened_data())
            self.assertEqual(labels[str(output_colors[0])], {"class": "resistor"})
            self.assertEqual(labels[str(output_colors[1])], {"class": "capacitor"})
            self.assertEqual(output_colors[2], (0, 0, 0, 255))
            self.assertEqual(output_colors[3], (0, 0, 0, 0))

    def test_supports_raw_ids_and_removes_internal_instance_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            instance_path = root / "instance_id_segmentation_0003.png"
            Image.fromarray(
                np.asarray([[7, 8]], dtype=np.uint16),
            ).save(instance_path)
            mapping_path = root / "instance_id_segmentation_mapping_0003.json"
            mapping_path.write_text(
                json.dumps(
                    {
                        "7": "/World/Panel/I0/Components/R1/body",
                        "8": "/World/Panel/I0/Substrate",
                    },
                ),
                encoding="utf-8",
            )

            project_instance_semantics(
                root,
                self.projection,
                colorized=False,
                remove_instance_artifacts=True,
            )

            with Image.open(root / "semantic_segmentation_0003.png") as image:
                output_ids = np.asarray(image, dtype=np.uint32).tolist()
            labels = json.loads(
                (root / "semantic_segmentation_labels_0003.json").read_text(
                    encoding="utf-8",
                ),
            )
            self.assertEqual(output_ids, [[2, 1]])
            self.assertEqual(labels["2"], {"class": "resistor"})
            self.assertFalse(instance_path.exists())
            self.assertFalse(mapping_path.exists())


class InstanceIdFrameValidationTest(unittest.TestCase):
    """Verify that unusable instance-ID frames fail instead of mislabeling."""

    def test_rejects_empty_frame_with_labeled_mapping(self):
        """An annotator that emitted nothing must not read as empty ground truth."""
        with self.assertRaises(ValidationError) as caught:
            _validate_source_ids(
                np.zeros((4, 4), np.uint32),
                {7: 2},
                Path("instance_id_segmentation_0000.png"),
                colorized=True,
            )
        self.assertIn("is empty", str(caught.exception))

    def test_rejects_raw_identifier_beyond_pixel_format(self):
        """A uint16 frame cannot carry a six-figure identifier."""
        with self.assertRaises(ValidationError) as caught:
            _validate_source_ids(
                np.ones((4, 4), np.uint16),
                {70000: 2},
                Path("instance_id_segmentation_0000.png"),
                colorized=False,
            )
        self.assertIn("colorize_instance_id_segmentation", str(caught.exception))

    def test_accepts_colorized_identifiers_beyond_uint16(self):
        """RGBA identifiers are 32-bit, so the same count is fine."""
        _validate_source_ids(
            np.ones((4, 4), np.uint32),
            {70000: 2},
            Path("instance_id_segmentation_0000.png"),
            colorized=True,
        )

    def test_accepts_empty_frame_when_nothing_is_labeled(self):
        """With no labeled instance, an empty frame is a legitimate result."""
        _validate_source_ids(
            np.zeros((4, 4), np.uint32),
            {},
            Path("instance_id_segmentation_0000.png"),
            colorized=True,
        )


if __name__ == "__main__":
    unittest.main()
