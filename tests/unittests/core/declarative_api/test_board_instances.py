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

"""Test typed shared-board instance execution policy without Isaac Sim."""

import unittest

from cdf import ValidationError
from pcb_aoi.execution.board_instances import (
    BoardInstanceCohort,
    ComponentProjectionRecord,
    InstanceSemanticProjection,
    prepare_instance_semantic_capture,
)


def _cohort() -> BoardInstanceCohort:
    """Return a two-placement cohort for pure policy tests."""
    return BoardInstanceCohort(
        instance_paths=("/World/Panel/I0", "/World/Panel/I1"),
        representative_instance_path="/World/Panel/I0",
        prototype_path="/__Prototype_1",
        prototype_source_path="/World/BoardDefinition",
    )


def _projection() -> InstanceSemanticProjection:
    """Return one catalog-derived component projection record."""
    return InstanceSemanticProjection(
        cohort=_cohort(),
        components=(
            ComponentProjectionRecord(
                source_path="/World/BoardDefinition/Components/R1",
                member_paths=(
                    "/World/BoardDefinition/Components/R1",
                    "/World/BoardDefinition/Layers/Bottom/Pad_R1_1",
                ),
                refdes="R1",
                class_name="resistor",
            ),
        ),
    )


class BoardInstanceCohortTest(unittest.TestCase):
    """Typed cohort path expansion and capture-policy tests."""

    def test_expand_source_path_preserves_cohort_order(self):
        targets = _cohort().expand_source_path(
            "/World/BoardDefinition/Components/R1",
        )

        self.assertEqual(
            tuple(target.composed_path for target in targets),
            (
                "/World/Panel/I0/Components/R1",
                "/World/Panel/I1/Components/R1",
            ),
        )
        self.assertEqual(tuple(target.instance_index for target in targets), (0, 1))

    def test_expand_source_path_rejects_another_board_root(self):
        with self.assertRaisesRegex(ValidationError, "outside prototype source"):
            _cohort().expand_source_path("/World/OtherBoard/Components/R1")

    def test_prepare_capture_enables_projection_and_defect_labels(self):
        writer = {
            "semantic_segmentation": True,
            "colorize_semantic_segmentation": True,
            "instance_id_segmentation": False,
            "colorize_instance_id_segmentation": False,
            "semantic_types": ("class", "defect"),
        }

        config = prepare_instance_semantic_capture(
            _projection(),
            writer,
            structural_records=(
                {
                    "path": "/World/BoardDefinition/Components/R1",
                    "defect": "shift",
                },
            ),
        )

        self.assertIsNotNone(config)
        self.assertFalse(writer["semantic_segmentation"])
        self.assertTrue(writer["instance_id_segmentation"])
        self.assertEqual(
            config["components"][0]["semantics"],
            {"class": "resistor", "defect": "shift"},
        )
        self.assertTrue(config["remove_instance_artifacts"])

    def test_prepare_capture_leaves_standalone_writer_unchanged(self):
        writer = {
            "semantic_segmentation": True,
            "instance_id_segmentation": False,
        }
        expected = dict(writer)

        config = prepare_instance_semantic_capture(None, writer)

        self.assertIsNone(config)
        self.assertEqual(writer, expected)

    def test_prepare_capture_rejects_unsupported_semantic_namespace(self):
        writer = {
            "semantic_segmentation": True,
            "semantic_types": ("material",),
        }

        with self.assertRaisesRegex(ValidationError, "only class and defect"):
            prepare_instance_semantic_capture(_projection(), writer)


if __name__ == "__main__":
    unittest.main()
