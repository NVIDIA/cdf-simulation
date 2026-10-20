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

"""Tests that the YAML config exposes stitching, variation and geometry labels."""

from __future__ import annotations

import unittest

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi domain
from cdf import CDFJob
from pcb_aoi.config.annotations import AnnotationsConfig, apply_annotations
from pcb_aoi.config.appearance import MaterialOverrideConfig
from pcb_aoi.config.postprocess import PostprocessConfig, apply_postprocess


class YamlStitchTest(unittest.TestCase):
    """Verify the stitch blend mode and several products per pass."""

    def test_blend_reaches_the_declaration(self):
        """A YAML job can ask for feathered overlap."""
        job = CDFJob(domain="pcb-aoi", random_seed=0)
        config = PostprocessConfig.model_validate(
            {"stitch": {"source_pass": "full_board", "blend": "feather"}},
        )
        apply_postprocess(job, config)
        self.assertEqual(job.postprocess.declared_stitch["blend"], "feather")

    def test_several_products_of_one_pass(self):
        """The image and its segmentation are both declared."""
        job = CDFJob(domain="pcb-aoi", random_seed=0)
        config = PostprocessConfig.model_validate(
            {
                "stitch": [
                    {"source_pass": "full_board", "blend": "feather"},
                    {
                        "source_pass": "full_board",
                        "image_type": "semantic_segmentation",
                        "output_filename": "mask.png",
                        "blend": "labels",
                    },
                ],
            },
        )
        apply_postprocess(job, config)
        self.assertEqual(
            [entry["image_type"] for entry in job.postprocess.declared_stitches],
            ["rgb", "semantic_segmentation"],
        )

    def test_blend_defaults_to_none(self):
        """An existing YAML job keeps pasting tiles."""
        config = PostprocessConfig.model_validate({"stitch": {}})
        self.assertEqual(config.stitch.blend, "none")


class YamlMaterialVariationTest(unittest.TestCase):
    """Verify that randomisation ranges survive the YAML round trip."""

    def test_variation_is_parsed(self):
        """Ranges are read as pairs, not strings."""
        config = MaterialOverrideConfig.model_validate(
            {
                "material_path": "/World/Looks/Carrier",
                "variation": {"roughness": [0.55, 0.85]},
            },
        )
        self.assertEqual(config.variation, {"roughness": (0.55, 0.85)})

    def test_variation_defaults_to_absent(self):
        """A material without ranges stays deterministic."""
        config = MaterialOverrideConfig.model_validate(
            {"material_path": "/World/Looks/Carrier"},
        )
        self.assertIsNone(config.variation)


class YamlGeometryLabelTest(unittest.TestCase):
    """Verify that geometry classes can be declared from YAML."""

    def test_geometry_classes_reach_the_declaration(self):
        """Solder and terminal classes are applied to the job."""
        job = CDFJob(domain="pcb-aoi", random_seed=0)
        config = AnnotationsConfig.model_validate(
            {"geometry": {"solder": "solder", "terminals": "pin"}},
        )
        apply_annotations(job, config, component_metadata_source=None)
        self.assertEqual(
            job.annotations.geometry_labels,
            {"solder": "solder", "terminals": "pin"},
        )

    def test_absent_geometry_leaves_labels_unset(self):
        """A YAML job that does not ask for them is unchanged."""
        job = CDFJob(domain="pcb-aoi", random_seed=0)
        apply_annotations(
            job,
            AnnotationsConfig.model_validate({}),
            component_metadata_source=None,
        )
        self.assertIsNone(job.annotations.geometry_labels)


if __name__ == "__main__":
    unittest.main()
