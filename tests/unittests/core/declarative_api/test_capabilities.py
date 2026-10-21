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

"""Consistency tests for the CDF batch capability matrix.

The registry is the source of truth binding config, Python API, skills,
preview guidance, and documentation. These tests fail when a binding points
at something that does not exist or the generated matrix is stale.
"""

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: F401  # Registers the PCB-AOI domain.
from cdf import (
    CDFJob,
    capabilities,
)


class CapabilityMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.caps = capabilities.CAPABILITIES
        cls.job = CDFJob(domain="pcb-aoi")

    def test_ids_unique_and_modes_valid(self):
        ids = [c.id for c in self.caps]
        self.assertEqual(len(ids), len(set(ids)))
        for cap in self.caps:
            self.assertTrue(cap.modes, f"{cap.id}: no modes")
            self.assertEqual(cap.modes, {"batch"}, cap.id)

    def test_api_bindings_exist_on_job(self):
        for cap in self.caps:
            if not cap.api:
                continue
            section_name, method_name = cap.api.split(".")
            with self.subTest(cap.id):
                section = getattr(self.job, section_name)
                self.assertTrue(
                    callable(getattr(section, method_name)),
                    f"{cap.id}: job.{cap.api} is not callable",
                )

    def test_batch_capabilities_have_config_keys(self):
        for cap in self.caps:
            if "batch" in cap.modes:
                with self.subTest(cap.id):
                    self.assertTrue(
                        cap.config, f"{cap.id}: batch mode without config key"
                    )

    def test_capabilities_have_preview(self):
        for cap in self.caps:
            with self.subTest(cap.id):
                self.assertTrue(cap.preview, f"{cap.id}: no preview guidance")

    def test_doc_references_exist(self):
        for cap in self.caps:
            for doc in cap.docs:
                with self.subTest(f"{cap.id} -> {doc}"):
                    self.assertTrue(
                        (_REPO_ROOT / doc).is_file(), f"{cap.id}: doc {doc} missing"
                    )

    def test_registry_does_not_reference_skill_artifacts(self):
        for cap in self.caps:
            for doc in cap.docs:
                with self.subTest(f"{cap.id} -> {doc}"):
                    self.assertNotIn("SKILL.md", doc)
                    self.assertNotIn("skill-card", doc)

        generated = capabilities.to_markdown()
        self.assertNotIn("SKILL.md", generated)
        self.assertNotIn("skill-card", generated)

    def test_capabilities_markdown_is_fresh(self):
        generated = _REPO_ROOT / "cdf" / "CAPABILITIES.md"
        self.assertTrue(
            generated.is_file(),
            "generate with: python -m cdf.capabilities "
            "--domain pcb-aoi "
            "> cdf/CAPABILITIES.md",
        )
        self.assertEqual(
            generated.read_text(encoding="utf-8"),
            capabilities.to_markdown(),
            "cdf/CAPABILITIES.md is stale — regenerate with: "
            "python -m cdf.capabilities --domain pcb-aoi "
            "> cdf/CAPABILITIES.md",
        )

    def test_capabilities_markdown_explains_surfaces_and_implementations(self):
        matrix = capabilities.to_markdown()

        self.assertIn("Five surfaces per capability", matrix)
        self.assertNotIn("cdf.live", matrix)


if __name__ == "__main__":
    unittest.main()
