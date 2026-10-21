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

"""Unit tests for ``pcb_aoi/pairing.py`` — the Isaac-free pair math.

numpy-only; runs in the CI unit-test image without Isaac or Pillow.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pcb_aoi import pairing as pu


class BboxUnionTest(unittest.TestCase):
    def test_union_of_two(self):
        self.assertEqual(pu.bbox_union([1, 1, 3, 3], [2, 2, 5, 6]), [1, 1, 5, 6])

    def test_ignores_none(self):
        self.assertEqual(pu.bbox_union([1, 1, 3, 3], None), [1, 1, 3, 3])

    def test_all_none_returns_none(self):
        self.assertIsNone(pu.bbox_union(None, None))
        self.assertIsNone(pu.bbox_union())


class ResolveOcclusionTest(unittest.TestCase):
    """occlusion = Replicator native occlusionRatio on the defect pass.

    Pure convention/clamp logic (the annotator itself is Isaac-only and is
    exercised by the live smoke, not here).
    """

    def test_semantic_constant_is_the_real_definition(self):
        self.assertEqual(
            pu.OCCLUSION_SEMANTIC, "replicator_occlusion_ratio_defect_pass"
        )
        self.assertNotIn("placeholder", pu.OCCLUSION_SEMANTIC.lower())

    def test_missing_is_one_by_convention(self):
        # Even if the annotator reported nothing, missing -> 1.0.
        self.assertEqual(pu.resolve_occlusion(None, is_missing=True), 1.0)
        self.assertEqual(pu.resolve_occlusion(0.3, is_missing=True), 1.0)

    def test_present_component_passes_through_the_ratio(self):
        val = pu.resolve_occlusion(0.42, is_missing=False)
        self.assertIsInstance(val, float)
        self.assertAlmostEqual(val, 0.42)

    def test_zero_ratio_fully_visible(self):
        self.assertEqual(pu.resolve_occlusion(0.0, is_missing=False), 0.0)

    def test_clamped_to_unit_range(self):
        self.assertEqual(pu.resolve_occlusion(1.7, is_missing=False), 1.0)
        self.assertEqual(pu.resolve_occlusion(-0.2, is_missing=False), 0.0)

    def test_unavailable_ratio_is_none_for_present_component(self):
        self.assertIsNone(pu.resolve_occlusion(None, is_missing=False))


class SeverityMagnitudeTest(unittest.TestCase):
    def test_shift_reads_offset_mm(self):
        mag, field, units = pu.severity_magnitude(
            "shift", {"translate_magnitude_mm": 0.18}
        )
        self.assertAlmostEqual(mag, 0.18)
        self.assertEqual(field, "translate_magnitude_mm")
        self.assertEqual(units, "mm")

    def test_tombstone_reads_angle_deg(self):
        mag, _field, units = pu.severity_magnitude("tombstone", {"angle_deg": 82.0})
        self.assertAlmostEqual(mag, 82.0)
        self.assertEqual(units, "deg")

    def test_sideflip_reads_angle_deg(self):
        mag, _f, units = pu.severity_magnitude("sideflip", {"angle_deg": 75.5})
        self.assertAlmostEqual(mag, 75.5)
        self.assertEqual(units, "deg")

    def test_missing_is_null(self):
        self.assertEqual(pu.severity_magnitude("missing", {}), (None, None, None))

    def test_reverse_polarity_is_null_even_with_angle(self):
        # reverse_polarity is a fixed 180deg spin: categorical, no magnitude.
        self.assertEqual(
            pu.severity_magnitude("reverse_polarity", {"angle_deg": 180.0}),
            (None, None, None),
        )

    def test_solder_defect_is_null(self):
        self.assertEqual(pu.severity_magnitude("solder_defect", {}), (None, None, None))

    def test_absent_field_is_null(self):
        self.assertEqual(pu.severity_magnitude("shift", {}), (None, None, None))

    def test_unknown_type_is_null(self):
        self.assertEqual(
            pu.severity_magnitude("nonsense", {"angle_deg": 5}), (None, None, None)
        )

    def test_none_parameters_safe(self):
        self.assertEqual(pu.severity_magnitude("shift", None), (None, None, None))


class SeverityBucketTest(unittest.TestCase):
    def test_default_off_returns_none(self):
        # Bucket thresholds are application-specific; the default must not invent one.
        self.assertIsNone(pu.severity_bucket("shift", 0.18))
        self.assertIsNone(pu.severity_bucket("tombstone", 82.0))

    def test_none_magnitude_returns_none(self):
        self.assertIsNone(pu.severity_bucket("shift", None))

    def test_bucket_is_null_until_real_thresholds_exist(self):
        # The field is in the record schema, but the minor/moderate/severe cut
        # points are application-specific. Emitting a guess would be worse than
        # emitting null.
        for defect_type, magnitude in (
            ("shift", 0.05),
            ("shift", 0.9),
            ("tombstone", 80.0),
            ("missing", 1.0),
        ):
            with self.subTest(defect_type=defect_type, magnitude=magnitude):
                self.assertIsNone(pu.severity_bucket(defect_type, magnitude))


class MissingNullHandlingTest(unittest.TestCase):
    """The `missing` defect: bbox_new null, occlusion 1.0, severity null."""

    def test_missing_pair_semantics(self):
        bbox_old = [4, 4, 8, 8]
        bbox_new = (
            None  # component hidden in the defect pass -> null, not a placeholder
        )
        self.assertEqual(pu.bbox_union(bbox_old, bbox_new), bbox_old)
        # missing -> occlusion 1.0 by convention (native ratio absent/ignored).
        self.assertEqual(pu.resolve_occlusion(None, is_missing=True), 1.0)
        self.assertEqual(pu.severity_magnitude("missing", {}), (None, None, None))


if __name__ == "__main__":
    unittest.main()
