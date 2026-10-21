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

"""Tests for dependency-free planar geometry operations."""

import unittest

from pcb_aoi.geometry.planar import convex_hull_xy, cross_xy, signed_turn_xy


class PlanarGeometryTest(unittest.TestCase):
    """Verify reusable planar geometry calculations."""

    def test_cross_xy(self):
        self.assertEqual(cross_xy((1.0, 0.0), (0.0, 1.0)), 1.0)
        self.assertEqual(cross_xy((0.0, 1.0), (1.0, 0.0)), -1.0)

    def test_signed_turn_xy(self):
        self.assertGreater(
            signed_turn_xy((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            0.0,
        )
        self.assertEqual(
            signed_turn_xy((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)),
            0.0,
        )

    def test_convex_hull_xy_removes_duplicates_and_interior_points(self):
        hull = convex_hull_xy(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.5, 0.5),
                (1.0, 1.0),
            ),
        )

        self.assertEqual(
            hull,
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        )

    def test_convex_hull_xy_preserves_single_point(self):
        self.assertEqual(convex_hull_xy(((1.25, -2.5),)), ((1.25, -2.5),))


if __name__ == "__main__":
    unittest.main()
