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

"""A sideflip with yaw must still hinge on the part's own support edge.

Raised in review on !151: the flip was built from a pivot and axis measured on
the UN-yawed frame, so once ``rotate_z_jitter_deg`` was non-zero the part rotated
about a line that was no longer one of its edges. The lift that follows then
pushed it back onto the board, which produced a plausible pose but not the one
the parameters describe.

The invariant this pins is exact and needs no stage: **the hinge line must be
fixed by the flip**. A point on the hinge cannot move when the part topples
about it -- that is what "hinge" means -- and it is precisely what breaks when
the axis and the geometry disagree.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kitless  # noqa: E402


def _load():
    """Import the rotation module with the Kit-only names stubbed out.

    ``pxr`` is among them, and has to be: the CI unit-test image has no USD, so
    stubbing only ``omni`` passes on a developer machine and errors in CI.
    """
    return kitless.import_standalone("defects_utils.component_rotation",
                                     "defects_utils.transform_utils")


class YawedHingeTest(unittest.TestCase):
    """The hinge travels with the yaw, so the flip still turns on the edge."""

    @classmethod
    def setUpClass(cls):
        cls.rotation, cls.transform = _load()

    @classmethod
    def tearDownClass(cls):
        kitless.forget_standalone()

    # A part 4 x 2 x 0.8, resting on the board plane at z = 0. Its long axis
    # runs along x; the support edge it topples over is the +y one.
    _CENTER = np.array([0.0, 0.0, 0.4])
    _NORMAL = np.array([0.0, 0.0, 1.0])
    _LONG_AXIS = np.array([1.0, 0.0, 0.0])
    _PIVOT = np.array([0.0, 1.0, 0.0])

    def _compose(self, yaw_deg, angle_deg):
        """Build the same yaw-then-flip the sideflip path builds."""
        pivot_rotation = self.transform._pivot_rotation_matrix
        local = np.eye(4)
        hinge_pivot, hinge_axis = self._PIVOT, self._LONG_AXIS
        if yaw_deg:
            yaw = pivot_rotation(self._CENTER, self._NORMAL, yaw_deg)
            local = local @ yaw
            hinge_pivot, hinge_axis = self.rotation._yawed_hinge(
                self._PIVOT, self._LONG_AXIS, yaw)
        local = local @ pivot_rotation(hinge_pivot, hinge_axis, angle_deg)
        return local, hinge_pivot, hinge_axis

    def _apply(self, point, matrix):
        homogeneous = np.append(np.asarray(point, dtype=float), 1.0)
        return (homogeneous @ matrix)[:3]

    def test_the_hinge_line_is_fixed_by_the_flip(self):
        for yaw_deg in (0.0, 3.0, -7.5, 15.0, 90.0):
            with self.subTest(yaw=yaw_deg):
                local, pivot, axis = self._compose(yaw_deg, angle_deg=70.0)
                flip = self.transform._pivot_rotation_matrix(pivot, axis, 70.0)
                for t in (-2.0, 0.0, 1.3):
                    on_hinge = np.asarray(pivot) + t * np.asarray(axis)
                    np.testing.assert_allclose(
                        self._apply(on_hinge, flip), on_hinge, atol=1e-9,
                        err_msg="the flip moved a point on its own hinge",
                    )

    def test_the_hinge_axis_stays_a_unit_direction(self):
        for yaw_deg in (3.0, -7.5, 15.0, 90.0):
            with self.subTest(yaw=yaw_deg):
                _local, _pivot, axis = self._compose(yaw_deg, angle_deg=70.0)
                self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0,
                                       places=9)

    def test_the_hinge_rotates_with_the_yaw(self):
        # At 90 degrees about the board normal, a hinge along +x becomes +y.
        _local, _pivot, axis = self._compose(90.0, angle_deg=70.0)
        np.testing.assert_allclose(axis, [0.0, 1.0, 0.0], atol=1e-9)

    def test_zero_yaw_changes_nothing(self):
        local_zero, pivot, axis = self._compose(0.0, angle_deg=70.0)
        np.testing.assert_allclose(pivot, self._PIVOT, atol=1e-12)
        np.testing.assert_allclose(axis, self._LONG_AXIS, atol=1e-12)
        # And the composed transform is exactly the plain flip, so the default
        # path is untouched by this change.
        plain = self.transform._pivot_rotation_matrix(
            self._PIVOT, self._LONG_AXIS, 70.0)
        np.testing.assert_allclose(local_zero, plain, atol=1e-12)

    def test_the_un_yawed_axis_would_not_have_been_fixed(self):
        """The regression this locks: hinging on the old axis moves the edge."""
        yaw_deg = 20.0
        local, pivot, axis = self._compose(yaw_deg, angle_deg=70.0)
        wrong = self.transform._pivot_rotation_matrix(
            self._PIVOT, self._LONG_AXIS, 70.0)
        on_hinge = np.asarray(pivot) + 1.0 * np.asarray(axis)
        moved = self._apply(on_hinge, wrong)
        self.assertGreater(
            float(np.linalg.norm(moved - on_hinge)), 1e-3,
            "the old hinge should visibly move the yawed edge; if it does not, "
            "this test is not exercising the defect it was written for",
        )


if __name__ == "__main__":
    unittest.main()
