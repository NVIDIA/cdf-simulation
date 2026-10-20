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

"""Kit-free tests for meniscus placement on a rotated component.

``resolve_pad_orientation`` answers with an axis *index* and a sign, so it can
express only four outward directions. A component at a non-quadrant rotation is
therefore measured in its own yaw frame and the finished meniscus is rotated
back, which is what ``MeniscusParameters.yaw_degrees`` carries.

The expected values here are derived independently — from the rotation matrix
and from the closed form for a rotated rectangle's bounding box — never by
restating what the implementation computes.
"""

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pcb_aoi.solder_generation.procedural_meniscus_geometry import (  # noqa: E402
    MeniscusParameters,
    build_pad_meniscus,
)
from pcb_aoi.solder_generation.procedural_meniscus_geometry import (  # noqa: E402
    _world_transform,
)

# A two-terminal chip at the origin, its right-hand land overhanging in +X.
PAD_BOUNDS = (0.60, -0.65, 0.0, 1.55, 0.65, 0.035)
BODY_BOUNDS = (-1.00, -0.63, 0.035, 1.00, 0.63, 1.07)
COMPONENT_BOUNDS = (-1.02, -0.65, 0.035, 1.02, 0.65, 1.07)


def _footprint(mesh) -> tuple[float, float]:
    """Return the XY footprint of a meniscus once its transform is applied.

    Args:
        mesh: Result of :func:`build_pad_meniscus`.

    Returns:
        Extent along X and along Y.
    """
    matrix = mesh.transform
    xs, ys = [], []
    for point in mesh.points:
        xs.append(
            point[0] * matrix[0] + point[1] * matrix[4] + point[2] * matrix[8] + matrix[12],
        )
        ys.append(
            point[0] * matrix[1] + point[1] * matrix[5] + point[2] * matrix[9] + matrix[13],
        )
    return max(xs) - min(xs), max(ys) - min(ys)


class WorldTransformYawTest(unittest.TestCase):
    """The yaw composes onto the meniscus transform as a planar rotation."""

    def test_zero_yaw_is_unchanged(self) -> None:
        """A zero yaw must leave every matrix element exactly as it was."""
        without = _world_transform((1.0, 2.0, 3.0), 0, 1.0, 2.0, 3.0, 4.0, bottom=False)
        with_zero = _world_transform(
            (1.0, 2.0, 3.0), 0, 1.0, 2.0, 3.0, 4.0, bottom=False, yaw_degrees=0.0,
        )
        self.assertEqual(without, with_zero)

    def test_yaw_rotates_basis_and_origin(self) -> None:
        """Each row rotates in the board plane; the normal row is untouched."""
        matrix = _world_transform(
            (1.0, 0.0, 3.0), 0, 1.0, 2.0, 3.0, 4.0, bottom=False, yaw_degrees=90.0,
        )
        # +X basis of length 2 turns into +Y; +Y basis of length 3 into -X;
        # the origin (1, 0) turns into (0, 1). The normal row keeps its scale.
        self.assertAlmostEqual(matrix[0], 0.0)
        self.assertAlmostEqual(matrix[1], 2.0)
        self.assertAlmostEqual(matrix[4], -3.0)
        self.assertAlmostEqual(matrix[5], 0.0)
        self.assertAlmostEqual(matrix[10], 4.0)
        self.assertAlmostEqual(matrix[12], 0.0)
        self.assertAlmostEqual(matrix[13], 1.0)
        self.assertAlmostEqual(matrix[14], 3.0)

    def test_yaw_preserves_orthogonality_and_scale(self) -> None:
        """An arbitrary yaw must stay a rigid rotation of the same basis."""
        matrix = _world_transform(
            (0.7, -0.3, 0.2), 1, -1.0, 1.3, 0.9, 0.4, bottom=False, yaw_degrees=37.0,
        )
        x_axis = matrix[0:3]
        y_axis = matrix[4:7]
        self.assertAlmostEqual(sum(a * b for a, b in zip(x_axis, y_axis)), 0.0)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in x_axis)), 1.3)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in y_axis)), 0.9)


class MeniscusYawFootprintTest(unittest.TestCase):
    """A meniscus built in a yaw frame lands rotated by exactly that yaw."""

    def setUp(self) -> None:
        """Build the shared unrotated reference meniscus."""
        self.parameters = MeniscusParameters(resolution=12, scene_units_per_mm=1.0)
        self.reference = build_pad_meniscus(
            PAD_BOUNDS, BODY_BOUNDS, COMPONENT_BOUNDS, "Top", 2, self.parameters,
        )
        self.assertIsNotNone(self.reference)

    def test_quadrant_yaw_leaves_geometry_identical(self) -> None:
        """Points are frame-independent; only the transform carries the yaw."""
        rotated = build_pad_meniscus(
            PAD_BOUNDS, BODY_BOUNDS, COMPONENT_BOUNDS, "Top", 2,
            replace(self.parameters, yaw_degrees=90.0),
        )
        self.assertEqual(rotated.points, self.reference.points)
        self.assertEqual(rotated.face_vertex_indices, self.reference.face_vertex_indices)

    def test_diagonal_yaw_gives_the_closed_form_footprint(self) -> None:
        """A rectangle w x h at 45 degrees spans (w + h) / sqrt(2) on both axes.

        This is the check that fails on the axis-index-only implementation: it
        can only place the meniscus along +/-X or +/-Y, so the footprint comes
        back strongly anisotropic instead of square.
        """
        width, height = _footprint(self.reference)
        rotated = build_pad_meniscus(
            PAD_BOUNDS, BODY_BOUNDS, COMPONENT_BOUNDS, "Top", 2,
            replace(self.parameters, yaw_degrees=45.0),
        )
        expected = (width + height) / math.sqrt(2.0)
        actual_x, actual_y = _footprint(rotated)
        self.assertAlmostEqual(actual_x, expected, places=6)
        self.assertAlmostEqual(actual_y, expected, places=6)

    def test_arbitrary_yaw_matches_the_rotation_of_every_point(self) -> None:
        """Every placed vertex equals the unrotated vertex turned by the yaw."""
        yaw = 37.0
        angle = math.radians(yaw)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = build_pad_meniscus(
            PAD_BOUNDS, BODY_BOUNDS, COMPONENT_BOUNDS, "Top", 2,
            replace(self.parameters, yaw_degrees=yaw),
        )

        def place(mesh, point):
            m = mesh.transform
            return (
                point[0] * m[0] + point[1] * m[4] + point[2] * m[8] + m[12],
                point[0] * m[1] + point[1] * m[5] + point[2] * m[9] + m[13],
                point[0] * m[2] + point[1] * m[6] + point[2] * m[10] + m[14],
            )

        for point in self.reference.points[::37]:
            base = place(self.reference, point)
            turned = (
                base[0] * cos_a - base[1] * sin_a,
                base[0] * sin_a + base[1] * cos_a,
                base[2],
            )
            got = place(rotated, point)
            for expected_value, actual_value in zip(turned, got):
                self.assertAlmostEqual(actual_value, expected_value, places=6)


class ComponentYawRecoveryTest(unittest.TestCase):
    """Yaw is read from a planar instance transform, and refused on a tilt."""

    class _Prim:
        """Minimal valid-prim stub."""

        def __bool__(self) -> bool:
            """Report the stub as present."""
            return True

        def IsValid(self) -> bool:  # noqa: N802 -- mirrors the USD API
            """Report the stub as valid."""
            return True

    class _Stage:
        """Stage stub returning one prim for any path."""

        def GetPrimAtPath(self, path):  # noqa: N802 -- mirrors the USD API
            """Return the stub prim.

            Args:
                path: Ignored prim path.

            Returns:
                The stub prim.
            """
            return ComponentYawRecoveryTest._Prim()

    class _XformCache:
        """Xform cache stub returning a fixed row-major matrix."""

        def __init__(self, rows):
            """Store the matrix rows.

            Args:
                rows: Four rows of four floats.
            """
            self.rows = rows

        def GetLocalToWorldTransform(self, prim):  # noqa: N802 -- mirrors USD
            """Return the stored matrix.

            Args:
                prim: Ignored prim.

            Returns:
                The stored matrix rows.
            """
            return self.rows

    @staticmethod
    def _planar(degrees: float):
        """Build a row-major planar rotation about the board normal.

        Args:
            degrees: Rotation about Z.

        Returns:
            Four rows of four floats.
        """
        angle = math.radians(degrees)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return (
            (cos_a, sin_a, 0.0, 0.0),
            (-sin_a, cos_a, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )

    def test_planar_rotation_is_recovered(self) -> None:
        """A pure Z rotation reads back as the angle that was authored."""
        from pcb_aoi.solder_generation.procedural_solver import _component_yaw

        for degrees in (0.0, 37.0, 45.0, 90.0, -120.0):
            with self.subTest(degrees=degrees):
                cache = self._XformCache(self._planar(degrees))
                recovered = _component_yaw(self._Stage(), "/any", cache)
                self.assertAlmostEqual(
                    math.cos(math.radians(recovered)),
                    math.cos(math.radians(degrees)),
                    places=9,
                )
                self.assertAlmostEqual(
                    math.sin(math.radians(recovered)),
                    math.sin(math.radians(degrees)),
                    places=9,
                )

    def test_out_of_plane_tilt_refuses_rather_than_guesses(self) -> None:
        """A tilted part keeps the previous behaviour instead of a false yaw.

        The yaw frame does not describe a tilted component, and returning a
        confident number for one would swap a visible error for a hidden one.
        """
        from pcb_aoi.solder_generation.procedural_solver import _component_yaw

        angle = math.radians(30.0)
        tilted = (
            (math.cos(angle), 0.0, -math.sin(angle), 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (math.sin(angle), 0.0, math.cos(angle), 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        recovered = _component_yaw(self._Stage(), "/any", self._XformCache(tilted))
        self.assertEqual(recovered, 0.0)


if __name__ == "__main__":
    unittest.main()
