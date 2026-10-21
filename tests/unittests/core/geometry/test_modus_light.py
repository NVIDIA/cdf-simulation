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

"""Pure-numpy geometry tests for the Modus light (2D-dome linear AOI rig).

``pcb_aoi.modus_light`` has zero Omniverse/USD dependency, so the full
geometry — light count, L/R symmetry, colour order, elevations, rect height,
and the aim-at-centre orientation — is unit-testable on the host. The USD
authoring wrapper (``common.build_modus_lights``) is a thin translation of
these specs into UsdLux prims and is exercised by the live render.
"""

import unittest

import numpy as np

from pcb_aoi.modus_light import (
    COLOR_ORDER,
    DEFAULT_HEIGHT_SCALE,
    compute_modus_layout,
)

# A CAD-board-scale board: 183.87 x 234.25 x 18.14 mm, centred off-origin so the
# tests catch any hard-coded world-origin assumptions.
_BOARD = dict(
    board_center=(91.9, 117.1, 9.07),
    board_top_z=18.14,
    board_x_extent=183.87,
    board_y_extent=234.25,
)


def _layout(**overrides):
    kwargs = dict(_BOARD)
    kwargs.update(overrides)
    return compute_modus_layout(**kwargs)


class ModusGeometryTest(unittest.TestCase):

    def setUp(self):
        self.lights = _layout()

    def test_exactly_six_lights(self):
        self.assertEqual(len(self.lights), 6)

    def test_three_colour_groups_two_each(self):
        by_group = {}
        for lt in self.lights:
            by_group.setdefault(lt.color_group, []).append(lt)
        self.assertEqual(set(by_group), {"white", "red", "blue"})
        for group, members in by_group.items():
            self.assertEqual(len(members), 2, group)
            sides = sorted(m.side for m in members)
            self.assertEqual(sides, [-1, 1], "%s not L/R symmetric" % group)

    def test_colour_order_white_red_blue_centre_out(self):
        # slot_from_center 1 = innermost (nearest camera) .. 3 = outermost
        slot = {lt.color_group: lt.slot_from_center for lt in self.lights}
        self.assertEqual(slot["white"], 1)
        self.assertEqual(slot["red"], 2)
        self.assertEqual(slot["blue"], 3)
        self.assertEqual(COLOR_ORDER, ("white", "red", "blue"))

    def test_left_right_symmetry_mirrors_in_x(self):
        cx = _BOARD["board_center"][0]
        for group in ("white", "red", "blue"):
            pair = [lt for lt in self.lights if lt.color_group == group]
            right = next(lt for lt in pair if lt.side == 1)
            left = next(lt for lt in pair if lt.side == -1)
            # X mirrored about board centre; Y and Z identical
            self.assertAlmostEqual(right.position[0] - cx, cx - left.position[0])
            self.assertAlmostEqual(right.position[1], left.position[1])
            self.assertAlmostEqual(right.position[2], left.position[2])

    def test_blue_elevation_is_45(self):
        blues = [lt for lt in self.lights if lt.color_group == "blue"]
        for lt in blues:
            self.assertAlmostEqual(lt.elevation_deg, 45.0)

    def test_default_elevations_even_15deg_steps(self):
        elev = {lt.color_group: lt.elevation_deg for lt in self.lights}
        self.assertAlmostEqual(elev["white"], 75.0)
        self.assertAlmostEqual(elev["red"], 60.0)
        self.assertAlmostEqual(elev["blue"], 45.0)

    def test_height_is_two_times_board_y(self):
        expected = DEFAULT_HEIGHT_SCALE * _BOARD["board_y_extent"]
        self.assertAlmostEqual(expected, 2.0 * _BOARD["board_y_extent"])
        for lt in self.lights:
            self.assertAlmostEqual(lt.height, expected)

    def test_height_scale_param_respected(self):
        lights = _layout(height_scale=3.0)
        for lt in lights:
            self.assertAlmostEqual(lt.height, 3.0 * _BOARD["board_y_extent"])

    def test_width_from_width_frac_of_board_x(self):
        lights = _layout(width_frac=0.2)
        for lt in lights:
            self.assertAlmostEqual(lt.width, 0.2 * _BOARD["board_x_extent"])

    def test_explicit_width_overrides_frac(self):
        lights = _layout(width=12.5, width_frac=0.99)
        for lt in lights:
            self.assertAlmostEqual(lt.width, 12.5)

    def test_all_lights_share_board_centre_y(self):
        cy = _BOARD["board_center"][1]
        for lt in self.lights:
            self.assertAlmostEqual(lt.position[1], cy)

    def test_long_axis_is_world_y(self):
        # the rect's long/height axis (local Y) must stay world +Y so it reads
        # as a vertical bar
        for lt in self.lights:
            np.testing.assert_allclose(lt.y_axis, (0.0, 1.0, 0.0), atol=1e-12)

    def test_lights_sit_above_the_board_top(self):
        top = _BOARD["board_top_z"]
        for lt in self.lights:
            self.assertGreater(lt.position[2], top)

    def test_normal_points_from_light_to_the_board_top_face(self):
        # The arc is placed relative to board_top_z, so the aim has to use the
        # same reference or the delivered elevation is not the requested one.
        # This used to aim at the bbox centre, which made the real beam angle
        # steeper than the spec by however thick the board was.
        cx, cy, _ = _BOARD["board_center"]
        aim = np.array((cx, cy, _BOARD["board_top_z"]), dtype=float)
        for lt in self.lights:
            pos = np.array(lt.position, dtype=float)
            to_aim = aim - pos
            to_aim_unit = to_aim / np.linalg.norm(to_aim)
            normal = np.array(lt.normal, dtype=float)
            np.testing.assert_allclose(normal, to_aim_unit, atol=1e-9)
            self.assertAlmostEqual(np.linalg.norm(normal), 1.0)

    def test_delivered_elevation_matches_the_request(self):
        # The point of sharing one Z reference: the angle the rig actually
        # delivers is the angle that was asked for, and it does not drift with
        # board thickness.
        cx, cy, _ = _BOARD["board_center"]
        aim = np.array((cx, cy, _BOARD["board_top_z"]), dtype=float)
        for lt in self.lights:
            pos = np.array(lt.position, dtype=float)
            d = pos - aim
            horizontal = float(np.hypot(d[0], d[1]))
            delivered = np.degrees(np.arctan2(float(d[2]), horizontal))
            self.assertAlmostEqual(delivered, float(lt.elevation_deg), places=6)

    def test_layout_is_independent_of_board_thickness(self):
        # Same top face, thicker board: the lights must not move.
        thick = _layout(board_center=(91.9, 117.1, 2.0))
        # strict: a missing or extra light would otherwise be skipped by zip
        # and the comparison would pass on a partial match.
        for a, b in zip(self.lights, thick, strict=True):
            np.testing.assert_allclose(np.array(a.position),
                                       np.array(b.position), atol=1e-9)
            np.testing.assert_allclose(np.array(a.normal),
                                       np.array(b.normal), atol=1e-9)

    def test_emission_is_local_minus_z(self):
        # UsdLux RectLight emits toward local -Z; z_axis must equal -normal
        for lt in self.lights:
            np.testing.assert_allclose(
                np.array(lt.z_axis), -np.array(lt.normal), atol=1e-12)

    def test_local_basis_is_right_handed_orthonormal(self):
        for lt in self.lights:
            x = np.array(lt.x_axis)
            y = np.array(lt.y_axis)
            z = np.array(lt.z_axis)
            for a in (x, y, z):
                self.assertAlmostEqual(np.linalg.norm(a), 1.0)
            self.assertAlmostEqual(float(np.dot(x, y)), 0.0)
            self.assertAlmostEqual(float(np.dot(y, z)), 0.0)
            self.assertAlmostEqual(float(np.dot(x, z)), 0.0)
            # right-handed: x = y cross z
            np.testing.assert_allclose(np.cross(y, z), x, atol=1e-9)

    def test_arc_radius_scales_with_board_size(self):
        # X offset from centre = R*cos(elev), Z offset above top = R*sin(elev);
        # doubling arc_radius_scale doubles both offsets
        base = {lt.name: lt for lt in _layout(arc_radius_scale=1.0)}
        wide = {lt.name: lt for lt in _layout(arc_radius_scale=2.0)}
        cx = _BOARD["board_center"][0]
        top = _BOARD["board_top_z"]
        for name, lt in base.items():
            wlt = wide[name]
            self.assertAlmostEqual(2.0 * (lt.position[0] - cx),
                                   wlt.position[0] - cx)
            self.assertAlmostEqual(2.0 * (lt.position[2] - top),
                                   wlt.position[2] - top)

    def test_arc_placement_matches_polar_formula(self):
        # X = cx +/- R*cos(elev), Z = top + R*sin(elev), R = scale*max(bx,by)
        scale = 1.5
        R = scale * max(_BOARD["board_x_extent"], _BOARD["board_y_extent"])
        cx = _BOARD["board_center"][0]
        top = _BOARD["board_top_z"]
        for lt in _layout(arc_radius_scale=scale):
            elev = np.radians(lt.elevation_deg)
            self.assertAlmostEqual(abs(lt.position[0] - cx), R * np.cos(elev))
            self.assertAlmostEqual(lt.position[2] - top, R * np.sin(elev), places=5)

    def test_per_group_intensity_and_colour_overrides(self):
        lights = _layout(
            intensities={"white": 1234.0, "red": 55.0, "blue": 7.0},
            colors={"white": (0.9, 0.9, 0.9), "red": (1.0, 0.0, 0.0),
                    "blue": (0.0, 0.0, 1.0)})
        by_group = {}
        for lt in lights:
            by_group[lt.color_group] = lt
        self.assertAlmostEqual(by_group["white"].intensity, 1234.0)
        self.assertAlmostEqual(by_group["red"].intensity, 55.0)
        self.assertAlmostEqual(by_group["blue"].intensity, 7.0)
        self.assertEqual(by_group["blue"].color, (0.0, 0.0, 1.0))

    def test_custom_elevations_respected(self):
        lights = _layout(elevations={"white": 80.0, "red": 65.0, "blue": 45.0})
        elev = {lt.color_group: lt.elevation_deg for lt in lights}
        self.assertAlmostEqual(elev["white"], 80.0)
        self.assertAlmostEqual(elev["red"], 65.0)


if __name__ == "__main__":
    unittest.main()
