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

"""Test through-hole lead variation and the limits that bound it."""

from __future__ import annotations

import math
import unittest

import numpy as np

from pcb_aoi.geometry.tht_pin_pose import (
    WALL_GAP_MM,
    Lead,
    PinVariation,
    apply_pin_variation,
    barrel_tilt_limit,
    densify_prism,
    max_radius_in_barrel,
    probe_azimuths,
    tilt_rotation,
    wall_overrun_mm,
)

_BOARD_TOP, _BOARD_BOTTOM = 0.0, -1.6


def square_lead(
    half_width: float = 0.25,
    hole_radius: float = 0.50,
    centre: tuple[float, float] = (0.0, 0.0),
    protrusion: float = 1.95,
) -> Lead:
    """Build a square-section lead through a board, far cap ring first.

    A square pin is the interesting case: its clearance depends on which way it
    leans, which is why the barrel limit cannot be a formula.

    Args:
        half_width: Half the lead's cross-section width.
        hole_radius: Bore radius.
        centre: Bore centre in XY.
        protrusion: How far the untrimmed lead stands proud of the far face.

    Returns:
        The lead.
    """
    corners = [
        (centre[0] - half_width, centre[1] - half_width),
        (centre[0] + half_width, centre[1] - half_width),
        (centre[0] + half_width, centre[1] + half_width),
        (centre[0] - half_width, centre[1] + half_width),
    ]
    low = _BOARD_BOTTOM - protrusion
    high = _BOARD_TOP + 2.0
    points = np.array(
        [(x, y, low) for x, y in corners] + [(x, y, high) for x, y in corners],
        float,
    )
    return Lead(
        points=points,
        hole_center=centre,
        hole_radius_mm=hole_radius,
        barrel_z=(_BOARD_BOTTOM, _BOARD_TOP),
        board_far_face_z=_BOARD_BOTTOM,
        tip_is_low=True,
    )


class TiltRotationTest(unittest.TestCase):
    """The rotation itself."""

    def test_is_orthonormal_with_unit_determinant(self) -> None:
        """A rotation may not scale or mirror the lead."""
        for tilt in (0.0, 3.7, 15.0, 47.0):
            for azimuth in (0.0, 37.0, 180.0, 311.0):
                with self.subTest(tilt=tilt, azimuth=azimuth):
                    m = tilt_rotation(tilt, azimuth)
                    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
                    self.assertAlmostEqual(float(np.linalg.det(m)), 1.0)

    def test_zero_tilt_is_the_identity(self) -> None:
        """No lean must mean no transform, at every azimuth."""
        for azimuth in (0.0, 90.0, 217.0):
            with self.subTest(azimuth=azimuth):
                np.testing.assert_allclose(
                    tilt_rotation(0.0, azimuth), np.eye(3), atol=1e-12,
                )

    def test_the_lead_leans_toward_the_azimuth(self) -> None:
        """The sampled azimuth has to mean what it says.

        A sign error here puts every joint on the wrong side of its pin and
        still renders as a plausible board, so it is asserted directly.
        """
        for azimuth in (0.0, 90.0, 180.0, 270.0, 311.0):
            with self.subTest(azimuth=azimuth):
                axis = tilt_rotation(12.0, azimuth) @ np.array([0.0, 0.0, 1.0])
                self.assertAlmostEqual(
                    math.degrees(math.atan2(axis[1], axis[0])) % 360.0,
                    azimuth % 360.0,
                )


class DensifyTest(unittest.TestCase):
    """The measuring instrument, and why it exists."""

    def test_raw_vertices_miss_the_contact_a_dense_probe_finds(self) -> None:
        """A prism has no vertex where a tilted lead touches the bore.

        This is the whole reason for densifying: measured on the raw eight
        vertices the lead reports as clear, and it is through the copper.
        """
        lead = square_lead(half_width=0.25, hole_radius=0.42)
        rotation = tilt_rotation(12.0, 45.0)
        z_low, z_high = lead.barrel_z
        raw = max_radius_in_barrel(
            lead.points, lead.pivot, rotation, lead.hole_center, z_low, z_high,
        )
        dense = max_radius_in_barrel(
            densify_prism(lead.points, z_low, z_high),
            lead.pivot,
            rotation,
            lead.hole_center,
            z_low,
            z_high,
        )
        self.assertGreater(dense, raw)
        self.assertLess(raw, lead.hole_radius_mm)
        self.assertGreater(dense, lead.hole_radius_mm)


class BarrelTiltLimitTest(unittest.TestCase):
    """The limit that competes with the request."""

    def test_a_tight_bore_is_the_binding_constraint(self) -> None:
        """A 15 degree request does not survive a normal clearance."""
        lead = square_lead(half_width=0.25, hole_radius=0.42)
        self.assertLess(barrel_tilt_limit(lead, 15.0), 15.0)

    def test_a_loose_bore_reaches_the_cap(self) -> None:
        """With room to spare the request is what binds, not the barrel."""
        lead = square_lead(half_width=0.20, hole_radius=1.60)
        self.assertEqual(barrel_tilt_limit(lead, 15.0), 15.0)

    def test_a_looser_bore_permits_more_tilt(self) -> None:
        """The limit tracks the clearance rather than a hardcoded angle."""
        tight = barrel_tilt_limit(square_lead(hole_radius=0.42), 15.0)
        loose = barrel_tilt_limit(square_lead(hole_radius=0.60), 15.0)
        self.assertGreater(loose, tight)

    def test_a_lead_already_against_the_wall_gets_no_tilt(self) -> None:
        """An interference fit is reported as zero, not as a small angle."""
        lead = square_lead(half_width=0.50, hole_radius=0.45)
        self.assertEqual(barrel_tilt_limit(lead, 15.0), 0.0)

    def test_a_uniform_ring_alone_steps_over_a_square_leads_worst_case(
        self,
    ) -> None:
        """The corner directions are load-bearing, not belt and braces.

        A 12-step ring probes every 30 degrees and never looks at 45, which is
        exactly where a square lead's corner is. Solving on the ring alone
        returns a limit that puts the pin through its own plating, so this
        pins the reason the corner directions are probed.
        """
        lead = square_lead(half_width=0.25, hole_radius=0.42)
        z_low, z_high = lead.barrel_z
        probe = densify_prism(lead.points, z_low, z_high)
        allowed = lead.hole_radius_mm - WALL_GAP_MM

        ring = np.linspace(0.0, 360.0, 12, endpoint=False)
        self.assertNotIn(45.0, ring)

        def worst(azimuths, tilt):
            """Return the furthest reach over a set of tilt directions.

            Args:
                azimuths: Directions to probe.
                tilt: Tilt to apply.

            Returns:
                The largest radius reached inside the barrel.
            """
            return max(
                max_radius_in_barrel(
                    probe,
                    lead.pivot,
                    tilt_rotation(tilt, azimuth),
                    lead.hole_center,
                    z_low,
                    z_high,
                )
                for azimuth in azimuths
            )

        limit = barrel_tilt_limit(lead, 15.0)
        self.assertLessEqual(worst(probe_azimuths(lead), limit), allowed + 1e-9)
        # The same tilt measured only on the ring looks safe, which is the trap.
        self.assertLess(worst(ring, limit), worst(probe_azimuths(lead), limit))

    def test_corner_directions_are_probed(self) -> None:
        """A square lead's diagonals must be among the probed azimuths."""
        azimuths = probe_azimuths(square_lead())
        for corner in (45.0, 135.0, 225.0, 315.0):
            with self.subTest(corner=corner):
                self.assertTrue(bool(np.isclose(azimuths, corner).any()))

    def test_the_limit_holds_in_every_direction(self) -> None:
        """A square lead's clearance is azimuth dependent, so all must clear."""
        lead = square_lead(half_width=0.25, hole_radius=0.42)
        limit = barrel_tilt_limit(lead, 15.0)
        z_low, z_high = lead.barrel_z
        probe = densify_prism(lead.points, z_low, z_high)
        for azimuth in list(range(0, 360, 15)) + [45, 135, 225, 315]:
            with self.subTest(azimuth=azimuth):
                self.assertLessEqual(
                    max_radius_in_barrel(
                        probe,
                        lead.pivot,
                        tilt_rotation(limit, azimuth),
                        lead.hole_center,
                        z_low,
                        z_high,
                    ),
                    lead.hole_radius_mm - WALL_GAP_MM + 1e-9,
                )


class ApplyVariationTest(unittest.TestCase):
    """Delivered versus requested, per axis."""

    def test_tilt_is_clamped_and_the_barrel_is_named(self) -> None:
        """An unreachable request is a reported pair, not a substitution."""
        lead = square_lead(half_width=0.25, hole_radius=0.42)
        result = apply_pin_variation(
            lead, PinVariation(tilt_deg=15.0, azimuth_deg=40.0), tilt_cap_deg=15.0,
        )
        self.assertEqual(result.tilt_requested_deg, 15.0)
        self.assertLess(result.tilt_deg, 15.0)
        self.assertEqual(result.tilt_limited_by, "barrel")

    def test_a_reachable_tilt_is_delivered_untouched(self) -> None:
        """What fits must pass through exactly, and say so."""
        lead = square_lead(half_width=0.20, hole_radius=1.60)
        result = apply_pin_variation(
            lead, PinVariation(tilt_deg=6.0, azimuth_deg=10.0), tilt_cap_deg=15.0,
        )
        self.assertAlmostEqual(result.tilt_deg, 6.0)
        self.assertEqual(result.tilt_limited_by, "request")

    def test_offset_spends_only_what_the_tilt_left(self) -> None:
        """The two axes share one budget and must not both spend it.

        Requesting the same offset with and without a tilt has to yield a
        smaller delivered offset in the tilted case.
        """
        lead = square_lead(half_width=0.20, hole_radius=0.60)
        upright = apply_pin_variation(
            lead, PinVariation(offset_mm=0.30), tilt_cap_deg=15.0,
        )
        leaning = apply_pin_variation(
            lead,
            PinVariation(tilt_deg=8.0, azimuth_deg=0.0, offset_mm=0.30),
            tilt_cap_deg=15.0,
        )
        self.assertLess(leaning.offset_mm, upright.offset_mm)
        self.assertEqual(leaning.offset_limited_by, "clearance")

    def test_a_varied_lead_stays_inside_its_barrel(self) -> None:
        """Whatever comes back must clear the plating it was solved against."""
        lead = square_lead(half_width=0.25, hole_radius=0.50)
        for azimuth in (0.0, 45.0, 137.0, 300.0):
            with self.subTest(azimuth=azimuth):
                result = apply_pin_variation(
                    lead,
                    PinVariation(
                        tilt_deg=15.0,
                        azimuth_deg=azimuth,
                        offset_mm=0.40,
                        offset_azimuth_deg=azimuth + 90.0,
                        protrusion_mm=0.8,
                    ),
                    tilt_cap_deg=15.0,
                )
                self.assertLessEqual(
                    wall_overrun_mm(lead, result.points, result.probe_points),
                    1e-9,
                )

    def test_the_trim_actually_moves_the_end_cap(self) -> None:
        """Selecting the cap by height silently no-ops on a tilted lead.

        A cap selected by height would leave leads at their full protrusion
        while the record claims the varied value, so the trim is asserted to
        bite.
        """
        lead = square_lead(protrusion=1.95)
        result = apply_pin_variation(
            lead,
            PinVariation(tilt_deg=6.0, azimuth_deg=30.0, protrusion_mm=0.40),
            tilt_cap_deg=15.0,
        )
        self.assertAlmostEqual(
            float(result.points[:, 2].min()),
            lead.board_far_face_z - 0.40,
            places=6,
        )

    def test_the_deepest_point_carries_the_requested_protrusion(self) -> None:
        """Anchoring the face's centroid reports a depth no point has."""
        lead = square_lead()
        result = apply_pin_variation(
            lead,
            PinVariation(protrusion_mm=0.9, cut_deg=20.0, cut_azimuth_deg=70.0),
            tilt_cap_deg=15.0,
        )
        self.assertAlmostEqual(
            float(result.points[:, 2].min()),
            lead.board_far_face_z - 0.9,
            places=6,
        )

    def test_a_steep_cut_is_limited_by_a_shallow_protrusion(self) -> None:
        """A tilted face spans a height a short protrusion cannot hold."""
        lead = square_lead()
        result = apply_pin_variation(
            lead,
            PinVariation(protrusion_mm=0.05, cut_deg=60.0),
            tilt_cap_deg=15.0,
        )
        self.assertLess(result.cut_deg, 60.0)
        self.assertEqual(result.cut_limited_by, "protrusion")

    def test_a_deep_protrusion_keeps_the_requested_cut(self) -> None:
        """With room for the face, the request is what binds."""
        lead = square_lead(half_width=0.10, protrusion=3.0)
        result = apply_pin_variation(
            lead,
            PinVariation(protrusion_mm=1.5, cut_deg=30.0),
            tilt_cap_deg=15.0,
        )
        self.assertAlmostEqual(result.cut_deg, 30.0, places=6)
        self.assertEqual(result.cut_limited_by, "request")

    def test_an_angled_cut_is_not_flat(self) -> None:
        """A cut plane has to actually tip the face, not just move it."""
        lead = square_lead(half_width=0.10, protrusion=3.0)
        flat = apply_pin_variation(
            lead, PinVariation(protrusion_mm=1.5), tilt_cap_deg=15.0,
        )
        angled = apply_pin_variation(
            lead,
            PinVariation(protrusion_mm=1.5, cut_deg=30.0),
            tilt_cap_deg=15.0,
        )
        self.assertLess(float(np.ptp(flat.points[:4, 2])), 1e-6)
        self.assertGreater(float(np.ptp(angled.points[:4, 2])), 0.05)

    def test_variation_is_deterministic(self) -> None:
        """The module draws nothing; identical requests give identical points."""
        lead = square_lead()
        request = PinVariation(
            tilt_deg=9.0, azimuth_deg=33.0, offset_mm=0.1, protrusion_mm=0.7,
        )
        first = apply_pin_variation(lead, request, tilt_cap_deg=15.0)
        second = apply_pin_variation(lead, request, tilt_cap_deg=15.0)
        np.testing.assert_array_equal(first.points, second.points)


class WallOverrunTest(unittest.TestCase):
    """The gate, and what it has to reject."""

    def test_an_untouched_lead_is_inside_its_bore(self) -> None:
        """A nominal pin reports a negative overrun."""
        lead = square_lead(half_width=0.25, hole_radius=0.50)
        z_low, z_high = lead.barrel_z
        probe = densify_prism(lead.points, z_low, z_high)
        self.assertLess(wall_overrun_mm(lead, probe), 0.0)

    def test_a_lead_driven_through_the_plating_is_caught(self) -> None:
        """Moving a lead past its wall must report positive.

        The gate takes points rather than a request precisely so that a pose
        the planner never sanctioned is still measurable.
        """
        lead = square_lead(half_width=0.25, hole_radius=0.50)
        z_low, z_high = lead.barrel_z
        shoved = densify_prism(lead.points, z_low, z_high)
        shoved[:, 0] += 0.60
        self.assertGreater(wall_overrun_mm(lead, shoved), 0.0)


if __name__ == "__main__":
    unittest.main()
