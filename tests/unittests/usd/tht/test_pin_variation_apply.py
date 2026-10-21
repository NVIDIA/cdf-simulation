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

"""Test group selection, sampling and the write-back of varied leads."""

from __future__ import annotations

import random
import unittest

import numpy as np
try:
    from pxr import UsdGeom
except ImportError:  # the CI unit-test image installs numpy only
    UsdGeom = None

from pcb_aoi.tht.lead_discovery import discover_through_hole_leads
from pcb_aoi.tht.pin_variation_apply import (
    plan_board_variation,
    plan_lead_variation,
    select_leads,
    summarise,
    write_lead_points,
)
from tests.unittests.usd.tht.test_lead_discovery import board

_FAR_FACE_Z = -1.65


def leads_of(stage):
    """Return the paired leads of a stage.

    Args:
        stage: Stage to discover.

    Returns:
        The paired leads.
    """
    paired, _rejected = discover_through_hole_leads(stage)
    return paired


@unittest.skipUnless(UsdGeom is not None, "usd-core is not installed")
class SelectionTest(unittest.TestCase):
    """Which group claims which lead."""

    def test_a_package_type_group_claims_the_lead(self) -> None:
        """Selecting by package type must reach a lead that declares one."""
        leads = leads_of(board())
        claimed = select_leads(leads, [{"package_types": ["TO92"]}])
        self.assertEqual(set(claimed.values()), {0})

    def test_a_refdes_group_claims_the_lead(self) -> None:
        """Selecting by refdes is the path a panelised THT board depends on."""
        leads = leads_of(board())
        claimed = select_leads(leads, [{"components": ["U1"]}])
        self.assertEqual(set(claimed.values()), {0})

    def test_an_unmatched_name_claims_nothing(self) -> None:
        """Naming a part the board does not have selects nothing, quietly."""
        leads = leads_of(board())
        self.assertEqual(select_leads(leads, [{"components": ["Q99"]}]), {})

    def test_the_last_group_naming_a_lead_wins(self) -> None:
        """A narrow group written after a sweep has to override it.

        Resolving it first-wins would make the order of the calls meaningless.
        """
        leads = leads_of(board())
        claimed = select_leads(
            leads,
            [{"package_types": ["TO92"]}, {"components": ["U1"]}],
        )
        self.assertEqual(set(claimed.values()), {1})

    def test_a_package_group_does_not_match_a_refdes(self) -> None:
        """The two selectors read different fields and must not cross over."""
        leads = leads_of(board())
        self.assertEqual(select_leads(leads, [{"package_types": ["U1"]}]), {})


@unittest.skipUnless(UsdGeom is not None, "usd-core is not installed")
class SamplingTest(unittest.TestCase):
    """What is drawn, and what is left alone."""

    def test_an_omitted_axis_is_not_varied(self) -> None:
        """A group that asks for tilt only must not move anything else.

        Protrusion and cut act on the lead's END, so an omitted axis is None
        rather than 0.0 -- a protrusion of zero is a real request for a lead
        cut flush with the board.
        """
        request = plan_lead_variation(random.Random(0), {"tilt_range": [0, 15]})
        self.assertEqual(request.offset_mm, 0.0)
        self.assertIsNone(request.protrusion_mm)
        self.assertIsNone(request.cut_deg)

    def test_a_tilt_only_group_leaves_the_lead_its_length(self) -> None:
        """The effect, not just the request.

        Asserting only the request object is what let an omitted
        `height_variation` map to protrusion 0.0 and trim every lead flush with
        the board while the tests stayed green. This measures the tip.
        """
        leads = leads_of(board())
        nominal = float(leads[0].points[:, 2].min())
        edits, outcomes, _skipped = plan_board_variation(
            leads, [{"components": ["U1"], "tilt_range": [0.0, 0.0]}],
            board_far_face_z=_FAR_FACE_Z, seed=0,
        )
        self.assertAlmostEqual(
            float(edits[0][1].points[:, 2].min()), nominal, places=5,
        )
        self.assertFalse(outcomes[0].trimmed)
        self.assertIsNone(outcomes[0].protrusion_mm)

    def test_an_explicit_zero_protrusion_does_trim(self) -> None:
        """None and 0.0 are different requests, and both must be honoured.

        A lead cut flush with the board is a real defect, so asking for it must
        not be confused with not asking at all.
        """
        leads = leads_of(board())
        edits, outcomes, _skipped = plan_board_variation(
            leads,
            [{"components": ["U1"], "height_variation": [0.0, 0.0]}],
            board_far_face_z=_FAR_FACE_Z, seed=0,
        )
        self.assertTrue(outcomes[0].trimmed)
        self.assertAlmostEqual(
            float(edits[0][1].points[:, 2].min()), _FAR_FACE_Z, places=5,
        )

    def test_a_cut_angle_alone_refaces_the_lead_where_it_ends(self) -> None:
        """A cut with no length request changes the angle, not the reach."""
        leads = leads_of(board())
        nominal = float(leads[0].points[:, 2].min())
        edits, outcomes, _skipped = plan_board_variation(
            leads, [{"components": ["U1"], "cut_angle": [20.0, 20.0]}],
            board_far_face_z=_FAR_FACE_Z, seed=0,
        )
        self.assertTrue(outcomes[0].trimmed)
        self.assertAlmostEqual(
            float(edits[0][1].points[:, 2].min()), nominal, places=4,
        )
        # the face is now angled rather than flat
        self.assertGreater(float(np.ptp(edits[0][1].points[:4, 2])), 0.01)

    def test_a_drawn_axis_lands_inside_its_band(self) -> None:
        """Bands are bounds, not hints."""
        for seed in range(20):
            with self.subTest(seed=seed):
                request = plan_lead_variation(
                    random.Random(seed),
                    {"tilt_range": [2.0, 6.0], "cut_angle": [10.0, 20.0]},
                )
                self.assertGreaterEqual(request.tilt_deg, 2.0)
                self.assertLessEqual(request.tilt_deg, 6.0)
                self.assertGreaterEqual(request.cut_deg, 10.0)
                self.assertLessEqual(request.cut_deg, 20.0)

    def test_azimuth_covers_the_circle(self) -> None:
        """A band says how far to lean, never which way.

        Leaning every pin the same way would be a rig, not a variation.
        """
        azimuths = [
            plan_lead_variation(random.Random(seed), {"tilt_range": [0, 15]}).azimuth_deg
            for seed in range(60)
        ]
        self.assertLess(min(azimuths), 90.0)
        self.assertGreater(max(azimuths), 270.0)


@unittest.skipUnless(UsdGeom is not None, "usd-core is not installed")
class PlanBoardTest(unittest.TestCase):
    """The whole board, without a write."""

    def test_only_claimed_leads_are_varied(self) -> None:
        """An unmatched board produces no edits and no records."""
        leads = leads_of(board())
        edits, outcomes, _skipped = plan_board_variation(
            leads, [{"components": ["Q99"], "tilt_range": [0, 15]}],
            board_far_face_z=_FAR_FACE_Z, seed=0,
        )
        self.assertEqual((edits, outcomes), ([], []))

    def test_a_varied_lead_stays_inside_its_own_plating(self) -> None:
        """The point of the whole exercise, asserted on the output."""
        leads = leads_of(board())
        _edits, outcomes, _skipped = plan_board_variation(
            leads,
            [{
                "package_types": ["TO92"],
                "tilt_range": [0, 15],
                "offset_range": [0.0, 0.3],
                "height_variation": [0.0, 1.5],
                "cut_angle": [0, 60],
            }],
            board_far_face_z=_FAR_FACE_Z,
            seed=7,
        )
        self.assertEqual(len(outcomes), 1)
        self.assertLessEqual(outcomes[0].wall_overrun_mm, 1e-9)

    def test_a_lead_outside_its_bore_at_rest_is_skipped_and_named(self) -> None:
        """"Stay inside the plating" cannot bind geometry that already is not.

        Connectors on a panelised THT board can author lead meshes wider than
        their own stated bores, and failing every lead because of a few is
        the wrong response to an asset defect in a couple of parts. The measurement is
        taken AT REST, so this can never excuse a breach the solver caused.
        """
        leads = leads_of(board(lead_half=0.25, hole_radius=0.50))
        wide = leads_of(board(lead_half=0.60, hole_radius=0.50))
        edits, outcomes, skipped = plan_board_variation(
            wide, [{"components": ["U1"], "tilt_range": [0, 10]}],
            board_far_face_z=_FAR_FACE_Z, seed=0,
        )
        self.assertEqual((edits, outcomes), ([], []))
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["reason"], "outside its own bore at rest")
        self.assertGreater(skipped[0]["rest_overrun_mm"], 0.0)

        # a lead that does fit is still varied
        edits, outcomes, skipped = plan_board_variation(
            leads, [{"components": ["U1"], "tilt_range": [0, 10]}],
            board_far_face_z=_FAR_FACE_Z, seed=0,
        )
        self.assertEqual(len(edits), 1)
        self.assertEqual(skipped, [])

    def test_the_summary_counts_the_skips(self) -> None:
        """A skip that is not reported is a silent substitution."""
        wide = leads_of(board(lead_half=0.60, hole_radius=0.50))
        _edits, outcomes, skipped = plan_board_variation(
            wide, [{"components": ["U1"], "tilt_range": [0, 10]}],
            board_far_face_z=_FAR_FACE_Z, seed=0,
        )
        report = summarise(outcomes, skipped)
        self.assertEqual(report["skipped"], 1)
        self.assertEqual(len(report["skipped_leads"]), 1)

    def test_the_same_seed_gives_the_same_board(self) -> None:
        """A run has to be reproducible from its seed alone."""
        group = [{"package_types": ["TO92"], "tilt_range": [0, 15]}]
        first, _o, _s = plan_board_variation(
            leads_of(board()), group, board_far_face_z=_FAR_FACE_Z, seed=3,
        )
        second, _o, _s = plan_board_variation(
            leads_of(board()), group, board_far_face_z=_FAR_FACE_Z, seed=3,
        )
        np.testing.assert_array_equal(first[0][1].points, second[0][1].points)

    def test_a_different_seed_gives_a_different_board(self) -> None:
        """Otherwise the seed is decoration."""
        group = [{"package_types": ["TO92"], "tilt_range": [5, 15]}]
        first, _o, _s = plan_board_variation(
            leads_of(board()), group, board_far_face_z=_FAR_FACE_Z, seed=1,
        )
        second, _o, _s = plan_board_variation(
            leads_of(board()), group, board_far_face_z=_FAR_FACE_Z, seed=2,
        )
        self.assertFalse(
            np.array_equal(first[0][1].points, second[0][1].points),
        )

    def test_a_lead_draws_from_its_own_path_not_a_shared_stream(self) -> None:
        """Adding a group must not reshuffle parts that were already varied.

        Seeding one stream for the whole board would make every lead's
        variation depend on how many leads came before it.
        """
        leads = leads_of(board())
        alone, _o1, _s1 = plan_board_variation(
            leads, [{"components": ["U1"], "tilt_range": [0, 15]}],
            board_far_face_z=_FAR_FACE_Z, seed=5,
        )
        with_other, _o2, _s2 = plan_board_variation(
            leads,
            [
                {"components": ["Q99"], "tilt_range": [0, 15]},
                {"components": ["U1"], "tilt_range": [0, 15]},
            ],
            board_far_face_z=_FAR_FACE_Z,
            seed=5,
        )
        np.testing.assert_array_equal(
            alone[0][1].points, with_other[0][1].points,
        )


@unittest.skipUnless(UsdGeom is not None, "usd-core is not installed")
class WriteBackTest(unittest.TestCase):
    """Putting the geometry on the stage."""

    def test_points_are_authored_in_local_space(self) -> None:
        """The solve is in world space; the prim carries its own transform.

        Writing world points onto a transformed prim applies that transform
        twice, which is invisible on an identity board and wrong on a real one.
        """
        stage = board()
        lead_prim = stage.GetPrimAtPath("/Board/U1/lead")
        UsdGeom.Xformable(
            stage.GetPrimAtPath("/Board/U1"),
        ).AddTranslateOp().Set((25.0, 0.0, 0.0))

        leads = leads_of(stage)
        self.assertEqual(len(leads), 0)  # the lead moved off its barrel

        # Move the barrel with it and re-discover.
        UsdGeom.Xformable(
            stage.GetPrimAtPath("/Board/hole"),
        ).AddTranslateOp().Set((25.0, 0.0, 0.0))
        leads = leads_of(stage)
        self.assertEqual(len(leads), 1)

        edits, _o, _s = plan_board_variation(
            leads, [{"components": ["U1"], "tilt_range": [0, 10]}],
            board_far_face_z=_FAR_FACE_Z, seed=0,
        )
        self.assertEqual(write_lead_points(stage, edits), 1)

        authored = np.array(
            [list(p) for p in UsdGeom.Mesh(lead_prim).GetPointsAttr().Get()],
            float,
        )
        # Local X stays near zero; a double-applied translate would put it at 50.
        self.assertLess(abs(float(authored[:, 0].mean())), 1.0)

    def test_a_missing_prim_is_an_error_not_a_skip(self) -> None:
        """A lead that vanished between discovery and write is a real fault."""
        stage = board()
        leads = leads_of(stage)
        edits, _o, _s = plan_board_variation(
            leads, [{"components": ["U1"], "tilt_range": [0, 10]}],
            board_far_face_z=_FAR_FACE_Z, seed=0,
        )
        stage.RemovePrim("/Board/U1/lead")
        with self.assertRaises(ValueError):
            write_lead_points(stage, edits)


@unittest.skipUnless(UsdGeom is not None, "usd-core is not installed")
class SummaryTest(unittest.TestCase):
    """The record the order hands back."""

    def test_an_empty_board_summarises_without_raising(self) -> None:
        """Zero leads is a number, not an exception."""
        self.assertEqual(summarise([])["leads"], 0)

    def test_the_summary_is_json_compatible(self) -> None:
        """Order output crosses a transport boundary."""
        import json

        leads = leads_of(board())
        _edits, outcomes, _skipped = plan_board_variation(
            leads, [{"package_types": ["TO92"], "tilt_range": [0, 15]}],
            board_far_face_z=_FAR_FACE_Z, seed=0,
        )
        report = summarise(outcomes)
        json.dumps(report)
        self.assertEqual(report["leads"], 1)
        self.assertEqual(report["breaches"], 0)
        self.assertIn("tilt_limited_by", report["per_lead"][0])


@unittest.skipUnless(UsdGeom is not None, "usd-core is not installed")
class SummaryCompletenessTest(unittest.TestCase):
    """Every axis reports what was ASKED as well as what was delivered.

    The report is the only way a caller learns a band was not honoured, and
    it is the same report on both surfaces. With the delivered value alone,
    `cut_limited_by: "request"` reads identically whether the ask was 13
    degrees or 30 — tilt carried both, offset and cut did not.
    """

    def _summary(self):
        leads = leads_of(board())
        groups = [{"package_types": ["TO92"], "tilt_range": (2, 6),
                   "offset_range": (0.02, 0.06), "cut_angle": (5, 12)}]
        edits, outcomes, skipped = plan_board_variation(
            leads, groups, board_far_face_z=_FAR_FACE_Z, seed=1)
        del edits
        return summarise(outcomes, skipped)

    def test_every_axis_reports_requested_and_delivered(self):
        record = self._summary()["per_lead"][0]
        for axis in ("tilt_requested_deg", "tilt_deg",
                     "offset_requested_mm", "offset_mm",
                     "cut_requested_deg", "cut_deg"):
            with self.subTest(axis=axis):
                self.assertIn(axis, record)

    def test_the_requested_value_is_inside_the_declared_band(self):
        record = self._summary()["per_lead"][0]
        self.assertGreaterEqual(record["cut_requested_deg"], 5.0)
        self.assertLessEqual(record["cut_requested_deg"], 12.0)
        self.assertGreaterEqual(record["offset_requested_mm"], 0.02)
        self.assertLessEqual(record["offset_requested_mm"], 0.06)


if __name__ == "__main__":
    unittest.main()
