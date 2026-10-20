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

"""Observable-parameter defects: records configure strategies, and the
post-solve checks verify what was asked for (then discard the numbers).

The parameter vocabulary is the solder-defects proposal's: every user-facing
value is measurable on the produced geometry (volume_ratio, gap_fraction),
strategies own the observable -> internal translation, and generation warns
when the output deviates from what the label claims.
"""

import os
import sys
import unittest

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    import numpy as np
    from pcb_aoi.solder_evolver import defects
    from pcb_aoi.solder_evolver.defects.base import sample_scalar
    from pcb_aoi.solder_generation import surface_evolver_solver as ses
except (Exception, SystemExit):  # noqa: BLE001
    raise unittest.SkipTest("solder_evolver defects not importable here")


class TestConfigure(unittest.TestCase):
    def test_volume_ratio_reaches_target_volume(self):
        for name, ratio in (("insufficient", 0.4), ("excess", 2.0)):
            base = defects.resolve(name)
            s = base.configure({"volume_ratio": ratio})
            self.assertIsNot(s, base)       # registry singleton never mutated
            self.assertEqual(s.target_volume(1.0, 0), ratio)
            # ...and the singleton keeps its built-in default.
            self.assertNotEqual(base.target_volume(1.0, 0), ratio)

    def test_record_without_observables_keeps_defaults(self):
        for name in ("insufficient", "excess", "nominal", "missing_solder"):
            base = defects.resolve(name)
            self.assertIs(base.configure({}), base)

    def test_observable_keys_cover_every_strategy_parameter(self):
        # _OBSERVABLE_KEYS is the ONLY slice of a defect record that reaches
        # strategy.configure in JobConfig runs; a key missing here is
        # silently dropped in production while direct solve_footprint
        # callers (the lab, tests) still see it. lead_climb/fullness/
        # pad_exposure were shipped without it once already.
        from pcb_aoi.solder_generation.surface_evolver_solver import (
            _OBSERVABLE_KEYS)
        needed = {"volume_ratio", "gap_fraction", "seed",
                  "lead_climb", "fullness", "pad_exposure"}
        self.assertTrue(needed.issubset(set(_OBSERVABLE_KEYS)),
                        needed - set(_OBSERVABLE_KEYS))

    def test_dry_contact_is_tht_only(self):
        s = defects.resolve("dry_contact")
        self.assertEqual(s.kinds, ("tht",))
        self.assertTrue(s.expect_genus0)
        # The volume strategies stay both-kind.
        self.assertIn("smt", defects.resolve("insufficient").kinds)


class TestSampleScalar(unittest.TestCase):
    def test_scalar_passes_through(self):
        self.assertEqual(sample_scalar(0.4, np.random.default_rng(0)), 0.4)

    def test_range_samples_inside(self):
        rng = np.random.default_rng(7)
        got = {sample_scalar([0.2, 0.6], rng) for _ in range(64)}
        self.assertTrue(all(0.2 <= v <= 0.6 for v in got))
        self.assertGreater(len(got), 1)     # it is actually sampling


class TestInterruptedTranslation(unittest.TestCase):
    def test_front_matches_circumference_math(self):
        # gap = arccos(front / r) / pi  <=>  front = r * cos(pi * gap),
        # measured against interrupt_region (see the module docstring).
        s = defects.resolve("dry_contact").configure(
            {"gap_fraction": 0.35, "seed": 11})
        kw = s.tht_cut_kwargs(0, hole_r=0.5)
        self.assertTrue(kw["interrupt"])
        self.assertAlmostEqual(kw["interrupt_front"],
                               0.5 * np.cos(np.pi * 0.35), places=9)
        self.assertTrue(0.0 <= kw["interrupt_angle"] < 2.0 * np.pi)

    def test_seeded_and_per_joint_independent(self):
        s = defects.resolve("dry_contact").configure(
            {"gap_fraction": [0.25, 0.5], "seed": 3})
        again = defects.resolve("dry_contact").configure(
            {"gap_fraction": [0.25, 0.5], "seed": 3})
        self.assertEqual(s.tht_cut_kwargs(0, 0.5), again.tht_cut_kwargs(0, 0.5))
        # Twin joints break at independent azimuths.
        self.assertNotEqual(s.tht_cut_kwargs(0, 0.5)["interrupt_angle"],
                            s.tht_cut_kwargs(1, 0.5)["interrupt_angle"])


class TestObservablesSlice(unittest.TestCase):
    def test_pose_bookkeeping_never_enters_identity(self):
        record = {"volume_ratio": 0.4, "seed": 9, "defect": "insufficient",
                  "path": "/x", "original_flat": [0.0] * 16,
                  "tombstone_final_flat": [0.0] * 16}
        self.assertEqual(ses._observables(record),
                         {"volume_ratio": 0.4, "seed": 9})
        self.assertEqual(ses._observables(None), {})


class TestVerifyVolumes(unittest.TestCase):
    #: A unit cube: 8 corners, 12 triangles, volume exactly 1.
    _P = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
          (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    _T = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
          0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6,
          1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7]

    def test_mesh_volume_unit_cube(self):
        self.assertAlmostEqual(ses._mesh_volume(self._P, self._T), 1.0)

    def _run(self, target):
        s = defects.resolve("insufficient").configure({"volume_ratio": 0.5})
        report = []
        ses._verify_volumes(
            s, {"k": (self._P, None, self._T)}, {0: (None, "k", target, None)},
            {"k": [0]}, ["k"], report)
        return report

    def test_within_tolerance_is_silent(self):
        self.assertEqual(self._run(target=0.95), [])

    def test_deviation_warns_once_then_discards(self):
        report = self._run(target=0.5)      # cube is 2x the requested volume
        self.assertEqual(len(report), 1)
        self.assertIn("deviates", report[0])
        self.assertIn("insufficient", report[0])

    def test_nominal_never_checks(self):
        report = []
        ses._verify_volumes(defects.resolve("nominal"),
                            {"k": (self._P, None, self._T)},
                            {0: (None, "k", 0.5, None)}, {"k": [0]}, ["k"],
                            report)
        self.assertEqual(report, [])


if __name__ == "__main__":
    unittest.main()
