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

"""solver_frame's board-side orientation: decided by extent, not vertex mean.

Regression for the U1 (QFN-32) flip: a flat package's mesh is vertex-dense at
the leads/thermal pad and sparse above, so its vertex MEAN can sit below the
paste mean even though the body extends far above it. The mean heuristic
flipped the solver frame 180 degrees about x -- the footprint solved upside
down AND y-mirrored, invisibly for symmetric two-pad passives.

numpy-only on purpose, like test_mesh_normals: runs in the starved CI job.
"""

import unittest

import numpy as np

from pcb_aoi.solder_generation.surface_evolver_stage_read import solver_frame


def _paste_slab():
    """One thin deposit above the pad plane: z is the thin (normal) axis."""
    x, y = np.meshgrid(np.linspace(-0.3, 0.3, 5), np.linspace(-0.2, 0.2, 5))
    lo = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.035)])
    hi = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.162)])
    return [np.vstack([lo, hi])]


def _qfn_like():
    """Dense at lead level, sparse above: vertex mean BELOW the paste mean,
    body extent far above it -- the exact shape that fooled the mean test."""
    rng = np.random.default_rng(7)
    leads = np.column_stack(
        [
            rng.uniform(-2.5, 2.5, 4000),
            rng.uniform(-2.5, 2.5, 4000),
            rng.uniform(0.0, 0.05, 4000),
        ]
    )
    body_top = np.column_stack(
        [rng.uniform(-2.5, 2.5, 200), rng.uniform(-2.5, 2.5, 200), np.full(200, 0.85)]
    )
    return np.vstack([leads, body_top])


class SolverFrameTest(unittest.TestCase):
    def test_qfn_like_body_keeps_the_frame_upright(self):
        paste = _paste_slab()
        comp = _qfn_like()
        # the trap must actually be armed: mean below the paste mean...
        self.assertLess(comp[:, 2].mean(), np.vstack(paste)[:, 2].mean())
        # ...but the body reaches much farther above than below.
        R, _Rinv = solver_frame(paste, comp)
        R3 = np.asarray(R, float).reshape(4, 4)[:3, :3]
        self.assertGreater(R3[2, 2], 0.9, "frame flipped on a QFN-like part")
        # paste must land ABOVE the solver's board plane
        z = (np.vstack(paste) @ R3)[:, 2]
        self.assertGreater(float(z.min()), 0.0)

    def test_body_below_pads_still_flips(self):
        # A part whose body genuinely extends below the pad plane (the case
        # the flip exists for): up must invert so the solve sees the body up.
        paste = _paste_slab()
        comp = _qfn_like().copy()
        comp[:, 2] *= -1.0  # body hangs below
        R, _Rinv = solver_frame(paste, comp)
        R3 = np.asarray(R, float).reshape(4, 4)[:3, :3]
        self.assertLess(R3[2, 2], -0.9, "frame did not flip for a below-board body")

    def test_rotation_is_proper_and_invertible(self):
        paste = _paste_slab()
        for comp in (_qfn_like(), None):
            R, Rinv = solver_frame(paste, comp)
            R3 = np.asarray(R, float).reshape(4, 4)[:3, :3]
            self.assertAlmostEqual(float(np.linalg.det(R3)), 1.0, places=9)
            got = np.asarray(R, float).reshape(4, 4) @ np.asarray(Rinv, float).reshape(
                4, 4
            )
            np.testing.assert_allclose(got, np.eye(4), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
