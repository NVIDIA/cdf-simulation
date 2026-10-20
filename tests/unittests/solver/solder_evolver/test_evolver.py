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

"""Physics gate for the Surface Evolver backend: equilibrium recovers the angle.

The sessile drop PRESCRIBES its contact angle; solving it and measuring the
angle back is the self-consistency test for the whole backend. Requires a
Surface Evolver binary (the vendored build, ``/usr/bin/evolver``, or
``$SOLDER_EVOLVER_BIN``); skipped, not failed, when none is present.
"""

import os
import sys
import unittest

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    import numpy as np
    from pcb_aoi.solder_evolver import backend as ev
    from pcb_aoi.solder_evolver.verify import (
        fit_sphere, sessile_contact_angle_deg,
    )
except (Exception, SystemExit) as _exc:                                 # noqa: BLE001
    raise unittest.SkipTest(f"solder_evolver deps unavailable: {_exc}")

_HAVE_EVOLVER = ev.evolver_available()


class TestSphereFit(unittest.TestCase):
    def test_recovers_known_sphere(self):
        rng = np.random.default_rng(0)
        p = rng.normal(size=(500, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        pts = np.array([2.0, -1.0, 3.0]) + 1.7 * p
        c, r = fit_sphere(pts)
        self.assertTrue(np.allclose(c, [2.0, -1.0, 3.0], atol=1e-6))
        self.assertLess(abs(r - 1.7), 1e-6)


@unittest.skipUnless(_HAVE_EVOLVER, "no Surface Evolver binary")
class TestSessilePhysics(unittest.TestCase):
    def test_sessile_recovers_contact_angle(self):
        for prescribed in (30.0, 55.0, 90.0, 120.0):
            mesh = ev.solve_sessile(contact_angle=prescribed, volume=1.0,
                                    nrefine=3)
            got = sessile_contact_angle_deg(mesh.arrays()[0])
            self.assertLess(abs(got - prescribed), 1.5,
                            f"prescribed {prescribed} -> measured {got}")

    def test_free_surface_isolation(self):
        mesh = ev.solve_sessile(contact_angle=60.0, volume=1.0, nrefine=2)
        self.assertGreater(mesh.n_triangles, 0)
        self.assertGreater(mesh.free_surface().n_triangles, 0)


if __name__ == "__main__":
    unittest.main()
