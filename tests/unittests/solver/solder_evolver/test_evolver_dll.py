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

"""Gate for the dynamic-load (``-DENABLE_DLL``) Surface Evolver build.

The grid-SDF keep-out constraint depends on Evolver being able to ``load_library``
a compiled ``.so`` and use it as a ``constraint ... formula:``. The Debian/Ubuntu
package is NOT built that way, so this capability has to be built and gated
explicitly (``pcb_aoi/solder_evolver/third_party/evolver/build.sh``).

If no DLL-capable binary is present these checks are skipped, not failed --
matching the convention in test_evolver.py.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import unittest

try:
    from pcb_aoi.solder_evolver import backend as ev
except (Exception, SystemExit) as _exc:                                # noqa: BLE001
    raise unittest.SkipTest(f"solder_evolver deps unavailable: {_exc}")

# A smooth sphere SDF: f = |x - c| - r. Signs match geometry.sdf's convention
# (>0 outside, =0 on the surface) and |grad f| = 1.
_SPHERE_C = """
#include <math.h>
#define FUNC_VALUE 1
#define FUNC_DERIV 2
#define MAXCOORD 6      /* must match Evolver's model.h */
#define REAL double     /* must match Evolver's include.h */
struct dstack { REAL value; REAL deriv[2*MAXCOORD];
                REAL second[2*MAXCOORD][2*MAXCOORD]; };

void sphere_sdf(int mode, REAL *x, struct dstack *s)
{
  REAL dx = x[0] - 0.5, dy = x[1] - 0.5, dz = x[2];
  REAL d = sqrt(dx*dx + dy*dy + dz*dz);
  if (d < 1e-12) d = 1e-12;
  s->value = d - 0.30;
  if (mode == FUNC_VALUE) return;
  s->deriv[0] = dx/d; s->deriv[1] = dy/d; s->deriv[2] = dz/d;
}
"""


def _have_dll():
    return ev.evolver_available() and ev.evolver_has_dll()


def _run(fe_text, wd):
    """Compile the sphere .so into `wd`, run `fe_text` there, return stdout."""
    (wd / "sphere.c").write_text(_SPHERE_C)
    subprocess.run(["gcc", "-O1", "-fPIC", "-shared", str(wd / "sphere.c"),
                    "-o", str(wd / "sphere.so"), "-lm"], check=True)
    (wd / "t.fe").write_text(fe_text)
    p = subprocess.run([ev.find_evolver(), "t.fe"], cwd=wd, input="",
                       capture_output=True, text=True, timeout=120)
    return p.stdout + p.stderr


def _radii(out):
    return [float(l.split()[1]) for l in out.splitlines() if l.startswith("R ")]


def test_apt_binary_is_not_dll_capable():
    """Documents WHY we build our own: the packaged binary cannot do this.

    Skipped if the system package isn't installed.
    """
    if not Path("/usr/bin/evolver").exists():
        print("skip: no /usr/bin/evolver")
        return
    assert not ev.evolver_has_dll("/usr/bin/evolver"), (
        "system evolver unexpectedly has ENABLE_DLL -- if the distro started "
        "shipping it, pcb_aoi/solder_evolver/third_party/evolver/build.sh may no longer be needed"
    )


def test_dll_equality_constraint_projects_onto_level_set():
    """A DLL function used as an equality constraint pins vertices to sdf=0.

    Exercises BOTH the value and the gradient returned by the .so: projection
    is a Newton step along grad f, so a wrong gradient lands off the sphere.
    """
    if not _have_dll():
        print("skip: no DLL-capable evolver (run pcb_aoi/solder_evolver/third_party/evolver/build.sh)")
        return
    fe = """
load_library "sphere.so"
constraint 1
formula: sphere_sdf
vertices
1  1.0  0.5  0.0  constraint 1
2  0.5  1.0  0.0  constraint 1
3  0.5  0.5  0.9  constraint 1
edges
faces
read
foreach vertex vv do printf "R %.6f\\n", sqrt((vv.x-0.5)^2+(vv.y-0.5)^2+vv.z^2);
quit 0
"""
    with tempfile.TemporaryDirectory() as d:
        r = _radii(_run(fe, Path(d)))
    assert len(r) == 3, f"expected 3 vertices, got {r}"
    for v in r:
        assert abs(v - 0.30) < 1e-6, f"vertex not on sdf=0: |v-c|={v}"


def test_dll_onesided_constraint_repels():
    """The load-bearing claim: a SMOOTH one-sided constraint really is enforced.

    docs/solver_paradigms.md records keep-outs coming out inert with the kinky
    `min-of-max` hull formula. This pins down that the inertness was about the
    FORMULA, not about one-sided constraints per se: vertices seeded strictly
    inside a smooth obstacle get pushed out to its surface, while a vertex
    already outside is left alone.
    """
    if not _have_dll():
        print("skip: no DLL-capable evolver (run pcb_aoi/solder_evolver/third_party/evolver/build.sh)")
        return
    fe = """
load_library "sphere.so"
constraint 1 nonnegative
formula: sphere_sdf
vertices
1  0.55 0.50  0.00  constraint 1
2  0.50 0.55  0.00  constraint 1
3  0.50 0.50  0.10  constraint 1
4  0.90 0.50  0.00  constraint 1
edges
faces
read
foreach vertex vv do printf "R %.6f\\n", sqrt((vv.x-0.5)^2+(vv.y-0.5)^2+vv.z^2);
quit 0
"""
    with tempfile.TemporaryDirectory() as d:
        r = _radii(_run(fe, Path(d)))
    assert len(r) == 4, f"expected 4 vertices, got {r}"
    for v in r[:3]:                      # seeded inside -> pushed out to sdf=0
        assert abs(v - 0.30) < 1e-6, f"violating vertex not expelled: |v-c|={v}"
    assert abs(r[3] - 0.40) < 1e-9, (    # already outside -> untouched
        f"non-violating vertex was moved: |v-c|={r[3]}"
    )


# unittest-discovery adapter: the checks above are plain functions (each guards
# itself when no DLL-capable binary is present); this exposes them to
# `python -m unittest discover` without rewriting them.


class TestEvolverDLL(unittest.TestCase):
    pass


for _name in [k for k, v in list(globals().items())
              if k.startswith("test_") and callable(v)]:
    setattr(TestEvolverDLL, _name, staticmethod(globals()[_name]))
