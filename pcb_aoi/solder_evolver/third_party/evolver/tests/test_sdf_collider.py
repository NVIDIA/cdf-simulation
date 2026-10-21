#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the vendored Evolver binary can use our SDF collider constraint.

Run after third_party/evolver/build.sh:

    .venv/bin/python pcb_aoi/solder_evolver/third_party/evolver/tests/test_sdf_collider.py

This is a gate on the BINARY, not on Python: every check drives the real
evolver on a real datafile and reads geometry back out. If the collider is not
compiled in, or the grid format drifts, or the gradient is wrong, these fail.

Exit code 0 = all passed, 1 = something failed, 77 = skipped (no binary).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..")))

from pcb_aoi.solder_evolver.sdf import (                                              # noqa: E402
    write_grid, sample, sphere_sdf, box_sdf,
)

# tests/ -> evolver/ -> third_party/ -> surface_evolver/
VENDOR = os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "vendor", "surface-evolver"))


# --------------------------------------------------------------------- harness

def evolver_bin():
    env = os.environ.get("SOLDER_EVOLVER_BIN")
    if env and os.path.exists(env):
        return env
    cand = os.path.join(VENDOR, "evolver")
    return cand if os.path.exists(cand) else None


def has_collider(binary):
    """The collider is compiled in iff its error string is present."""
    with open(binary, "rb") as fh:
        return b"sdf_collider" in fh.read()


def run(binary, wd, fe_text, cmds):
    open(os.path.join(wd, "t.fe"), "w").write(fe_text)
    p = subprocess.run([binary, "t.fe"], cwd=wd, input="\n".join(cmds) + "\nq\n",
                       capture_output=True, text=True, timeout=300)
    # Evolver writes its prompt WITHOUT a trailing newline, so the first line of
    # real output arrives glued to it as "Enter command: R 0.2988...". Break the
    # prompts out before parsing, or every result silently loses its first row.
    return (p.stdout + p.stderr).replace("Enter command:", "\n")


def values(out, tag):
    return [float(l.split()[1]) for l in out.splitlines()
            if l.strip().startswith(tag + " ")]


def points(out, tag="P"):
    return [[float(t) for t in l.split()[1:4]] for l in out.splitlines()
            if l.strip().startswith(tag + " ")]


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------- the ctypes probe
# Second derivatives cannot be reached from Evolver's datafile language at all:
# `usr3` in a formula yields the value, and the Hessian is consumed silently
# inside `hessian`, where a wrong answer surfaces only as a solve that
# misbehaves for no visible reason. So compile the SAME sdf_collider.c into a
# tiny shim (tests/sdf_probe.c) and call the evaluator directly. See that file
# for what the shim fakes and what this consequently cannot catch.

PROBE_SRC = os.path.join(HERE, "sdf_probe.c")


def build_probe(wd):
    """Compile sdf_probe.c -> .so and return a ctypes handle, or None."""
    import ctypes
    import shutil

    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if not cc:
        return None
    so = os.path.join(wd, "sdf_probe.so")
    # Same dialect flags build.sh uses, because this compiles Evolver-era C.
    r = subprocess.run(
        [cc, "-O2", "-fPIC", "-shared", "-std=gnu89", "-fcommon",
         "-Wno-implicit-function-declaration", PROBE_SRC, "-o", so, "-lm"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"probe failed to compile:\n{r.stdout}\n{r.stderr}")
    lib = ctypes.CDLL(so)
    lib.probe_load.restype = ctypes.c_int
    lib.probe_load.argtypes = [ctypes.c_char_p]
    lib.probe_eval.restype = ctypes.c_double
    lib.probe_slot_seconds.restype = ctypes.c_double
    lib.probe_slot_deriv.restype = ctypes.c_double
    return lib


def probe_at(lib, x):
    """(value, grad[3], hess[3,3]) at world point `x`, via probe_eval."""
    import ctypes
    xa = (ctypes.c_double * 3)(*[float(t) for t in x])
    g = (ctypes.c_double * 3)()
    h = (ctypes.c_double * 9)()
    v = lib.probe_eval(xa, g, h)
    return v, np.array(g[:]), np.array(h[:]).reshape(3, 3)


def fd_hessian(lib, x, d):
    """Central difference of the ANALYTIC gradient. Differencing the gradient
    rather than the value keeps this a test of `seconds` alone: a bug shared by
    value and gradient would cancel out of a value-only stencil."""
    H = np.zeros((3, 3))
    for i in range(3):
        e = np.zeros(3)
        e[i] = d
        H[:, i] = (probe_at(lib, x + e)[1] - probe_at(lib, x - e)[1]) / (2 * d)
    return H


# ----------------------------------------------------------------- the checks

@case
def test_equality_constraint_projects_onto_level_set(binary, wd):
    """Vertices pinned to the collider's zero level set land on the sphere.

    Exercises value AND gradient: projection is a Newton step along grad f, so a
    wrong or badly-scaled gradient lands off the surface (or not at all).
    """
    C, R = (0.5, 0.4, 0.3), 0.30
    write_grid(os.path.join(wd, "sdf_1.bin"),
               sample(sphere_sdf(C, R), (-0.6, -0.7, -0.8), (1.6, 1.5, 1.4), 64),
               (-0.6, -0.7, -0.8), (1.6, 1.5, 1.4))
    fe = """
constraint 1
formula: usr3
vertices
1  1.30  0.40  0.30  constraint 1
2  0.50  1.20  0.30  constraint 1
3  0.50  0.40  1.10  constraint 1
4  0.10  0.05  0.02  constraint 1
edges
faces
read
"""
    out = run(binary, wd, fe, [
        'foreach vertex vv do printf "R %.9f\\n", '
        'sqrt((vv.x-0.5)^2+(vv.y-0.4)^2+(vv.z-0.3)^2);'])
    r = values(out, "R")
    assert len(r) == 4, f"expected 4 vertices, got {r}\n{out[-800:]}"
    err = max(abs(v - R) for v in r)
    assert err < 2e-3, f"vertices not on the sphere: radii {r}, max err {err:.2e}"
    return f"4 vertices projected onto |x-c|={R}, max err {err:.2e}"


@case
def test_onesided_constraint_expels_interior_vertices(binary, wd):
    """`nonnegative` pushes vertices OUT of the solid and leaves outsiders alone.

    This is the keep-out behaviour the whole thing exists for.
    """
    C, R = (0.5, 0.5, 0.5), 0.30
    write_grid(os.path.join(wd, "sdf_1.bin"),
               sample(sphere_sdf(C, R), (-0.5, -0.5, -0.5), (1.5, 1.5, 1.5), 64),
               (-0.5, -0.5, -0.5), (1.5, 1.5, 1.5))
    fe = """
constraint 1 nonnegative
formula: usr3
vertices
1  0.55 0.50 0.50  constraint 1
2  0.50 0.58 0.50  constraint 1
3  0.50 0.50 0.42  constraint 1
4  0.50 0.50 0.50  constraint 1
5  0.95 0.50 0.50  constraint 1
edges
faces
read
"""
    out = run(binary, wd, fe, [
        'foreach vertex vv do printf "R %.9f\\n", '
        'sqrt((vv.x-0.5)^2+(vv.y-0.5)^2+(vv.z-0.5)^2);'])
    r = values(out, "R")
    assert len(r) == 5, f"expected 5 vertices, got {r}\n{out[-800:]}"
    for i, v in enumerate(r[:4]):
        assert v > R - 3e-3, (
            f"vertex {i+1} was seeded INSIDE and not expelled: |v-c|={v:.6f} < {R}")
    assert abs(r[4] - 0.45) < 1e-6, (
        f"vertex 5 was already outside and should not have moved: {r[4]:.6f} != 0.45")
    return (f"4 interior vertices expelled to >={R}, exterior one untouched "
            f"(radii {[round(v, 4) for v in r]})")


@case
def test_bounds_are_read_from_the_file(binary, wd):
    """The SAME sphere on DIFFERENT bounds must give the same surface.

    Guards the lo/hi header: if bounds were ignored or mis-scaled, the level set
    would move. Two grids covering different regions at different spacings.
    """
    C, R = (1.0, 1.0, 1.0), 0.5
    got = []
    for lo, hi, n in [((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), 48),
                      ((-1.0, -0.5, 0.2), (3.0, 2.5, 1.9), 56)]:
        write_grid(os.path.join(wd, "sdf_1.bin"),
                   sample(sphere_sdf(C, R), lo, hi, n), lo, hi)
        fe = """
constraint 1
formula: usr3
vertices
1  1.90  1.00  1.00  constraint 1
edges
faces
read
"""
        out = run(binary, wd, fe, [
            'printf "R %.9f\\n", sqrt((vertex[1].x-1)^2+(vertex[1].y-1)^2+(vertex[1].z-1)^2);'])
        v = values(out, "R")
        assert v, f"no output for bounds {lo}..{hi}\n{out[-800:]}"
        got.append(v[0])
    for v in got:
        assert abs(v - R) < 3e-3, f"level set moved with the bounds: radii {got}"
    assert abs(got[0] - got[1]) < 3e-3, f"bounds not honoured: {got}"
    return f"same sphere on two different grids/bounds -> radii {[round(v,5) for v in got]}"


@case
def test_outside_the_grid_is_finite_and_positive(binary, wd):
    """A vertex far outside the sampled box gets a sane value, not NaN or a cliff.

    The field is extended as f(clamp(x)) + |x-clamp(x)|, so the constraint stays
    satisfied out there and a stray vertex is not yanked back or NaN-poisoned.
    """
    C, R = (0.0, 0.0, 0.0), 0.3
    lo, hi = (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)
    write_grid(os.path.join(wd, "sdf_1.bin"),
               sample(sphere_sdf(C, R), lo, hi, 32), lo, hi)
    fe = """
constraint 1 nonnegative
formula: usr3
vertices
1   8.0  0.0  0.0  constraint 1
2   0.0 -9.0  0.0  constraint 1
3   5.0  5.0  5.0  constraint 1
edges
faces
read
"""
    out = run(binary, wd, fe, [
        'foreach vertex vv do printf "P %.9f %.9f %.9f\\n", vv.x, vv.y, vv.z;'])
    pts = points(out)
    assert len(pts) == 3, f"expected 3 vertices, got {pts}\n{out[-800:]}"
    P = np.array(pts)
    assert np.isfinite(P).all(), f"non-finite coordinates outside the grid: {P}"
    moved = np.abs(P - np.array([[8, 0, 0], [0, -9, 0], [5, 5, 5]], float)).max()
    assert moved < 1e-6, f"far-outside vertices should satisfy f>=0 and not move; moved {moved:.3e}"
    return "3 vertices far outside the grid: finite, satisfied, unmoved"


@case
def test_box_collider_reproduces_flat_faces(binary, wd):
    """A box SDF, so the level set has flat faces and sharp edges.

    A sphere alone would not catch a scaling error that happens to be isotropic;
    an anisotropic box on an anisotropic grid does.
    """
    C, H = (0.0, 0.0, 0.0), (0.5, 0.3, 0.2)
    lo, hi = (-1.2, -1.0, -0.9), (1.2, 1.0, 0.9)
    write_grid(os.path.join(wd, "sdf_1.bin"),
               sample(box_sdf(C, H), lo, hi, (72, 64, 60)), lo, hi)
    fe = """
constraint 1
formula: usr3
vertices
1   1.00  0.00  0.00  constraint 1
2   0.00  0.90  0.00  constraint 1
3   0.00  0.00  0.80  constraint 1
edges
faces
read
"""
    out = run(binary, wd, fe, [
        'foreach vertex vv do printf "P %.9f %.9f %.9f\\n", vv.x, vv.y, vv.z;'])
    pts = points(out)
    assert len(pts) == 3, f"expected 3 vertices, got {pts}\n{out[-800:]}"
    # each vertex approaches its own face along its own axis
    got = [pts[0][0], pts[1][1], pts[2][2]]
    for axis, (g, want) in enumerate(zip(got, H)):
        assert abs(g - want) < 8e-3, (
            f"axis {axis}: landed at {g:.5f}, box face is at {want} "
            f"-- anisotropic spacing likely mishandled")
    return f"3 faces hit at {[round(v,4) for v in got]} (expected {list(H)})"


# ------------------------------------------------- second derivatives (Hessian)
# These exist because `hessian` / `hessian_seek` minimisation consults
# sdf1_seconds, and a wrong Hessian does not fail loudly -- it just makes Newton
# take bad steps. The sphere is the ideal subject: an exact sphere SDF has a
# Hessian we know in closed form, (I - n n^T)/r, and it is nowhere flat, so a
# zeroed or mis-scaled curvature has nothing to hide behind.

SEC_C, SEC_R = (0.0, 0.0, 0.0), 0.5
SEC_LO, SEC_HI, SEC_N = (-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), 128

# FD step. The field is a cubic B-spline, so a central difference of its
# gradient carries O(d^2 * f'''') truncation error, while the analytic Hessian
# itself is exact -- meaning the residual we measure is almost entirely the
# STENCIL's error, not the code's. d = 1e-3 mm sits ~24x below the 0.0236 mm
# grid spacing (small enough that f'''' is the local cell's, not a smear across
# several) and ~1e10 above the double-precision noise floor of differencing two
# O(1) gradients. Measured residual at that step is 3e-5; the tolerance below is
# 10x that, which leaves room for a different machine's rounding but not for an
# actual sign, scaling or transpose error, all of which are O(1) here.
SEC_FD_STEP = 1e-3
SEC_TOL = 3e-4


def _sec_grid(wd):
    path = os.path.join(wd, "sphere.bin")
    write_grid(path, sample(sphere_sdf(SEC_C, SEC_R), SEC_LO, SEC_HI, SEC_N),
               SEC_LO, SEC_HI)
    return path


def _sec_points(rng, n=40):
    """Sample points well clear of BOTH singular regions: the sphere's centre,
    where the true SDF has a |x| cusp and infinite curvature, and the grid
    boundary, where the cubic stencil starts reading clamped border samples."""
    P = []
    while len(P) < n:
        x = rng.uniform(-1.1, 1.1, 3)
        if np.linalg.norm(x) > 0.15:
            P.append(x)
    return P


@case
def test_seconds_match_finite_differences_of_the_gradient(binary, wd):
    """The analytic Hessian agrees with differences of the analytic gradient.

    This is the primary check and it is self-contained: it does not care what
    the field MEANS, only that `seconds` really is the derivative of `deriv`.
    """
    lib = build_probe(wd)
    if lib is None:
        return "SKIPPED (no C compiler to build the probe)"
    assert lib.probe_load(_sec_grid(wd).encode()), "probe could not load the grid"

    rng = np.random.default_rng(20260731)
    worst, worst_at = 0.0, None
    for x in _sec_points(rng):
        H = probe_at(lib, x)[2]
        e = np.abs(H - fd_hessian(lib, x, SEC_FD_STEP)).max()
        if e > worst:
            worst, worst_at = e, x
    assert worst < SEC_TOL, (
        f"analytic Hessian disagrees with FD of the gradient by {worst:.3e} "
        f"at {np.round(worst_at, 4)} (tolerance {SEC_TOL:.1e})")
    return f"40 interior points, max |H - FD(grad)| = {worst:.2e}"


@case
def test_seconds_match_the_exact_sphere_hessian(binary, wd):
    """...and the derivative it converges to is the RIGHT one.

    The FD check above would pass just as happily on a consistently wrong field.
    An exact sphere SDF f = |x-c| - R has Hessian (I - n n^T)/r, so compare
    against that. The bar is looser because the B-spline is an APPROXIMATING
    basis (no prefilter), so it smooths the samples: the value is O(h^2) off and
    each differentiation costs another order. 5% is what a 128^3 grid buys.
    """
    lib = build_probe(wd)
    if lib is None:
        return "SKIPPED (no C compiler to build the probe)"
    assert lib.probe_load(_sec_grid(wd).encode()), "probe could not load the grid"

    c = np.asarray(SEC_C, float)
    rng = np.random.default_rng(20260731)
    worst = 0.0
    for x in _sec_points(rng):
        r = np.linalg.norm(x - c)
        n = (x - c) / r
        exact = (np.eye(3) - np.outer(n, n)) / r
        H = probe_at(lib, x)[2]
        worst = max(worst, np.abs(H - exact).max() / np.abs(exact).max())
    assert worst < 0.05, (
        f"interpolated curvature is {100 * worst:.1f}% off the exact sphere "
        "Hessian -- more than interpolation error can explain")
    return f"max relative error vs (I-nn^T)/r = {100 * worst:.2f}%"


@case
def test_seconds_outside_the_grid_match_finite_differences(binary, wd):
    """The out-of-box extension has curvature too, and it is not zero.

    f(x) = f(clamp x) + |x - clamp x| is LINEAR along the overhang but not
    across it: the distance term contributes (I - n n^T)/|o| on the clamped
    axes. This used to return an all-zero Hessian out here, which is worse than
    inexact -- a Newton step against zero curvature is unbounded, so a vertex
    that strayed off the grid would be thrown to infinity.

    Three cases on purpose: one axis clamped, two, and three. The one- and
    two-axis cases are the ones that catch a block-structure mistake, since they
    mix interpolated curvature with distance curvature in the same matrix.
    """
    lib = build_probe(wd)
    if lib is None:
        return "SKIPPED (no C compiler to build the probe)"
    assert lib.probe_load(_sec_grid(wd).encode()), "probe could not load the grid"

    got = []
    for x in (np.array([3.0, 0.2, 0.1]),        # x clamped only
              np.array([2.0, 2.0, 0.0]),        # x and y
              np.array([-4.0, 1.9, 2.4])):      # all three
        H = probe_at(lib, x)[2]
        e = np.abs(H - fd_hessian(lib, x, SEC_FD_STEP)).max()
        assert e < SEC_TOL, (
            f"out-of-grid Hessian at {x} off by {e:.3e} (tolerance {SEC_TOL:.1e})")
        assert np.abs(H).max() > 1e-3, (
            f"out-of-grid Hessian at {x} is ~zero; the transverse 1/|o| "
            "curvature of the distance term is missing")
        got.append(e)
    return ("1-, 2- and 3-axis overhangs match FD to "
            + ", ".join(f"{e:.1e}" for e in got))


@case
def test_seconds_are_symmetric_and_reach_evolvers_slot(binary, wd):
    """The Hessian is symmetric, and sdf1_seconds() marshals it correctly.

    Evolver hands `seconds` in as an array of ROW POINTERS, which is easy to
    fill transposed or short. Going through sdf1_seconds() (rather than the
    internal evaluator) covers the marshalling that the shipped binary uses, and
    checks its `partials` agree with sdf1_deriv().
    """
    import ctypes

    lib = build_probe(wd)
    if lib is None:
        return "SKIPPED (no C compiler to build the probe)"
    assert lib.probe_load(_sec_grid(wd).encode()), "probe could not load the grid"

    rng = np.random.default_rng(7)
    asym = diff = dg = 0.0
    for x in _sec_points(rng, 12):
        xa = (ctypes.c_double * 3)(*x)
        p = (ctypes.c_double * 3)()
        s = (ctypes.c_double * 9)()
        v = lib.probe_slot_seconds(xa, p, s)
        H = np.array(s[:]).reshape(3, 3)
        asym = max(asym, np.abs(H - H.T).max())
        diff = max(diff, np.abs(H - probe_at(lib, x)[2]).max())

        pd = (ctypes.c_double * 3)()
        vd = lib.probe_slot_deriv(xa, pd)
        dg = max(dg, np.abs(np.array(p[:]) - np.array(pd[:])).max(),
                 abs(v - vd))
    assert asym < 1e-12, f"Hessian is not symmetric: max |H - H^T| = {asym:.2e}"
    assert diff < 1e-12, f"slot marshalling altered the Hessian by {diff:.2e}"
    assert dg < 1e-12, f"seconds' value/partials disagree with deriv's by {dg:.2e}"
    return (f"symmetric to {asym:.1e}, slot round-trip exact, "
            f"value/partials agree with sdf1_deriv to {dg:.1e}")


DROP_ON_COLLIDER_BODY = """
vertices
1  -0.5 -0.5  0.0 %(con)s
2   0.5 -0.5  0.0 %(con)s
3   0.5  0.5  0.0 %(con)s
4  -0.5  0.5  0.0 %(con)s
5  -0.5 -0.5  1.0 %(con)s
6   0.5 -0.5  1.0 %(con)s
7   0.5  0.5  1.0 %(con)s
8  -0.5  0.5  1.0 %(con)s
edges
1 1 2 %(con)s
2 2 3 %(con)s
3 3 4 %(con)s
4 4 1 %(con)s
5 5 6 %(con)s
6 6 7 %(con)s
7 7 8 %(con)s
8 8 5 %(con)s
9 1 5 %(con)s
10 2 6 %(con)s
11 3 7 %(con)s
12 4 8 %(con)s
faces
1 1 10 -5 -9
2 2 11 -6 -10
3 3 12 -7 -11
4 4 9 -8 -12
5 5 6 7 8
6 -4 -3 -2 -1
bodies
1 1 2 3 4 5 6 volume 1
read
"""

# A big sphere used as a curved FLOOR: radius 2 centred at z=-2, apex at the
# origin, with a unit-cube drop resting on it. The drop wants to ball up
# (free sphere r=0.62) and the floor is in the way, so contact is unavoidable.
FLOOR_C, FLOOR_R = (0.0, 0.0, -2.0), 2.0
FLOOR_LO, FLOOR_HI, FLOOR_N = (-1.6, -1.6, -0.7), (1.6, 1.6, 1.7), 72
EVOLVE = "r; g 40; u; r; g 60; u; g 60;"


def _floor_grid(wd):
    write_grid(os.path.join(wd, "sdf_1.bin"),
               sample(sphere_sdf(FLOOR_C, FLOOR_R), FLOOR_LO, FLOOR_HI, FLOOR_N),
               FLOOR_LO, FLOOR_HI)


def _deepest(out):
    """Smallest |v - floor centre| over all vertices; < R means penetration."""
    r = values(out, "R")
    assert r, f"no vertex output\n{out[-1000:]}"
    return min(r)


REPORT_R = ('foreach vertex vv do printf "R %.6f\\n", '
            'sqrt(vv.x^2+vv.y^2+(vv.z+2)^2);')


@case
def test_penalty_energy_pushes_vertices_out(binary, wd):
    """THE WORKING KEEP-OUT: an energy, not a constraint.

    A one-sided constraint is inert under minimisation (see the next test), but
    a penalty ENERGY built on the same field is not: energies are evaluated
    every step by construction, so there is no acquisition step to miss.

        quantity keepout energy method facet_scalar_integral global
        scalar_integrand: kpen*(maximum(-(usr3),0))^2

    `facet_scalar_integral` uses Gaussian cubature, so the integrand is sampled
    INSIDE each facet, not only at its corners -- which also addresses the
    facet-spanning worry that per-vertex enforcement cannot.

    Note `-(usr3)` and not `-usr3`: the lexer mis-tokenises `usrN` followed
    directly by an operator (`Illegal token '-0r2'`), so always parenthesise.
    """
    _floor_grid(wd)
    fe = ("""
parameter kpen = 200

quantity keepout energy method facet_scalar_integral global
scalar_integrand: kpen*(maximum(-(usr3),0))^2
""" + DROP_ON_COLLIDER_BODY % {"con": ""})

    got = {}
    for k in (0, 20, 200, 2000):
        out = run(binary, wd, fe, [f"kpen := {k};", EVOLVE, REPORT_R])
        got[k] = FLOOR_R - _deepest(out)

    assert got[0] > 0.10, (
        f"with kpen=0 the drop should sink freely; penetration {got[0]:.4f}")
    for a, b in zip((0, 20, 200), (20, 200, 2000)):
        assert got[b] < got[a], (
            f"penetration did not fall from kpen={a} to {b}: "
            f"{got[a]:.4f} -> {got[b]:.4f}")
    assert got[2000] < 0.005, (
        f"stiff penalty still lets the surface {got[2000]:.4f} inside")
    return ("penetration vs stiffness: " +
            ", ".join(f"kpen={k} -> {v:.4f}" for k, v in got.items()))


@case
def test_KNOWN_one_sided_constraint_is_inert_under_minimisation(binary, wd):
    """CHARACTERISATION TEST -- documents a LIMITATION, not a success.

    The field and the projection are correct (see the checks above). But a
    one-sided constraint on a real surface under tension is INERT during
    minimisation: the drop sinks straight through the collider and lands on the
    free-sphere solution, bit-for-bit identical to a run with no constraint at
    all.

        with the collider   deepest |v-c| = 1.879162
        no constraint       deepest |v-c| = 1.879162
        free sphere of vol 1 centred (0,0,0.5), lowest point -> 1.879650

    This asserts the CURRENT behaviour so the suite is honest and green. If it
    ever starts FAILING, that is good news -- the constraint began to bite, and
    the SDF collider can then be used as a keep-out rather than only as a pinning
    surface.

    Note this reproduces docs/solver_paradigms.md ("solder solved with vs without
    a lead came out bit-for-bit identical") with a SMOOTH, verified field -- so
    the inertness is NOT caused by the kinky min-of-max formula, as was
    previously assumed.

    ROOT CAUSE, measured with Evolver's own diagnostics:

        vertices that START in violation   hit_constraint 1, value -> 0.000000
        vertices that BECOME violating     hit_constraint 0, value    -0.120478

    Evolver ACQUIRES a one-sided constraint only at load time. A vertex that
    wanders in during `g` is never flagged, and `recalc` does not re-acquire it
    (13 violating vertices, 0 hit, before and after). The manual says as much in
    passing: "'g' will check for a vertex wanting to LEAVE a one-sided constraint
    IT HAS HIT" -- there is no acquisition step.

    Use the penalty energy above instead.

    A drop resting on a curved floor: sphere radius 2 centred at z=-2, so its
    apex is the origin. (An enclosing topology cannot test this at all -- a body
    around a sphere of radius R needs volume >= (4/3)pi R^3, whose free sphere
    has radius exactly R, so tension stops it at the obstacle for reasons that
    have nothing to do with the obstacle.)
    """
    C, R = (0.0, 0.0, -2.0), 2.0
    lo, hi = (-1.6, -1.6, -0.7), (1.6, 1.6, 1.7)
    write_grid(os.path.join(wd, "sdf_1.bin"),
               sample(sphere_sdf(C, R), lo, hi, 72), lo, hi)
    # unit cube resting on the collider's apex; volume 1 -> free sphere r~0.62
    fe = """
constraint 1 nonnegative
formula: usr3
vertices
1  -0.5 -0.5  0.0  constraint 1
2   0.5 -0.5  0.0  constraint 1
3   0.5  0.5  0.0  constraint 1
4  -0.5  0.5  0.0  constraint 1
5  -0.5 -0.5  1.0  constraint 1
6   0.5 -0.5  1.0  constraint 1
7   0.5  0.5  1.0  constraint 1
8  -0.5  0.5  1.0  constraint 1
edges
1   1 2 constraint 1
2   2 3 constraint 1
3   3 4 constraint 1
4   4 1 constraint 1
5   5 6 constraint 1
6   6 7 constraint 1
7   7 8 constraint 1
8   8 5 constraint 1
9   1 5 constraint 1
10  2 6 constraint 1
11  3 7 constraint 1
12  4 8 constraint 1
faces
1   1 10 -5  -9
2   2 11 -6 -10
3   3 12 -7 -11
4   4  9 -8 -12
5   5  6  7   8
6  -4 -3 -2  -1
bodies
1   1 2 3 4 5 6  volume 1
read
"""
    out = run(binary, wd, fe, [
        'printf "V0 %.9f\\n", body[1].volume;',
        'printf "E0 %.9f\\n", total_energy;',
        "r; g 40; u; r; g 60; u; g 60;",
        'printf "V1 %.9f\\n", body[1].volume;',
        'printf "E1 %.9f\\n", total_energy;',
        'foreach vertex vv do printf "R %.9f\\n", '
        'sqrt(vv.x^2+vv.y^2+(vv.z+2)^2);',
    ])
    v0, v1 = values(out, "V0"), values(out, "V1")
    e0, e1 = values(out, "E0"), values(out, "E1")
    r = np.array(values(out, "R"))
    assert v0 and v1 and e0 and e1 and len(r), f"missing output\n{out[-1200:]}"
    assert abs(v1[0] - 1.0) < 1e-4, f"volume not conserved: {v0[0]} -> {v1[0]}"
    assert e1[0] <= e0[0] + 1e-9, f"energy increased: {e0[0]} -> {e1[0]}"

    worst = r.min()
    free = abs(0.5 - (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) + 2.0)   # 1.87965

    # Assert the LIMITATION, so a future improvement trips this loudly.
    assert worst < R - 0.05, (
        "the one-sided constraint appears to be WORKING now -- the surface no "
        f"longer sinks into the collider (deepest |v-c|={worst:.6f} vs R={R}). "
        "That is good news: re-read this test, delete it, and replace it with a "
        "real keep-out assertion.")
    assert abs(worst - free) < 0.02, (
        f"surface settled at |v-c|={worst:.6f}, which is neither blocked by the "
        f"collider (R={R}) nor the free-sphere solution ({free:.6f}); something "
        "third is going on and this test's story is wrong")
    return (f"INERT as documented: {len(r)} verts, vol {v1[0]:.6f}, energy "
            f"{e0[0]:.4f} -> {e1[0]:.4f}; sank to |v-c|={worst:.6f} "
            f"(free sphere {free:.6f}, collider R={R})")


# ---------------------------------------------------------------------- main

def main():
    binary = evolver_bin()
    if not binary:
        print(f"SKIP: no evolver binary. Build it:\n"
              f"  {os.path.normpath(os.path.join(HERE, '..', 'build.sh'))}")
        return 77
    print(f"binary: {binary}")
    if not has_collider(binary):
        print("SKIP: this binary has no SDF collider compiled in.\n"
              "  Rebuild with: third_party/evolver/build.sh --clean")
        return 77

    npass = nfail = 0
    for fn in CASES:
        name = fn.__name__
        with tempfile.TemporaryDirectory(prefix="sdfcol_") as wd:
            try:
                note = fn(binary, wd)
                print(f"  PASS  {name}\n          {note}")
                npass += 1
            except AssertionError as e:
                print(f"  FAIL  {name}\n          {e}")
                nfail += 1
            except Exception as e:                      # noqa: BLE001
                print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
                nfail += 1
    print(f"\n{npass} passed, {nfail} failed")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
