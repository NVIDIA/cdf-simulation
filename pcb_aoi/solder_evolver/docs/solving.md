# Newton (Hessian) minimisation in the solder solve

> Part of the `pcb_aoi.solder_evolver` module docs. The measurements below were
> taken with an earlier development harness; module names refer to the current
> layout (boolean / emit / solve).

Status: **available, opt-in, measured. Not the default, and it should not be.**

**Summary.** Newton is a 6-11x wall-clock win *when the meniscus is not pressed
against the collider*, and it stalls well short of the right answer when it is.
The deciding variable is CONTACT, not the fill and not the collider as such.
That makes it a valuable opt-in and a bad default, since low fill against a
collider is exactly the case the defect taxonomy cares about.

Gradient descent on a starved 0805 needs ~7000 iterations to settle, and every
iteration costs ~8x more once the SDF collider is on (1370 s for one solve).
Speeding that up is the entire point of looking at a second-order method.

## The short version

Use `hessian_seek`, never raw `hessian`.

That one substitution is why this works at all. Raw `hessian` steps converge
faster but overshoot on these free surfaces -- they invert triangles and blow
up -- which is why gradient descent is the default and Hessian is opt-in only.

```
--hessian off       gradient descent -- the default, and the reference
--hessian seek      hessian_seek blocks in place of the `g` blocks
--hessian polish    the gradient pass in full, then Newton on top
```

Available as `--hessian` on `python -m pcb_aoi.solder_evolver.solve` and as
`hessian=` on `solve.solve_stages()`. The package backend's `use_hessian=True`
(`solder_evolver.backend`) also issues `hessian_seek`, though its default
remains `False` and its shapes have not been measured.

## Why raw `hessian` blows up, and `hessian_seek` does not

Two independent reasons, and both are structural rather than bad luck.

**1. The matrix is indefinite by construction.** A fixed body volume is not
projected out; Evolver borders the Hessian with one Lagrange row per fixed
volume or fixed quantity, whose multiplier *is* the pressure
(`hessian.c:2040-2085`, `hessian3.c:252`, `bk.c:1489`). The assembled system is
therefore a saddle-point system and has negative eigenvalues in every single
solve. Evolver factors it with an explicitly indefinite factorisation and, when
it finds negative pivots, prints

```
WARNING 1825: Hessian not positive definite. Index: 1
```

...and then **takes the step anyway** (`bk.c:1799`). Raw `hessian` moves at
stepsize exactly 1.0 along whatever that solve returned. On a floppy free
surface that is how you get inverted triangles.

**2. The keep-out penalty is C1 but not C2.** Our collider energy is

```
quantity keepout energy method facet_scalar_integral
scalar_integrand: kpen*(maximum(-(usr3),0))^2
```

`max(-f,0)^2` has a second derivative that jumps from `2*kpen*|grad f|^2` to `0`
across the contact boundary. Evolver does not complain: `eval_sec.c`'s
`MAXIMUM_NODE` copies the winning branch's whole derivative stack, so you get a
valid *one-sided* Hessian and no warning. Newton's quadratic model is simply
wrong for any step that crosses the boundary.

`hessian_seek` fixes both by line-searching along the Newton direction and
committing only to a step that actually lowers the energy. An indefinite solve
and a wrong-branch quadratic model both produce bad *proposals*, and a line
search rejects bad proposals for free.

### Smoothing the penalty to C2: tested, helps, does not fix it

The obvious alternative is to replace `max(-f,0)^2` with something C2. We tried
the cheapest such swap, `max(-f,0)^3`, on the case where Newton actually loses
(fill 0.1, collider on). A cubic's restoring force is `~kpen*d^2` rather than
`~kpen*d`, so matching the quadratic's stiffness at a ~0.01 mm penetration needs
about 30x the `kpen`; energies are therefore NOT comparable between the two
penalties, and each gets its own gradient baseline.

| penalty | gradient | seek | shortfall | indef |
|---|---|---|---|---|
| `^2` (C1), kpen 1e3 | -0.765506 | -0.737456 | **2.81e-2** | 281 |
| `^3` (C2), kpen 3e4 | -0.769922 | -0.759849 | **1.01e-2** | 280 |

So making the energy C2 recovers about **two thirds** of Newton's shortfall --
the kink is doing real damage, and this is the measurement that proves it -- but
it does **not** close the gap. Newton still stops 1.3% short of its own gradient
baseline, and, tellingly, **the indefinite-factorisation count does not move at
all** (281 -> 280).

That last number is the important one. If the penalty's Hessian discontinuity
were the source of the indefiniteness, smoothing it would have reduced the
count. It did not. The indefiniteness is dominated by the CONTACT GEOMETRY
itself -- a surface pressed flat along an obstacle has near-zero curvature in
the directions it is free to move, which is a genuinely ill-conditioned Newton
problem no matter how smooth the energy is.

We therefore did not adopt the cubic penalty:

* it buys 2/3 of a fix for a case where we are recommending gradient descent
  anyway, so it buys nothing where it is used;
* it changes the **physics**, not just the numerics. The equilibrium penetration
  depth is set by where the penalty gradient balances surface tension, so `kpen`
  and `--sdf-inflate` would both have to be re-tuned against measured
  penetration -- a real cost paid across every existing calibration.

If the collider low-fill case ever needs to be fast, the cubic penalty is a
component of the answer but not the whole of it; the conditioning of the contact
patch has to be dealt with too.

If a future need for a genuinely C2 energy appears (a trust-region method that
cannot line-search, say), the smoothed penalty is the right answer then.

## Analytic second derivatives for the grid SDF

`sdf1_seconds` / `sdf2_seconds` in `third_party/evolver/src/sdf_collider.c` were
already implemented analytically -- the field is a cubic B-spline, which is C2,
so the exact Hessian is a tensor product of the basis' second derivatives. Two
defects were found and fixed:

* the out-of-grid extension `f(x) = f(clamp x) + |x - clamp x|` returned an
  **all-zero** Hessian. It is linear *along* the overhang but not across it: the
  distance term contributes `(I - n n^T)/|o|` on the clamped axes. Zero curvature
  makes a Newton step unbounded, so a vertex that strayed off the grid would be
  thrown to infinity.
* `sdf_basis()` returned before filling `dw[]`/`ddw[]` when asked for curvature
  but not gradient, leaving both as stack garbage. Latent -- nothing took that
  path -- but it is exactly the path a future caller would take.

`tests/test_sdf_collider.py` now checks the Hessian against central differences
of the analytic gradient (max residual 2.4e-5, which is the *stencil's*
truncation error, not the code's), against the exact sphere Hessian
`(I - n n^T)/r` (0.09% -- that residual is the B-spline's approximation error),
and out of the grid on 1-, 2- and 3-axis overhangs. Evolver's datafile language
cannot ask for a second derivative at all, so the test compiles the same
`sdf_collider.c` into a ctypes shim (`tests/sdf_probe.c`) and calls the
evaluator directly.

## Measurements

`PASSIVE_0805_g02` joint 0, sphere blob, `--nrefine 2`, gradient pass
`--nconv 80 --convblock 100`, Newton pass `--nhess 50 --hessblock 20`. Mesh
metrics are over the FREE surface only (the fixed wetted patches are not
touched by the minimiser, so they would only dilute the comparison).

### Collider OFF

| fill | mode | energy | wall | g it | h it | indef | ed med/min | rough p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| 0.1 | gradient | -0.844136 | 170.0 s | 8000 | - | 0 | 1797 | 0.0444 / 0.1542 |
| 0.1 | seek | -0.842599 | **34.4 s** | - | 212 | 112 | 332 | 0.0500 / 0.2162 |
| 0.1 | polish | **-0.852218** | 208.9 s | 8000 | 294 | 32 | **36.5** | 0.0428 / 0.1517 |
| 1.0 | gradient | -0.6063278 | 77.9 s | 3340 | - | 0 | 44.7 | 0.0180 / 0.1436 |
| 1.0 | seek | -0.6063272 | **22.3 s** | - | 99 | 0 | 44.6 | 0.0195 / 0.1472 |
| 1.0 | polish | -0.6063282 | 97.7 s | 3340 | 80 | 0 | 45.0 | 0.0167 / 0.1406 |

### Collider ON (`--sdf`, kpen 1000, 64^3 grid, inflate 0.01)

| fill | mode | energy | wall | g it | h it | indef | ed med/min | rough p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| 0.1 | gradient | **-0.765506** | 1370.2 s | 7440 | - | 0 | 2.4e7 | 0.0587 / 0.2465 |
| 0.1 | seek | -0.737223 | 206.1 s | - | 294 | 231 | 456 | 0.0637 / 0.4308 |
| 0.1 | polish | -0.765477 | 1462.5 s | 7440 | 50 | 50 | **131** | 0.0557 / 0.2000 |
| 1.0 | gradient | -0.6027470 | 1169.4 s | 5980 | - | 0 | 44.8 | 0.0180 / 0.1470 |
| 1.0 | seek | -0.6027463 | **102.3 s** | - | 99 | 0 | 44.6 | 0.0195 / 0.1461 |
| 1.0 | polish | -0.6027464 | 1263.7 s | 5980 | 80 | 0 | 45.0 | 0.0167 / 0.1405 |

### Reading it

**The clean win is fill 1.0 with the collider on: 11.4x faster for the same
energy to 7 significant figures.** Seven of those minutes per joint, across
thousands of joints, is the entire scaling argument.

**The clean loss is fill 0.1 with the collider on.** `seek` gives up at
-0.7372 against gradient's -0.7655 -- a 2.8e-2 shortfall, which is not a
tolerance question, it is the wrong shape. `polish` cannot rescue it either: its
Newton pass managed only 50 steps before stalling, for a 7% wall-clock penalty
and no gain.

**`indef` is the tell, and it is not the volume row.** Note the pattern: 0
indefinite factorisations at fill 1.0 with the collider (surface barely touching
it), 231 at fill 0.1 with the collider (surface pressed along it), 112 at fill
0.1 without. Indefiniteness tracks CONTACT AREA, and the runs where Newton
stalls are exactly the runs with a lot of it.

It is tempting to blame the C1 kink in `max(-f,0)^2` for this, and it is worth
about a third of the blame -- see the C2 experiment above, which recovers two
thirds of the shortfall by smoothing the penalty. But that same experiment left
the indefinite count unchanged, so the conditioning problem is the contact
geometry itself: a surface pressed flat along an obstacle has almost no
curvature in the directions it is free to move.

### Two cautions on the numbers

*Wall clock is single-run on a shared 32-core box.* Differences of a few percent
are noise. The differences that decided this were 1.1x-11.4x.

*`ed med/min` is a minimum, so it reports the single worst edge and says nothing
about how many there are.* A 1st-percentile companion is worth checking for
exactly this reason. Worked example: `--hessian polish --nconv 10` (a short
gradient prefix) reaches -0.852123 in 87 s -- better than the gradient baseline's
-0.844136 in 170 s on BOTH axes -- but scores med/min 9.4e8 while its
1st-percentile ratio is a healthy 13.5. That is one collapsed edge, not a
degenerate surface. It is reproducible bit-for-bit across repeats, and `t`
cannot delete it because it runs to a pinned contact-line vertex; adding a final
`u; V; t` after the Newton pass was tried and changed nothing measurable, so it
was not kept.

## Recommendation

Keep gradient descent as the default. Reach for `--hessian seek` when the
meniscus is not in sustained contact with the collider (normal and excess fill),
where it is worth 6-11x. Do not use it for starved joints against a collider:
that case needs both a C2 penalty (worth 2/3 of the gap, at the cost of
re-tuning `kpen` and `--sdf-inflate` against measured penetration) and something
about the conditioning of the contact patch, and neither is done.

## Staged Newton: put it in the ladder, not the tail

`solve_stages(hess_stage=N)` runs a short `hessian_seek` burst at each level of
the refine ladder -- once on the coarse mesh right after `g 30`, then after
each refine. The fixed gradient recipe holds a faceted "golfball" shape through
every block and snaps into the right one the moment Newton first touches it;
doing that early is nearly free because factorisation cost goes with vertex
count, and the coarse levels are where the faceting is.

Measured (SWITCH_DIP j0, sphere seed, nominal fill, no convergence tail):

    plain, no Newton                  energy 0.95961   11.0 s
    hess_stage 10   (ladder only)     energy 0.22682   13.0 s
    --hessian seek  (tail only)       energy 0.22806   57.9 s
    both                              energy 0.22679   29.8 s

Ladder-only beats the tail on energy and is 4.5x faster; adding the tail on top
buys ~3e-5 for 17 s. The caveat above still binds: Newton loses against the
keep-out collider on starved joints, so the `insufficient`/`missing` strategies
stay gradient-only and `hess_stage` is enabled per-strategy (nominal and the
pose defects), not globally.

