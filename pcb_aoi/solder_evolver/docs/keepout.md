# The SDF keep-out: an experimental collision energy

Pinning fixes where the solder *ends*, not where it may *pass*. A starved joint
has nothing else holding its meniscus out of the component, and the surface
happily tunnels through the part on its way to a smaller area. The keep-out is
the answer: a signed-distance field of the component, baked to a grid, turned
into an energy that punishes any free vertex inside it.

## Why a penalty and not a constraint

Surface Evolver's one-sided (`nonnegative`/`nonpositive`) constraints are
**inert under minimisation** — verified directly, and worth restating because
it is the single most surprising fact in this corner: a vertex placed on the
wrong side of a one-sided constraint is not pushed back by `g` steps. Only an
*energy* participates in minimisation, so the keep-out is a penalty:

    E = kpen * integral( max(0, -sdf)^2 )   over the free surface

`kpen = 1000` in the adapter defaults. The penalty is C1 but not C2, which is
one of the two reasons raw `hessian` steps are untrustworthy here (see
[solving.md](solving.md)) — `hessian_seek` line-searches its way around it.

## The pipeline

1. `sdf.mesh_sdf_grid(V, F, lo, hi, n)` — warp `wp.Mesh` +
   `mesh_query_point_sign_winding_number` on GPU (`support_winding_number=True`
   is load-bearing), or a trimesh fallback on CPU. Grid resolution 64³ in the
   adapter; the bake is near-instant on GPU.
2. `sdf.bake_keepout_sdf(path, ...)` — folds the board plane in via `floor_z`
   (solder may not pass below the pad either), inflates by a skin thickness,
   writes the grid file. Evolver resolves the path against its working
   directory, so the `.bin` sits next to the datafile.
3. `emit_datafile(..., keepout={"grid": "collider.bin", "kpen": ...})` — the
   grid lands in `usrN` slots and the penalty quantity attaches to the free
   facets.

The grid file format (`MAGIC`/version/dims/bounds, x-fastest float64 values) is
pinned by `tests/unittests/solver/solder_evolver/test_sdf.py` — it is a wire
contract with the Evolver-side reader, which needs the `-DENABLE_DLL` build
(`test_evolver_dll.py` gates that capability).

## When it is on

Per-strategy, not global. Nominal joints run **without** it — enough solder to
sit on the footprint, and the keep-out costs roughly 8× per iteration. The
starved strategies (`insufficient`, `missing_solder`) run **with** it, and
gradient-only: Newton stalls ~2.8e-2 short of gradient when the meniscus is
pressed along the keep-out — a wrong shape, not a tolerance.

## Young's-equation spreading: incomplete

The same field could in principle free the contact line — let it slide under
Young's law instead of pinning it, with a hydrophobic pad boundary stopping it
at the copper edge. Status, honestly:

* Pinned patches make `theta` a constant in the energy (measured: 5°/35°/80°
  give bit-identical geometry). The knob is dead by construction today.
* Freeing the pad patch onto a plane constraint with `tension -cos(theta)`
  works at θ ≈ 70° and **diverges catastrophically below it**: the patch
  inverts, and an inverted patch satisfies the volume constraint (divergence
  theorem) while its area grows without bound — at `-cos 20° = -0.94` that is
  free energy. The attempt remains an unmerged experiment.
* The likely fix is the **constraint-integrand formulation**: wetting energy as
  a vector integrand on the constraint (Stokes over the contact-line edges), no
  bottom facets at all, body closed by `content`. Nothing to invert.
  `datafiles/sessile.fe` is kept precisely as the working reference for that
  formulation — it recovers a prescribed contact angle to < 0.5°.

Until then, "flood the pad" is expressed geometrically: the `skirt` (SMT) and
the pad annulus (THT) put the pinned contact line where a spread meniscus would
have ended up.
