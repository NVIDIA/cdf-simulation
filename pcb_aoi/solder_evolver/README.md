# pcb_aoi.solder_evolver

Equilibrium solder-joint geometry on Ken Brakke's Surface Evolver, for
synthetic PCB-AOI training data.

## The method

Start an initial **blob** of solder. **Intersect** it against the mesh geometry
of the board and component (manifold3d booleans, with per-triangle provenance,
so every surviving facet knows which solid it came from). **Pin the
intersection-outline boundary curves.** Then update the free surface to
**minimise surface-area energy while constraining the volume**.

The pinned formulation has one deliberate consequence: the contact line is
*prescribed* by the cut, not solved for. Contact angles (`theta`) are inert
while the wetted patches are fixed — what actually sets the meniscus shape is
the footprint the cut produced against the volume the solve targets. That split
(blob sets footprint; target sets volume) is load-bearing everywhere: it is
what keeps "less solder" distinct from "wets less far".

There is **experimental support for an SDF-based "collision" energy** — a
penalty field keeping the free surface out of the component (and below-board,
via `floor_z`). The same machinery could also serve to allow **Young's-equation
spreading of the boundary curve** instead of pinning it, but that work is
**incomplete**: the free-boundary attempt diverges below ~70° (recorded on
branch `denomme/young-pad`); `datafiles/sessile.fe` is kept as the reference
for the constraint-integrand formulation that is the likely path back in. See
[docs/keepout.md](docs/keepout.md).

## The API

Everything blessed is in `__all__`:

```python
import pcb_aoi.solder_evolver as se

# geometry: seed a blob, intersect against mesh geometry, label provenance
se.make_blob, se.cut_smt_joint, se.cut_tht_joint, se.check_body, se.cluster_rects

# energy: pin the intersection outlines, minimise area at fixed volume
se.emit_datafile, se.solve_stages, se.solve_joint      # solve_joint = one call

# volume policy and defect strategies
se.volume_plan, se.resolve_defect, se.DefectStrategy

# experimental SDF collision energy
se.bake_keepout_sdf, se.mesh_sdf_grid, se.write_grid

# verification
se.measure_contact_angle
```

One joint, end to end:

```python
meshes, stats, cut = se.solve_joint(
    d, 0, kind="smt",
    cut_kw=dict(blob="aperture_offset", stencil=bs, fill=1.0, skirt=0.10),
    target_volume=real_solder_volume,
    solve_kw=dict(nrefine=2, hess_stage=10))
```

The `SolderSolverBase` implementation that drives this from the cdf workflow
lives beside the contract it implements:
`pcb_aoi/solder_generation/surface_evolver_solver.py`.

## Module map

| module | role |
|---|---|
| `boolean` | SMT seed blobs, the cut, labels, conformity checks, clustering |
| `tht` | through-hole seed (plug + crown), board-with-hole, interrupted region |
| `emit` | writes the `.fe`: pinning, tensions, volume constraint, keep-out penalty |
| `solve` | drives Evolver: refine ladder, staged Newton, mesh maintenance |
| `backend` | the Evolver process, datafile fill-in, dump parser (`vertex_fixed`) |
| `sdf` | keep-out grids (warp or trimesh backend), grid file format, analytic SDFs |
| `volume` | the one shared volume rule (`volume_plan`) |
| `defects/` | per-defect strategies — see [defects/README.md](defects/README.md) |
| `tht_spec` | THT inputs shaped like the PCBA→USD contract, provenance-tagged |
| `verify` | equilibrium signatures: contact angle, sessile-drop recovery |

## Docs

- [docs/method.md](docs/method.md) — the boolean cut, labels, conformity
- [docs/solving.md](docs/solving.md) — minimisation, staged Newton, maintenance
- [docs/keepout.md](docs/keepout.md) — the experimental collision energy + Young status
- [docs/inputs.md](docs/inputs.md) — what the solver needs from a board; THT contract gaps
- [docs/scaling.md](docs/scaling.md) — footprint-keyed reuse, board-scale cost

Defect tuning is documented separately in
[defects/README.md](defects/README.md) — the module docs stay about the solver.

Tests: `tests/unittests/solver/solder_evolver/test_*.py`.

Developer setup — building the vendored Evolver and installing the solver's
wheels into a local Isaac Sim — is in the repo root
[README](../../README.md#developer-setup-simulated-solder-solver-optional).

The solder tree, in three parts: `pcb_aoi/solder_generation/` is the
solver-agnostic FRAMEWORK (extension, work items, registry, the procedural and
pass-through solvers); this package is the ENGINE (physics, defect strategies,
its vendored Evolver source and build under `third_party/evolver/`, its binary
under the gitignored `vendor/`); the DEV TOOLS live flat under `scripts/`
(`solder_board.py` for headless board bakes and `isaac_render.py` for RTX
renders).
