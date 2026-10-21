# Solder generation developer guide

This package connects the declarative PCB-AOI Python API to interchangeable
solder-joint generators. It owns component discovery, solver dispatch,
fallback, geometry attachment, source-paste visibility, and placement
manifests.

The central separation is:

- `SolderGenerationExtension` integrates solder generation with the live CDF
  execution context and is the only package component that authors generated
  solder into the live USD stage.
- `SolderSolverBase` is the interface implemented by pluggable solvers. A
  solver prepares inputs and returns geometry; it never modifies the live
  stage.

## Runtime flow

The public API exposes one operation:

```python
job.operations.solder_joints(
    solver="procedural",
    source="components",
    component_types=["PASSIVE_SMD_0805"],
    solver_options={
        "height_ratio": 0.9,
        "resolution": 64,
        "solid": True,
        "solder_material_prim_path": "/World/Looks/SolderPaste",
    },
)
result = job.execute(
    serialization_path="board_with_solder.usda",
    windowed=False,
)
```

Final-stage and solder-layer serialization do not require a capture declaration.
Omitting capture causes `job.execute()` to stop after all non-capture orders,
including USD serialization. Supplying `serialization_path` directly to
`solder_joints(...)` instead exports only the composable solder edit layer.

The JobOrder compiler lowers that declaration into four internal operations:

```text
scene and board metadata catalog
            |
            v
 solder_joints.discover
            |
            v
 structural defects
            |
            v
  solder_joints.collect
            |
            v
 solder_joints.generate
            |
            v
  solder_joints.attach
            |
            v
    capture and render
```

Semantic dependencies determine this order, not the order in which calls
appear in a declarative script. Defects are applied before solver inputs are
collected, and capture depends on completed solder attachment.

The legacy YAML renderer does not execute these orders. Solder generation must
run through `compile_orders()` and `render_async()` with the JobOrder executor.

## Architecture boundaries

### `SolderGenerationExtension`

`solder_generation_extension.py` is the integration facade between CDF and
the solver registry. It:

1. Resolves catalog records into stable component identities.
2. Builds nominal and defect-specific `SolderWorkItem` objects.
3. Invokes the selected solver's preparation and generation methods.
4. Validates request, work-item, artifact, and coverage correlation.
5. Falls back per failed or uncovered work item.
6. Creates an extension-owned USD edit layer.
7. Authors returned meshes and suppresses replaced source paste.
8. Restores source paste for pass-through results.
9. Rolls back unsafe attachment or serialization attempts.
10. Returns a `SolderPlacementManifest` describing stage changes.

Solvers must not duplicate any of these responsibilities.

### `SolderSolverBase`

`solver_base.py` defines the plugin interface:

```python
class SolderSolverBase(ABC):
    solver_id: str
    solver_type: SolderSolverType
    available: bool
    unavailable_reason: str | None
    requires_source_paste: bool

    def prepare(
        self,
        request: SolderGenerationRequest,
        work_items: Sequence[SolderWorkItem],
    ) -> SolderSolverRequest:
        ...

    def generate(
        self,
        request: SolderSolverRequest,
    ) -> SolderSolverResult:
        ...
```

`prepare()` runs on the main execution thread. It may inspect the live stage
from `SolderGenerationRequest`, but it must not modify it. Its output must be a
stage-free `SolderSolverRequest` containing serializable values suitable for a
thread, worker process, or external service.

`generate()` receives only the stage-free request. It returns geometry,
coverage, warnings, and backend metadata in `SolderSolverResult`. It must not
read or modify the live stage.

## Contract flow

```text
SolderGenerationRequest
  In-process extension request. May hold the live stage, metadata catalog,
  discovered candidates, applied defect records, and serialization settings.

        | SolderGenerationExtension builds work items
        | solver.prepare(...)
        v

SolderSolverRequest
  Stage-free request. Contains stable work items, solver options, provenance,
  and solver-conditioned numeric or serializable payload data.

        | solver.generate(...)
        v

SolderSolverResult
  Artifact manifest. Contains inline meshes or external USD references,
  per-work-item coverage, warnings, coordinate frames, and transforms.

        | SolderGenerationExtension validates and attaches
        v

SolderPlacementManifest
  Runtime result. Records generated prims, suppressed/restored source paste,
  final coverage, warnings, the edit layer, and serialized output paths.
```

The common data classes and enums live in `data_types.py`.

## Identity and work items

The authoritative component-instance identity is its prim path. A
`ComponentIdentity` also carries its refdes, side, component-definition path,
and optional prototype path.

Each `SolderWorkItem` represents a nominal component placement or one applied
defect variant. Work-item identifiers are stable and readable:

```text
/Board/Components/Top/C1#nominal
/Board/Components/Top/C1#defect-tombstone-000
```

A component can therefore produce more than one work item. Solvers must
preserve `request_id`, `work_item_id`, and component-path correlation in their
results.

Discovery and work construction use `BoardMetadataCatalog`, regardless of
whether its provider decoded embedded USD metadata or a legacy sidecar. In
particular:

- component and prototype records supply identities, process classes, body and
  contact geometry, and reusable package information;
- paste and pad association records supply fitted paste and through-hole sites.

Legacy schema v1.2 sidecars may provide `solder_pin_pad_associations`; the
sidecar provider normalizes them into catalog pad records. Each entry pairs one
instance-space
`solder_contact_prim` with the opposite-side board conductor
`solder_pad_prim`. Its `pin_center` records the detected world-space XY center
in stage units. Multiple entries may reference the same contact mesh when a
single mesh contains disconnected pins. The association is instance-specific;
it must not be stored on the reusable component definition.

Prepare these associations with the board assets before running CDF.
The board-specific sidecar must identify the correct contacts and pads;
CAD-to-USD preparation tooling is not included in this release.

Source-paste prims are the authority for reversible suppression and
pass-through restoration. A solver may use other geometry during preparation,
but it must not independently hide or delete source paste.

## Geometry and transform contract

Solvers return `SolderGeometryArtifact` values containing either:

- Inline `SolderMesh` geometry.
- References to source prims, as used by the pass-through solver.
- An external USD path and optional prim path for a future external solver.

Every artifact declares a coordinate frame and reuse scope. Every mesh carries
a row-major 4x4 transform. The current inline attachment implementation accepts
world-frame artifacts and applies each mesh transform while authoring USD.

Defect transforms are normalized as component-local row-major deltas:

```text
T_delta = inverse(T_nominal_world) @ T_defect_world
```

Do not use prototype identity as an instance identity, and do not return an
uncorrelated collection of mesh files.

## Coverage and fallback

A solver reports one `SolderWorkItemCoverage` record for each work item it
handles. Relevant statuses are:

- `GENERATED`
- `PASSTHROUGH`
- `NO_SOURCE_GEOMETRY`
- `INELIGIBLE`
- `SOLVER_FAILED`
- `PLACEMENT_FAILED`

Preparation exceptions, generation exceptions, invalid results, incomplete
coverage, and ineligible work items fall back independently through
`PassThroughSolderSolver`. A through-hole component with no source paste may
legitimately finish as `NO_SOURCE_GEOMETRY`.

Failures that prevent safe stage mutation are extension-level failures and do
not fall back. Attachment is performed through a replaceable edit layer so a
failed rerun does not leave mixed source and generated solder visible.

## Serialization

`SolderSerializationMode` defines three policies:

- `NONE` changes only the live stage.
- `SOLDER_JOINTS_ONLY` exports the extension-owned solder edit layer.
- `ALL` exports the flattened composed PCBA after defects and solder
  replacement.

A serialization path is required when either export mode is selected. The
source PCBA path remains provenance; the live stage is authoritative during
collection and attachment.

## Registry and solver discovery

`solver_registry.py` owns the in-process registry. Solver IDs must be stable,
lowercase identifiers using letters, digits, and `.`, `_`, or `-` separators.

The package currently registers:

| ID | Type | Implementation | Behavior |
|---|---|---|---|
| `none` | `none` | `PassThroughSolderSolver` | Retains or restores sidecar-listed source paste |
| `procedural` | `procedural` | `ProceduralSolderSolver` | Generates SMT menisci and THT volcano meshes |

The public API can inspect the registry with:

```python
job.operations.get_solder_joint_solvers(mode_filter=None)
```

The result includes solver ID, type, availability, and an unavailability reason
when applicable.

## Adding a solver

Create a `SolderSolverBase` implementation and register one instance before
`job.operations.solder_joints(...)` validates the selected ID:

```python
from collections.abc import Sequence

from pcb_aoi.solder_generation import (
    SolderGenerationRequest,
    SolderSolverBase,
    SolderSolverRequest,
    SolderSolverResult,
    SolderSolverType,
    SolderWorkItem,
    register_solver,
)


class ExampleSolver(SolderSolverBase):
    solver_id = "example"
    solver_type = SolderSolverType.SIMULATED
    available = True
    unavailable_reason = None
    requires_source_paste = False

    def prepare(
        self,
        request: SolderGenerationRequest,
        work_items: Sequence[SolderWorkItem],
    ) -> SolderSolverRequest:
        # Read and normalize any required stage geometry here. Do not author USD.
        return SolderSolverRequest(
            request_id=request.request_id,
            solver_id=self.solver_id,
            source_usd_path=request.source_usd_path,
            root_path=request.root_path,
            work_items=tuple(work_items),
            solver_options=dict(request.solver_options),
            payload={"example_inputs": ()},
        )

    def generate(self, request: SolderSolverRequest) -> SolderSolverResult:
        # Run without stage access and return correlated artifacts and coverage.
        return SolderSolverResult(
            request_id=request.request_id,
            solver_id=self.solver_id,
            artifacts=(),
            coverage=(),
        )


register_solver(ExampleSolver())
```

The abbreviated `generate()` above demonstrates the interface only. A real
solver must report complete, correctly correlated coverage or the extension
will fall back uncovered work items.

Built-in solvers are imported and registered in `__init__.py`. An external
solver package must ensure its registration module is imported before the
public solder declaration is evaluated.

### Simulated solver recipe

A simulated solder solver follows the same boundary regardless of its backend:

1. Implement `SolderSolverBase` and register its stable solver ID and
   `SolderSolverType.SIMULATED` type.
2. In `prepare()`:
   - Read the required pad, termination, and component geometry.
   - Incorporate the current defect state.
   - Normalize geometry, coordinates, and units.
   - Convert the data into inputs understood by the simulation backend.
   - Never modify the USD stage.
3. In `generate()`:
   - Run the simulation backend using the prepared inputs.
   - Convert the simulation output into `SolderGeometryArtifact` objects.
   - Report coverage for every requested work item.
   - Never modify the USD stage.
4. Let `SolderGenerationExtension`:
   - Add generated geometry to USD.
   - Hide or restore source paste.
   - Handle fallback, rollback, and serialization.

The defect state must be incorporated exactly once. A solver can read geometry
from the already-defected stage, or it can apply a recorded defect transform to
copied nominal geometry. It must not do both.

For a Surface Evolver backend, the implementation should be named
`surface_evolver_solver.py`, registered with `solver_id = "surface-evolver"`,
and declared as `SolderSolverType.SIMULATED`. Converting prepared inputs into a
Surface Evolver `.fe` datafile and parsing its output mesh are backend-specific
steps inside `generate()`; they are not part of the common solver contract.

## Procedural implementation

`ProceduralSolderSolver` has two distinct phases:

- `prepare()` routes explicit `through_hole` records through their
  `solder_pin_pad_associations`. Other records use the established SMT
  paste/contact measurements, including contact-foot mesh points touching each
  pad. All measurements come from the already-defected stage and are converted
  into stage-free numeric inputs.
- `generate()` dispatches SMT inputs to `procedural_meniscus_geometry.py` and
  THT inputs to `procedural_tht_geometry.py`, returning per-instance meshes.

`process_class` is the compatibility gate for THT support. A metadata record
without that field follows the legacy SMT path and is never inferred to be
THT. An explicit `through_hole` record with missing or invalid associations
returns `NO_SOURCE_GEOMETRY` and a warning without blocking other components.

`procedural_meniscus_geometry.py` is stage-free. It contains the shared CPU
reference for
the meniscus heightfield, deterministic surface noise, tapered pad edge,
watertight solid closure, top/bottom pad-aware placement, lead-local height and
toe sizing, two-pad wall climb, and covered area-array suppression. The pad-aware
path in `scripts/sdg/editor/editor_solder_fillet_board.py` consumes the same
geometry core. USD material binding and mesh authoring remain outside that
module.

`procedural_tht_stage_analysis.py` resolves each declared pin center to its
contact region and opposite-side conductor geometry during `prepare()`.
`procedural_tht_geometry.py` builds the stage-free annular volcano surface and
optional watertight solid. Neither module authors USD geometry.

Functions whose names begin with `fallback_` are narrowly scoped geometry
containment paths. They run only when otherwise valid pin-pad metadata resolves
to connected or ambiguous contact geometry that extends beyond the associated
pad neighborhood, or when optional pad-contour metadata is unavailable. These
functions isolate a local contact region, substitute a pad-centered throat, or
recover the already-associated pad's contour from its mesh. They do not repair,
reinterpret, or compensate for an incorrect metadata association; invalid
pin-pad metadata must be corrected at its source.

For THT components, only pad associations on the side opposite component
insertion are eligible for solder generation. If those associations are
missing, the solver warns and skips the component instead of silently using
same-side pads that would place solder on the wrong board surface.

The USD reader stores `pcba:pin:crossSection`, `pcba:board:padContour`, and the
associated hole's `holeCenterMm`, `holeDimsMm`, and `holeShape` fields in the
source-neutral metadata catalog. Catalog construction precomputes dimensional
metadata gaps, so preparation checks them without revisiting the USD stage.
Missing dimensional metadata is non-fatal: the result contains one aggregated
warning that identifies the absent fields and states that heavier USD mesh
analysis will be used to infer the solder-joint dimensions.

Supported procedural options include:

- `components` is an exact refdes or component-instance prim-path allowlist.
  Omit it or pass an empty list to generate solder for every discovered component.
- `height_ratio`, `inset_ratio`, and `shift_ratio` compatibility controls.
- `tht_profile_exponent` (default `1.0`, valid range `0.5 < value <= 1.0`)
  raises the intermediate through-hole fillet surface when reduced, while
  preserving the throat height and outer pad footprint. It applies to bath-on
  and bath-off fillets, not the pin tip or SMT geometry. The pad-rim tangent
  stays horizontal. Combine with `height_ratio` to raise the fillet's neck;
  changing the exponent alone does not raise the neck. Existing configurations
  retain the original profile. Keep `tht_pad_edge_inset_mm: 0.0` to reach the
  available pad contour (or the radius fallback when a contour is unavailable).
- `tht_crease_angle_deg` (optional, valid range `0 < value < 180`) switches
  procedural THT meshes to crease-aware vertex normals. Faces meeting above
  the threshold receive separate shading vertices, which preserves the exact
  geometry while making strong cut-face, chamfer, and rim boundaries visible.
  Omitting it preserves the existing smooth-normal behavior.
- `long_edge`, `x_scale`, `platform_start`, and `platform_end`.
- `smoothness_y`, `bump`, `delta_x_decay`, and `pad_decay`.
- `noise_scale`, `noise_octaves`, and `noise_amplitude`.
- `resolution`, `solid`, and `solid_depth_fraction`.
- `wetting_overlap_ratio` and `z_offset_mm`.
- `tht_angular_segments` and `tht_radial_segments` control THT volcano
  tessellation independently. Their smooth defaults are 64 angular segments
  and 16 radial rings. The shared `resolution` option remains a backward-
  compatible angular override when `tht_angular_segments` is omitted.
- `tht_pad_edge_inset_mm` controls the radial clearance between a THT joint
  and its associated pad edge. The default `0.0` fills to the pad boundary,
  including non-circular pad contours.
- `tht_solder_bath` extends each generated THT joint so solder envelops the
  exposed solder-side portion of the associated pin. This is the option's
  primary behavior and models the PCB assembly solder-bath process. The
  envelope covers the outward pin end face, including a chamfered or angled
  cut; use the separate, annotation-only `operations.pin_tip` proxy when that
  face needs its own solder-hidden semantic class. Additional transition rings
  and blended normals smooth the join into the generated volcano at its neck;
  pin geometry on the opposite side of the neck is not coated. It defaults to
  `false`, preserving the existing volcano-only geometry.
- THT meshes author area-weighted vertex normals, smoothing adjacent surface
  facets and the transition from non-circular pin throats without rounding the
  actual pin-contact boundary.
- Surface Evolver meshes author crease-aware area-weighted vertex normals
  (`mesh_normals.py`): the curved free surface shades smooth while the crease
  against the flat wet patches stays sharp (vertices split at the crease, so
  a normal bent at a big planar patch's rim cannot smear shading across it).
- The default THT radial height profile uses a concave quadratic falloff,
  keeping the joint inside the straight pin-to-pad chord instead of producing
  an outward upper bulge.
- `lead_local_sizing` (enabled by default) switches pad placement from the
  package union bounds to the lead foot touching each pad. Its calibrated
  geometry constants remain internal to the solver.

World-space per-instance output is intentional for the current integration.
Component-local geometry and reusable prototype instancing remain future
contract extensions.

## Package map

| File | Responsibility |
|---|---|
| `data_types.py` | Shared requests, work items, artifacts, coverage, manifests, transforms, and enums |
| `metadata_discovery.py` | Source-neutral component eligibility and candidate discovery |
| `solver_base.py` | Abstract solver interface |
| `solver_registry.py` | Solver registration, lookup, filtering, and availability validation |
| `pass_through_solver.py` | Source-paste retention and restoration behavior |
| `procedural_solver.py` | Stage preparation and pad-aware procedural dispatch |
| `procedural_meniscus_geometry.py` | Shared stage-free procedural meniscus geometry and placement math |
| `procedural_tht_stage_analysis.py` | Live-stage adaptation of THT pin-pad associations into numeric sites |
| `procedural_tht_geometry.py` | Stage-free annular THT volcano geometry |
| `solder_generation_extension.py` | Work construction, validation, fallback, USD attachment, rollback, and serialization |
| `__init__.py` | Public exports and built-in solver registration |

## Testing

Run the focused solder tests (the stage-backed module needs `usd-core`):

```powershell
uv run --with-requirements requirements.txt `
  python -m unittest `
    tests.unittests.core.declarative_api.test_solder_planning `
    tests.unittests.usd.declarative_api.test_solder_planning_stage -v
```

Run the complete declarative API suite (core and USD-backed tests):

```powershell
uv run --with-requirements requirements.txt `
  python -m unittest discover -s tests/unittests/core/declarative_api -p "test_*.py"
uv run --with-requirements requirements.txt `
  python -m unittest discover -s tests/unittests/usd/declarative_api -p "test_*.py"
```

Run the solver lane after building the vendored Surface Evolver binary:

```powershell
uv run --with-requirements requirements.txt `
  python scripts/ci/run_unittests.py `
    --lane solver-linux `
    --groups core,solver `
    --report-file test-results/solver-linux.json
```

See `tests/unittests/README.md` for the other group compositions.

The solder tests cover registry behavior, discovery, multiple defect work
items, fallback, stage-free generation, in-memory USD preparation, bottom-side
placement, and watertight mesh topology.

## Current limitations

- Real Isaac/Kit visual comparison between declarative and editor pad-aware
  output is still required.
- Rotated hierarchies, parent transforms, bottom-side boards, and structurally
  defected components require broader production validation.
- Inline attachment currently supports world-frame meshes only.
- Procedural meshes are emitted per instance; prototype instancing and cache
  invalidation are not implemented.
- External-worker timeout, cancellation, artifact retention, and bounded
  parallelism are not implemented.
- Materials: a mesh naming a `material_source_prim_path` (or a job passing
  `solder_material_prim_path`) binds that composed material; every other
  solder mesh -- procedural or simulated -- gets the packaged reflowed-metal
  default (`assets/solder.usda`, referenced under `<output root>/Looks/Solder`
  and flattened into `all` serialization).
- Incremental replacement after defects change is unsupported; reload and
  rerun is the supported lifecycle.
- Surface Evolver and Newton solvers are not registered.
- The legacy YAML renderer intentionally does not support solder generation.
