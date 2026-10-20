# SDG USD Input Requirements (draft)

The minimal set a USD file must carry so our pipeline can simulate solder and
generate defect data from it — **no JSON sidecar; everything is USD properties.**

## The provider must supply

1. **Component meshes** — geometry for every placed component.
2. **Solder-contact meshes** — the wettable metal on each component (leads /
   terminations / tabs), cleanly separable from the non-wettable body. Solder
   adheres only here.
3. **Board mesh** — the PCB, with through-holes cut for THT parts.
4. **Solder pads** — the copper landing areas, as a texture map or explicit mesh.
5. **Solder mask (wet map)** — which areas should be wetted with solder after the
   simulation; this defines each joint's footprint.

## Ground rules

- Conveyed entirely as **USD properties** — no JSON sidecar.
- Sufficient for the pipeline to set up each joint's solder solve automatically.

## Our pipeline's job (not the provider's)

Pose/placement defects, solder volume + wetting knobs, and the Surface Evolver
setup are configured by us from the inputs above.

## To iterate

This fixes the **what**, not yet the **how** — concrete USD attribute
names/types come next.

## SMT joint identity: aperture-driven now, should be pad-driven

**What ships today (stopgap).** With no USD ingestion yet, joints are read from
the sidecar's **paste apertures** (`solder_paste_prims`) — one aperture is one
joint (`DEFAULTS["gap"] = 0.0`; only genuinely *overlapping* apertures fuse).
This is correct for the common case (each pad prints one aperture) and fixes the
blobbing that a positive `gap` caused on fine-pitch parts — a TSSOP's ~0.2 mm
pin spacing was being fused into one lump per side.

**Where it is wrong.** A **windowpaned thermal pad** — one physical pad printed
as a grid of separate paste squares (a D2PAK drain tab is ~15) — is *one* joint,
but 1:1 splits it into one blob per square. There is no reliable way to recover
"these N apertures are one pad" from aperture geometry alone: the webbing gap
between windowpane squares can be smaller than, equal to, or larger than a
neighbouring part's pin pitch, so no proximity threshold separates the two cases
in general. The old `gap=0.2` merged the tab correctly but mis-merged pins; 1:1
splits pins correctly but mis-splits the tab. Neither is right because the
**source data lacks pad identity**.

**The real fix (with true USD ingestion, replacing the sidecar).** The draft
USD spec already carries the right unit: a joint is a **`board_pad`**, with a
**`padNumber`** and a pin linkage (`pcba:board:componentPinNumber`) — *"boards
have pads, components have pins … this linkage is what lets the pipeline place
solder."* `board_paste` is a **separate** role that supplies *volume*, not joint
identity. Key joints off `board_pad` and the windowpane case vanishes: the 15
tab squares share one `padNumber` → one joint, and each fine-pitch pin has its
own pad → its own joint. No clustering, no `gap`, no heuristic. "Windowpane"
never appears in the spec precisely because the pad, not the aperture, is the
unit.

| need                | interim source              | contract status |
|---------------------|-----------------------------|-----------------|
| joint identity      | paste apertures, 1:1        | GAP -- should be one `board_pad` (`padNumber`); sidecar carries no pad grouping |
| paste → pad grouping| none (aperture is the unit) | GAP -- windowpane pads mis-split until `board_pad` + pin linkage are ingested |
| paste volume        | aperture area × thickness   | OK as interim; spec keeps `board_paste` for exactly this |

Until then `gap` stays a knob (default 0.0) for a one-off windowpane part, but
it is a workaround for missing pad identity, not a model of the physics.

## Through-hole: what the contract cannot express yet

SMT rides on one datum -- paste carries both footprint and volume. THT has no
paste (wave/selective soldered), so today every input is synthesised from raw
board layers by `tht_spec.from_raw_layers` and tagged with the contract's own
`pcba:prov: "not_yet_available"` convention. The ask is five fields plus one
standing assumption:

| need                | interim source            | contract status |
|---------------------|---------------------------|-----------------|
| joint discovery     | Platings + lead-through   | CLOSED (interim) -- the production solver classifies per instance: paste -> SMT, no paste -> `from_raw_layers` occupied plated holes. The contract-level gap (board_pad/board_hole roles) still stands |
| the barrel          | `Board/Platings` mesh     | GAP (filed); raw plating is the sanctioned interim |
| pad contour         | copper ^ mask opening     | GAP -- spec has `board_pad`+`padContour`, neither reaches `SolderWorkItem` |
| mask opening        | `Board/Soldermask*`       | GAP -- load-bearing: without it a "pad" is a 12.4 mm^2 pour |
| lead tagging        | all geometry under part   | GAP -- `solder_contact_prims` is the right field and is empty |
| solder volume       | `barrel_fill`/`solve_fill`| no datum exists; a stated parameter either way |

One deliberate divergence from the spec's letter: `padContour` is defined as
the mask opening, which is right for SMT -- but a THT hole sits in a mask
relief several times its annular ring, so the wettable area here is
copper INTERSECT opening, capped by `pad_ratio` where the copper is a pour.

