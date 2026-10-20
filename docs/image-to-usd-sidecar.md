# The `image_to_usd.json` sidecar

`postprocess.stitch` writes `image_<stitched image stem>_to_usd.json` beside the
stitched PNG. It answers one question: **given a pixel in the stitched image,
where is that point on the board in USD stage space** — and the reverse.

That is what an external consumer needs in order to take an AOI annotation drawn
on the image and put it back on the model. The file is a **contract**, not debug
output. It states its conventions in its own fields so a reader does not have to
infer them from names. Get any convention wrong and the picture still looks
right, while coordinates are off by a row flip or half a pixel.

## Contents

- [Support and stability](#support-and-stability)
- [Conventions you must read before using it](#conventions-you-must-read-before-using-it)
- [Using it](#using-it)
- [Field reference](#field-reference)
- [What it does not cover](#what-it-does-not-cover)
- [Related documentation](#related-documentation)

## Support and stability

The sidecar identity and emit rules are as follows:

| Item | Value |
|---|---|
| Written by | `cdf.postprocess.image_stitching`, on every `postprocess.stitch` |
| Result field | `image_to_usd_path` on the stitch result |
| Path | `image_<image stem>_to_usd.json`, beside the image. The prefix comes first so a consumer filtering on `.../artifact/image` finds the sidecar as well as the image it describes. |
| Schema | `schema_version: "pcb_aoi.image_to_usd/1.0.0"` |
| Emitted when | the layout is *supported* — refer to [What it does not cover](#what-it-does-not-cover) |

`schema_version` is `<name>/<major>.<minor>.<patch>`. **Check the major before
parsing.** A minor bump adds fields and does not move or reinterpret an existing
one; a major bump may. A consumer that does not recognize the major should fail
loudly rather than read fields that happen to share a name.

## Conventions you must read before using it

Four of them, each stated in the file itself as well:

**Pixel origin is top-left, y down** (`pixel_coordinate_system:
"top-left-xy-pixels"`). This is the raster convention PNG, PIL, and NumPy use.
The scanner vendor's proposed example uses a lower-left `[0, 0]` origin, and this file
deliberately does **not** — a consumer expecting y-up must flip the row index
first. Nothing detects that mistake for you; the result is a board that looks
mirrored about its own center line.

**Integer pixel coordinates are edges, not centers**
(`pixel_indexing_convention: "pixel-edge"`). Column 0 is the left edge of the
first pixel, so the *center* of pixel `(col, row)` is at `(col + 0.5, row + 0.5)`.
Using centers where the file means edges puts everything half a pixel off — small
enough to survive review and large enough to matter on a 0402.

**Coordinates are USD stage (world) space, in stage units**
(`coordinate_space: "usd_stage_world"`, `units: "usd_stage_units"`). Not meters
and not millimeters. `stage_meters_per_unit` converts; `stage_up_axis` tells you
which axis is up. A stage whose `metersPerUnit` disagrees with its real geometry
is a known hazard on this project — the sidecar reports what the stage claims.

**The mapping is a plane, not a projection** (`projection: "orthographic"`,
`reference_plane_z`). Every pixel maps to a point at one Z. Refer to
[What it does not cover](#what-it-does-not-cover).

## Using it

Convert in either direction as follows.

Forward — pixel to stage:

```python
import json

sidecar = json.load(open("image_scan_top_to_usd.json"))
assert sidecar["schema_version"].split("/")[1].split(".")[0] == "1"

affine = sidecar["affine_pixel_to_usd"]
ox, oy = affine["origin_xy"]
ux, uy = affine["u_per_px_xy"]
vx, vy = affine["v_per_px_xy"]
z = sidecar["reference_plane_z"]

def pixel_to_usd(px, py):
    """px, py are pixel-EDGE coordinates; add 0.5 each for a pixel center."""
    return (ox + px * ux + py * vx,
            oy + px * uy + py * vy,
            z)
```

Reverse — stage to pixel — is the inverse of the 2×2, which is well conditioned
because `u_per_px` and `v_per_px` are the two scan axes:

```python
det = ux * vy - uy * vx
def usd_to_pixel(x, y):
    dx, dy = x - ox, y - oy
    return ((dx * vy - dy * vx) / det,
            (dy * ux - dx * uy) / det)
```

To get millimeters, multiply a stage-unit distance by
`stage_meters_per_unit * 1000`.

## Field reference

The sidecar fields are as follows:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | `pcb_aoi.image_to_usd/1.0.0` |
| `coordinate_space` | string | `usd_stage_world` |
| `units` | string | `usd_stage_units` |
| `pixel_coordinate_system` | string | `top-left-xy-pixels` |
| `pixel_indexing_convention` | string | `pixel-edge` |
| `projection` | string | `orthographic` |
| `image_size_px` | `[int, int]` | Stitched image width, height. For a `capture.fov` capture this is the size the scanner states for the rectangle (`area / pitch`, rounded once per axis). The stitch crops the rendered mosaic to it, and every other field here describes the cropped image. |
| `camera_side` | string | Which side of the board was scanned |
| `reference_plane_source` | string | How `reference_plane_z` was derived — `substrate_z_range` when the metadata catalog measured the board, otherwise a bbox route |
| `reference_plane_z` | float | Stage Z of the plane every pixel maps onto. **Read `reference_plane_source` before treating this as a board face.** `substrate_z_range` is the PCB surface as measured by the metadata catalog. Bbox routes are an axis-aligned minimum over whatever the scope contains; under a tall part or a long pin they track that part, so they are BOM-dependent. |
| `stage_meters_per_unit` | float | Stage units to meters. This is the scale the capture was framed with (the job override when the job carries one), not necessarily the stage's own declaration. |
| `stage_meters_per_unit_source` | str | `stage_declaration` or `job_override`. A delivered stage can declare a scale its geometry does not use; a board that declares 0.01 and measures 183.9 units would be a 1.8 m PCB |
| `stage_up_axis` | string | `Y` or `Z` |
| `source_usd` | string | The stage the layout was computed from |
| `corners_usd` | object | `top_left` / `top_right` / `bottom_left` / `bottom_right`, each `{pixel_xy, usd_xyz}` |
| `affine_pixel_to_usd` | object | `origin_xy`, plus `world_per_px_x` / `world_per_px_y` (and their original `u_per_px_xy` / `v_per_px_xy` spellings, kept as aliases because two sample packs shipped with them) |
| `authoritative_mapping` | string | Which field wins if the affine and the corners ever disagree. Always `affine_pixel_to_usd`; the corners are that affine evaluated at the four extremes |
| `instances_in_frame` | array or null | Board placement prim paths this image covers, so a stitched panel does not need guessing at. `null` — not `[]` — when the scene carries a single board, so "no cohort" stays distinguishable from "looked and found none" |
| `source_usd_provenance` | object or null | `creator`, `version`, `comment`, `modified_utc` from the stage's root layer, whichever it carries. `source_usd` names a file; two exports a week apart share that path and are different stages |
| `fov` | object | The same capture stated the way a scanner is: `scan_area_mm`, `dpi`, `mm_per_px`, `grid`, `square_pixels`, plus `centered_on` and `center_mm` (world XY in mm when the field of view was stated as corners, `null` when it was centered on the board). Derived from the same layout as the affine, so the two cannot disagree. Absent when the stage scale is unknown |

`corners_usd` and `affine_pixel_to_usd` are two views of the same mapping. The
corners are there so a consumer can sanity-check its own arithmetic against four
known points before trusting the affine on the other few million.

## What it does not cover

Stated here rather than discovered later:

- **One plane.** Every pixel maps to `reference_plane_z`. A tall component's top
  face is not at that Z, so a point picked on the top of a connector maps to
  where it would be *if it were flat on the board*. For component-level work,
  map to the plane and take height from the model, not from the image.
- **Orthographic only.** `compute_pixel_layout` reports a supported layout only
  for an orthographic camera. Under a perspective camera the stitch still runs
  and no sidecar is written, because a flat reference plane would not describe
  it — you would need camera intrinsics instead.
- **Cardinal scan axes only.** A camera yawed off the scan grid's axes is not a
  supported layout and gets no sidecar, for the same reason.
- **Not a lens model.** No distortion, no vignetting. It is the geometric
  placement of tiles on a plane, nothing about the optics that made them.
- **The affine describes the pixels that were written**, not the ideal layout.
  The stitcher rounds each tile's destination rectangle before pasting it, so the
  transform is re-anchored on that rounding. With fractional tile offsets an
  affine taken from the unrounded layout would be up to half a pixel out.

## Related documentation

- Python stitch API: [`cdf/API_SPEC.md`](../cdf/API_SPEC.md) (`postprocess.stitch`)
- Capability matrix: [`cdf/CAPABILITIES.md`](../cdf/CAPABILITIES.md)
