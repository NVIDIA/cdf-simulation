# `scripts/crop/`

Crop helpers for rendered RGB and annotation outputs. CDF's
`postprocess.crop()` uses the bbox-tight per-component path:

| Algorithm | Consumer | Knob documentation |
|---|---|---|
| bbox-tight per-Xform crop (`bbox_tight.py` via `pipeline_adapter.py`) | `cdf` `postprocess.crop` | [`cdf/API_SPEC.md`](../../cdf/API_SPEC.md) |

Any `crop:` knob left unspecified falls back to
[`crop_defaults.py`](crop_defaults.py) — the single source of truth for
defaults (`offset`, `min_coverage`, `xform_depth`, and default crop types).
The facade also exposes low-level connected-component and fixed-region
helpers; these are not separate CDF execution workflows.

## Files

| File | Purpose |
|---|---|
| `facade.py` | Single entry point: dispatches a config dict to the right algorithm |
| `pipeline_adapter.py` | Adapts the CDF `postprocess.crop` configuration |
| `bbox_tight.py` / `bbox_grouping.py` | bbox-tight per-component crop + bbox grouping helpers |
| `label_cc.py` | Label connected-component image crop |
| `region.py` | Fixed-region crop of SDG annotator outputs |
| `crop_primitives.py` | Low-level image/bbox helpers shared by the algorithms |
| `crop_defaults.py` | Canonical default knob values (see module docstring) |
