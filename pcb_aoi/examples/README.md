# PCB-AOI Python examples

Python scripts that build PCB-AOI jobs through `CDFJob`. The versioned YAML
examples, one per scenario, live in
[`../config/examples/`](../config/examples/README.md).

## Python examples

Each script compiles and prints its `JobOrderPlan` by default and renders with
`--execute` when `ISAAC_SIM_PATH` points at an Isaac Sim installation. They
are importable modules (`pcb_aoi.examples.<name>`), and the unit suite compiles
the ones that need no external asset.

| Script | Purpose |
|---|---|
| `allegro_defect_job.py` | CAD-exported board example with its `pcb.components` sidecar, pose defects, and crops. Expects the board export under `_boards/`. |
| `allegro_v7_defects_with_solder.py` | Six SMT defects with procedural solder close-ups on a CAD-exported board. |
| `distributable_board_defects.py` | Six SMT defects with procedural solder on a board that may be redistributed. |
| `environment_sensor_defects.py` | Six SMT defects and component close-ups using only metadata embedded in the composed USD stage; no sidecar, no solder. |
| `panel_golden_bottom.py` | Defect-free full-panel golden render for any panelised export; the scene file and root prim are command-line arguments, so the script carries no board of its own. |

```bash
python pcb_aoi/examples/environment_sensor_defects.py             # compile only
python pcb_aoi/examples/environment_sensor_defects.py --execute   # render (needs ISAAC_SIM_PATH)
python pcb_aoi/examples/panel_golden_bottom.py panel.usda --root-path /World/Panel
```
