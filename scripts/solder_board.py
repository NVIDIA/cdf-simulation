#!/usr/bin/env python
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

"""Headless solder-board generation over the PRODUCTION extension.

Point at a board (USD + components sidecar), get one self-contained USD with
simulated solder -- optionally with SMT defects on named components. This drives
the same ``SolderGenerationExtension`` the Kit/Isaac pipeline runs
(``prepare -> generate -> attach``); the only difference is it opens the stage
with plain ``pxr.Usd`` instead of ``omni.usd``, so it needs no Kit and no env
vars (the Evolver binary is auto-discovered from the vendored build).

    .venv/bin/python scripts/solder_board.py <sidecar.components.json> \
        --out out.usd
    # with SMT defects on specific components:
    .venv/bin/python scripts/solder_board.py <sidecar.components.json> \
        --out out.usd --defect C10=insufficient --defect C20=excess

``--mode all`` (default) flattens board+solder into one shareable file;
``--mode solder-joints-only`` writes just the solder layer.

Scope: SMT (paste-driven) and THT (plated-hole-driven; tht_spec reads the raw
board layers, no sidecar pin/pad association needed). Pad-driven SMT discovery
still lands with the USD-spec ingestion (Real B). See
pcb_aoi/solder_evolver/docs/inputs.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _load_board(sidecar_path, usd_path=None):
    """Resolve a components sidecar to its board USD.

    By default the USD is a sibling named by the sidecar's ``board_name`` -- a
    convenience, not a rule (production takes the USD path explicitly, as its own
    scene input, decoupled from the sidecar). Pass ``usd_path`` to name it
    directly when the file does not follow that convention. The sidecar is the
    handle, not the directory -- a folder may hold many USD+sidecar pairs."""
    sidecar_path = os.path.abspath(os.path.expanduser(sidecar_path))
    if os.path.isdir(sidecar_path):
        raise SystemExit(
            f"{sidecar_path} is a directory; pass the *.components.json file "
            f"itself (a folder may hold several boards)")
    if not os.path.isfile(sidecar_path):
        raise SystemExit(f"sidecar not found: {sidecar_path}")
    sidecar = json.load(open(sidecar_path))
    if usd_path:
        board_usd = os.path.abspath(os.path.expanduser(usd_path))
        if not os.path.isfile(board_usd):
            raise SystemExit(f"board USD not found: {board_usd}")
    else:
        board_usd = os.path.join(os.path.dirname(sidecar_path),
                                 sidecar["board_name"] + ".usd")
        if not os.path.isfile(board_usd):
            raise SystemExit(
                f"board USD not found next to the sidecar: {board_usd} "
                f"(from board_name={sidecar.get('board_name')!r}). Pass --usd "
                f"to name the board USD explicitly.")
    return board_usd, sidecar, sidecar_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sidecar",
                    help="path to a components sidecar (.json); its board USD "
                         "sits beside it, named by the sidecar's board_name")
    ap.add_argument("--usd", default=None, metavar="PATH",
                    help="board USD path; overrides the <board_name>.usd sibling "
                         "(use when the USD filename does not match board_name)")
    ap.add_argument("--out", required=True, help="output USD path")
    ap.add_argument("--mode", default="all",
                    choices=["all", "solder-joints-only"],
                    help="all = flattened board+solder (shareable); "
                         "solder-joints-only = just the solder layer")
    ap.add_argument("--solver", default="surface_evolver")
    ap.add_argument("--defect", action="append", default=[], metavar="REFDES=TYPE",
                    help="SMT defect on a component, e.g. C10=insufficient "
                         "(repeatable). Types: the registered solder strategies.")
    ap.add_argument("--workers", type=int, default=0,
                    help="cores the solve may use (0 = the machine, minus "
                         "headroom for the OS)")
    ap.add_argument("--only", default="", metavar="SUBSTR",
                    help="dev: restrict to candidate prototypes matching SUBSTR")
    a = ap.parse_args(argv)

    from pxr import Usd
    from pcb_aoi.board_metadata import BoardMetadataRequest, SidecarMetadataProvider
    from pcb_aoi.solder_generation import (
        SolderGenerationExtension,
        SolderGenerationRequest,
        SolderSerializationMode,
        discover_solder_components,
        resolve_solver,
    )

    board_usd, sidecar, sidecar_path = _load_board(a.sidecar, a.usd)
    root_path = sidecar.get("board_prim_path") or "/Board_" + sidecar["board_name"]
    stage = Usd.Stage.Open(board_usd)

    # The sidecar becomes a source-neutral metadata catalog, exactly as the
    # production runtime's sidecar-load handler does.
    catalog = SidecarMetadataProvider().load(
        BoardMetadataRequest(source_identifier=sidecar_path, sidecar_data=sidecar))

    # Paste eligibility is the solver's call (mirrors the production runtime):
    # surface_evolver is paste-driven (SMT), procedural also takes paste-less
    # THT candidates.
    solver = resolve_solver(a.solver)
    candidates = discover_solder_components(
        catalog, require_sample_paste=solver.requires_source_paste)
    if a.only:
        candidates = tuple(c for c in candidates if a.only in c.prototype_id)
        print(f"--only {a.only!r}: {len(candidates)} candidate footprint(s)")

    # REFDES=TYPE -> defect records the extension turns into defect work items.
    refdes_to_path = {i.get("refdes"): i["path"]
                      for i in sidecar.get("component_instances", ())
                      if i.get("refdes")}
    defect_records = []
    for spec in a.defect:
        refdes, _, dtype = spec.partition("=")
        if not dtype or refdes not in refdes_to_path:
            raise SystemExit(f"--defect {spec!r}: need REFDES=TYPE with a known refdes")
        defect_records.append({"path": refdes_to_path[refdes], "defect": dtype})

    req = SolderGenerationRequest(
        request_id="solder_board",
        solver_id=a.solver,
        source_usd_path=board_usd,
        root_path=root_path,
        output_root_path=root_path.rstrip("/") + "/SolderJoints",
        stage=stage,
        board_metadata=catalog,
        candidates=candidates,
        defect_records=tuple(defect_records),
        # `workers` is a surface_evolver option; other solvers validate their
        # option set strictly and would reject it.
        solver_options=({"workers": a.workers}
                        if a.solver == "surface_evolver" else {}),
        serialization=SolderSerializationMode(a.mode),
        serialization_path=os.path.abspath(os.path.expanduser(a.out)),
    )

    ext = SolderGenerationExtension()
    print(f"board {sidecar['board_name']}  ->  {len(candidates)} footprints, "
          f"{len(defect_records)} defect(s)")
    solver_request = ext.prepare(req)                       # discover + collect
    result = ext.generate(req, solver_request)              # solve
    ext.attach(req, solver_request, result)                 # author + serialize

    from collections import Counter
    cov = Counter(c.status.value for c in result.coverage)
    print("coverage:", ", ".join(f"{k}={v}" for k, v in cov.items()))
    for w in result.warnings[:4]:
        print("  WARN", w.splitlines()[0])
    print("wrote", req.serialization_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
