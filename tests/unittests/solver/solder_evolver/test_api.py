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

"""Test the public solder-evolver API and coupon-derived formulations.

The integration tests run the whole method in one call -- seed a blob,
intersect it against mesh geometry, pin the intersection outlines, and
minimize area at fixed volume. Their geometry is prepared from the maintained
ICT and Artix USD-schema coupons through the production solver path.
"""

import os
import sys
import unittest
from pathlib import Path

_REPO = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
from pxr import Usd

import pcb_aoi.solder_evolver as se
from pcb_aoi.board_metadata import BoardMetadataRequest
from pcb_aoi.board_metadata.usd_schema import UsdSchemaMetadataProvider
from pcb_aoi.solder_evolver import backend as ev
from pcb_aoi.solder_generation import (
    SolderGenerationExtension,
    SolderGenerationRequest,
    discover_solder_components,
)
from pcb_aoi.solder_generation.surface_evolver_solver import (
    SurfaceEvolverSolver,
    _as_joint_dict,
)

_RESOURCE_DIR = Path(_REPO) / "tests" / "functional" / "resources"
_SMT_COUPON = _RESOURCE_DIR / "ICT_C0805_C22_coupon.usdc"
_THT_COUPON = _RESOURCE_DIR / "ARTIX_PINHEADER_J13_coupon.usdc"


def _coupon_entry(path: Path, *, require_sample_paste: bool) -> dict:
    """Prepare one coupon component through the production solver path."""
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open coupon: {path}")
    catalog = UsdSchemaMetadataProvider().load(
        BoardMetadataRequest(
            stage=stage,
            board_root="/Board",
            source_identifier=str(path),
        ),
    )
    request = SolderGenerationRequest(
        request_id="/Board/SolderJoints@surface_evolver",
        solver_id="surface_evolver",
        source_usd_path=str(path),
        root_path="/Board",
        output_root_path="/Board/SolderJoints",
        stage=stage,
        board_metadata=catalog,
        candidates=discover_solder_components(
            catalog,
            require_sample_paste=require_sample_paste,
        ),
        defect_records=(),
    )
    prepared = SurfaceEvolverSolver().prepare(
        request,
        SolderGenerationExtension().build_work_items(request),
    )
    entries = prepared.payload["entries"]
    if len(entries) != 1:
        raise RuntimeError(
            f"expected one prepared component from {path}, got {len(entries)}: "
            f"{prepared.payload.get('skipped', ())!r}",
        )
    return entries[0]


class TestApiSurface(unittest.TestCase):
    def test_all_names_resolve(self):
        missing = [n for n in se.__all__ if not hasattr(se, n)]
        self.assertEqual(missing, [], f"__all__ names not bound: {missing}")

    def test_volume_plan_invariant(self):
        """blob_stencil inflates the CUT and target_fraction deflates the SOLVE,
        so their product must be exactly thickness x metal_fraction -- the
        printed solder. If this breaks, footprint and volume have re-coupled."""
        bs, tf = se.volume_plan(0.076, 0.5, 1.6)
        self.assertAlmostEqual(bs * tf, 0.076 * 0.5, places=12)
        bs2, tf2 = se.volume_plan(0.076, 0.5, 2.2)
        self.assertAlmostEqual(bs2 * tf2, 0.076 * 0.5, places=12)
        self.assertGreater(bs2, bs)  # bigger blob_scale, bigger cut

    def test_cluster_rects(self):
        far = [(0, 1, 0, 1), (5, 6, 0, 1)]
        near = [(0, 1, 0, 1), (1.05, 2, 0, 1)]
        self.assertEqual(len(se.cluster_rects(far, gap=0.2)), 2)
        self.assertEqual(len(se.cluster_rects(near, gap=0.2)), 1)
        flattened = [
            item for cluster in se.cluster_rects(near, gap=0.2) for item in cluster
        ]
        self.assertEqual(sorted(flattened), [0, 1])


class TestDefectStrategies(unittest.TestCase):
    def test_unregistered_is_none_not_nominal(self):
        """A defect with no strategy must be INELIGIBLE, never silently nominal
        -- a defect that quietly produced a good joint is mislabelled data."""
        self.assertIsNone(se.resolve_defect("no_such_defect"))

    def test_fill_fractions(self):
        self.assertEqual(se.resolve_defect("nominal").target_volume(1.0, 0), 1.0)
        self.assertAlmostEqual(
            se.resolve_defect("insufficient").target_volume(1.0, 0), 0.5
        )
        self.assertAlmostEqual(se.resolve_defect("excess").target_volume(1.0, 0), 2.6)
        self.assertFalse(se.resolve_defect("missing_solder").target_volume(1.0, 0))

    def test_minimiser_split(self):
        """Newton is documented to stall against the keep-out on starved joints,
        so the starved strategies must stay gradient-only with the collider on,
        and nominal must stay Newton with it off."""
        base = {"blob_scale": 1.25, "skirt": 0.0, "hess_stage": 0, "sdf": True}
        o = se.resolve_defect("nominal").options(dict(base))
        self.assertEqual((o["sdf"], o["hess_stage"]), (False, 10))
        self.assertAlmostEqual(o["skirt"], 0.10)
        o = se.resolve_defect("insufficient").options(dict(base))
        self.assertEqual((o["sdf"], o["hess_stage"]), (True, 0))


@unittest.skipUnless(
    _SMT_COUPON.is_file(),
    "approved SMT coupon fixture is not shipped in the public release",
)
@unittest.skipUnless(ev.evolver_available(), "no Surface Evolver binary")
class TestSolveJointSmt(unittest.TestCase):
    def test_formulation_end_to_end(self):
        entry = _coupon_entry(_SMT_COUPON, require_sample_paste=True)
        d = _as_joint_dict(
            entry["rects"],
            entry["pad_z"],
            entry["mesh_points"],
            entry["mesh_faces"],
            gap=0.0,
            polys=entry.get("polys"),
            pad_polys=entry.get("pad_polys"),
        )
        joint = int(np.asarray(d["joint_id"], int)[0])
        joint_polys = [
            polygon
            for polygon, joint_id in zip(d["polys"], d["joint_id"], strict=True)
            if int(joint_id) == joint and polygon is not None
        ]
        stencil = float(entry["paste_thickness"])
        bs, tf = se.volume_plan(stencil, 0.5, 1.6)
        _meshes, _stats, cut = se.solve_joint(
            d,
            joint,
            kind="smt",
            cut_kw={
                "blob": "aperture_offset",
                "stencil": bs,
                "fill": 1.0,
                "skirt": 0.10,
                "polys": joint_polys,
                "pad_polys": entry.get("pad_polys"),
            },
            solve_kw={"nrefine": 2, "converge": False, "hess_stage": 10},
        )
        # re-target to real solder volume, the way the adapter does
        _meshes, stats, cut = se.solve_joint(
            d,
            joint,
            kind="smt",
            cut_kw={
                "blob": "aperture_offset",
                "stencil": bs,
                "fill": 1.0,
                "skirt": 0.10,
                "polys": joint_polys,
                "pad_polys": entry.get("pad_polys"),
            },
            target_volume=cut["target_volume"] * tf,
            solve_kw={"nrefine": 2, "converge": False, "hess_stage": 10},
        )
        self.assertAlmostEqual(stats["VOL"], cut["target_volume"] * tf, places=6)
        ok = se.check_body(cut["V"], cut["F"], np.asarray(cut["lab"]).astype(str))
        self.assertEqual(ok.get("euler"), 2)  # a closed SMT joint is a sphere


@unittest.skipUnless(
    _THT_COUPON.is_file(),
    "approved THT coupon fixture is not shipped in the public release",
)
@unittest.skipUnless(ev.evolver_available(), "no Surface Evolver binary")
class TestSolveJointTht(unittest.TestCase):
    def test_ring_topology_survives_the_solve(self):
        entry = _coupon_entry(_THT_COUPON, require_sample_paste=False)
        d = entry["tht"]
        joint = int(np.asarray(d["barrels"], float)[0, 3])
        meshes, _stats, _cut = se.solve_joint(
            d,
            joint,
            kind="tht",
            solve_kw={"nrefine": 2, "converge": False, "hess_stage": 10},
        )
        P = np.asarray(meshes[-1].vertices, float)
        T = np.asarray(meshes[-1].triangles, int).reshape(-1, 3)
        genus = (2 - (len(P) - len(T) * 3 // 2 + len(T))) // 2
        self.assertEqual(genus, 1, "a nominal THT joint must ring the pin")


if __name__ == "__main__":
    unittest.main()
