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

"""Unit tests for ``scripts/postprocess/daft.py`` — the Isaac-free DAFT
packager for Cosmos-3 aligned defect pairs.

Pure stdlib (no Pillow / no Isaac); validates the DAFT structure with synthetic
pair records and dummy raw files.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS_POSTPROCESS = Path(__file__).resolve().parents[3] / "scripts" / "postprocess"
if str(_SCRIPTS_POSTPROCESS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_POSTPROCESS))

import daft  # noqa: E402


def _view_rec(view_id, *, clean=True, defect=True, bbox_old=None, bbox_new=None,
              occlusion=0.0):
    rec = {
        "resolution_px": [1920, 1080],
        "depth_format": "exr_float32", "depth_units": "mm",
        "depth_definition": "distance_to_camera",
        "bbox_old_xyxy": bbox_old,
        "bbox_new_xyxy": bbox_new,
        "bbox_union_xyxy": daft_pack_union(bbox_old, bbox_new),
        "occlusion": occlusion,
        "calibration": {
            "K": [[1.0, 0, 960], [0, 1.0, 540], [0, 0, 1]],
            "world_to_camera": [[1, 0, 0, 0]] * 4,
            "camera_to_world": [[1, 0, 0, 0]] * 4,
        },
        "clean_seg_part_pixel_counts": {"body": 10, "lead": 4, "paste": 2},
        "defect_seg_part_pixel_counts": {"body": 8, "lead": 3, "paste": 2},
    }
    rec["clean"] = ({"image_uri": f"images/pair_00000_golden/{view_id}.png",
                     "depth_uri": f"depth/pair_00000_golden/{view_id}.exr"}
                    if clean else {"image_uri": None, "depth_uri": None})
    rec["defect"] = ({"image_uri": f"images/pair_00000_defect/{view_id}.png",
                      "depth_uri": f"depth/pair_00000_defect/{view_id}.exr"}
                     if defect else {"image_uri": None, "depth_uri": None})
    return rec


def daft_pack_union(a, b):
    present = [x for x in (a, b) if x is not None]
    if not present:
        return None
    return [min(x[0] for x in present), min(x[1] for x in present),
            max(x[2] for x in present), max(x[3] for x in present)]


def _record(scene_id="pair_00000", defect_type="tombstoning", missing_view=False):
    views = {}
    for v in ("top_down", "oblique_N_45"):
        if missing_view:
            views[v] = _view_rec(v, bbox_old=[10, 10, 40, 40], bbox_new=None, occlusion=1.0)
        else:
            views[v] = _view_rec(v, bbox_old=[10, 10, 40, 40],
                                 bbox_new=[12, 12, 44, 44], occlusion=0.21)
    return {
        "schema_version": "cosmos3_pair_raw_v0.1",
        "scene_id": scene_id, "capture_mode": "pair",
        "view_ids": ["top_down", "oblique_N_45"],
        "board_plane_global": {"normal_xyz": [0, 0, 1], "origin_xyz_mm": [0, 0, 0],
                               "equation_ax_by_cz_d": [0, 0, 1, 0], "units": "mm"},
        "bbox_3d_global": None if missing_view else {"center_xyz_mm": [1, 2, 3]},
        "segmentation": {"format": "part_colored_rgb", "classes": ["body", "lead", "paste"]},
        "defect": {
            "has_defect": True, "defect_id": "def_0001",
            "defect_type": defect_type, "defect_class_id": "C01",
            "affected_component_id": "R1", "affected_pin_id": None,
            "severity": {"magnitude": None if missing_view else 82.0,
                         "magnitude_field": None if missing_view else "angle_deg",
                         "magnitude_units": None if missing_view else "deg",
                         "bucket": None,
                         "bucket_status": "TODO_placeholder_pending_requester_thresholds"},
            "occlusion_semantic": "replicator_occlusion_ratio_defect_pass",
            "defect_parameters": None if missing_view else {"angle_deg": 82.0},
        },
        "views": views,
        "generation": {"lighting": "dome", "pixel_aligned": True},
    }


class SceneContextTest(unittest.TestCase):
    def test_board_plane_rehomed_into_scene(self):
        ctx = daft.scene_context(_record())
        self.assertEqual(ctx["board_plane_global"]["normal_xyz"], [0, 0, 1])
        self.assertEqual(ctx["scene_id"], "pair_00000")
        self.assertEqual(ctx["view_ids"], ["top_down", "oblique_N_45"])
        self.assertIn("bbox_3d_global", ctx)
        self.assertIn("segmentation", ctx)


class CalibrationContextTest(unittest.TestCase):
    def test_one_record_per_view_with_intrinsics_extrinsics(self):
        ctx = daft.calibration_context(_record())
        self.assertEqual(set(ctx["views"]), {"top_down", "oblique_N_45"})
        for view in ctx["views"].values():
            self.assertIn("K", view)
            self.assertIn("world_to_camera", view)
            self.assertIn("camera_to_world", view)
            self.assertEqual(view["resolution_px"], [1920, 1080])


class ImagesContextTest(unittest.TestCase):
    def test_uris_rehomed_to_daft_tree(self):
        ctx = daft.images_context(_record())
        td = ctx["views"]["top_down"]
        self.assertEqual(td["passes"]["clean"]["rgb"], "raw/rgb/top_down/clean.png")
        self.assertEqual(td["passes"]["clean"]["depth"], "raw/depth/top_down/clean.exr")
        self.assertEqual(td["passes"]["defect"]["rgb"], "raw/rgb/top_down/defect.png")
        self.assertEqual(td["depth_definition"], "distance_to_camera")


class DefectsContextTest(unittest.TestCase):
    def test_bbox_old_new_union_and_occlusion_per_view(self):
        ctx = daft.defects_context(_record())
        td = ctx["views"]["top_down"]
        self.assertEqual(td["bbox_old_xyxy"], [10, 10, 40, 40])
        self.assertEqual(td["bbox_new_xyxy"], [12, 12, 44, 44])
        self.assertEqual(td["bbox_union_xyxy"], [10, 10, 44, 44])
        self.assertIsInstance(td["occlusion"], float)
        self.assertEqual(ctx["defect"]["defect_type"], "tombstoning")

    def test_missing_defect_bbox_new_is_null_not_placeholder(self):
        ctx = daft.defects_context(_record(defect_type="missing", missing_view=True))
        td = ctx["views"]["top_down"]
        self.assertIsNone(td["bbox_new_xyxy"])  # null, not a sentinel string
        self.assertEqual(td["bbox_old_xyxy"], [10, 10, 40, 40])
        self.assertEqual(td["occlusion"], 1.0)

    def test_severity_bucket_is_null_placeholder(self):
        ctx = daft.defects_context(_record())
        self.assertIsNone(ctx["defect"]["severity"]["bucket"])


class RawFilePlanTest(unittest.TestCase):
    def test_maps_source_uris_to_dst_relpaths(self):
        plan = dict(daft.raw_file_plan(_record()))
        self.assertEqual(plan["images/pair_00000_golden/top_down.png"],
                         "raw/rgb/top_down/clean.png")
        self.assertEqual(plan["depth/pair_00000_defect/top_down.exr"],
                         "raw/depth/top_down/defect.exr")
        # 2 views x 2 passes x (rgb + depth) = 8 files.
        self.assertEqual(len(plan), 8)


class ManifestTest(unittest.TestCase):
    def test_manifest_indexes_scenes(self):
        mani = daft.manifest([_record("pair_00000"), _record("pair_00001")])
        self.assertEqual(mani["scene_count"], 2)
        ids = [s["scene_id"] for s in mani["scenes"]]
        self.assertEqual(ids, ["pair_00000", "pair_00001"])
        self.assertEqual(mani["scenes"][0]["defect_type"], "tombstoning")


class PackFilesystemTest(unittest.TestCase):
    def _seed_raw(self, root: Path, record: dict) -> None:
        for src_uri, _dst in daft.raw_file_plan(record):
            p = root / src_uri
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"dummy")

    def test_pack_dir_materializes_full_daft_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw"
            dst = Path(tmp) / "daft"
            src.mkdir()
            rec = _record("pair_00000")
            self._seed_raw(src, rec)
            (src / "pair_00000.pair.json").write_text(json.dumps(rec), encoding="utf-8")

            mani = daft.pack_dir(src, dst, strict=True)

            self.assertEqual(mani["scene_count"], 1)
            self.assertTrue((dst / "manifest.json").exists())
            scene = dst / "scenes" / "pair_00000"
            # raw rgb/depth for both passes + all contextual jsons.
            self.assertTrue((scene / "raw/rgb/top_down/clean.png").exists())
            self.assertTrue((scene / "raw/rgb/top_down/defect.png").exists())
            self.assertTrue((scene / "raw/depth/oblique_N_45/defect.exr").exists())
            for name in ("scene", "calibration", "images", "defects", "generation"):
                ctx_path = scene / "contextual" / f"{name}.json"
                self.assertTrue(ctx_path.exists(), name)
                json.loads(ctx_path.read_text(encoding="utf-8"))  # valid JSON

            # scene.json carries the board plane; calibration.json has per-view K.
            sc = json.loads((scene / "contextual/scene.json").read_text(encoding="utf-8"))
            self.assertEqual(sc["board_plane_global"]["normal_xyz"], [0, 0, 1])
            cal = json.loads((scene / "contextual/calibration.json").read_text(encoding="utf-8"))
            self.assertIn("K", cal["views"]["top_down"])

    def test_pack_injectable_copy_records_calls(self):
        calls = []
        rec = _record("pair_00000")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw"
            dst = Path(tmp) / "daft"
            self._seed_raw(src, rec)
            daft.pack(rec, src, dst, copy=lambda a, b: calls.append((a, b)))
        self.assertEqual(len(calls), 8)  # 2 views x 2 passes x (rgb + depth)

    def test_strict_raises_on_missing_source(self):
        rec = _record("pair_00000")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw"
            src.mkdir()
            with self.assertRaises(FileNotFoundError):
                daft.pack(rec, src, Path(tmp) / "daft", strict=True)


class PathTraversalTest(unittest.TestCase):
    """Record fields are data, not trusted paths."""

    def _record(self, scene_id="pair_00000", image_uri="images/a/top.png"):
        return {
            "scene_id": scene_id,
            "view_ids": ["top_down"],
            "views": {"top_down": {"clean": {"image_uri": image_uri}}},
        }

    def test_scene_id_cannot_escape_the_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                daft.pack(self._record(scene_id="../../etc"),
                          root / "raw", root / "out")

    def test_absolute_scene_id_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                daft.pack(self._record(scene_id="/tmp/anywhere"),
                          root / "raw", root / "out")

    def test_image_uri_cannot_read_outside_the_raw_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                daft.pack(self._record(image_uri="../../../secrets.png"),
                          root / "raw", root / "out")

    def test_an_ordinary_record_still_packs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            (raw / "images" / "a").mkdir(parents=True)
            (raw / "images" / "a" / "top.png").write_bytes(b"x")
            scene_dir = daft.pack(self._record(), raw, root / "out")
            self.assertTrue(scene_dir.is_dir())


class UriGatingTest(unittest.TestCase):
    """images.json must describe the files raw_file_plan actually copies."""

    def _record(self, **prec):
        return {"scene_id": "s", "view_ids": ["top"], "views": {"top": {"clean": prec}}}

    def test_depth_is_null_when_only_an_image_exists(self):
        ctx = daft.images_context(self._record(image_uri="a.png"))
        passes = ctx["views"]["top"]["passes"]["clean"]
        self.assertIsNotNone(passes["rgb"])
        self.assertIsNone(passes["depth"])

    def test_rgb_is_null_when_only_depth_exists(self):
        ctx = daft.images_context(self._record(depth_uri="a.exr"))
        passes = ctx["views"]["top"]["passes"]["clean"]
        self.assertIsNone(passes["rgb"])
        self.assertIsNotNone(passes["depth"])

    def test_context_matches_the_copy_plan(self):
        for prec in ({"image_uri": "a.png"}, {"depth_uri": "a.exr"},
                     {"image_uri": "a.png", "depth_uri": "a.exr"}, {}):
            with self.subTest(prec=sorted(prec)):
                record = self._record(**prec)
                planned = {dst for _src, dst in daft.raw_file_plan(record)}
                passes = daft.images_context(record)["views"]["top"]["passes"]["clean"]
                advertised = {v for v in passes.values() if v}
                self.assertEqual(advertised, planned)


class DuplicateSceneIdTest(unittest.TestCase):
    """Two records with one scene_id would silently lose a scene."""

    def test_pack_dir_refuses_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            for name in ("a", "b"):
                (raw / f"{name}.pair.json").write_text(json.dumps(
                    {"scene_id": "pair_00000", "view_ids": [], "views": {}}),
                    encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                daft.pack_dir(raw, Path(tmp) / "out")
            self.assertIn("duplicate scene_id", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
