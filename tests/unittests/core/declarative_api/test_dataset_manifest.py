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

"""Test run-scoped dataset manifest collection and serialization."""

from __future__ import annotations

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from typing import get_type_hints

from cdf.errors import ValidationError
from cdf.execution.dataset_manifest import (
    DatasetManifestObserver,
    ManifestROISelection,
)
from cdf.execution.isaac.capture_execution import (
    _discover_frame_artifacts,
    _refine_frame_defect_associations,
)
from pcb_aoi.postprocess.roi_output import format_aoi_roi_output


class DatasetManifestObserverTest(unittest.TestCase):
    """Verify normalized dataset relationships and compatibility output."""

    def test_roi_selection_contract_matches_pin_record_shapes(self):
        annotations = get_type_hints(ManifestROISelection)

        self.assertEqual(annotations["pin_number"], str | None)
        self.assertEqual(annotations["pad_paths"], tuple[str, ...])

    def test_records_defect_identity_severity_frames_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observer = DatasetManifestObserver()
            observer.reset({"random_seed": 7})
            observer.begin_capture_pass(
                "capture.full_board.render",
                output_uri="dataset/full_board",
                local_root=root,
            )
            labels_path = root / "trigger_0000" / "semantic_labels_0000.json"
            labels_path.parent.mkdir()
            labels_path.write_text(json.dumps({"1,2,3,255": {"class": "insufficient"}}))
            defect_ids = observer.observe_defects(
                [
                    {
                        "path": "/Board/Components/R1",
                        "defect": "shift",
                        "shift_params": {
                            "translate": [0.1, -0.2, 0.0],
                            "rotate_z_deg": 4.5,
                        },
                    },
                    {
                        "path": "/Board/Components/U2",
                        "defect": "insufficient",
                        "volume_ratio": 0.35,
                    },
                ],
                lambda path: {
                    "prim_path": path,
                    "refdes": path.rsplit("/", 1)[-1],
                },
            )
            document = observer.record_trigger(
                "capture.full_board.render",
                0,
                {"trigger_idx": 0, "pixel_layout": {"supported": True}},
                [
                    {
                        "frame_id": "frame-0",
                        "frame": 0,
                        "role": "capture",
                        "roi_selection": {
                            "refdes": "U2",
                            "component_path": "/Board/Components/U2",
                            "component_type": "ic",
                            "package_name": "USON-10",
                            "package_type": "USON",
                            "footprint_id": "USON-10_2.5x1mm_P0.5mm",
                            "side": "top",
                            "process_class": "SMT",
                            "process_classes": ["SMT"],
                        },
                        "camera": {
                            "roi_selection": {
                                "refdes": "U2",
                                "component_path": "/Board/Components/U2",
                                "component_type": "ic",
                                "package_name": "USON-10",
                                "package_type": "USON",
                                "footprint_id": "USON-10_2.5x1mm_P0.5mm",
                                "side": "top",
                                "process_class": "SMT",
                                "process_classes": ["SMT"],
                            },
                        },
                        "defect_ids": defect_ids,
                        "artifacts": {
                            "rgb": ["trigger_0000/rgb_0000.png"],
                            "semantic_labels": [
                                "trigger_0000/semantic_labels_0000.json"
                            ],
                        },
                    }
                ],
                [
                    {
                        "artifact_id": "rgb-0",
                        "frame_id": "frame-0",
                        "kind": "rgb",
                        "path": "trigger_0000/rgb_0000.png",
                    },
                    {
                        "artifact_id": "labels-0",
                        "frame_id": "frame-0",
                        "kind": "semantic_labels",
                        "path": "trigger_0000/semantic_labels_0000.json",
                    },
                ],
            )

            self.assertEqual(document["defects"][0]["refdes"], "R1")
            self.assertEqual(
                document["defects"][0]["severity"]["rotate_z_deg"],
                4.5,
            )
            self.assertEqual(document["defects"][1]["type"], "insufficient")
            self.assertEqual(document["defects"][1]["severity"]["volume_ratio"], 0.35)
            self.assertEqual(document["frames"][0]["defect_ids"], defect_ids)
            self.assertEqual(
                document["frames"][0]["roi_selection"]["component_type"],
                "ic",
            )
            self.assertEqual(
                document["frames"][0]["camera"]["roi_selection"]["footprint_id"],
                "USON-10_2.5x1mm_P0.5mm",
            )
            self.assertEqual(
                document["frames"][0]["artifacts"]["rgb"],
                ["trigger_0000/rgb_0000.png"],
            )
            self.assertTrue((root / "trigger_0000" / "metadata.json").is_file())
            index = json.loads((root / "dataset_index.json").read_text())
            self.assertEqual(index["passes"][0]["name"], "full_board")
            self.assertEqual(len(index["artifacts"]), 2)
            self.assertEqual(
                index["artifacts"][0]["pass_id"],
                "capture.full_board.render",
            )
            loaded = observer.serializer.read_json(root / "dataset_index.json")
            self.assertEqual(loaded["dataset_id"], document["dataset_id"])
            hydrated = DatasetManifestObserver.from_index(root / "dataset_index.json")
            self.assertEqual(hydrated.view().dataset_id, document["dataset_id"])
            self.assertEqual(
                hydrated.view().frames("capture.full_board.render")[0]["frame_id"],
                "frame-0",
            )
            labels = json.loads(labels_path.read_text())
            self.assertEqual(labels["1,2,3,255"]["defects"][0]["refdes"], "U2")

    def test_hydration_keeps_the_persisted_provenance(self):
        # index_document() writes the provenance record, so from_index has to
        # read it back. It did not, and a hydrate-then-rewrite round trip
        # silently replaced a real record with null.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observer = DatasetManifestObserver()
            observer.begin_capture_pass(
                "capture.rgb.render",
                output_uri="rgb",
                local_root=root,
            )
            observer.record_trigger("capture.rgb.render", 0, {"trigger_idx": 0}, [], [])
            observer.record_provenance(
                {"fingerprint_version": "1.1", "phase": "complete"}
            )

            hydrated = DatasetManifestObserver.from_index(root / "dataset_index.json")

            self.assertEqual(
                hydrated.index_document()["provenance"],
                {"fingerprint_version": "1.1", "phase": "complete"},
            )

    def test_hydrates_sibling_passes_from_copied_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observer = DatasetManifestObserver()
            for pass_id, pass_name in (
                ("capture.rgb.render", "rgb"),
                ("capture.roi.render", "roi"),
            ):
                pass_root = root / pass_name
                observer.begin_capture_pass(
                    pass_id,
                    output_uri=pass_name,
                    local_root=pass_root,
                )
                observer.record_trigger(
                    pass_id,
                    0,
                    {"trigger_idx": 0},
                    [],
                    [],
                )

            hydrated = DatasetManifestObserver.from_index(
                root / "rgb" / "dataset_index.json"
            )

            self.assertEqual(
                hydrated.view().capture_pass("capture.rgb.render")["local_root"],
                str(root / "rgb"),
            )
            self.assertEqual(
                hydrated.view().capture_pass("capture.roi.render")["local_root"],
                str(root / "roi"),
            )

    def test_rejects_non_object_manifest_collection_entries(self):
        observer = DatasetManifestObserver()
        observer.begin_capture_pass(
            "capture.render",
            output_uri="output",
            local_root="output",
        )

        with self.assertRaisesRegex(
            ValidationError,
            "frames entries must be objects",
        ):
            observer.record_trigger(
                "capture.render",
                0,
                {"trigger_idx": 0},
                [42],
                [],
            )

    def test_missing_reference_and_defective_frames_share_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = DatasetManifestObserver()
            observer.begin_capture_pass(
                "capture.render",
                output_uri=directory,
                local_root=directory,
            )
            defect_ids = observer.observe_defects(
                [{"path": "/Board/Components/C1", "defect": "missing"}],
                lambda path: {"prim_path": path, "refdes": "C1"},
            )
            document = observer.record_trigger(
                "capture.render",
                0,
                {"trigger_idx": 0},
                [
                    {
                        "frame_id": "reference",
                        "frame": 0,
                        "role": "reference",
                        "pair_id": "pair-0",
                        "defect_ids": [],
                        "artifacts": {},
                    },
                    {
                        "frame_id": "defective",
                        "frame": 0,
                        "role": "defective",
                        "pair_id": "pair-0",
                        "defect_ids": defect_ids,
                        "artifacts": {},
                    },
                ],
                [],
            )

            self.assertEqual(document["frames"][0]["pair_id"], "pair-0")
            self.assertEqual(document["frames"][1]["pair_id"], "pair-0")
            self.assertEqual(
                document["defects"][0]["visible_in_roles"],
                ["defective"],
            )

    def test_reconciles_writer_files_and_visible_defects_to_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trigger = root / "trigger_0000"
            trigger.mkdir()
            (trigger / "rgb_0000.png").touch()
            (trigger / "bounding_box_2d_tight_prim_paths_0000.json").write_text(
                json.dumps(["/Board/Components/R1/Body"])
            )

            frames = [
                {
                    "frame_id": "frame-0",
                    "frame": 0,
                    "role": "capture",
                    "defect_ids": ["visible", "not-visible"],
                    "artifacts": {},
                }
            ]
            artifacts = _discover_frame_artifacts(root, trigger, frames, [])
            _refine_frame_defect_associations(
                root,
                frames,
                {
                    "visible": {
                        "type": "shift",
                        "usd_path": "/Board/Components/R1",
                    },
                    "not-visible": {
                        "type": "shift",
                        "usd_path": "/Elsewhere",
                    },
                },
            )

            self.assertEqual(
                frames[0]["artifacts"]["rgb"],
                ["trigger_0000/rgb_0000.png"],
            )
            self.assertEqual(frames[0]["defect_ids"], ["visible"])
            self.assertEqual(
                frames[0]["defect_association"],
                "bounding_box_2d_tight",
            )
            self.assertEqual(len(artifacts), 2)


def _calibration(**overrides: object) -> dict[str, object]:
    """Return a well-formed calibration record, with optional field overrides."""
    record: dict[str, object] = {
        "projection": "orthographic",
        "camera_path": "/World/camera_light/Camera",
        "resolution_px": [900, 900],
        "horizontal_aperture_mm": 2.8,
        "vertical_aperture_mm": 2.8,
        "mm_per_px": [0.003111, 0.003111],
        "intrinsic_matrix": [
            [321.4, 0.0, 450.0],
            [0.0, -321.4, 450.0],
            [0.0, 0.0, 1.0],
        ],
        "world_to_camera": [[1.0, 0.0, 0.0, 0.0]] * 4,
        "camera_to_world": [[1.0, 0.0, 0.0, 0.0]] * 4,
        "position": [0.0, 0.0, 10.0],
        "stage_meters_per_unit": 0.01,
    }
    record.update(overrides)
    return record


class CameraCalibrationManifestTest(unittest.TestCase):
    """The calibration is part of the frame contract, not a loose nested dict."""

    def _record(self, observer, root, calibration):
        observer.begin_capture_pass(
            "capture.render",
            output_uri="dataset/render",
            local_root=root,
        )
        return observer.record_trigger(
            "capture.render",
            0,
            {"trigger_idx": 0},
            [
                {
                    "frame_id": "frame-0",
                    "frame": 0,
                    "trigger": 0,
                    "role": "capture",
                    "camera": {
                        "position": [0.0, 0.0, 10.0],
                        "calibration": calibration,
                    },
                    "defect_ids": [],
                },
            ],
            [
                {
                    "artifact_id": "artifact:capture:trigger:0000:camera_calibration",
                    "kind": "camera_calibration",
                    "path": "calibration.json",
                    "role": "capture",
                    "trigger_idx": 0,
                },
            ],
        )

    def test_calibration_survives_write_view_rewrite_and_hydration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observer = DatasetManifestObserver()
            observer.reset({"random_seed": 7})
            self._record(observer, root, _calibration())

            # 1. reachable through the view
            frames = observer.view().frames("capture.render")
            self.assertEqual(
                frames[0]["camera"]["calibration"]["intrinsic_matrix"][0][2],
                450.0,
            )

            # 2. and it actually reached the file on disk
            persisted = json.loads(
                (root / "trigger_0000" / "metadata.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                persisted["frames"][0]["camera"]["calibration"]["camera_path"],
                "/World/camera_light/Camera",
            )

            # 3. the pass-level sidecar is a discoverable artifact. It has no
            #    frame number in its name, so _discover_frame_artifacts would
            #    never have found it -- it is only here because it was recorded.
            sidecars = observer.view().artifacts(
                "capture.render",
                kind="camera_calibration",
            )
            self.assertEqual([item["path"] for item in sidecars], ["calibration.json"])

            # 4. a postprocessor rewriting the pass must not drop it
            observer.rewrite_trigger_documents("capture.render")
            rewritten = json.loads(
                (root / "trigger_0000" / "metadata.json").read_text(encoding="utf-8"),
            )
            self.assertIn("calibration", rewritten["frames"][0]["camera"])

            # 5. nor may a reader that only has the files on disk
            hydrated = DatasetManifestObserver.from_index(root / "dataset_index.json")
            hydrated_frames = hydrated.view().frames("capture.render")
            self.assertEqual(
                hydrated_frames[0]["camera"]["calibration"]["mm_per_px"],
                [0.003111, 0.003111],
            )

    def test_normalizes_numbers_that_are_not_plain_floats(self):
        """USD and numpy scalars must be coerced before they reach the file."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observer = DatasetManifestObserver()
            observer.reset({"random_seed": 7})
            self._record(
                observer,
                root,
                _calibration(stage_meters_per_unit=Fraction(1, 100)),
            )

            persisted = json.loads(
                (root / "trigger_0000" / "metadata.json").read_text(encoding="utf-8"),
            )
            value = persisted["frames"][0]["camera"]["calibration"][
                "stage_meters_per_unit"
            ]
            self.assertIsInstance(value, float)
            self.assertAlmostEqual(value, 0.01)

    def test_rejects_a_matrix_that_is_not_the_shape_it_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = DatasetManifestObserver()
            observer.reset({"random_seed": 7})
            with self.assertRaisesRegex(ValidationError, "3x3 numeric matrix"):
                self._record(
                    observer,
                    Path(directory),
                    _calibration(intrinsic_matrix=[[1.0, 0.0], [0.0, 1.0]]),
                )

    def test_an_unavailable_calibration_is_a_valid_record(self):
        """A frame that rendered is kept even when its calibration failed."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observer = DatasetManifestObserver()
            observer.reset({"random_seed": 7})
            self._record(observer, root, {"unavailable": "ValueError: no camera"})

            frames = observer.view().frames("capture.render")
            self.assertEqual(
                frames[0]["camera"]["calibration"]["unavailable"],
                "ValueError: no camera",
            )


class RoiOutputManifestTest(unittest.TestCase):
    """Verify ROI formatting cannot overwrite rich capture metadata."""

    def test_preserves_metadata_and_reports_derived_label_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb_trigger = root / "rgb" / "trigger_0000"
            roi_trigger = root / "roi" / "trigger_0000"
            rgb_trigger.mkdir(parents=True)
            roi_trigger.mkdir(parents=True)
            original = {"schema_name": "cdf.dataset_manifest", "frames": []}
            for trigger in (rgb_trigger, roi_trigger):
                (trigger / "metadata.json").write_text(json.dumps(original))
            (roi_trigger / "semantic_segmentation_labels_x0_y0.json").write_text(
                json.dumps({"0,0,0,255": {"class": "BACKGROUND"}})
            )
            metadata = [{"trigger_idx": 0, "pixel_layout": {"supported": True}}]

            result = format_aoi_roi_output(
                root / "rgb",
                root / "roi",
                metadata,
                metadata,
            )

            self.assertEqual(
                json.loads((rgb_trigger / "metadata.json").read_text()), original
            )
            self.assertEqual(
                json.loads((roi_trigger / "metadata.json").read_text()), original
            )
            self.assertEqual(result["artifacts"][0]["kind"], "semantic_labels")


if __name__ == "__main__":
    unittest.main()
