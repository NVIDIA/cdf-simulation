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

"""Tests for joint-ROI identity parsing and semantic label formatting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pcb_aoi.board_metadata import (
    ComponentRecord,
    HoleRecord,
    MetadataOrigin,
    MetadataSource,
    PadRecord,
)
from pcb_aoi.postprocess.joint_rois import (
    _component_metadata_by_ref,
    _parse_component_key,
    _stitched_output_name,
    build_pad_joint_keys,
    generate_joint_rois,
)
from pcb_aoi.postprocess.roi_output import _merge_semantic_labels


class TestJointRois(unittest.TestCase):
    """Cover physical leads and their annotation-only pin-tip proxies."""

    def test_parse_physical_component_path(self) -> None:
        """Parse a physical lead directly below the component scope."""
        path = "/World/Carrier/StepRepeats/SR_0_0/Components/J1/pkg_lead_7"

        self.assertEqual(_parse_component_key(path), ("SR_0_0", "J1", "7"))

    def test_parse_pin_tip_proxy_path(self) -> None:
        """Parse a lead proxy below the intermediate PinTips scope."""
        path = (
            "/World/Carrier/StepRepeats/SR_0_0/SegmentationProxies/"
            "PinTips/Components/J1/pkg_lead_7"
        )

        self.assertEqual(_parse_component_key(path), ("SR_0_0", "J1", "7"))

    def test_stitched_output_names_avoid_trigger_collisions(self) -> None:
        """Use a simple name for one trigger and indexed names for several."""
        self.assertEqual(_stitched_output_name(0, 1), "joint_rois.json")
        self.assertEqual(
            _stitched_output_name(3, 4),
            "joint_rois_trigger_0003.json",
        )

    def test_merged_labels_retain_optional_pin_tip(self) -> None:
        """Keep the optional pin-tip class in the formatted label table."""
        labels = {
            "(0, 0, 0, 0)": {"class": "BACKGROUND"},
            "(0, 255, 255, 255)": {"class": "pin_tip"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic_segmentation_labels_x0_y0.json"
            path.write_text(json.dumps(labels), encoding="utf-8")

            merged = _merge_semantic_labels(Path(directory), set())

        self.assertEqual(merged, labels)

    def test_component_metadata_prefers_authored_type(self) -> None:
        """Prefer authored component type and retain secondary grouping fields."""
        source = MetadataSource(MetadataOrigin.USD, "board.usda")
        component = ComponentRecord(
            board_root="/Board",
            refdes="R1",
            prim_path="/Board/Components/R1",
            source=source,
            component_type="RESISTOR",
            family="PASSIVE",
            package_type="THT_AXIAL",
            package_name="AXIAL_RESISTOR",
            footprint_id="R_AXIAL_10MM",
            process_class="through_hole",
        )

        metadata = _component_metadata_by_ref((component,))

        self.assertEqual(
            metadata["R1"],
            {
                "component_type": "RESISTOR",
                "component_type_source": "component_type",
                "package_type": "THT_AXIAL",
                "package_name": "AXIAL_RESISTOR",
                "footprint_id": "R_AXIAL_10MM",
                "process_class": "through_hole",
            },
        )

    def test_component_metadata_falls_back_to_footprint(self) -> None:
        """Use the footprint for grouping when semantic type fields are unavailable."""
        source = MetadataSource(MetadataOrigin.USD, "board.usda")
        component = ComponentRecord(
            board_root="/Board",
            refdes="R1",
            prim_path="/Board/Components/R1",
            source=source,
            package_type="not_yet_available",
            package_name="R_AXIAL_10MM",
            footprint_id="R_AXIAL_10MM",
        )

        metadata = _component_metadata_by_ref((component,))

        self.assertEqual(metadata["R1"]["component_type"], "R_AXIAL_10MM")
        self.assertEqual(metadata["R1"]["component_type_source"], "footprint_id")
        self.assertIsNone(metadata["R1"]["package_type"])
        self.assertEqual(metadata["R1"]["package_name"], "R_AXIAL_10MM")

    def test_component_metadata_omits_conflicting_refdes(self) -> None:
        """Avoid assigning one board's metadata to an ambiguous repeated refdes."""
        source = MetadataSource(MetadataOrigin.USD, "panel.usda")
        components = (
            ComponentRecord(
                board_root="/Panel/BoardA",
                refdes="R1",
                prim_path="/Panel/BoardA/Components/R1",
                source=source,
                component_type="RESISTOR",
            ),
            ComponentRecord(
                board_root="/Panel/BoardB",
                refdes="R1",
                prim_path="/Panel/BoardB/Components/R1",
                source=source,
                component_type="THERMISTOR",
            ),
        )

        metadata = _component_metadata_by_ref(components)

        self.assertNotIn("R1", metadata)

    def test_generate_joint_rois_writes_component_and_pin_metadata(self) -> None:
        """Write explicit pin and component classification fields for every ROI."""
        bbox_dtype = np.dtype(
            [
                ("x_min", np.float32),
                ("y_min", np.float32),
                ("x_max", np.float32),
                ("y_max", np.float32),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            roi_root = output_root / "roi"
            rgb_root = output_root / "rgb"
            trigger_dir = roi_root / "trigger_0000"
            trigger_dir.mkdir(parents=True)
            np.save(
                trigger_dir / "bounding_box_2d_tight_x0_y0.npy",
                np.array(
                    [(1.0, 2.0, 11.0, 12.0), (2.0, 3.0, 10.0, 11.0)],
                    dtype=bbox_dtype,
                ),
            )
            paths = [
                "/World/Carrier/StepRepeats/SR_0_0/Components/R1/pkg_lead_7",
                (
                    "/World/Carrier/StepRepeats/SR_0_0/SegmentationProxies/"
                    "PinTips/Components/R1/pkg_lead_7"
                ),
            ]
            (trigger_dir / "bounding_box_2d_tight_prim_paths_x0_y0.json").write_text(
                json.dumps(paths),
                encoding="utf-8",
            )
            trigger_metadata = [
                {
                    "trigger_idx": 0,
                    "pixel_layout": {
                        "tiles": [
                            {
                                "x_idx": 0,
                                "y_idx": 0,
                                "dst_rect_xywh_px": [100, 200, 20, 20],
                            },
                        ],
                    },
                },
            ]

            result = generate_joint_rois(
                roi_root,
                trigger_metadata,
                output_roots_by_pass={
                    "full_board": rgb_root,
                    "full_board_roi": roi_root,
                },
                instance_label_by_segment={"SR_0_0": "0"},
                component_metadata={
                    "R1": {
                        "component_type": "RESISTOR",
                        "component_type_source": "component_type",
                        "package_type": "THT_AXIAL",
                        "package_name": "AXIAL_RESISTOR",
                        "footprint_id": "R_AXIAL_10MM",
                        "process_class": "through_hole",
                    },
                },
            )

            output = json.loads((trigger_dir / "joint_rois.json").read_text())
            stitched_output = json.loads(
                (roi_root / "stitched" / "joint_rois.json").read_text()
            )
            rgb_trigger_output = json.loads(
                (rgb_root / "trigger_0000" / "joint_rois.json").read_text()
            )
            rgb_stitched_output = json.loads(
                (rgb_root / "stitched" / "joint_rois.json").read_text()
            )

            generate_joint_rois(
                roi_root,
                trigger_metadata,
                instance_label_by_segment={"SR_0_0": "0"},
            )
            output_without_metadata = json.loads(
                (trigger_dir / "joint_rois.json").read_text()
            )

        self.assertEqual(output["schema_version"], "1.0")
        self.assertEqual(
            output["items"],
            [
                {
                    "joint_id": "0/R1/7",
                    "board_instance": "0",
                    "component_ref": "R1",
                    "pin_number": "7",
                    "component_type": "RESISTOR",
                    "component_type_source": "component_type",
                    "package_type": "THT_AXIAL",
                    "package_name": "AXIAL_RESISTOR",
                    "footprint_id": "R_AXIAL_10MM",
                    "process_class": "through_hole",
                    "source_prim_paths": [
                        "/World/Carrier/StepRepeats/SR_0_0/Components/R1/pkg_lead_7",
                        (
                            "/World/Carrier/StepRepeats/SR_0_0/SegmentationProxies/"
                            "PinTips/Components/R1/pkg_lead_7"
                        ),
                    ],
                    "bbox_xyxy": [101, 202, 111, 212],
                    "pad_prim_paths": [],
                    "joint_bbox_xyxy": [101, 202, 111, 212],
                    "joint_bbox_source": "pin_only",
                },
            ],
        )
        self.assertEqual(stitched_output, output)
        self.assertEqual(rgb_trigger_output, output)
        self.assertEqual(rgb_stitched_output, output)
        self.assertEqual(
            [
                (artifact["capture_pass"], artifact["kind"])
                for artifact in result["artifacts"]
            ],
            [
                ("full_board", "joint_rois"),
                ("full_board", "joint_rois_stitched"),
                ("full_board_roi", "joint_rois"),
                ("full_board_roi", "joint_rois_stitched"),
            ],
        )
        self.assertEqual(
            {
                key: output_without_metadata["items"][0][key]
                for key in (
                    "component_type",
                    "component_type_source",
                    "package_type",
                    "package_name",
                    "footprint_id",
                    "process_class",
                )
            },
            {
                "component_type": None,
                "component_type_source": None,
                "package_type": None,
                "package_name": None,
                "footprint_id": None,
                "process_class": None,
            },
        )


_BBOX_DTYPE = np.dtype(
    [
        ("semanticId", np.uint32),
        ("x_min", np.int32),
        ("y_min", np.int32),
        ("x_max", np.int32),
        ("y_max", np.int32),
    ],
)
_PANEL = "/World/Panel/StepRepeats"


def _lead(instance: str, ref: str, lead: str) -> str:
    return f"{_PANEL}/{instance}/SegmentationProxies/PinTips/Components/{ref}/{lead}"


def _pad(instance: str, name: str) -> str:
    return f"{_PANEL}/{instance}/Layers/Bottom/SignalLayer_0/{name}"


class TestJointBoxWithPads(unittest.TestCase):
    """Cover ``joint_bbox_xyxy``: the pin box widened by its stated pads."""

    def _run(
        self,
        tiles: dict[tuple[int, int], tuple[list[float], list[tuple[str, tuple]]]],
        pad_joint_keys: dict[str, tuple[str, str]] | None,
        instance_labels: dict[str, str] | None = None,
    ) -> tuple[dict[str, dict], dict[str, object]]:
        """Write per-tile box files, run the generator, return items by id."""
        with tempfile.TemporaryDirectory() as directory:
            roi_root = Path(directory) / "roi"
            trigger_dir = roi_root / "trigger_0000"
            trigger_dir.mkdir(parents=True)
            layout = []
            for (x_idx, y_idx), (rect, entries) in tiles.items():
                layout.append(
                    {"x_idx": x_idx, "y_idx": y_idx, "dst_rect_xywh_px": rect},
                )
                np.save(
                    trigger_dir / f"bounding_box_2d_tight_x{x_idx}_y{y_idx}.npy",
                    np.array([(1, *box) for _, box in entries], dtype=_BBOX_DTYPE),
                )
                (
                    trigger_dir
                    / f"bounding_box_2d_tight_prim_paths_x{x_idx}_y{y_idx}.json"
                ).write_text(
                    json.dumps([path for path, _ in entries]),
                    encoding="utf-8",
                )
            result = generate_joint_rois(
                roi_root,
                [{"trigger_idx": 0, "pixel_layout": {"tiles": layout}}],
                instance_label_by_segment=instance_labels,
                pad_joint_keys=pad_joint_keys,
            )
            document = json.loads((trigger_dir / "joint_rois.json").read_text())
        return {item["joint_id"]: item for item in document["items"]}, result

    def test_pin_with_one_pad(self) -> None:
        """Union the pin box with its single pad; keep bbox_xyxy as the pin."""
        items, result = self._run(
            {
                (0, 0): (
                    [100, 200, 50, 50],
                    [
                        (_lead("SR_0_0", "J1", "pkg_lead_1"), (10, 10, 14, 14)),
                        (_pad("SR_0_0", "Pad_7"), (6, 5, 18, 19)),
                    ],
                ),
            },
            {_pad("SR_0_0", "Pad_7"): ("J1", "1")},
        )

        item = items["SR_0_0/J1/1"]
        self.assertEqual(item["bbox_xyxy"], [110, 210, 114, 214])
        self.assertEqual(item["joint_bbox_xyxy"], [106, 205, 118, 219])
        self.assertEqual(item["joint_bbox_source"], "pin_and_pads")
        self.assertEqual(item["pad_prim_paths"], [_pad("SR_0_0", "Pad_7")])
        self.assertEqual(
            item["source_prim_paths"],
            [_lead("SR_0_0", "J1", "pkg_lead_1")],
        )
        self.assertEqual(result["pin_only_joint_count"], 0)

    def test_pad_split_across_tiles(self) -> None:
        """Union the pad's partial boxes from two tiles in board coordinates."""
        pad = _pad("SR_0_0", "Pad_7")
        items, _ = self._run(
            {
                (0, 0): (
                    [0, 0, 100, 100],
                    [
                        (_lead("SR_0_0", "J1", "pkg_lead_1"), (94, 40, 99, 46)),
                        (pad, (88, 34, 100, 52)),
                    ],
                ),
                (1, 0): (
                    [100, 0, 100, 100],
                    [(pad, (0, 34, 6, 52))],
                ),
            },
            {pad: ("J1", "1")},
        )

        item = items["SR_0_0/J1/1"]
        self.assertEqual(item["bbox_xyxy"], [94, 40, 99, 46])
        self.assertEqual(item["joint_bbox_xyxy"], [88, 34, 106, 52])
        self.assertEqual(item["pad_prim_paths"], [pad])

    def test_pin_without_pad_is_flagged(self) -> None:
        """Keep the pin box and flag a joint whose pad is absent from the pass."""
        items, result = self._run(
            {
                (0, 0): (
                    [0, 0, 50, 50],
                    [
                        (_lead("SR_0_0", "J1", "pkg_lead_1"), (10, 10, 14, 14)),
                        (_lead("SR_0_0", "J1", "pkg_lead_2"), (30, 10, 34, 14)),
                        (_pad("SR_0_0", "Pad_7"), (6, 6, 18, 18)),
                    ],
                ),
            },
            {
                _pad("SR_0_0", "Pad_7"): ("J1", "1"),
                _pad("SR_0_0", "Pad_8"): ("J1", "2"),
            },
        )

        item = items["SR_0_0/J1/2"]
        self.assertEqual(item["joint_bbox_xyxy"], item["bbox_xyxy"])
        self.assertEqual(item["joint_bbox_source"], "pin_only")
        self.assertEqual(item["pad_prim_paths"], [])
        self.assertEqual(items["SR_0_0/J1/1"]["joint_bbox_source"], "pin_and_pads")
        self.assertEqual(result["pin_only_joint_count"], 1)

    def test_pads_stay_with_their_board_instance(self) -> None:
        """Attribute each placement's pad to the same pin on that placement only."""
        board = "/World/_class_Board"
        keys = build_pad_joint_keys(
            (
                PadRecord(
                    board_root=board,
                    prim_path=f"{board}/Layers/Bottom/SignalLayer_0/Pad_7",
                    source=MetadataSource(MetadataOrigin.USD, "panel.usda"),
                    component_path=f"{board}/Components/J1",
                    terminal_number="1",
                    contact_prim_path=f"{board}/Components/J1/pkg_lead_1",
                ),
            ),
            source_root=board,
            instance_paths=(f"{_PANEL}/SR_0_0", f"{_PANEL}/SR_0_1"),
        )
        items, _ = self._run(
            {
                (0, 0): (
                    [0, 0, 100, 100],
                    [
                        (_lead("SR_0_0", "J1", "pkg_lead_1"), (10, 10, 14, 14)),
                        (_pad("SR_0_0", "Pad_7"), (6, 6, 18, 18)),
                        (_lead("SR_0_1", "J1", "pkg_lead_1"), (60, 10, 64, 14)),
                        (_pad("SR_0_1", "Pad_7"), (57, 7, 67, 17)),
                    ],
                ),
            },
            keys,
            instance_labels={"SR_0_0": "0", "SR_0_1": "1"},
        )

        self.assertEqual(items["0/J1/1"]["joint_bbox_xyxy"], [6, 6, 18, 18])
        self.assertEqual(items["1/J1/1"]["joint_bbox_xyxy"], [57, 7, 67, 17])
        self.assertEqual(
            items["1/J1/1"]["pad_prim_paths"],
            [_pad("SR_0_1", "Pad_7")],
        )

    def test_pad_without_visible_pin_creates_no_joint(self) -> None:
        """Never create a joint from a pad box alone."""
        items, result = self._run(
            {
                (0, 0): (
                    [0, 0, 50, 50],
                    [(_pad("SR_0_0", "Pad_7"), (6, 6, 18, 18))],
                ),
            },
            {_pad("SR_0_0", "Pad_7"): ("J1", "1")},
        )

        self.assertEqual(items, {})
        self.assertEqual(result["joint_count"], 0)

    def test_build_keys_prefer_contact_lead_and_drop_conflicts(self) -> None:
        """Use the contact lead name, fall back to the number, refuse conflicts."""
        source = MetadataSource(MetadataOrigin.USD, "board.usda")
        records = (
            PadRecord(
                board_root="/B",
                prim_path="/B/Layers/Pad_1",
                source=source,
                component_path="/B/Components/U1",
                terminal_number="A1",
                contact_prim_path="/B/Components/U1/pkg_lead_5",
            ),
            PadRecord(
                board_root="/B",
                prim_path="/B/Layers/Pad_2",
                source=source,
                component_path="/B/Components/U1",
                terminal_number="6",
            ),
            PadRecord(board_root="/B", prim_path="/B/Layers/Pad_3", source=source),
            PadRecord(
                board_root="/B",
                prim_path="/B/Layers/Pad_4",
                source=source,
                component_path="/B/Components/U1",
            ),
            HoleRecord(
                board_root="/B",
                prim_path="/B/Layers/Pad_2",
                source=source,
                component_path="/B/Components/U2",
                terminal_number="6",
            ),
        )

        keys = build_pad_joint_keys(records)

        self.assertEqual(keys, {"/B/Layers/Pad_1": ("U1", "5")})

    def test_build_keys_normalize_terminal_number_like_lead_names(self) -> None:
        """A fallback terminal number such as ``A1`` keys as ``1``, like ``pkg_lead_A1``."""
        source = MetadataSource(MetadataOrigin.USD, "board.usda")
        records = (
            PadRecord(
                board_root="/B",
                prim_path="/B/Layers/Pad_1",
                source=source,
                component_path="/B/Components/U1",
                terminal_number="A1",
            ),
        )

        keys = build_pad_joint_keys(records)

        self.assertEqual(keys, {"/B/Layers/Pad_1": ("U1", "1")})


if __name__ == "__main__":
    unittest.main()
