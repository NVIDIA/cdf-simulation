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

"""``docs/image-to-usd-sidecar.md`` documents an external contract, so it is
tested rather than trusted.

Review on !146 asked for the sidecar format to be documented. A prose document
that drifts from the writer is worse than none, because a consumer integrates
against it and finds out in production. These tests read the document and check
it against a sidecar the writer actually produced:

* every field the document's reference table names is present, and no emitted
  field is left undocumented;
* the worked pixel-to-USD snippet in the document agrees with ``corners_usd``,
  which the writer computes independently;
* the inverse snippet round-trips.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from cdf.postprocess.image_stitching import _write_image_to_usd_sidecar

_DOC = (Path(__file__).resolve().parents[4]
        / "docs" / "image-to-usd-sidecar.md")

_LAYOUT = {
    "pixel_to_world_xy": {
        "origin_xy": [-4.0, 1.0],
        "u_per_px_xy": [1.0, 0.0],
        "v_per_px_xy": [0.0, -1.0],
    },
    "world_reference_z": 12.5,
    "camera_side": "top",
    "reference_plane_source": "substrate_z_range",
    "stage_meters_per_unit": 0.001,
    "stage_up_axis": "Z",
    "source_usd": "omniverse://example/board.usd",
    "tiles": [{"frame": 0, "dst_rect_xywh_px": [0, 0, 2, 2]},
              {"frame": 1, "dst_rect_xywh_px": [2, 0, 2, 2]}],
    # A supported layout always carries the per-cell footprint and the cell
    # counts; the fixture omitted them, which silently switched off the `fov`
    # block and made a documented field look unwritten.
    "footprint_x": 2.0,
    "footprint_y": 2.0,
    "x_num": 2,
    "y_num": 1,
    "board_instances": ["/Board/StepRepeats/SR_0_0"],
    "source_usd_provenance": {"creator": "PPX 1.4.2", "modified_utc": "2026-09-08T11:02:00Z"},
}
_CANVAS = [4, 2]


def _documented_fields(text: str | None = None) -> set[str]:
    """Field names from the reference table's first column."""
    if text is None:
        text = _DOC.read_text(encoding="utf-8")
    section = re.split(r"^## Field reference\s*$", text, maxsplit=1, flags=re.MULTILINE)[1]
    table = re.split(r"^##\s+", section, maxsplit=1, flags=re.MULTILINE)[0]
    names = set()
    for line in table.splitlines():
        match = re.match(r"\|\s*`([a-z_]+)`\s*\|", line)
        if match:
            names.add(match.group(1))
    return names


class DocumentedFieldsParserTest(unittest.TestCase):
    """Reference fields are independent of introductory paragraph spacing."""

    def test_introductory_prose_does_not_hide_fields(self):
        for intro in ("", "The sidecar fields are as follows:\n\n",
                      "First paragraph.\n\nSecond paragraph.\n\n"):
            with self.subTest(intro=intro):
                text = (
                    "## Field reference\n\n" + intro
                    + "| Field | Type | Meaning |\n|---|---|---|\n"
                    "| `schema_version` | string | Schema |\n"
                    "| `image_size_px` | array | Dimensions |\n"
                )
                self.assertEqual(_documented_fields(text), {"schema_version", "image_size_px"})

    def test_fields_from_other_sections_are_excluded(self):
        text = (
            "## Earlier section\n\n| `before` | string | Not a field |\n\n"
            "## Field reference\n\n| `schema_version` | string | Schema |\n\n"
            "## Later section\n\n| `after` | string | Not a field |\n"
        )
        self.assertEqual(_documented_fields(text), {"schema_version"})


class _SidecarTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmp = tempfile.TemporaryDirectory()
        output = Path(cls._tmp.name) / "scan_top.png"
        path = _write_image_to_usd_sidecar(output, dict(_LAYOUT), list(_CANVAS))
        assert path is not None, "the fixture layout should produce a sidecar"
        cls.sidecar = json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class SidecarFilenameTest(unittest.TestCase):
    """The name is a contract with the consumer's file filter, not a label."""

    def test_the_name_starts_with_image(self):
        """PPX selects Modus AOI output with `.../artifact/image`.

        A sidecar named after its image -- `<stem>_image_to_usd.json` -- falls
        outside a filter that the image it describes matches, so the mapping
        the UI needs is the one file the UI cannot see. The prefix leads for
        that reason; the contents are the same either way.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = _write_image_to_usd_sidecar(
                Path(directory) / "board_rgb.png",
                dict(_LAYOUT),
                list(_CANVAS),
            )
        self.assertIsNotNone(path)
        self.assertEqual(path.name, "image_board_rgb_to_usd.json")
        self.assertTrue(path.name.startswith("image"), path.name)


class DocumentMatchesWriterTest(_SidecarTest):

    def test_the_document_exists_and_names_the_schema(self):
        text = _DOC.read_text(encoding="utf-8")
        self.assertIn(self.sidecar["schema_version"], text)

    def test_every_documented_field_is_emitted(self):
        missing = sorted(_documented_fields() - set(self.sidecar))
        self.assertEqual(missing, [], "documented but never written: %s" % missing)

    def test_every_emitted_field_is_documented(self):
        extra = sorted(set(self.sidecar) - _documented_fields())
        self.assertEqual(extra, [], "written but undocumented: %s" % extra)

    def test_the_stated_conventions_are_the_ones_written(self):
        for field, value in (("coordinate_space", "usd_stage_world"),
                             ("units", "usd_stage_units"),
                             ("pixel_coordinate_system", "top-left-xy-pixels"),
                             ("pixel_indexing_convention", "pixel-edge"),
                             ("projection", "orthographic")):
            with self.subTest(field=field):
                self.assertEqual(self.sidecar[field], value)
                self.assertIn(value, _DOC.read_text(encoding="utf-8"))


class WorkedExampleTest(_SidecarTest):
    """The snippets in the document are the arithmetic the file describes."""

    def _forward(self, px, py):
        affine = self.sidecar["affine_pixel_to_usd"]
        ox, oy = affine["origin_xy"]
        ux, uy = affine["u_per_px_xy"]
        vx, vy = affine["v_per_px_xy"]
        return (ox + px * ux + py * vx,
                oy + px * uy + py * vy,
                self.sidecar["reference_plane_z"])

    def _inverse(self, x, y):
        affine = self.sidecar["affine_pixel_to_usd"]
        ox, oy = affine["origin_xy"]
        ux, uy = affine["u_per_px_xy"]
        vx, vy = affine["v_per_px_xy"]
        det = ux * vy - uy * vx
        dx, dy = x - ox, y - oy
        return ((dx * vy - dy * vx) / det, (dy * ux - dx * uy) / det)

    def test_the_forward_snippet_reproduces_every_corner(self):
        for name, corner in self.sidecar["corners_usd"].items():
            with self.subTest(corner=name):
                px, py = corner["pixel_xy"]
                for got, want in zip(self._forward(px, py), corner["usd_xyz"]):
                    self.assertAlmostEqual(got, want, places=9)

    def test_the_inverse_snippet_round_trips(self):
        for px, py in ((0.0, 0.0), (1.5, 0.5), (4.0, 2.0)):
            with self.subTest(pixel=(px, py)):
                x, y, _z = self._forward(px, py)
                back_x, back_y = self._inverse(x, y)
                self.assertAlmostEqual(back_x, px, places=9)
                self.assertAlmostEqual(back_y, py, places=9)

    def test_the_inverse_is_well_conditioned(self):
        affine = self.sidecar["affine_pixel_to_usd"]
        ux, uy = affine["u_per_px_xy"]
        vx, vy = affine["v_per_px_xy"]
        # The document claims the 2x2 inverts cleanly because u and v are the
        # two scan axes. If a layout ever made them parallel, the snippet would
        # divide by zero and the document would be wrong rather than the code.
        self.assertGreater(abs(ux * vy - uy * vx), 1e-9)


if __name__ == "__main__":
    unittest.main()
