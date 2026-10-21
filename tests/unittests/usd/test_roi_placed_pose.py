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

"""Focused ROI crops frame a defect-moved part where the board places it.

An AOI photographs each part at its CAD placement, so a tombstone shows the
part standing on one pad with the other pad bare. A crop framed from the
defective pose instead zooms onto the standing part and leaves that pad out,
and a shift or rotation moves and rescales the crop. Only the crop window stays
put: the defect itself stays on the stage, where its labels are drawn.
"""

from __future__ import annotations

import unittest

from cdf import IsaacExecutionContext, IsaacSdgRuntime, ValidationError
from cdf.execution.isaac.capture_execution import (
    _record_placed_bounds,
    _roi_selection_specs,
)
from pcb_aoi.board_metadata import (
    BoardMetadataCatalog,
    BoardRecord,
    ComponentRecord,
    MetadataOrigin,
    MetadataSource,
)

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:  # pragma: no cover - exercised by the usd lane only
    Usd = None

_PART = "/Board/Components/R18"
_PLACED = (10.0, 20.0, 0.0)
#: A 0402 body in millimetres.
_BODY = (1.0, 0.5, 0.35)


def _stage():
    """Return a millimetre stage with one 0402 part at its placement."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.Xform.Define(stage, "/Board")
    UsdGeom.Xform.Define(stage, "/Board/Components")
    part = UsdGeom.Xform.Define(stage, _PART)
    part.AddTranslateOp().Set(Gf.Vec3d(*_PLACED))
    body = UsdGeom.Cube.Define(stage, f"{_PART}/body")
    body.CreateSizeAttr(1.0)
    body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, _BODY[2] / 2.0))
    body.AddScaleOp().Set(Gf.Vec3f(*_BODY))
    return stage


def _context(stage):
    """Return an execution context whose board metadata lists the part."""
    source = MetadataSource(MetadataOrigin.USD, "board.usda")
    component = ComponentRecord(
        board_root="/Board",
        refdes="R18",
        prim_path=_PART,
        source=source,
        component_type="resistor",
        package_name="0402",
        side="top",
        process_class="SMT",
        process_classes=("SMT",),
    )
    context = IsaacExecutionContext(
        runtime=IsaacSdgRuntime(),
        stage=stage,
        scene={"board_root_path": "/Board"},
    )
    context.board_metadata["usd"] = BoardMetadataCatalog(
        boards=(BoardRecord("/Board", "Board", source),),
        components=(component,),
    )
    return context


def _window(context):
    """Return the crop centre and apertures the capture would use for the part."""
    config = {
        "roi_selection": {
            "selections": ({"refdes": ("R18",), "count": None, "padding": 0.5},),
        },
        "pcba_root": "/Board",
        "resolution": (512, 512),
        "random_seed": 0,
        "camera_rotation": {"z_fixed": -90.0},
        "camera_side": "top",
    }
    (spec,) = _roi_selection_specs(context, context.stage, config, 100.0)
    return (
        spec["position"],
        spec["horizontal_aperture"],
        spec["vertical_aperture"],
    )


def _flat(matrix):
    """Flatten a matrix row-major, as defect records store transforms."""
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _tombstone(angle_deg):
    """Stand the part on its -X end cap, rotating about that cap's bottom edge."""
    pivot = Gf.Vec3d(-_BODY[0] / 2.0, 0.0, 0.0)
    rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), -angle_deg))
    return (
        Gf.Matrix4d().SetTranslate(-pivot)
        * rotation
        * Gf.Matrix4d().SetTranslate(pivot)
    )


def _shift(dx, dy, rotate_z_deg):
    """Slide the part and turn it about its own centre."""
    rotation = Gf.Matrix4d().SetRotate(
        Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rotate_z_deg),
    )
    return rotation * Gf.Matrix4d().SetTranslate(Gf.Vec3d(dx, dy, 0.0))


def _record(stage, defect, delta):
    """Build the record ``defect_ops.prepare_defects`` returns for the part."""
    original = UsdGeom.Xformable(stage.GetPrimAtPath(_PART)).GetLocalTransformation()
    return {
        "path": _PART,
        "defect": defect,
        "original_flat": _flat(original),
        f"{defect}_final_flat": _flat(delta * original),
    }


def _part_center(stage):
    """Return the part's live world-bound centre: where its labels are drawn."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bounds = cache.ComputeWorldBound(stage.GetPrimAtPath(_PART))
    return bounds.ComputeAlignedRange().GetMidpoint()


@unittest.skipIf(Usd is None, "usd-core is not installed")
class PlacedPoseRoiTest(unittest.TestCase):
    """A defect must not move or rescale the focused crop of its part."""

    def _assert_same_window(self, actual, expected):
        """Assert two crop windows match to well below a pixel."""
        (actual_xyz, *actual_apertures), (expected_xyz, *expected_apertures) = (
            actual,
            expected,
        )
        for got, want in zip(
            (*actual_xyz, *actual_apertures),
            (*expected_xyz, *expected_apertures),
            strict=True,
        ):
            self.assertAlmostEqual(got, want, places=9)

    def _apply(self, stage, context, record):
        """Apply a defect the way the runtime does: measure, then author."""
        _record_placed_bounds(context, stage, [record])
        IsaacSdgRuntime._author_defect_transforms(stage, [record])

    def test_tombstone_crop_keeps_the_clean_window(self):
        """A tombstoned part is cropped exactly like the clean part, bare pad in."""
        stage = _stage()
        context = _context(stage)
        clean = _window(context)

        self._apply(stage, context, _record(stage, "tombstone", _tombstone(80.0)))

        self._assert_same_window(_window(context), clean)
        # The defect stays on the stage: the part stands on its -X cap.
        self.assertLess(_part_center(stage)[0], _PLACED[0] - 0.4)

    def test_shift_and_rotation_crop_keeps_the_clean_window(self):
        """A shifted, turned part neither moves nor rescales its crop."""
        stage = _stage()
        context = _context(stage)
        clean = _window(context)

        self._apply(stage, context, _record(stage, "shift", _shift(0.3, 0.0, 10.0)))

        self._assert_same_window(_window(context), clean)
        self.assertAlmostEqual(_part_center(stage)[0], _PLACED[0] + 0.3, places=6)

    def test_placed_pose_is_measured_when_the_stage_already_holds_the_defect(self):
        """Without Replicator, defect_ops authors the stage before this runs."""
        stage = _stage()
        context = _context(stage)
        clean = _window(context)
        record = _record(stage, "tombstone", _tombstone(80.0))
        final = Gf.Matrix4d(*record["tombstone_final_flat"])
        UsdGeom.Xformable(stage.GetPrimAtPath(_PART)).MakeMatrixXform().Set(final)

        self._apply(stage, context, record)

        self._assert_same_window(_window(context), clean)
        local = UsdGeom.Xformable(stage.GetPrimAtPath(_PART)).GetLocalTransformation()
        self.assertTrue(Gf.IsClose(local, final, 1e-9))

    def test_a_part_no_defect_moved_is_framed_from_its_live_bounds(self):
        """Parts without a defect record keep the existing live-bounds framing."""
        stage = _stage()
        context = _context(stage)
        clean = _window(context)
        UsdGeom.Xformable(stage.GetPrimAtPath(_PART)).MakeMatrixXform().Set(
            Gf.Matrix4d().SetTranslate(Gf.Vec3d(12.0, 20.0, 0.0)),
        )

        (x, y, _z), *_apertures = _window(context)

        self.assertAlmostEqual(x, clean[0][0] + 2.0, places=9)
        self.assertAlmostEqual(y, clean[0][1], places=9)

    def test_a_record_without_its_placed_transform_is_refused(self):
        """Framing the defect instead of the placement must not pass silently."""
        stage = _stage()
        context = _context(stage)
        record = _record(stage, "tombstone", _tombstone(80.0))
        del record["original_flat"]

        with self.assertRaisesRegex(ValidationError, "has no placed transform"):
            _record_placed_bounds(context, stage, [record])


if __name__ == "__main__":
    unittest.main()
