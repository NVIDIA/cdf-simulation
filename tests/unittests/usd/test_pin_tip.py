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

"""Tests for through-hole pin-tip annotation-proxy geometry selection."""

from __future__ import annotations

import asyncio
import sys
import unittest
import warnings
from math import cos, radians, sin, tan, tau
from types import ModuleType, SimpleNamespace
from unittest import mock

from cdf import JobOrder
from cdf.execution.isaac.execution_context import IsaacExecutionContext
from cdf.execution.isaac.job_order_parameters import CaptureVisibilityParameters
from pcb_aoi.execution.isaac_adapter import (
    PIN_TIP_ORDER_TYPE,
    handle_pin_tip,
    isaac_handlers,
)
from pcb_aoi.execution.pin_tip_contract import PinTipParameters
from pcb_aoi.pin_tip import create_pin_tip_proxy, select_pin_tip_faces
from pcb_aoi.refdes_selection import select_refdes


class PinTipFaceSelectionTest(unittest.TestCase):
    """Verify complete end-face selection without selecting lead sidewalls."""

    def test_selects_bottom_quad(self) -> None:
        points = (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        )
        counts = (4, 4, 4)
        indices = (
            0,
            3,
            2,
            1,
            4,
            5,
            6,
            7,
            0,
            1,
            5,
            4,
        )

        selected = select_pin_tip_faces(
            points,
            counts,
            indices,
            direction=-1.0,
        )

        self.assertIsNotNone(selected)
        faces, normal = selected
        self.assertEqual(faces, ((0, 3, 2, 1),))
        self.assertLess(normal[2], -0.99)

    def test_selects_every_triangle_of_an_angled_cut(self) -> None:
        points = (
            (-1.0, -1.0, -1.2),
            (1.0, -1.0, -0.8),
            (1.0, 1.0, -0.8),
            (-1.0, 1.0, -1.2),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
        )
        counts = (3, 3, 4)
        indices = (0, 2, 1, 0, 3, 2, 0, 1, 5, 4)

        selected = select_pin_tip_faces(
            points,
            counts,
            indices,
            direction=-1.0,
        )

        self.assertIsNotNone(selected)
        faces, normal = selected
        self.assertEqual(faces, ((0, 2, 1), (0, 3, 2)))
        self.assertLess(normal[2], -0.9)

    def test_selects_steep_cut_through_supported_angle_range(self) -> None:
        for angle in (60.0, 75.0, 89.0):
            with self.subTest(angle=angle):
                slope = tan(radians(angle))
                points = (
                    (-1.0, -1.0, -5.0 - slope),
                    (1.0, -1.0, -5.0 + slope),
                    (1.0, 1.0, -5.0 + slope),
                    (-1.0, 1.0, -5.0 - slope),
                    (-1.0, -1.0, 100.0),
                    (1.0, -1.0, 100.0),
                    (1.0, 1.0, 100.0),
                    (-1.0, 1.0, 100.0),
                )
                counts = (3, 3, 4, 4, 4, 4, 4)
                indices = (
                    0,
                    2,
                    1,
                    0,
                    3,
                    2,
                    4,
                    5,
                    6,
                    7,
                    0,
                    1,
                    5,
                    4,
                    1,
                    2,
                    6,
                    5,
                    2,
                    3,
                    7,
                    6,
                    3,
                    0,
                    4,
                    7,
                    4,
                    7,
                    6,
                    5,
                )

                selected = select_pin_tip_faces(
                    points,
                    counts,
                    indices,
                    direction=-1.0,
                )

                self.assertIsNotNone(selected)
                faces, normal = selected
                self.assertEqual(faces, ((0, 2, 1), (0, 3, 2)))
                self.assertLess(normal[2], 0.0)

    def test_selects_connected_curved_tip_patch(self) -> None:
        points = (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, -1.0, -2.0),
            (1.0, -1.0, -2.0),
            (1.0, 1.0, -2.0),
            (-1.0, 1.0, -2.0),
        )
        counts = (3, 3, 3, 3, 4, 4, 4, 4, 4)
        indices = (
            0,
            1,
            4,
            1,
            2,
            4,
            2,
            3,
            4,
            3,
            0,
            4,
            0,
            5,
            6,
            1,
            1,
            6,
            7,
            2,
            2,
            7,
            8,
            3,
            3,
            8,
            5,
            0,
            5,
            8,
            7,
            6,
        )

        selected = select_pin_tip_faces(
            points,
            counts,
            indices,
            direction=1.0,
        )

        self.assertIsNotNone(selected)
        faces, normal = selected
        self.assertEqual(
            faces,
            ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
        )
        self.assertGreater(normal[2], 0.99)

    def test_flat_terminal_cap_excludes_adjacent_chamfers(self) -> None:
        points = (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-1.0, -1.0, -2.85),
            (1.0, -1.0, -2.85),
            (1.0, 1.0, -2.85),
            (-1.0, 1.0, -2.85),
            (-0.7, -0.7, -3.2),
            (0.7, -0.7, -3.2),
            (0.7, 0.7, -3.2),
            (-0.7, 0.7, -3.2),
        )
        counts = (4,) * 10
        indices = (
            0,
            1,
            2,
            3,
            0,
            1,
            5,
            4,
            1,
            2,
            6,
            5,
            2,
            3,
            7,
            6,
            3,
            0,
            4,
            7,
            4,
            5,
            9,
            8,
            5,
            6,
            10,
            9,
            6,
            7,
            11,
            10,
            7,
            4,
            8,
            11,
            11,
            10,
            9,
            8,
        )

        selected = select_pin_tip_faces(
            points,
            counts,
            indices,
            direction=-1.0,
        )

        self.assertIsNotNone(selected)
        faces, normal = selected
        self.assertEqual(faces, ((11, 10, 9, 8),))
        self.assertLess(normal[2], -0.99)

    def test_rounded_tip_stops_at_its_chamfer(self) -> None:
        """A bullet-shaped lead keeps its chamfer as lead, not tip.

        The tip is a tessellated dome on a cylindrical shaft, so no coplanar
        cap exists and selection falls back to growing from the apex. Without
        a deviation bound the patch crawls down the whole dome and the
        ``pin`` class loses the chamfer it should own.
        """
        rings = 8
        side_count = 16
        radius = 0.2
        angles = tuple(index * tau / side_count for index in range(side_count))
        points = [(0.0, 0.0, -radius)]
        for ring in range(1, rings + 1):
            polar = radians(90.0 * ring / rings)
            for angle in angles:
                points.append(
                    (
                        radius * sin(polar) * cos(angle),
                        radius * sin(polar) * sin(angle),
                        -radius * cos(polar),
                    )
                )
        shaft_base = len(points)
        points.extend(
            (radius * cos(angle), radius * sin(angle), 3.0) for angle in angles
        )
        counts: list[int] = []
        indices: list[int] = []
        for index in range(side_count):
            counts.append(3)
            indices.extend((0, 1 + (index + 1) % side_count, 1 + index))
        for ring in range(rings - 1):
            inner = 1 + ring * side_count
            outer = 1 + (ring + 1) * side_count
            for index in range(side_count):
                nxt = (index + 1) % side_count
                counts.append(4)
                indices.extend(
                    (inner + index, inner + nxt, outer + nxt, outer + index)
                )
        rim = 1 + (rings - 1) * side_count
        for index in range(side_count):
            nxt = (index + 1) % side_count
            counts.append(4)
            indices.extend(
                (rim + index, rim + nxt, shaft_base + nxt, shaft_base + index)
            )

        selected = select_pin_tip_faces(
            tuple(points),
            tuple(counts),
            tuple(indices),
            direction=-1.0,
        )

        self.assertIsNotNone(selected)
        faces, normal = selected
        self.assertLess(normal[2], -0.99)
        vertices = {vertex for face in faces for vertex in face}
        depth = max(points[vertex][2] for vertex in vertices) - min(
            points[vertex][2] for vertex in vertices
        )
        # The dome alone is 0.2 deep; a bounded patch stays near the apex and
        # never reaches the shaft, which starts at z = 3.0.
        self.assertLess(depth, radius * (1.0 - cos(radians(45.0))))
        self.assertTrue(
            all(points[vertex][2] < -radius * 0.5 for vertex in vertices),
            "tip patch escaped past the chamfer",
        )

    def test_tessellated_slanted_cut_is_selected_whole(self) -> None:
        """A planar cut subdivided into many facets is still taken entirely.

        The deviation bound is measured against the seed normal, so coplanar
        facets all qualify however finely the cut is meshed.
        """
        columns = 6
        slope = tan(radians(30.0))
        grid = tuple(
            tuple(
                (
                    -1.0 + 2.0 * column / columns,
                    -1.0 + 2.0 * row / columns,
                    -3.0 + slope * (-1.0 + 2.0 * column / columns),
                )
                for column in range(columns + 1)
            )
            for row in range(columns + 1)
        )
        points = tuple(point for row in grid for point in row)
        counts: list[int] = []
        indices: list[int] = []
        stride = columns + 1
        for row in range(columns):
            for column in range(columns):
                base = row * stride + column
                counts.append(4)
                indices.extend((base, base + 1, base + stride + 1, base + stride))

        selected = select_pin_tip_faces(
            points,
            tuple(counts),
            tuple(indices),
            direction=-1.0,
        )

        self.assertIsNotNone(selected)
        faces, _normal = selected
        self.assertEqual(len(faces), columns * columns)

    def test_flat_polygonal_cap_excludes_chamfer_ring(self) -> None:
        side_count = 16
        angles = tuple(index * tau / side_count for index in range(side_count))
        outer_ring = tuple((cos(angle), sin(angle), -2.85) for angle in angles)
        cap_ring = tuple((0.7 * cos(angle), 0.7 * sin(angle), -3.2) for angle in angles)
        points = outer_ring + cap_ring
        chamfer_indices = tuple(
            vertex
            for index in range(side_count)
            for vertex in (
                index,
                (index + 1) % side_count,
                side_count + (index + 1) % side_count,
                side_count + index,
            )
        )
        cap_indices = tuple(reversed(range(side_count, side_count * 2)))

        selected = select_pin_tip_faces(
            points,
            (4,) * side_count + (side_count,),
            chamfer_indices + cap_indices,
            direction=-1.0,
        )

        self.assertIsNotNone(selected)
        faces, normal = selected
        self.assertEqual(faces, (cap_indices,))
        self.assertLess(normal[2], -0.99)


class PinTipContractTest(unittest.TestCase):
    """Verify the executable pin-tip parameter boundary."""

    def test_accepts_a_filtered_bottom_proxy(self) -> None:
        parameters = PinTipParameters(
            side="bottom",
            components=("X205",),
        )

        self.assertEqual(parameters.class_name, "pin_tip")
        self.assertEqual(parameters.offset_mm, 0.002)

    def test_rejects_duplicate_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            PinTipParameters(
                side="bottom",
                components=("X205", "X205"),
            )

    def test_handler_is_registered_by_the_domain(self) -> None:
        self.assertIn(PIN_TIP_ORDER_TYPE, isaac_handlers())

    def test_refdes_selection_accounts_for_partial_matches(self) -> None:
        selection = select_refdes(
            ["X204", "X205"],
            ["X205", "X999"],
            key=str,
        )

        self.assertEqual(selection.selected, ("X205",))
        self.assertEqual(selection.matched, ("X205",))
        self.assertEqual(selection.unmatched, ("X999",))

    def test_handler_warns_and_reports_partial_selection(self) -> None:
        pxr = ModuleType("pxr")
        pxr.UsdGeom = SimpleNamespace(GetStageMetersPerUnit=lambda _stage: 0.001)
        common = SimpleNamespace(
            refresh_authored_semantics=lambda _stage, paths: len(paths)
        )
        leads = (
            SimpleNamespace(refdes="X205", path="/Board/Components/X205/lead_1"),
            SimpleNamespace(refdes="X205", path="/Board/Components/X205/lead_2"),
        )
        context = IsaacExecutionContext(runtime=object(), stage=object())
        context.scene = {"root_path": "/Board", "board_root_path": "/Board"}
        order = JobOrder(
            id="operation.pin_tip",
            type=PIN_TIP_ORDER_TYPE,
            parameters={"side": "bottom", "components": ["X205", "X999"]},
        )
        tht = ModuleType("pcb_aoi.tht")
        tht.discover_through_hole_leads = lambda _stage, _catalogs: (leads, ())
        pin_tip = ModuleType("pcb_aoi.pin_tip")
        generated = iter(("/Board/SegmentationProxies/PinTips/lead_1", None))
        pin_tip.create_pin_tip_proxy = lambda *_args, **_kwargs: next(generated)

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "pxr": pxr,
                    "common": common,
                    "pcb_aoi.tht": tht,
                    "pcb_aoi.pin_tip": pin_tip,
                },
            ),
            mock.patch("pcb_aoi.execution.isaac_adapter.mark_instanced_stage_mutation"),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            output = asyncio.run(handle_pin_tip(order, context))

        self.assertEqual(output["matched_components"], ["X205"])
        self.assertEqual(output["unmatched_components"], ["X999"])
        self.assertEqual(output["selected"], 2)
        self.assertEqual(output["generated"], 1)
        self.assertEqual(
            output["failed_prim_paths"],
            ["/Board/Components/X205/lead_2"],
        )
        self.assertEqual(len(output["warnings"]), 2)
        self.assertEqual(len(caught), 2)

    def test_handler_keeps_job_alive_when_filter_matches_nothing(self) -> None:
        pxr = ModuleType("pxr")
        pxr.UsdGeom = SimpleNamespace(GetStageMetersPerUnit=lambda _stage: 0.001)
        context = IsaacExecutionContext(runtime=object(), stage=object())
        context.scene = {"root_path": "/Board", "board_root_path": "/Board"}
        order = JobOrder(
            id="operation.pin_tip",
            type=PIN_TIP_ORDER_TYPE,
            parameters={"side": "bottom", "components": ["X999"]},
        )
        tht = ModuleType("pcb_aoi.tht")
        tht.discover_through_hole_leads = lambda _stage, _catalogs: ((), ())

        with (
            mock.patch.dict(sys.modules, {"pxr": pxr, "pcb_aoi.tht": tht}),
            self.assertWarnsRegex(RuntimeWarning, "X999"),
        ):
            output = asyncio.run(handle_pin_tip(order, context))

        self.assertEqual(output["unmatched_components"], ["X999"])
        self.assertEqual(output["generated"], 0)
        self.assertEqual(output["semantic_prims"], 0)

    def test_isaac_visibility_contract_accepts_pin_tip(self) -> None:
        parameters = CaptureVisibilityParameters(show=("hole", "pin_tip"))

        self.assertEqual(parameters.show, ("hole", "pin_tip"))

    def test_isaac_visibility_contract_rejects_show_hide_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "show and hide"):
            CaptureVisibilityParameters(
                hide=("pin_tip",),
                show=("pin_tip",),
            )

    def test_proxy_is_magenta_under_grouped_xform_parent(self) -> None:
        try:
            from pxr import Gf, Usd, UsdGeom, UsdShade
        except ModuleNotFoundError:
            self.skipTest("usd-core not installed (CI installs numpy only)")

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 0.001)
        lead = UsdGeom.Mesh.Define(stage, "/World/Board/Components/X205/pkg_lead_1")
        lead.CreatePointsAttr(
            [
                Gf.Vec3f(-0.5, -0.5, -1.0),
                Gf.Vec3f(0.5, -0.5, -1.0),
                Gf.Vec3f(0.5, 0.5, -1.0),
                Gf.Vec3f(-0.5, 0.5, -1.0),
                Gf.Vec3f(-0.5, -0.5, 1.0),
                Gf.Vec3f(0.5, -0.5, 1.0),
                Gf.Vec3f(0.5, 0.5, 1.0),
                Gf.Vec3f(-0.5, 0.5, 1.0),
            ]
        )
        lead.CreateFaceVertexCountsAttr([4, 4, 4])
        lead.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4])
        material = UsdShade.Material.Define(stage, "/World/Looks/Pin")
        UsdShade.MaterialBindingAPI.Apply(lead.GetPrim()).Bind(material)
        path = create_pin_tip_proxy(
            stage,
            str(lead.GetPath()),
            output_root_path="/World/Board/SegmentationProxies/PinTips",
            class_name="pin_tip",
            side="bottom",
            offset_mm=0.002,
            meters_per_unit=0.001,
        )

        self.assertIsNotNone(path)
        self.assertTrue(
            stage.GetPrimAtPath("/World/Board/SegmentationProxies").IsA(UsdGeom.Xform)
        )
        self.assertEqual(
            UsdGeom.Imageable(stage.GetPrimAtPath("/World/Board/SegmentationProxies"))
            .GetVisibilityAttr()
            .Get(),
            UsdGeom.Tokens.invisible,
        )
        self.assertTrue(
            stage.GetPrimAtPath("/World/Board/SegmentationProxies/PinTips").IsA(
                UsdGeom.Xform
            )
        )
        self.assertEqual(
            UsdGeom.Imageable(
                stage.GetPrimAtPath("/World/Board/SegmentationProxies/PinTips")
            )
            .GetVisibilityAttr()
            .Get(),
            UsdGeom.Tokens.invisible,
        )
        proxy = stage.GetPrimAtPath(path)
        bound, _relationship = UsdShade.MaterialBindingAPI(proxy).ComputeBoundMaterial()
        self.assertFalse(bound.GetPrim().IsValid())
        color = UsdGeom.Mesh(proxy).GetDisplayColorAttr().Get()
        self.assertEqual(tuple(color[0]), (1.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
