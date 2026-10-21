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

"""Check opt-in fillet fullness without moving the pad boundary or pin tip."""

import importlib.util
import math
import unittest
from collections import Counter
from dataclasses import replace

from pcb_aoi.solder_generation.procedural_solver import ProceduralSolderSolver
from pcb_aoi.solder_generation.procedural_tht_geometry import (
    ThroughHoleParameters,
    ThroughHoleSite,
    build_through_hole_volcano,
)


class SolderProfileTest(unittest.TestCase):
    """Verify profile geometry and solver option transport."""

    def test_options_and_validation(self) -> None:
        """Keep defaults and reject singular, nonfinite, or unsupported exponents."""
        self.assertEqual(ThroughHoleParameters().profile_exponent, 1.0)
        self.assertIsNone(ThroughHoleParameters().crease_angle_deg)
        parameters = ProceduralSolderSolver._through_hole_parameters(
            {"tht_profile_exponent": 0.7, "tht_crease_angle_deg": 45.0}, 1.0
        )
        self.assertEqual(parameters.profile_exponent, 0.7)
        self.assertEqual(parameters.crease_angle_deg, 45.0)
        for value in (0.5, 0.0, -1.0, 1.1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ProceduralSolderSolver._through_hole_parameters(
                    {"tht_profile_exponent": value}, 1.0
                )
        for value in (0.0, -1.0, 180.0, math.nan, math.inf):
            with self.subTest(crease_angle=value), self.assertRaises(ValueError):
                ProceduralSolderSolver._through_hole_parameters(
                    {"tht_crease_angle_deg": value}, 1.0
                )

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
    def test_crease_normals_split_vertices_without_moving_geometry(self) -> None:
        """Split shading across strong THT creases while preserving the surface."""
        pin_points = tuple(
            (x, y, z)
            for z in (-2.0, 2.0)
            for x, y in ((-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2))
        )
        pin_indices = (
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
        )
        site = ThroughHoleSite(
            center_xy=(0.0, 0.0),
            pin_radius=0.2,
            pad_radius=0.9,
            base_z=0.0,
            grow_toward_negative_z=True,
            pin_points=pin_points,
            pin_face_vertex_counts=(4,) * 6,
            pin_face_vertex_indices=pin_indices,
        )
        parameters = ThroughHoleParameters(
            resolution=32,
            radial_segments=16,
            solder_bath=True,
        )
        control = build_through_hole_volcano(site, parameters)
        creased = build_through_hole_volcano(
            site,
            replace(parameters, crease_angle_deg=35.0),
        )

        self.assertGreater(len(creased.points), len(control.points))
        self.assertEqual(set(creased.face_vertex_counts), {3})
        self.assertEqual(set(creased.points), set(control.points))
        for normal in creased.normals:
            self.assertAlmostEqual(sum(value * value for value in normal), 1.0)

    def test_fullness_preserves_footprint_tip_and_topology(self) -> None:
        """Raise intermediate fillet rings for both sides and bath modes."""
        pin_points = tuple(
            (x, y, z)
            for z in (-2.0, 2.0)
            for x, y in ((-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2))
        )
        pin_indices = (
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
        )
        for negative in (False, True):
            for bath in (False, True):
                for contour in (
                    (),
                    ((-1.1, -0.9), (1.1, -0.9), (1.1, 0.9), (-1.1, 0.9)),
                ):
                    with self.subTest(negative=negative, bath=bath, contour=contour):
                        site = ThroughHoleSite(
                            center_xy=(0.0, 0.0),
                            pin_radius=0.2,
                            pad_radius=0.9,
                            base_z=0.0,
                            grow_toward_negative_z=negative,
                            pad_shape_xy=contour,
                            pin_points=pin_points,
                            pin_face_vertex_counts=(4,) * 6,
                            pin_face_vertex_indices=pin_indices,
                        )
                        parameters = ThroughHoleParameters(
                            resolution=32, radial_segments=16, solder_bath=bath
                        )
                        control = build_through_hole_volcano(site, parameters)
                        fuller = build_through_hole_volcano(
                            site, replace(parameters, profile_exponent=0.7)
                        )
                        self.assertEqual(
                            control.face_vertex_counts, fuller.face_vertex_counts
                        )
                        self.assertEqual(
                            control.face_vertex_indices, fuller.face_vertex_indices
                        )
                        self.assertEqual(
                            tuple(p[:2] for p in control.points),
                            tuple(p[:2] for p in fuller.points),
                        )
                        self.assertEqual(
                            [p for p in control.points if abs(p[2]) < 1e-10],
                            [p for p in fuller.points if abs(p[2]) < 1e-10],
                        )
                        direction = -1 if negative else 1
                        self.assertAlmostEqual(
                            max(direction * p[2] for p in control.points),
                            max(direction * p[2] for p in fuller.points),
                        )
                        changes = [
                            direction * (new[2] - old[2])
                            for old, new in zip(
                                control.points, fuller.points, strict=True
                            )
                        ]
                        self.assertGreater(max(changes), 0.01)
                        self.assertGreaterEqual(min(changes), -1e-10)
                        edges = Counter()
                        cursor = 0
                        for count in fuller.face_vertex_counts:
                            face = fuller.face_vertex_indices[cursor : cursor + count]
                            cursor += count
                            for a, b in zip(face, face[1:] + face[:1], strict=True):
                                edges[tuple(sorted((a, b)))] += 1
                        self.assertEqual(set(edges.values()), {2})
                        for normal in fuller.normals:
                            self.assertAlmostEqual(sum(v * v for v in normal), 1.0)
