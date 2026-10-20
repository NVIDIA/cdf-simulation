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

"""Unit tests for tombstone solder-contact geometry and eligibility filters."""

from __future__ import annotations

import importlib
import types
import unittest
from unittest import mock

import numpy as np

from tests.unittests.core.defects.mock_utils import (
    DefectsApiMockEnvironment,
    MockPrim,
    MockXformPrim,
)


class TombstoneContactsTestBase(unittest.TestCase):
    """Shared mock-environment fixture for tombstone contact-geometry tests."""

    def setUp(self) -> None:
        """Create an isolated defects API fixture with module access."""
        self._environment = DefectsApiMockEnvironment()
        self.addCleanup(self._environment.stop)
        self.defects_api, self.component = self._environment.start()
        self.stage = self._environment.get_stage()
        self.tombstone = importlib.import_module("defects_utils.tombstone")
        self.transform_utils = importlib.import_module("defects_utils.transform_utils")


class ComponentTransformTest(TombstoneContactsTestBase):
    """Validate component-local transform resolution."""

    def test_authored_matrix_transform_is_preserved(self) -> None:
        """Prefer an authored matrix operation when one is available."""
        expected = np.eye(4)
        expected[3, :3] = (1.0, 2.0, 3.0)
        prim = MockXformPrim(
            ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
            xform=expected.tolist(),
            composed_xform=(np.eye(4) * 2.0).tolist(),
        )

        result = self.transform_utils._local_transform_matrix(prim)

        np.testing.assert_array_equal(result, expected)

    def test_composed_transform_supports_multiple_xform_ops(self) -> None:
        """Fall back to the composed matrix for translate/rotate stacks."""
        expected = np.eye(4)
        expected[0, 0] = -1.0
        expected[3, :3] = (26.23, -0.6, 1.362)
        prim = MockXformPrim(
            ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
            composed_xform=expected.tolist(),
        )

        result = self.transform_utils._local_transform_matrix(prim)

        np.testing.assert_array_equal(result, expected)

    def test_missing_transform_is_rejected(self) -> None:
        """Raise an actionable error when a prim is not transformable."""
        prim = MockPrim(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))

        with self.assertRaisesRegex(ValueError, "not UsdGeom.Xformable"):
            self.transform_utils._local_transform_matrix(prim)


class TombstoneContactGeometryTest(TombstoneContactsTestBase):
    """Validate the solder-paste/contact tombstone pivot path."""

    def test_paste_geometry_pivot_and_sign_contract(self) -> None:
        """Verify paste-driven pivots land on the support-plane body ends."""
        metadata = {
            "side": "Top",
            "solder_paste_prims": ["/paste_0", "/paste_1"],
        }
        np.random.seed(3)

        result = self.defects_api.tombstone(
            self.component,
            np.eye(4),
            angle_min=70.0,
            angle_max=90.0,
            component_instance=metadata,
        )
        parameters = result.parameters

        self.assertTrue(parameters["use_contacts"])
        self.assertEqual(parameters["contact_source"], "paste")
        self.assertIn(parameters["sign"], (-1, 1))
        self.assertIn(parameters["pivot_idx"], (0, 1))
        self.assertGreaterEqual(abs(parameters["angle_deg"]), 70.0)
        self.assertLessEqual(abs(parameters["angle_deg"]), 90.0)

        # Both candidate pivots are the paste centroids projected onto the
        # board-facing support plane (z=-0.2) and the nearest body end (x=+-2).
        np.testing.assert_allclose(parameters["c0_pivot_local"], (-2.0, 0.0, -0.2))
        np.testing.assert_allclose(parameters["c1_pivot_local"], (2.0, 0.0, -0.2))
        np.testing.assert_allclose(np.abs(parameters["rot_axis"]), (0.0, 1.0, 0.0))
        np.testing.assert_allclose(
            parameters["pivot_local"],
            parameters["c0_pivot_local"]
            if parameters["pivot_idx"] == 0
            else parameters["c1_pivot_local"],
        )

        # The fixed contact edge stays put under the returned transform.
        pivot = np.append(parameters["pivot_local"], 1.0)
        final = np.asarray(result.flat_transform).reshape(4, 4)
        np.testing.assert_allclose(pivot @ final, pivot, atol=1e-12)

    def test_angle_sign_is_rotation_sign_in_both_branches(self) -> None:
        """Verify ``angle_sign`` always matches ``angle_deg``'s sign.

        ``sign`` keeps its historical per-branch meaning (contacts: rotation
        sign; bbox: lifted-end direction) — ``angle_sign`` is the
        unambiguous rotation-sign key shared by both branches.
        """
        metadata = {
            "side": "Top",
            "solder_paste_prims": ["/paste_0", "/paste_1"],
        }
        np.random.seed(3)
        contact_result = self.defects_api.tombstone(
            self.component,
            np.eye(4),
            angle_min=70.0,
            angle_max=90.0,
            component_instance=metadata,
        )
        np.random.seed(3)
        bbox_result = self.defects_api.tombstone(
            self.component,
            np.eye(4),
            angle_min=70.0,
            angle_max=90.0,
        )

        for result in (contact_result, bbox_result):
            parameters = result.parameters
            self.assertIn(parameters["angle_sign"], (-1, 1))
            self.assertEqual(
                parameters["angle_sign"],
                1 if parameters["angle_deg"] >= 0.0 else -1,
            )
        # Documented divergence: contacts "sign" IS the rotation sign;
        # bbox "sign" is the lifted-end direction (may differ from it).
        self.assertEqual(
            contact_result.parameters["sign"],
            contact_result.parameters["angle_sign"],
        )
        self.assertIn(bbox_result.parameters["sign"], (-1, 1))

    def test_paste_transform_differs_from_bbox_transform_for_same_seed(self) -> None:
        """Verify contact pivots produce a different pose than bbox pivots."""
        metadata = {
            "side": "Top",
            "solder_paste_prims": ["/paste_0", "/paste_1"],
        }

        np.random.seed(3)
        contact_result = self.defects_api.tombstone(
            self.component,
            np.eye(4),
            angle_min=80.0,
            angle_max=80.0,
            component_instance=metadata,
        )
        np.random.seed(3)
        bbox_result = self.defects_api.tombstone(
            self.component,
            np.eye(4),
            angle_min=80.0,
            angle_max=80.0,
        )

        self.assertTrue(contact_result.parameters["use_contacts"])
        self.assertFalse(bbox_result.parameters["use_contacts"])
        self.assertEqual(bbox_result.parameters["contact_source"], "bbox")
        self.assertFalse(
            np.allclose(contact_result.flat_transform, bbox_result.flat_transform)
        )

    def test_contact_prims_used_when_no_paste_available(self) -> None:
        """Verify solder-contact centroids drive the pivot without paste."""
        self.stage.add_prim(
            "/contact_0",
            MockPrim(((-1.7, -0.5, -0.25), (-1.3, 0.5, -0.15))),
        )
        self.stage.add_prim(
            "/contact_1",
            MockPrim(((1.3, -0.5, -0.25), (1.7, 0.5, -0.15))),
        )
        metadata = {
            "side": "Top",
            "solder_contact_prims": ["/contact_0", "/contact_1"],
        }
        np.random.seed(5)

        result = self.defects_api.tombstone(
            self.component,
            np.eye(4),
            angle_min=80.0,
            angle_max=80.0,
            component_instance=metadata,
        )
        parameters = result.parameters

        self.assertTrue(parameters["use_contacts"])
        self.assertEqual(parameters["contact_source"], "contact")
        np.testing.assert_allclose(parameters["c0_pivot_local"], (-2.0, 0.0, -0.2))
        np.testing.assert_allclose(parameters["c1_pivot_local"], (2.0, 0.0, -0.2))

    def test_contact_prims_apply_instance_proxy_xform(self) -> None:
        """Verify contact centroids honor an authored xformOp:transform."""
        identity_rows = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        contact_bounds = ((-0.2, -0.5, -0.05), (0.2, 0.5, 0.05))
        self.stage.add_prim(
            "/xform_contact_0",
            MockXformPrim(
                contact_bounds,
                xform=identity_rows + [[-1.5, 0.0, -0.2, 1.0]],
            ),
        )
        self.stage.add_prim(
            "/xform_contact_1",
            MockXformPrim(
                contact_bounds,
                xform=identity_rows + [[1.5, 0.0, -0.2, 1.0]],
            ),
        )
        metadata = {
            "side": "Top",
            "solder_contact_prims": ["/xform_contact_0", "/xform_contact_1"],
        }
        np.random.seed(5)

        result = self.defects_api.tombstone(
            self.component,
            np.eye(4),
            angle_min=80.0,
            angle_max=80.0,
            component_instance=metadata,
        )
        parameters = result.parameters

        self.assertEqual(parameters["contact_source"], "contact")
        np.testing.assert_allclose(parameters["c0_local"], (-1.5, 0.0, -0.2))
        np.testing.assert_allclose(parameters["c1_local"], (1.5, 0.0, -0.2))
        np.testing.assert_allclose(parameters["c0_pivot_local"], (-2.0, 0.0, -0.2))

    def test_bottom_side_paste_orients_board_normal_into_body(self) -> None:
        """Verify paste above the body flips the inferred board normal."""
        self.stage.add_prim(
            "/paste_top_0",
            MockPrim(
                ((-0.2, -0.5, -0.05), (0.2, 0.5, 0.05)),
                ((-1.7, -0.5, 0.15), (-1.3, 0.5, 0.25)),
            ),
        )
        self.stage.add_prim(
            "/paste_top_1",
            MockPrim(
                ((-0.2, -0.5, -0.05), (0.2, 0.5, 0.05)),
                ((1.3, -0.5, 0.15), (1.7, 0.5, 0.25)),
            ),
        )
        metadata = {
            "side": "Bottom",
            "solder_paste_prims": ["/paste_top_0", "/paste_top_1"],
        }
        np.random.seed(9)

        result = self.defects_api.tombstone(
            self.component,
            np.eye(4),
            angle_min=80.0,
            angle_max=80.0,
            component_instance=metadata,
        )
        parameters = result.parameters

        self.assertTrue(parameters["use_contacts"])
        np.testing.assert_allclose(
            parameters["board_normal_local"],
            (0.0, 0.0, -1.0),
            atol=1e-12,
        )
        self.assertAlmostEqual(parameters["support_projection"], -0.2)
        self.assertAlmostEqual(parameters["pivot_local"][2], 0.2)

    def test_length_axis_along_y_keeps_right_handed_frame(self) -> None:
        """Verify a Y-long paste pair yields an X rotation axis."""
        component = MockPrim(((-1.0, -2.0, -0.2), (1.0, 2.0, 0.2)))
        self.stage.add_prim(
            "/paste_y0",
            MockPrim(((-0.5, -1.7, -0.25), (0.5, -1.3, -0.15))),
        )
        self.stage.add_prim(
            "/paste_y1",
            MockPrim(((-0.5, 1.3, -0.25), (0.5, 1.7, -0.15))),
        )
        metadata = {
            "side": "Top",
            "solder_paste_prims": ["/paste_y0", "/paste_y1"],
        }
        np.random.seed(13)

        result = self.defects_api.tombstone(
            component,
            np.eye(4),
            angle_min=80.0,
            angle_max=80.0,
            component_instance=metadata,
        )
        parameters = result.parameters

        self.assertTrue(parameters["use_contacts"])
        np.testing.assert_allclose(np.abs(parameters["rot_axis"]), (1.0, 0.0, 0.0))
        np.testing.assert_allclose(parameters["c0_pivot_local"], (0.0, -2.0, -0.2))
        np.testing.assert_allclose(parameters["c1_pivot_local"], (0.0, 2.0, -0.2))

    def test_singular_component_transform_uses_pseudoinverse(self) -> None:
        """Verify a singular original transform does not raise."""
        metadata = {
            "side": "Top",
            "solder_paste_prims": ["/paste_0", "/paste_1"],
        }
        singular = np.diag([1.0, 1.0, 0.0, 1.0])
        np.random.seed(17)

        result = self.defects_api.tombstone(
            self.component,
            singular,
            angle_min=80.0,
            angle_max=80.0,
            component_instance=metadata,
        )

        self.assertTrue(result.parameters["use_contacts"])
        self.assertEqual(len(result.flat_transform), 16)
        self.assertTrue(np.all(np.isfinite(result.flat_transform)))

    def test_sideflip_uses_paste_support_plane(self) -> None:
        """Verify sideflip shares the paste-derived support geometry."""
        metadata = {
            "side": "Top",
            "solder_paste_prims": ["/paste_0", "/paste_1"],
        }
        np.random.seed(19)

        result = self.defects_api.sideflip(
            self.component,
            np.eye(4),
            angle_min=80.0,
            angle_max=80.0,
            component_instance=metadata,
        )
        parameters = result.parameters

        self.assertEqual(parameters["axis_source"], "paste")
        np.testing.assert_allclose(np.abs(parameters["rot_axis"]), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(parameters["support_projection"], -0.2)
        self.assertAlmostEqual(parameters["pivot_local"][2], -0.2)
        self.assertAlmostEqual(abs(parameters["pivot_local"][1]), 1.0)

        pivot = np.append(parameters["pivot_local"], 1.0)
        final = np.asarray(result.flat_transform).reshape(4, 4)
        np.testing.assert_allclose(pivot @ final, pivot, atol=1e-12)


class TombstoneContactFallbackTest(TombstoneContactsTestBase):
    """Validate clean bbox fallbacks for degraded contact metadata."""

    def _assert_bbox_fallback(self, metadata: dict) -> None:
        """Run tombstone with metadata and assert a clean bbox fallback.

        Args:
            metadata: Component-instance metadata expected to be unusable.
        """
        np.random.seed(23)
        result = self.defects_api.tombstone(
            self.component,
            np.eye(4),
            angle_min=70.0,
            angle_max=90.0,
            component_instance=metadata,
        )
        parameters = result.parameters
        self.assertFalse(parameters["use_contacts"])
        self.assertEqual(parameters["contact_source"], "bbox")
        self.assertEqual(len(result.flat_transform), 16)
        self.assertTrue(np.all(np.isfinite(result.flat_transform)))

    def test_empty_paste_prim_list_falls_back_to_bbox(self) -> None:
        """Verify an empty paste list falls back to the bbox heuristic."""
        self._assert_bbox_fallback({"side": "Top", "solder_paste_prims": []})

    def test_wrong_paste_prim_count_falls_back_to_bbox(self) -> None:
        """Verify one or three paste prims fall back to the bbox heuristic."""
        self._assert_bbox_fallback({"side": "Top", "solder_paste_prims": ["/paste_0"]})
        self._assert_bbox_fallback(
            {
                "side": "Top",
                "solder_paste_prims": ["/paste_0", "/paste_1", "/paste_0"],
            }
        )

    def test_paste_prims_missing_from_stage_fall_back_to_bbox(self) -> None:
        """Verify unresolved paste prim paths fall back to the bbox heuristic."""
        self._assert_bbox_fallback(
            {"side": "Top", "solder_paste_prims": ["/missing_0", "/missing_1"]}
        )
        self._assert_bbox_fallback(
            {"side": "Top", "solder_paste_prims": ["/paste_0", "/missing_1"]}
        )

    def test_missing_contact_prims_fall_back_to_bbox(self) -> None:
        """Verify unresolved contact prim paths fall back to the bbox heuristic."""
        self._assert_bbox_fallback(
            {"side": "Top", "solder_contact_prims": ["/missing_0", "/missing_1"]}
        )

    def test_coincident_paste_centroids_fall_back_to_bbox(self) -> None:
        """Verify degenerate paste geometry falls back to the bbox heuristic."""
        degenerate = MockPrim(((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)))
        self.stage.add_prim("/paste_dup_0", degenerate)
        self.stage.add_prim("/paste_dup_1", degenerate)
        self._assert_bbox_fallback(
            {"side": "Top", "solder_paste_prims": ["/paste_dup_0", "/paste_dup_1"]}
        )

    def test_zero_size_component_bounds_do_not_raise(self) -> None:
        """Verify a zero-size component bbox still yields a valid transform."""
        degenerate_component = MockPrim(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        np.random.seed(23)

        result = self.defects_api.tombstone(
            degenerate_component,
            np.eye(4),
            angle_min=70.0,
            angle_max=90.0,
        )

        self.assertFalse(result.parameters["use_contacts"])
        self.assertEqual(len(result.flat_transform), 16)
        self.assertTrue(np.all(np.isfinite(result.flat_transform)))


class TombstoneEligibilityTest(TombstoneContactsTestBase):
    """Validate tombstone component-eligibility helpers."""

    def test_component_type_tokens_variants(self) -> None:
        """Verify component-type token extraction and matching."""
        module = self.tombstone
        self.assertEqual(module._component_type_tokens({}), [])
        self.assertEqual(
            module._component_type_tokens({"component_types": "R0402"}),
            ["R0402"],
        )
        self.assertEqual(
            module._component_type_tokens({"component_types": ["R0402", "C0603", ""]}),
            ["R0402", "C0603"],
        )
        self.assertTrue(module._matches_component_types("/pcb/R0402_1", ["R0402"]))
        self.assertTrue(module._matches_component_types("/pcb/C3", ["/pcb/C3"]))
        self.assertFalse(module._matches_component_types("/pcb/C30", ["/pcb/C3"]))
        self.assertFalse(
            module._matches_component_types("/pcb/SOT23_1", ["R0402", "C0603"])
        )

    def test_auto_filter_cfg_normalization(self) -> None:
        """Verify auto-filter defaults, aliases, and invalid values."""
        module = self.tombstone

        default_cfg = module._tombstone_auto_filter_cfg({})
        self.assertTrue(default_cfg["enabled"])
        self.assertEqual(default_cfg["require_solder_contact_count"], 2)

        disabled_cfg = module._tombstone_auto_filter_cfg({"auto_filter": False})
        self.assertFalse(disabled_cfg["enabled"])

        alias_cfg = module._tombstone_auto_filter_cfg(
            {"auto_filter": {"require_solder_paste_count": 3}}
        )
        self.assertEqual(alias_cfg["require_solder_contact_count"], 3)

        explicit_cfg = module._tombstone_auto_filter_cfg(
            {"auto_filter": {"require_solder_contact_count": 4}}
        )
        self.assertEqual(explicit_cfg["require_solder_contact_count"], 4)

        invalid_cfg = module._tombstone_auto_filter_cfg({"auto_filter": 3})
        self.assertFalse(invalid_cfg["enabled"])

    def test_solder_contact_count_metadata_sources(self) -> None:
        """Verify contact-count lookups across all metadata sources."""
        count = self.tombstone._tombstone_solder_contact_count

        self.assertIsNone(count("/c", None))
        self.assertIsNone(count("/c", {}))
        self.assertEqual(
            count(
                "/c",
                {
                    "component_defs_by_path": {
                        "/c": {"solder_contact_meshes": ["a", "b"]}
                    }
                },
            ),
            2,
        )
        self.assertEqual(
            count(
                "/c",
                {
                    "component_defs_by_path": {"/c": {"family": "R"}},
                    "contact_index": {"/c": {"solder_contact_meshes": ["a"]}},
                },
            ),
            1,
        )
        self.assertEqual(
            count(
                "/c",
                {"contact_index": {"/c": {"solder_contact_prims": ["a", "b"]}}},
            ),
            2,
        )
        self.assertEqual(
            count("/c", {"manifest_by_path": {"/c": {"solder_contact_count": "3"}}}),
            3,
        )
        self.assertEqual(
            count(
                "/c",
                {
                    "manifest_by_path": {
                        "/c": {"solder_contact_count": None, "family_contact_count": 4}
                    }
                },
            ),
            4,
        )
        self.assertEqual(
            count("/c", {"manifest_by_path": {"/c": {"solder_paste_count": 2}}}),
            2,
        )
        self.assertIsNone(
            count("/c", {"manifest_by_path": {"/c": {"solder_contact_count": "abc"}}})
        )
        self.assertIsNone(count("/c", {"manifest_by_path": {"/c": {"unrelated": 1}}}))
        self.assertIsNone(count("/c", {"manifest_by_path": {"/other": {}}}))

    def test_solder_paste_count_metadata_sources(self) -> None:
        """Verify paste-count lookups across all metadata sources."""
        count = self.tombstone._tombstone_solder_paste_count

        self.assertIsNone(count("/c", None))
        self.assertIsNone(count("/c", {}))
        self.assertEqual(
            count(
                "/c",
                {"contact_index": {"/c": {"solder_paste_prims": ["p0", "p1"]}}},
            ),
            2,
        )
        self.assertEqual(
            count("/c", {"contact_index": {"/c": {"solder_paste_prims": None}}}),
            0,
        )
        self.assertEqual(
            count("/c", {"manifest_by_path": {"/c": {"solder_paste_count": "2"}}}),
            2,
        )
        self.assertIsNone(
            count("/c", {"manifest_by_path": {"/c": {"solder_paste_count": None}}})
        )
        self.assertIsNone(
            count("/c", {"manifest_by_path": {"/c": {"solder_paste_count": "abc"}}})
        )

    def test_component_world_dims_measurement_and_cache(self) -> None:
        """Verify world-dimension measurement, caching, and unit scaling."""
        dims_mm = self.tombstone._component_world_dims_mm
        self.stage.add_prim(
            "/comp/measured",
            MockPrim(((0.0, 0.0, 0.0), (2.0, 1.0, 0.5))),
        )

        self.assertIsNone(dims_mm("/comp/measured", None))
        self.assertIsNone(dims_mm("/comp/unknown", {}))

        context: dict = {}
        first = dims_mm("/comp/measured", context)
        np.testing.assert_allclose(first, (2.0, 1.0, 0.5))
        self.assertIn("/comp/measured", context["world_bbox_mm_by_path"])

        # A second lookup must come from the cache, not the stage.
        context["world_bbox_mm_by_path"]["/comp/measured"] = (9.0, 9.0, 9.0)
        self.assertEqual(dims_mm("/comp/measured", context), (9.0, 9.0, 9.0))

        scaled_context = {"stage_meters_per_unit": 0.01}
        np.testing.assert_allclose(
            dims_mm("/comp/measured", scaled_context),
            (20.0, 10.0, 5.0),
        )

    def test_component_world_dims_empty_bound_returns_none(self) -> None:
        """Verify an empty world bound yields no measurement."""

        class _EmptyRange:
            def IsEmpty(self) -> bool:
                return True

        class _EmptyBound:
            def GetRange(self) -> _EmptyRange:
                return _EmptyRange()

        class _EmptyBBoxCache:
            def __init__(self, *_args: object):
                pass

            def ComputeWorldBound(self, _prim: object) -> _EmptyBound:
                return _EmptyBound()

        self.stage.add_prim(
            "/comp/empty",
            MockPrim(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
        )
        empty_usd_geom = types.SimpleNamespace(
            BBoxCache=_EmptyBBoxCache,
            Tokens=types.SimpleNamespace(default_="default"),
        )
        with mock.patch.object(self.tombstone, "UsdGeom", empty_usd_geom):
            self.assertIsNone(
                self.tombstone._component_world_dims_mm("/comp/empty", {})
            )

    def test_size_filter_limits(self) -> None:
        """Verify size-limit checks and the missing-measurement rejection."""
        passes = self.tombstone._passes_tombstone_size_filter
        self.stage.add_prim(
            "/comp/sized",
            MockPrim(((0.0, 0.0, 0.0), (2.0, 1.0, 0.5))),
        )
        context: dict = {}

        self.assertTrue(passes("/comp/sized", {}, None))
        self.assertFalse(passes("/comp/unknown", {"min_long_axis_mm": 1.0}, {}))
        self.assertTrue(
            passes(
                "/comp/sized",
                {
                    "min_long_axis_mm": 1.5,
                    "max_long_axis_mm": 3.0,
                    "min_short_axis_mm": 0.5,
                    "max_short_axis_mm": 1.5,
                    "min_height_mm": 0.1,
                    "max_height_mm": 1.0,
                },
                context,
            )
        )
        self.assertFalse(passes("/comp/sized", {"max_long_axis_mm": 1.5}, context))
        self.assertFalse(passes("/comp/sized", {"min_height_mm": 1.0}, context))

    def test_eligible_components_manual_type_filter(self) -> None:
        """Verify manual component_types bypass the automatic filters."""
        eligible = self.tombstone._tombstone_eligible_components
        pool = ["/pcb/R0402_1", "/pcb/C0603_2", "/pcb/SOT23_3"]

        self.assertEqual(
            eligible(pool, set(), {"component_types": "R0402"}, None),
            ["/pcb/R0402_1"],
        )
        self.assertEqual(
            eligible(
                pool,
                {"/pcb/R0402_1"},
                {"component_types": ["R0402", "C0603"]},
                None,
                log=False,
            ),
            ["/pcb/C0603_2"],
        )

    def test_eligible_components_requires_metadata_and_enablement(self) -> None:
        """Verify disabled auto-filter or missing metadata yields no components."""
        eligible = self.tombstone._tombstone_eligible_components
        pool = ["/pcb/R0402_1"]

        self.assertEqual(eligible(pool, set(), {"auto_filter": False}, None), [])
        self.assertEqual(eligible(pool, set(), {}, None), [])
        self.assertEqual(eligible(pool, set(), {}, {}, log=False), [])

    def test_eligible_components_auto_filter_rejections(self) -> None:
        """Verify every automatic rejection path plus one eligible component."""
        eligible = self.tombstone._tombstone_eligible_components
        two_contacts = ["a", "b"]
        two_pastes = ["p0", "p1"]
        context = {
            "contact_index": {
                "/comp/wrong_contacts": {"solder_contact_prims": ["a", "b", "c"]},
                "/comp/no_paste": {"solder_contact_prims": two_contacts},
                "/comp/wrong_paste": {
                    "solder_contact_prims": two_contacts,
                    "solder_paste_prims": ["p0"],
                },
                "/comp/too_big": {
                    "solder_contact_prims": two_contacts,
                    "solder_paste_prims": two_pastes,
                },
                "/comp/good": {
                    "solder_contact_prims": two_contacts,
                    "solder_paste_prims": two_pastes,
                },
                "/comp/used": {
                    "solder_contact_prims": two_contacts,
                    "solder_paste_prims": two_pastes,
                },
            },
        }
        self.stage.add_prim("/comp/good", MockPrim(((0.0, 0.0, 0.0), (2.0, 1.0, 0.5))))
        self.stage.add_prim(
            "/comp/too_big", MockPrim(((0.0, 0.0, 0.0), (10.0, 5.0, 3.0)))
        )
        pool = [
            "/comp/no_meta",
            "/comp/wrong_contacts",
            "/comp/no_paste",
            "/comp/wrong_paste",
            "/comp/too_big",
            "/comp/good",
            "/comp/used",
        ]

        result = eligible(
            pool,
            {"/comp/used"},
            {"auto_filter": {"max_long_axis_mm": 5.0}},
            context,
        )

        self.assertEqual(result, ["/comp/good"])

    def test_eligible_components_invalid_required_count_uses_default(self) -> None:
        """Verify an invalid required contact count falls back to two."""
        eligible = self.tombstone._tombstone_eligible_components
        context = {
            "contact_index": {
                "/comp/pair": {
                    "solder_contact_prims": ["a", "b"],
                    "solder_paste_prims": ["p0", "p1"],
                },
            },
        }

        result = eligible(
            ["/comp/pair"],
            set(),
            {"auto_filter": {"require_solder_contact_count": "two"}},
            context,
        )

        self.assertEqual(result, ["/comp/pair"])


class TransformUtilsEdgeCaseTest(TombstoneContactsTestBase):
    """Cover degenerate-geometry branches in the shared transform helpers."""

    def test_normalized_rejects_zero_vectors(self) -> None:
        """Verify normalization returns None for unstable directions."""
        module = self.transform_utils
        self.assertIsNone(module._normalized(np.zeros(3)))
        np.testing.assert_allclose(
            module._normalized(np.array([3.0, 0.0, 0.0])),
            (1.0, 0.0, 0.0),
        )

    def test_row_major_axis_rotation_degenerate_axis_is_identity(self) -> None:
        """Verify a zero rotation axis yields the identity transform."""
        np.testing.assert_array_equal(
            self.transform_utils._row_major_axis_rotation(np.zeros(3), 1.0),
            np.eye(4),
        )

    def test_bbox_wide_and_normal_axes_degenerate_inputs(self) -> None:
        """Verify degenerate bbox inputs return no axes."""
        module = self.transform_utils
        half = np.array([2.0, 1.0, 0.2])
        self.assertEqual(
            module._bbox_wide_and_normal_axes(half, np.zeros(3)),
            (None, None),
        )
        self.assertEqual(
            module._bbox_wide_and_normal_axes(np.zeros(3), np.array([1.0, 0.0, 0.0])),
            (None, None),
        )

    def test_support_axis_and_projection_degenerate_inputs(self) -> None:
        """Verify degenerate support queries return no orientation."""
        module = self.transform_utils
        vmin = np.array([-2.0, -1.0, -0.2])
        vmax = np.array([2.0, 1.0, 0.2])
        self.assertEqual(
            module._support_axis_and_projection(vmin, vmax, np.zeros(3), [np.zeros(3)]),
            (None, None),
        )
        self.assertEqual(
            module._support_axis_and_projection(
                vmin, vmax, np.array([0.0, 0.0, 1.0]), []
            ),
            (None, None),
        )

    def test_project_point_to_support_end_degenerate_long_axis(self) -> None:
        """Verify a degenerate length axis skips the body-end projection."""
        module = self.transform_utils
        projected = module._project_point_to_support_end(
            np.array([0.5, 0.0, 0.1]),
            np.array([-2.0, -1.0, -0.2]),
            np.array([2.0, 1.0, 0.2]),
            np.array([0.0, 0.0, 1.0]),
            -0.2,
            np.zeros(3),
        )
        np.testing.assert_allclose(projected, (0.5, 0.0, -0.2))


if __name__ == "__main__":
    unittest.main()
