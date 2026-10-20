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

"""Test normalized component-filter resolution."""

import unittest

from pydantic import ValidationError as PydanticValidationError

from cdf import ComponentFilter
from cdf.errors import ValidationError
from pcb_aoi.board_metadata import (
    BoardMetadataCatalog,
    ComponentRecord,
    MetadataOrigin,
    MetadataSource,
    select_components,
)

_SOURCE = MetadataSource(MetadataOrigin.USD, "board.usda")


def _component(
    refdes: str,
    *,
    component_type: str,
    package_type: str | None,
    side: str,
    process_classes: tuple[str, ...],
    footprint_id: str | None = None,
) -> ComponentRecord:
    """Build one component record for selector tests."""
    return ComponentRecord(
        board_root="/Board",
        refdes=refdes,
        prim_path=f"/Board/Components/{refdes}",
        source=_SOURCE,
        component_type=component_type,
        package_type=package_type,
        footprint_id=footprint_id,
        side=side,
        process_class=process_classes[0],
        process_classes=process_classes,
    )


class ComponentFilterTest(unittest.TestCase):
    """Verify explicit fields, composition, and parent intersection."""

    def setUp(self) -> None:
        """Create a small mixed component catalog."""
        self.catalog = BoardMetadataCatalog(
            boards=(),
            components=(
                _component(
                    "R1",
                    component_type="resistor",
                    package_type="0603",
                    side="Top",
                    process_classes=("SMT",),
                    footprint_id="RES_0603",
                ),
                _component(
                    "C1",
                    component_type="capacitor",
                    package_type="0603",
                    side="Top",
                    process_classes=("SMT",),
                ),
                _component(
                    "J1",
                    component_type="connector",
                    package_type="USB-C",
                    side="Bottom",
                    process_classes=("through_hole",),
                ),
            ),
        )

    def test_fields_intersect_and_values_within_a_field_union(self) -> None:
        """Compose fields with AND and same-field values with OR."""
        selected = select_components(
            self.catalog,
            {
                "component_type": ["resistor", "capacitor"],
                "package_type": ["0603"],
                "side": ["Top"],
                "process_classes": ["SMT"],
            },
        )

        self.assertEqual([component.refdes for component in selected], ["R1", "C1"])

    def test_every_normalized_field_uses_exact_metadata(self) -> None:
        """Resolve all scalar and plural fields without overloaded matching."""
        selected = select_components(
            self.catalog,
            {
                "refdes": ["R1"],
                "component_type": ["resistor"],
                "package_type": ["0603"],
                "footprint_id": ["RES_0603"],
                "side": ["Top"],
                "process_class": ["SMT"],
                "process_classes": ["SMT"],
            },
        )

        self.assertEqual([component.refdes for component in selected], ["R1"])

    def test_child_filter_cannot_expand_parent_population(self) -> None:
        """Intersect a child match with its already-resolved parent paths."""
        selected = select_components(
            self.catalog,
            {"package_type": ["0603"]},
            parent_paths=("/Board/Components/R1",),
        )

        self.assertEqual([component.refdes for component in selected], ["R1"])

    def test_empty_filter_preserves_parent_population(self) -> None:
        """Treat an empty filter as an unrestricted child."""
        selected = select_components(
            self.catalog,
            {},
            parent_paths=("/Board/Components/C1",),
        )

        self.assertEqual([component.refdes for component in selected], ["C1"])

    def test_unavailable_field_is_an_explicit_error(self) -> None:
        """Distinguish missing metadata from an ordinary non-match."""
        catalog = BoardMetadataCatalog(
            boards=(),
            components=(
                _component(
                    "R1",
                    component_type="resistor",
                    package_type=None,
                    side="Top",
                    process_classes=("SMT",),
                ),
            ),
        )

        with self.assertRaisesRegex(ValidationError, "package_type"):
            select_components(catalog, {"package_type": ["0603"]})

    def test_mixed_process_component_matches_either_process_class(self) -> None:
        # pcb-builder exports: processClass="Mixed", processClasses=[THT, SMT].
        catalog = BoardMetadataCatalog(
            boards=(),
            components=(
                ComponentRecord(
                    board_root="/Board",
                    refdes="J1",
                    prim_path="/Board/Components/J1",
                    source=_SOURCE,
                    component_type="connector",
                    side="Top",
                    process_class="Mixed",
                    process_classes=("through_hole", "SMT"),
                ),
            ),
        )

        by_refdes = select_components(catalog, {"refdes": ["J1"]})
        by_tht = select_components(catalog, {"process_classes": ["through_hole"]})
        by_smt = select_components(catalog, {"process_classes": ["SMT"]})
        by_aggregate = select_components(catalog, {"process_class": ["Mixed"]})

        self.assertEqual(by_refdes, by_tht)
        self.assertEqual(by_refdes, by_smt)
        self.assertEqual(by_refdes, by_aggregate)
        with self.assertRaisesRegex(ValidationError, "matched no components"):
            select_components(catalog, {"process_class": ["through_hole"]})

    def test_nonmatching_filter_is_an_explicit_error(self) -> None:
        """Reject a populated filter that silently selects nothing."""
        with self.assertRaisesRegex(ValidationError, "matched no components"):
            select_components(self.catalog, {"refdes": ["R404"]})

    def test_filter_fields_reject_empty_or_duplicate_values(self) -> None:
        """Reject ambiguous empty and duplicate field lists at validation."""
        with self.assertRaises(PydanticValidationError):
            ComponentFilter(refdes=())
        with self.assertRaises(PydanticValidationError):
            ComponentFilter(refdes=("R1", "R1"))


if __name__ == "__main__":
    unittest.main()
