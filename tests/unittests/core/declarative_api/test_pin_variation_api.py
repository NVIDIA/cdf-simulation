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

"""Test the pin_variation operation's selector and its group accumulation."""

from __future__ import annotations

import unittest

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi CDF domain
from cdf import CDFJob
from cdf.errors import ValidationError


def job() -> CDFJob:
    """Return a bare pcb-aoi job.

    Returns:
        A job with no operations declared.

    """
    return CDFJob(domain="pcb-aoi")


def compilable_job() -> CDFJob:
    """Return a job configured far enough to compile a plan.

    Comparing compiled plans is stronger than comparing the intermediate
    `pin_variation_spec`: the spec is what both surfaces feed in, while the
    plan is what the executor actually receives, so a divergence introduced
    anywhere below the spec would still be caught.
    """
    model = CDFJob(domain="pcb-aoi", random_seed=7)
    model.scene.configure(usd_path="board.usd", root_path="/Board")
    model.capture.output_start(0)
    model.capture.renderer(
        mode="pathtracing", resolution=(64, 64), spp=1, total_spp=1, max_bounces=1
    )
    model.capture.camera(
        side="bottom", rotation_range={"x": (0, 0), "y": (0, 0), "z": 0}
    )
    model.capture.scan_grid(x_count=1, y_count=1)
    model.capture.limits(triggers=1, max_images=1)
    model.capture.outputs(rgb=True)
    return model


class SelectorTest(unittest.TestCase):
    """Refdes and package type are one or the other, never both."""

    def test_components_alone_is_accepted(self) -> None:
        """Selecting by refdes is the dependable path and must work."""
        model = job()
        model.operations.pin_variation(
            components=("C1001", "C1002"),
            tilt_range=(0, 15),
        )
        group = model.operations.pin_variation_spec["groups"][0]
        self.assertEqual(group["components"], ["C1001", "C1002"])
        self.assertNotIn("package_types", group)

    def test_package_types_alone_is_accepted(self) -> None:
        """Selecting by package type must work where the metadata is good."""
        model = job()
        model.operations.pin_variation(package_types=("TO92",), tilt_range=(0, 15))
        group = model.operations.pin_variation_spec["groups"][0]
        self.assertEqual(group["package_types"], ["TO92"])

    def test_both_selectors_is_rejected(self) -> None:
        """Combining them would have to mean union or intersection.

        A caller who wrote both almost certainly meant one; guessing turns a
        typo into a silently different sample, so it is refused.
        """
        with self.assertRaises(ValidationError):
            job().operations.pin_variation(
                components=("C1001",),
                package_types=("TO92",),
                tilt_range=(0, 15),
            )

    def test_neither_selector_is_rejected(self) -> None:
        """An unscoped call would silently vary the whole board."""
        with self.assertRaises(ValidationError):
            job().operations.pin_variation(tilt_range=(0, 15))

    def test_an_empty_selector_is_rejected(self) -> None:
        """An empty list selects nothing and is a caller bug, not a no-op."""
        with self.assertRaises(ValidationError):
            job().operations.pin_variation(components=(), tilt_range=(0, 15))

    def test_a_non_string_name_is_rejected(self) -> None:
        """Names have to be names."""
        with self.assertRaises(ValidationError):
            job().operations.pin_variation(components=("C1001", 7), tilt_range=(0, 15))


class RangeTest(unittest.TestCase):
    """Every axis is an optional (lo, hi) band, but not all of them are."""

    def test_all_four_axes_are_carried(self) -> None:
        """The op has to express what the dataset actually varies."""
        model = job()
        model.operations.pin_variation(
            components=("U1",),
            tilt_range=(0, 15),
            cut_angle=(0, 60),
            height_variation=(0.0, 1.5),
            offset_range=(0.0, 0.2),
        )
        # One call, two groups. Leaning and shifting a lead has to happen
        # before the joint is grown, because the joint is sized from where the
        # lead ended up; trimming has to happen after, because the blade takes
        # the lead and the joint together. The caller is not asked to know
        # that -- they still write one call and still get all four axes.
        pose = model.operations.pin_variation_spec["groups"][0]
        self.assertEqual(pose["tilt_range"], [0, 15])
        self.assertEqual(pose["offset_range"], [0.0, 0.2])
        self.assertNotIn("cut_angle", pose)
        self.assertNotIn("height_variation", pose)

        trim = model.operations.pin_cut_spec["groups"][0]
        self.assertEqual(trim["cut_angle"], [0, 60])
        self.assertEqual(trim["height_variation"], [0.0, 1.5])
        self.assertNotIn("tilt_range", trim)
        self.assertNotIn("offset_range", trim)

        # Both groups claim the SAME parts, or the trim would land on leads
        # that were never posed.
        self.assertEqual(pose["components"], trim["components"])

    def test_a_call_with_no_range_is_rejected(self) -> None:
        """Naming parts without varying anything is a no-op worth flagging."""
        with self.assertRaises(ValidationError):
            job().operations.pin_variation(components=("U1",))

    def test_a_reversed_band_is_rejected(self) -> None:
        """(hi, lo) is a typo, not an empty range."""
        with self.assertRaises(ValidationError):
            job().operations.pin_variation(components=("U1",), tilt_range=(15, 0))

    def test_a_negative_magnitude_is_rejected(self) -> None:
        """Offset and protrusion are magnitudes; direction is drawn separately.

        A negative one is a sign error, not the opposite direction, and would
        otherwise pull a lead back inside the board.
        """
        for axis in ("offset_range", "height_variation"):
            with self.subTest(axis=axis), self.assertRaises(ValidationError):
                job().operations.pin_variation(
                    components=("U1",),
                    **{axis: (-0.5, 0.5)},
                )

    def test_a_degenerate_angle_is_rejected(self) -> None:
        """90 degrees off the axis is degenerate, not merely large.

        A lead lying flat has no axis through the board, and a cut plane
        parallel to the lead is not a cut.
        """
        for axis in ("tilt_range", "cut_angle"):
            with self.subTest(axis=axis):
                with self.assertRaises(ValidationError):
                    job().operations.pin_variation(
                        components=("U1",),
                        **{axis: (0, 120)},
                    )
                with self.assertRaises(ValidationError):
                    job().operations.pin_variation(
                        components=("U1",),
                        **{axis: (-5, 15)},
                    )

    def test_a_boolean_band_is_rejected(self) -> None:
        """`bool` is a subclass of `int`, so a typo silently becomes 0..1."""
        with self.assertRaises(ValidationError):
            job().operations.pin_variation(
                components=("U1",),
                tilt_range=(False, True),
            )

    def test_a_scalar_is_rejected(self) -> None:
        """Every axis is a band; a bare number is ambiguous."""
        with self.assertRaises(ValidationError):
            job().operations.pin_variation(components=("U1",), tilt_range=15)


class AccumulationTest(unittest.TestCase):
    """Several calls, one order."""

    def test_repeated_calls_accumulate_groups_in_order(self) -> None:
        """Different sets of parts get different ranges, which is the point."""
        model = job()
        model.operations.pin_variation(
            package_types=("TO92",),
            tilt_range=(0, 15),
        ).pin_variation(
            components=("C1001", "C1002"),
            tilt_range=(0, 4),
            cut_angle=(0, 60),
        )
        groups = model.operations.pin_variation_spec["groups"]
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["package_types"], ["TO92"])
        self.assertEqual(groups[1]["components"], ["C1001", "C1002"])

    def test_it_stays_a_single_operation(self) -> None:
        """Repeat calls must not race to register the operation twice."""
        model = job()
        for index in range(3):
            model.operations.pin_variation(
                components=(f"U{index}",),
                tilt_range=(0, 5),
            )
        self.assertEqual(len(model.operations.pin_variation_spec["groups"]), 3)
        orders = [
            d.order_id
            for d in model._declarations
            if d.order_id == "operation.pin_variation"
        ]
        self.assertEqual(len(orders), 1)

    def test_the_declaration_tracks_every_group(self) -> None:
        """A group added after the first call still has to reach an order.

        These two calls vary DIFFERENT axes, so each reaches the order its
        axis belongs to: U1 is posed before the solder, U2 is trimmed after it.
        """
        model = job()
        model.operations.pin_variation(components=("U1",), tilt_range=(0, 5))
        model.operations.pin_variation(components=("U2",), cut_angle=(0, 60))
        pose = next(
            d for d in model._declarations if d.order_id == "operation.pin_variation"
        )
        self.assertEqual(
            [group["components"] for group in pose.parameters["groups"]],
            [["U1"]],
        )
        trim = next(d for d in model._declarations if d.order_id == "operation.pin_cut")
        self.assertEqual(
            [group["components"] for group in trim.parameters["groups"]],
            [["U2"]],
        )

    def test_a_group_that_varies_both_reaches_both_orders(self) -> None:
        """One call naming pose AND trim axes has to appear in both."""
        model = job()
        model.operations.pin_variation(
            components=("U1",), tilt_range=(0, 5), cut_angle=(0, 60)
        )
        model.operations.pin_variation(components=("U2",), offset_range=(0, 0.1))
        pose = next(
            d for d in model._declarations if d.order_id == "operation.pin_variation"
        )
        trim = next(d for d in model._declarations if d.order_id == "operation.pin_cut")
        self.assertEqual(
            [group["components"] for group in pose.parameters["groups"]],
            [["U1"], ["U2"]],
        )
        self.assertEqual(
            [group["components"] for group in trim.parameters["groups"]],
            [["U1"]],
        )

    def test_chaining_returns_the_section(self) -> None:
        """The op follows the section's call-chaining convention."""
        model = job()
        self.assertIs(
            model.operations.pin_variation(components=("U1",), tilt_range=(0, 5)),
            model.operations,
        )


class YamlParityTest(unittest.TestCase):
    """The YAML surface has to reach the same declaration as the Python one."""

    def test_yaml_groups_reach_the_same_spec(self) -> None:
        """Parity is the claim; a different shape would be a second contract."""
        import yaml

        from pcb_aoi.config.operations import OperationsConfig, apply_operations

        document = yaml.safe_load(
            """
            pin_variation:
              - package_types: [TO92]
                tilt: [0, 15]
                height: [0.0, 1.5]
              - components: [C1001, C1002]
                tilt: [0, 4]
                cut_angle: [0, 60]
            """,
        )
        from_yaml = job()
        apply_operations(
            from_yaml,
            OperationsConfig(**document),
            base_dir=".",
            component_metadata_source="usd",
            selected_types=None,
        )

        from_python = job()
        from_python.operations.pin_variation(
            package_types=("TO92",),
            tilt_range=(0, 15),
            height_variation=(0.0, 1.5),
        ).pin_variation(
            components=("C1001", "C1002"),
            tilt_range=(0, 4),
            cut_angle=(0, 60),
        )

        self.assertEqual(
            from_yaml.operations.pin_variation_spec,
            from_python.operations.pin_variation_spec,
        )

    def test_yaml_and_python_compile_to_the_same_plan(self) -> None:
        """The plan is what the executor sees, so parity has to hold there.

        Matching `pin_variation_spec` only shows the two surfaces agree on the
        input to compilation. Anything that read the declaration differently
        below that point would still diverge, and this is the artifact that
        actually gets transported.
        """
        import yaml

        from pcb_aoi.config.operations import OperationsConfig, apply_operations

        document = yaml.safe_load(
            """
            pin_variation:
              - package_types: [TO92]
                tilt: [0, 15]
                height: [0.0, 1.5]
              - components: [C1001, C1002]
                tilt: [0, 4]
                cut_angle: [0, 60]
            """,
        )
        from_yaml = compilable_job()
        apply_operations(
            from_yaml,
            OperationsConfig(**document),
            base_dir=".",
            component_metadata_source="usd",
            selected_types=None,
        )

        from_python = compilable_job()
        from_python.operations.pin_variation(
            package_types=("TO92",),
            tilt_range=(0, 15),
            height_variation=(0.0, 1.5),
        ).pin_variation(
            components=("C1001", "C1002"),
            tilt_range=(0, 4),
            cut_angle=(0, 60),
        )

        self.assertEqual(
            from_yaml.compile_orders(output_dir="out"),
            from_python.compile_orders(output_dir="out"),
        )

    def test_yaml_is_a_list_so_repeated_groups_survive(self) -> None:
        """A mapping could only ever express one group.

        The Python API takes repeated calls, so the YAML has to be a sequence
        or the two surfaces would not be able to say the same thing.
        """
        import yaml

        from pcb_aoi.config.operations import OperationsConfig, apply_operations

        document = yaml.safe_load(
            """
            pin_variation:
              - components: [U1]
                tilt: [0, 5]
              - components: [U2]
                tilt: [0, 9]
              - components: [U3]
                cut_angle: [0, 30]
            """,
        )
        model = job()
        apply_operations(
            model,
            OperationsConfig(**document),
            base_dir=".",
            component_metadata_source="usd",
            selected_types=None,
        )
        # U1 and U2 lean, U3 is cut: two groups pose, one trims. The
        # sequence still carries three groups; they simply reach the two
        # orders the axes belong to.
        self.assertEqual(len(model.operations.pin_variation_spec["groups"]), 2)
        self.assertEqual(len(model.operations.pin_cut_spec["groups"]), 1)

    def test_yaml_inherits_the_authoring_validation(self) -> None:
        """A YAML typo must fail the same way a Python one does."""
        import yaml

        from pcb_aoi.config.operations import OperationsConfig

        document = yaml.safe_load(
            """
            pin_variation:
              - components: [U1]
                package_types: [TO92]
                tilt: [0, 15]
            """,
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            OperationsConfig(**document)


if __name__ == "__main__":
    unittest.main()
