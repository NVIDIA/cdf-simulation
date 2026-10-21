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

"""Test the pin_variation execution contract and its handler registration."""

from __future__ import annotations

import unittest

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi CDF domain
from cdf import CDFJob
from cdf.execution.isaac.default_handler_registry import (
    create_default_isaac_handler_registry,
)
from pcb_aoi.execution import PinVariationGroup, PinVariationParameters
from pydantic import ValidationError as PydanticValidationError

_ORDER_TYPE = "operation.pin_variation.apply"


class ContractTest(unittest.TestCase):
    """The contract is the second gate, not a restatement of the first."""

    def test_a_valid_group_is_accepted(self) -> None:
        """The ordinary shape has to pass."""
        parameters = PinVariationParameters(
            groups=({"components": ("U1",), "tilt_range": (0.0, 15.0)},),
        )
        self.assertEqual(parameters.groups[0].components, ("U1",))

    def test_the_order_carries_no_seed(self) -> None:
        """The JOB owns the seed; an order-level one would shadow it.

        A `seed` field defaulting to 0 made `CDFJob(random_seed=...)`
        silently inert for this operation while every other sampling order
        honoured it.
        """
        with self.assertRaises(PydanticValidationError):
            PinVariationParameters(
                groups=({"components": ("U1",), "tilt_range": (0.0, 15.0)},),
                seed=7,
            )

    def test_both_selectors_is_rejected(self) -> None:
        """The authoring surface refuses it; so must the transport."""
        with self.assertRaises(PydanticValidationError):
            PinVariationGroup(
                components=("U1",),
                package_types=("TO92",),
                tilt_range=(0.0, 15.0),
            )

    def test_neither_selector_is_rejected(self) -> None:
        """An unscoped group would vary the whole board."""
        with self.assertRaises(PydanticValidationError):
            PinVariationGroup(tilt_range=(0.0, 15.0))

    def test_a_group_with_no_band_is_rejected(self) -> None:
        """Naming parts without varying anything is a no-op."""
        with self.assertRaises(PydanticValidationError):
            PinVariationGroup(components=("U1",))

    def test_a_reversed_band_is_rejected(self) -> None:
        """(hi, lo) is a typo, not an empty range."""
        with self.assertRaises(PydanticValidationError):
            PinVariationGroup(components=("U1",), tilt_range=(15.0, 0.0))

    def test_an_empty_group_list_is_rejected(self) -> None:
        """An order that varies nothing should not reach a runtime."""
        with self.assertRaises(PydanticValidationError):
            PinVariationParameters(groups=())

    def test_an_unknown_field_is_rejected(self) -> None:
        """A misspelled axis must fail loudly rather than be dropped."""
        with self.assertRaises(PydanticValidationError):
            PinVariationGroup(components=("U1",), tilt_rnge=(0.0, 15.0))

    def test_an_empty_name_is_rejected(self) -> None:
        """A blank refdes selects nothing and hides the mistake."""
        with self.assertRaises(PydanticValidationError):
            PinVariationGroup(components=("",), tilt_range=(0.0, 15.0))

    def test_the_contract_enforces_the_same_bounds(self) -> None:
        """The transport gate must not be looser than the authoring one.

        A JobOrder can arrive from a file or another process, so the contract
        is the only gate that has definitely run.
        """
        for kwargs in (
            {"offset_range": (-0.5, 0.5)},
            {"height_variation": (-1.0, 1.0)},
            {"tilt_range": (0.0, 120.0)},
            {"cut_angle": (-5.0, 30.0)},
            {"tilt_range": (False, True)},
        ):
            with self.subTest(**kwargs), self.assertRaises(
                PydanticValidationError,
            ):
                PinVariationGroup(components=("U1",), **kwargs)

    def test_the_contract_is_frozen(self) -> None:
        """Order parameters must not be mutated by a runtime."""
        parameters = PinVariationParameters(
            groups=({"components": ("U1",), "tilt_range": (0.0, 15.0)},),
        )
        with self.assertRaises(PydanticValidationError):
            parameters.seed = 4


class HandlerTest(unittest.TestCase):
    """The order type has to resolve to the runtime method."""

    def test_the_order_type_is_registered(self) -> None:
        """Without this the order compiles and then fails at execution."""
        registry = create_default_isaac_handler_registry()
        self.assertIn(_ORDER_TYPE, registry.handlers)

    def test_the_handler_is_callable(self) -> None:
        """A registered name that resolves to nothing still fails at run time."""
        registry = create_default_isaac_handler_registry()
        self.assertTrue(callable(registry.handler(_ORDER_TYPE)))

    def test_the_handler_is_the_domain_owned_one(self) -> None:
        """CDF must not own a PCB order's contract or implementation.

        `DEVELOPMENT.md` names the growing list of PCB-specific contracts and
        handlers embedded in the Isaac backend as a boundary violation not to
        extend. This asserts the order arrives through the domain adapter.
        """
        from pcb_aoi.execution import handle_pin_variation

        registry = create_default_isaac_handler_registry()
        self.assertIs(registry.handler(_ORDER_TYPE), handle_pin_variation)

    def test_cdf_does_not_own_the_pcb_contract(self) -> None:
        """The contract belongs to whoever invented the order."""
        import cdf.execution.isaac.job_order_parameters as core
        import cdf.execution.isaac.job_order_runtime as runtime

        self.assertFalse(hasattr(core, "PinVariationParameters"))
        self.assertFalse(hasattr(runtime.IsaacSdgRuntime, "apply_pin_variation"))

    def test_the_domain_registers_itself_on_import(self) -> None:
        """Importing the domain is what makes its orders executable."""
        from cdf.execution.isaac.domain_adapters import registered_domains

        self.assertIn("pcb-aoi", registered_domains())

    def test_the_authoring_surface_declares_that_order_type(self) -> None:
        """The declaration and the handler have to agree on the name."""
        job = CDFJob(domain="pcb-aoi")
        job.operations.pin_variation(components=("U1",), tilt_range=(0, 15))
        declaration = next(
            d for d in job._declarations if d.order_id == "operation.pin_variation"
        )
        self.assertEqual(declaration.order_type, _ORDER_TYPE)

    def test_the_declared_parameters_satisfy_the_contract(self) -> None:
        """Authoring output must survive transport validation unchanged.

        This is the join the two gates exist to protect: a section method that
        emits something its own contract rejects fails only at run time.
        """
        job = CDFJob(domain="pcb-aoi")
        job.operations.pin_variation(
            package_types=("TO92",), tilt_range=(0, 15), cut_angle=(0, 60),
        ).pin_variation(
            components=("C1001", "C1002"), height_variation=(0.0, 1.5),
        )
        # BOTH orders carry the same contract. They are two halves of one
        # declaration and a group that transports for one has to transport for
        # the other, or the split would be a place for a group to go missing.
        pose = next(
            d for d in job._declarations
            if d.order_id == "operation.pin_variation"
        )
        parameters = PinVariationParameters(**pose.parameters)
        self.assertEqual(len(parameters.groups), 1)
        self.assertEqual(parameters.groups[0].package_types, ("TO92",))
        self.assertEqual(parameters.groups[0].tilt_range, (0.0, 15.0))

        trim = next(
            d for d in job._declarations if d.order_id == "operation.pin_cut"
        )
        trim_parameters = PinVariationParameters(**trim.parameters)
        self.assertEqual(len(trim_parameters.groups), 2)
        self.assertEqual(trim_parameters.groups[0].cut_angle, (0.0, 60.0))
        self.assertEqual(
            trim_parameters.groups[1].components, ("C1001", "C1002"))
        self.assertEqual(
            trim_parameters.groups[1].height_variation, (0.0, 1.5))


if __name__ == "__main__":
    unittest.main()
