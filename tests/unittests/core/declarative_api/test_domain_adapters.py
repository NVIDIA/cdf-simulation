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

"""Test the extension point domains use to contribute Isaac handlers."""

from __future__ import annotations

import unittest

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi domain and its adapter
from cdf.errors import CDFError, ValidationError
from cdf.execution.isaac import domain_adapters


class RegistrationTest(unittest.TestCase):
    """What may and may not be registered."""

    def setUp(self) -> None:
        """Snapshot the registry so a test cannot leak into its neighbours."""
        self._saved = dict(domain_adapters._ADAPTERS)  # noqa: SLF001

    def tearDown(self) -> None:
        """Restore the registry."""
        domain_adapters._ADAPTERS.clear()  # noqa: SLF001
        domain_adapters._ADAPTERS.update(self._saved)  # noqa: SLF001

    def test_a_domain_can_contribute_a_handler(self) -> None:
        """The ordinary case: a domain adds an order CDF never heard of."""
        async def handler(order, context):  # noqa: ANN001, ANN202, ARG001
            """Stand in for a domain's real order handler."""
            return {}

        domain_adapters.register_isaac_adapter(
            "demo", lambda: {"operation.demo.apply": handler},
        )
        self.assertIn("demo", domain_adapters.registered_domains())
        self.assertIs(
            domain_adapters.domain_handlers()["operation.demo.apply"], handler,
        )

    def test_registering_the_same_factory_twice_is_harmless(self) -> None:
        """Importing a domain package twice must not be an error."""
        factory = dict
        domain_adapters.register_isaac_adapter("demo", factory)
        domain_adapters.register_isaac_adapter("demo", factory)
        self.assertIn("demo", domain_adapters.registered_domains())

    def test_a_second_different_factory_is_refused(self) -> None:
        """Silent replacement would make behaviour depend on import order."""
        domain_adapters.register_isaac_adapter("demo", dict)
        with self.assertRaises(CDFError):
            domain_adapters.register_isaac_adapter("demo", lambda: {})

    def test_two_domains_claiming_one_order_type_is_refused(self) -> None:
        """Otherwise which implementation runs depends on import order."""
        async def handler(order, context):  # noqa: ANN001, ANN202, ARG001
            """Stand in for a domain's real order handler."""
            return {}

        domain_adapters.register_isaac_adapter(
            "alpha", lambda: {"operation.clash.apply": handler},
        )
        domain_adapters.register_isaac_adapter(
            "beta", lambda: {"operation.clash.apply": handler},
        )
        with self.assertRaises(CDFError):
            domain_adapters.domain_handlers()

    def test_a_nameless_domain_is_refused(self) -> None:
        """A blank name would collide with the next blank name."""
        with self.assertRaises(ValidationError):
            domain_adapters.register_isaac_adapter("  ", dict)

    def test_a_non_callable_factory_is_refused(self) -> None:
        """The failure should be at registration, not at execution."""
        with self.assertRaises(ValidationError):
            domain_adapters.register_isaac_adapter("demo", {"a": 1})

    def test_the_factory_is_called_lazily(self) -> None:
        """A domain may keep heavy imports out of registration."""
        calls = []

        def factory():  # noqa: ANN202
            """Record that registration asked for the handler table."""
            calls.append(1)
            return {}

        domain_adapters.register_isaac_adapter("lazy", factory)
        self.assertEqual(calls, [])
        domain_adapters.domain_handlers()
        self.assertEqual(len(calls), 1)


class BuiltInRegistryTest(unittest.TestCase):
    """The default registry has to actually consult the extension point."""

    def test_domain_handlers_reach_the_default_registry(self) -> None:
        """A hook nothing calls is not an extension point."""
        from cdf.execution.isaac.default_handler_registry import (
            create_default_isaac_handler_registry,
        )

        registry = create_default_isaac_handler_registry()
        self.assertIn("operation.pin_variation.apply", registry.handlers)

    def test_generic_orders_are_still_built_in(self) -> None:
        """The extension point must not have displaced the neutral handlers."""
        from cdf.execution.isaac.default_handler_registry import (
            create_default_isaac_handler_registry,
        )

        registry = create_default_isaac_handler_registry()
        for order_type in ("scene.load", "capture.render", "stage.serialize"):
            with self.subTest(order_type=order_type):
                self.assertIn(order_type, registry.handlers)


class GeometryRoleRegistryTest(unittest.TestCase):
    """The geometry-role hook that keeps role vocabulary out of the runtime."""

    def setUp(self) -> None:
        self._saved = dict(domain_adapters._GEOMETRY_ROLES)  # noqa: SLF001
        domain_adapters._GEOMETRY_ROLES.clear()  # noqa: SLF001

    def tearDown(self) -> None:
        domain_adapters._GEOMETRY_ROLES.clear()  # noqa: SLF001
        domain_adapters._GEOMETRY_ROLES.update(self._saved)  # noqa: SLF001

    def test_no_registered_domain_means_no_traversal_targets(self) -> None:
        # The generic runtime must carry no role vocabulary of its own: with
        # nothing registered it labels no traversal geometry at all.
        self.assertEqual(domain_adapters.geometry_roles(), {})

    def test_registering_twice_with_the_same_factory_is_harmless(self) -> None:
        def factory():
            return {"barrels": domain_adapters.GeometryRole(role="board_barrel")}

        domain_adapters.register_geometry_roles("demo", factory)
        domain_adapters.register_geometry_roles("demo", factory)
        self.assertEqual(
            domain_adapters.geometry_roles()["barrels"].role,
            "board_barrel",
        )

    def test_a_different_factory_for_a_taken_domain_is_an_error(self) -> None:
        domain_adapters.register_geometry_roles("demo", dict)
        with self.assertRaises(CDFError):
            domain_adapters.register_geometry_roles("demo", lambda: {})

    def test_two_domains_claiming_one_target_is_an_error(self) -> None:
        # Otherwise which resolution wins would depend on import order.
        spec = domain_adapters.GeometryRole(role="board_barrel")
        domain_adapters.register_geometry_roles("a", lambda: {"barrels": spec})
        domain_adapters.register_geometry_roles("b", lambda: {"barrels": spec})
        with self.assertRaises(CDFError):
            domain_adapters.geometry_roles()

    def test_a_factory_returning_a_bare_string_is_rejected_at_merge(self) -> None:
        # Otherwise the runtime fails later reading .role off a str, with
        # nothing naming the domain that declared it.
        domain_adapters.register_geometry_roles(
            "demo",
            lambda: {"body": "pkg_body"},
        )
        with self.assertRaises(ValidationError) as caught:
            domain_adapters.geometry_roles()
        self.assertIn("demo", str(caught.exception))

    def test_a_factory_returning_an_unusable_target_name_is_rejected(self) -> None:
        spec = domain_adapters.GeometryRole(role="pkg_body")
        domain_adapters.register_geometry_roles("demo", lambda: {"": spec})
        with self.assertRaises(ValidationError):
            domain_adapters.geometry_roles()

    def test_an_unknown_ownership_kind_is_rejected_at_declaration(self) -> None:
        with self.assertRaises(ValidationError):
            domain_adapters.GeometryRole(role="pkg_body", ownership="nearby")

    def test_an_empty_role_token_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            domain_adapters.GeometryRole(role="  ")

    def test_pcb_aoi_declares_package_geometry_ancestor_owned_and_projected(
        self,
    ) -> None:
        # The behaviour the runtime depends on, asserted at the declaration.
        from pcb_aoi.execution.isaac_adapter import geometry_roles

        roles = geometry_roles()
        self.assertEqual(roles["body"].ownership, domain_adapters.OWNERSHIP_ANCESTOR)
        self.assertTrue(roles["body"].projected)
        self.assertEqual(roles["marking"].role, "pkg_marking")
        self.assertEqual(
            roles["barrels"].ownership,
            domain_adapters.OWNERSHIP_RELATIONSHIP,
        )
        self.assertFalse(roles["barrels"].projected)


if __name__ == "__main__":
    unittest.main()
