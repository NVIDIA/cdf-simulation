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

"""Register executor handlers for semantic CDF job-order types."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from ..errors import CDFError, ValidationError
from .job_order_types import JobOrder

JobOrderHandler = Callable[[JobOrder, object | None], object]


class JobOrderHandlerRegistry:
    """Map semantic JobOrder types to executor handler callables.

    Args:
        handlers: Optional initial mapping of order types to handlers.

    Raises:
        ValidationError: If an order type or handler is invalid.
        CDFError: If an order type is registered more than once with
            different handlers.

    Example:

    .. code-block:: python

        from cdf import JobOrderHandlerRegistry

        registry = JobOrderHandlerRegistry({
            "scene.load": lambda order, context: {"loaded": True},
        })
        print(sorted(registry.handlers))
        # ['scene.load']

    """

    def __init__(
        self,
        handlers: Mapping[str, JobOrderHandler] | None = None,
    ) -> None:
        self._handlers: dict[str, JobOrderHandler] = {}
        for order_type, handler in dict(handlers or {}).items():
            self.register(order_type, handler)

    def register(
        self,
        order_type: str,
        handler: JobOrderHandler,
    ) -> JobOrderHandler:
        """Register a handler for a semantic order type.

        Args:
            order_type: Stable semantic order identifier.
            handler: Callable receiving an order and execution context.

        Returns:
            Registered handler, allowing decorator-style use.

        Raises:
            ValidationError: If the order type or handler is invalid.
            CDFError: If another handler already owns the order type.

        Example:

        .. code-block:: python

            registry.register("capture.run", lambda order, context: None)
            print("capture.run" in registry.handlers)
            # True

        """
        if not isinstance(order_type, str) or not order_type.strip():
            raise ValidationError("handler order_type must be a non-empty string")
        if not callable(handler):
            raise ValidationError("handler for %r must be callable" % order_type)
        key = order_type.strip()
        existing = self._handlers.get(key)
        if existing is not None and existing is not handler:
            raise CDFError("JobOrder handler %r is already registered" % key)
        self._handlers[key] = handler
        return handler

    def handler(self, order_type: str) -> JobOrderHandler:
        """Resolve the handler registered for an order type.

        Args:
            order_type: Stable semantic order identifier.

        Returns:
            Registered executor handler.

        Raises:
            ValidationError: If no handler supports the order type.

        Example:

        .. code-block:: python

            print(callable(registry.handler("scene.load")))
            # True

        """
        try:
            return self._handlers[order_type]
        except KeyError:
            raise ValidationError(
                "no JobOrder handler registered for %r" % order_type,
            ) from None

    @property
    def handlers(self) -> Mapping[str, JobOrderHandler]:
        """Return a read-only view of registered handlers.

        Returns:
            Mapping from semantic order types to handler callables.

        Example:

        .. code-block:: python

            print(sorted(registry.handlers))
            # ['scene.load']

        """
        return MappingProxyType(self._handlers)
