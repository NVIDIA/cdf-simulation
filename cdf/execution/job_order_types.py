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

"""Domain-neutral executable job-order primitives.

``CDFJob`` remains the mutable authoring surface.  A future compiler lowers
that description into an immutable :class:`JobOrderPlan`; executors consume
the plan without reaching back into the job.  This module deliberately has no
Kit, USD, or domain imports, so plans can be validated, compared, and
transported from ordinary Python.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

from ..errors import ValidationError


def _freeze(value: object, path: str = "parameters") -> object:
    """Normalize a transport value into immutable Python values.

    Args:
        value: Value to normalize.
        path: Diagnostic path identifying the value within an order.

    Returns:
        Immutable representation of the supplied value.

    Raises:
        ValidationError: If the value cannot be represented by the transport.

    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("%s must contain only finite numbers" % path)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(
                    "%s keys must be strings, got %r" % (path, key),
                )
            frozen[key] = _freeze(item, "%s.%s" % (path, key))
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(_freeze(item, "%s[]" % path) for item in value)
    raise ValidationError(
        "%s contains unsupported value %r (%s)" % (path, value, type(value).__name__)
    )


def _thaw(value: object) -> object:
    """Convert an immutable order value into transport containers.

    Args:
        value: Immutable value to convert.

    Returns:
        JSON-compatible representation of the supplied value.

    """
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveVersion = Annotated[int, Field(strict=True, ge=1)]


class JobOrder(BaseModel):
    """One immutable unit of executable intent.

    ``order_type`` is a stable semantic identifier such as
    ``"scene.load"`` or ``"defect.tombstone"``.  It selects an executor
    handler; it is not a Python module or function name.  ``order_id`` is
    unique within a plan and is the target of dependency edges.

    Args:
        order_id: Unique identifier within the containing plan.
        order_type: Stable semantic identifier used to resolve a handler.
        parameters: Transport-safe parameters consumed by the handler.
        depends_on: Identifiers of orders that must complete first.
        version: Schema version for this order type.

    Raises:
        pydantic.ValidationError: If identifiers, dependencies, parameters, or the
            version are invalid.

    Example:

    .. code-block:: python

        from cdf import JobOrder

        order = JobOrder(
            order_id="load-board",
            order_type="scene.load",
            parameters={"usd_path": "/data/board.usd"},
        )
        print(order.order_type)
        # scene.load

    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    order_id: NonEmptyString = Field(alias="id")
    #: Identifier referenced by dependency edges within the containing plan.

    order_type: NonEmptyString = Field(alias="type")
    #: Semantic identifier used to resolve an executor handler.

    parameters: Mapping[str, object] = Field(default_factory=dict)
    #: Immutable parameters passed to the executor handler.

    depends_on: tuple[NonEmptyString, ...] = ()
    #: Identifiers of orders that must complete before this order.

    version: PositiveVersion = 1
    #: Schema version for the order payload.

    @field_validator("parameters", mode="after")
    @classmethod
    def freeze_parameters(
        cls,
        parameters: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Normalize parameters into deeply immutable transport values.

        Args:
            parameters: Validated parameter mapping.

        Returns:
            Deeply immutable transport-safe parameter mapping.

        Raises:
            ValueError: If a nested value cannot be transported.

        """
        try:
            return _freeze(parameters)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    @field_serializer("parameters", when_used="json")
    def serialize_parameters(self, parameters: Mapping[str, object]) -> object:
        """Convert immutable parameters into JSON-compatible containers.

        Args:
            parameters: Deeply immutable parameter mapping.

        Returns:
            JSON-compatible parameter mapping.

        """
        return _thaw(parameters)

    @field_validator("depends_on", mode="after")
    @classmethod
    def validate_dependencies(
        cls,
        dependencies: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Validate dependencies that do not require plan context.

        Args:
            dependencies: Normalized dependency identifiers.

        Returns:
            Validated dependency identifiers.

        Raises:
            ValueError: If dependency identifiers are duplicated.

        """
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("dependencies must be unique")
        return dependencies

    @model_validator(mode="after")
    def validate_self_dependency(self) -> JobOrder:
        """Reject an order that depends on itself.

        Returns:
            Validated order without a self-dependency.

        Raises:
            ValueError: If the order depends on its own identifier.

        """
        if self.order_id in self.depends_on:
            raise ValueError("an order cannot depend on itself")
        return self


class JobOrderPlan(BaseModel):
    """Immutable dependency graph of :class:`JobOrder` objects.

    Args:
        orders: Orders in deterministic declaration order.
        metadata: Transport-safe plan-level context such as a random seed.
        version: Schema version for the plan envelope.

    Raises:
        pydantic.ValidationError: If the plan envelope contains invalid values.

    Example:

    .. code-block:: python

        from cdf import JobOrder, JobOrderPlan

        plan = JobOrderPlan(orders=(
            JobOrder(order_id="load", order_type="scene.load"),
        ))
        print(len(plan.orders))
        # 1

    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    orders: tuple[JobOrder, ...] = ()
    #: Orders in deterministic declaration order.

    metadata: Mapping[str, object] = Field(default_factory=dict)
    #: Immutable plan-level execution context.

    version: PositiveVersion = Field(default=1, alias="plan_version")
    #: Schema version for the plan envelope.

    @field_validator("metadata", mode="after")
    @classmethod
    def freeze_metadata(
        cls,
        metadata: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Normalize metadata into deeply immutable transport values.

        Args:
            metadata: Validated plan metadata.

        Returns:
            Deeply immutable transport-safe metadata.

        Raises:
            ValueError: If a nested value cannot be transported.

        """
        try:
            return _freeze(metadata, "metadata")
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    @field_serializer("metadata", when_used="json")
    def serialize_metadata(self, metadata: Mapping[str, object]) -> object:
        """Convert immutable metadata into JSON-compatible containers.

        Args:
            metadata: Deeply immutable plan metadata.

        Returns:
            JSON-compatible metadata mapping.

        """
        return _thaw(metadata)

    def ordered(self) -> tuple[JobOrder, ...]:
        """Validate dependencies and return a stable topological ordering.

        Call order is the tie-breaker: independent orders retain their plan
        order, making execution and serialized comparisons deterministic.

        Returns:
            Orders in stable dependency order.

        Raises:
            ValidationError: If identifiers are duplicated, dependencies are
                missing, or the dependency graph contains a cycle.

        Example:

        .. code-block:: python

            print([item.order_id for item in plan.ordered()])
            # ['load']

        """
        by_id = {}
        positions = {}
        for position, order in enumerate(self.orders):
            if order.order_id in by_id:
                raise ValidationError("duplicate JobOrder id %r" % order.order_id)
            by_id[order.order_id] = order
            positions[order.order_id] = position

        missing = {
            dependency
            for order in self.orders
            for dependency in order.depends_on
            if dependency not in by_id
        }
        if missing:
            raise ValidationError(
                "JobOrder dependencies do not exist: %s" % ", ".join(sorted(missing))
            )

        remaining = {order.order_id: set(order.depends_on) for order in self.orders}
        ready = sorted(
            (
                order_id
                for order_id, dependencies in remaining.items()
                if not dependencies
            ),
            key=positions.get,
        )
        result = []
        while ready:
            order_id = ready.pop(0)
            result.append(by_id[order_id])
            del remaining[order_id]
            newly_ready = []
            for candidate, dependencies in remaining.items():
                dependencies.discard(order_id)
                if not dependencies and candidate not in ready:
                    newly_ready.append(candidate)
            ready.extend(newly_ready)
            ready.sort(key=positions.get)

        if remaining:
            raise ValidationError(
                "JobOrder dependency cycle involves: %s"
                % ", ".join(sorted(remaining, key=positions.get))
            )
        return tuple(result)
