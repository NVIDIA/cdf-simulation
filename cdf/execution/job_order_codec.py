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

"""Serialize CDF job-order plans for cross-process execution."""

from __future__ import annotations

import json

from ..errors import ValidationError
from .job_order_types import JobOrderPlan


class JobOrderCodec:
    """Deterministic JSON transport for a :class:`JobOrderPlan`.

    JSON is an implementation detail of the cross-process executor. The
    in-process API continues to pass ``JobOrderPlan`` objects directly.

    Example:

    .. code-block:: python

        from cdf import JobOrderCodec, JobOrderPlan

        payload = JobOrderCodec.dumps(JobOrderPlan())
        print(JobOrderCodec.loads(payload).orders)
        # ()

    """

    @staticmethod
    def dumps(plan: JobOrderPlan) -> str:
        """Serialize a job-order plan deterministically.

        Args:
            plan: Validated plan to serialize.

        Returns:
            Compact JSON representation with stable key ordering.

        Raises:
            ValidationError: If the plan or dependency graph is invalid.

        Example:

        .. code-block:: python

            print(JobOrderCodec.dumps(JobOrderPlan()).startswith("{"))
            # True

        """
        if not isinstance(plan, JobOrderPlan):
            raise ValidationError("JobOrderCodec.dumps expects JobOrderPlan")
        plan.ordered()
        return json.dumps(
            plan.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @staticmethod
    def loads(payload: str | bytes | bytearray) -> JobOrderPlan:
        """Deserialize and validate a job-order plan.

        Args:
            payload: Serialized plan representation.

        Returns:
            Reconstituted immutable job-order plan.

        Raises:
            pydantic.ValidationError: If the payload schema is invalid.
            ValidationError: If the dependency graph is invalid.

        Example:

        .. code-block:: python

            plan = JobOrderCodec.loads('{"orders": []}')
            print(plan.orders)
            # ()

        """
        plan = JobOrderPlan.model_validate_json(payload)
        plan.ordered()
        return plan
