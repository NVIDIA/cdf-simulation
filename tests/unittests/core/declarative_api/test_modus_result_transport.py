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

"""Verify nested Modus settings survive the result transport."""

import asyncio
import json
import unittest
from dataclasses import asdict
from types import MappingProxyType

from cdf.execution.isaac.default_handler_registry import (
    create_default_isaac_handler_registry,
)
from cdf.execution.isaac.execution_context import IsaacExecutionContext
from cdf.execution.isaac.in_process_executor import IsaacInProcessExecutor
from cdf.execution.isaac.job_order_runtime import IsaacSdgRuntime
from cdf.execution.job_order_types import JobOrder, JobOrderPlan


class ModusResultTransportTest(unittest.TestCase):
    """Exercise the same frozen order path used by the render subprocess."""

    def test_nested_modus_settings_are_json_serializable(self):
        intent = {
            "elevations": {"white": 60.0, "red": 40.0, "blue": 20.0},
            "colors": {"white": [1.0, 1.0, 1.0]},
        }
        order = JobOrder(
            order_id="lighting",
            order_type="appearance.lighting.configure",
            parameters={"mode": "modus_light", "modus": intent},
        )
        context = IsaacExecutionContext(runtime=IsaacSdgRuntime(), stage=object())
        executor = IsaacInProcessExecutor(
            create_default_isaac_handler_registry(),
            context=context,
        )
        result = asyncio.run(executor.execute_async(JobOrderPlan(orders=(order,))))
        self.assertTrue(result.success)
        json.dumps(asdict(result))
        self.assertEqual(context.state["appearance.lighting"]["modus"], intent)
        self.assertIsInstance(order.parameters["modus"]["elevations"], MappingProxyType)


if __name__ == "__main__":
    unittest.main()
