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

"""Verify the public Isaac application runner uses JobOrder execution."""

import asyncio
import sys
import types
import unittest
from unittest import mock

from cdf import JobResult
from cdf.execution.isaac import run_plan_in_isaac


class _SimulationApp:
    """Drive the host event loop while recording application lifecycle."""

    def __init__(self, settings: dict[str, object]) -> None:
        self.settings = settings
        self.closed = False

    def is_running(self) -> bool:
        """Report that the test application remains active."""
        return True

    def update(self) -> None:
        """Advance the asyncio task by one turn."""
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0))

    def close(self) -> None:
        """Record application shutdown."""
        self.closed = True


class IsaacAppRunnerTest(unittest.TestCase):
    """Verify direct execution without the legacy render adapter."""

    def test_run_plan_uses_executor_and_closes_application(self):
        expected = JobResult(success=True)
        plan = object()
        executor = mock.Mock()
        executor.execute_async = mock.AsyncMock(return_value=expected)
        on_complete = mock.Mock()
        application = _SimulationApp({})
        isaacsim = types.ModuleType("isaacsim")
        isaacsim.SimulationApp = mock.Mock(return_value=application)

        with (
            mock.patch.dict(sys.modules, {"isaacsim": isaacsim}),
            mock.patch(
                "cdf.execution.isaac.app_runner.create_default_isaac_executor",
                return_value=executor,
            ),
        ):
            result = run_plan_in_isaac(
                plan,
                resolution=(1200, 800),
                on_complete=on_complete,
            )

        # The runner returns the executor result enriched with the boot cost,
        # so it is the same result rather than the same object.
        self.assertEqual(result.success, expected.success)
        self.assertEqual(result.order_results, expected.order_results)
        self.assertGreater(result.startup_s, 0.0)
        isaacsim.SimulationApp.assert_called_once_with(
            {
                "headless": True,
                "hide_ui": True,
                "width": 1200,
                "height": 800,
                "renderer": "RaytracedLighting",
                "create_new_stage": False,
                "multi_gpu": False,
            },
        )
        executor.execute_async.assert_awaited_once_with(plan)
        on_complete.assert_called_once_with(result, None)
        self.assertTrue(application.closed)


if __name__ == "__main__":
    unittest.main()
