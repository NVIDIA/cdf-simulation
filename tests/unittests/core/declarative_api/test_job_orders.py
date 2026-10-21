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

"""Kit-free tests for JobOrder planning, transport, and execution."""

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cdf import (  # noqa: E402
    IsaacInProcessExecutor,
    JobExecutionError,
    JobOrder,
    JobOrderCodec,
    JobOrderPlan,
    JobOrderResult,
    JobResult,
    ValidationError,
)


class JobOrderModelTest(unittest.TestCase):
    def test_order_is_deeply_immutable_and_normalizes_paths_and_sequences(self):
        source = {"path": Path("board.usd"), "range": [1, 2]}
        order = JobOrder(
            order_id="load",
            order_type="scene.load",
            parameters=source,
        )
        source["path"] = "changed.usd"
        source["range"].append(3)

        self.assertEqual(order.parameters["path"], "board.usd")
        self.assertEqual(order.parameters["range"], (1, 2))
        with self.assertRaises(TypeError):
            order.parameters["new"] = True

    def test_order_rejects_non_transport_values(self):
        with self.assertRaisesRegex(PydanticValidationError, "unsupported value"):
            JobOrder(
                order_id="bad",
                order_type="test.bad",
                parameters={"callback": lambda: None},
            )
        with self.assertRaisesRegex(PydanticValidationError, "finite"):
            JobOrder(
                order_id="bad",
                order_type="test.bad",
                parameters={"ratio": float("nan")},
            )
        with self.assertRaises(PydanticValidationError):
            JobOrder(
                order_id="bad",
                order_type="test.bad",
                depends_on="another-order",
            )

    def test_codec_is_stable_and_round_trips(self):
        first = JobOrderPlan(
            orders=(
                JobOrder(
                    order_id="lighting",
                    order_type="appearance.lighting",
                    parameters={"z": 3, "nested": {"b": 2, "a": 1}},
                ),
            ),
            metadata={"random_seed": 7},
        )
        second = JobOrderPlan(
            orders=(
                JobOrder(
                    order_id="lighting",
                    order_type="appearance.lighting",
                    parameters={"nested": {"a": 1, "b": 2}, "z": 3},
                ),
            ),
            metadata={"random_seed": 7},
        )

        encoded = JobOrderCodec.dumps(first)
        self.assertEqual(encoded, JobOrderCodec.dumps(second))
        decoded = JobOrderCodec.loads(encoded)
        self.assertEqual(decoded, first)
        self.assertEqual(decoded.ordered(), first.orders)

        model_payload = first.model_dump(mode="json", by_alias=True)
        self.assertEqual(model_payload["plan_version"], 1)
        self.assertEqual(JobOrderPlan.model_validate(model_payload), first)

    def test_pydantic_rejects_unknown_fields_and_frozen_assignment(self):
        with self.assertRaisesRegex(PydanticValidationError, "extra_forbidden"):
            JobOrder(order_id="bad", order_type="test.bad", unknown=True)

        order = JobOrder(order_id="load", order_type="scene.load")
        with self.assertRaisesRegex(PydanticValidationError, "frozen_instance"):
            order.order_type = "scene.other"


class JobOrderDependencyTest(unittest.TestCase):
    def test_dependencies_are_topologically_sorted_with_stable_ties(self):
        plan = JobOrderPlan(
            orders=(
                JobOrder(
                    order_id="capture",
                    order_type="capture.run",
                    depends_on=("appearance",),
                ),
                JobOrder(order_id="scene", order_type="scene.load"),
                JobOrder(order_id="metrics", order_type="metrics.start"),
                JobOrder(
                    order_id="appearance",
                    order_type="appearance.apply",
                    depends_on=("scene",),
                ),
            )
        )
        self.assertEqual(
            [order.order_id for order in plan.ordered()],
            ["scene", "metrics", "appearance", "capture"],
        )

    def test_duplicate_missing_and_cyclic_dependencies_are_rejected(self):
        cases = (
            (
                JobOrder(order_id="same", order_type="a"),
                JobOrder(order_id="same", order_type="b"),
            ),
            (
                JobOrder(
                    order_id="a",
                    order_type="a",
                    depends_on=("missing",),
                ),
            ),
            (
                JobOrder(order_id="a", order_type="a", depends_on=("b",)),
                JobOrder(order_id="b", order_type="b", depends_on=("a",)),
            ),
        )
        messages = ("duplicate", "do not exist", "cycle")
        for orders, message in zip(cases, messages):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValidationError, message):
                    JobOrderPlan(orders=orders).ordered()


class IsaacInProcessExecutorTest(unittest.TestCase):
    def test_executor_sequences_handlers_and_returns_structured_results(self):
        calls = []
        context = {"stage": "fake-stage"}

        def handle(order, received_context):
            calls.append((order.order_id, received_context))
            return {"handled": order.order_id}

        plan = JobOrderPlan(
            orders=(
                JobOrder(
                    order_id="capture",
                    order_type="capture.run",
                    depends_on=("scene",),
                ),
                JobOrder(order_id="scene", order_type="scene.load"),
            )
        )
        executor = IsaacInProcessExecutor(
            {
                "scene.load": handle,
                "capture.run": handle,
            },
            context=context,
        )

        result = executor.execute(plan)

        self.assertTrue(result.success)
        self.assertEqual([item[0] for item in calls], ["scene", "capture"])
        self.assertTrue(all(item[1] is context for item in calls))
        self.assertEqual(
            result.outputs,
            {
                "scene": {"handled": "scene"},
                "capture": {"handled": "capture"},
            },
        )

    def test_unknown_handler_fails_preflight_before_any_handler_runs(self):
        calls = []
        plan = JobOrderPlan(
            orders=(
                JobOrder(order_id="known", order_type="known"),
                JobOrder(
                    order_id="unknown",
                    order_type="unknown",
                    depends_on=("known",),
                ),
            )
        )
        executor = IsaacInProcessExecutor(
            {
                "known": lambda order, context: calls.append(order.order_id),
            }
        )

        with self.assertRaisesRegex(ValidationError, "unknown"):
            executor.execute(plan)
        self.assertEqual(calls, [])

    def test_handler_failure_carries_partial_result_and_stops_execution(self):
        calls = []

        def handle(order, context):
            calls.append(order.order_id)
            if order.order_id == "fail":
                raise RuntimeError("boom")
            return order.order_id

        plan = JobOrderPlan(
            orders=(
                JobOrder(order_id="first", order_type="work"),
                JobOrder(
                    order_id="fail",
                    order_type="work",
                    depends_on=("first",),
                ),
                JobOrder(
                    order_id="never",
                    order_type="work",
                    depends_on=("fail",),
                ),
            )
        )
        executor = IsaacInProcessExecutor({"work": handle})

        with self.assertRaises(JobExecutionError) as caught:
            executor.execute(plan)

        self.assertEqual(calls, ["first", "fail"])
        self.assertEqual(caught.exception.order.order_id, "fail")
        self.assertFalse(caught.exception.result.success)
        self.assertEqual(
            [item.status for item in caught.exception.result.order_results],
            ["succeeded", "failed"],
        )


class JobResultTest(unittest.TestCase):
    def test_public_accessors_expose_artifacts_and_failure_diagnostics(self):
        result = JobResult(
            success=False,
            order_results=(
                JobOrderResult(
                    order_id="capture.render",
                    order_type="capture.render",
                    status="succeeded",
                    output={"output_dir": "artifacts", "frame_count": 1},
                ),
                JobOrderResult(
                    order_id="postprocess.crop",
                    order_type="postprocess.crop",
                    status="succeeded",
                    output={"output_dir": "artifacts/cropped", "executed": True},
                ),
                JobOrderResult(
                    order_id="publish",
                    order_type="artifact.publish",
                    status="failed",
                    error="RuntimeError: unavailable",
                ),
            ),
        )

        self.assertEqual(result.capture_output["frame_count"], 1)
        self.assertTrue(result.crop_output["executed"])
        self.assertEqual(result.failed_order.order_id, "publish")
        self.assertEqual(
            result.output_for_type("artifact.publish"),
            None,
        )


if __name__ == "__main__":
    unittest.main()
