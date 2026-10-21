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

"""Isaac Sim child-process entry point for JobOrder transport."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from ..job_order_codec import JobOrderCodec
from ..job_result_types import JobResult


def _arguments() -> argparse.Namespace:
    """Parse transport arguments before Isaac Sim consumes process options."""
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan", type=Path)
    source.add_argument("--plans", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--resolution", type=int, nargs=2, required=True)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--domain-module", action="append", default=[])
    parser.add_argument("--experience", default=None)
    return parser.parse_args()


def main() -> int:
    """Deserialize, execute, and return a JobResult transport envelope."""
    arguments = _arguments()
    # Kit also parses `sys.argv`; hide CDF's transport-only arguments from it.
    sys.argv = [sys.argv[0]]

    def write_completion(
        result: JobResult | tuple[JobResult, ...] | None,
        error: BaseException | None,
    ) -> None:
        """Persist completion before ``SimulationApp.close`` can exit Kit.

        A batch writes ``results``, a single plan writes ``result``. The two
        envelopes stay distinct rather than one becoming a list of one, so a
        reader never has to guess which shape it is holding.
        """
        if error is None and isinstance(result, tuple):
            envelope = {
                "success": True,
                "result": {"success": True, "results": [asdict(r) for r in result]},
            }
        elif error is None and result is not None:
            envelope = {"success": True, "result": asdict(result)}
        else:
            envelope = {
                "success": False,
                "error_type": type(error).__name__ if error else "RuntimeError",
                "error": str(error or "Isaac execution returned no result"),
                "traceback": "".join(traceback.format_exception(error))
                if error
                else "",
            }
        arguments.result.parent.mkdir(parents=True, exist_ok=True)
        arguments.result.write_text(
            json.dumps(envelope, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    try:
        # A domain registers its Isaac handlers as an import side effect, so the
        # child must import the same domain modules the parent compiled against
        # or domain-owned orders arrive with no handler to run them.
        import importlib

        for module_name in arguments.domain_module:
            importlib.import_module(module_name)

        from .app_runner import run_plan_in_isaac, run_plans_in_isaac

        if arguments.plans is not None:
            documents = json.loads(arguments.plans.read_text(encoding="utf-8"))
            if not isinstance(documents, list):
                raise ValueError("--plans must contain a JSON array of plans")
            plans = tuple(
                JobOrderCodec.loads(json.dumps(document).encode("utf-8"))
                for document in documents
            )
            result = run_plans_in_isaac(
                plans,
                resolution=tuple(arguments.resolution),
                windowed=arguments.windowed,
                experience=arguments.experience,
                on_complete=write_completion,
            )
        else:
            result = run_plan_in_isaac(
                JobOrderCodec.loads(arguments.plan.read_bytes()),
                resolution=tuple(arguments.resolution),
                windowed=arguments.windowed,
                experience=arguments.experience,
                on_complete=write_completion,
            )
        if not arguments.result.is_file():
            write_completion(result, None)
        return_code = 0
    except Exception as error:  # noqa: BLE001 - child must transport all failures
        if not arguments.result.is_file():
            write_completion(None, error)
        return_code = 1
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
