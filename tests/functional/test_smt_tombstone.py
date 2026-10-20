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

"""Exercise one SMT tombstone through the CDF container API.

Compile the JobOrder plan from the repository root::

    python tests/functional/test_smt_tombstone.py

Execute the plan inside the candidate container::

    docker run --rm --gpus all <candidate-image> \
        job-script tests/functional/test_smt_tombstone.py --execute
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cdf import CDFJob
from tests.functional.smt_scenario import SmtFunctionalScenario


def _configure_tombstone(job: CDFJob, selection_ratio: float) -> None:
    """Inject the tombstone-specific operation into the shared SMT job.

    Args:
        job: Shared SMT job receiving the tombstone declaration.
        selection_ratio: Ratio that selects the coupon component.

    """
    job.operations.add_tombstone(
        ratio=selection_ratio,
        angle_range=(70, 90),
        auto_filter=True,
        eligibility_source="usd",
    )


SCENARIO = SmtFunctionalScenario(
    name="smt_tombstone",
    description=__doc__ or "SMT tombstone functional test",
    defect_label="tombstone",
    operation_type="operation.tombstone.apply",
    configure_operations=_configure_tombstone,
    output_subdir="tombstone_closeups",
)


def build_job(
    *,
    resolution: tuple[int, int] = (1920, 1080),
    total_spp: int = 32,
) -> tuple[CDFJob, int, float]:
    """Build the single-component SMT tombstone functional job.

    Args:
        resolution: Capture width and height.
        total_spp: Total path-tracing samples per pixel.

    Returns:
        Authored job, eligible component count, and tombstone ratio.

    Example:

    .. code-block:: python

        job, eligible_count, ratio = build_job(
            resolution=(320, 180),
            total_spp=2,
        )
        print(eligible_count, ratio)
        # 1 1.0

    """
    return SCENARIO.build_job(resolution=resolution, total_spp=total_spp)


def main() -> int:
    """Compile or execute the SMT tombstone functional job.

    Returns:
        Process exit status.

    """
    return SCENARIO.main()


if __name__ == "__main__":
    sys.exit(main())
