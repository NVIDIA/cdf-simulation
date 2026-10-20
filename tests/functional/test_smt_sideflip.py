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

"""Exercise one SMT sideflip through the CDF container API."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cdf import CDFJob
from tests.functional.smt_scenario import (
    SmtFunctionalScenario,
)


def _configure_sideflip(job: CDFJob, selection_ratio: float) -> None:
    """Inject the sideflip-specific operation into the shared SMT job.

    Args:
        job: Shared SMT job receiving the sideflip declaration.
        selection_ratio: Ratio that selects the coupon component.

    """
    job.operations.add_sideflip(
        ratio=selection_ratio,
        angle_range=(75.0, 75.0),
    )


SCENARIO = SmtFunctionalScenario(
    name="smt_sideflip",
    description=__doc__ or "SMT sideflip functional test",
    defect_label="sideflip",
    operation_type="operation.sideflip.apply",
    configure_operations=_configure_sideflip,
    output_subdir="sideflip_closeups",
)


def build_job(
    *,
    resolution: tuple[int, int] = (1920, 1080),
    total_spp: int = 32,
) -> tuple[CDFJob, int, float]:
    """Build the single-component SMT sideflip functional job.

    Args:
        resolution: Capture width and height.
        total_spp: Total path-tracing samples per pixel.

    Returns:
        Authored job, eligible component count, and sideflip ratio.

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
    """Compile or execute the SMT sideflip functional job.

    Returns:
        Process exit status.

    """
    return SCENARIO.main()


if __name__ == "__main__":
    sys.exit(main())
