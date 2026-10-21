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

"""Exercise one reversed-polarity SMT component through the container API."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cdf import CDFJob
from tests.functional.smt_scenario import (
    RESOURCE_DIR,
    SmtFunctionalScenario,
)

COMPONENT_TYPE = "LED_0603_1608Metric_G"


def _configure_reverse_polarity(job: CDFJob, selection_ratio: float) -> None:
    """Inject the reverse-polarity operation into the shared SMT job.

    Args:
        job: Shared SMT job receiving the reverse-polarity declaration.
        selection_ratio: Ratio that selects the coupon component.

    """
    job.operations.add_reverse_polarity(
        ratio=selection_ratio,
    )


SCENARIO = SmtFunctionalScenario(
    name="smt_reverse_polarity",
    description=__doc__ or "SMT reverse-polarity functional test",
    defect_label="reverse_polarity",
    operation_type="operation.reverse_polarity.apply",
    configure_operations=_configure_reverse_polarity,
    output_subdir="reverse_polarity_closeups",
    board_usd=RESOURCE_DIR / "ARTIX_LED_D1_coupon.usdc",
    component_type=COMPONENT_TYPE,
)


def build_job(
    *,
    resolution: tuple[int, int] = (1920, 1080),
    total_spp: int = 32,
) -> tuple[CDFJob, int, float]:
    """Build the single-component reverse-polarity functional job.

    Args:
        resolution: Capture width and height.
        total_spp: Total path-tracing samples per pixel.

    Returns:
        Authored job, eligible component count, and selection ratio.

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
    """Compile or execute the reverse-polarity functional job.

    Returns:
        Process exit status.

    """
    return SCENARIO.main()


if __name__ == "__main__":
    sys.exit(main())
