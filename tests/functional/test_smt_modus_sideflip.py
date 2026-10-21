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

"""Exercise one sideflipped SMT component under the Modus light.

This is the only functional scenario that renders with
``appearance.lighting(mode="modus_light")``: six RectLights on an arc,
white/red/blue centre-out, instead of the tri-colour ring every other
scenario uses. The sideflip defect is the visual subject -- rolling a chip
part onto its long axis changes which faces catch the red and blue arms, so
the frame exercises the rig's colour separation rather than just its
brightness.

Compile the JobOrder plan from the repository root::

    python tests/functional/test_smt_modus_sideflip.py

Execute the plan inside the candidate container::

    docker run --rm --gpus all <candidate-image> \\
        job-script tests/functional/test_smt_modus_sideflip.py --execute
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cdf import CDFJob  # noqa: E402
from tests.functional.smt_scenario import (  # noqa: E402
    SmtFunctionalScenario,
)


#: Modus arm intensities. Collapsed to single values (not ranges) so the run is
#: reproducible frame for frame, and separated enough that the gamut check can
#: tell the three arms apart in the rendered pixels.
MODUS_INTENSITIES = {"white": 6000.0, "red": 9000.0, "blue": 9000.0}

#: Saturated primaries on the red and blue arms. The white arm stays neutral, so
#: a correct render shows a warm side, a cool side and a neutral top.
MODUS_COLORS = {
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
}


def _configure_modus_appearance(job: CDFJob) -> None:
    """Declare the Modus rig instead of the shared AOI ring.

    Args:
        job: Shared SMT job receiving the appearance declarations.
    """
    job.appearance.lighting(
        mode="modus_light",
        exposure_range=(0.0, 0.0),
        modus={
            "arc_radius_scale": 1.5,
            "height_scale": 2.0,
            "width_frac": 0.12,
            "intensities": MODUS_INTENSITIES,
            "colors": MODUS_COLORS,
        },
    )
    job.appearance.components(vantablack=False)


def _configure_sideflip(job: CDFJob, selection_ratio: float) -> None:
    """Inject the sideflip-specific operation into the shared SMT job.

    Args:
        job: Shared SMT job receiving the sideflip declaration.
        selection_ratio: Ratio that selects the coupon component.
    """
    job.operations.add_sideflip(
        ratio=selection_ratio,
        angle_range=(70.0, 90.0),
    )


SCENARIO = SmtFunctionalScenario(
    name="smt_modus_sideflip",
    description=__doc__ or "SMT sideflip under the Modus light",
    defect_label="sideflip",
    operation_type="operation.sideflip.apply",
    configure_operations=_configure_sideflip,
    output_subdir="sideflip_closeups",
    configure_appearance=_configure_modus_appearance,
)


def build_job(
    *,
    resolution: tuple[int, int] = (1920, 1080),
    total_spp: int = 32,
) -> tuple[CDFJob, int, float]:
    """Build the Modus-lit SMT sideflip functional job.

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
    """Compile or execute the Modus-lit SMT sideflip functional job.

    Returns:
        Process exit status.

    """
    return SCENARIO.main()


if __name__ == "__main__":
    sys.exit(main())
