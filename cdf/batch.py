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

"""Render several jobs through one Isaac process.

A dataset is several variants of one board, and building it one
``job.execute()`` at a time pays the same fixed cost per variant. Measured on a
KiCad board: 101.5 s for a render in a fresh process, 4.8 s for the second
render in the same one. Roughly 40 s of the difference is booting Kit and
roughly 98 s is RTX pipeline and shader compilation, which the replicator
orchestrator waits out on the first render and never again.

So::

    from cdf import BatchJob, render_batch

    results = render_batch([
        BatchJob(golden_job, output_dir=out / "golden"),
        BatchJob(tombstone_job, output_dir=out / "tombstone"),
    ])

Seven variants that cost about seventeen minutes as seven processes cost about
three as one. Nothing about how a job is built changes: `job.execute` and
`job.capture.render` are untouched, and a batch of one behaves like either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import ValidationError
from .execution.job_result_types import JobResult

if TYPE_CHECKING:
    from .job import CDFJob


@dataclass(frozen=True)
class BatchJob:
    """One job in a batch, with the destinations `job.execute` would take.

    Args:
        job: Built job to compile and run.
        output_dir: Destination required by a job with capture declarations.
        serialization_path: Optional final composed-stage USD destination.

    Example:

    .. code-block:: python

        item = BatchJob(job, output_dir="out/golden")
        print(item.output_dir)
        # out/golden

    """

    job: CDFJob
    #: Built job to compile and run.

    output_dir: str | Path | None = None
    #: Destination required by a job with capture declarations.

    serialization_path: str | Path | None = None
    #: Optional final composed-stage USD destination.


def render_batch(
    items: Sequence[BatchJob],
    *,
    windowed: bool = False,
    experience: str | None = None,
    isaac_path: str | Path | None = None,
    timeout: float | None = None,
) -> tuple[JobResult, ...]:
    """Execute several jobs in one Isaac Sim process.

    Every job is validated and compiled BEFORE the child launches, so a batch
    that cannot run costs nothing rather than failing four minutes in. Once it
    is running, a job that fails does not end the batch: the boot and the
    shader compilation the remaining jobs need are already paid for, and the
    failure is reported in that job's own result.

    The Isaac window is per process, so its resolution is the largest any job
    asks for. A job declaring no capture contributes nothing to it.

    Args:
        items: Jobs and their destinations, executed in the order given.
        windowed: Whether to display the Isaac application window.
        experience: Optional Kit experience file. `apps/pcb_aoi.sdg.kit` boots
            29 fewer extensions for a bit-identical frame.
        isaac_path: Optional Isaac installation or bundled Python launcher.
        timeout: Optional child-process timeout in seconds, for the whole batch.

    Returns:
        One result per item, in the order given. The first carries the boot
        cost in ``startup_s``; the rest carry zero, because they did not pay it.

    Raises:
        ValidationError: If an entry is not a BatchJob, or a job's declarations
            and destinations disagree.
        CDFError: If the child fails to return a usable batch envelope.

    Example:

    .. code-block:: python

        print(render_batch([]))
        # ()

    """
    ordered = tuple(items)
    for position, item in enumerate(ordered):
        if not isinstance(item, BatchJob):
            raise ValidationError(
                "render_batch expects BatchJob at position %d" % position,
            )
    if not ordered:
        return ()

    from .execution.isaac import IsaacSubprocessExecutor

    plans = []
    width, height = 64, 64
    for item in ordered:
        has_capture = item.job._validate_execution_outputs(
            item.output_dir,
            item.serialization_path,
        )
        if has_capture:
            job_width, job_height = item.job.capture._render_resolution()
            width, height = max(width, job_width), max(height, job_height)
        plans.append(
            item.job.compile_orders(
                output_dir=item.output_dir,
                serialization_path=item.serialization_path,
            ),
        )

    return IsaacSubprocessExecutor(
        resolution=(width, height),
        windowed=windowed,
        experience=experience,
        isaac_path=isaac_path,
        timeout=timeout,
    ).execute_batch(plans)
