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

"""CDFJob — the declarative job model."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .domain import get_domain
from .sections import (
    AnnotationsSection,
    AppearanceSection,
    CaptureSection,
    OperationsSection,
    PostprocessSection,
    SceneSection,
)

if TYPE_CHECKING:
    from .execution.job_order_types import JobOrderPlan
    from .execution.job_result_types import JobResult


@dataclass(frozen=True)
class _JobDeclaration:
    """Validated CDF intent recorded in Python API call order."""

    order_id: str
    order_type: str
    parameters: Mapping[str, object]


class CDFJob:
    """Build a declarative job from six public areas.

    ``scene``, ``appearance``, ``operations``, ``annotations``, ``capture``,
    and ``postprocess``. Description calls add specifications to the job plan
    and never mutate the USD stage; everything is compiled, validated, and
    executed by :meth:`execute`. ``capture.render()`` remains a
    backwards-compatible convenience for jobs that declare capture passes.

    The ``domain`` names a registered :class:`~cdf.domain.DomainSpec`;
    domain modules register themselves on import (``import pcb_aoi``).
    """

    def __init__(self, *, domain, random_seed=0):
        """Initialize an empty job for a registered domain."""
        self.domain = get_domain(domain)
        self.random_seed = int(random_seed)
        #: Where this job's declarations came from, when a file authored them
        #: (``load_job`` stamps ``config_path`` / ``config_sha256``). Carried
        #: into ``JobOrderPlan.metadata["source"]`` so provenance can name the
        #: config a render was made from. ``None`` for jobs built in Python.
        self.source: dict[str, object] | None = None
        self._declarations: list[_JobDeclaration] = []
        self.scene = SceneSection(self)
        self.appearance = AppearanceSection(self)
        operations_factory = self.domain.operations_section_factory or OperationsSection
        self.operations = operations_factory(self)
        self.annotations = AnnotationsSection(self)
        capture_factory = self.domain.capture_section_factory or CaptureSection
        self.capture = capture_factory(self)
        self.postprocess = PostprocessSection(self)

    def _record_declaration(
        self,
        order_id: str,
        order_type: str,
        parameters: Mapping[str, object],
        *,
        replace: bool = True,
        preserve_position: bool = False,
    ) -> None:
        """Record validated intent without changing legacy section state."""
        declaration = _JobDeclaration(
            order_id=order_id,
            order_type=order_type,
            parameters=copy.deepcopy(dict(parameters)),
        )
        matching = [
            index
            for index, existing in enumerate(self._declarations)
            if existing.order_id == order_id
        ]
        if matching and not replace:
            raise ValueError(f"declaration {order_id!r} is already recorded")
        if matching and preserve_position:
            first = matching[0]
            self._declarations[first] = declaration
            for index in reversed(matching[1:]):
                del self._declarations[index]
            return
        for index in reversed(matching):
            del self._declarations[index]
        self._declarations.append(declaration)

    def compile(self, output_dir=None):
        """Normalize and validate the description into a legacy JobPlan.

        This does not write files or execute the plan.
        """
        from . import plan as _plan

        return _plan.compile_plan(self, output_dir=output_dir)

    def compile_orders(
        self,
        output_dir: str | Path | None = None,
        serialization_path: str | Path | None = None,
    ) -> JobOrderPlan:
        """Compile the description into executable intent without Isaac Sim.

        Python API declaration order is preserved except for declared
        postprocessing, which is placed after the synthetic terminal capture
        order. The legacy :meth:`compile` method remains unchanged.

        Args:
            output_dir: Output directory supplied to the capture order.
            serialization_path: Destination for the flattened composed USD stage.

        Returns:
            Immutable, validated JobOrder plan.

        Raises:
            ValidationError: If required declarations or their known hard
                prerequisites are missing or incorrectly ordered.

        Example:

        .. code-block:: python

            plan = job.compile_orders(output_dir="/tmp/output")
            print([order.order_type for order in plan.orders])

        """
        from .execution.job_order_compiler import _compile_job_orders

        return _compile_job_orders(
            self,
            output_dir=output_dir,
            serialization_path=serialization_path,
        )

    def _has_capture_declarations(self) -> bool:
        """Return whether the job contains any capture-pass declarations."""
        return any(
            declaration.order_type.startswith("capture.")
            for declaration in self._declarations
        )

    def _validate_execution_outputs(
        self,
        output_dir: str | Path | None,
        serialization_path: str | Path | None,
    ) -> bool:
        """Validate execution destinations and return capture presence."""
        from .errors import ValidationError

        has_capture = self._has_capture_declarations()
        if has_capture and output_dir is None:
            raise ValidationError(
                "job.execute requires output_dir when capture is configured",
            )
        if not has_capture and output_dir is not None:
            raise ValidationError(
                "job.execute output_dir requires a configured render pass",
            )
        if serialization_path is not None and not str(
            serialization_path
        ).lower().endswith(
            (".usd", ".usda", ".usdc"),
        ):
            raise ValidationError(
                "serialization_path must end with .usd, .usda, or .usdc",
            )
        return has_capture

    def execute(
        self,
        *,
        output_dir: str | Path | None = None,
        serialization_path: str | Path | None = None,
        windowed: bool = False,
        experience: str | Path | None = None,
    ) -> JobResult:
        """Execute every declared job area through an isolated Isaac process.

        Capture is optional. When capture declarations are absent, the job
        executes its scene, operation, annotation, and serialization orders
        without creating render products.

        Args:
            output_dir: Destination required by jobs with capture declarations.
            serialization_path: Optional final composed-stage USD destination.
            windowed: Whether to display the Isaac application window.
            experience: Optional Kit experience file for the Isaac child. This
                is the path the option is FOR: a batch amortizes the boot over
                jobs known in advance, which does not apply to one cold render.
                `apps/pcb_aoi.sdg.kit`
                boots 29 fewer extensions for a bit-identical frame; it has to
                sit in the Isaac install's `apps/` directory, because its
                extension-folder paths are relative to its own location.

        Returns:
            Structured result transported from the Isaac JobOrder executor.

        Raises:
            ValidationError: If execution outputs are missing or inconsistent.

        Example:

        .. code-block:: python

            job.operations.solder_joints(solver="procedural")
            result = job.execute(
                serialization_path="board_with_solder.usda",
                windowed=False,
            )
            print(result.success)
            # True

        """
        has_capture = self._validate_execution_outputs(
            output_dir,
            serialization_path,
        )
        resolution = self.capture._render_resolution() if has_capture else (64, 64)

        from .execution.isaac import IsaacSubprocessExecutor

        plan = self.compile_orders(
            output_dir=output_dir,
            serialization_path=serialization_path,
        )
        return IsaacSubprocessExecutor(
            resolution=resolution,
            windowed=windowed,
            experience=str(experience) if experience else None,
        ).execute(plan)
