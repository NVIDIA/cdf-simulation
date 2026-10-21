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

"""Execute a JobOrder plan in an isolated Isaac Sim Python process."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from ...errors import CDFError, ValidationError
from ..executor_base import JobExecutor
from ..job_order_codec import JobOrderCodec
from ..job_order_types import JobOrderPlan
from ..job_result_types import JobOrderResult, JobResult


def resolve_isaac_python(isaac_path: str | Path | None = None) -> Path:
    """Resolve the Python launcher bundled with Isaac Sim.

    Args:
        isaac_path: Isaac installation directory or a launcher within it.
            When omitted, ``ISAAC_SIM_PATH`` is used.

    Returns:
        Existing Isaac Sim Python launcher.

    Raises:
        ValidationError: If an Isaac installation or Python launcher cannot be
            resolved.

    """
    configured = isaac_path or os.environ.get("ISAAC_SIM_PATH")
    if not configured:
        raise ValidationError(
            "capture.render requires ISAAC_SIM_PATH to point to the Isaac Sim "
            "installation directory or bundled Python launcher"
        )

    # Strip first: a batch file's `set VAR=value && ...` keeps the blank before
    # the `&&`, and the resulting error names a path with a space wedged into
    # the middle of it (`C:\isaac-sim-6.0 \python.bat`), which reads as
    # anything but a trailing space.
    path = Path(str(configured).strip()).expanduser()
    if path.is_file():
        if path.name.lower() in {"isaac-sim.bat", "isaac-sim.sh"}:
            path = path.with_name("python.bat" if os.name == "nt" else "python.sh")
        elif path.suffix.lower() not in {".bat", ".sh", ".exe"}:
            raise ValidationError("Isaac Python launcher is not executable: %s" % path)
    elif path.is_dir():
        names = (
            ("python.bat", "python.exe")
            if os.name == "nt"
            else (
                "python.sh",
                "python",
            )
        )
        path = next(
            (path / name for name in names if (path / name).is_file()), path / names[0]
        )

    if not path.is_file():
        raise ValidationError("Isaac Python launcher does not exist: %s" % path)
    return path.resolve()


def _decode_result(payload: object) -> JobResult:
    """Validate and reconstruct a transported JobResult."""
    if not isinstance(payload, dict) or not isinstance(payload.get("success"), bool):
        raise CDFError("Isaac child returned an invalid JobResult payload")
    raw_results = payload.get("order_results", [])
    if not isinstance(raw_results, list):
        raise CDFError("Isaac child returned invalid order_results")
    try:
        results = tuple(JobOrderResult(**item) for item in raw_results)
    except (TypeError, ValueError) as error:
        raise CDFError("Isaac child returned invalid order result data") from error
    startup = payload.get("startup_s", 0.0)
    if not isinstance(startup, (int, float)):
        raise CDFError("Isaac child returned an invalid startup duration")
    return JobResult(
        success=payload["success"],
        order_results=results,
        startup_s=float(startup),
    )


def _decode_batch_result(payload: object) -> tuple[JobResult, ...]:
    """Validate and reconstruct a transported sequence of JobResults."""
    if not isinstance(payload, dict) or not isinstance(payload.get("success"), bool):
        raise CDFError("Isaac child returned an invalid batch payload")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise CDFError("Isaac child returned invalid batch results")
    return tuple(_decode_result(item) for item in raw_results)


class IsaacSubprocessExecutor(JobExecutor):
    """Transport and execute a plan using Isaac Sim's bundled Python.

    The caller only needs ordinary Python and CDF's authoring dependencies.
    Isaac-specific modules are imported exclusively by the child process.

    Args:
        resolution: Width and height supplied to the Isaac application window.
        windowed: Whether the child displays the Isaac application window.
        experience: Optional Kit experience file for the child to boot.
        isaac_path: Optional Isaac installation or bundled Python launcher.
        timeout: Optional child-process timeout in seconds.

    """

    def __init__(
        self,
        *,
        resolution: tuple[int, int],
        windowed: bool = False,
        experience: str | None = None,
        isaac_path: str | Path | None = None,
        timeout: float | None = None,
    ) -> None:
        self.resolution = resolution
        self.windowed = windowed
        self.experience = experience
        self.isaac_path = isaac_path
        self.timeout = timeout

    def execute(self, plan: JobOrderPlan) -> JobResult:
        """Serialize and execute a JobOrder plan in an Isaac child process."""
        return _decode_result(self._run_child("--plan", JobOrderCodec.dumps(plan)))

    def execute_batch(
        self,
        plans: "Sequence[JobOrderPlan]",
    ) -> tuple[JobResult, ...]:
        """Execute several plans in ONE Isaac child process.

        The saving is not the boot. Measured on a KiCad board, a render costs
        101.5 s in a fresh process and 4.8 s as the second render in the same
        one, because the ~98 s the replicator orchestrator waits out is RTX
        pipeline and shader compilation and it is paid once per process.

        A plan that fails does not end the batch; its failure is reported in
        its own result. Only a child that dies without writing anything raises.

        Args:
            plans: Validated plans to execute in order.

        Returns:
            One result per plan, in the order given. Empty for an empty batch,
            which never launches a child.

        Raises:
            CDFError: If the child fails to return a usable batch envelope.

        Example:

        .. code-block:: python

            results = executor.execute_batch([first, second])  # doctest: +SKIP
            print(len(results))  # doctest: +SKIP

        """
        ordered = tuple(plans)
        if not ordered:
            return ()
        document = "[%s]" % ",".join(JobOrderCodec.dumps(plan) for plan in ordered)
        return _decode_batch_result(self._run_child("--plans", document))

    def _run_child(self, flag: str, document: str) -> object:
        """Launch the Isaac child on one transport document and return its result."""
        launcher = resolve_isaac_python(self.isaac_path)
        repository_root = Path(__file__).resolve().parents[3]

        with tempfile.TemporaryDirectory(prefix="cdf-isaac-") as temp_dir:
            transport_dir = Path(temp_dir)
            plan_path = transport_dir / "plan.json"
            result_path = transport_dir / "result.json"
            plan_path.write_text(document, encoding="utf-8")
            command = [
                str(launcher),
                "-m",
                "cdf.execution.isaac.subprocess_entrypoint",
                flag,
                str(plan_path),
                "--result",
                str(result_path),
                "--resolution",
                str(self.resolution[0]),
                str(self.resolution[1]),
            ]
            for module_name in _registered_domain_modules():
                command += ["--domain-module", module_name]
            if self.windowed:
                command.append("--windowed")
            if self.experience:
                command.extend(("--experience", str(self.experience)))

            completed = subprocess.run(
                command,
                cwd=repository_root,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            if not result_path.is_file():
                diagnostic = (completed.stderr or completed.stdout).strip()
                if len(diagnostic) > 4000:
                    diagnostic = diagnostic[-4000:]
                raise CDFError(
                    "Isaac child exited with code %d without a result%s"
                    % (completed.returncode, ": " + diagnostic if diagnostic else "")
                )

            envelope = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict):
                raise CDFError("Isaac child returned an invalid result envelope")
            if not envelope.get("success"):
                child_traceback = envelope.get("traceback")
                traceback_suffix = (
                    "\n" + child_traceback.strip()
                    if isinstance(child_traceback, str) and child_traceback.strip()
                    else ""
                )
                raise CDFError(
                    "Isaac child failed: %s: %s%s"
                    % (
                        envelope.get("error_type", "Error"),
                        envelope.get("error", ""),
                        traceback_suffix,
                    )
                )
            if completed.returncode != 0:
                raise CDFError("Isaac child exited with code %d" % completed.returncode)
            return envelope.get("result")


def _registered_domain_modules() -> tuple[str, ...]:
    """Return the import names of every domain module registered in THIS process.

    A domain contributes its Isaac handlers as an import side effect -- importing
    ``pcb_aoi`` is what registers ``operation.pin_variation.apply``. The parent
    imports the domain to compile the plan, but the Isaac child was only ever
    given the plan, so domain-owned orders compiled fine and then failed in the
    child with "no JobOrder handler registered". Passing the module names across
    is what makes the child's registry match the one the plan was compiled against.

    Returns:
        Module names, sorted, safe to `import` in the child.
    """
    import sys

    from cdf.domain import DomainSpec

    names = set()
    for module_name, module in list(sys.modules.items()):
        if module is None or "." in module_name:
            continue
        spec = getattr(module, "DOMAIN", None)
        if isinstance(spec, DomainSpec):
            names.add(module_name)
    return tuple(sorted(names))
