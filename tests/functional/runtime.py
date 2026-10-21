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

"""Shared Isaac Sim lifecycle and failure reporting for functional tests."""

from __future__ import annotations

import asyncio
import json
import math
import sys
import traceback
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from cdf import JobExecutionError, JobResult

ERROR_ARTIFACT_NAME = "functional-test-error.json"
MISSING_FIXTURE_EXIT_CODE = 77


def _skip_missing_fixture(fixture: Path) -> int | None:
    """Return a skip status when a release-excluded fixture is unavailable.

    Args:
        fixture: Coupon asset required by the functional workflow.

    Returns:
        None when the fixture exists, otherwise the functional-test skip code.

    """
    if fixture.is_file():
        return None
    print(f"SKIP: approved board fixture is not shipped: {fixture.name}")
    print("Provide an approved board asset to enable this functional workflow.")
    return MISSING_FIXTURE_EXIT_CODE


def _validate_generated_meshes(
    stage: object,
    generated_prim_paths: list[str],
) -> tuple[str, ...]:
    """Validate generated USD meshes on a supplied stage.

    Args:
        stage: Live USD stage containing the generated solder prims.
        generated_prim_paths: Paths reported by the solder attachment manifest.

    Returns:
        Validated mesh prim paths.

    Raises:
        RuntimeError: If no valid generated mesh is present on the stage.

    """
    from pxr import Usd, UsdGeom

    mesh_prims: dict[str, object] = {}
    missing_roots = []
    for root_path in generated_prim_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            missing_roots.append(root_path)
            continue
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                mesh_prims[str(prim.GetPath())] = prim

    problems = []
    if missing_roots:
        problems.append(f"missing generated prims {sorted(missing_roots)!r}")
    if not mesh_prims:
        problems.append("attachment manifest resolves to no USD mesh prims")

    for mesh_path, prim in sorted(mesh_prims.items()):
        mesh = UsdGeom.Mesh(prim)
        points = list(mesh.GetPointsAttr().Get() or ())
        counts = [int(value) for value in mesh.GetFaceVertexCountsAttr().Get() or ()]
        indices = [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get() or ()]
        if not points:
            problems.append(f"{mesh_path} has no points")
            continue
        if not counts or any(count < 3 for count in counts):
            problems.append(f"{mesh_path} has no valid polygon faces")
        if sum(counts) != len(indices):
            problems.append(f"{mesh_path} face counts do not match its indices")
        if any(index < 0 or index >= len(points) for index in indices):
            problems.append(f"{mesh_path} has out-of-range face indices")
        if any(not math.isfinite(float(value)) for point in points for value in point):
            problems.append(f"{mesh_path} has non-finite point coordinates")

    if problems:
        raise RuntimeError("invalid generated solder geometry: " + "; ".join(problems))
    return tuple(sorted(mesh_prims))


def _require_generated_meshes(generated_prim_paths: list[str]) -> tuple[str, ...]:
    """Require generated USD meshes with finite, nonempty valid topology.

    Args:
        generated_prim_paths: Paths reported by the solder attachment manifest.

    Returns:
        Validated mesh prim paths.

    Raises:
        RuntimeError: If the live stage or valid generated geometry is absent.

    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("solder mesh validation found no live USD stage")
    return _validate_generated_meshes(stage, generated_prim_paths)


def _require_semantic_capture(
    output_dir: Path,
    *,
    expected_label: str | None = None,
) -> dict[str, object]:
    """Require aligned RGB, semantic-mask, and semantic-label artifacts.

    Args:
        output_dir: Capture directory containing the first trigger.
        expected_label: Semantic value that must appear in the label mapping.

    Returns:
        Paths, dimensions, and semantic values validated by the check.

    Raises:
        RuntimeError: If the capture artifacts are missing, empty, or misaligned.

    """
    from PIL import Image

    trigger_dir = output_dir / "trigger_0000"
    rgb_path = trigger_dir / "rgb_0000.png"
    mask_path = trigger_dir / "semantic_segmentation_0000.png"
    labels_path = trigger_dir / "semantic_segmentation_labels_0000.json"
    missing = [
        str(path)
        for path in (rgb_path, mask_path, labels_path)
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            f"semantic capture artifacts are missing or empty: {missing!r}"
        )

    with Image.open(rgb_path) as rgb_image, Image.open(mask_path) as mask_image:
        rgb_size = rgb_image.size
        mask_size = mask_image.size
        mask_bbox = mask_image.getbbox()
    if rgb_size != mask_size:
        raise RuntimeError(
            f"semantic mask size {mask_size!r} does not match RGB size {rgb_size!r}",
        )
    if mask_bbox is None:
        raise RuntimeError("semantic mask contains no labelled pixels")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict) or not labels:
        raise RuntimeError("semantic label sidecar is not a nonempty mapping")
    semantic_values = {
        str(nested_value)
        for value in labels.values()
        for nested_value in (value.values() if isinstance(value, dict) else (value,))
    }
    if expected_label is not None and expected_label not in semantic_values:
        raise RuntimeError(
            f"semantic label {expected_label!r} is absent from {sorted(semantic_values)!r}",
        )
    return {
        "rgb": str(rgb_path),
        "mask": str(mask_path),
        "labels": str(labels_path),
        "size": rgb_size,
        "semantic_values": sorted(semantic_values),
    }


class _SimulationApp(Protocol):
    """Subset of ``SimulationApp`` used by the functional-test runtime."""

    def is_running(self) -> bool:
        """Return whether Kit is still running."""

    def update(self) -> None:
        """Advance Kit by one application update."""

    def close(self) -> None:
        """Close Kit and release its resources."""


class _Task(Protocol):
    """Subset of ``asyncio.Task`` used by the runtime loop."""

    def done(self) -> bool:
        """Return whether the asynchronous job has completed."""

    def cancel(self) -> bool:
        """Cancel the asynchronous job."""

    def result(self) -> JobResult:
        """Return the job result or raise its exception."""


def _failure_payload(exception: BaseException) -> dict[str, object]:
    """Build the JSON-compatible functional-test failure record.

    Args:
        exception: Exception raised by the asynchronous functional job.

    Returns:
        Failure details, including JobOrder context when available.

    """
    payload: dict[str, object] = {
        "success": False,
        "recorded_at": datetime.now(UTC).isoformat(),
        "exception": {
            "type": type(exception).__name__,
            "message": str(exception),
        },
        "traceback": traceback.format_exception(
            type(exception),
            exception,
            exception.__traceback__,
        ),
    }
    if not isinstance(exception, JobExecutionError):
        return payload

    payload["failed_order"] = {
        "id": exception.order.order_id,
        "type": exception.order.order_type,
    }
    payload["cause"] = {
        "type": type(exception.cause).__name__,
        "message": str(exception.cause),
    }
    payload["order_results"] = [
        {
            "id": result.order_id,
            "type": result.order_type,
            "status": result.status,
            "error": result.error,
        }
        for result in exception.result.order_results
    ]
    return payload


def _write_failure_artifact(output_dir: Path, exception: BaseException) -> Path:
    """Atomically write a structured functional-test failure artifact.

    Args:
        output_dir: Directory retained by CI on failure.
        exception: Exception raised by the asynchronous functional job.

    Returns:
        Path to the completed JSON artifact.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / ERROR_ARTIFACT_NAME
    temporary_path = output_dir / f".{ERROR_ARTIFACT_NAME}.tmp"
    temporary_path.write_text(
        json.dumps(_failure_payload(exception), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(artifact_path)
    return artifact_path


def _run_task(
    simulation_app: _SimulationApp,
    task: _Task,
    output_dir: Path,
) -> JobResult:
    """Drive an asynchronous functional job and close Kit safely.

    Args:
        simulation_app: Running Isaac Sim application.
        task: Scheduled functional job.
        output_dir: Directory retained by CI on failure.

    Returns:
        Successful functional job result.

    Raises:
        BaseException: Re-raises the functional job failure after recording it.
        RuntimeError: If Isaac Sim stops before the job completes.

    """
    try:
        while simulation_app.is_running() and not task.done():
            simulation_app.update()
        if not task.done():
            task.cancel()
            raise RuntimeError("Isaac Sim stopped before the functional job completed")
        return task.result()
    except BaseException as exception:
        traceback.print_exception(
            type(exception),
            exception,
            exception.__traceback__,
            file=sys.stderr,
        )
        try:
            artifact_path = _write_failure_artifact(output_dir, exception)
            print(f"Functional failure recorded at {artifact_path}", file=sys.stderr)
        # Failure reporting must never replace the original job exception.
        except Exception as artifact_exception:  # noqa: BLE001
            print(
                "Failed to write functional-test error artifact: "
                f"{type(artifact_exception).__name__}: {artifact_exception}",
                file=sys.stderr,
            )
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()


def execute_in_isaac(
    execute_job: Callable[[], Awaitable[JobResult]],
    output_dir: Path,
    *,
    windowed: bool,
    resolution: tuple[int, int],
) -> JobResult:
    """Boot Isaac Sim and execute a functional job with failure reporting.

    Args:
        execute_job: Factory returning the asynchronous functional job.
        output_dir: Directory retained by CI on failure.
        windowed: Whether to show the Isaac Sim window.
        resolution: Width and height supplied to the application window.

    Returns:
        Structured result from the in-process executor.

    """
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not windowed,
            "width": resolution[0],
            "height": resolution[1],
            "renderer": "PathTracing",
        },
    )
    task = asyncio.ensure_future(execute_job())
    return _run_task(simulation_app, task, output_dir)
