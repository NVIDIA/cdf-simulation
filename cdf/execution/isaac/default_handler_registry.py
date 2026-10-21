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

"""Register the built-in handlers that have concrete Isaac implementations."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from pydantic import BaseModel

from ...errors import ValidationError
from ..handler_registry import JobOrderHandlerRegistry
from ..job_order_types import JobOrder
from .domain_adapters import domain_handlers
from .execution_context import IsaacExecutionContext
from .job_order_parameters import (
    AugmentationParameters,
    CameraParameters,
    CaptureLimitParameters,
    CaptureOutputParameters,
    CaptureRenderParameters,
    CaptureVisibilityParameters,
    CaptureWriterParameters,
    LightingApplyParameters,
    LightingParameters,
    MaterialOverrideParameters,
    PostprocessCloseOpeningsParameters,
    PostprocessCropParameters,
    PostprocessStitchParameters,
    RendererParameters,
    RoiSelectionParameters,
    ScanGridParameters,
    SceneLoadParameters,
    SidecarLoadParameters,
    StageSerializationParameters,
)


def _require_context(context: object | None) -> IsaacExecutionContext:
    """Return the typed context required by built-in Isaac handlers.

    Args:
        context: Context supplied to the in-process executor.

    Returns:
        Typed Isaac execution context.

    Raises:
        ValidationError: If the executor received another context type.

    """
    if not isinstance(context, IsaacExecutionContext):
        raise ValidationError(
            "built-in Isaac handlers require IsaacExecutionContext",
        )
    return context


async def _invoke[Parameters: BaseModel](
    order: JobOrder,
    raw_context: object | None,
    contract: type[Parameters],
    runtime_method: str,
) -> object:
    """Validate an order, invoke its runtime adapter, and retain its output.

    Args:
        order: JobOrder being executed.
        raw_context: Context supplied by the executor.
        contract: Version-1 Pydantic parameter contract.
        runtime_method: Method name provided by the runtime adapter.

    Returns:
        Runtime adapter output.

    Raises:
        ValidationError: If the order version, context, or runtime is invalid.

    """
    if order.version != 1:
        raise ValidationError(
            f"{order.order_type} handler supports version 1, got {order.version}",
        )
    context = _require_context(raw_context)
    # Thaw nested order mappings before validation. Values typed as `object`
    # otherwise retain mapping proxies that cannot cross the result transport.
    parameters = contract.model_validate(order.model_dump(mode="json")["parameters"])
    method = getattr(context.runtime, runtime_method, None)
    if not callable(method):
        raise ValidationError(
            f"Isaac runtime {type(context.runtime).__name__} does not implement "
            f"{runtime_method}()",
        )
    context._active_order_id = order.order_id
    try:
        output = method(context, parameters)
        if inspect.isawaitable(output):
            output = await output
    finally:
        context._active_order_id = None
    context.outputs[order.order_id] = output
    return output


async def handle_scene_load(
    order: JobOrder,
    context: object | None,
) -> object:
    """Execute a version-1 ``scene.load`` order.

    Args:
        order: Scene-load JobOrder.
        context: Isaac execution context.

    Returns:
        Runtime scene-load output.

    """
    return await _invoke(order, context, SceneLoadParameters, "load_scene")


async def handle_sidecar_load(
    order: JobOrder,
    context: object | None,
) -> object:
    """Execute a version-1 ``scene.sidecar.load`` order.

    Args:
        order: Sidecar-load JobOrder.
        context: Isaac execution context.

    Returns:
        Runtime sidecar-load output.

    """
    return await _invoke(order, context, SidecarLoadParameters, "load_sidecar")


def _runtime_handler[Parameters: BaseModel](
    contract: type[Parameters],
    runtime_method: str,
) -> Callable:
    """Create an async handler for one validated runtime adapter method.

    Args:
        contract: Version-1 Pydantic parameter contract.
        runtime_method: Runtime adapter method that implements the order.

    Returns:
        Async JobOrder handler.

    """

    async def handler(order: JobOrder, context: object | None) -> object:
        return await _invoke(order, context, contract, runtime_method)

    return handler


def create_default_isaac_handler_registry() -> JobOrderHandlerRegistry:
    """Create the registry of built-in handlers implemented so far.

    Returns:
        Registry containing runtime and context-owned configuration handlers.

    Example:

    .. code-block:: python

        registry = create_default_isaac_handler_registry()
        print("scene.load" in registry.handlers)
        # True

    """
    handlers: dict[str, Callable] = {
        "scene.load": handle_scene_load,
        "scene.sidecar.load": handle_sidecar_load,
        "appearance.lighting.configure": _runtime_handler(
            LightingParameters,
            "configure_lighting",
        ),
        "appearance.lighting.apply": _runtime_handler(
            LightingApplyParameters,
            "apply_lighting",
        ),
        "appearance.material.override": _runtime_handler(
            MaterialOverrideParameters,
            "apply_material_override",
        ),
        "stage.serialize": _runtime_handler(
            StageSerializationParameters,
            "serialize_stage",
        ),
        "capture.output.configure": _runtime_handler(
            CaptureOutputParameters,
            "configure_capture_output",
        ),
        "capture.renderer.configure": _runtime_handler(
            RendererParameters,
            "configure_renderer",
        ),
        "capture.camera.configure": _runtime_handler(
            CameraParameters,
            "configure_camera",
        ),
        "capture.scan_grid.configure": _runtime_handler(
            ScanGridParameters,
            "configure_scan_grid",
        ),
        "capture.roi_selection.configure": _runtime_handler(
            RoiSelectionParameters,
            "configure_roi_selection",
        ),
        "capture.limits.configure": _runtime_handler(
            CaptureLimitParameters,
            "configure_capture_limits",
        ),
        "capture.visibility.configure": _runtime_handler(
            CaptureVisibilityParameters,
            "configure_capture_visibility",
        ),
        "capture.outputs.configure": _runtime_handler(
            CaptureWriterParameters,
            "configure_writer",
        ),
        "capture.augmentation.configure": _runtime_handler(
            AugmentationParameters,
            "configure_augmentation",
        ),
        "capture.render": _runtime_handler(
            CaptureRenderParameters,
            "render_capture",
        ),
        "postprocess.crop": _runtime_handler(
            PostprocessCropParameters,
            "crop_capture",
        ),
        "postprocess.close_openings": _runtime_handler(
            PostprocessCloseOpeningsParameters,
            "close_capture_openings",
        ),
        "postprocess.stitch": _runtime_handler(
            PostprocessStitchParameters,
            "stitch_capture",
        ),
    }
    registry = JobOrderHandlerRegistry(handlers)
    # Domains contribute their own orders. The executor machinery stays here;
    # the contracts and implementations belong to whoever invented the order.
    for order_type, handler in domain_handlers().items():
        registry.register(order_type, handler)
    return registry
