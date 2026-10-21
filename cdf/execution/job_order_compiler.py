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

"""Compile CDF Python declarations into domain-neutral JobOrders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..api_version import CDF_API_VERSION
from ..errors import ValidationError
from ..image_formats import WRITER_NATIVE_IMAGE_FORMATS
from ..job import _JobDeclaration
from .compiler_adapters import DomainCompilerAdapter, compiler_adapter
from .job_order_types import JobOrder, JobOrderPlan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..job import CDFJob


_REQUIRED_ORDER_IDS = ("scene.load",)

_REQUIRED_CAPTURE_TYPES = frozenset(
    {
        "capture.renderer.configure",
        "capture.camera.configure",
        "capture.scan_grid.configure",
        "capture.outputs.configure",
    },
)

_STATIC_LIGHTING_MODES = frozenset({"rgb_light", "modus_light"})


def _capture_pass_name(declaration: _JobDeclaration) -> str | None:
    """Return the render-pass name for one capture configuration declaration."""
    if not declaration.order_type.startswith("capture."):
        return None
    parts = declaration.order_id.split(".")
    if len(parts) == 2:
        return "default"
    if len(parts) == 3 and parts[0] == "capture":
        return parts[1]
    raise ValidationError(
        f"capture declaration has invalid order ID {declaration.order_id!r}"
    )


def _static_lighting_mode(
    declarations: Sequence[_JobDeclaration],
) -> str | None:
    """Return the configured static rig mode, if the job declares one."""
    lighting = next(
        (
            declaration
            for declaration in declarations
            if declaration.order_type == "appearance.lighting.configure"
        ),
        None,
    )
    if lighting is None:
        return None
    mode = str(lighting.parameters.get("mode"))
    return mode if mode in _STATIC_LIGHTING_MODES else None


def _capture_camera_side(
    declarations: Sequence[_JobDeclaration],
    pass_name: str,
) -> str:
    """Return the declared camera side for one capture pass."""
    camera = next(
        (
            declaration
            for declaration in declarations
            if declaration.order_type == "capture.camera.configure"
            and _capture_pass_name(declaration) == pass_name
        ),
        None,
    )
    return str(camera.parameters.get("side", "top")) if camera is not None else "top"


def _lighting_apply_order(
    pass_name: str | None,
    camera_side: str,
    dependencies: tuple[str, ...],
) -> JobOrder:
    """Build the internal order that authors one capture pass's static rig."""
    suffix = f".{pass_name}" if pass_name is not None else ""
    order_id = f"appearance.lighting{suffix}.apply"
    return JobOrder(
        order_id=order_id,
        order_type="appearance.lighting.apply",
        parameters={"camera_side": camera_side},
        depends_on=tuple(
            dependency for dependency in dependencies if dependency != order_id
        ),
    )


def _postprocess_rgb_source_passes(
    declaration: _JobDeclaration,
    capture_passes: Sequence[str],
) -> frozenset[str]:
    """Return the capture passes whose rgb frames a postprocess order opens.

    Narrower than the passes an order reads from at all: stitching a
    segmentation, closing openings in a stitched mask, cropping only
    annotation types and generating joint ROIs from bounding boxes read other
    artifacts of a pass and never open its rgb frames.

    Args:
        declaration: A recorded declaration of any type.
        capture_passes: Every render pass the job declares.

    Returns:
        The pass names the order reads rgb frames from; empty when it reads
        none.

    """
    parameters = declaration.parameters
    if declaration.order_type == "postprocess.crop":
        if "rgb" not in (parameters.get("types") or ()):
            return frozenset()
        return frozenset({"default"} & set(capture_passes))
    if declaration.order_type == "postprocess.stitch":
        if str(parameters.get("image_type", "rgb")) != "rgb":
            return frozenset()
        return frozenset({str(parameters.get("source_pass", "default"))})
    if declaration.order_type == "postprocess.format_roi_output":
        # The ROI pass contributes masks and label sidecars; only rgb_pass is
        # opened for its rgb tiles.
        return frozenset({str(parameters.get("rgb_pass", "full_board"))})
    return frozenset()


def _pass_output_dir(output_dir: str | Path | None, pass_name: str):
    """Return the destination for one named render pass."""
    if output_dir is None or pass_name == "default":
        return output_dir
    text = str(output_dir)
    if "://" in text:
        return f"{text.rstrip('/')}/{pass_name}"
    return Path(output_dir) / pass_name


def _validate_declarations(
    declarations: tuple[_JobDeclaration, ...],
    *,
    domain_adapter: DomainCompilerAdapter,
    has_final_serialization: bool = False,
) -> dict[str, int]:
    """Validate required declarations and demonstrated hard prerequisites.

    Args:
        declarations: Recorded declarations in Python API call order.
        domain_adapter: Domain hooks for validation and output detection.
        has_final_serialization: Whether execution exports the composed stage.

    Returns:
        Mapping from order identifiers to declaration positions.

    Raises:
        ValidationError: If required intent or a known prerequisite is invalid.

    """
    positions = {
        declaration.order_id: position
        for position, declaration in enumerate(declarations)
    }
    problems = []
    for order_id in _REQUIRED_ORDER_IDS:
        if order_id not in positions:
            problems.append(f"{order_id} was never declared")

    capture_passes: dict[str, set[str]] = {}
    for declaration in declarations:
        pass_name = _capture_pass_name(declaration)
        if pass_name is not None:
            capture_passes.setdefault(pass_name, set()).add(declaration.order_type)
    for declaration in declarations:
        if declaration.order_type != "postprocess.stitch":
            continue
        source_pass = str(declaration.parameters.get("source_pass", "default"))
        if source_pass not in capture_passes:
            problems.append(
                f"postprocess.stitch references unknown capture pass {source_pass!r}",
            )
    # A converted format (bmp) re-encodes only the rgb frames, once the capture
    # handler has flushed the pass; masks, bounding boxes and label sidecars
    # keep the writer's own formats. A postprocess step that opens the rgb
    # frames of such a pass would look for png/jpeg names that no longer
    # exist, so that combination is refused here, as a declaration error
    # rather than an empty crop or stitch at the end of a run. Steps that read
    # other artifacts of the pass, and any step on a pass with rgb disabled,
    # never meet the converted frames and stay valid.
    converted_rgb_formats = {
        _capture_pass_name(declaration): str(declaration.parameters["image_format"])
        for declaration in declarations
        if declaration.order_type == "capture.outputs.configure"
        and bool(declaration.parameters.get("rgb", True))
        and str(declaration.parameters.get("image_format", "png"))
        not in WRITER_NATIVE_IMAGE_FORMATS
    }
    if converted_rgb_formats:
        for declaration in declarations:
            consumed = _postprocess_rgb_source_passes(
                declaration,
                tuple(capture_passes),
            )
            for pass_name in sorted(consumed & set(converted_rgb_formats)):
                problems.append(
                    f"{declaration.order_type} opens the rgb frames of capture "
                    f"pass {pass_name!r} as the writer's png/jpeg, but that pass "
                    f"declares image_format {converted_rgb_formats[pass_name]!r}, "
                    "which re-encodes the rgb frames after capture; give that "
                    "pass a native format, or read only its other artifacts "
                    "(masks, bounding boxes, stitched images)",
                )
    stitched = {
        (
            str(declaration.parameters.get("source_pass", "default")),
            str(declaration.parameters.get("output_subdir", "stitched")),
            str(declaration.parameters.get("output_filename", "rgb.png")),
        )
        for declaration in declarations
        if declaration.order_type == "postprocess.stitch"
    }
    for declaration in declarations:
        if declaration.order_type not in _STITCH_CONSUMER_TYPES:
            continue
        key = (
            str(declaration.parameters.get("source_pass", "default")),
            str(declaration.parameters.get("output_subdir", "stitched")),
            str(declaration.parameters.get("output_filename", "rgb.png")),
        )
        if key not in stitched:
            problems.append(
                f"{declaration.order_type} names a stitched image nothing "
                f"produces: source_pass={key[0]!r} output_subdir={key[1]!r} "
                f"output_filename={key[2]!r}",
            )

    has_domain_output = domain_adapter.produces_output(declarations)
    if not capture_passes and not has_domain_output and not has_final_serialization:
        problems.append(
            "no capture render pass or serialization output was configured",
        )
    for pass_name, order_types in capture_passes.items():
        missing_types = _REQUIRED_CAPTURE_TYPES - order_types
        if missing_types:
            missing = ", ".join(
                sorted(
                    order_type.removeprefix("capture.").removesuffix(".configure")
                    for order_type in missing_types
                )
            )
            problems.append(f"capture render pass {pass_name!r} is missing: {missing}")

    scene_position = positions.get("scene.load")
    if scene_position is not None:
        for position, declaration in enumerate(declarations):
            if declaration.order_id == "scene.load":
                continue
            if declaration.order_type.startswith("postprocess."):
                continue
            if position < scene_position:
                problems.append(
                    f"{declaration.order_type} must be declared after scene.configure(...)",
                )

    problems.extend(domain_adapter.validate_declarations(declarations, positions))

    if problems:
        raise ValidationError(
            "invalid JobOrder description:\n- " + "\n- ".join(problems),
        )
    return positions


def _declaration_dependencies(
    declaration: _JobDeclaration,
    positions: dict[str, int],
    *,
    declarations: tuple[_JobDeclaration, ...],
    domain_adapter: DomainCompilerAdapter,
) -> tuple[str, ...]:
    """Build semantic dependencies for a recorded declaration.

    Args:
        declaration: Declaration being lowered into a JobOrder.
        positions: Mapping from order identifiers to declaration positions.
        declarations: All normalized declarations in the job.
        domain_adapter: Hooks contributed by the selected domain.

    Returns:
        Prerequisite order identifiers in deterministic order.

    """
    dependencies = []
    if declaration.order_id != "scene.load":
        dependencies.append("scene.load")

    dependencies.extend(
        domain_adapter.declaration_dependencies(
            declaration,
            positions,
            declarations,
        ),
    )

    return tuple(dict.fromkeys(dependencies))


#: Postprocess orders that consume a stitched image rather than the raw frames.
_STITCH_CONSUMER_TYPES = frozenset({"postprocess.close_openings"})


def _stitch_key(declaration: _JobDeclaration) -> tuple[str, str, str]:
    """Return what identifies the stitched image a declaration refers to.

    Args:
        declaration: A stitch, or an order that consumes one.

    Returns:
        The capture pass, output directory and filename, which together name
        exactly one stitched image.

    """
    return (
        str(declaration.parameters.get("source_pass", "default")),
        str(declaration.parameters.get("output_subdir", "stitched")),
        str(declaration.parameters.get("output_filename", "rgb.png")),
    )


def _stitch_producers(
    declarations: Sequence[_JobDeclaration],
) -> dict[tuple[str, str, str], str]:
    """Return the order that produces each stitched image.

    Args:
        declarations: The postprocess declarations of one job.

    Returns:
        A map from stitched-image identity to the order that writes it.

    Raises:
        ValidationError: If two stitches write the same image, which leaves the
            consumer with no single producer to depend on.

    """
    producers: dict[tuple[str, str, str], str] = {}
    clashes = []
    for declaration in declarations:
        if declaration.order_type != "postprocess.stitch":
            continue
        key = _stitch_key(declaration)
        if key in producers:
            clashes.append(key)
            continue
        producers[key] = declaration.order_id
    if clashes:
        listed = "; ".join(
            f"source_pass={key[0]!r} output_subdir={key[1]!r} "
            f"output_filename={key[2]!r}"
            for key in clashes
        )
        message = (
            "more than one postprocess.stitch writes the same image, so an "
            f"order that consumes it has no single producer: {listed}"
        )
        raise ValidationError(message)
    return producers


def _compile_job_orders(
    job: CDFJob,
    output_dir: str | Path | None = None,
    serialization_path: str | Path | None = None,
) -> JobOrderPlan:
    """Lower recorded CDF declarations into an immutable JobOrder plan.

    Python API call order is the stable tie-breaker for independent work.
    Semantic dependencies may override that order. Domain adapters can insert
    lifecycle orders around recorded declarations. Declared postprocessing
    consumes the synthetic terminal ``capture.render`` order added by this
    compiler.

    Args:
        job: Declarative CDF job to compile.
        output_dir: Output directory supplied to the capture order.
        serialization_path: Destination for the flattened composed USD stage.

    Returns:
        Validated JobOrder plan in deterministic declaration order.

    Raises:
        ValidationError: If declarations or their known hard prerequisites are
            invalid.

    """
    domain_adapter = compiler_adapter(job.domain.name)
    declarations = domain_adapter.prepare_declarations(tuple(job._declarations))
    positions = _validate_declarations(
        declarations,
        domain_adapter=domain_adapter,
        has_final_serialization=serialization_path is not None,
    )
    runtime_declarations = tuple(
        declaration
        for declaration in declarations
        if (
            not declaration.order_type.startswith("postprocess.")
            and declaration.order_type not in domain_adapter.configuration_order_types
        )
    )
    capture_declarations = tuple(
        declaration
        for declaration in runtime_declarations
        if _capture_pass_name(declaration) is not None
    )
    shared_declarations = tuple(
        declaration
        for declaration in runtime_declarations
        if _capture_pass_name(declaration) is None
    )
    postprocess_declarations = tuple(
        declaration
        for declaration in declarations
        if declaration.order_type.startswith("postprocess.")
    )
    pass_names = tuple(
        dict.fromkeys(
            pass_name
            for declaration in capture_declarations
            if (pass_name := _capture_pass_name(declaration)) is not None
        )
    )
    static_lighting_mode = _static_lighting_mode(declarations)
    capture_camera_sides = {
        pass_name: _capture_camera_side(declarations, pass_name)
        for pass_name in pass_names
    }
    legacy_default_pass = pass_names == ("default",)
    initial_declarations = (
        runtime_declarations if legacy_default_pass else shared_declarations
    )

    orders = [
        JobOrder(
            order_id=declaration.order_id,
            order_type=declaration.order_type,
            parameters=declaration.parameters,
            depends_on=_declaration_dependencies(
                declaration,
                positions,
                declarations=declarations,
                domain_adapter=domain_adapter,
            ),
        )
        for declaration in initial_declarations
    ]
    orders = domain_adapter.expand_orders(orders, declarations, positions)
    serialized_lighting_order_id = None
    if serialization_path is not None and static_lighting_mode is not None:
        serialized_pass = pass_names[0] if pass_names else None
        camera_side = (
            capture_camera_sides[serialized_pass]
            if serialized_pass is not None
            else "top"
        )
        lighting_order = _lighting_apply_order(
            serialized_pass,
            camera_side,
            tuple(order.order_id for order in orders),
        )
        orders.append(lighting_order)
        serialized_lighting_order_id = lighting_order.order_id
    if serialization_path is not None:
        orders.append(
            JobOrder(
                order_id="stage.serialize",
                order_type="stage.serialize",
                parameters={"serialization_path": str(serialization_path)},
                depends_on=tuple(order.order_id for order in orders),
            ),
        )
    render_order_ids = []
    for pass_name in pass_names:
        pass_declarations = tuple(
            declaration
            for declaration in capture_declarations
            if _capture_pass_name(declaration) == pass_name
        )
        if not legacy_default_pass:
            orders.extend(
                JobOrder(
                    order_id=declaration.order_id,
                    order_type=declaration.order_type,
                    parameters=declaration.parameters,
                    depends_on=_declaration_dependencies(
                        declaration,
                        positions,
                        declarations=declarations,
                        domain_adapter=domain_adapter,
                    ),
                )
                for declaration in pass_declarations
            )
        if static_lighting_mode is not None:
            lighting_order = _lighting_apply_order(
                pass_name,
                capture_camera_sides[pass_name],
                tuple(order.order_id for order in orders),
            )
            if lighting_order.order_id != serialized_lighting_order_id:
                orders.append(lighting_order)
        render_order_id = (
            "capture.render"
            if pass_name == "default"
            else f"capture.{pass_name}.render"
        )
        capture_parameters = {}
        pass_output = _pass_output_dir(output_dir, pass_name)
        if pass_output is not None:
            capture_parameters["output_dir"] = pass_output
        orders.append(
            JobOrder(
                order_id=render_order_id,
                order_type="capture.render",
                parameters=capture_parameters,
                depends_on=tuple(order.order_id for order in orders),
            ),
        )
        render_order_ids.append(render_order_id)
    stitch_producers = _stitch_producers(postprocess_declarations)
    # Emit producers before the orders that read them, whatever order they were
    # declared in. The dependency edges alone are enough to execute correctly,
    # but leaving the sequence to follow declaration order lets it show up in
    # the plan -- and the YAML path applies closure before stitch while a
    # Python caller naturally writes stitch first, so the two surfaces compiled
    # the same job into two different plans. The sort is stable, so anything
    # unrelated keeps the order it was written in.
    ordered_postprocess = sorted(
        postprocess_declarations,
        key=lambda entry: entry.order_type in _STITCH_CONSUMER_TYPES,
    )
    for declaration in ordered_postprocess:
        dependencies = tuple(render_order_ids)
        if declaration.order_type == "postprocess.stitch":
            source_pass = str(declaration.parameters["source_pass"])
            dependencies = (
                "capture.render"
                if source_pass == "default"
                else f"capture.{source_pass}.render",
            )
        elif declaration.order_type in _STITCH_CONSUMER_TYPES:
            # Depend on the stitch that produces the image, found by what the
            # two of them name -- not by the order the caller happened to
            # declare them in, which carries no meaning.
            dependencies = (stitch_producers[_stitch_key(declaration)],)
        orders.append(
            JobOrder(
                order_id=declaration.order_id,
                order_type=declaration.order_type,
                parameters=declaration.parameters,
                depends_on=dependencies,
            ),
        )

    metadata: dict[str, object] = {
        "api_version": CDF_API_VERSION,
        "domain": job.domain.name,
        "random_seed": job.random_seed,
    }
    source = getattr(job, "source", None)
    if source:
        # The plan is what survives into the Isaac child process; the job does
        # not. Provenance therefore reads the config identity from here.
        metadata["source"] = dict(source)
    if output_dir is not None:
        # The root every capture pass files under. Provenance derives its
        # artifact root from the passes and must never enumerate outside the
        # run; this is the boundary it checks against.
        metadata["output_root"] = str(output_dir)
    plan = JobOrderPlan(orders=tuple(orders), metadata=metadata)
    plan.ordered()
    return plan
