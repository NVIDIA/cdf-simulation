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

"""PCB-AOI declarative sections: the domain's operations and its capture."""

from __future__ import annotations

import json
import numbers
from typing import TYPE_CHECKING

from cdf.component_filter import normalize_component_filter
from cdf.errors import ValidationError
from cdf.sections import (
    CapturePassSection,
    CaptureSection,
    OperationsSection,
    _pair,
    _ratio,
)

if TYPE_CHECKING:
    from cdf.job import CDFJob


def _observable_spec(name, value, *, lo, hi):
    """Validate a defect observable: a number or a [min, max] range.

    Bounds are EXCLUSIVE -- the boundary values are different classes (a
    volume_ratio of 0 is missing_solder, 1 is nominal; a gap_fraction of 0
    does not interrupt and 1 leaves no fillet).
    """
    values = value if isinstance(value, (list, tuple)) else (value,)
    if isinstance(value, (list, tuple)):
        _pair(name, value)

    for v in values:
        if not isinstance(v, numbers.Real) or not lo < float(v) < hi:
            raise ValidationError(
                f"{name} must be within ({lo}, {hi}), got {v!r}",
            )
    return list(values) if isinstance(value, (list, tuple)) else float(value)


def _targeting(
    operation_name: str,
    *,
    ratio: float | None,
    count: int | None,
    default_ratio: float | None = None,
) -> dict[str, float | int]:
    """Normalize one mutually exclusive defect-targeting mode."""
    if ratio is None and count is None and default_ratio is not None:
        ratio = default_ratio
    if (ratio is None) == (count is None):
        raise ValidationError(
            f"{operation_name} requires exactly one of ratio or count",
        )
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValidationError(f"{operation_name}.count must be an int >= 1")
        return {"count": count}
    assert ratio is not None
    return {"ratio": _ratio(f"{operation_name}.ratio", ratio)}


def _normalize_filter(
    where: str,
    component_filter: object | None,
    *,
    legacy_selector: object | None = None,
) -> dict[str, list[str]] | None:
    """Validate one normalized filter and its legacy-selector boundary.

    Args:
        where: Public API location used in validation messages.
        component_filter: Filter model or mapping supplied by the caller.
        legacy_selector: Mutually exclusive legacy selector, when applicable.

    Returns:
        JSON-compatible component filter, or None when omitted.

    Raises:
        ValidationError: If the filter is invalid or mixed with a legacy selector.

    """
    if component_filter is not None and legacy_selector is not None:
        raise ValidationError(
            f"{where}: legacy selector and component_filter are mutually exclusive",
        )
    try:
        return normalize_component_filter(component_filter, where=where)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def _selector(
    components: list[str] | tuple[str, ...] | None,
    package_types: list[str] | tuple[str, ...] | None,
    component_filter: object | None,
) -> dict[str, object]:
    """Normalize one mutually exclusive way of naming a set of parts.

    Refdes and package type are deliberately **not** combinable. Accepting both
    would have to mean either union or intersection, and a caller who wrote
    both almost certainly meant one of them; guessing turns a typo into a
    silently different sample. Multiple calls express multiple sets.

    Args:
        components: Reference designators, e.g. ``("C1001", "C1002")``.
        package_types: Package-type names, e.g. ``("TO92", "0805")``.
        component_filter: Optional exact metadata-field filter.

    Returns:
        A single-key mapping naming the selected set.

    Raises:
        ValidationError: If neither or both are given, or an entry is not a
            non-empty string.

    """
    supplied = sum(
        value is not None for value in (components, package_types, component_filter)
    )
    if supplied != 1:
        raise ValidationError(
            "pin_variation requires exactly one of components, package_types, "
            "or component_filter",
        )
    if component_filter is not None:
        return {
            "component_filter": _normalize_filter(
                "pin_variation",
                component_filter,
            ),
        }
    key = "components" if components is not None else "package_types"
    values = components if components is not None else package_types
    if not isinstance(values, (list, tuple)) or not values:
        raise ValidationError(
            f"pin_variation.{key} must be a non-empty list of names",
        )
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(
                f"pin_variation.{key} entries must be non-empty strings, got {value!r}",
            )
    return {key: [str(v) for v in values]}


#: The axes that are ONE PLANE through the finished assembly rather than a
#: pose applied to the lead. `height_variation` is where that plane sits and
#: `cut_angle` is how it tips; they were separate only because the trim used to
#: run before the joint existed.
_CUT_AXES = frozenset({"cut_angle", "height_variation"})

#: Keys that name WHICH leads a group claims, and so belong to both orders.
_SELECTOR_KEYS = frozenset({"components", "package_types", "component_filter"})

#: Physical bounds for each pin_variation axis. Tilt and cut are angles off an
#: axis, so 90 degrees is degenerate rather than merely large; protrusion and
#: offset are magnitudes whose direction is drawn separately, so a negative one
#: is a sign error rather than the opposite direction.
PIN_VARIATION_BOUNDS = {
    "tilt_range": (0.0, 89.0),
    "cut_angle": (0.0, 89.0),
    "height_variation": (0.0, None),
    "offset_range": (0.0, None),
}


def _variant_index(name, value):
    """Validate a stratification index at the API boundary.

    Args:
        name: Parameter name, for the message.
        value: Caller-supplied index, or None.

    Returns:
        The index, or None when unset.

    Raises:
        ValidationError: If it is a bool, not an integer, or negative.

    """
    if value is None:
        return None
    # bool is an int in Python, and add_shift(variant_index=True) is far more
    # likely a mistake than a request for index 1.
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValidationError(
            f"{name} must be a non-negative integer, got {value!r}",
        )
    if value < 0:
        raise ValidationError(
            f"{name} must be non-negative, got {value!r}",
        )
    return int(value)


def _non_negative(name, value):
    """Validate a non-negative magnitude at the API boundary.

    Args:
        name: Parameter name, for the message.
        value: Caller-supplied magnitude.

    Returns:
        The value as a float.

    Raises:
        ValidationError: If it is a bool, not a number, or negative.

    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(
            f"{name} must be a non-negative number, got {value!r}",
        )
    if value < 0:
        raise ValidationError(
            f"{name} must be non-negative, got {value!r}",
        )
    return float(value)


def _bounded_pair(
    name: str,
    value: object,
    bounds: tuple[float, float | None],
) -> list[float]:
    """Validate one pin_variation band against its own physical limits.

    Args:
        name: Parameter name, used in the message.
        value: The ``(lo, hi)`` pair supplied by the caller.
        bounds: Inclusive minimum and optional maximum for the axis.

    Returns:
        The validated pair.

    Raises:
        ValidationError: If either end is a bool, or falls outside the axis's
            own limits.

    """
    if isinstance(value, (list, tuple)) and any(
        isinstance(entry, bool) for entry in value
    ):
        raise ValidationError(f"{name} bounds must be numbers, not booleans")
    pair = _pair(name, value)
    low, high = bounds
    for entry in pair:
        if entry < low:
            raise ValidationError(f"{name} must not be below {low}, got {entry}")
        if high is not None and entry > high:
            raise ValidationError(f"{name} must not exceed {high}, got {entry}")
    return pair


class PcbAoiOperationsSection(OperationsSection):
    """PCB-AOI defects and generated solder-joint geometry.

    Args:
        job: Declarative job that receives recorded PCB operation intent.

    Example:

    .. code-block:: python

        import pcb_aoi
        from cdf import CDFJob

        job = CDFJob(domain="pcb-aoi")
        print(type(job.operations).__name__)
        # PcbAoiOperationsSection

    """

    def __init__(self, job: CDFJob) -> None:
        """Initialize PCB operation state for the declarative job."""
        super().__init__(job)
        self.solder_joints_spec: dict[str, object] | None = None
        self.solder_defect_specs: dict[str, dict[str, object]] = {}
        #: Accumulated pin_variation groups. Repeat calls extend this one
        #: spec rather than declaring the operation more than once.
        #: Accumulated trim groups, kept apart from the pose groups because
        #: they are executed after the joint exists rather than before.
        self.pin_cut_spec: dict[str, list[dict[str, object]]] = {"groups": []}
        self.pin_variation_spec: dict[str, list[dict[str, object]]] = {
            "groups": [],
        }

    def add_shift(
        self,
        *,
        ratio: float | None = None,
        count: int | None = None,
        translate_range: float,
        rotate_z_range: float,
        variant_index: int | None = None,
        component_filter: object | None = None,
    ) -> PcbAoiOperationsSection:
        """Add a pick-and-place translation and rotation defect.

        Args:
            ratio: Fraction of selected components to modify.
            count: Exact number of components to modify.
            translate_range: Symmetric XY translation bound in millimeters.
            rotate_z_range: Symmetric board-normal rotation bound in degrees.
            variant_index: Position in a repeating coverage sequence. Pass the
                sample's own index and the defect is *assigned* a region of the
                parameter space instead of drawn at random from all of it, so a
                run covers the space evenly rather than clumping. The index
                selects the mode (``translate`` / ``rotate`` / ``combined``,
                every 1), one of eight 45-degree direction sectors in the
                component's own frame (every 3), the translation magnitude band
                (every 24), the rotation magnitude band (every 72) and the
                rotation sign (every 144). The strides form a mixed-radix
                odometer, so the axes are independent rather than locked to one
                another: the full 3 x 8 x 3 x 2 x 2 grid repeats after **288**,
                and any 288 consecutive indices cover it exactly once. Values
                inside the assigned region are still random, so samples sharing
                an index are not duplicates. Omit it
                and sampling is uniform, as before. See
                ``docs/defect-variants.md``.
            component_filter: Optional exact metadata-field filter.

        Returns:
            This operation section for call chaining.

        Raises:
            ValidationError: If targeting is missing, ambiguous, or invalid.

        Example:

        .. code-block:: python

            job.operations.add_shift(
                count=2,
                translate_range=0.2,
                rotate_z_range=15.0,
            )

        Stratified over a run, passing each sample its own index:

        .. code-block:: python

            for n in range(72):          # one full coverage cycle
                job = build_job()
                job.operations.add_shift(count=1, variant_index=n)

        """
        parameters = _targeting("shift", ratio=ratio, count=count)
        parameters.update(
            {
                "translate_range": translate_range,
                "rotate_z_range": rotate_z_range,
                # Omitted unless asked for: an unset knob must not appear in the
                # compiled flow, or every existing config gains a null key.
                **(
                    {}
                    if variant_index is None
                    else {
                        "variant_index": _variant_index(
                            "shift.variant_index", variant_index
                        )
                    }
                ),
            },
        )
        declaration = {
            **parameters,
            "component_filter": _normalize_filter("shift", component_filter),
        }
        return self._add("shift", parameters, declaration_params=declaration)

    def add_tombstone(
        self,
        *,
        ratio: float | None = None,
        count: int | None = None,
        angle_range: tuple[float, float],
        component_types: list[str] | tuple[str, ...] | None = None,
        component_filter: object | None = None,
        auto_filter: bool = True,
        eligibility_source: str | None = None,
    ) -> PcbAoiOperationsSection:
        """Add a tombstone defect over an angle range.

        Args:
            ratio: Fraction of eligible components to modify.
            count: Exact number of eligible components to modify.
            angle_range: Minimum and maximum tombstone angles in degrees.
            component_types: Optional component-type allow-list.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``component_types``.
            auto_filter: Whether to apply metadata-catalog eligibility filtering.
            eligibility_source: Optional active component-metadata source used for
                filtering.

        Returns:
            This operation section for call chaining.

        Raises:
            ValidationError: If targeting or ``angle_range`` is invalid.

        Example:

        .. code-block:: python

            job.operations.add_tombstone(
                count=2,
                angle_range=(70.0, 90.0),
            )

        """
        lo, hi = _pair("tombstone.angle_range", angle_range)
        targeting = _targeting("tombstone", ratio=ratio, count=count)
        normalized_filter = _normalize_filter(
            "tombstone",
            component_filter,
            legacy_selector=component_types,
        )
        return self._add(
            "tombstone",
            {
                **targeting,
                "angle_min": lo,
                "angle_max": hi,
                "component_types": component_types,
                "auto_filter": auto_filter,
                "eligibility_source": eligibility_source,
            },
            declaration_params={
                **targeting,
                "angle_range": [lo, hi],
                "component_types": component_types,
                "component_filter": normalized_filter,
                "auto_filter": auto_filter,
                "eligibility_source": eligibility_source,
            },
        )

    def add_sideflip(
        self,
        *,
        ratio: float | None = None,
        count: int | None = None,
        angle_range: tuple[float, float],
        variant_index: int | None = None,
        rotate_z_jitter_deg: float = 0.0,
        slide_range_mm: float = 0.0,
        component_filter: object | None = None,
    ) -> PcbAoiOperationsSection:
        """Add a component length-axis sideflip defect.

        Args:
            ratio: Fraction of selected components to modify.
            count: Exact number of components to modify.
            angle_range: Minimum and maximum sideflip angles in degrees.
            variant_index: Position in a repeating coverage sequence, as on
                :meth:`add_shift`. Here it selects one of three equal sub-bins
                of ``angle_range`` (every 1) and which of the two long support
                edges the part hinges on (every 3), so the sequence repeats
                after **6**. Without it a run can flip every part onto the same
                edge and cluster the angles. See ``docs/defect-variants.md``.
            rotate_z_jitter_deg: Symmetric board-normal rotation applied before
                the flip, in degrees. Same axis as ``rotate_z_range`` on
                :meth:`add_shift`. A pure geometric flip leaves the part
                perfectly square to its pads, which is a giveaway no real board
                shows; this breaks that alignment.
            slide_range_mm: Symmetric slide along the hinge edge, in
                millimeters, so the flipped part is not always centred on it.
            component_filter: Optional exact metadata-field filter.

        Returns:
            This operation section for call chaining.

        Raises:
            ValidationError: If targeting or ``angle_range`` is invalid.

        Example:

        .. code-block:: python

            job.operations.add_sideflip(
                count=2,
                angle_range=(60.0, 80.0),
            )

        """
        lo, hi = _pair("sideflip.angle_range", angle_range)
        targeting = _targeting("sideflip", ratio=ratio, count=count)
        # Omitted unless asked for, so an existing flow config is unchanged.
        # Validated here rather than left to compile time: these go straight
        # into the declaration payload, and a wrong-shaped value that only fails
        # later surfaces a long way from the call that caused it.
        perturbation = {}
        if variant_index is not None:
            perturbation["variant_index"] = _variant_index(
                "sideflip.variant_index", variant_index
            )
        jitter = _non_negative("sideflip.rotate_z_jitter_deg", rotate_z_jitter_deg)
        slide = _non_negative("sideflip.slide_range_mm", slide_range_mm)
        if jitter:
            perturbation["rotate_z_jitter_deg"] = jitter
        if slide:
            perturbation["slide_range_mm"] = slide
        return self._add(
            "sideflip",
            {
                **targeting,
                "angle_min": lo,
                "angle_max": hi,
                **perturbation,
            },
            declaration_params={
                **targeting,
                "angle_range": [lo, hi],
                **perturbation,
                "component_filter": _normalize_filter(
                    "sideflip",
                    component_filter,
                ),
            },
        )

    def add_reverse_polarity(
        self,
        *,
        ratio: float | None = None,
        count: int | None = None,
        component_types: list[str] | tuple[str, ...] | None = None,
        component_filter: object | None = None,
    ) -> PcbAoiOperationsSection:
        """Add a 180-degree polarity reversal.

        Args:
            ratio: Fraction of selected components to modify. Defaults to one
                when neither targeting mode is supplied.
            count: Exact number of components to modify.
            component_types: Optional component-type allow-list.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``component_types``.

        Returns:
            This operation section for call chaining.

        Raises:
            ValidationError: If targeting is ambiguous or invalid.

        Example:

        .. code-block:: python

            job.operations.add_reverse_polarity(
                ratio=0.25,
                component_types=["IC_SMD"],
            )

        """
        targeting = _targeting(
            "reverse_polarity",
            ratio=ratio,
            count=count,
            default_ratio=1.0,
        )
        normalized_filter = _normalize_filter(
            "reverse_polarity",
            component_filter,
            legacy_selector=component_types,
        )
        return self._add(
            "reverse_polarity",
            {
                **targeting,
                "component_types": component_types,
            },
            declaration_params={
                **targeting,
                "component_types": component_types,
                "component_filter": normalized_filter,
            },
        )

    def add_missing(
        self,
        *,
        ratio: float | None = None,
        count: int | None = None,
        component_filter: object | None = None,
    ) -> PcbAoiOperationsSection:
        """Add a missing-component defect.

        Args:
            ratio: Fraction of selected components to hide.
            count: Exact number of components to hide.
            component_filter: Optional exact metadata-field filter.


        Returns:
            This operation section for call chaining.

        Raises:
            ValidationError: If targeting is missing, ambiguous, or invalid.

        Example:

        .. code-block:: python

            job.operations.add_missing(count=2)

        """
        targeting = _targeting("missing", ratio=ratio, count=count)
        return self._add(
            "missing",
            targeting,
            declaration_params={
                **targeting,
                "component_filter": _normalize_filter("missing", component_filter),
            },
        )

    def pin_variation(
        self,
        *,
        components: list[str] | tuple[str, ...] | None = None,
        package_types: list[str] | tuple[str, ...] | None = None,
        component_filter: object | None = None,
        tilt_range: tuple[float, float] | None = None,
        cut_angle: tuple[float, float] | None = None,
        height_variation: tuple[float, float] | None = None,
        offset_range: tuple[float, float] | None = None,
    ) -> PcbAoiOperationsSection:
        """Vary through-hole leads inside their own plated barrels.

        Call it more than once to give different sets of parts different
        ranges; each call adds a group and the groups are applied in order.
        A lead named by more than one group takes the LAST group that names it,
        so a broad ``package_types`` sweep can be overridden by a narrow
        ``components`` one written after it.

        Every range is a *request*. Tilt competes with the barrel, offset with
        whatever clearance the tilt left, and the cut angle with the protrusion
        it has to fit inside; the delivered value and the constraint that bound
        it are reported per lead.

        Args:
            components: Reference designators to vary. Mutually exclusive with
                ``package_types``.
            package_types: Package types to vary. Mutually exclusive with
                ``components``. Prefer ``components`` where the board's package
                metadata is not dependable.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``components`` and ``package_types``.
            tilt_range: Tilt off vertical, degrees.
            cut_angle: Trim-plane angle off perpendicular, degrees.
            height_variation: Lead protrusion past the far board face,
                millimetres.
            offset_range: Off-centre distance within the bore, millimetres.

        Returns:
            This operation section for call chaining.

        Raises:
            ValidationError: If the selector is missing or ambiguous, if no
                range is given, or if a range is malformed.

        Example:

        .. code-block:: python

            job.operations.pin_variation(
                package_types=("TO92",),
                tilt_range=(0, 15),
                height_variation=(0.0, 1.5),
            ).pin_variation(
                components=("C1001", "C1002"),
                tilt_range=(0, 4),
                cut_angle=(0, 60),
            )

        """
        group: dict[str, object] = _selector(
            components,
            package_types,
            component_filter,
        )
        ranges = {
            "tilt_range": tilt_range,
            "cut_angle": cut_angle,
            "height_variation": height_variation,
            "offset_range": offset_range,
        }
        if all(value is None for value in ranges.values()):
            raise ValidationError(
                "pin_variation needs at least one of tilt_range, cut_angle, "
                "height_variation or offset_range",
            )
        for name, value in ranges.items():
            if value is not None:
                group[name] = _bounded_pair(
                    f"pin_variation.{name}",
                    value,
                    PIN_VARIATION_BOUNDS[name],
                )

        if "pin_variation" not in self._domain.job_operations:
            raise ValidationError(
                "operation 'pin_variation' is not supported by domain "
                f"{self._domain.name!r}",
            )
        # A job-level geometry operation, like solder_joints -- NOT a
        # per-component defect. It is deliberately absent from the live panel's
        # defect vocabulary, which applies one defect to one selected part.
        #
        # ONE call, TWO orders. Leaning and shifting a lead has to happen
        # before the joint is grown, because the joint is sized from where the
        # lead ended up. Trimming it has to happen AFTER, because a blade does
        # not stop at the boundary between copper and solder -- it takes the
        # lead and the joint together and leaves one section across both.
        #
        # The split is internal. A caller still writes one `pin_variation(...)`
        # and still gets a lead that leans, shifts, stands proud and is cut on
        # a slanted plane; they simply do not have to know that two of those
        # four are the same plane applied later.
        pose_group = {
            key: value for key, value in group.items() if key not in _CUT_AXES
        }
        cut_group = {
            key: value
            for key, value in group.items()
            if key in _CUT_AXES or key in _SELECTOR_KEYS
        }
        if any(key in group for key in _CUT_AXES - _SELECTOR_KEYS):
            self.pin_cut_spec["groups"].append(cut_group)
            self._job._record_declaration(
                "operation.pin_cut",
                "operation.pin_cut.apply",
                self.pin_cut_spec,
                preserve_position=True,
            )
        if any(key not in _CUT_AXES and key not in _SELECTOR_KEYS for key in group):
            self.pin_variation_spec["groups"].append(pose_group)
            self._job._record_declaration(
                "operation.pin_variation",
                "operation.pin_variation.apply",
                self.pin_variation_spec,
                preserve_position=True,
            )
        return self

    def solder_joints(
        self,
        *,
        solver: str = "none",
        component_types: list[str] | tuple[str, ...] | None = None,
        component_filter: object | None = None,
        source: str | None = None,
        solver_options: dict[str, object] | None = None,
        output_root_path: str | None = None,
        serialization: str | None = None,
        serialization_path: str | None = None,
    ) -> PcbAoiOperationsSection:
        """Configure implicit solder-joint geometry generation.

        Args:
            solver: Registered solder-generation solver ID.
            component_types: Optional component-name or family allow-list. ``None``
                uses contact-mesh plus sample-instance paste discovery.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``component_types``.
            source: Board metadata source name. Embedded USD metadata is used
                when omitted.
            solver_options: Solver-specific JSON-compatible configuration.
            output_root_path: Absolute destination prim path. ``None`` resolves
                to ``<pcba-root>/SolderJoints``.
            serialization: Deprecated compatibility policy. Omit it and use
                ``serialization_path`` to export the solder edit layer.
            serialization_path: Optional solder edit-layer USD destination.

        Returns:
            This operation section for call chaining.

        Raises:
            ValidationError: If the domain, solver, component list, or source
                is invalid.

        Example:

        .. code-block:: python

            job.operations.solder_joints(
                solver="procedural",
                source="usd",
            )

        """
        if "solder_joints" not in self._domain.job_operations:
            raise ValidationError(
                f"operation 'solder_joints' is not supported by domain {self._domain.name!r}"
            )
        if not isinstance(solver, str) or not solver:
            raise ValidationError(
                "operations.solder_joints: solver must be a non-empty "
                "registered solver ID",
            )
        from .solder_generation import resolve_solver

        resolve_solver(solver)
        if component_types is not None and not (
            isinstance(component_types, (list, tuple))
            and all(isinstance(value, str) and value for value in component_types)
        ):
            raise ValidationError(
                "operations.solder_joints: component_types must be a list "
                "of non-empty names or None"
            )
        normalized_filter = _normalize_filter(
            "operations.solder_joints",
            component_filter,
            legacy_selector=component_types,
        )
        if solver_options is None:
            solver_options = {}
        if not isinstance(solver_options, dict) or not all(
            isinstance(key, str) and key for key in solver_options
        ):
            raise ValidationError(
                "operations.solder_joints: solver_options must be an object "
                "with non-empty string keys",
            )
        try:
            json.dumps(solver_options)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "operations.solder_joints: solver_options must be JSON-compatible",
            ) from exc
        if output_root_path is not None and (
            not isinstance(output_root_path, str)
            or not output_root_path.startswith("/")
            or output_root_path == "/"
            or "//" in output_root_path
        ):
            raise ValidationError(
                "operations.solder_joints: output_root_path must be an "
                "absolute non-root prim path or None",
            )
        from .solder_generation import SolderSerializationMode

        if serialization is None:
            serialization_mode = (
                SolderSerializationMode.SOLDER_JOINTS_ONLY
                if serialization_path is not None
                else SolderSerializationMode.NONE
            )
        else:
            try:
                serialization_mode = SolderSerializationMode(serialization)
            except ValueError as exc:
                raise ValidationError(
                    "operations.solder_joints: serialization must be 'none', "
                    "'solder-joints-only', or 'all'",
                ) from exc
        if serialization_path is not None and (
            not isinstance(serialization_path, str) or not serialization_path
        ):
            raise ValidationError(
                "operations.solder_joints: serialization_path must be a "
                "non-empty USD path",
            )
        if serialization_mode is SolderSerializationMode.NONE:
            if serialization_path is not None:
                raise ValidationError(
                    "operations.solder_joints: legacy serialization='none' "
                    "cannot be combined with serialization_path",
                )
        elif serialization_path is None:
            raise ValidationError(
                "operations.solder_joints: legacy serialization requires "
                "serialization_path",
            )

        if source is not None and source != "usd":
            sidecar = self._job.scene.sidecars.get(source)
            if sidecar is None:
                raise ValidationError(
                    f"operations.solder_joints: unknown sidecar {source!r}"
                )
            if sidecar["kind"] != "pcb.components":
                raise ValidationError(
                    f"operations.solder_joints: sidecar {source!r} must have kind "
                    "'pcb.components'"
                )
        else:
            source = next(
                (
                    name
                    for name, sidecar in self._job.scene.sidecars.items()
                    if sidecar["kind"] == "pcb.components"
                ),
                "usd",
            )

        spec = {
            "solver": solver,
            "component_types": (
                list(dict.fromkeys(component_types))
                if component_types is not None
                else None
            ),
            "component_filter": normalized_filter,
            "source": source,
            "solver_options": dict(solver_options),
            "output_root_path": output_root_path,
            "serialization": serialization_mode.value,
            "serialization_path": serialization_path,
        }
        self.solder_joints_spec = spec
        self._job._record_declaration(
            "operation.solder_joints",
            "operation.solder_joints.configure",
            spec,
        )
        return self

    def hole_plane(
        self,
        *,
        class_name: str = "hole",
        side: str,
        z_offset_mm: float = 0.1,
        output_prim_path: str | None = None,
        components: list[str] | None = None,
        component_filter: object | None = None,
    ) -> PcbAoiOperationsSection:
        """Generate a board-sized hole-reveal plane for segmentation passes.

        Creates a quad mesh at ``z_offset_mm`` inside the board substrate from
        the chosen ``side``.  Substrate geometry occludes the plane everywhere
        except at through-hole voids, making it visible only where drilled holes
        penetrate the board.

        The z reference is the ``board_substrate`` (FR4) face as stored in
        ``BoardRecord.substrate_z_range`` — not the solder-mask or paste layers.

        Generated hole geometry is hidden by default. Use
        ``visibility.show: ["hole"]`` in an annotation pass so the
        segmentation annotator sees it through substrate voids.

        Use ``side="both"`` for boards with THT components on both faces.  Two
        planes are created: one at the bottom (solder) face and one at the top
        (component) face. Both carry ``pcba:mesh:role = "board_hole"`` so a
        single ``visibility.show: ["hole"]`` reveals them together.

        Args:
            class_name: Semantic class to label the generated plane(s) with.
            side: Which substrate face to measure from: ``"bottom"`` for the
                solder side (z_lo), ``"top"`` for the component side (z_hi),
                or ``"both"`` to create planes on both faces (for boards with
                THT components on both sides).
            z_offset_mm: Distance in millimetres to offset inward from the
                chosen substrate face.  Measured from the FR4 substrate face,
                not from the solder-mask or paste layers.
            output_prim_path: Absolute destination prim path. ``None`` places
                the plane at
                ``<pcba-root>/SegmentationProxies/HolePlane`` (single side) or
                sibling ``HolePlane_bottom`` / ``HolePlane_top`` Xforms
                (``side="both"``).
            components: Optional allowlist of refdes strings.  When set, one
                small plane is created per hole belonging to these components
                instead of a single board-wide plane.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``components``.

        Returns:
            This operation section for call chaining.

        Raises:
            ValidationError: If the domain does not support ``hole_plane`` or
                any parameter is invalid.

        Example:

        .. code-block:: python

            job.operations.hole_plane(side="bottom")
            job.operations.hole_plane(side="both")  # THT on both sides
            job.operations.hole_plane(side="bottom", components=["C1", "C2"])

        """
        if "hole_plane" not in self._domain.job_operations:
            raise ValidationError(
                f"operation 'hole_plane' is not supported by domain {self._domain.name!r}"
            )
        if not isinstance(class_name, str) or not class_name:
            raise ValidationError(
                "operations.hole_plane: class_name must be a non-empty string"
            )
        if side not in ("top", "bottom", "both"):
            raise ValidationError(
                "operations.hole_plane: side must be 'top', 'bottom', or 'both'"
            )
        if not isinstance(z_offset_mm, (int, float)) or z_offset_mm < 0:
            raise ValidationError(
                "operations.hole_plane: z_offset_mm must be a non-negative number"
            )
        if output_prim_path is not None and (
            not isinstance(output_prim_path, str)
            or not output_prim_path.startswith("/")
            or output_prim_path == "/"
            or "//" in output_prim_path
        ):
            raise ValidationError(
                "operations.hole_plane: output_prim_path must be an "
                "absolute non-root prim path or None"
            )
        if components is not None and (
            not isinstance(components, (list, tuple))
            or not all(isinstance(c, str) and c for c in components)
        ):
            raise ValidationError(
                "operations.hole_plane: components must be a non-empty list of strings"
            )
        normalized_filter = _normalize_filter(
            "operations.hole_plane",
            component_filter,
            legacy_selector=components,
        )
        spec: dict[str, object] = {
            "class_name": class_name,
            "side": side,
            "z_offset_mm": float(z_offset_mm),
            "output_prim_path": output_prim_path,
        }
        if components is not None:
            spec["components"] = tuple(str(c) for c in components)
        spec["component_filter"] = normalized_filter
        self._job._record_declaration(
            "operation.hole_plane",
            "hole_plane.generate",
            spec,
        )
        return self

    def pin_tip(
        self,
        *,
        side: str,
        class_name: str = "pin_tip",
        offset_mm: float = 0.002,
        components: list[str] | tuple[str, ...] | None = None,
        component_filter: object | None = None,
    ) -> PcbAoiOperationsSection:
        """Generate annotation-only meshes over through-hole lead tip surfaces.

        Each proxy copies the terminal's final connected tip polygons, sits just
        outside the physical pin to avoid z-fighting, and uses a diagnostic
        magenta display color. Selection stops at vertical lead walls and where
        the surface tilts more than 30 degrees from the tip's outward normal, so
        a rounded lead end keeps its chamfer in the terminal's own class. The proxies are grouped under
        ``<pcba-root>/SegmentationProxies/PinTips`` and remain hidden unless an
        annotation pass uses ``visibility.show: ["pin_tip"]``.

        Args:
            side: Board side carrying the visible lead tips.
            class_name: Semantic class authored on each proxy.
            offset_mm: Outward annotation-proxy displacement.
            components: Optional refdes allowlist.  The full THT lead set is
                used when omitted.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``components``.

        Returns:
            This operation section for call chaining.

        Raises:
            ValidationError: If the domain does not support pin-tip proxies or
                any parameter is invalid.

        Example:

        .. code-block:: python

            job.operations.pin_tip(
                side="bottom",
                class_name="pin_tip",
                components=["X205"],
            )

        """
        if "pin_tip" not in self._domain.job_operations:
            raise ValidationError(
                f"operation 'pin_tip' is not supported by domain {self._domain.name!r}"
            )
        if side not in ("top", "bottom"):
            raise ValidationError("operations.pin_tip: side must be 'top' or 'bottom'")
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValidationError(
                "operations.pin_tip: class_name must be a non-empty string"
            )
        if (
            isinstance(offset_mm, bool)
            or not isinstance(offset_mm, (int, float))
            or offset_mm <= 0.0
        ):
            raise ValidationError(
                "operations.pin_tip: offset_mm must be a positive number"
            )
        if components is not None and (
            not isinstance(components, (list, tuple))
            or not components
            or not all(isinstance(value, str) and value.strip() for value in components)
        ):
            raise ValidationError(
                "operations.pin_tip: components must be a non-empty list of strings"
            )
        if components is not None and len(set(components)) != len(components):
            raise ValidationError(
                "operations.pin_tip: components must not contain duplicates"
            )
        normalized_filter = _normalize_filter(
            "operations.pin_tip",
            component_filter,
            legacy_selector=components,
        )
        spec = {
            "class_name": class_name,
            "side": side,
            "offset_mm": float(offset_mm),
            "components": tuple(components) if components is not None else None,
            "component_filter": normalized_filter,
        }
        self._job._record_declaration(
            "operation.pin_tip",
            "operation.pin_tip.generate",
            spec,
        )
        return self

    def add_insufficient(
        self,
        *,
        ratio: float,
        volume_ratio: object = None,
        lead_climb: object = None,
        fullness: object = None,
        component_types: list[str] | tuple[str, ...] | None = None,
        component_filter: object | None = None,
    ) -> PcbAoiOperationsSection:
        """Add starved solder joints: reduced volume on the nominal footprint.

        On THT this is the "less solder" class: shape it with low
        ``lead_climb`` and low ``fullness`` (a valley around the pin). When
        either shape parameter is given it drives the THT target directly
        and ``volume_ratio`` becomes a measured observable there.

        Args:
            ratio: Fraction of the eligible pool to select.
            volume_ratio: Solder volume as a fraction of nominal, in (0, 1) --
                a value or a ``[min, max]`` range sampled per component.
                ``None`` uses the solver default (SMT severity).
            lead_climb: THT shape: wetted lead height over measured pin
                protrusion, in (0, 2) -- a value or a ``[min, max]`` range
                sampled per joint. > 1 buries the tip. ``None`` uses the
                solver default (nominal 0.8).
            fullness: Fillet volume over the joint's own straight-cone
                fillet, in (0, 2) -- a value or a ``[min, max]`` range
                sampled per joint. 1.0 is a flat silhouette, < 1 concave,
                > 1 convex. Shapes BOTH kinds: THT always (nominal 0.85);
                SMT only when given -- an unset SMT joint keeps the
                paste-driven nominal (the printed deposit is mass
                conservation).
            component_types: Optional component-family allow-list. ``None``
                draws from the scene's selected component pool.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``component_types``.

        Returns:
            This operation section for call chaining.

        Example:

        .. code-block:: python

            job.operations.add_insufficient(
                ratio=0.05, volume_ratio=[0.3, 0.7],
            )

        """
        return self._add_solder_defect(
            "insufficient",
            ratio,
            component_types,
            component_filter,
            ("volume_ratio", volume_ratio, 0.0, 1.0),
            ("lead_climb", lead_climb, 0.0, 2.0),
            ("fullness", fullness, 0.0, 2.0),
        )

    def add_excess(
        self,
        *,
        ratio: float,
        volume_ratio: object = None,
        lead_climb: object = None,
        fullness: object = None,
        component_types: list[str] | tuple[str, ...] | None = None,
        component_filter: object | None = None,
    ) -> PcbAoiOperationsSection:
        """Add over-filled solder joints: increased volume, same footprint.

        On THT this is the "hidden/excess" class: ``lead_climb`` > 1 buries
        the pin tip and ``fullness`` >= 1 bulges the fillet. When either
        shape parameter is given it drives the THT target directly and
        ``volume_ratio`` becomes a measured observable there.

        Args:
            ratio: Fraction of the eligible pool to select.
            volume_ratio: Solder volume as a multiple of nominal, > 1 -- a
                value or a ``[min, max]`` range sampled per component.
                ``None`` uses the solver default (SMT severity).
            lead_climb: THT shape: wetted lead height over measured pin
                protrusion, in (0, 2) -- a value or a ``[min, max]`` range
                sampled per joint. > 1 buries the tip. ``None`` uses the
                solver default (nominal 0.8).
            fullness: Fillet volume over the joint's own straight-cone
                fillet, in (0, 2) -- a value or a ``[min, max]`` range
                sampled per joint. 1.0 is a flat silhouette, < 1 concave,
                > 1 convex. Shapes BOTH kinds: THT always (nominal 0.85);
                SMT only when given -- an unset SMT joint keeps the
                paste-driven nominal (the printed deposit is mass
                conservation).
            component_types: Optional component-family allow-list. ``None``
                draws from the scene's selected component pool.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``component_types``.

        Returns:
            This operation section for call chaining.

        Example:

        .. code-block:: python

            job.operations.add_excess(ratio=0.05, volume_ratio=[1.8, 2.6])

        """
        return self._add_solder_defect(
            "excess",
            ratio,
            component_types,
            component_filter,
            ("volume_ratio", volume_ratio, 1.0, float("inf")),
            ("lead_climb", lead_climb, 0.0, 2.0),
            ("fullness", fullness, 0.0, 2.0),
        )

    def add_bright(
        self,
        *,
        ratio: float,
        lead_climb: object = None,
        fullness: object = None,
        component_types: list[str] | tuple[str, ...] | None = None,
        component_filter: object | None = None,
    ) -> PcbAoiOperationsSection:
        """Add bright solder joints: a steep, starved THT meniscus.

        The AOI "bright" class: too little solder pulled taut between the
        pad edge and a respectable climb up the lead, so the surface necks
        in and reads shiny under ring lighting. Distinct from insufficient
        ("less solder") along both shape axes: bright climbs the lead,
        less-solder barely wets it.

        Through-hole only -- an SMT component selected here is reported
        INELIGIBLE by the solver rather than silently soldered nominal.

        Args:
            ratio: Fraction of the eligible pool to select.
            lead_climb: Wetted lead height over measured pin protrusion, in
                (0, 2) -- a value or a ``[min, max]`` range sampled per
                joint. ``None`` uses the defect default ([0.6, 0.9]).
            fullness: Fillet volume over the joint's own straight-cone
                fillet, in (0, 2) -- a value or a ``[min, max]`` range
                sampled per joint. ``None`` uses the defect default
                ([0.35, 0.6]).
            component_types: Optional component-family allow-list. ``None``
                draws from the scene's selected component pool.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``component_types``.

        Returns:
            This operation section for call chaining.

        Example:

        .. code-block:: python

            job.operations.add_bright(
                ratio=0.03, lead_climb=[0.6, 0.9], fullness=[0.35, 0.6],
            )

        """
        return self._add_solder_defect(
            "bright",
            ratio,
            component_types,
            component_filter,
            ("lead_climb", lead_climb, 0.0, 2.0),
            ("fullness", fullness, 0.0, 2.0),
        )

    def add_interrupted(
        self,
        *,
        ratio: float,
        pad_exposure: object = None,
        lead_climb: object = None,
        fullness: object = None,
        component_types: list[str] | tuple[str, ...] | None = None,
        component_filter: object | None = None,
    ) -> PcbAoiOperationsSection:
        """Add interrupted solder joints: the pad itself is partially exposed.

        The fillet's foot stops short of the pad edge, leaving bare copper
        visible -- a noisy dewetting front pushed in from a random azimuth.
        Both kinds; on THT the ring around the pin stays CLOSED (a broken
        ring is ``dry_contact``, a different class, and is reported).

        Args:
            ratio: Fraction of the eligible pool to select.
            pad_exposure: Exposed pad area over total pad area, in (0, 1) --
                a value or a ``[min, max]`` range sampled per joint. The
                exposure azimuth is always randomized and never configured.
                ``None`` uses the defect default ([0.2, 0.5]).
            lead_climb: THT shape override; see :meth:`add_insufficient`.
            fullness: Shape override, both kinds; see :meth:`add_insufficient`.
            component_types: Optional component-family allow-list. ``None``
                draws from the scene's selected component pool.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``component_types``.

        Returns:
            This operation section for call chaining.

        Example:

        .. code-block:: python

            job.operations.add_interrupted(
                ratio=0.03, pad_exposure=[0.2, 0.5],
            )

        """
        return self._add_solder_defect(
            "interrupted",
            ratio,
            component_types,
            component_filter,
            ("pad_exposure", pad_exposure, 0.0, 1.0),
            ("lead_climb", lead_climb, 0.0, 2.0),
            ("fullness", fullness, 0.0, 2.0),
        )

    def add_missing_solder(
        self,
        *,
        ratio: float,
        component_types: list[str] | tuple[str, ...] | None = None,
        component_filter: object | None = None,
    ) -> PcbAoiOperationsSection:
        """Add pasted-but-unsoldered sites: the component stays, solder does not.

        Distinct from the ``missing`` component operation -- here the part is
        present and its solder (and printed paste) is not. No parameters: the
        volume ratio is definitionally zero.

        Args:
            ratio: Fraction of the eligible pool to select.
            component_types: Optional component-family allow-list. ``None``
                draws from the scene's selected component pool.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``component_types``.

        Returns:
            This operation section for call chaining.

        Example:

        .. code-block:: python

            job.operations.add_missing_solder(ratio=0.02)

        """
        return self._add_solder_defect(
            "missing_solder",
            ratio,
            component_types,
            component_filter,
        )

    def add_dry_contact(
        self,
        *,
        ratio: float,
        gap_fraction: object = None,
        lead_climb: object = None,
        fullness: object = None,
        component_types: list[str] | tuple[str, ...] | None = None,
        component_filter: object | None = None,
    ) -> PcbAoiOperationsSection:
        """Add dry-contact through-hole fillets: the solder does not ring the pin.

        (Renamed from ``interrupted`` before the API 1.0.0 release: in
        the AOI taxonomy "interrupted" means partially visible pad
        structures, a different defect; the hole-in-the-solder one is dry
        contact.)

        Through-hole only -- an SMT component selected here is reported
        INELIGIBLE by the solver rather than silently soldered nominal. The
        gap's azimuth is always randomized per joint and never configured.

        Args:
            ratio: Fraction of the eligible pool to select.
            gap_fraction: Fraction of the barrel circumference NOT wetted, in
                (0, 1) -- a value or a ``[min, max]`` range sampled per joint.
                ``None`` uses the solver default.
            lead_climb: THT shape: wetted lead height over measured pin
                protrusion, in (0, 2) -- a value or a ``[min, max]`` range
                sampled per joint. > 1 buries the tip. ``None`` uses the
                solver default (nominal 0.8).
            fullness: Fillet volume over the joint's own straight-cone
                fillet, in (0, 2) -- a value or a ``[min, max]`` range
                sampled per joint. 1.0 is a flat silhouette, < 1 concave,
                > 1 convex. Shapes BOTH kinds: THT always (nominal 0.85);
                SMT only when given -- an unset SMT joint keeps the
                paste-driven nominal (the printed deposit is mass
                conservation).
            component_types: Optional component-family allow-list. ``None``
                draws from the scene's selected component pool.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``component_types``.

        Returns:
            This operation section for call chaining.

        Example:

        .. code-block:: python

            job.operations.add_dry_contact(
                ratio=1.0, gap_fraction=[0.25, 0.5],
                component_types=["TO254"],
            )

        """
        return self._add_solder_defect(
            "dry_contact",
            ratio,
            component_types,
            component_filter,
            ("gap_fraction", gap_fraction, 0.0, 1.0),
            ("lead_climb", lead_climb, 0.0, 2.0),
            ("fullness", fullness, 0.0, 2.0),
        )

    def _add_solder_defect(
        self,
        defect: str,
        ratio: float,
        component_types: list[str] | tuple[str, ...] | None,
        component_filter: object | None,
        *params: tuple[str, object, float, float],
    ) -> PcbAoiOperationsSection:
        """Record one solder-defect operation (shared by the four adders).

        Selection mirrors the pose operations: each defect type independently
        selects ``ratio`` of the (remaining) eligible pool, non-overlapping
        across types and with the pose operations. Every parameter is a
        POST-SOLVE OBSERVABLE -- measurable on the produced geometry, never a
        solver internal (see the solder defects in ``pcb_aoi/config/README.md``).

        Raises:
            ValidationError: If arguments are malformed,
                ``operations.solder_joints`` was not declared first, or its
                solver does not honour defect intent.

        """
        where = f"operations.{defect}"
        if "solder_defects" not in self._domain.job_operations:
            raise ValidationError(
                f"operation {defect!r} is not supported by domain {self._domain.name!r}"
            )
        # The solve everything rides on, declared first so the solver is
        # known. A solver that silently ignores defect records (procedural)
        # would produce nominal joints under defect labels -- mislabelled
        # training data -- so only defect-honouring (simulated) solvers pass.
        if self.solder_joints_spec is None:
            raise ValidationError(
                f"{where} requires operations.solder_joints(...) to be declared first",
            )
        from .solder_generation import SolderSolverType, resolve_solver

        solver_id = str(self.solder_joints_spec["solver"])
        if resolve_solver(solver_id).solver_type is not SolderSolverType.SIMULATED:
            raise ValidationError(
                f"{where}: solver {solver_id!r} does not honour defect intent; "
                "a defect it ignores is mislabelled data. Use a simulated solver "
                "(e.g. 'surface_evolver')",
            )
        if component_types is not None and not (
            isinstance(component_types, (list, tuple))
            and component_types
            and all(isinstance(t, str) and t for t in component_types)
        ):
            raise ValidationError(
                f"{where}: component_types must be a list of non-empty family "
                "names or None",
            )
        normalized_filter = _normalize_filter(
            where,
            component_filter,
            legacy_selector=component_types,
        )
        spec: dict[str, object] = {
            "defect": defect,
            "ratio": _ratio(f"{where}.ratio", ratio),
            "component_types": (
                list(dict.fromkeys(component_types))
                if component_types is not None
                else None
            ),
            "component_filter": normalized_filter,
        }
        for pname, pvalue, plo, phi in params:
            if pvalue is None:
                continue
            spec[pname] = _observable_spec(f"{where}.{pname}", pvalue, lo=plo, hi=phi)
        self.solder_defect_specs[defect] = spec
        self._job._record_declaration(
            f"operation.solder_defects.{defect}",
            "operation.solder_defects.configure",
            spec,
        )
        return self

    def get_solder_joint_solvers(
        self,
        mode_filter: str | None = None,
    ) -> list[dict[str, object]]:
        """List registered solder-generation solvers.

        Args:
            mode_filter: Optional ``none``, ``procedural``, or ``simulated``
                solver-type filter.

        Returns:
            Solver IDs, types, availability, and optional unavailability reason.

        Raises:
            ValidationError: If solder generation is unsupported by the domain
                or the filter is invalid.

        Example:

        .. code-block:: python

            solvers = job.operations.get_solder_joint_solvers(
                mode_filter="simulated",
            )

        """
        if "solder_joints" not in self._domain.job_operations:
            raise ValidationError(
                f"operation 'solder_joints' is not supported by domain {self._domain.name!r}"
            )
        from .solder_generation import registered_solvers

        results = []
        for solver in registered_solvers(mode_filter):
            result = {
                "id": solver.solver_id,
                "type": solver.solver_type.value,
                "available": solver.available,
            }
            if solver.unavailable_reason:
                result["reason"] = solver.unavailable_reason
            results.append(result)
        return results


class PcbAoiCapturePassSection(CapturePassSection):
    """One render pass, statable the way an AOI test program is written.

    Core capture is stated in the terms a renderer uses: a resolution, a camera
    and a grid of cells. An AOI test program is not written that way -- it says
    "scan this rectangle of the board at this dpi" -- so this adds
    :meth:`fov`, which is the one place the two vocabularies meet.
    """

    def fov(
        self,
        *,
        scan_area_mm=None,
        corners_mm=None,
        dpi=None,
        mm_per_px=None,
        image_size_px=None,
        max_tile_px=(4096, 4096),
        centered_on="unspecified",
        guard_px=None,
    ):
        """State the capture as a scan rectangle in millimetres at a resolution.

        Calling this derives the renderer resolution and the scan-grid counts
        from that rectangle, so the pixel count is computed in one place instead
        of independently on each side of the interface.

        The call order does not matter. The frame turns with the camera's yaw,
        so a yaw declared after this one is applied to the derivation, and a
        renderer declared after it keeps the derived resolution.

        Args:
            scan_area_mm: ``(length_x, breadth_y)`` of board to capture,
                centered on the board.
            corners_mm: Two opposite corners of the region to capture, each
                ``(x, y)`` or ``(x, y, z)`` in world millimetres -- the
                lower-left and upper-right points a machine test program states
                its field of view as. They give the size and also place the
                rectangle; ``z`` is ignored. Exactly one of ``scan_area_mm``
                and ``corners_mm``.
            dpi: Scanning resolution. A number, or a pair of equal numbers:
                pixels are square, because an orthographic render cannot
                deliver two pitches, so an unequal pair is refused. The
                scanner offers 600 or 1200; anything else renders but is
                reported as having no counterpart on the machine.
            mm_per_px: The same resolution as a pixel pitch, the way a sensor
                spec sheet states it (0.042333 for 600 dpi). A number or an
                equal pair, as for ``dpi``.
            image_size_px: ``(width, height)`` of the stitched canvas. Give this
                instead of a resolution to pin the output size and let dpi
                follow. The pixel count is kept exactly; the pitch is the
                coarser of ``area / pixels`` on the two axes, used on both, so
                when the canvas does not have the area's aspect the other axis
                covers more than was asked, centred on the requested rectangle.
            max_tile_px: Largest single rendered frame, guard band included;
                larger canvases tile.
            centered_on: ``"panel"``, ``"instance"`` or ``"unspecified"`` -- what
                the rectangle is centred on. Recorded and carried to the
                sidecar, never inferred: for a multi-up panel "the board" is
                ambiguous.
            guard_px: Pixels each cell renders beyond every edge and crops
                before its tile is written, so the renderer's unreliable outer
                row and column never reach a seam. ``None`` takes
                :data:`pcb_aoi.geometry.fov.DEFAULT_GUARD_PX`; ``0`` renders the
                delivered tile directly. Tiles, pixel layout and calibration are
                the same shape either way.

        Returns:
            This section, for chaining.

        Raises:
            ValidationError: If not exactly one of ``scan_area_mm`` and
                ``corners_mm``, or of ``dpi``, ``mm_per_px`` and
                ``image_size_px``, is given, a dimension is not positive, a
                ``dpi`` or ``mm_per_px`` pair asks for non-square pixels, or
                the pass already states its resolution directly.

        """
        from pcb_aoi.geometry.fov import DEFAULT_GUARD_PX, FovError, resolve_fov

        if self.resolution_source == "author":
            raise ValidationError(
                "capture.fov: this pass already states renderer.resolution; "
                "give one or the other -- a fov derives the resolution"
            )
        if self.grid_source == "author":
            raise ValidationError(
                "capture.fov: this pass already states capture.scan_grid; "
                "give one or the other -- a fov derives the grid"
            )
        try:
            resolved = resolve_fov(
                # The stage scale is not known at declaration time; the capture
                # stage supplies the real value. This only sets the units the
                # apertures are reported in here, and cancels out of the pixel
                # arithmetic, which is what actually leaves this section.
                stage_meters_per_unit=0.01,
                scan_area_mm=list(scan_area_mm) if scan_area_mm is not None else None,
                corners_mm=(
                    [list(corner) for corner in corners_mm]
                    if corners_mm is not None
                    else None
                ),
                dpi=list(dpi) if isinstance(dpi, (list, tuple)) else dpi,
                mm_per_px=(
                    list(mm_per_px)
                    if isinstance(mm_per_px, (list, tuple))
                    else mm_per_px
                ),
                image_size_px=list(image_size_px) if image_size_px else None,
                max_tile_px=list(max_tile_px),
                centered_on=centered_on,
                guard_px=DEFAULT_GUARD_PX if guard_px is None else guard_px,
            )
        except FovError as exc:
            raise ValidationError(f"capture.fov: {exc}") from exc

        self.fov_spec = resolved
        self.resolution_source = "derived"
        self._refresh_derived_capture()
        return self

    def _refresh_derived_capture(self):
        """Re-derive the frame and the grid from the current camera.

        Called again whenever the camera or the renderer changes, which is what
        makes the public calls commute. Deriving once at :meth:`fov` time read
        whatever yaw happened to be declared already -- zero, if the camera came
        second -- and a capture framed square to the wrong axis is not visible
        in any number the job records.
        """
        resolved = getattr(self, "fov_spec", None)
        if not resolved:
            return
        # The rectangle is stated in world axes; the rendered frame is stated in
        # the camera's own axes. A fixed yaw of +-90 deg puts the camera's WIDTH
        # along world Y, so the frame turns with it. The scan grid stays
        # world-axis-aligned either way.
        z_fixed = float((getattr(self, "camera_spec", None) or {}).get("z_fixed", 0.0))
        swap = round(z_fixed / 90.0) % 2 == 1
        tile_w, tile_h = resolved["tile_size_px"]
        frame_px = [tile_h, tile_w] if swap else [tile_w, tile_h]
        image_w, image_h = resolved["image_size_px"]
        image_px = [image_h, image_w] if swap else [image_w, image_h]
        resolved["camera_z_rotation_deg"] = z_fixed
        resolved["frame_size_px"] = frame_px
        guard = int(resolved.get("guard_px", 0))
        resolved["render_frame_size_px"] = [value + 2 * guard for value in frame_px]

        # The resolution is derived, not accepted: a fov states the pixel count
        # through the pitch, and a separately authored resolution could only
        # disagree with it. Re-record the renderer in place rather than asking
        # the caller to compute the same number a second time.
        if getattr(self, "renderer_spec", None):
            self.renderer_spec = {**self.renderer_spec, "resolution": frame_px}
            self._job._record_declaration(
                self._order_id("renderer"),
                "capture.renderer.configure",
                self.renderer_spec,
                replace=True,
                preserve_position=True,
            )
        # The millimetres go to the grid, not a derived aperture. Two reasons,
        # both found on a real render: the grid's orthographic auto-fit grows
        # any authored aperture back to bbox/count, and the conversion from
        # millimetres to scene units needs the stage's own scale, which is not
        # known here. Guessing it put the frame out by a factor of ten.
        # Corners place the rectangle; a size alone is centered on the board and
        # carries the label saying which reading of "the board" was meant.
        placed = resolved.get("center_mm") is not None
        self.scan_grid(
            x_count=resolved["x_num"],
            y_count=resolved["y_num"],
            # The cells span the whole mosaic; the stitch crops it to the image
            # size, which is what the scanner would have produced.
            scan_area_mm=resolved["scan_area_mm_rendered"],
            centered_on=None if placed else resolved["centered_on"],
            scan_center_mm=resolved["center_mm"] if placed else None,
            image_size_px=image_px,
            # Square: the band is the same on every side, so the camera's yaw
            # does not change it.
            guard_px=guard,
            _derived=True,
        )
        # No declaration of its own, and no aperture written here: every effect
        # a fov has is already on the record as capture.renderer.configure plus
        # capture.scan_grid.configure, and the aperture is derived at render
        # time from those millimetres and the stage's own metersPerUnit.
        # Recording an order with no runtime behaviour would need a no-op
        # handler and would put a lie in the job order -- a step that executes
        # nothing.
        return


class PcbAoiCaptureSection(CaptureSection, PcbAoiCapturePassSection):
    """The domain's capture section: named passes speak the same vocabulary."""

    _pass_class = PcbAoiCapturePassSection
