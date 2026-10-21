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

"""PCB-AOI operation models and Python API interpretation."""

from __future__ import annotations

import os
from typing import ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from cdf.component_filter import ComponentFilter
from cdf.config._common import ConfigModel, Number, resolve_path
from cdf.job import CDFJob
from pcb_aoi.config.scene import (
    ComponentMetadataSource,
    component_metadata_api_source,
)


class _TargetConfig(ConfigModel):
    """Shared exact-count or proportional defect targeting."""

    default_ratio: ClassVar[float | None] = None
    ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    count: int | None = Field(default=None, ge=1)
    component_filter: ComponentFilter | None = None

    @model_validator(mode="after")
    def validate_targeting(self) -> _TargetConfig:
        """Require one targeting mode unless the operation defines a default.

        Returns:
            Validated operation targeting.

        Raises:
            ValueError: If targeting is missing, ambiguous, or invalid.

        """
        if self.ratio is not None and self.count is not None:
            raise ValueError("ratio and count are mutually exclusive")
        if self.ratio is None and self.count is None and self.default_ratio is None:
            raise ValueError("exactly one of ratio or count is required")
        if (
            self.component_filter is not None
            and getattr(self, "component_types", None) is not None
        ):
            raise ValueError(
                "component_types and component_filter are mutually exclusive",
            )
        return self


class ShiftConfig(_TargetConfig):
    """Settings for positional and rotational component-shift defects."""

    translate: float
    rotate_z: float
    #: Deterministic stratification index; omit to keep the random draw.
    #: Bounded here as well as on the execution parameters, so a bad config
    #: fails at load rather than during order compilation.
    variant_index: int | None = Field(default=None, ge=0)


class TombstoneConfig(_TargetConfig):
    """Tombstone-defect and component-eligibility settings."""

    angle: tuple[Number, Number]
    component_types: tuple[str, ...] | None = None
    auto_filter: bool = True
    eligibility_source: str | None = None


class SideflipConfig(_TargetConfig):
    """Side-flip defect ratio and rotation-range settings."""

    angle: tuple[Number, Number]
    #: Deterministic stratification index; omit to keep the random draw.
    variant_index: int | None = Field(default=None, ge=0)
    #: Extra rotation spread about the board normal, in degrees.
    rotate_z_jitter_deg: float = Field(default=0.0, ge=0.0)
    #: Extra in-plane slide applied after the flip, in millimetres.
    slide_range_mm: float = Field(default=0.0, ge=0.0)


class ReversePolarityConfig(_TargetConfig):
    """Polarity-reversal settings for eligible component types."""

    default_ratio: ClassVar[float | None] = 1.0
    component_types: tuple[str, ...] | None = None


class MissingConfig(_TargetConfig):
    """Settings for component-removal defects."""

    #: Prune the prim instead of authoring USD invisibility, so the component
    #: is absent from RGB as well as from segmentation.


class PinVariationGroupConfig(ConfigModel):
    """One set of parts and the bands to vary their through-hole leads over."""

    components: tuple[str, ...] | None = None
    package_types: tuple[str, ...] | None = None
    component_filter: ComponentFilter | None = None
    tilt: tuple[Number, Number] | None = None
    cut_angle: tuple[Number, Number] | None = None
    height: tuple[Number, Number] | None = None
    offset: tuple[Number, Number] | None = None


class SolderJointsConfig(ConfigModel):
    """Procedural solder generation, selection, and serialization settings."""

    solver: str = "none"
    component_types: tuple[str, ...] | None = None
    component_filter: ComponentFilter | None = None
    source: str | None = None
    options: dict[str, object] = Field(default_factory=dict)
    output_root_path: str | None = None
    serialization: Literal["none", "solder-joints-only", "all"] | None = None
    serialization_path: str | None = None


class HolePlaneConfig(ConfigModel):
    """Board-sized hole-reveal plane for through-hole segmentation.

    A quad mesh is placed at ``z_offset_mm`` inside the board substrate (FR4)
    from the chosen substrate face.  Substrate geometry occludes the plane
    everywhere except at through-hole voids so the segmentation annotator sees
    it only where holes exist.

    The z reference is the ``board_substrate`` (FR4) face — not the solder-mask
    or paste layers. Generated meshes are hidden by default; use
    ``visibility.show: [hole]`` in an annotation pass to reveal them.

    Generated meshes are children of an Xform at ``output_prim_path`` (or the
    default ``SegmentationProxies/HolePlane`` path), so the proxies can be shown
    or hidden as one group. When ``components`` is set, one small plane is
    created per pad belonging to those components instead of a single
    board-wide plane.
    """

    class_name: str = "hole"
    side: Literal["top", "bottom", "both"]
    z_offset_mm: float = Field(default=0.1, ge=0.0)
    output_prim_path: str | None = None
    components: list[str] | None = None
    component_filter: ComponentFilter | None = None

    @field_validator("output_prim_path")
    @classmethod
    def validate_output_prim_path(cls, value: str | None) -> str | None:
        """Require an absolute non-root output prim path when provided.

        Args:
            value: Optional output prim path.

        Returns:
            Validated path or None.

        Raises:
            ValueError: If the path is not an absolute non-root prim path.

        """
        if value is not None and (
            not value.startswith("/") or value == "/" or "//" in value
        ):
            raise ValueError("output_prim_path must be an absolute non-root prim path")
        return value


class PinTipConfig(ConfigModel):
    """Annotation-only through-hole lead-tip proxy settings.

    The proxy covers the terminal's outward tip surface up to a vertical wall
    or 30 degrees of tilt from the tip's outward normal, whichever comes first,
    so the chamfer on a rounded lead end keeps the terminal's own class.
    """

    side: Literal["top", "bottom"]
    class_name: str = "pin_tip"
    offset_mm: float = Field(default=0.002, gt=0.0)
    components: tuple[str, ...] | None = None
    component_filter: ComponentFilter | None = None


class InsufficientConfig(ConfigModel):
    """Starved-solder defect: reduced volume on the nominal footprint.

    ``volume_ratio`` is a post-solve observable (mesh volume over nominal, in
    (0, 1)): a value or a ``[min, max]`` range sampled per component.
    ``lead_climb`` (THT-only) and ``fullness`` (both kinds) shape the
    joints -- on THT the "less solder" valley: values or ``[min, max]``
    ranges sampled per joint; when either is given it drives the target and
    ``volume_ratio`` is measured-only on the joints it shapes.
    """

    ratio: float
    volume_ratio: Number | tuple[Number, Number] | None = None
    lead_climb: Number | tuple[Number, Number] | None = None
    fullness: Number | tuple[Number, Number] | None = None
    component_types: tuple[str, ...] | None = None
    component_filter: ComponentFilter | None = None


class ExcessConfig(ConfigModel):
    """Excess-solder defect: increased volume on the nominal footprint.

    ``volume_ratio`` is a post-solve observable (mesh volume over nominal,
    > 1): a value or a ``[min, max]`` range sampled per component.
    ``lead_climb`` (THT-only; > 1 buries the tip -- the "hidden/excess"
    class) and ``fullness`` (both kinds; >= 1 bulges the fillet) shape the
    joints: values or ``[min, max]`` ranges sampled per joint; when either
    is given it drives the target and ``volume_ratio`` is measured-only on
    the joints it shapes.
    """

    ratio: float
    volume_ratio: Number | tuple[Number, Number] | None = None
    lead_climb: Number | tuple[Number, Number] | None = None
    fullness: Number | tuple[Number, Number] | None = None
    component_types: tuple[str, ...] | None = None
    component_filter: ComponentFilter | None = None


class BrightConfig(ConfigModel):
    """Bright-solder defect: a steep, starved THT meniscus. THT-only.

    ``lead_climb`` and ``fullness`` are the shape parameters (values or
    ``[min, max]`` ranges sampled per joint): a respectable climb with a
    starved fill, so the surface necks in and reads shiny.
    """

    ratio: float
    lead_climb: Number | tuple[Number, Number] | None = None
    fullness: Number | tuple[Number, Number] | None = None
    component_types: tuple[str, ...] | None = None
    component_filter: ComponentFilter | None = None


class InterruptedConfig(ConfigModel):
    """Interrupted-solder defect: the pad itself is partially exposed.

    ``pad_exposure`` (exposed pad area / total pad area, in (0, 1)) is a
    value or ``[min, max]`` range sampled per joint; the exposure azimuth is
    always randomized. Both kinds; on THT the ring around the pin stays
    closed (a broken ring is ``dry_contact``). ``lead_climb``/``fullness``
    optionally shape the remaining fillet.
    """

    ratio: float
    pad_exposure: Number | tuple[Number, Number] | None = None
    lead_climb: Number | tuple[Number, Number] | None = None
    fullness: Number | tuple[Number, Number] | None = None
    component_types: tuple[str, ...] | None = None
    component_filter: ComponentFilter | None = None


class MissingSolderConfig(ConfigModel):
    """Missing-solder defect: the component stays, its solder does not."""

    ratio: float
    component_types: tuple[str, ...] | None = None
    component_filter: ComponentFilter | None = None


class DryContactConfig(ConfigModel):
    """Interrupted through-hole fillet: solder does not ring the pin.

    ``gap_fraction`` is a post-solve observable (unwetted fraction of the
    barrel circumference, in (0, 1)): a value or a ``[min, max]`` range
    sampled per joint. The gap's azimuth is always randomized per joint.
    """

    ratio: float
    gap_fraction: Number | tuple[Number, Number] | None = None
    lead_climb: Number | tuple[Number, Number] | None = None
    fullness: Number | tuple[Number, Number] | None = None
    component_types: tuple[str, ...] | None = None
    component_filter: ComponentFilter | None = None


class OperationsConfig(ConfigModel):
    """Optional PCB structural and solder operations for a job."""

    shift: ShiftConfig | None = None
    tombstone: TombstoneConfig | None = None
    sideflip: SideflipConfig | None = None
    reverse_polarity: ReversePolarityConfig | None = None
    missing: MissingConfig | None = None
    #: A LIST, not a mapping: the Python API accepts repeated
    #: `pin_variation(...)` calls for different sets of parts, and a single
    #: mapping could only ever express one of them.
    pin_variation: tuple[PinVariationGroupConfig, ...] | None = None
    solder_joints: SolderJointsConfig | None = None
    hole_plane: HolePlaneConfig | None = None
    pin_tip: PinTipConfig | None = None
    insufficient: InsufficientConfig | None = None
    excess: ExcessConfig | None = None
    bright: BrightConfig | None = None
    interrupted: InterruptedConfig | None = None
    missing_solder: MissingSolderConfig | None = None
    dry_contact: DryContactConfig | None = None

    @model_validator(mode="after")
    def validate_component_filters(self) -> OperationsConfig:
        """Reject mixed normalized and overloaded selectors.

        Returns:
            Validated operation configuration.

        Raises:
            ValueError: If an operation supplies both selector forms.

        """
        for index, group in enumerate(self.pin_variation or ()):
            supplied = sum(
                value is not None
                for value in (
                    group.components,
                    group.package_types,
                    group.component_filter,
                )
            )
            if supplied != 1:
                raise ValueError(
                    f"pin_variation.{index} requires exactly one of components, "
                    "package_types, or component_filter",
                )
        for name in ("hole_plane", "pin_tip"):
            value = getattr(self, name)
            if (
                value is not None
                and value.components is not None
                and value.component_filter is not None
            ):
                raise ValueError(
                    f"{name}.components and component_filter are mutually exclusive",
                )
        for name in (
            "solder_joints",
            "insufficient",
            "excess",
            "bright",
            "interrupted",
            "missing_solder",
            "dry_contact",
        ):
            value = getattr(self, name)
            if (
                value is not None
                and value.component_types is not None
                and value.component_filter is not None
            ):
                raise ValueError(
                    f"{name}.component_types and component_filter are mutually exclusive",
                )
        return self


def apply_operations(
    job: CDFJob,
    config: OperationsConfig,
    *,
    base_dir: str | os.PathLike[str],
    component_metadata_source: ComponentMetadataSource,
    selected_types: tuple[str, ...] | None,
) -> None:
    """Apply PCB-AOI operation declarations."""
    active_source = component_metadata_api_source(component_metadata_source)
    if config.shift is not None:
        value = config.shift
        job.operations.add_shift(
            ratio=value.ratio,
            count=value.count,
            translate_range=value.translate,
            rotate_z_range=value.rotate_z,
            variant_index=value.variant_index,
            component_filter=value.component_filter,
        )
    if config.tombstone is not None:
        value = config.tombstone
        eligibility_source = value.eligibility_source or active_source
        if eligibility_source != active_source:
            raise ValueError(
                "tombstone eligibility_source must match the component metadata "
                f"source selected by scene configuration ({active_source!r})"
            )
        job.operations.add_tombstone(
            ratio=value.ratio,
            count=value.count,
            angle_range=value.angle,
            component_types=value.component_types,
            auto_filter=value.auto_filter,
            eligibility_source=eligibility_source,
            component_filter=value.component_filter,
        )
    if config.sideflip is not None:
        job.operations.add_sideflip(
            ratio=config.sideflip.ratio,
            count=config.sideflip.count,
            angle_range=config.sideflip.angle,
            variant_index=config.sideflip.variant_index,
            rotate_z_jitter_deg=config.sideflip.rotate_z_jitter_deg,
            slide_range_mm=config.sideflip.slide_range_mm,
            component_filter=config.sideflip.component_filter,
        )
    if config.reverse_polarity is not None:
        job.operations.add_reverse_polarity(
            ratio=config.reverse_polarity.ratio,
            count=config.reverse_polarity.count,
            component_types=config.reverse_polarity.component_types,
            component_filter=config.reverse_polarity.component_filter,
        )
    if config.missing is not None:
        job.operations.add_missing(
            ratio=config.missing.ratio,
            count=config.missing.count,
            component_filter=config.missing.component_filter,
        )
    if config.pin_variation is not None:
        for group in config.pin_variation:
            job.operations.pin_variation(
                components=group.components,
                package_types=group.package_types,
                component_filter=group.component_filter,
                tilt_range=group.tilt,
                cut_angle=group.cut_angle,
                height_variation=group.height,
                offset_range=group.offset,
            )
    if config.solder_joints is not None:
        value = config.solder_joints
        source = value.source or active_source
        if source != active_source:
            raise ValueError(
                "solder_joints source must match the component metadata source "
                f"selected by scene configuration ({active_source!r})"
            )
        job.operations.solder_joints(
            solver=value.solver,
            component_types=(
                value.component_types or selected_types
                if value.component_filter is None
                else None
            ),
            component_filter=value.component_filter,
            source=source,
            solver_options=value.options,
            output_root_path=value.output_root_path,
            serialization=value.serialization,
            serialization_path=(
                resolve_path(value.serialization_path, base_dir)
                if value.serialization_path is not None
                else None
            ),
        )
    if config.hole_plane is not None:
        value = config.hole_plane
        job.operations.hole_plane(
            class_name=value.class_name,
            side=value.side,
            z_offset_mm=value.z_offset_mm,
            output_prim_path=value.output_prim_path,
            components=value.components,
            component_filter=value.component_filter,
        )
    if config.pin_tip is not None:
        value = config.pin_tip
        job.operations.pin_tip(
            side=value.side,
            class_name=value.class_name,
            offset_mm=value.offset_mm,
            components=value.components,
            component_filter=value.component_filter,
        )
    # Solder defects come after solder_joints on purpose: each adder
    # validates that the declared solver honours defect intent.
    if config.insufficient is not None:
        job.operations.add_insufficient(
            ratio=config.insufficient.ratio,
            volume_ratio=config.insufficient.volume_ratio,
            lead_climb=config.insufficient.lead_climb,
            fullness=config.insufficient.fullness,
            component_types=config.insufficient.component_types,
            component_filter=config.insufficient.component_filter,
        )
    if config.excess is not None:
        job.operations.add_excess(
            ratio=config.excess.ratio,
            volume_ratio=config.excess.volume_ratio,
            lead_climb=config.excess.lead_climb,
            fullness=config.excess.fullness,
            component_types=config.excess.component_types,
            component_filter=config.excess.component_filter,
        )
    if config.bright is not None:
        job.operations.add_bright(
            ratio=config.bright.ratio,
            lead_climb=config.bright.lead_climb,
            fullness=config.bright.fullness,
            component_types=config.bright.component_types,
            component_filter=config.bright.component_filter,
        )
    if config.interrupted is not None:
        job.operations.add_interrupted(
            ratio=config.interrupted.ratio,
            pad_exposure=config.interrupted.pad_exposure,
            lead_climb=config.interrupted.lead_climb,
            fullness=config.interrupted.fullness,
            component_types=config.interrupted.component_types,
            component_filter=config.interrupted.component_filter,
        )
    if config.missing_solder is not None:
        job.operations.add_missing_solder(
            ratio=config.missing_solder.ratio,
            component_types=config.missing_solder.component_types,
            component_filter=config.missing_solder.component_filter,
        )
    if config.dry_contact is not None:
        job.operations.add_dry_contact(
            ratio=config.dry_contact.ratio,
            gap_fraction=config.dry_contact.gap_fraction,
            lead_climb=config.dry_contact.lead_climb,
            fullness=config.dry_contact.fullness,
            component_types=config.dry_contact.component_types,
            component_filter=config.dry_contact.component_filter,
        )
