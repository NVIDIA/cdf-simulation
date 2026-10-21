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

"""Validate version-1 parameters consumed by Isaac JobOrder handlers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from cdf.component_filter import ComponentFilter

from ...image_formats import ImageFormat

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class SceneLoadParameters(BaseModel):
    """Version-1 parameters for ``scene.load``.

    Args:
        usd_path: USD stage path to open.
        archive_member: Root-level USD filename to select from a ZIP archive.
        root_path: Absolute prim path containing the board.
        camera_path: Optional camera prim path.
        camera_xform_path: Optional camera transform prim path.
        ring_light_root: Optional AOI ring-light root path.
        horizontal_aperture: Optional camera aperture override.
        meters_per_unit: Optional metres-per-scene-unit override -- USD's own
            ``metersPerUnit``, so ``0.001`` for a millimetre stage -- for a
            stage that declares a scale its geometry does not use.
        instance_paths: Ordered board-instance cohort, when configured.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    usd_path: NonEmptyString
    root_path: NonEmptyString
    archive_member: NonEmptyString | None = None
    camera_path: str | None = None
    camera_xform_path: str | None = None
    ring_light_root: str | None = None
    horizontal_aperture: float | None = None
    meters_per_unit: float | None = Field(default=None, gt=0.0)
    instance_paths: tuple[NonEmptyString, ...] | None = None

    @model_validator(mode="after")
    def validate_root_path(self) -> SceneLoadParameters:
        """Require absolute scene and board-instance prim paths.

        Returns:
            Validated scene parameters.

        Raises:
            ValueError: If the board root is not absolute.

        """
        if not self.root_path.startswith("/"):
            raise ValueError("root_path must be an absolute prim path")
        if self.instance_paths is not None:
            if not self.instance_paths:
                raise ValueError("instance_paths cannot be empty")
            if len(set(self.instance_paths)) != len(self.instance_paths):
                raise ValueError("instance_paths must be unique")
            if any(
                not path.startswith("/") or path == "/" or "//" in path
                for path in self.instance_paths
            ):
                raise ValueError(
                    "instance_paths must contain absolute non-root prim paths",
                )
        return self


class SidecarLoadParameters(BaseModel):
    """Version-1 parameters for ``scene.sidecar.load``.

    Args:
        name: Name used by dependent JobOrders.
        kind: Semantic sidecar kind registered by the CDF domain.
        path: JSON sidecar path to load.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: NonEmptyString
    kind: NonEmptyString
    path: NonEmptyString


class ComponentSelectionParameters(BaseModel):
    """Version-1 parameters for ``scene.components.select``.

    Args:
        source: Optional board metadata source name.
        types: Optional explicit component-family names.
        component_filter: Optional exact metadata-field filter.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str | None = None
    types: tuple[NonEmptyString, ...] | None = None
    component_filter: ComponentFilter | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> ComponentSelectionParameters:
        """Reject mixing normalized and overloaded selector forms.

        Returns:
            Validated selection parameters.

        Raises:
            ValueError: If both selector forms are supplied.

        """
        if self.types is not None and self.component_filter is not None:
            raise ValueError("types and component_filter are mutually exclusive")
        return self


LightingMode = Literal[
    "rgb_light",
    "white_light",
    "dome_light",
    "modus_light",
    "usd_light",
]
#: The five rigs ``appearance.lighting`` can select. Mirrors
#: :data:`cdf.sections.LIGHTING_MODES`; kept as a Literal so an unknown rig
#: is rejected at the transport boundary rather than deep inside the pipeline.


class LightingParameters(BaseModel):
    """Version-1 parameters for ``appearance.lighting.configure``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The rig this job builds. Two rigs can never be active at once, so one
    #: required selector replaces the former use_scene_lights / ring_light pair.
    mode: LightingMode
    preserve_scene_light_color: bool = False
    exposure_range: tuple[float, float] | None = None
    cone_angle_range: tuple[float, float] | None = None
    cone_softness_range: tuple[float, float] | None = None
    dome_fill: Mapping[str, object] | None = None
    layers: Mapping[str, Mapping[str, object]] | None = None
    white_light: Mapping[str, object] | None = None
    #: Modus 2D-dome linear illuminator block; only for ``mode="modus_light"``.
    modus: Mapping[str, object] | None = None


class LightingApplyParameters(BaseModel):
    """Version-1 parameters for ``appearance.lighting.apply``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    camera_side: Literal["top", "bottom"] = "top"


class ComponentAppearanceParameters(BaseModel):
    """Version-1 parameters for ``appearance.components.configure``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vantablack: bool = False


class MaterialOverrideParameters(BaseModel):
    """Version-1 parameters for ``appearance.material.override``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    material_path: NonEmptyString
    diffuse_texture: NonEmptyString | None = None
    roughness_texture: NonEmptyString | None = None
    normal_texture: NonEmptyString | None = None
    diffuse_color: tuple[float, float, float] = (0.2, 0.2, 0.2)
    diffuse_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    albedo_brightness: float = Field(default=1.0, ge=0.0)
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    roughness: float = Field(default=0.5, ge=0.0, le=1.0)
    roughness_texture_influence: float = Field(default=1.0, ge=0.0, le=1.0)
    metallic: float = Field(default=0.0, ge=0.0, le=1.0)
    specular_level: float = Field(default=0.5, ge=0.0, le=1.0)
    normal_strength: float = Field(default=1.0, ge=0.0)
    normal_format: Literal["opengl", "directx"] = "opengl"
    texture_scale: tuple[float, float] = (1.0, 1.0)
    texture_rotation: float = 0.0
    project_uvw: bool = False
    world_space_projection: bool = False
    variation: dict[str, tuple[float, float]] = Field(default_factory=dict)

    @field_validator("variation")
    @classmethod
    def validate_variation(
        cls,
        value: dict[str, tuple[float, float]],
    ) -> dict[str, tuple[float, float]]:
        """Require known controls and ordered, finite bounds."""
        allowed = {
            "albedo_brightness",
            "roughness",
            "metallic",
            "specular_level",
            "normal_strength",
            "texture_rotation",
            "texture_scale",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown material variation controls: {unknown}")
        for control, bounds in value.items():
            if bounds[0] > bounds[1]:
                raise ValueError(
                    f"material variation {control} bounds must be ordered",
                )
        return value

    @field_validator("material_path")
    @classmethod
    def validate_material_path(cls, value: str) -> str:
        """Require an absolute non-root material prim path."""
        if not value.startswith("/") or value == "/":
            raise ValueError("material_path must be an absolute non-root prim path")
        return value


class OperationLabelParameters(BaseModel):
    """Version-1 parameters for ``annotations.operations.configure``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: NonEmptyString = "defect"


class ComponentAnnotationParameters(BaseModel):
    """Version-1 parameters for ``annotations.components.apply``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str | None = None
    classes: Mapping[str, tuple[NonEmptyString, ...]] | None = None


class GeometryAnnotationParameters(BaseModel):
    """Version-1 parameters for ``annotations.geometry.apply``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    solder: NonEmptyString | None = None
    terminals: NonEmptyString | None = None
    pads: NonEmptyString | None = None
    holes: NonEmptyString | None = None
    barrels: NonEmptyString | None = None
    body: NonEmptyString | None = None
    marking: NonEmptyString | None = None
    components: tuple[NonEmptyString, ...] | None = None
    component_filter: ComponentFilter | None = None

    @model_validator(mode="after")
    def validate_targets(self) -> GeometryAnnotationParameters:
        """Require at least one geometry class.

        Returns:
            Validated geometry-annotation parameters.

        Raises:
            ValueError: If no geometry target was given a class or selector
                forms are mixed.

        """
        if not any(
            (
                self.solder,
                self.terminals,
                self.pads,
                self.holes,
                self.barrels,
                self.body,
                self.marking,
            )
        ):
            raise ValueError(
                "geometry annotations require at least one geometry class",
            )
        if self.components is not None and self.component_filter is not None:
            raise ValueError("components and component_filter are mutually exclusive")
        return self


class _TargetParameters(BaseModel):
    """Shared exact-count or proportional structural-operation targeting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    count: int | None = Field(default=None, ge=1)
    component_filter: ComponentFilter | None = None

    @model_validator(mode="after")
    def validate_targeting(self) -> _TargetParameters:
        """Require exactly one structural-operation targeting mode.

        Returns:
            Validated targeting parameters.

        Raises:
            ValueError: If both or neither targeting modes are supplied.

        """
        if (self.ratio is None) == (self.count is None):
            raise ValueError("exactly one of ratio or count is required")
        if (
            self.component_filter is not None
            and getattr(self, "component_types", None) is not None
        ):
            raise ValueError(
                "component_types and component_filter are mutually exclusive",
            )
        return self


class _AngleRangeParameters(_TargetParameters):
    """Shared validated angle range for structural operations."""

    angle_range: tuple[float, float]

    @field_validator("angle_range")
    @classmethod
    def validate_angle_range(
        cls,
        angle_range: tuple[float, float],
    ) -> tuple[float, float]:
        """Require an ordered structural-defect angle range.

        Args:
            angle_range: Minimum and maximum defect angles.

        Returns:
            Validated angle range.

        Raises:
            ValueError: If the minimum angle exceeds the maximum.

        """
        if angle_range[0] > angle_range[1]:
            raise ValueError("angle_range minimum must not exceed maximum")
        return angle_range


class ShiftParameters(_TargetParameters):
    """Version-1 parameters for ``operation.shift.apply``."""

    translate_range: float = Field(ge=0.0)
    rotate_z_range: float = Field(ge=0.0)
    #: Deterministic stratification index; ``None`` keeps the random draw.
    variant_index: int | None = Field(default=None, ge=0)


class TombstoneParameters(_AngleRangeParameters):
    """Version-1 parameters for ``operation.tombstone.apply``."""

    component_types: tuple[NonEmptyString, ...] | None = None
    auto_filter: bool = True
    eligibility_source: str | None = None


class SideflipParameters(_AngleRangeParameters):
    """Version-1 parameters for ``operation.sideflip.apply``."""

    #: Deterministic stratification index; ``None`` keeps the random draw.
    variant_index: int | None = Field(default=None, ge=0)
    #: Extra yaw spread applied around the flipped pose, in degrees.
    rotate_z_jitter_deg: float = Field(default=0.0, ge=0.0)
    #: Extra in-plane slide applied after the flip, in millimetres.
    slide_range_mm: float = Field(default=0.0, ge=0.0)


class ReversePolarityParameters(_TargetParameters):
    """Version-1 parameters for ``operation.reverse_polarity.apply``."""

    component_types: tuple[NonEmptyString, ...] | None = None


class MissingParameters(_TargetParameters):
    """Version-1 parameters for ``operation.missing.apply``."""

    #: Prune the prim instead of authoring USD invisibility. Under the Fabric
    #: Scene Delegate a mid-run MakeInvisible does not reach the beauty pass,
    #: so the component still shows in RGB; pruning removes it everywhere.


class SolderJointParameters(BaseModel):
    """Validate internal solder-joint generation parameters.

    Args:
        solver: Registered solder-generation solver ID.
        source: Board metadata source name.
        component_types: Optional component-name or family allow-list.
        component_filter: Optional exact metadata-field filter.
        solver_options: Solver-specific JSON-compatible configuration.
        output_root_path: Optional absolute solder output prim path.
        serialization: Optional USD serialization policy.
        serialization_path: Optional serialized USD output path.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    solver: NonEmptyString
    #: Registered solder-generation solver ID.

    source: NonEmptyString = "usd"
    #: Board metadata source name.

    component_types: tuple[NonEmptyString, ...] | None = None
    #: Component-name or family allow-list, if provided.

    component_filter: ComponentFilter | None = None
    #: Explicit metadata-field filter, if provided.

    solver_options: dict[str, object] = Field(default_factory=dict)
    #: Solver-specific JSON-compatible configuration.

    output_root_path: NonEmptyString | None = None
    #: Absolute solder output prim path, or None to use the PCBA default.

    serialization: Literal["none", "solder-joints-only", "all"] = "none"
    #: Optional USD serialization policy.

    serialization_path: NonEmptyString | None = None
    #: Serialized USD output path when serialization is enabled.

    @field_validator("output_root_path")
    @classmethod
    def validate_output_root_path(cls, value: str | None) -> str | None:
        """Require an absolute non-root output prim path.

        Args:
            value: Optional output prim path.

        Returns:
            Validated output prim path, or None.

        Raises:
            ValueError: If the path is not an absolute non-root prim path.

        """
        if value is not None and (
            not value.startswith("/") or value == "/" or "//" in value
        ):
            raise ValueError("output_root_path must be an absolute non-root path")
        return value

    @model_validator(mode="after")
    def validate_serialization_config(self) -> SolderJointParameters:
        """Require a path exactly when serialization is enabled.

        Returns:
            Validated solder-joint parameters.

        Raises:
            ValueError: If path presence conflicts with serialization policy.

        """
        if self.serialization == "none" and self.serialization_path is not None:
            raise ValueError("serialization_path requires serialization")
        if self.serialization != "none" and self.serialization_path is None:
            raise ValueError("serialization_path is required by serialization")
        if self.component_types is not None and self.component_filter is not None:
            raise ValueError(
                "component_types and component_filter are mutually exclusive",
            )
        return self


class HolePlaneParameters(BaseModel):
    """Version-1 parameters for ``hole_plane.generate``.

    Args:
        class_name: Semantic class name authored on the generated prim.
        side: Which board substrate face to measure from: ``"bottom"`` for
            the solder side (z_lo), ``"top"`` for the component side (z_hi),
            or ``"both"`` to create planes on both faces.
        z_offset_mm: Distance in millimetres to offset inward from the chosen
            board substrate (FR4) face.
        output_prim_path: Optional absolute prim path for the generated quad
            mesh.  ``None`` resolves to
            ``<pcba-root>/SegmentationProxies/HolePlane`` (single side) or
            sibling ``HolePlane_bottom`` / ``HolePlane_top`` Xforms
            (``side="both"``) at runtime.
        components: Optional refdes allowlist.  When set, one small plane is
            created per pad prim belonging to a listed component (sized from
            the pad's world-space geometry) instead of a single board-wide
            plane.  Requires ``MetadataCapability.PADS`` and
            ``MetadataCapability.COMPONENTS``. An empty allowlist or missing
            capabilities emits a warning and creates no planes; explicit
            filters never widen to a board-wide plane.
        component_filter: Optional exact metadata-field filter. Mutually
            exclusive with ``components``.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    class_name: NonEmptyString = "hole"
    side: Literal["top", "bottom", "both"]
    z_offset_mm: float = Field(default=0.1, ge=0.0)
    output_prim_path: NonEmptyString | None = None
    components: tuple[NonEmptyString, ...] | None = None
    component_filter: ComponentFilter | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> HolePlaneParameters:
        """Reject mixing normalized and legacy selectors.

        Returns:
            Validated hole-plane parameters.

        Raises:
            ValueError: If both selector forms are supplied.

        """
        if self.components is not None and self.component_filter is not None:
            raise ValueError("components and component_filter are mutually exclusive")
        return self

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
            raise ValueError("output_prim_path must be an absolute non-root path")
        return value


class StageSerializationParameters(BaseModel):
    """Version-1 parameters for ``stage.serialize``.

    Args:
        serialization_path: Destination for the flattened composed USD stage.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    serialization_path: NonEmptyString

    @field_validator("serialization_path")
    @classmethod
    def validate_serialization_path(cls, value: str) -> str:
        """Require a supported USD destination filename.

        Args:
            value: Requested local path or remote URI.

        Returns:
            Validated serialization destination.

        Raises:
            ValueError: If the destination has no supported USD extension.

        """
        if not value.lower().endswith((".usd", ".usda", ".usdc")):
            raise ValueError(
                "serialization_path must end with .usd, .usda, or .usdc",
            )
        return value


#: Which observable each solder-defect type takes, with its EXCLUSIVE bounds
#: (the boundary values are different classes: a volume_ratio of 0 is
#: missing_solder, 1 is nominal; a gap_fraction of 0 does not interrupt and 1
#: leaves no fillet). Mirrors the declarative surface's ``_observable_spec``
#: so an order arriving over transport is held to the same contract.
_SOLDER_DEFECT_OBSERVABLES: dict[str, dict[str, tuple[float, float]]] = {
    "insufficient": {
        "volume_ratio": (0.0, 1.0),
        "lead_climb": (0.0, 2.0),
        "fullness": (0.0, 2.0),
    },
    "excess": {
        "volume_ratio": (1.0, float("inf")),
        "lead_climb": (0.0, 2.0),
        "fullness": (0.0, 2.0),
    },
    "bright": {"lead_climb": (0.0, 2.0), "fullness": (0.0, 2.0)},
    "interrupted": {
        "pad_exposure": (0.0, 1.0),
        "lead_climb": (0.0, 2.0),
        "fullness": (0.0, 2.0),
    },
    "dry_contact": {
        "gap_fraction": (0.0, 1.0),
        "lead_climb": (0.0, 2.0),
        "fullness": (0.0, 2.0),
    },
    "missing_solder": {},
}


class SolderDefectEntryParameters(BaseModel):
    """One validated solder-defect operation (one defect type).

    Args:
        defect: Solder defect class name.
        ratio: Selected fraction of the (remaining) eligible pool.
        component_types: Optional family allow-list; ``None`` draws from the
            scene's selected component pool.
        volume_ratio: Volume observable (value or ``[min, max]``), only for
            ``insufficient`` (within (0, 1)) or ``excess`` (> 1).
        gap_fraction: Interrupted-arc observable (value or ``[min, max]``
            within (0, 1)), only for ``dry_contact``.
        lead_climb: THT shape (value or ``[min, max]`` within (0, 2)):
            wetted lead height over measured pin protrusion; not for
            ``missing_solder``.
        fullness: Shape (value or ``[min, max]`` within (0, 2)): fillet
            volume over the joint's own straight-cone fillet. Drives THT
            joints and, when given, SMT ones too (an unset SMT joint keeps
            the paste-driven nominal); not for ``missing_solder``.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    defect: Literal[
        "insufficient",
        "excess",
        "bright",
        "interrupted",
        "missing_solder",
        "dry_contact",
    ]
    ratio: float = Field(gt=0.0, le=1.0)
    component_types: tuple[NonEmptyString, ...] | None = None
    component_filter: ComponentFilter | None = None
    volume_ratio: float | tuple[float, float] | None = None
    gap_fraction: float | tuple[float, float] | None = None
    pad_exposure: float | tuple[float, float] | None = None
    lead_climb: float | tuple[float, float] | None = None
    fullness: float | tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_observable(self) -> SolderDefectEntryParameters:
        """Hold each defect type's observable to its exclusive bounds.

        Returns:
            The validated model.

        Raises:
            ValueError: If an observable does not belong to the defect type,
                a range is not ordered, or a value sits on or outside the
                type's exclusive bounds.

        """
        if self.component_types is not None and self.component_filter is not None:
            raise ValueError(
                "component_types and component_filter are mutually exclusive",
            )
        allowed = _SOLDER_DEFECT_OBSERVABLES[self.defect]
        for name in (
            "volume_ratio",
            "gap_fraction",
            "pad_exposure",
            "lead_climb",
            "fullness",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if name not in allowed:
                takes = ", ".join(sorted(allowed)) or "no observable"
                raise ValueError(
                    f"{self.defect} takes {takes}, not {name}",
                )
            lo, hi = allowed[name]
            values = value if isinstance(value, tuple) else (value,)
            if isinstance(value, tuple) and values[0] > values[1]:
                raise ValueError(
                    f"{self.defect}.{name}: min ({values[0]}) must be "
                    f"<= max ({values[1]})",
                )
            for item in values:
                if not lo < float(item) < hi:
                    raise ValueError(
                        f"{self.defect}.{name} must be within ({lo}, {hi}), got {item}",
                    )
        return self


class SolderDefectParameters(BaseModel):
    """Version-1 parameters for ``solder_defects.apply``.

    Args:
        entries: Validated solder-defect entries.
        source: Attached components sidecar name (refdes resolution).

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[SolderDefectEntryParameters, ...] = Field(min_length=1)
    #: Validated solder-defect entries.

    source: NonEmptyString
    #: Attached components sidecar name.


class CaptureOutputParameters(BaseModel):
    """Version-1 parameters for ``capture.output.configure``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)


class RendererParameters(BaseModel):
    """Version-1 parameters for ``capture.renderer.configure``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["pathtracing", "realtime"]
    resolution: tuple[int, int]
    spp: int = Field(ge=1)
    total_spp: int = Field(ge=1)
    max_bounces: int = Field(ge=0)
    #: rt_subframes for "realtime"; path tracing ignores it. Defaulted so a
    #: plan encoded before this field existed still decodes.
    realtime_subframes: int = Field(default=1, ge=1)

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, resolution: tuple[int, int]) -> tuple[int, int]:
        """Require positive image dimensions.

        Args:
            resolution: Requested image width and height.

        Returns:
            Validated resolution.

        Raises:
            ValueError: If either image dimension is not positive.

        """
        if min(resolution) < 1:
            raise ValueError("resolution dimensions must be positive")
        return resolution


class CameraParameters(BaseModel):
    """Version-1 parameters for ``capture.camera.configure``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    side: Literal["top", "bottom"] = "top"
    rotation_range: Mapping[str, object]

    @field_validator("rotation_range")
    @classmethod
    def validate_axes(
        cls, rotation_range: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Reject unsupported camera-rotation axes.

        Args:
            rotation_range: Axis names and fixed or ranged rotations.

        Returns:
            Validated rotation mapping.

        Raises:
            ValueError: If an unsupported axis is present.

        """
        unknown = set(rotation_range) - {"x", "y", "z"}
        if unknown:
            raise ValueError(f"unsupported rotation axes: {sorted(unknown)}")
        return rotation_range


class ScanGridParameters(BaseModel):
    """Version-1 parameters for ``capture.scan_grid.configure``.

    Args:
        x_count: Cells across world X.
        y_count: Cells across world Y.
        scan_area_mm: World rectangle the grid spans, ``(length_x, breadth_y)``
            in millimetres, centred on the board. ``None`` spans the whole
            board, which is what every capture did before a field of view
            could be stated. Millimetres rather than an aperture because only
            the stage knows its own scale.
        centered_on: Opaque label naming what ``scan_area_mm`` is centred on.
            The vocabulary belongs to the domain -- ``pcb-aoi`` uses ``panel``
            and ``instance`` -- and is carried, never interpreted: the runtime
            centres on the bbox midpoint either way and reports which reading
            was meant, so a consumer of a stitched panel image is not left to
            guess which placement it is looking at.
        scan_center_mm: World ``(x, y)`` in millimetres the rectangle is
            centered on instead of the board bbox midpoint, for a field of view
            placed by corners. Requires ``scan_area_mm``.
        image_size_px: ``(width, height)`` in image axes the stitched canvas is
            cropped to, centred. Cells are whole pixels, so the mosaic can be
            larger than the size the capture states; ``None`` keeps it whole.
        guard_px: Pixels each cell renders beyond every edge and crops from
            every per-tile output before it is written. The aperture grows by
            the same number of pixel pitches, so cell centres, tiles, pixel
            layout and calibration are those of the delivered tile. Requires
            ``scan_area_mm``; ``0`` renders the tile as-is.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x_count: int = Field(ge=1)
    y_count: int = Field(ge=1)
    scan_area_mm: tuple[float, float] | None = None
    centered_on: str | None = None
    scan_center_mm: tuple[float, float] | None = None
    image_size_px: tuple[PositiveInt, PositiveInt] | None = None
    guard_px: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_scan_center(self):
        """A center places a stated rectangle; without one it has nothing to place."""
        if self.scan_center_mm is not None and self.scan_area_mm is None:
            raise ValueError("scan_center_mm places scan_area_mm, which is not set")
        if self.guard_px and self.scan_area_mm is None:
            raise ValueError(
                "guard_px widens the scan_area_mm aperture, which is not set"
            )
        if self.scan_center_mm is not None and self.centered_on is not None:
            raise ValueError(
                "centered_on labels a rectangle centered on the board; "
                "scan_center_mm places it explicitly -- give one"
            )
        return self


class RoiSelectorParameters(BaseModel):
    """One component selector within an ROI-selection declaration.

    Args:
        refdes: Ordered explicit component reference designators.
        pins: Ordered explicit ``(refdes, pin_number)`` identities.
        component_types: Metadata component types eligible for selection.
        process_classes: Manufacturing process classes eligible for selection.
        component_filter: Exact metadata-field filter eligible for selection.
        all_components: Whether every cataloged component is eligible.
        defects: Applied defect types eligible for selection.
        focus: Whether to frame each component or each cataloged pin.
        count: Deterministic subset size for type, process, or defect selection.
        padding: Additional fractional framing around each target.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    refdes: tuple[NonEmptyString, ...] | None = None
    pins: tuple[tuple[NonEmptyString, NonEmptyString], ...] | None = None
    component_types: tuple[NonEmptyString, ...] | None = None
    process_classes: tuple[NonEmptyString, ...] | None = None
    component_filter: ComponentFilter | None = None
    all_components: bool = False
    defects: (
        tuple[
            Literal["shift", "tombstone", "sideflip", "reverse_polarity", "missing"],
            ...,
        ]
        | None
    ) = None
    focus: Literal["component", "pin"] | None = None
    count: int | None = Field(default=None, ge=1)
    padding: float = Field(default=0.5, ge=0.0)

    @model_validator(mode="after")
    def validate_selector(self):
        """Require one nonempty selector and compatible count semantics."""
        values = (
            self.refdes,
            self.pins,
            self.component_types,
            self.process_classes,
            self.component_filter,
            True if self.all_components else None,
            self.defects,
        )
        if sum(value is not None for value in values) != 1:
            raise ValueError(
                "ROI selector requires exactly one of refdes, pins, "
                "component_types, process_classes, component_filter, "
                "all_components, or defects",
            )
        selected = next(value for value in values if value is not None)
        if not isinstance(selected, ComponentFilter):
            if selected is not True and not selected:
                raise ValueError("ROI selector values cannot be empty")
            if selected is not True and len(set(selected)) != len(selected):
                raise ValueError("ROI selector values must be unique")
        if (
            self.refdes is not None or self.pins is not None or self.all_components
        ) and self.count is not None:
            raise ValueError(
                "ROI selector count is not valid with explicit refdes, pins, "
                "or all_components"
            )
        if self.pins is not None and self.focus == "component":
            raise ValueError("ROI selector exact pins cannot use component focus")
        return self


class RoiSelectionParameters(BaseModel):
    """Version-1 parameters for ``capture.roi_selection.configure``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selections: tuple[RoiSelectorParameters, ...] = Field(min_length=1)


class CaptureLimitParameters(BaseModel):
    """Version-1 parameters for ``capture.limits.configure``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    triggers: int = Field(ge=1)
    max_images: int = Field(ge=-1)


class CaptureVisibilityParameters(BaseModel):
    """Version-1 parameters for ``capture.visibility.configure``.

    Accepted ``hide`` values:

    - ``"solder"`` — hides procedurally generated solder-joint prims via the
      placement manifest.  Requires a ``simulated`` solder-joints operation to
      have completed before this pass.
    - Board geometry role shortnames (``pcba:mesh:role`` without ``board_``
      prefix): ``"copper"``, ``"mask"``, ``"pad"``, ``"paste"``, ``"hole"``,
      ``"barrel"``, ``"substrate"``, ``"pin_tip"``.  Every prim with a matching
      role attribute under the board root is hidden for the pass duration and
      restored after.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hide: tuple[
        Literal[
            "solder",
            "copper",
            "mask",
            "pad",
            "paste",
            "hole",
            "barrel",
            "substrate",
            "pin_tip",
        ],
        ...,
    ] = ()
    show: tuple[
        Literal[
            "copper",
            "mask",
            "pad",
            "paste",
            "hole",
            "barrel",
            "substrate",
            "pin_tip",
        ],
        ...,
    ] = ()

    @model_validator(mode="after")
    def validate_targets_are_disjoint(self):
        """Reject geometry requested as both shown and hidden."""
        overlap = sorted(set(self.hide) & set(self.show))
        if overlap:
            raise ValueError(
                f"visibility cannot show and hide the same target(s): {overlap}"
            )
        return self


class CaptureWriterParameters(BaseModel):
    """Version-1 parameters for ``capture.outputs.configure``."""

    model_config = ConfigDict(frozen=True, extra="allow")

    rgb: bool = True
    image_format: ImageFormat = "png"
    bounding_box_2d_tight: bool = False
    bounding_box_2d_loose: bool = False
    bounding_box_3d: bool = False
    semantic_segmentation: bool = False
    instance_id_segmentation: bool = False
    colorize_semantic_segmentation: bool = False
    colorize_instance_id_segmentation: bool = False
    semantic_types: tuple[NonEmptyString, ...] | None = None
    frame_padding: int = Field(default=4, ge=1)
    rename_to_grid_index: bool = False


class AugmentationParameters(BaseModel):
    """Version-1 parameters for ``capture.augmentation.configure``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    motion_blur: Mapping[str, object] | None = None


class CaptureRenderParameters(BaseModel):
    """Version-1 parameters for ``capture.render``.

    Args:
        output_dir: Directory that receives capture artifacts.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    output_dir: NonEmptyString


class PostprocessCropParameters(BaseModel):
    """Version-1 parameters for ``postprocess.crop``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: NonEmptyString
    types: tuple[NonEmptyString, ...]
    offset: int = Field(ge=0)
    min_coverage: float = Field(ge=0.0, le=1.0)
    class_filter: tuple[NonEmptyString, ...] | None = None
    xform_depth: int = Field(ge=0)
    output_subdir: NonEmptyString
    mode: str | None = None
    input_subdir: str | None = None
    reference_subdir: str | None = None

    @model_validator(mode="after")
    def validate_mode_inputs(self) -> PostprocessCropParameters:
        """Require paired input directories when a crop mode is selected.

        Returns:
            Validated crop parameters.

        Raises:
            ValueError: If a mode omits either required input directory.

        """
        if self.mode is not None and not (self.input_subdir and self.reference_subdir):
            raise ValueError(
                "mode requires input_subdir and reference_subdir",
            )
        return self


class PostprocessCloseOpeningsParameters(BaseModel):
    """Version-1 parameters for ``postprocess.close_openings``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_pass: NonEmptyString = "default"
    class_name: NonEmptyString
    output_subdir: NonEmptyString = "stitched"
    output_filename: NonEmptyString = "rgb.png"
    max_diameter_px: int = Field(default=48, ge=1)
    min_circularity: float = Field(default=0.55, gt=0.0, le=1.0)


class PostprocessStitchParameters(BaseModel):
    """Version-1 parameters for ``postprocess.stitch``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_pass: NonEmptyString = "default"
    image_type: NonEmptyString = "rgb"
    output_subdir: NonEmptyString = "stitched"
    output_filename: NonEmptyString = "rgb.png"
    blend: Literal["none", "feather", "labels"] = "none"
    write_labels: bool = False
    recolor_tiles: bool = False
    color_map: dict[str, tuple[int, int, int, int]] | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> PostprocessStitchParameters:
        """Require simple relative output names.

        Returns:
            Validated stitch parameters.

        Raises:
            ValueError: If output names contain path traversal or no extension.

        """
        if PurePath(self.output_subdir).name != self.output_subdir:
            raise ValueError("output_subdir must be a single directory name")
        output_path = PurePath(self.output_filename)
        if output_path.name != self.output_filename or not output_path.suffix:
            raise ValueError("output_filename must be a filename with an extension")
        if self.recolor_tiles and self.blend != "labels":
            raise ValueError("recolor_tiles requires blend='labels'")
        return self


class PostprocessFormatRoiOutputParameters(BaseModel):
    """Version-1 parameters for ``postprocess.format_roi_output``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rgb_pass: NonEmptyString = "full_board"
    roi_pass: NonEmptyString = "full_board_roi"


class PostprocessGenerateJointRoisParameters(BaseModel):
    """Version-1 parameters for ``postprocess.generate_joint_rois``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    roi_pass: NonEmptyString = "full_board_roi"
    rgb_pass: NonEmptyString = "full_board"
