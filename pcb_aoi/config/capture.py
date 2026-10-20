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

"""Capture configuration models and Python API interpretation."""

from typing import Literal

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from cdf.component_filter import ComponentFilter
from cdf.config._common import ConfigModel, Number, RangeValue, expand_range
from cdf.image_formats import ImageFormat
from cdf.job import CDFJob
from pcb_aoi.geometry.fov import DEFAULT_GUARD_PX, FovError, isotropic_resolution

KNOWN_ANNOTATORS = frozenset(
    {
        "rgb",
        "bounding_box_2d_tight",
        "bounding_box_2d_loose",
        "bounding_box_3d",
        "semantic_segmentation",
        "instance_id_segmentation",
    }
)

RENDER_PASS_API_CONFIG_FIELDS = {
    "output_start": "output_start",
    "renderer": "renderer",
    "camera": "camera",
    "scan_grid": "scan_grid",
    "fov": "fov",
    "visibility": "visibility",
    "roi_selection": "roi_selection",
    "limits": "limits",
    "augmentation": "motion_blur",
    "outputs": "outputs",
}

CAPTURE_API_CONFIG_FIELDS = {
    **RENDER_PASS_API_CONFIG_FIELDS,
    "render_pass": "passes",
}


class RendererConfig(ConfigModel):
    """Render resolution and sampling-quality settings.

    ``"realtime"`` renders about 1.47x faster on a scan grid and is a
    measurably different picture -- 37.8 dB against a 128-sample reference
    where path tracing is 46.7 dB -- so the two are not interchangeable within
    one dataset.
    """

    mode: Literal["pathtracing", "realtime"] = "pathtracing"
    #: Optional when the pass carries a ``fov`` section, which derives it.
    resolution: tuple[int, int] | None = None
    spp: int = 1
    total_spp: int = 32
    max_bounces: int = 4
    #: ``rt_subframes`` for ``"realtime"``; path tracing ignores it.
    realtime_subframes: int = Field(default=1, ge=1)


class RotationConfig(ConfigModel):
    """Camera X/Y rotation ranges and fixed Z-rotation settings."""

    x: RangeValue | None = None
    y: RangeValue | None = None
    z: Number | None = None


class CameraConfig(ConfigModel):
    """Board-side and camera-orientation settings."""

    side: Literal["top", "bottom"] = "top"
    rotation: RotationConfig


class ScanGridConfig(ConfigModel):
    """The object form of ``scan_grid``; ``[x_count, y_count]`` stays valid.

    Mirrors ``capture.scan_grid(...)`` so a YAML job can state everything the
    Python API can: the cell counts, the world rectangle the grid spans, its
    label or its explicit center, and the size the stitched canvas is cropped to.
    """

    x_count: PositiveInt
    y_count: PositiveInt
    scan_area_mm: tuple[PositiveFloat, PositiveFloat] | None = None
    centered_on: str | None = None
    scan_center_mm: tuple[float, float] | None = None
    image_size_px: tuple[PositiveInt, PositiveInt] | None = None
    #: Pixels each cell renders beyond every edge and crops before the tile is
    #: written. Off by default here: a hand-authored grid keeps the frames it
    #: always rendered. ``fov`` turns it on.
    guard_px: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_scan_center(self):
        """Refuse at parse time what ``capture.scan_grid`` refuses at declaration."""
        if self.scan_center_mm is not None and self.scan_area_mm is None:
            raise ValueError(
                "scan_grid.scan_center_mm places scan_area_mm, which is not set"
            )
        if self.guard_px and self.scan_area_mm is None:
            raise ValueError(
                "scan_grid.guard_px widens the scan_area_mm aperture, which is not set"
            )
        if self.scan_center_mm is not None and self.centered_on is not None:
            raise ValueError(
                "scan_grid.centered_on labels a rectangle centered on the board; "
                "scan_center_mm places it explicitly -- give one"
            )
        return self


class LimitsConfig(ConfigModel):
    """Limits on capture triggers and generated images.

    ``max_images`` counts capture positions (image sets) across all triggers;
    ``-1`` means unlimited.  In a ``missing`` pipeline each position renders
    one aligned ``reference``/``defective`` pair, which counts once.
    """

    triggers: int = 1
    max_images: int = -1


class VisibilityConfig(ConfigModel):
    """Geometry shown or hidden only while one render pass executes.

    Accepted values for ``hide``:

    - ``"solder"`` — hides procedurally generated solder-joint prims (requires
      a ``solder_joints`` operation with a ``simulated`` solver).
    - Board geometry role shortnames — ``pcba:mesh:role`` without the
      ``board_`` prefix: ``"copper"``, ``"mask"``, ``"pad"``, ``"paste"``,
      ``"hole"``, ``"barrel"``, ``"substrate"``, ``"pin_tip"``.  All matching
      prims under the board root are hidden for this pass duration and restored
      after.

    ``show`` accepts board geometry role shortnames. Generated ``hole`` and
    ``pin_tip`` annotation proxies are hidden by default and must be shown in
    the annotation pass that consumes them.
    """

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


class RoiSelectorConfig(ConfigModel):
    """One metadata or applied-defect selector for focused capture."""

    refdes: tuple[str, ...] | None = None
    pins: tuple[tuple[str, str], ...] | None = None
    component_types: tuple[str, ...] | None = None
    process_classes: tuple[str, ...] | None = None
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
                "roi_selection requires exactly one of refdes, pins, "
                "component_types, process_classes, component_filter, "
                "all_components, or defects",
            )
        selected = next(value for value in values if value is not None)
        if not isinstance(selected, ComponentFilter):
            if selected is not True and not selected:
                raise ValueError("roi_selection selector values cannot be empty")
            if selected is not True and len(set(selected)) != len(selected):
                raise ValueError("roi_selection selector values must be unique")
        if (
            self.refdes is not None or self.pins is not None or self.all_components
        ) and self.count is not None:
            raise ValueError(
                "roi_selection count is not valid with explicit refdes, pins, "
                "or all_components"
            )
        if self.pins is not None and self.focus == "component":
            raise ValueError("roi_selection exact pins cannot use component focus")
        return self


class OutputsConfig(ConfigModel):
    """Capture annotator, encoding, semantic, and filename settings."""

    annotators: tuple[str, ...] = ("rgb",)
    #: ``bmp`` is produced by re-encoding the rgb frames after capture; the
    #: writer itself only encodes png and jpeg. See :mod:`cdf.image_formats`.
    image_format: ImageFormat = "png"
    semantic_types: tuple[str, ...] | None = None
    colorize: tuple[
        Literal["semantic_segmentation", "instance_id_segmentation"], ...
    ] = ()
    frame_padding: int = 4
    rename_to_grid_index: bool = False


class FovConfig(ConfigModel):
    """A capture stated the way a scanner is stated: millimetres and dpi.

    An AOI test program is written as "scan this rectangle of the board at this
    resolution", never as an aperture in scene units. Supplying this section
    derives ``renderer.resolution`` and ``scan_grid`` from it, so the pixel count
    is computed once, here, rather than independently on each side of the
    interface and left to drift.

    Give the area as ``scan_area_mm`` (a size, centered on the board) or as
    ``corners_mm`` (two opposite corners in world millimetres, which also place
    it), with exactly one of ``dpi``, ``mm_per_px`` or ``image_size_px`` --
    area, resolution and pixel count are three names for two quantities, and
    dpi and pixel pitch are one quantity written two ways.
    """

    scan_area_mm: tuple[PositiveFloat, PositiveFloat] | None = None
    #: Lower-left and upper-right corners of the field of view in world
    #: millimetres, ``[x, y]`` or ``[x, y, z]`` each (``z`` is ignored) -- the
    #: way a machine test program states it. Size and placement both follow.
    corners_mm: (
        tuple[
            tuple[float, float] | tuple[float, float, float],
            tuple[float, float] | tuple[float, float, float],
        ]
        | None
    ) = None
    #: A number, or a pair that must be equal: pixels are square, because an
    #: orthographic render cannot deliver two pitches.
    dpi: PositiveFloat | tuple[PositiveFloat, PositiveFloat] | None = None
    #: The sensor's x,y resolution as a pitch, the way a machine spec sheet
    #: states it: 0.042333 mm/px is 600 dpi, 0.021167 mm/px is 1200 dpi. A pair
    #: must be equal, as for ``dpi``.
    mm_per_px: PositiveFloat | tuple[PositiveFloat, PositiveFloat] | None = None
    #: Kept exactly; delivered at one pitch, the coarser of area / pixels on
    #: the two axes, so a canvas without the area's aspect covers more than
    #: the area on the other axis, centred on it.
    image_size_px: tuple[PositiveInt, PositiveInt] | None = None
    #: Largest single rendered frame, guard band included; a bigger canvas is
    #: tiled into cells.
    max_tile_px: tuple[PositiveInt, PositiveInt] = (4096, 4096)
    #: Pixels each cell renders beyond every edge, cropped from every per-tile
    #: output before it is written: the renderer's outermost row and column are
    #: unreliable and the cells abut, so without it every seam shows them. Tiles,
    #: pixel layout and calibration are the same shape at any value; ``0``
    #: renders the delivered tile directly.
    guard_px: NonNegativeInt = DEFAULT_GUARD_PX
    #: What ``scan_area_mm`` is centred on. Carried to the sidecar; never
    #: inferred, because for a multi-up panel "the board" is ambiguous and a
    #: silent guess in a coordinate contract is expensive.
    centered_on: Literal["panel", "instance", "unspecified"] = "unspecified"

    @model_validator(mode="after")
    def validate_fov_inputs(self):
        """Refuse an over- or under-determined capture at parse time.

        Area, resolution and pixel count are three names for two quantities.
        Checking it here rather than where the arithmetic runs keeps invalid
        declarative input inside the normal validation boundary -- and makes
        the generated schema describe the contract that is actually enforced.
        """
        if (self.scan_area_mm is None) == (self.corners_mm is None):
            raise ValueError(
                "fov requires exactly one of scan_area_mm or corners_mm; got %s"
                % ("both" if self.corners_mm is not None else "neither")
            )
        if self.corners_mm is not None:
            (x0, y0), (x1, y1) = (corner[:2] for corner in self.corners_mm)
            if x0 == x1 or y0 == y1:
                raise ValueError("fov.corners_mm must span a positive area")
            if self.centered_on != "unspecified":
                raise ValueError(
                    "fov.centered_on describes a rectangle centered on the board; "
                    "corners_mm place it themselves"
                )
        stated = [
            name
            for name in ("dpi", "mm_per_px", "image_size_px")
            if getattr(self, name) is not None
        ]
        if len(stated) != 1:
            raise ValueError(
                "fov requires exactly one of dpi, mm_per_px or image_size_px "
                "alongside scan_area_mm (they are not independent); got %s"
                % (", ".join(stated) or "none")
            )
        # Square pixels only: the renderer cannot produce two pitches. The same
        # check, and so the same message, as the Python API's resolve_fov.
        for name in ("dpi", "mm_per_px"):
            value = getattr(self, name)
            if value is not None:
                try:
                    isotropic_resolution(
                        list(value) if isinstance(value, tuple) else value, name
                    )
                except FovError as exc:
                    raise ValueError(f"fov.{exc}") from exc
        return self


class RenderPassConfig(ConfigModel):
    """Complete configuration for one named render pass."""

    output_start: int = 0
    renderer: RendererConfig
    camera: CameraConfig
    scan_grid: tuple[int, int] | ScanGridConfig | None = None
    fov: FovConfig | None = None
    visibility: VisibilityConfig = Field(default_factory=VisibilityConfig)
    roi_selection: tuple[RoiSelectorConfig, ...] | None = Field(
        default=None,
        min_length=1,
    )
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    motion_blur: dict[str, object] | None = None
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)


class CaptureConfig(RenderPassConfig):
    """Legacy default-pass settings and optional named render passes."""

    renderer: RendererConfig | None = None
    camera: CameraConfig | None = None
    scan_grid: tuple[int, int] | ScanGridConfig | None = None
    fov: FovConfig | None = None
    passes: dict[str, RenderPassConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_capture_modes(self):
        """Require a complete default pass or at least one named pass."""
        legacy_values = (self.renderer, self.camera, self.scan_grid or self.fov)
        if any(value is not None for value in legacy_values) and not all(
            value is not None for value in legacy_values
        ):
            raise ValueError(
                "capture.renderer, capture.camera, and capture.scan_grid must "
                "be supplied together"
            )
        if not any(value is not None for value in legacy_values) and not self.passes:
            raise ValueError("capture requires a default configuration or named passes")
        _require_resolution_or_fov(self)
        for name, section in self.passes.items():
            try:
                _require_resolution_or_fov(section)
            except ValueError as exc:
                raise ValueError(f"capture.passes.{name}: {exc}") from exc
        return self


def _require_resolution_or_fov(section) -> None:
    """Require each pass to specify its frame size in exactly one way.

    ``renderer.resolution`` states it in pixels; ``fov`` states it in
    millimetres and dpi and derives the pixels. Exactly one is needed -- and
    supplying both is refused rather than silently letting one win, because
    which one won would only be visible in the output size.
    """
    renderer = getattr(section, "renderer", None)
    if renderer is None:
        return
    has_res = getattr(renderer, "resolution", None) is not None
    has_fov = getattr(section, "fov", None) is not None
    if has_res and has_fov:
        raise ValueError(
            "give renderer.resolution or fov, not both -- fov derives the resolution"
        )
    if not has_res and not has_fov:
        raise ValueError(
            "renderer.resolution is required unless a fov section derives it"
        )
    # A fov derives the cell counts too, so an authored scan_grid beside one is
    # not merely redundant: it is a second answer to a question that already
    # has one, and the derived answer wins. That is invisible in the output
    # except as a frame count, so refuse it rather than drop it.
    if has_fov and getattr(section, "scan_grid", None) is not None:
        raise ValueError(
            "give scan_grid or fov, not both -- fov derives the cell counts"
        )


def _pass_fov_section(section, config):
    """Return the section's ``fov`` entry point, or refuse the declaration.

    A field of view is domain vocabulary -- how a capture is stated in the
    terms the domain's own instrument is configured in -- so the method exists
    only on a domain capture section. Configuration reaches it the way it
    reaches every other domain behaviour: through the section the domain
    contributed, never by importing the domain package.
    """
    if config.fov is None:
        return None
    entry_point = getattr(section, "fov", None)
    if entry_point is None:
        raise ValueError(
            "capture.fov is not supported by this job's domain; its capture "
            "section does not define fov()"
        )
    return entry_point


def _apply_render_pass(section, config: RenderPassConfig | CaptureConfig) -> None:
    """Apply one validated render-pass configuration."""
    assert config.renderer is not None
    assert config.camera is not None
    fov_entry_point = _pass_fov_section(section, config)
    renderer = {k: v for k, v in config.renderer.model_dump().items() if v is not None}
    section.output_start(config.output_start)
    # No resolution when a fov states the frame size: the section derives it,
    # and the two calls commute, so the renderer may be configured first and
    # keep whatever the derivation puts there.
    section.renderer(**renderer)
    rotation_spec = config.camera.rotation
    rotation = {}
    if rotation_spec.x is not None:
        rotation["x"] = expand_range(rotation_spec.x)
    if rotation_spec.y is not None:
        rotation["y"] = expand_range(rotation_spec.y)
    if rotation_spec.z is not None:
        rotation["z"] = rotation_spec.z
    section.camera(side=config.camera.side, rotation_range=rotation)
    if fov_entry_point is not None:
        # Go through the public API, not just its arithmetic: the section is
        # where the derived resolution, the cell counts and the scan rectangle
        # all reach the job together. Deriving them here instead once left the
        # rectangle behind -- the YAML path honoured the resolution and the
        # cell counts but not the millimetres, so a capture asking for
        # 2836x3543 rendered 3010x2362.
        fov_entry_point(
            scan_area_mm=(
                list(config.fov.scan_area_mm)
                if config.fov.scan_area_mm is not None
                else None
            ),
            corners_mm=(
                [list(corner) for corner in config.fov.corners_mm]
                if config.fov.corners_mm is not None
                else None
            ),
            dpi=list(config.fov.dpi)
            if isinstance(config.fov.dpi, tuple)
            else config.fov.dpi,
            mm_per_px=list(config.fov.mm_per_px)
            if isinstance(config.fov.mm_per_px, tuple)
            else config.fov.mm_per_px,
            image_size_px=(
                list(config.fov.image_size_px) if config.fov.image_size_px else None
            ),
            max_tile_px=list(config.fov.max_tile_px),
            centered_on=config.fov.centered_on,
            guard_px=config.fov.guard_px,
        )
    else:
        assert config.scan_grid is not None
        if isinstance(config.scan_grid, ScanGridConfig):
            grid = config.scan_grid
            section.scan_grid(
                x_count=grid.x_count,
                y_count=grid.y_count,
                scan_area_mm=list(grid.scan_area_mm) if grid.scan_area_mm else None,
                centered_on=grid.centered_on,
                scan_center_mm=list(grid.scan_center_mm)
                if grid.scan_center_mm
                else None,
                image_size_px=list(grid.image_size_px) if grid.image_size_px else None,
                guard_px=grid.guard_px,
            )
        else:
            section.scan_grid(
                x_count=config.scan_grid[0],
                y_count=config.scan_grid[1],
            )
    if config.visibility.hide or config.visibility.show:
        section.visibility(
            hide=config.visibility.hide,
            show=config.visibility.show,
        )
    for selector in config.roi_selection or ():
        section.roi_selection(**selector.model_dump())
    section.limits(**config.limits.model_dump())
    if config.motion_blur is not None:
        section.augmentation(motion_blur=config.motion_blur)

    requested = set(config.outputs.annotators)
    parameters: dict[str, object] = {
        name: name in requested for name in KNOWN_ANNOTATORS
    }
    parameters.update({name: True for name in requested - KNOWN_ANNOTATORS})
    parameters.update(
        image_format=config.outputs.image_format,
        semantic_types=config.outputs.semantic_types,
        frame_padding=config.outputs.frame_padding,
        rename_to_grid_index=config.outputs.rename_to_grid_index,
        colorize_semantic_segmentation=(
            "semantic_segmentation" in config.outputs.colorize
        ),
        colorize_instance_id_segmentation=(
            "instance_id_segmentation" in config.outputs.colorize
        ),
    )
    section.outputs(**parameters)


def apply_capture(job: CDFJob, config: CaptureConfig) -> None:
    """Apply default and named render-pass declarations."""
    if config.renderer is not None:
        _apply_render_pass(job.capture, config)
    for name, pass_config in config.passes.items():
        with job.capture.render_pass(name) as render_pass:
            _apply_render_pass(render_pass, pass_config)
