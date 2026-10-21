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

"""Appearance configuration models and Python API interpretation."""

from pydantic import Field, PositiveFloat, model_validator

from cdf.config._common import ConfigModel, Number, RangeValue, expand_range
from cdf.job import CDFJob

#: Modus rig light groups. Mirrors ``cdf.sections._MODUS_GROUPS`` -- kept as
#: a separate constant because config models must not import the section
#: layer that applies them.
_MODUS_GROUPS = frozenset({"white", "red", "blue"})

APPEARANCE_API_CONFIG_FIELDS = {
    "components": "vantablack_components",
    "lighting": "lighting",
    "material_override": "materials",
}


class LightLayerConfig(ConfigModel):
    """Intensity, color, and visibility settings for one light layer."""

    intensity: RangeValue
    color: tuple[Number, Number, Number] | None = None
    color_r: RangeValue | None = None
    color_g: RangeValue | None = None
    color_b: RangeValue | None = None
    visible: bool = False

    @model_validator(mode="after")
    def validate_color_forms(self):
        """Reject ambiguous fixed-color and channel-range declarations."""
        channels = (self.color_r, self.color_g, self.color_b)
        if self.color is not None and any(value is not None for value in channels):
            raise ValueError(
                "light layer color cannot be combined with color_r, color_g, "
                "or color_b",
            )
        return self


def _validate_modus_groups(name: str, value: dict[str, object] | None) -> None:
    """Reject a Modus group mapping keyed by anything but white/red/blue."""
    if value is None:
        return
    unknown = sorted(set(value) - _MODUS_GROUPS)
    if unknown:
        raise ValueError(
            f"modus.{name}: unknown group(s) {unknown} (want white/red/blue)"
        )


class ModusLightingConfig(ConfigModel):
    """Knobs for the Modus rig -- a 2D-dome linear AOI illuminator.

    Six ``UsdLux.RectLight`` objects on an arc, in white/red/blue pairs working
    centre-out. Every field is optional; the pipeline builder supplies
    defaults for whatever is left unset.
    """

    arc_radius_scale: PositiveFloat | None = None
    width: PositiveFloat | None = None
    width_frac: PositiveFloat | None = None
    height_scale: PositiveFloat | None = None
    #: Elevation angle (degrees) per group.
    elevations: dict[str, Number] | None = None
    #: RectLight intensity per group.
    intensities: dict[str, Number] | None = None
    #: RectLight colour per group.
    colors: dict[str, tuple[Number, Number, Number]] | None = None
    #: Optional per-trigger randomization, same schema as the ring's
    #: ``lighting.layers``. A group absent here keeps its fixed
    #: ``intensities``/``colors`` value, re-stamped only by ``exposure``.
    layers: dict[str, LightLayerConfig] | None = None

    @model_validator(mode="after")
    def validate_width_forms(self):
        """Reject stating the rect width both as an absolute and a fraction."""
        if self.width is not None and self.width_frac is not None:
            raise ValueError(
                "modus.width cannot be combined with modus.width_frac -- "
                "give one or the other"
            )
        return self

    @model_validator(mode="after")
    def validate_group_keys(self):
        """Reject a group mapping keyed by anything but white/red/blue."""
        _validate_modus_groups("elevations", self.elevations)
        _validate_modus_groups("intensities", self.intensities)
        _validate_modus_groups("colors", self.colors)
        _validate_modus_groups("layers", self.layers)
        return self


class LightingConfig(ConfigModel):
    """Settings for the lighting rig, selected by ``mode``.

    ``mode`` names the rig and decides which payload applies: ``rgb_light``
    takes ``layers``, ``white_light`` and ``dome_light`` take ``dome``,
    ``modus_light`` takes ``modus``, and ``usd_light`` takes neither (the
    board USD's own lights are reused). ``dome_fill`` is accepted with the
    RGB ring and batch Modus rigs.
    """

    mode: str
    preserve_scene_light_color: bool = False
    exposure: RangeValue | None = None
    cone_angle: RangeValue | None = None
    cone_softness: RangeValue | None = None
    dome_fill: LightLayerConfig | None = None
    layers: dict[str, LightLayerConfig] | None = None
    dome: LightLayerConfig | None = None
    modus: ModusLightingConfig | None = None


class MaterialOverrideConfig(ConfigModel):
    """OmniPBR override for one existing bound USD material."""

    material_path: str
    diffuse_texture: str | None = None
    roughness_texture: str | None = None
    normal_texture: str | None = None
    diffuse_color: tuple[Number, Number, Number] = (0.2, 0.2, 0.2)
    diffuse_tint: tuple[Number, Number, Number] = (1.0, 1.0, 1.0)
    albedo_brightness: Number = 1.0
    opacity: Number = Field(default=1.0, ge=0.0, le=1.0)
    roughness: Number = 0.5
    roughness_texture_influence: Number = 1.0
    metallic: Number = 0.0
    specular_level: Number = 0.5
    normal_strength: Number = 1.0
    normal_format: str = "opengl"
    texture_scale: tuple[Number, Number] = (1.0, 1.0)
    texture_rotation: Number = 0.0
    project_uvw: bool = False
    world_space_projection: bool = False
    variation: dict[str, tuple[Number, Number]] | None = None
    #: Per-job randomisation ranges keyed by control name.


class AppearanceConfig(ConfigModel):
    """Optional lighting and component-material appearance settings."""

    lighting: LightingConfig | None = None
    materials: dict[str, MaterialOverrideConfig] | None = None
    vantablack_components: bool | None = None


def _light_layer(config: LightLayerConfig) -> dict[str, object]:
    layer: dict[str, object] = {"intensity": expand_range(config.intensity)}
    if config.visible:
        layer["visible"] = True
    if config.color is not None:
        red, green, blue = config.color
        layer.update(
            color_r=(red, red),
            color_g=(green, green),
            color_b=(blue, blue),
        )
    else:
        for name in ("color_r", "color_g", "color_b"):
            value = getattr(config, name)
            if value is not None:
                layer[name] = expand_range(value)
    return layer


def _modus_config(config: ModusLightingConfig) -> dict[str, object]:
    """Convert validated Modus settings into a plain-dictionary payload.

    The payload is the form
    :meth:`~cdf.sections.CapturePassSection.lighting` (and, underneath it,
    ``cdf.sections._modus_block``) accepts -- the same conversion
    :func:`_light_layer` does for the ring's layers, so the Python API stays
    the single place that payload is interpreted.
    """
    modus: dict[str, object] = {}
    for name in ("arc_radius_scale", "width", "width_frac", "height_scale"):
        value = getattr(config, name)
        if value is not None:
            modus[name] = value
    if config.elevations is not None:
        modus["elevations"] = dict(config.elevations)
    if config.intensities is not None:
        modus["intensities"] = dict(config.intensities)
    if config.colors is not None:
        modus["colors"] = {group: list(color) for group, color in config.colors.items()}
    if config.layers is not None:
        modus["layers"] = {
            group: _light_layer(layer) for group, layer in config.layers.items()
        }
    return modus


def apply_appearance(job: CDFJob, config: AppearanceConfig) -> None:
    """Apply appearance declarations."""
    if config.lighting is not None:
        lighting = config.lighting
        job.appearance.lighting(
            mode=lighting.mode,
            preserve_scene_light_color=lighting.preserve_scene_light_color,
            exposure_range=(
                expand_range(lighting.exposure)
                if lighting.exposure is not None
                else None
            ),
            cone_angle_range=(
                expand_range(lighting.cone_angle)
                if lighting.cone_angle is not None
                else None
            ),
            cone_softness_range=(
                expand_range(lighting.cone_softness)
                if lighting.cone_softness is not None
                else None
            ),
            dome_fill=(
                _light_layer(lighting.dome_fill)
                if lighting.dome_fill is not None
                else None
            ),
            layers=(
                {name: _light_layer(layer) for name, layer in lighting.layers.items()}
                if lighting.layers is not None
                else None
            ),
            dome=(_light_layer(lighting.dome) if lighting.dome is not None else None),
            modus=(
                _modus_config(lighting.modus) if lighting.modus is not None else None
            ),
        )
    for name, material in (config.materials or {}).items():
        job.appearance.material_override(
            name,
            material_path=material.material_path,
            diffuse_texture=material.diffuse_texture,
            roughness_texture=material.roughness_texture,
            normal_texture=material.normal_texture,
            diffuse_color=material.diffuse_color,
            diffuse_tint=material.diffuse_tint,
            albedo_brightness=material.albedo_brightness,
            opacity=material.opacity,
            roughness=material.roughness,
            roughness_texture_influence=material.roughness_texture_influence,
            metallic=material.metallic,
            specular_level=material.specular_level,
            normal_strength=material.normal_strength,
            normal_format=material.normal_format,
            texture_scale=material.texture_scale,
            texture_rotation=material.texture_rotation,
            project_uvw=material.project_uvw,
            world_space_projection=material.world_space_projection,
        )
    if config.vantablack_components is not None:
        job.appearance.components(vantablack=config.vantablack_components)
