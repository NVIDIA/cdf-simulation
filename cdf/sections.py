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

"""The six declarative job-description areas.

Every public call here only records validated specifications on the job plan;
nothing touches the USD stage or the filesystem. Compilation to the pipeline's
YAML schema happens in :mod:`cdf.plan`; execution happens in
``Capture.render()`` / ``Postprocess.crop()``.
"""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import TYPE_CHECKING

from .component_filter import normalize_component_filter
from .errors import ValidationError
from .image_formats import validate_image_format

if TYPE_CHECKING:
    from .execution.executor_base import JobExecutor
    from .execution.job_result_types import JobResult
    from .job import CDFJob

_ALL = "ALL"  # component_types keyword sdg_pipeline resolves against the board's own registry


def _pair(name, value, *, ordered=True):
    """Validate a (lo, hi) range and return it as a yaml-ready 2-list."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(v, numbers.Real) for v in value)
    ):
        raise ValidationError(
            f"{name} must be a (lo, hi) pair of numbers, got {value!r}"
        )
    lo, hi = value
    if ordered and lo > hi:
        raise ValidationError(f"{name}: lo ({lo}) must be <= hi ({hi})")
    return [lo, hi]


def _ratio(name, value):
    if not isinstance(value, numbers.Real) or not 0.0 <= value <= 1.0:
        raise ValidationError(f"{name} must be within [0, 1], got {value!r}")
    return value


def _color_layer(name, spec, *, allow_visible=False):
    """Validate an {intensity, color_r/g/b: (lo, hi)} lighting-layer mapping."""
    if not isinstance(spec, dict):
        raise ValidationError(f"lighting layer {name!r} must be a mapping")
    out = {}
    for key, value in spec.items():
        if key == "visible" and allow_visible:
            if not isinstance(value, bool):
                raise ValidationError(f"lighting layer {name!r}: visible must be bool")
            out[key] = value
            continue
        if key not in ("intensity", "color_r", "color_g", "color_b"):
            raise ValidationError(f"lighting layer {name!r}: unknown key {key!r}")
        out[key] = _pair(f"{name}.{key}", value)
    return out


_MODUS_GROUPS = ("white", "red", "blue")


def _modus_group_map(name, value, *, item):
    """Validate a ``{white|red|blue: <item>}`` Modus group mapping."""
    if not isinstance(value, dict):
        raise ValidationError(f"lighting.modus.{name} must be a mapping")
    out = {}
    for group, entry in value.items():
        if group not in _MODUS_GROUPS:
            raise ValidationError(
                f"lighting.modus.{name}: unknown group {group!r} (want white/red/blue)"
            )
        out[group] = item(f"modus.{name}.{group}", entry)
    return out


def _modus_scalar(name, value):
    if not isinstance(value, numbers.Real):
        raise ValidationError(f"lighting.{name} must be a number, got {value!r}")
    return value


def _modus_color(name, value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or not all(isinstance(v, numbers.Real) for v in value)
    ):
        raise ValidationError(
            f"lighting.{name} must be an (r, g, b) triple, got {value!r}"
        )
    return [float(v) for v in value]


def _modus_block(spec):
    """Validate the ``lighting.modus:`` block (the 2D-dome linear illuminator).

    Every knob is optional (the pipeline builder supplies defaults). Keys:
    ``arc_radius_scale`` / ``width`` / ``width_frac`` / ``height_scale``
    (numbers, ``width`` and ``width_frac`` mutually exclusive), ``elevations``
    / ``intensities`` ({group: number}), ``colors`` ({group: (r,g,b)}), and
    ``layers`` ({group: {intensity, color_r, color_g, color_b}} ranges) for
    optional per-trigger randomization -- same schema as ``lighting.layers``
    on the ring, restricted to the white/red/blue groups.
    """
    if not isinstance(spec, dict):
        raise ValidationError("lighting.modus must be a mapping")
    if "width" in spec and "width_frac" in spec:
        raise ValidationError(
            "lighting.modus.width cannot be combined with "
            "lighting.modus.width_frac -- give one or the other"
        )
    out = {}
    for key, value in spec.items():
        if key in ("arc_radius_scale", "width", "width_frac", "height_scale"):
            out[key] = _modus_scalar(key, value)
        elif key == "elevations":
            out[key] = _modus_group_map("elevations", value, item=_modus_scalar)
        elif key == "intensities":
            out[key] = _modus_group_map("intensities", value, item=_modus_scalar)
        elif key == "colors":
            out[key] = _modus_group_map("colors", value, item=_modus_color)
        elif key == "layers":
            out[key] = _modus_group_map("layers", value, item=_color_layer)
        else:
            raise ValidationError(f"lighting.modus: unknown key {key!r}")
    return out


LIGHTING_MODES = (
    "rgb_light",
    "white_light",
    "dome_light",
    "modus_light",
    "usd_light",
)
#: Lighting rigs the pipeline can build, one per authoring mode.
#:
#: ``rgb_light``    the tri-colour AOI ring (needs ``layers``; ``dome_fill`` optional)
#: ``white_light``  one DomeLight pinned to pure white (``dome`` carries intensity only)
#: ``dome_light``   one DomeLight with an authored/randomized colour (``dome``)
#: ``modus_light``  the Modus 2D-dome linear illuminator (``modus``)
#: ``usd_light``    reuse the lights authored in the board USD

_WHITE = [1.0, 1.0]

_MODE_PAYLOAD = {
    "rgb_light": ("layers",),
    "white_light": ("dome",),
    "dome_light": ("dome",),
    "modus_light": ("modus",),
    "usd_light": (),
}


def _lighting_mode(mode):
    """Validate the rig selector."""
    if mode is None:
        raise ValidationError(f"lighting requires mode=<{'|'.join(LIGHTING_MODES)}>")
    if mode not in LIGHTING_MODES:
        raise ValidationError(
            f"lighting.mode must be one of {', '.join(LIGHTING_MODES)}, got {mode!r}"
        )
    return mode


def _dome_block(mode, dome):
    """Validate the DomeLight payload for the white/dome modes.

    ``white_light`` is the neutral flood: intensity is authored, the colour is
    pinned to (1, 1, 1). ``dome_light`` is the same rig with an authored or
    randomized colour, so it is the mode to use whenever a tint is wanted.
    """
    block = _color_layer("dome", dome if dome is not None else {}, allow_visible=True)
    tinted = [k for k in ("color_r", "color_g", "color_b") if k in block]
    if mode == "white_light":
        if tinted:
            raise ValidationError(
                "lighting.dome: mode='white_light' pins the dome to white; "
                f"use mode='dome_light' to author {', '.join(sorted(tinted))}"
            )
        block["color_r"] = list(_WHITE)
        block["color_g"] = list(_WHITE)
        block["color_b"] = list(_WHITE)
    elif len(tinted) != 3:
        raise ValidationError(
            "lighting.dome: mode='dome_light' needs color_r, color_g and "
            "color_b (use mode='white_light' for a pure-white dome)"
        )
    if "intensity" not in block:
        raise ValidationError("lighting.dome requires an intensity (lo, hi) pair")
    return block


class SceneSection:
    """Scene — USD assets, sidecars, camera bindings, and scene entities."""

    def __init__(self, job):
        """Initialize scene declarations for a job."""
        self._job = job
        self._domain = job.domain
        self.config = None
        self.instance_paths: tuple[str, ...] | None = None
        self.sidecars = {}
        self.selection = None

    def configure(
        self,
        *,
        usd_path,
        root_path,
        archive_member=None,
        camera_path=None,
        camera_xform_path=None,
        ring_light_root=None,
        horizontal_aperture=None,
        meters_per_unit=None,
    ):
        """Bind the board USD and its prim paths.

        PCBA config fields include ``scene``, ``pcba_root``, ``archive_member``,
        ``camera_path``, ``camera_xform_path``, ``ring_light_root``, and
        ``horizontal_aperture``.

        Args:
            usd_path: Path to the board USD asset.
            root_path: Absolute prim path to the board root.
            archive_member: USD member to load when ``usd_path`` is an archive.
            camera_path: Absolute prim path to the capture camera.
            camera_xform_path: Absolute prim path to the camera transform.
            ring_light_root: Absolute prim path to the ring-light root.
            horizontal_aperture: Camera horizontal aperture override.
            meters_per_unit: Metres per scene unit -- USD's own
                ``metersPerUnit``, so ``0.001`` for a millimetre stage --
                overriding the stage's declaration. Stated in that direction
                because the reciprocal is a plausible-looking number too, and
                getting it backwards frames the capture wrong by the square of
                the scale. Only a capture stated in millimetres reads it --
                everything else in this pipeline works in scene units and never
                converts, which is why a wrong declaration goes unnoticed until
                one does. Never inferred from geometry: a stage that lies is
                still the stage every other consumer reads.

        """
        if not usd_path:
            raise ValidationError("scene.configure: usd_path is required")
        if archive_member is not None and not str(archive_member).strip():
            raise ValidationError("scene.configure: archive_member cannot be empty")
        if not root_path or not str(root_path).startswith("/"):
            raise ValidationError(
                f"scene.configure: root_path must be an absolute prim path, got {root_path!r}"
            )
        self.config = {
            "usd_path": str(usd_path),
            "root_path": str(root_path),
            "archive_member": (
                str(archive_member) if archive_member is not None else None
            ),
            "camera_path": str(camera_path) if camera_path else None,
            "camera_xform_path": str(camera_xform_path) if camera_xform_path else None,
            "ring_light_root": str(ring_light_root) if ring_light_root else None,
            "horizontal_aperture": horizontal_aperture,
            "meters_per_unit": (
                float(meters_per_unit) if meters_per_unit is not None else None
            ),
            "instance_paths": self.instance_paths,
        }
        self._job._record_declaration(
            "scene.load",
            "scene.load",
            self.config,
        )
        return self

    def instances(
        self,
        *,
        prim_paths: list[str] | tuple[str, ...],
    ) -> SceneSection:
        """Register one complete shared-prototype board-instance cohort.

        The first path is the representative board used to calculate operations.
        Runtime validation requires every path to be a USD instance of the same
        prototype and requires the cohort to include every matching instance in
        the configured scene root.

        Args:
            prim_paths: Ordered absolute prim paths for all board instances.

        Returns:
            This scene section for call chaining.

        Raises:
            ValidationError: If scene configuration is missing or paths are
                empty, invalid, or duplicated.

        Example:

        .. code-block:: python

            job.scene.instances(
                prim_paths=(
                    "/World/Panel/StepRepeats/SR_0_0",
                    "/World/Panel/StepRepeats/SR_0_1",
                ),
            )

        """
        if self.config is None:
            raise ValidationError(
                "scene.instances requires scene.configure first",
            )
        if not isinstance(prim_paths, (list, tuple)) or not prim_paths:
            raise ValidationError(
                "scene.instances: prim_paths must be a non-empty list or tuple",
            )
        if not all(isinstance(path, str) for path in prim_paths):
            raise ValidationError(
                "scene.instances: every prim path must be a string",
            )
        if any(
            sidecar["kind"] == "pcb.components" for sidecar in self.sidecars.values()
        ):
            raise ValidationError(
                "scene.instances requires embedded USD component metadata; "
                "pcb.components sidecars cannot describe a shared prototype cohort",
            )
        normalized = tuple(prim_paths)
        invalid = tuple(
            path
            for path in normalized
            if not path.startswith("/") or path == "/" or "//" in path
        )
        if invalid:
            raise ValidationError(
                "scene.instances: every prim path must be absolute and non-root; "
                f"invalid paths: {list(invalid)!r}",
            )
        if len(set(normalized)) != len(normalized):
            raise ValidationError("scene.instances: prim_paths must be unique")
        self.instance_paths = normalized
        self.config["instance_paths"] = normalized
        self._job._record_declaration(
            "scene.load",
            "scene.load",
            self.config,
            preserve_position=True,
        )
        return self

    def attach_sidecar(self, *, name, kind, path):
        """Attach a metadata sidecar file.

        For example, kind ``pcb.components`` maps to flow ``components_json``;
        ``pcb.sim_manifest`` maps to ``sim_manifest``.
        """
        if kind not in self._domain.sidecar_kinds:
            raise ValidationError(
                f"scene.attach_sidecar: kind {kind!r} not supported by domain {self._domain.name!r} "
                f"(supported: {sorted(self._domain.sidecar_kinds)})"
            )
        if kind == "pcb.components" and self.instance_paths is not None:
            raise ValidationError(
                "scene.instances requires embedded USD component metadata; "
                "pcb.components sidecars cannot describe a shared prototype cohort",
            )
        if name in self.sidecars:
            raise ValidationError(f"scene.attach_sidecar: {name!r} already attached")
        self.sidecars[name] = {"kind": kind, "path": str(path)}
        self._job._record_declaration(
            f"scene.sidecar.{name}",
            "scene.sidecar.load",
            {"name": name, **self.sidecars[name]},
            replace=False,
        )
        return self

    def select_components(self, *, source=None, types=None, component_filter=None):
        """Scope component families using embedded or attached metadata."""
        if source is not None and source != "usd" and source not in self.sidecars:
            raise ValidationError(
                f"scene.select_components: unknown metadata source {source!r} "
                f"(attached: {sorted(self.sidecars)})"
            )
        if types is not None and not (
            isinstance(types, (list, tuple)) and all(isinstance(t, str) for t in types)
        ):
            raise ValidationError(
                "scene.select_components: types must be a list of names or None"
            )
        if types is not None and component_filter is not None:
            raise ValidationError(
                "scene.select_components: types and component_filter are mutually exclusive"
            )
        try:
            normalized_filter = normalize_component_filter(
                component_filter,
                where="scene.select_components",
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        self.selection = {
            "source": source,
            "types": list(types) if types else None,
            "component_filter": normalized_filter,
        }
        self._job._record_declaration(
            "scene.components.select",
            "scene.components.select",
            self.selection,
        )
        return self


class AppearanceSection:
    """Appearance — lighting, materials, and presentation overrides."""

    def __init__(self, job):
        """Initialize appearance declarations for a job."""
        self._job = job
        self.lighting_spec = None
        self.components_spec = None
        self.material_specs = {}

    def lighting(
        self,
        *,
        mode=None,
        preserve_scene_light_color=False,
        exposure_range=None,
        cone_angle_range=None,
        cone_softness_range=None,
        layers=None,
        dome_fill=None,
        dome=None,
        modus=None,
    ):
        """Describe lighting: pick a rig with ``mode`` and give it its payload.

        ``mode`` is one of :data:`LIGHTING_MODES`; each takes exactly one
        payload argument, and supplying another rig's payload is an error
        rather than a silent no-op:

        ============== ================= ==================================
        mode           payload           rig
        ============== ================= ==================================
        ``rgb_light``  ``layers``        tri-colour AOI ring (``dome_fill``
                                         adds the neutral white dome)
        ``white_light`` ``dome``         one DomeLight pinned to white
        ``dome_light``  ``dome``         one DomeLight, colour authored
        ``modus_light`` ``modus``        Modus 2D-dome linear illuminator
                                         (6 RectLights on an arc:
                                         white/red/blue, centre-out;
                                         ``dome_fill`` adds neutral fill)
        ``usd_light``   --               reuse the board USD's own lights
        ============== ================= ==================================

        Range arguments are ``(lo, hi)`` pairs sampled per trigger.
        ``exposure_range`` / ``cone_angle_range`` / ``cone_softness_range``
        apply to the spotlight rigs; a DomeLight ignores the cone pair.

        ``preserve_scene_light_color`` only applies to ``usd_light``: it
        keeps the authored colours instead of whitening them.

        The Modus rig places six ``UsdLux.RectLight``s on an arc centred on
        the board, in white/red/blue pairs working centre-out, each aimed at
        the board centre. Defaults: elevations 75/60/45 degrees, arc radius
        1.5x and height 2.0x the board size, rect width 0.12 of it (``width``
        and ``width_frac`` state the same thing two ways and cannot both be
        given). Every knob is overridable through the ``modus`` payload,
        including per-trigger randomization of a group's intensity/colour
        through ``modus["layers"]`` -- same schema as ``layers`` above,
        restricted to white/red/blue; a group left out of it keeps its fixed
        ``intensities``/``colors`` value. Batch capture places the rig on the
        inspected side selected by the capture camera.

        Args:
            mode: Lighting rig selector.
            preserve_scene_light_color: Whether ``usd_light`` retains colors
                authored in the scene.
            exposure_range: Per-trigger exposure range.
            cone_angle_range: Per-trigger spotlight cone-angle range.
            cone_softness_range: Per-trigger spotlight cone-softness range.
            layers: Named RGB ring-light layers.
            dome_fill: Optional neutral fill for RGB ring or batch Modus rigs.
            dome: Dome-light settings for white or authored-color dome modes.
            modus: Modus rig geometry, color, intensity, and layer settings.

        Returns:
            This appearance section for call chaining.

        Raises:
            ValidationError: If the mode, payload, range, or Modus settings are
                invalid or incompatible.

        Example:

        .. code-block:: python

            job.appearance.lighting(
                mode="modus_light",
                modus={},
                dome_fill={"intensity": (500.0, 700.0)},
            )

        """
        mode = _lighting_mode(mode)

        supplied = {
            "layers": layers,
            "dome": dome,
            "modus": modus,
            "dome_fill": dome_fill,
        }
        allowed = set(_MODE_PAYLOAD[mode])
        if mode in ("rgb_light", "modus_light"):
            allowed.add("dome_fill")
        for key, value in supplied.items():
            if value is not None and key not in allowed:
                raise ValidationError(
                    f"lighting.{key} is not part of mode={mode!r} "
                    f"(that mode takes {', '.join(sorted(allowed)) or 'no payload'})"
                )

        spec = {
            "mode": mode,
            "preserve_scene_light_color": bool(preserve_scene_light_color),
        }
        if mode == "modus_light":
            if modus is None:
                raise ValidationError(
                    "lighting.modus is required for mode='modus_light' "
                    "(pass {} to accept the rig defaults)"
                )
            spec["modus"] = _modus_block(modus)
        if mode == "rgb_light":
            if layers is None:
                raise ValidationError(
                    "lighting.layers is required for mode='rgb_light' "
                    "(per-ring intensity and colour drive the AOI ring)"
                )
            spec["layers"] = {
                name: _color_layer(name, layer) for name, layer in layers.items()
            }
        if mode in ("white_light", "dome_light"):
            spec["white_light"] = _dome_block(mode, dome)
        if exposure_range is not None:
            spec["exposure_range"] = _pair("lighting.exposure_range", exposure_range)
        if cone_angle_range is not None:
            spec["cone_angle_range"] = _pair(
                "lighting.cone_angle_range", cone_angle_range
            )
        if cone_softness_range is not None:
            spec["cone_softness_range"] = _pair(
                "lighting.cone_softness_range", cone_softness_range
            )
        if dome_fill is not None:
            spec["dome_fill"] = _color_layer("dome_fill", dome_fill)
        self.lighting_spec = spec
        self._job._record_declaration(
            "appearance.lighting",
            "appearance.lighting.configure",
            self.lighting_spec,
        )
        return self

    def material_override(
        self,
        name,
        *,
        material_path,
        diffuse_texture=None,
        roughness_texture=None,
        normal_texture=None,
        diffuse_color=(0.2, 0.2, 0.2),
        diffuse_tint=(1.0, 1.0, 1.0),
        albedo_brightness=1.0,
        opacity=1.0,
        roughness=0.5,
        roughness_texture_influence=1.0,
        metallic=0.0,
        specular_level=0.5,
        normal_strength=1.0,
        normal_format="opengl",
        texture_scale=(1.0, 1.0),
        texture_rotation=0.0,
        project_uvw=False,
        world_space_projection=False,
        variation=None,
    ):
        """Override an existing bound material with an OmniPBR appearance.

        Args:
            name: Unique job-local material override name.
            material_path: Absolute path of the existing USD material prim.
            diffuse_texture: Optional albedo texture asset path. A texture
                shipped with the scene is addressed as
                ``scene://textures/albedo.png``, which resolves beside the USD
                or inside the directory a scene ZIP was extracted into. A bare
                relative path is tried on disk first and falls back to the
                scene, so an archive stays self-contained without fixing any
                path in advance.
            roughness_texture: Optional scalar roughness texture asset path,
                resolved the same way.
            normal_texture: Optional tangent-space normal texture asset path,
                resolved the same way.
            diffuse_color: Base color used when no albedo texture is present.
            diffuse_tint: Color multiplied over the albedo result.
            albedo_brightness: Albedo brightness multiplier.
            opacity: Constant surface opacity from transparent to opaque.
            roughness: Constant surface roughness.
            roughness_texture_influence: Roughness texture blend weight.
            metallic: Metallic surface response.
            specular_level: Surface specular intensity.
            normal_strength: Normal-map strength.
            normal_format: Normal-map convention, ``opengl`` or ``directx``.
            texture_scale: Projected texture tiling along two axes.
            texture_rotation: Projected texture rotation in degrees.
            project_uvw: Whether OmniPBR generates UVW coordinates.
            world_space_projection: Whether projection uses world space.
            variation: Optional ``{control: (low, high)}`` ranges sampled once per
                job from its ``random_seed``, so a dataset can vary surface
                appearance without editing the scene. Supported controls are
                ``albedo_brightness``, ``roughness``, ``metallic``,
                ``specular_level``, ``normal_strength``, ``texture_rotation`` and
                ``texture_scale`` (a uniform multiplier over the declared scale).
                A sampled control replaces its constant.

        Returns:
            Appearance section for chained declarations.

        Raises:
            ValidationError: If a name, path, color, or numeric control is invalid.

        Example:

        .. code-block:: python

            job.appearance.material_override(
                "carrier",
                material_path="/World/Looks/Carrier",
                diffuse_texture="scene://carrier_albedo.png",
                project_uvw=True,
            )

        """
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("appearance.material_override: name must be nonempty")
        if name in self.material_specs:
            raise ValidationError(
                f"appearance.material_override: duplicate name {name!r}"
            )
        if not isinstance(material_path, str) or not material_path.startswith("/"):
            raise ValidationError(
                "appearance.material_override: material_path must be absolute"
            )
        textures = {
            "diffuse_texture": diffuse_texture,
            "roughness_texture": roughness_texture,
            "normal_texture": normal_texture,
        }
        for field_name, value in textures.items():
            if value is not None and not isinstance(value, (str, Path)):
                raise ValidationError(
                    f"appearance.material_override: {field_name} must be a path"
                )
        if normal_format not in ("opengl", "directx"):
            raise ValidationError(
                "appearance.material_override: normal_format must be "
                "'opengl' or 'directx'"
            )
        for field_name, color in (
            ("diffuse_color", diffuse_color),
            ("diffuse_tint", diffuse_tint),
        ):
            if (
                not isinstance(color, (list, tuple))
                or len(color) != 3
                or not all(isinstance(value, numbers.Real) for value in color)
            ):
                raise ValidationError(
                    f"appearance.material_override: {field_name} must be three numbers"
                )
        variation_spec = {}
        if variation is not None:
            if not isinstance(variation, dict):
                raise ValidationError(
                    "appearance.material_override: variation must be a mapping"
                )
            allowed = (
                "albedo_brightness",
                "roughness",
                "metallic",
                "specular_level",
                "normal_strength",
                "texture_rotation",
                "texture_scale",
            )
            unknown = sorted(set(variation) - set(allowed))
            if unknown:
                raise ValidationError(
                    "appearance.material_override: unknown variation controls "
                    f"{unknown}; allowed {list(allowed)}"
                )
            for control, bounds in variation.items():
                variation_spec[control] = _pair(
                    f"appearance.material_override.variation.{control}",
                    bounds,
                )
        texture_scale = _pair(
            "appearance.material_override.texture_scale",
            texture_scale,
            ordered=False,
        )
        numeric = {
            "albedo_brightness": albedo_brightness,
            "opacity": opacity,
            "roughness": roughness,
            "roughness_texture_influence": roughness_texture_influence,
            "metallic": metallic,
            "specular_level": specular_level,
            "normal_strength": normal_strength,
            "texture_rotation": texture_rotation,
        }
        if not all(isinstance(value, numbers.Real) for value in numeric.values()):
            raise ValidationError(
                "appearance.material_override: material controls must be numeric"
            )
        spec = {
            "material_path": material_path,
            **{
                field_name: str(value)
                for field_name, value in textures.items()
                if value is not None
            },
            "diffuse_color": list(diffuse_color),
            "diffuse_tint": list(diffuse_tint),
            **numeric,
            "normal_format": normal_format,
            "texture_scale": texture_scale,
            "project_uvw": bool(project_uvw),
            "world_space_projection": bool(world_space_projection),
        }
        if variation_spec:
            spec["variation"] = variation_spec
        self.material_specs[name] = spec
        self._job._record_declaration(
            f"appearance.material.{name}",
            "appearance.material.override",
            spec,
        )
        return self

    def components(self, *, vantablack=False):
        """Component presentation overrides (flow ``vantablack_components``)."""
        self.components_spec = {"vantablack": bool(vantablack)}
        self._job._record_declaration(
            "appearance.components",
            "appearance.components.configure",
            self.components_spec,
        )
        return self


class OperationsSection:
    """Domain-neutral base for a domain's declarative operation section.

    Domain packages subclass this type and expose their own public methods.
    The base only records normalized operation declarations.

    Args:
        job: Declarative job that receives recorded operation intent.

    """

    def __init__(self, job: CDFJob) -> None:
        """Initialize operation declarations for a job."""
        self._job = job
        self._domain = job.domain
        self.ops: dict[str, dict[str, object]] = {}

    def _add(
        self,
        name: str,
        params: dict[str, object],
        declaration_params: dict[str, object] | None = None,
    ) -> OperationsSection:
        """Record a domain operation using the standard JobOrder naming.

        Args:
            name: Operation name registered by the domain.
            params: Normalized operation parameters used by legacy compilation.
            declaration_params: Optional parameters recorded for JobOrder
                compilation.

        Returns:
            This operation section for call chaining.

        Raises:
            ValidationError: If the operation is unsupported or already added.

        """
        if name not in self._domain.operations:
            raise ValidationError(
                f"operation {name!r} not supported by domain {self._domain.name!r} (supported: {sorted(self._domain.operations)})"
            )
        if name in self.ops:
            raise ValidationError(f"operation {name!r} already added")
        self.ops[name] = params
        self._job._record_declaration(
            f"operation.{name}",
            f"operation.{name}.apply",
            declaration_params if declaration_params is not None else params,
            replace=False,
        )
        return self


class AnnotationsSection:
    """Annotations — baseline semantics and labeling rules."""

    def __init__(self, job):
        """Initialize annotation declarations for a job."""
        self._job = job
        self.component_labels = None
        self.operation_labels = None
        self.geometry_labels = None

    def label_components(self, *, source=None, classes=None):
        """Declare component class labels for ground truth.

        An explicit ``{class: [refdes]}`` map compiles to PCBA
        ``component_classes``; ``None`` derives labels from component scope names.
        """
        if classes is not None and not isinstance(classes, dict):
            raise ValidationError(
                "annotations.label_components: classes must be a "
                "{class: [refdes, ...]} mapping or None (derive from sidecar)"
            )
        self.component_labels = {"source": source, "classes": classes}
        self._job._record_declaration(
            "annotations.components",
            "annotations.components.apply",
            self.component_labels,
        )
        return self

    def label_geometry(self, *, components=None, component_filter=None, **labels):
        """Declare semantic classes for geometry that is not a component.

        ``label_components`` labels whole component scopes, which is the right
        granularity for detection but too coarse for joint-level ground truth:
        a pin and the solder fillet around it share one component scope and so
        collapse into one class. This declares a class for each separately.

        Which geometry a board has is domain vocabulary, so the accepted
        keywords come from the registered domain rather than from here. For
        ``pcb-aoi`` they are ``solder``, ``terminals``, ``pads``, ``holes``,
        ``barrels``, ``body`` and ``marking`` (see ``pcb_aoi.geometry_labels``).
        ``body`` and ``marking`` are the package body mesh and the pin-1 dot /
        polarity band mesh of a component prototype, so a mask can separate
        the polarity marker from the body it sits on.

        Args:
            components: Optional allowlist of refdes strings.  When set, only
                pads, terminals, and holes belonging to these components are
                labeled; the full board vocabulary is labeled when ``None``.
            component_filter: Optional exact metadata-field filter. Mutually
                exclusive with ``components``.
            **labels: One class name per geometry target the domain declares.

        Returns:
            This section, for chaining.

        Raises:
            ValidationError: If a target is unknown or its class is not a name.

        Example:

        .. code-block:: python

            job.annotations.label_geometry(solder="solder", terminals="pin")
            job.annotations.label_geometry(
                pads="pad", components=["C1", "C2"]
            )

        """
        domain = self._job.domain
        targets = getattr(domain, "geometry_targets", frozenset())
        if not targets:
            raise ValidationError(
                f"annotations.label_geometry: domain {domain.name!r} "
                "declares no labelable geometry"
            )
        unknown = sorted(set(labels) - targets)
        if unknown:
            raise ValidationError(
                "annotations.label_geometry: unknown target(s) "
                f"{unknown}; {domain.name} supports {sorted(targets)}"
            )
        if not labels:
            raise ValidationError(
                f"annotations.label_geometry: declare at least one of {sorted(targets)}"
            )
        for target, value in labels.items():
            if not value or not isinstance(value, str):
                raise ValidationError(
                    f"annotations.label_geometry: {target} must be a class name"
                )
        if components is not None and (
            not isinstance(components, (list, tuple))
            or not components
            or not all(isinstance(c, str) and c for c in components)
        ):
            raise ValidationError(
                "annotations.label_geometry: components must be a non-empty list of strings"
            )
        if components is not None and component_filter is not None:
            raise ValidationError(
                "annotations.label_geometry: components and component_filter "
                "are mutually exclusive"
            )
        try:
            normalized_filter = normalize_component_filter(
                component_filter,
                where="annotations.label_geometry",
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        spec = dict(labels)
        if components is not None:
            spec["components"] = tuple(str(c) for c in components)
        if normalized_filter is not None:
            spec["component_filter"] = normalized_filter
        self.geometry_labels = spec
        self._job._record_declaration(
            "annotations.geometry",
            "annotations.geometry.apply",
            self.geometry_labels,
        )
        return self

    def enable_operation_labels(self, *, namespace="defect"):
        """Label operated components under ``namespace``.

        This sets the ``writer.semantic_types`` default so operation labels
        land in the segmentation output for defect flows.
        """
        if not namespace or not isinstance(namespace, str):
            raise ValidationError(
                "annotations.enable_operation_labels: namespace must be a name"
            )
        self.operation_labels = {"namespace": namespace}
        self._job._record_declaration(
            "annotations.operations",
            "annotations.operations.configure",
            self.operation_labels,
        )
        return self


class CapturePassSection:
    """Configuration for one render pass within a capture job."""

    _RENDER_MODES = ("pathtracing", "realtime")

    def __init__(self, job, *, name="default", order_prefix="capture"):
        """Initialize one pass and its declaration namespace."""
        self._job = job
        self.name = name
        self._order_prefix = order_prefix
        self.start = 0
        self.renderer_spec = None
        #: Where ``renderer_spec["resolution"]`` came from: ``"author"`` when a
        #: caller stated it, ``"derived"`` when this pass computed it from
        #: another statement of the same quantity, ``None`` when the pass has no
        #: resolution yet. Keeping the provenance is what lets the two ways of
        #: stating a frame size be order-independent and still refuse to be
        #: given both.
        self.resolution_source = None
        self.camera_spec = None
        self.camera_side = "top"
        self.scan_grid_spec = None
        #: Where ``scan_grid_spec`` came from: ``"author"`` when a caller
        #: stated it directly, ``"derived"`` when :meth:`fov` computed it from
        #: a scan rectangle, ``None`` when the pass has no grid yet. Mirrors
        #: ``resolution_source`` so a fov and a directly authored grid are
        #: order-independent and still refuse to be given both.
        self.grid_source = None
        self.visibility_spec = {"hide": [], "show": []}
        self.roi_selection_specs = []
        self.limits_spec = {"triggers": 1, "max_images": -1}
        self.augmentation_spec = None
        self.outputs_spec = None

    def __enter__(self):
        """Return this render pass for scoped configuration."""
        return self

    def __exit__(self, exception_type, exception, traceback):
        """Validate a successfully configured scoped render pass."""
        if exception_type is None:
            self._validate_required_configuration()
        return False

    def _order_id(self, leaf):
        """Return a namespaced JobOrder identifier for this pass."""
        return f"{self._order_prefix}.{leaf}"

    def _validate_required_configuration(self):
        """Require the configuration needed to execute this render pass."""
        missing = [
            name
            for name, value in (
                ("renderer", self.renderer_spec),
                ("camera", self.camera_spec),
                ("scan_grid", self.scan_grid_spec),
                ("outputs", self.outputs_spec),
            )
            if value is None
        ]
        if missing:
            raise ValidationError(
                f"capture render pass {self.name!r} is missing: {', '.join(missing)}"
            )
        if not self.renderer_spec.get("resolution"):
            raise ValidationError(
                f"capture render pass {self.name!r} has no resolution: give "
                "renderer.resolution, or state the frame size in the form that "
                "derives it"
            )

    def _refresh_derived_capture(self):
        """Recompute anything this pass derives from the camera and renderer.

        A no-op here: core capture is stated directly, in pixels and cells, so
        there is nothing to recompute. A domain capture section that lets a
        pass be stated in the domain's own terms overrides this and re-derives
        from the current camera, which is what makes the public calls
        order-independent -- a yaw set after the derivation still turns the
        frame, instead of silently leaving it square to the wrong axis.
        """
        return

    def output_start(self, start):
        """Set the first written frame index.

        Flow ``seed`` skips the first ``start`` scan-grid cells on trigger 0.
        """
        if not isinstance(start, int) or start < 0:
            raise ValidationError("capture.output_start must be an int >= 0")
        self.start = start
        self._job._record_declaration(
            self._order_id("output"),
            "capture.output.configure",
            {"start": self.start},
        )
        return self

    def renderer(
        self,
        *,
        mode,
        resolution=None,
        spp,
        total_spp,
        max_bounces,
        realtime_subframes=1,
    ):
        """Configure renderer settings.

        This maps to flow ``resolution`` and the ``pathtracing:`` block.
        ``"realtime"`` renders about **1.47x** faster on a scan grid and is a
        measurably different picture -- 37.8 dB against a 128-sample reference
        where path tracing is 46.7 dB. It is not a drop-in for a dataset that
        has already been shot path-traced.

        Args:
            mode: ``"pathtracing"`` or ``"realtime"``.
            resolution: Image width and height. Optional, and only optional,
                when the pass states its frame size another way -- a domain
                capture section may derive it from a statement of the same
                quantity in the domain's own terms. A pass that reaches
                execution with neither is refused.
            spp: Samples per rendered subframe (path tracing).
            total_spp: Samples accumulated per frame (path tracing).
            max_bounces: Path-tracing bounce limit.
            realtime_subframes: ``rt_subframes`` for ``"realtime"``; ignored by
                path tracing.

        """
        if mode not in self._RENDER_MODES:
            raise ValidationError(
                f"capture.renderer: mode must be one of {self._RENDER_MODES}, got {mode!r}"
            )
        if not isinstance(realtime_subframes, int) or realtime_subframes < 1:
            raise ValidationError(
                "capture.renderer: realtime_subframes must be an int >= 1, "
                f"got {realtime_subframes!r}"
            )
        if resolution is None:
            # Keep whatever a derivation already put here, so the two calls
            # commute: stating the frame size first and the renderer second
            # must produce the same pass as the reverse order.
            previous = (self.renderer_spec or {}).get("resolution")
            size = list(previous) if previous else None
        else:
            if self.resolution_source == "derived":
                raise ValidationError(
                    "capture.renderer: this pass already derives its resolution; "
                    "give renderer.resolution or the derived form, not both"
                )
            width, height = _pair(
                "capture.renderer.resolution", resolution, ordered=False
            )
            size = [int(width), int(height)]
            self.resolution_source = "author"
        self.renderer_spec = {
            "mode": mode,
            "resolution": size,
            "spp": int(spp),
            "total_spp": int(total_spp),
            "max_bounces": int(max_bounces),
            "realtime_subframes": int(realtime_subframes),
        }
        self._job._record_declaration(
            self._order_id("renderer"),
            "capture.renderer.configure",
            self.renderer_spec,
        )
        self._refresh_derived_capture()
        return self

    def camera(self, *, rotation_range, side="top"):
        """Configure the camera side and per-trigger rotation.

        Args:
            rotation_range: Rotation ranges for ``x`` and ``y`` and a fixed
                rotation for ``z``.
            side: Board side viewed by the camera.

        Returns:
            This capture section for chained declarations.

        Raises:
            ValidationError: If ``side`` or a rotation axis is unsupported.

        Example:

        .. code-block:: python

            job.capture.camera(
                side="bottom",
                rotation_range={"x": (0, 0), "y": (0, 0), "z": -90},
            )

        """
        if side not in {"top", "bottom"}:
            raise ValidationError(
                f"capture.camera: side must be 'top' or 'bottom', got {side!r}"
            )
        unknown = set(rotation_range) - {"x", "y", "z"}
        if unknown:
            raise ValidationError(
                f"capture.camera: unknown rotation axes {sorted(unknown)}"
            )
        spec = {}
        for axis in ("x", "y"):
            if axis in rotation_range:
                spec[f"{axis}_range"] = _pair(
                    f"camera.rotation_range.{axis}", rotation_range[axis]
                )
        if "z" in rotation_range:
            z = rotation_range["z"]
            if isinstance(z, (list, tuple)):
                raise ValidationError(
                    "capture.camera: z rotation is fixed in the pipeline "
                    "(z_fixed) — pass a single number"
                )
            spec["z_fixed"] = z
        self.camera_spec = spec
        self.camera_side = side
        declaration_spec = {
            "side": side,
            "rotation_range": {
                axis: list(value) if isinstance(value, (list, tuple)) else value
                for axis, value in rotation_range.items()
            },
        }
        self._job._record_declaration(
            self._order_id("camera"),
            "capture.camera.configure",
            declaration_spec,
        )
        self._refresh_derived_capture()
        return self

    def scan_grid(
        self,
        *,
        x_count,
        y_count,
        scan_area_mm=None,
        centered_on=None,
        scan_center_mm=None,
        image_size_px=None,
        guard_px=0,
        _derived=False,
    ):
        """Configure scan-grid dimensions.

        Flow ``scan_grid.x_num`` and ``y_num`` produce one frame per cell per
        trigger.

        Args:
            x_count: Cells across the board's world X.
            y_count: Cells across the board's world Y.
            scan_area_mm: World rectangle the grid spans,
                ``(length_x, breadth_y)`` in millimetres, centred on the board.
                ``None`` spans the whole board. Stating it switches off the
                orthographic auto-fit, which otherwise grows the aperture to
                ``bbox / count`` and never shrinks it -- a field of view that
                scans LESS than the board is exactly the case it refuses.
                Millimetres rather than an aperture because only the stage
                knows its own scale.
            centered_on: Label naming what ``scan_area_mm`` is centred on. Free
                text here: the rectangle is always centred on the board bbox
                midpoint, and this records which reading of "the board" was
                meant. The vocabulary belongs to the domain, which is the only
                side that knows whether a stage holds one board or a panel of
                them. Travels to the image-to-USD sidecar, so a consumer of a
                stitched image does not have to guess.
            scan_center_mm: World ``(x, y)`` in millimetres to center
                ``scan_area_mm`` on instead of the board bbox midpoint -- a field
                of view stated as corners by the machine's test program is
                placed, not centered. Requires ``scan_area_mm``; converted to
                scene units at render time, like the area.
            image_size_px: ``(width, height)`` the stitched canvas is cropped
                to, centred, in image axes. Cells are whole pixels, so a mosaic
                can exceed the size a scanner states by under one pixel per
                cell; this restores it. ``None`` keeps the full mosaic.
            guard_px: Pixels each cell renders beyond every edge, with its
                aperture widened by the same number of pixel pitches, and crops
                from every per-tile output before it is written. The
                renderer's outermost row and column are unreliable, and cells
                of a stated ``scan_area_mm`` abut with no overlap, so without
                this every seam carries them. Tiles, pixel layout and
                calibration describe the delivered tile either way. Requires
                ``scan_area_mm``; ``0`` renders the tile as-is.

        """
        if min(int(x_count), int(y_count)) < 1:
            raise ValidationError("capture.scan_grid counts must be >= 1")
        if centered_on is not None and scan_area_mm is None:
            raise ValidationError(
                "capture.scan_grid: centered_on describes scan_area_mm, which "
                "is not set -- a grid spanning the whole board is centred on "
                "the board by construction"
            )
        if scan_center_mm is not None and scan_area_mm is None:
            raise ValidationError(
                "capture.scan_grid: scan_center_mm places scan_area_mm, which "
                "is not set"
            )
        if scan_center_mm is not None and not (
            isinstance(scan_center_mm, (list, tuple))
            and len(scan_center_mm) == 2
            and all(
                isinstance(value, numbers.Real) and not isinstance(value, bool)
                for value in scan_center_mm
            )
        ):
            raise ValidationError(
                "capture.scan_grid: scan_center_mm must be a world [x, y] in "
                f"millimetres, got {scan_center_mm!r}"
            )
        if scan_center_mm is not None and centered_on is not None:
            raise ValidationError(
                "capture.scan_grid: centered_on labels a rectangle centered on "
                "the board; scan_center_mm places it explicitly -- give one"
            )
        if image_size_px is not None and not (
            isinstance(image_size_px, (list, tuple))
            and len(image_size_px) == 2
            and all(
                isinstance(value, numbers.Integral)
                and not isinstance(value, bool)
                and value > 0
                for value in image_size_px
            )
        ):
            raise ValidationError(
                "capture.scan_grid: image_size_px must be a positive integer "
                f"[width, height] in pixels, got {image_size_px!r}"
            )
        if (
            isinstance(guard_px, bool)
            or not isinstance(guard_px, numbers.Integral)
            or guard_px < 0
        ):
            raise ValidationError(
                "capture.scan_grid: guard_px must be a non-negative integer "
                f"number of pixels, got {guard_px!r}"
            )
        if guard_px and scan_area_mm is None:
            raise ValidationError(
                "capture.scan_grid: guard_px widens the aperture derived from "
                "scan_area_mm, which is not set -- a grid spanning the whole "
                "board fits its aperture to the board instead"
            )
        if not _derived and self.grid_source == "derived":
            raise ValidationError(
                "capture.scan_grid: this pass already derives its grid from "
                "capture.fov; give one or the other -- a fov derives the grid"
            )
        self.grid_source = "derived" if _derived else "author"
        self.scan_grid_spec = {"x_num": int(x_count), "y_num": int(y_count)}
        parameters = {
            "x_count": int(x_count),
            "y_count": int(y_count),
            "scan_area_mm": (
                [float(scan_area_mm[0]), float(scan_area_mm[1])]
                if scan_area_mm is not None
                else None
            ),
            "centered_on": str(centered_on) if centered_on is not None else None,
            "scan_center_mm": (
                [float(scan_center_mm[0]), float(scan_center_mm[1])]
                if scan_center_mm is not None
                else None
            ),
        }
        # Only when stated, so a grid without a crop records the same order as
        # before this field existed.
        if image_size_px is not None:
            parameters["image_size_px"] = [int(image_size_px[0]), int(image_size_px[1])]
        # Same rule: a grid without a guard band records the order it always did.
        if guard_px:
            parameters["guard_px"] = int(guard_px)
        self._job._record_declaration(
            self._order_id("scan_grid"),
            "capture.scan_grid.configure",
            parameters,
        )
        return self

    def visibility(self, *, hide=(), show=()):
        """Show or hide selected domain geometry while this pass renders.

        Two categories of target are accepted:

        - ``"solder"`` — hides procedurally generated solder-joint prims using
          the placement manifest.  Requires a ``solder_joints`` operation with a
          ``simulated`` solver to have run before this pass.
        - Board geometry roles — shortnames of ``pcba:mesh:role`` values
          (without the ``board_`` prefix), e.g. ``"copper"``.  All prims whose
          ``pcba:mesh:role`` attribute equals ``board_<name>`` under the board
          root are hidden for this pass and restored immediately after.

        When ``"hole"`` is listed but no ``hole_plane`` prim exists (e.g. on
        an SMD-only board), it is silently skipped.  Other missing roles emit
        a warning.

        Args:
            hide: Geometry targets hidden for this render pass.  Accepts
                ``"solder"`` and any board geometry role registered by the
                domain (e.g. ``"copper"``, ``"mask"``, ``"hole"``).  Empty
                by default.
            show: Board geometry roles temporarily shown for this render pass.
                Generated ``"hole"`` and ``"pin_tip"`` annotation proxies are
                hidden by default and should be shown in annotation passes.

        Returns:
            This capture section for chained declarations.

        Raises:
            ValidationError: If a target cannot be hidden per pass.

        Example:

        .. code-block:: python

            roi_pass.visibility(
                hide=("solder", "copper"),
                show=("hole", "pin_tip"),
            )

        """
        if isinstance(hide, str):
            hide = (hide,)
        if isinstance(show, str):
            show = (show,)
        hide_values = tuple(dict.fromkeys(str(value).strip() for value in hide))
        show_values = tuple(dict.fromkeys(str(value).strip() for value in show))
        if any(not value for value in (*hide_values, *show_values)):
            raise ValidationError("capture.visibility targets must be names")
        domain = self._job.domain
        hide_supported = ({"solder"} & set(domain.geometry_targets)) | set(
            domain.board_geometry_roles
        )
        unknown_hide = sorted(set(hide_values) - hide_supported)
        if unknown_hide:
            raise ValidationError(
                "capture.visibility cannot hide target(s) "
                f"{unknown_hide}; supported targets are {sorted(hide_supported)}",
            )
        show_supported = set(domain.board_geometry_roles)
        unknown_show = sorted(set(show_values) - show_supported)
        if unknown_show:
            raise ValidationError(
                "capture.visibility cannot show target(s) "
                f"{unknown_show}; supported targets are {sorted(show_supported)}",
            )
        overlap = sorted(set(hide_values) & set(show_values))
        if overlap:
            raise ValidationError(
                f"capture.visibility cannot show and hide target(s) {overlap}",
            )
        self.visibility_spec = {
            "hide": list(hide_values),
            "show": list(show_values),
        }
        self._job._record_declaration(
            self._order_id("visibility"),
            "capture.visibility.configure",
            self.visibility_spec,
        )
        return self

    def roi_selection(
        self,
        *,
        refdes=None,
        pins=None,
        component_types=None,
        process_classes=None,
        component_filter=None,
        all_components=False,
        defects=None,
        focus=None,
        count=None,
        padding=0.5,
    ):
        """Add one component or pin selector to this focused render pass.

        Args:
            refdes: Ordered component reference designators to render exactly.
            pins: Ordered ``(refdes, pin_number)`` identities to render exactly.
            component_types: Metadata component types from which to select.
            process_classes: Manufacturing process classes from which to select.
            component_filter: Exact metadata-field filter to select.
            all_components: Select every cataloged component.
            defects: Applied structural-defect types whose targets should render.
            focus: Render each selected component or each of its cataloged pins.
            count: Deterministic subset size for type, process, or defect selection.
            padding: Additional fractional width and height around each target.

        Returns:
            This capture section for chained declarations.

        Raises:
            ValidationError: If the selector declaration is invalid.

        Example:

        .. code-block:: python

            roi_pass.roi_selection(refdes=("J1", "J2"), padding=0.5)
            roi_pass.roi_selection(pins=(("J1", "1"), ("J1", "2")))
            roi_pass.roi_selection(component_types=("CAPACITOR",), count=2)
            roi_pass.roi_selection(
                process_classes=("through_hole",),
                focus="pin",
            )
            roi_pass.roi_selection(all_components=True, focus="pin")
            roi_pass.roi_selection(defects=("shift", "tombstone"))

        """
        selector_values = {
            "refdes": refdes,
            "pins": pins,
            "component_types": component_types,
            "process_classes": process_classes,
            "component_filter": component_filter,
            "all_components": True if all_components else None,
            "defects": defects,
        }
        supplied = [
            name for name, value in selector_values.items() if value is not None
        ]
        if len(supplied) != 1:
            raise ValidationError(
                "capture.roi_selection requires exactly one of refdes, pins, "
                "component_types, process_classes, component_filter, "
                "all_components, or defects",
            )

        selector_name = supplied[0]
        raw_values = selector_values[selector_name]
        if selector_name == "component_filter":
            try:
                values = normalize_component_filter(
                    raw_values,
                    where="capture.roi_selection",
                )
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
        elif selector_name == "all_components":
            values = True
        elif selector_name == "pins":
            try:
                normalized_pins = []
                for value in raw_values:
                    if isinstance(value, str) or len(value) != 2:
                        raise ValueError
                    identity = (str(value[0]).strip(), str(value[1]).strip())
                    if not all(identity):
                        raise ValueError
                    normalized_pins.append(identity)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "capture.roi_selection pins must contain "
                    "(refdes, pin_number) pairs",
                ) from exc
            values = tuple(dict.fromkeys(normalized_pins))
        else:
            if isinstance(raw_values, str):
                raw_values = (raw_values,)
            try:
                values = tuple(
                    dict.fromkeys(str(value).strip() for value in raw_values),
                )
            except TypeError as exc:
                raise ValidationError(
                    f"capture.roi_selection {selector_name} must be an iterable",
                ) from exc
        if selector_name not in {"all_components", "component_filter"} and (
            not values or any(not value for value in values)
        ):
            raise ValidationError(
                f"capture.roi_selection requires {selector_name} values",
            )
        if selector_name == "defects":
            supported_defects = {
                "missing",
                "reverse_polarity",
                "shift",
                "sideflip",
                "tombstone",
            }
            unknown = set(values) - supported_defects
            if unknown:
                raise ValidationError(
                    "capture.roi_selection has unsupported defects: "
                    + ", ".join(sorted(unknown)),
                )
        if count is not None and (not isinstance(count, int) or count < 1):
            raise ValidationError("capture.roi_selection count must be an int >= 1")
        if selector_name in {"refdes", "pins", "all_components"} and count is not None:
            raise ValidationError(
                "capture.roi_selection count is not valid with explicit "
                f"{selector_name}",
            )
        if focus not in {None, "component", "pin"}:
            raise ValidationError(
                "capture.roi_selection focus must be 'component' or 'pin'",
            )
        if selector_name == "pins" and focus == "component":
            raise ValidationError(
                "capture.roi_selection exact pins cannot use component focus",
            )
        if float(padding) < 0.0:
            raise ValidationError("capture.roi_selection padding must be >= 0")

        if selector_name == "pins":
            selector_payload = [list(value) for value in values]
        elif selector_name == "component_filter":
            selector_payload = values
        elif selector_name == "all_components":
            selector_payload = True
        else:
            selector_payload = list(values)
        selection = {
            selector_name: selector_payload,
            "count": count,
            "padding": float(padding),
        }
        if focus is not None:
            selection["focus"] = focus
        self.roi_selection_specs.append(selection)
        self._job._record_declaration(
            self._order_id("roi_selection"),
            "capture.roi_selection.configure",
            {"selections": self.roi_selection_specs},
        )
        return self

    def limits(self, *, triggers=1, max_images=-1):
        """Configure the capture run size.

        This maps to flow ``num_triggers`` and ``max_image_count``;
        ``max_images=-1`` means unlimited.  ``max_images`` counts capture
        positions (image sets) across all triggers: in a ``missing`` pipeline
        each position renders one aligned ``reference``/``defective`` pair,
        and that pair counts once.
        """
        if int(triggers) < 1:
            raise ValidationError("capture.limits: triggers must be >= 1")
        self.limits_spec = {"triggers": int(triggers), "max_images": int(max_images)}
        self._job._record_declaration(
            self._order_id("limits"),
            "capture.limits.configure",
            self.limits_spec,
        )
        return self

    def augmentation(self, *, motion_blur=None):
        """Configure post-render augmentation.

        When never called, compilation emits a disabled ``motion_blur`` block
        because the pipeline reads it unconditionally; probability 0 disables it.
        """
        spec = {}
        if motion_blur is not None:
            mb = dict(motion_blur)
            if "probability" in mb:
                mb["probability"] = _ratio("motion_blur.probability", mb["probability"])
            if "alpha_range" in mb:
                mb["alpha_range"] = _pair("motion_blur.alpha_range", mb["alpha_range"])
            spec["motion_blur"] = mb
        self.augmentation_spec = spec
        self._job._record_declaration(
            self._order_id("augmentation"),
            "capture.augmentation.configure",
            self.augmentation_spec,
        )
        return self

    def outputs(
        self,
        *,
        rgb=True,
        image_format="png",
        bounding_box_2d_tight=False,
        bounding_box_2d_loose=False,
        bounding_box_3d=False,
        semantic_segmentation=False,
        colorize_semantic_segmentation=False,
        instance_id_segmentation=False,
        colorize_instance_id_segmentation=False,
        semantic_types=None,
        frame_padding=4,
        rename_to_grid_index=False,
        **extra_annotators,
    ):
        """Configure writer and annotator switches.

        Extra YAML writer keys, such as ``distance_to_camera``, pass through by name;
        ``semantic_types=None`` defaults to ``[class]`` or the operation
        namespace when ``annotations.enable_operation_labels`` was called.
        """
        spec = {
            "rgb": bool(rgb),
            "image_output_format": validate_image_format(
                image_format,
                name="outputs.image_format",
            ),
            "bounding_box_2d_tight": bool(bounding_box_2d_tight),
            "bounding_box_2d_loose": bool(bounding_box_2d_loose),
            "bounding_box_3d": bool(bounding_box_3d),
            "semantic_segmentation": bool(semantic_segmentation),
            "instance_id_segmentation": bool(instance_id_segmentation),
            "colorize_semantic_segmentation": bool(colorize_semantic_segmentation),
            "colorize_instance_id_segmentation": bool(
                colorize_instance_id_segmentation
            ),
        }
        # Pass-through writer flags (e.g. distance_to_camera) stay by-name so
        # every YAML writer key has a Python spelling.
        for key, value in extra_annotators.items():
            spec[key] = value
        if semantic_types is not None:
            spec["semantic_types"] = list(semantic_types)
        spec["frame_padding"] = int(frame_padding)
        spec["rename_to_grid_index"] = bool(rename_to_grid_index)
        self.outputs_spec = spec
        declaration_spec = dict(self.outputs_spec)
        declaration_spec["image_format"] = declaration_spec.pop(
            "image_output_format",
        )
        self._job._record_declaration(
            self._order_id("outputs"),
            "capture.outputs.configure",
            declaration_spec,
        )
        return self


class CaptureSection(CapturePassSection):
    """Capture configuration, named render passes, and execution."""

    #: Class used for named render passes. A domain capture section sets this
    #: to its own pass class, so ``render_pass("top")`` offers the same
    #: vocabulary as the default pass; otherwise how a capture may be stated
    #: would depend on which of the two you happened to configure.
    _pass_class = CapturePassSection

    def __init__(self, job):
        """Initialize the implicit default pass and named-pass registry."""
        super().__init__(job)
        self._render_passes = {}

    def render_pass(self, name):
        """Create a named render pass that shares the job's prepared scene.

        Args:
            name: Stable name used for JobOrder IDs and the output subdirectory.

        Returns:
            A render-pass section usable directly or as a context manager.

        Raises:
            ValidationError: If the name is invalid, reserved, or already used.

        Example:
            ``with job.capture.render_pass("full_board") as full_board: ...``

        """
        import re

        pass_name = str(name)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", pass_name):
            raise ValidationError(
                "capture.render_pass name must start with a letter and contain "
                "only letters, digits, underscores, or hyphens"
            )
        if pass_name == "default":
            raise ValidationError("capture.render_pass name 'default' is reserved")
        if pass_name in self._render_passes:
            raise ValidationError(f"capture render pass {pass_name!r} already exists")
        render_pass = self._pass_class(
            self._job,
            name=pass_name,
            order_prefix=f"capture.{pass_name}",
        )
        self._render_passes[pass_name] = render_pass
        return render_pass

    def _render_resolution(self):
        """Return a launch resolution large enough for all configured passes."""
        renderer_specs = [
            render_pass.renderer_spec
            for render_pass in (self, *self._render_passes.values())
            if isinstance(render_pass.renderer_spec, dict)
        ]
        if not renderer_specs:
            raise ValidationError("capture.render requires a configured render pass")
        resolutions = [spec.get("resolution") for spec in renderer_specs]
        if any(resolution is None for resolution in resolutions):
            raise ValidationError(
                "capture.renderer has no resolution: give renderer.resolution, "
                "or state the frame size in the form that derives it"
            )
        if any(
            not isinstance(resolution, (list, tuple)) or len(resolution) != 2
            for resolution in resolutions
        ):
            raise ValidationError("capture.renderer resolution must contain two values")
        return (
            max(int(resolution[0]) for resolution in resolutions),
            max(int(resolution[1]) for resolution in resolutions),
        )

    def render(
        self,
        output_dir: str | Path | None = None,
        *,
        serialization_path: str | Path | None = None,
        windowed: bool = False,
        experience: str | Path | None = None,
    ) -> JobResult:
        """Execute the job through an isolated Isaac Sim Python process.

        Ordinary Python serializes the same immutable JobOrder plan used by
        :meth:`render_async`. Isaac Sim executes that plan directly without
        generating legacy YAML or invoking ``sdg_pipeline.py``.

        Args:
            output_dir: Directory or supported remote prefix receiving artifacts.
                Required when at least one render pass is configured.
            serialization_path: Optional final composed-stage USD destination.
            windowed: Whether to display the Isaac application window.
            experience: Optional Kit experience file for the Isaac child. A
                cold render does not benefit from `render_batch`, so it is the
                case this is for. See
                :meth:`~cdf.job.CDFJob.execute`.

        Returns:
            Structured result transported from the Isaac JobOrder executor.

        Raises:
            ValidationError: If neither capture nor final-stage serialization is
                configured, or if their destinations are inconsistent.

        """
        if not self._job._has_capture_declarations() and serialization_path is None:
            raise ValidationError(
                "capture.render requires a render pass or serialization_path",
            )
        return self._job.execute(
            output_dir=output_dir,
            serialization_path=serialization_path,
            windowed=windowed,
            experience=experience,
        )

    async def render_async(
        self,
        output_dir: str | Path | None = None,
        *,
        executor: JobExecutor,
        serialization_path: str | Path | None = None,
    ) -> JobResult:
        """Compile JobOrders and execute them without the legacy YAML path.

        The caller must already be running inside the environment required by
        the selected executor.

        Args:
            output_dir: Directory that receives capture artifacts when rendering.
            executor: Public JobExecutor selected by the caller.
            serialization_path: Optional final composed-stage USD destination.

        Returns:
            Structured JobOrder execution result.

        Raises:
            ValidationError: If ``executor`` does not implement the public
                executor contract or returns an invalid result.
            JobExecutionError: If an order fails after execution begins.

        Example:

        .. code-block:: python

            from cdf import create_default_isaac_executor

            result = await job.capture.render_async(
                output_dir="output",
                executor=create_default_isaac_executor(),
            )
            print(result.success)
            # True

        """
        from .execution.executor_base import JobExecutor
        from .execution.job_result_types import JobResult

        if not isinstance(executor, JobExecutor):
            raise ValidationError(
                "capture.render_async executor must implement JobExecutor",
            )
        if not self._job._has_capture_declarations() and serialization_path is None:
            raise ValidationError(
                "capture.render_async requires a render pass or serialization_path",
            )
        self._job._validate_execution_outputs(output_dir, serialization_path)
        plan = self._job.compile_orders(
            output_dir=output_dir,
            serialization_path=serialization_path,
        )
        result = await executor.execute_async(plan)
        if not isinstance(result, JobResult):
            raise ValidationError(
                "capture.render_async executor must return JobResult",
            )
        return result


class PostprocessSection:
    """Postprocess — optional cropping and other post-capture transforms."""

    def __init__(self, job):
        """Initialize postprocess declarations for a job."""
        self._job = job
        self.declared_crop = None
        self.declared_stitch = None
        self.declared_stitches = []
        self.declared_closures = []

    @staticmethod
    def _crop_rules():
        """Load the shared crop validation rules.

        The stdlib-only ``scripts/crop/crop_validation.py`` rules are also
        enforced by ``pipeline_adapter.py`` and reached through the same
        ``sys.path`` mechanism used by ``plan.execute_crop``.
        """
        import sys

        from . import plan as _plan  # local import keeps section import light

        if str(_plan._CROP_DIR) not in sys.path:
            sys.path.insert(0, str(_plan._CROP_DIR))
        import crop_validation

        return crop_validation

    def _crop_block(
        self,
        method,
        types,
        offset,
        min_coverage,
        class_filter,
        xform_depth,
        output_subdir,
        mode,
        input_subdir,
        reference_subdir,
    ):
        rules = self._crop_rules()
        method_error = rules.crop_method_error(method)
        if method_error:
            raise ValidationError(f"postprocess.crop: {method_error}")
        types_list = list(types) if types else ["rgb", rules.DEFAULT_SEMANTIC_TYPE]
        types_error = rules.crop_types_error(types_list)
        if types_error:
            raise ValidationError(f"postprocess.crop: {types_error}")
        block = {
            "enabled": True,
            "method": method,
            "types": types_list,
            "offset": int(offset),
            "min_coverage": min_coverage,
            "class_filter": class_filter,
            "xform_depth": int(xform_depth),
            "output_subdir": output_subdir,
        }
        if mode is not None:
            block["mode"] = mode
            if not (input_subdir and reference_subdir):
                raise ValidationError(
                    f"postprocess.crop: mode={mode!r} requires input_subdir and "
                    "reference_subdir"
                )
            block["input_subdir"] = input_subdir
            block["reference_subdir"] = reference_subdir
        return block

    def crop(
        self,
        *,
        source=None,
        method="bbox_tight",
        types=None,
        offset=10,
        min_coverage=0.2,
        class_filter=None,
        xform_depth=7,
        output_subdir="cropped",
        mode=None,
        input_subdir=None,
        reference_subdir=None,
        dry_run=False,
        runner=None,
    ):
        """Declare a crop or invoke the legacy post-hoc compatibility path.

        Without ``source``, the crop is recorded as a JobOrder that runs after
        capture. Passing a legacy ``RenderResult`` invokes the retained
        post-hoc adapter against existing captured artifacts.
        """
        block = self._crop_block(
            method,
            types,
            offset,
            min_coverage,
            class_filter,
            xform_depth,
            output_subdir,
            mode,
            input_subdir,
            reference_subdir,
        )
        if source is None:
            self.declared_crop = block
            self._job._record_declaration(
                "postprocess.crop",
                "postprocess.crop",
                {
                    key: value
                    for key, value in self.declared_crop.items()
                    if key != "enabled"
                },
            )
            return None
        from . import plan as _plan

        return _plan.execute_crop(source, block, dry_run=dry_run, runner=runner)

    def close_openings(
        self,
        *,
        source_pass: str = "default",
        class_name: str,
        output_subdir: str = "stitched",
        output_filename: str = "rgb.png",
        max_diameter_px: int = 48,
        min_circularity: float = 0.55,
    ) -> None:
        """Fill closed rings of one class with that same class.

        Geometry that encloses a void projects to a rim rather than a disc, so
        a drilled hole is labelled only around its edge and the opening itself
        stays unlabelled. This recovers the opening from the rim that was
        actually rendered. Only unlabelled pixels are claimed, so anything
        inside the opening keeps its own class, and a rim that does not close
        is left alone rather than guessed at.

        It runs on the STITCHED mask, not on the frames: a hole lying across a
        tile boundary has an incomplete rim in every frame that sees only part
        of it, so closing per frame recovers a fraction of the openings and
        leaves mismatched fragments wherever tiles overlap. The stitch that
        produces the mask is found by the ``source_pass``, ``output_subdir``
        and ``output_filename`` the two of them share, so declaration order
        does not matter and a missing or ambiguous producer is a compilation
        error rather than a silently wrong plan.

        Args:
            source_pass: Named capture pass, or ``default`` for legacy capture.
            class_name: Class whose rings are closed.
            output_subdir: Directory the stitch wrote into.
            output_filename: Stitched mask filename.
            max_diameter_px: Largest opening accepted, in pixels.
            min_circularity: Least filled-area-to-enclosing-disc ratio accepted.

        Raises:
            ValidationError: If a name is empty or a bound is out of range.

        Example:

        .. code-block:: python

            job.postprocess.close_openings(
                source_pass="full_board",
                class_name="hole",
                output_filename="panel_mask.png",
            )

        """
        for name, value in (
            ("source_pass", source_pass),
            ("class_name", class_name),
            ("output_subdir", output_subdir),
            ("output_filename", output_filename),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(
                    f"postprocess.close_openings: {name} must be a name",
                )
        if not isinstance(max_diameter_px, int) or max_diameter_px < 1:
            raise ValidationError(
                "postprocess.close_openings: max_diameter_px must be a "
                "positive whole number of pixels",
            )
        if not 0.0 < float(min_circularity) <= 1.0:
            raise ValidationError(
                "postprocess.close_openings: min_circularity must fall in (0, 1]",
            )
        values = {
            "source_pass": source_pass,
            "class_name": class_name,
            "output_subdir": output_subdir,
            "output_filename": output_filename,
            "max_diameter_px": int(max_diameter_px),
            "min_circularity": float(min_circularity),
        }
        self.declared_closures.append(values)
        self._job._record_declaration(
            f"postprocess.close_openings.{source_pass}.{class_name}",
            "postprocess.close_openings",
            values,
        )

    def stitch(
        self,
        *,
        source_pass: str = "default",
        image_type: str = "rgb",
        output_subdir: str = "stitched",
        output_filename: str = "rgb.png",
        blend: str = "none",
        write_labels: bool = False,
        recolor_tiles: bool = False,
        color_map: dict | None = None,
    ) -> None:
        """Stitch one orthographic scan-grid pass after capture.

        The capture pass supplies canonical pixel-layout metadata, so this
        operation remains independent of rendering and camera execution.

        Args:
            source_pass: Named capture pass, or ``default`` for legacy capture.
            image_type: BasicWriter image prefix to stitch.
            output_subdir: Directory created beneath the capture-pass output.
            output_filename: Filename used for a single-trigger result.
            blend: Overlap policy. ``none`` pastes tiles, which leaves a visible
                join wherever the scan step equals the camera footprint;
                ``feather`` cross-fades across the overlap the pixel layout
                implies. Author a camera aperture wider than the cell pitch to
                create that overlap -- the orthographic auto-fit only ever grows
                the aperture, so the step stays ``bbox / count``.
            write_labels: When ``True`` and ``blend`` is not ``labels``, merge
                the per-tile label sidecars and write a ``*_labels.json`` file
                alongside the stitched image without remapping pixel colors.
                Emits a ``RuntimeWarning`` if tiles use different colors for
                the same class; use ``blend='labels'`` in that case.
                Cannot be combined with ``blend='labels'``.
            recolor_tiles: When ``True`` with ``blend='labels'``, rewrite each
                source tile and its label sidecar with the unified label colors.
            color_map: Optional mapping of class name to an RGBA 4-tuple
                ``(R, G, B, A)`` with values in ``[0, 255]``.  Only applies
                when ``blend='labels'``.  Named classes use the supplied color
                instead of the hash-derived assignment.  BACKGROUND and
                UNLABELLED cannot be overridden.

        Raises:
            ValidationError: If a name is empty or contains a path separator.

        Example:

        .. code-block:: python

            job.postprocess.stitch(
                source_pass="full_board",
                output_filename="panel.png",
            )

        """
        string_values = {
            "source_pass": source_pass,
            "image_type": image_type,
            "output_subdir": output_subdir,
            "output_filename": output_filename,
            "blend": blend,
        }
        if not all(
            isinstance(value, str) and value.strip() for value in string_values.values()
        ):
            raise ValidationError("postprocess.stitch values must be non-empty strings")
        if not isinstance(write_labels, bool):
            raise ValidationError("postprocess.stitch write_labels must be a boolean")
        if not isinstance(recolor_tiles, bool):
            raise ValidationError("postprocess.stitch recolor_tiles must be a boolean")
        if "/" in output_subdir or "\\" in output_subdir:
            raise ValidationError(
                "postprocess.stitch output_subdir must be a single directory name",
            )
        if "/" in output_filename or "\\" in output_filename:
            raise ValidationError(
                "postprocess.stitch output_filename must be a simple filename",
            )
        if blend not in ("none", "feather", "labels"):
            raise ValidationError(
                "postprocess.stitch blend must be none, feather, or labels "
                f"(got {blend!r})",
            )
        if write_labels and blend == "labels":
            raise ValidationError(
                "postprocess.stitch write_labels may not be combined with "
                "blend='labels' (labels mode already writes a remapped label sidecar)",
            )
        if recolor_tiles and blend != "labels":
            raise ValidationError(
                "postprocess.stitch recolor_tiles requires blend='labels'",
            )
        if color_map is not None:
            if not isinstance(color_map, dict):
                raise ValidationError("postprocess.stitch color_map must be a dict")
            if blend != "labels":
                raise ValidationError(
                    "postprocess.stitch color_map only applies when blend='labels'",
                )
            for name, rgba in color_map.items():
                if not isinstance(name, str):
                    raise ValidationError(
                        "postprocess.stitch color_map keys must be strings",
                    )
                if (
                    not isinstance(rgba, (list, tuple))
                    or len(rgba) != 4
                    or not all(isinstance(c, int) and 0 <= c <= 255 for c in rgba)
                ):
                    raise ValidationError(
                        f"postprocess.stitch color_map[{name!r}] must be an RGBA "
                        "4-tuple of integers in [0, 255]",
                    )
        values = {
            **string_values,
            "write_labels": write_labels,
            "recolor_tiles": recolor_tiles,
            "color_map": color_map,
        }
        # a job may stitch several products of one pass -- typically the image
        # and its segmentation -- so each declaration gets its own order id
        suffix = f".{source_pass}.{image_type}" if self.declared_stitches else ""
        self.declared_stitch = values
        self.declared_stitches.append(values)
        self._job._record_declaration(
            f"postprocess.stitch{suffix}",
            "postprocess.stitch",
            values,
        )

    def format_roi_output(
        self,
        *,
        rgb_pass: str = "full_board",
        roi_pass: str = "full_board_roi",
    ) -> None:
        """Format capture outputs to the AOI ROI output contract.

        For each trigger, renames ``semantic_segmentation_x*_y*.png`` to
        ``semantic_mask_x*_y*.png``, merges per-tile label JSONs into a
        single ``semantic_segmentation_labels.json`` (retaining the five
        required geometry classes, optional ``pin_tip``, and declared defect
        classes), preserves each pass's schema-rich ``metadata.json``, and
        runs RGB/ROI acceptance checks.

        **Capture-pass requirement**: both ``rgb_pass`` and ``roi_pass`` must
        have ``rename_to_grid_index: true`` set in their ``outputs`` block.
        File matching relies on the ``x<col>_y<row>`` suffix that option
        produces; without it no tiles are found and a ``ValidationError`` is
        raised.

        Args:
            rgb_pass: Named capture pass that produced the RGB tiles.
            roi_pass: Named capture pass that produced the semantic mask tiles.

        Raises:
            ValidationError: If either pass name is empty.

        Example:

        .. code-block:: python

            job.postprocess.format_roi_output(
                rgb_pass="full_board",
                roi_pass="full_board_roi",
            )

        """
        if not (isinstance(rgb_pass, str) and rgb_pass.strip()):
            raise ValidationError(
                "postprocess.format_roi_output: rgb_pass must be non-empty"
            )
        if not (isinstance(roi_pass, str) and roi_pass.strip()):
            raise ValidationError(
                "postprocess.format_roi_output: roi_pass must be non-empty"
            )
        self._job._record_declaration(
            f"postprocess.format_roi_output.{rgb_pass}.{roi_pass}",
            "postprocess.format_roi_output",
            {"rgb_pass": rgb_pass, "roi_pass": roi_pass},
        )

    def generate_joint_rois(
        self,
        *,
        roi_pass: str = "full_board_roi",
        rgb_pass: str = "full_board",
    ) -> None:
        """Generate ``joint_rois.json`` from tight bounding box outputs.

        Reads ``bounding_box_2d_tight_x*_y*.npy`` files in the ROI trigger
        directories, transforms per-tile pixel coordinates to full-board
        canvas coordinates using each tile's ``dst_rect_xywh_px`` from the
        pixel layout, unions bboxes that span multiple tiles for the same
        component lead, and writes ``joint_rois.json`` to each ROI trigger
        directory.

        **Capture-pass requirement**: ``roi_pass`` must have
        ``rename_to_grid_index: true`` set in its ``outputs`` block.  File
        matching relies on the ``x<col>_y<row>`` suffix that option produces;
        without it no bounding-box files are found and the output contains
        zero joints.

        Args:
            roi_pass: Named capture pass that produced the bounding box files.
            rgb_pass: Source pass name recorded in ``joint_rois.json``.

        Raises:
            ValidationError: If either pass name is empty.

        Example:

        .. code-block:: python

            job.postprocess.generate_joint_rois(
                roi_pass="full_board_roi",
                rgb_pass="full_board",
            )

        """
        if not (isinstance(roi_pass, str) and roi_pass.strip()):
            raise ValidationError(
                "postprocess.generate_joint_rois: roi_pass must be non-empty"
            )
        if not (isinstance(rgb_pass, str) and rgb_pass.strip()):
            raise ValidationError(
                "postprocess.generate_joint_rois: rgb_pass must be non-empty"
            )
        self._job._record_declaration(
            f"postprocess.generate_joint_rois.{roi_pass}",
            "postprocess.generate_joint_rois",
            {"roi_pass": roi_pass, "rgb_pass": rgb_pass},
        )
