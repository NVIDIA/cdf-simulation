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

"""Scene configuration models and Python API interpretation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from pydantic import model_validator

from cdf.component_filter import ComponentFilter
from cdf.config._common import ConfigModel, Number, resolve_path
from cdf.job import CDFJob

SCENE_API_CONFIG_FIELDS = {
    "configure": (
        "usd",
        "root",
        "camera",
        "camera_xform",
        "ring_light",
        "horizontal_aperture",
        "meters_per_unit",
    ),
    "instances": ("instances",),
    "attach_sidecar": ("components",),
    "select_components": ("components",),
}

ComponentMetadataSource = Literal["embedded_usd", "legacy_sidecar"]


@dataclass(frozen=True)
class AppliedScene:
    """Resolved scene metadata selection for downstream config adapters."""

    component_metadata_source: ComponentMetadataSource
    #: Source category selected by the scene declaration.

    selected_types: tuple[str, ...] | None
    #: Component types selected while applying the scene.


def component_metadata_api_source(source: ComponentMetadataSource) -> str:
    """Return the existing Python API source name for a metadata category."""
    if source == "embedded_usd":
        return "usd"
    return "components"


class ComponentsConfig(ConfigModel):
    """Component-metadata source and optional type-selection settings."""

    sidecar: str | None = None
    types: tuple[str, ...] | None = None
    component_filter: ComponentFilter | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> ComponentsConfig:
        """Reject mixing normalized and overloaded component selectors.

        Returns:
            Validated component settings.

        Raises:
            ValueError: If both selector forms are supplied.

        """
        if self.types is not None and self.component_filter is not None:
            raise ValueError("types and component_filter are mutually exclusive")
        return self


class SceneConfig(ConfigModel):
    """USD scene, PCB root, rig-path, and component-metadata settings."""

    usd: str
    root: str
    camera: str | None = None
    camera_xform: str | None = None
    ring_light: str | None = None
    horizontal_aperture: Number | None = None
    #: Metres per scene unit -- the same direction USD states ``metersPerUnit``
    #: in, so ``0.001`` for a millimetre stage -- when the stage's own
    #: declaration is wrong. The reciprocal is a plausible-looking number too,
    #: and stating it backwards frames the capture wrong by the square of the
    #: scale with nothing in the pipeline to contradict it.
    #: Only a capture stated in millimetres -- ``capture.fov`` -- reads this;
    #: everything else works in scene units and never converts. Set it when a
    #: stage declares a scale its geometry does not use, which shows up as an
    #: implausible board size (for example a PCB metres across).
    meters_per_unit: Number | None = None
    instances: tuple[str, ...] | None = None
    components: ComponentsConfig | None = None


def apply_scene(
    job: CDFJob,
    config: SceneConfig,
    base_dir: str | os.PathLike[str],
) -> AppliedScene:
    """Apply scene declarations and return the resolved metadata selection."""
    job.scene.configure(
        usd_path=resolve_path(config.usd, base_dir),
        root_path=config.root,
        camera_path=config.camera,
        camera_xform_path=config.camera_xform,
        ring_light_root=config.ring_light,
        horizontal_aperture=config.horizontal_aperture,
        meters_per_unit=config.meters_per_unit,
    )
    if config.instances is not None:
        job.scene.instances(prim_paths=config.instances)
    if config.components is None:
        return AppliedScene(
            component_metadata_source="embedded_usd",
            selected_types=None,
        )
    selected_types = config.components.types
    if config.components.sidecar is not None:
        job.scene.attach_sidecar(
            name="components",
            kind="pcb.components",
            path=resolve_path(config.components.sidecar, base_dir),
        )
        job.scene.select_components(
            source="components" if selected_types is None else None,
            types=selected_types,
            component_filter=config.components.component_filter,
        )
        component_metadata_source: ComponentMetadataSource = "legacy_sidecar"
    else:
        job.scene.select_components(
            source="usd",
            types=selected_types,
            component_filter=config.components.component_filter,
        )
        component_metadata_source = "embedded_usd"
    return AppliedScene(
        component_metadata_source=component_metadata_source,
        selected_types=selected_types,
    )
