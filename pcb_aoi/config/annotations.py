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

"""Annotation configuration models and Python API interpretation."""

from __future__ import annotations

from pydantic import model_validator

from cdf.component_filter import ComponentFilter
from cdf.config._common import ConfigModel
from cdf.job import CDFJob
from pcb_aoi.config.scene import (
    ComponentMetadataSource,
    component_metadata_api_source,
)


class AnnotationsGeometryConfig(ConfigModel):
    """Class names for each board geometry target, with an optional component filter.

    Each target field maps to the semantic class that Replicator will assign.
    ``barrels`` labels the cylindrical copper inner wall of drilled holes
    (``pcba:mesh:role = "board_barrel"`` prims); set it to the same class as
    ``pads`` to eliminate the unlabeled ring between pad and hole in semantic
    segmentation.  ``body`` labels the package body mesh
    (``pcba:mesh:role = "pkg_body"``) and ``marking`` the package's pin-1 dot /
    polarity band / notch mesh (``pkg_marking``), so a mask can separate the
    polarity marker from the body it sits on.  ``components`` restricts
    labeling to pads, terminals, holes, barrels, bodies and markings that
    belong to the listed refdes components; the full board is labeled when
    omitted.
    """

    solder: str | None = None
    terminals: str | None = None
    pads: str | None = None
    holes: str | None = None
    barrels: str | None = None
    body: str | None = None
    marking: str | None = None
    components: list[str] | None = None
    component_filter: ComponentFilter | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> AnnotationsGeometryConfig:
        """Reject mixing normalized and legacy selectors.

        Returns:
            Validated geometry annotation settings.

        Raises:
            ValueError: If both selector forms are supplied.

        """
        if self.components is not None and self.component_filter is not None:
            raise ValueError("components and component_filter are mutually exclusive")
        return self


class AnnotationsConfig(ConfigModel):
    """Component-semantic and operation-label annotation settings."""

    components: bool = False
    component_classes: dict[str, list[str]] | None = None
    operation_labels: str | None = None
    geometry: AnnotationsGeometryConfig | None = None


def apply_annotations(
    job: CDFJob,
    config: AnnotationsConfig,
    *,
    component_metadata_source: ComponentMetadataSource,
) -> None:
    """Apply annotation declarations from config to the job.

    Args:
        job: Mutable job being built.
        config: Parsed annotation settings.
        component_metadata_source: Resolved metadata source for component
            label declarations.

    """
    if config.components:
        job.annotations.label_components(
            source=component_metadata_api_source(component_metadata_source),
            classes=config.component_classes,
        )
    if config.geometry:
        g = config.geometry
        labels = {
            k: v
            for k, v in {
                "solder": g.solder,
                "terminals": g.terminals,
                "pads": g.pads,
                "holes": g.holes,
                "barrels": g.barrels,
                "body": g.body,
                "marking": g.marking,
            }.items()
            if v is not None
        }
        if labels:
            kwargs: dict[str, object] = {}
            if g.components is not None:
                kwargs["components"] = g.components
            if g.component_filter is not None:
                kwargs["component_filter"] = g.component_filter
            job.annotations.label_geometry(**labels, **kwargs)
    if config.operation_labels is not None:
        job.annotations.enable_operation_labels(namespace=config.operation_labels)
