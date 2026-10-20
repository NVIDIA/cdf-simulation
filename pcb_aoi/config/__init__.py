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

"""Register PCB-AOI's versioned JobConfig extensions."""

from cdf.config.domain_adapters import DomainConfigAdapter, register_domain_config

from .annotations import AnnotationsConfig, apply_annotations
from .appearance import AppearanceConfig, apply_appearance
from .capture import CaptureConfig, apply_capture
from .operations import OperationsConfig, apply_operations
from .postprocess import PostprocessConfig, apply_postprocess
from .scene import SceneConfig, apply_scene


def register() -> None:
    """Register PCB-AOI's operation schema and interpreter with CDF."""
    register_domain_config(
        "pcb-aoi",
        DomainConfigAdapter(
            scene_model=SceneConfig,
            apply_scene=apply_scene,
            appearance_model=AppearanceConfig,
            apply_appearance=apply_appearance,
            annotations_model=AnnotationsConfig,
            apply_annotations=apply_annotations,
            operations_model=OperationsConfig,
            apply_operations=apply_operations,
            capture_model=CaptureConfig,
            apply_capture=apply_capture,
            postprocess_model=PostprocessConfig,
            apply_postprocess=apply_postprocess,
        ),
    )


__all__ = [
    "AnnotationsConfig",
    "AppearanceConfig",
    "CaptureConfig",
    "OperationsConfig",
    "PostprocessConfig",
    "SceneConfig",
    "apply_annotations",
    "apply_appearance",
    "apply_capture",
    "apply_operations",
    "apply_postprocess",
    "apply_scene",
    "register",
]
