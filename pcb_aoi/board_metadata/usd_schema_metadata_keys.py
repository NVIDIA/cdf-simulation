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

"""Define ordered USD metadata keys used by board-metadata consumers.

Each tuple lists the expected schema key first and compatibility fallbacks after
it. Keep schema-name changes in this registry rather than distributing aliases
through metadata readers and downstream geometry adapters.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

USD_SCHEMA_METADATA_KEYS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "board_feature.component_relationship": ("pcba:board:component",),
        "board_feature.refdes": (
            "pcba:board:refdes",
            "userProperties:refdes",
        ),
        "board_feature.role": ("pcba:mesh:role",),
        "hole.center_mm": ("pcba:board:holeCenterMm",),
        "hole.dimensions_mm": ("pcba:board:holeDimsMm",),
        "hole.shape": ("pcba:board:holeShape",),
        "hole.terminal_number": ("pcba:board:componentPinNumber",),
        "pad.contour": ("pcba:board:padContour",),
        "pad.terminal_number": (
            "pcba:board:componentPinNumber",
            "pcba:board:padNumber",
        ),
        "pad.type": ("userProperties:padType",),
        "paste.refdes": (
            "pcba:board:refdes",
            "userProperties:refdes",
        ),
        "paste.site_kind": ("userProperties:siteKind",),
        "paste.terminal_number": (
            "pcba:board:componentPinNumber",
            "userProperties:padNumber",
        ),
        "pin.cross_section": ("pcba:pin:crossSection",),
        "site.kind": ("userProperties:siteKind",),
    },
)


#: The mesh-role attribute as a standalone constant for GEOMETRY readers
#: (pkg_body silhouettes, board_substrate scans, board meshes by role).
#: Sourced from the registry above so a schema rename stays a one-line edit.
MESH_ROLE_ATTR = USD_SCHEMA_METADATA_KEYS["board_feature.role"][0]
