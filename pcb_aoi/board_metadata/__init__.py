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

"""Source-neutral PCB metadata for USD and legacy sidecar consumers."""

from .catalog_json_writer import (
    CATALOG_FORMAT,
    CATALOG_FORMAT_VERSION,
    catalog_to_json_data,
    write_catalog_json,
)
from .component_filter import select_components
from .metadata_catalog import (
    HOLE_ASSOCIATION_ORIGIN_KEY,
    BoardMetadataCatalog,
    BoardRecord,
    ComponentRecord,
    Diagnostic,
    HoleRecord,
    MetadataCapability,
    MetadataOrigin,
    MetadataSource,
    NetRecord,
    PadRecord,
    PasteDepositRecord,
    PrototypeRecord,
    TerminalRecord,
)
from .metadata_provider import BoardMetadataProvider, BoardMetadataRequest
from .sidecar import SidecarMetadataProvider
from .usd_schema_metadata_keys import MESH_ROLE_ATTR

__all__ = [
    "CATALOG_FORMAT",
    "CATALOG_FORMAT_VERSION",
    "HOLE_ASSOCIATION_ORIGIN_KEY",
    "MESH_ROLE_ATTR",
    "BoardMetadataCatalog",
    "BoardMetadataProvider",
    "BoardMetadataRequest",
    "BoardRecord",
    "ComponentRecord",
    "Diagnostic",
    "HoleRecord",
    "MetadataCapability",
    "MetadataOrigin",
    "MetadataSource",
    "NetRecord",
    "PadRecord",
    "PasteDepositRecord",
    "PrototypeRecord",
    "SidecarMetadataProvider",
    "TerminalRecord",
    "catalog_to_json_data",
    "select_components",
    "write_catalog_json",
]
