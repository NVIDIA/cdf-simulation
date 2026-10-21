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

"""Provider protocol and input request for canonical PCB metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .metadata_catalog import BoardMetadataCatalog


@dataclass(frozen=True)
class BoardMetadataRequest:
    """Inputs available to a board metadata provider.

    Args:
        board_root: Absolute board-root path to read, if explicitly selected.
        source_identifier: Stable path or label used for source provenance.
        sidecar_data: Decoded sidecar mapping, when using a sidecar provider.
        stage: Open composed USD stage, when using a USD schema provider.

    """

    board_root: str | None = None
    source_identifier: str | None = None
    sidecar_data: Mapping[str, object] | None = None
    stage: object | None = None


@runtime_checkable
class BoardMetadataProvider(Protocol):
    """Source adapter that produces a source-neutral metadata catalog."""

    def load(self, request: BoardMetadataRequest) -> BoardMetadataCatalog:
        """Decode the requested source into an immutable catalog.

        Args:
            request: Provider inputs and source identity.

        Returns:
            Immutable catalog populated from the requested source.

        Raises:
            ValidationError: If the source is unavailable, malformed, or
                unsupported by the provider.

        """
