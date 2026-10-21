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

"""Compatibility result records for post-hoc processing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RenderResult:
    """Describe an existing capture supplied to post-hoc processing."""

    output_dir: str
    pipeline_type: str
    flow_config: dict = field(default_factory=dict)
    pcba_config: dict = field(default_factory=dict)
    flow_config_path: str = ""
    pcba_config_path: str = ""
    command: list = field(default_factory=list)
    executed: bool = False
    returncode: int = None


@dataclass
class CropResult:
    """Post-hoc crop execution result."""

    output_dir: str
    config: dict = field(default_factory=dict)
    executed: bool = False
