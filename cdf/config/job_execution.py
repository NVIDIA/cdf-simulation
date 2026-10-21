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

"""Execution-output configuration shared by YAML and Python APIs."""

from pydantic import field_validator

from ._common import ConfigModel

EXECUTION_API_CONFIG_FIELDS = {
    "output_dir": "output_dir",
    "serialization_path": "execution.serialization_path",
}


class ExecutionConfig(ConfigModel):
    """Optional outputs produced by whole-job execution."""

    serialization_path: str | None = None

    @field_validator("serialization_path")
    @classmethod
    def validate_serialization_path(cls, value: str | None) -> str | None:
        """Require final-stage output to use a supported USD extension."""
        if value is not None and not value.lower().endswith((".usd", ".usda", ".usdc")):
            raise ValueError(
                "serialization_path must end with .usd, .usda, or .usdc",
            )
        return value
