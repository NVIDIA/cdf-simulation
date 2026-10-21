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

"""Shared types and normalization helpers for JobConfig sections."""

import os
import posixpath
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict


Number = int | float
RangeValue = Number | tuple[Number, Number]


class ConfigModel(BaseModel):
    """Strict base model shared by all JobConfig blocks."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


def expand_range(value: RangeValue) -> tuple[Number, Number]:
    """Expand a scalar into the constant range expected by the Python API."""
    if isinstance(value, tuple):
        return value
    return value, value


def resolve_path(path: str | os.PathLike[str], base_dir: str | os.PathLike[str]) -> str:
    """Resolve a local or S3 config-relative path without requiring it to exist."""
    value = os.fspath(path)
    if "://" in value:
        return value

    local_path = Path(value).expanduser()
    if local_path.is_absolute():
        return str(local_path.resolve())

    base = os.fspath(base_dir)
    if base.startswith("s3://"):
        parsed = urlparse(base)
        key = posixpath.normpath(
            posixpath.join(parsed.path.lstrip("/"), value.replace("\\", "/")),
        )
        return f"s3://{parsed.netloc}/{key}"

    return str((Path(base).expanduser() / local_path).resolve())
