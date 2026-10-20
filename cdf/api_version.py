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

"""Expose the authoring API version independently of package releases."""

from __future__ import annotations

import re
from pathlib import Path


def _load_api_contract() -> tuple[str, str]:
    """Read the authoritative API version and status from `VERSION.md`."""
    version_path = Path(__file__).resolve().parent / "VERSION.md"
    contents = version_path.read_text(encoding="utf-8")
    version_match = re.search(
        r"^Current API version: `([0-9]+\.[0-9]+\.[0-9]+)`$",
        contents,
        flags=re.MULTILINE,
    )
    status_match = re.search(
        r"^Current API status: `(unreleased|released)`$",
        contents,
        flags=re.MULTILINE,
    )
    if version_match is None or status_match is None:
        raise RuntimeError(f"invalid CDF API version file: {version_path}")
    return version_match.group(1), status_match.group(1)


_api_version, _api_status = _load_api_contract()
#: Current authoring and JobConfig contract version.
CDF_API_VERSION = _api_version
#: Publication status of the current authoring and JobConfig contract.
CDF_API_STATUS = _api_status
