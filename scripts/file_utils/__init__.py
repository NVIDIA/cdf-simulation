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

"""Expose backend-neutral file access helpers.

``open``, ``exists``, and ``is_file`` cover individual paths or objects.
``localize_tree`` and ``output_tree`` adapt tools that require local filesystem
paths or directories.
"""

from .api import exists, is_file, is_remote, localize_tree, open, output_tree
from .archives import extract_zip
from .path_safety import confined_path, safe_path_segment

__all__ = [
    "confined_path",
    "exists",
    "extract_zip",
    "is_file",
    "is_remote",
    "localize_tree",
    "open",
    "output_tree",
    "safe_path_segment",
]
