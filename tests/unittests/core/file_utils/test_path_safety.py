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

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_FILE_UTILS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "file_utils"
if str(_FILE_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_FILE_UTILS_DIR))

from path_safety import confined_path, safe_path_segment  # noqa: E402


class OutputPathSafetyTest(unittest.TestCase):
    """Tests for output path safety helpers."""

    def test_untrusted_segments_become_safe_single_names(self):
        self.assertEqual(safe_path_segment("side flip", "fallback"), "side_flip")
        self.assertEqual(safe_path_segment("../../escape", "fallback"), "escape")
        self.assertEqual(safe_path_segment(r"..\..\escape", "fallback"), "escape")
        expected_reserved_name = "_CON" if os.name == "nt" else "CON"
        self.assertEqual(safe_path_segment("CON", "fallback"), expected_reserved_name)
        self.assertEqual(safe_path_segment("...", "fallback"), "fallback")

    def test_confined_path_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                confined_path(Path(temp_dir) / "output", "..", "escaped.png")


if __name__ == "__main__":
    unittest.main()
