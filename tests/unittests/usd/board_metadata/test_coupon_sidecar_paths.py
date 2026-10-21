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

"""Every checked-in component sidecar must resolve against its coupon USD.

A sidecar names absolute prim paths (board root, component prototypes,
instances, bodies, solder contacts). If the coupon's root prim is renamed and
the sidecar is not, or the other way round, stage-backed component and solder
discovery silently finds nothing. This test opens each pair and asserts the
paths exist.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from pxr import Usd

_RESOURCES = Path(__file__).resolve().parents[4] / "tests" / "functional" / "resources"


def _sidecar_pairs():
    for sidecar in sorted(_RESOURCES.glob("*.components.json")):
        usd = sidecar.with_name(sidecar.name.replace(".components.json", ".usdc"))
        if usd.is_file():
            yield sidecar, usd


@unittest.skipUnless(
    any(_sidecar_pairs()),
    "approved SMT coupon fixture is not shipped in the public release",
)
class CouponSidecarPathsTest(unittest.TestCase):
    """Sidecar prim paths and the coupon stage agree."""

    def test_at_least_one_pair_is_checked(self):
        self.assertGreaterEqual(len(list(_sidecar_pairs())), 2)

    def test_sidecar_paths_exist_on_stage(self):
        for sidecar, usd in _sidecar_pairs():
            with self.subTest(coupon=usd.name):
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                stage = Usd.Stage.Open(str(usd))
                missing = []

                def check(path, stage=stage, missing=missing):
                    if path and not stage.GetPrimAtPath(path).IsValid():
                        missing.append(path)

                check(data["board_prim_path"])
                for component in data.get("components", ()):
                    check(component.get("prim_path"))
                    for path in component.get("instance_paths", ()):
                        check(path)
                for instance in data.get("component_instances", ()):
                    check(instance.get("path"))
                    for key, value in instance.items():
                        if key.endswith("_prims") and isinstance(value, list):
                            for path in value:
                                check(path)
                self.assertEqual(
                    missing,
                    [],
                    f"{sidecar.name} names prims absent from {usd.name}:\n  "
                    + "\n  ".join(missing[:20]),
                )


if __name__ == "__main__":
    unittest.main()
