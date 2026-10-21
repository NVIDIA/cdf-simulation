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

"""Tests for the bounded on-disk normal-map cache in
``scripts/sdg/editor/tin_noise_patch.py``.

``randomize_tin_noise`` draws a fresh seed per sample and the seed is part
of the cache key, so ``make_tin_normalmap`` writes one new PNG per trigger
into a FIXED shared temp dir — the cache must evict or it grows without
bound across campaigns.

``tin_noise_patch`` imports omni/pxr and the fillet editor at module level
(it is a Kit editor script), so this test injects lightweight stubs to load
it outside Kit; the normal-map baking under test is pure numpy + PIL. The
whole class self-skips without PIL (the CI image is numpy-only).
"""

import importlib
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    from PIL import Image  # noqa: F401
except ImportError:  # CI unit-test image installs numpy only
    Image = None

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EDITOR_DIR = _REPO_ROOT / "scripts" / "sdg" / "editor"

_FILLET_STUB_NAMES = (
    "_component_axes", "_compute_world_bbox", "_compute_z_warp",
    "_find_instances", "_fractal_noise_2d", "_mm_to_scene",
    "_set_transform", "_wp_init",
)


def _load_tin_noise_patch():
    """Import tin_noise_patch with omni/pxr/fillet-editor stubs, then pop
    the stubs from sys.modules so no other test observes them (e.g. the
    declarative-API test asserting omni/pxr are never imported)."""
    stubs = {}
    omni = types.ModuleType("omni")
    omni_usd = types.ModuleType("omni.usd")
    omni.usd = omni_usd
    stubs["omni"] = omni
    stubs["omni.usd"] = omni_usd
    pxr = types.ModuleType("pxr")
    for name in ("Gf", "Sdf", "UsdGeom", "UsdShade"):
        setattr(pxr, name, types.SimpleNamespace())
    stubs["pxr"] = pxr
    fillet = types.ModuleType("editor_solder_fillet_board")
    for name in _FILLET_STUB_NAMES:
        setattr(fillet, name, lambda *args, **kwargs: None)
    stubs["editor_solder_fillet_board"] = fillet

    injected = [name for name in stubs if name not in sys.modules]
    for name in injected:
        sys.modules[name] = stubs[name]
    sys.path.insert(0, str(_EDITOR_DIR))
    prior = sys.modules.pop("tin_noise_patch", None)
    try:
        return importlib.import_module("tin_noise_patch")
    finally:
        sys.path.remove(str(_EDITOR_DIR))
        sys.modules.pop("tin_noise_patch", None)
        if prior is not None:
            sys.modules["tin_noise_patch"] = prior
        for name in injected:
            sys.modules.pop(name, None)


@unittest.skipUnless(Image is not None, "PIL not installed (CI is numpy-only)")
class NormalmapCacheEvictionTest(unittest.TestCase):

    _CACHE_MAX = 6
    _SIZE = 32  # small maps keep the suite fast; keying is size-agnostic

    @classmethod
    def setUpClass(cls):
        cls.tnp = _load_tin_noise_patch()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tin_normal_cache_test_")
        self._old_dir = self.tnp._NORMALMAP_DIR
        self._old_max = self.tnp._NORMALMAP_CACHE_MAX
        self.tnp._NORMALMAP_DIR = self.tmp
        self.tnp._NORMALMAP_CACHE_MAX = self._CACHE_MAX

    def tearDown(self):
        self.tnp._NORMALMAP_DIR = self._old_dir
        self.tnp._NORMALMAP_CACHE_MAX = self._old_max
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _bake(self, seed=None, **kwargs):
        return self.tnp.make_tin_normalmap(
            self.tnp.TinNoiseParams(), size=self._SIZE, seed=seed, **kwargs)

    def _cached_files(self):
        return [f for f in os.listdir(self.tmp)
                if f.startswith("tin_normal_") and f.endswith(".png")]

    def test_distinct_seeds_beyond_cap_evict_down_to_cap(self):
        # N + K distinct seeds (the randomize_tin_noise pattern) must not
        # retain more than N files on disk. Back-date each file after its
        # bake so eviction order is deterministic even on filesystems with
        # coarse mtime resolution (fast successive writes can tie).
        import time
        base = time.time() - 1000.0
        last = ""
        for i, seed in enumerate(range(self._CACHE_MAX + 4)):
            last = self._bake(seed=seed)
            os.utime(last, (base + i, base + i))
        files = self._cached_files()
        self.assertLessEqual(len(files), self._CACHE_MAX)
        # the just-written file is never the eviction victim
        self.assertTrue(os.path.isfile(last))

    def test_fixed_seed_setup_path_still_caches_single_file(self):
        first = self._bake(seed=None)   # historical fixed seed 42
        second = self._bake(seed=None)  # cache hit — same path, no rewrite
        self.assertEqual(first, second)
        self.assertEqual(len(self._cached_files()), 1)

    def test_png_bytes_unchanged_for_a_given_cache_key(self):
        path = self._bake(seed=7)
        with open(path, "rb") as handle:
            before = handle.read()
        rebuilt = self._bake(seed=7, force_rebuild=True)
        self.assertEqual(rebuilt, path)
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), before)


if __name__ == "__main__":
    unittest.main()
