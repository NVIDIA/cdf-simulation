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

"""Tests for referencing companion assets relative to the loaded scene."""

from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi domain
from cdf.errors import ValidationError
from cdf.execution.isaac.execution_context import (
    SCENE_URI_SCHEME,
    IsaacExecutionContext,
)


def _write_archive(directory: Path, *, members) -> Path:
    """Write a ZIP holding one root-level USD plus the given companions."""
    archive = directory / "panel.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("panel.usda", "#usda 1.0\n")
        for name, payload in members.items():
            bundle.writestr(name, payload)
    return archive


class SceneArchiveAssetTest(unittest.TestCase):
    """Verify placeholder expansion against an extracted scene archive."""

    def test_scene_scheme_resolves_to_the_extracted_member(self):
        """A texture shipped in the archive resolves to its extracted copy."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _write_archive(root, members={"albedo.png": b"stub"})
            context = IsaacExecutionContext()
            with context.storage_scope():
                context.localize_scene(archive)
                resolved = context.resolve_scene_asset(
                    f"{SCENE_URI_SCHEME}albedo.png",
                )
                self.assertTrue(Path(resolved).is_file())
                self.assertEqual(Path(resolved).name, "albedo.png")
                self.assertEqual(Path(resolved).read_bytes(), b"stub")

    def test_scene_scheme_resolves_a_nested_member(self):
        """Companions may sit at any depth inside the archive."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _write_archive(root, members={"tex/normal.png": b"stub"})
            context = IsaacExecutionContext()
            with context.storage_scope():
                context.localize_scene(archive)
                resolved = context.resolve_scene_asset(
                    f"{SCENE_URI_SCHEME}tex/normal.png",
                )
                self.assertTrue(Path(resolved).is_file())

    def test_leaves_rooted_paths_untouched(self):
        """A rooted path is not relative, even without a Windows drive letter."""
        context = IsaacExecutionContext()
        self.assertEqual(
            context.resolve_scene_asset("/assets/albedo.png"),
            str(Path("/assets/albedo.png")),
        )

    def test_scene_scheme_works_for_a_plain_usd_scene(self):
        """The scheme means "beside the scene", which a plain USD also has."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "panel.usda").write_text("#usda 1.0" + chr(10), encoding="utf-8")
            (root / "albedo.png").write_bytes(b"stub")
            context = IsaacExecutionContext()
            with context.storage_scope():
                context.localize_scene(root / "panel.usda")
                resolved = context.resolve_scene_asset(
                    f"{SCENE_URI_SCHEME}albedo.png",
                )
                self.assertEqual(Path(resolved).read_bytes(), b"stub")

    def test_rejects_scene_reference_before_a_scene_is_loaded(self):
        """The scheme is meaningless until a scene has been configured."""
        context = IsaacExecutionContext()
        with self.assertRaises(ValidationError) as caught:
            context.resolve_scene_asset(f"{SCENE_URI_SCHEME}albedo.png")
        self.assertIn("scene.configure", str(caught.exception))

    def test_rejects_a_member_the_scene_lacks(self):
        """A typo names the available assets rather than failing at render."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _write_archive(root, members={"albedo.png": b"stub"})
            context = IsaacExecutionContext()
            with context.storage_scope():
                context.localize_scene(archive)
                with self.assertRaises(ValidationError) as caught:
                    context.resolve_scene_asset(f"{SCENE_URI_SCHEME}albedoo.png")
            self.assertIn("albedo.png", str(caught.exception))

    def test_rejects_a_reference_escaping_the_scene(self):
        """A traversal must not reach files outside the scene directory."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "secret.png").write_bytes(b"stub")
            archive = _write_archive(root, members={"albedo.png": b"stub"})
            context = IsaacExecutionContext()
            with context.storage_scope():
                context.localize_scene(archive)
                with self.assertRaises(ValidationError) as caught:
                    context.resolve_scene_asset(
                        f"{SCENE_URI_SCHEME}../../secret.png",
                    )
            self.assertIn("escapes the scene", str(caught.exception))


class RelativeAssetFallbackTest(unittest.TestCase):
    """Verify that a bare relative path prefers disk and falls back to scene."""

    def test_prefers_a_path_that_exists_relative_to_the_working_directory(self):
        """A local asset keeps winning, so existing jobs are unaffected."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _write_archive(root, members={"albedo.png": b"in-archive"})
            local = root / "local"
            local.mkdir()
            (local / "albedo.png").write_bytes(b"on-disk")
            context = IsaacExecutionContext()
            previous = os.getcwd()
            os.chdir(root)
            try:
                with context.storage_scope():
                    context.localize_scene(archive)
                    resolved = context.resolve_scene_asset("local/albedo.png")
                    self.assertEqual(Path(resolved).read_bytes(), b"on-disk")
            finally:
                os.chdir(previous)

    def test_falls_back_to_the_scene_when_not_on_disk(self):
        """An asset only present in the archive still resolves."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _write_archive(root, members={"tex/albedo.png": b"in-archive"})
            context = IsaacExecutionContext()
            previous = os.getcwd()
            os.chdir(root)
            try:
                with context.storage_scope():
                    context.localize_scene(archive)
                    resolved = context.resolve_scene_asset("tex/albedo.png")
                    self.assertEqual(Path(resolved).read_bytes(), b"in-archive")
            finally:
                os.chdir(previous)

    def test_reports_both_places_when_an_asset_is_in_neither(self):
        """The error says it looked on disk and in the scene."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _write_archive(root, members={"albedo.png": b"stub"})
            context = IsaacExecutionContext()
            with context.storage_scope():
                context.localize_scene(archive)
                with self.assertRaises(ValidationError) as caught:
                    context.resolve_scene_asset("missing.png")
            message = str(caught.exception)
            self.assertIn("not on disk", message)
            self.assertIn("albedo.png", message)

    def test_absolute_paths_are_untouched(self):
        """An absolute asset path is never rewritten or required to exist."""
        context = IsaacExecutionContext()
        absolute = str(Path(tempfile.gettempdir()) / "nowhere" / "albedo.png")
        self.assertEqual(context.resolve_scene_asset(absolute), absolute)


if __name__ == "__main__":
    unittest.main()
