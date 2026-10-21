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

"""How the batch pipeline reads a pcba target's board-scoped keys.

The legacy pipeline resolved ``component_types`` keywords
and picked component classes at module scope, behind a Kit import, so neither
rule had a test. Both now live in ``scripts/sdg/standalone/pcba_target.py``;
this file pins them, including two edge cases: an
unquoted YAML ``0`` reaching component processing as an int, and an
environment-backed ``component_registry`` path being resolved before the
configuration was expanded.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import yaml
except ImportError:  # the CI unit-test image installs PyYAML, but be honest
    yaml = None

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STANDALONE = _REPO_ROOT / "scripts" / "sdg" / "standalone"


def _load():
    sys.path.insert(0, str(_STANDALONE))
    try:
        return importlib.import_module("pcba_target")
    finally:
        sys.path.remove(str(_STANDALONE))


pcba_target = _load()

_REGISTRY = {
    "all": ["CAP_SMD_0603", "SOIC8", "TSSOP20"],
    "subsets": {"passives": ["CAP_SMD_0603"]},
}


@unittest.skipUnless(yaml is not None, "PyYAML not installed")
class ComponentTypesKeywordTest(unittest.TestCase):
    """Keyword resolution against the registry travelling with the target."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.board = Path(self.tmp.name) / "board"
        self.board.mkdir()
        self.target = self.board / "pcba_target.yaml"

    def _write_target(self, text):
        self.target.write_text(text, encoding="utf-8")
        with open(self.target, encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _write_registry(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(_REGISTRY), encoding="utf-8")

    # -- the two YAML spellings of "no components" -------------------------

    def test_an_unquoted_zero_means_no_components(self):
        # yaml loads `component_types: 0` as an int. The old string-only branch
        # let it through untouched, so component processing received 0.
        cfg = self._write_target("component_types: 0\n")
        self.assertIsInstance(cfg["component_types"], int)
        self.assertEqual(pcba_target.resolve_component_types(cfg, self.target), [])

    def test_a_quoted_zero_means_no_components(self):
        cfg = self._write_target('component_types: "0"\n')
        self.assertIsInstance(cfg["component_types"], str)
        self.assertEqual(pcba_target.resolve_component_types(cfg, self.target), [])

    def test_zero_needs_no_registry(self):
        # No components.yaml beside the target, and none asked for.
        for spelling in (0, "0"):
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    pcba_target.resolve_component_types(
                        {"component_types": spelling}, self.target
                    ),
                    [],
                )

    def test_a_boolean_is_not_a_zero(self):
        # False == 0 in Python; a boolean there is a typo, not a selection,
        # and must not be read as "no components".
        self.assertIsNone(
            pcba_target.resolve_component_types({"component_types": False}, self.target)
        )

    # -- lists pass through, keywords resolve -----------------------------

    def test_a_list_needs_nothing_and_is_left_alone(self):
        cfg = self._write_target("component_types: [SOIC8, TSSOP20]\n")
        self.assertIsNone(pcba_target.resolve_component_types(cfg, self.target))

    def test_all_resolves_against_the_sibling_registry(self):
        self._write_registry(self.board / "components.yaml")
        cfg = self._write_target("component_types: ALL\n")
        self.assertEqual(
            pcba_target.resolve_component_types(cfg, self.target),
            _REGISTRY["all"],
        )

    def test_a_subset_key_resolves(self):
        self._write_registry(self.board / "components.yaml")
        cfg = self._write_target("component_types: passives\n")
        self.assertEqual(
            pcba_target.resolve_component_types(cfg, self.target),
            ["CAP_SMD_0603"],
        )

    def test_component_registry_is_relative_to_the_target(self):
        self._write_registry(self.board / "registry" / "parts.yaml")
        cfg = self._write_target(
            "component_types: ALL\ncomponent_registry: registry/parts.yaml\n"
        )
        self.assertEqual(
            pcba_target.resolve_component_types(cfg, self.target),
            _REGISTRY["all"],
        )

    def test_an_environment_backed_registry_path_resolves(self):
        # `${PCB_REGISTRY_DIR}/components.yaml` is what the loader promises to
        # expand. It used to be joined onto the target directory unexpanded,
        # because the merged configuration was expanded only afterwards.
        registry_dir = Path(self.tmp.name) / "shared_registry"
        self._write_registry(registry_dir / "components.yaml")
        cfg = self._write_target(
            "component_types: ALL\n"
            "component_registry: ${PCB_REGISTRY_DIR}/components.yaml\n"
        )
        with mock.patch.dict(os.environ, {"PCB_REGISTRY_DIR": str(registry_dir)}):
            resolved = pcba_target.resolve_component_types(cfg, self.target)
        self.assertEqual(resolved, _REGISTRY["all"])

    # -- refusals name what to fix -----------------------------------------

    def test_a_missing_registry_names_the_path_and_the_key(self):
        cfg = self._write_target("component_types: ALL\n")
        with self.assertRaises(pcba_target.ComponentRegistryError) as ctx:
            pcba_target.resolve_component_types(cfg, self.target)
        message = str(ctx.exception)
        self.assertIn(str(self.board / "components.yaml"), message)
        self.assertIn("component_registry", message)

    def test_a_keyword_without_a_target_is_refused(self):
        with self.assertRaisesRegex(
            pcba_target.ComponentRegistryError, "--pcba-config"
        ):
            pcba_target.resolve_component_types({"component_types": "ALL"}, None)

    def test_a_non_string_registry_is_refused_not_joined(self):
        cfg = {"component_types": "ALL", "component_registry": ["a.yaml"]}
        with self.assertRaisesRegex(
            pcba_target.ComponentRegistryError, "must be a path string"
        ):
            pcba_target.resolve_component_types(cfg, self.target)

    def test_an_unknown_keyword_lists_what_is_allowed(self):
        self._write_registry(self.board / "components.yaml")
        cfg = self._write_target("component_types: everything\n")
        with self.assertRaisesRegex(
            pcba_target.ComponentRegistryError, "ALL, 0, or a key"
        ):
            pcba_target.resolve_component_types(cfg, self.target)


class ComponentClassTest(unittest.TestCase):
    """An instance the map does not name is a `component`, and is reported."""

    def test_the_default_is_neutral_not_a_package_family(self):
        self.assertEqual(pcba_target.DEFAULT_COMPONENT_CLASS, "component")

    def test_an_unmapped_ic_is_not_labelled_a_capacitor(self):
        # The board maps its passives and forgets the IC. Labelling U1 as a
        # capacitor is the silent mislabel the old fallback produced.
        classes, unmapped = pcba_target.classify_components(
            ["C1", "U1", "J3"], {"capacitor": ["C1"]}
        )
        self.assertEqual(
            classes, {"C1": "capacitor", "U1": "component", "J3": "component"}
        )
        self.assertEqual(unmapped, ["U1", "J3"])

    def test_without_a_map_every_instance_is_on_the_default_by_design(self):
        classes, unmapped = pcba_target.classify_components(["C1", "U1"], None)
        self.assertEqual(classes, {"C1": "component", "U1": "component"})
        self.assertEqual(unmapped, [])

    def test_the_board_may_name_its_own_default(self):
        classes, _ = pcba_target.classify_components(
            ["C1", "U1"], {"ic": ["U1"]}, default_class="capacitor"
        )
        self.assertEqual(classes, {"C1": "capacitor", "U1": "ic"})

    def test_an_empty_class_list_is_tolerated(self):
        classes, unmapped = pcba_target.classify_components(
            ["U1"], {"ic": None, "connector": []}
        )
        self.assertEqual(classes, {"U1": "component"})
        self.assertEqual(unmapped, [])


class PipelineWiringTest(unittest.TestCase):
    """Shared pipeline helpers do not restore removed board assumptions."""

    _COMMON = _STANDALONE / "common.py"

    def test_common_has_no_capacitor_fallback(self):
        source = self._COMMON.read_text(encoding="utf-8")
        self.assertNotIn('"capacitor"', source)
        self.assertIn("classify_components", source)


if __name__ == "__main__":
    unittest.main()
