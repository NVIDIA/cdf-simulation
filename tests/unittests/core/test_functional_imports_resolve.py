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

"""Names the functional tests import from each other must actually exist.

The functional suite runs only in CI, inside the container, on a GPU runner.
That makes a broken cross-import the most expensive kind of trivial mistake
here: renaming a helper and missing one caller costs a full image build and a
GPU job before anything says so.

That is exactly what happened on the Modus MR. ``_configure_appearance`` in
``tests/functional/smt_scenario.py`` became the public hook
``configure_ring_appearance`` so the Modus scenario could substitute its own,
and ``test_smt_solder_defect.py`` was left importing the old private name:

    ImportError: cannot import name '_configure_appearance' from
    'tests.functional.smt_scenario'

These modules import Isaac at module scope, so they cannot be imported here to
find that out. They can be read. This walks every intra-suite ``from
tests.functional.X import ...`` and checks each name against what X defines --
static, no Isaac, milliseconds.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_FUNCTIONAL = Path(__file__).resolve().parents[3] / "tests" / "functional"


def _defined_names(path: Path) -> set[str]:
    """Every module-level name *path* binds, by any of the usual means."""
    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    names.update(e.id for e in target.elts
                                 if isinstance(e, ast.Name))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # A re-export is a legitimate way to provide a name.
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _intra_suite_imports(path: Path):
    """(module, imported name, line) for each `from tests.functional.X import`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith("tests.functional."):
            continue
        module = node.module.split(".")[-1]
        for alias in node.names:
            if alias.name != "*":
                yield module, alias.name, node.lineno


class FunctionalImportsResolveTest(unittest.TestCase):

    def setUp(self):
        self.modules = sorted(_FUNCTIONAL.glob("*.py"))
        self.assertTrue(self.modules, "no functional modules found")

    def test_the_suite_is_actually_being_scanned(self):
        # A guard on the guard: if the glob silently stops matching, the test
        # below passes vacuously and this whole file becomes decoration.
        checked = sum(1 for path in self.modules
                      for _ in _intra_suite_imports(path))
        self.assertGreater(checked, 0,
                           "no intra-suite imports found; is the path right?")

    def test_every_imported_name_exists_in_its_module(self):
        for path in self.modules:
            for module, name, line in _intra_suite_imports(path):
                with self.subTest(source=path.name, module=module, name=name):
                    target = _FUNCTIONAL / ("%s.py" % module)
                    self.assertTrue(
                        target.is_file(),
                        "%s:%d imports from tests.functional.%s, which does "
                        "not exist" % (path.name, line, module))
                    self.assertIn(
                        name, _defined_names(target),
                        "%s:%d imports %r from %s.py, which does not define "
                        "it" % (path.name, line, name, module))


if __name__ == "__main__":
    unittest.main()
