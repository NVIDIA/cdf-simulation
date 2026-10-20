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

"""Tests for solder and terminal geometry semantic labels."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi domain
from cdf import IsaacExecutionContext, CDFJob
from cdf.errors import ValidationError
from cdf.execution.isaac.default_handler_registry import (
    create_default_isaac_handler_registry,
)
from cdf.execution.isaac.job_order_parameters import GeometryAnnotationParameters
from cdf.execution.isaac.job_order_runtime import (
    _label_geometry_prims,
    _project_geometry_class,
)
from pcb_aoi.execution.board_instances import BoardInstanceCohort

_FAKE_PXR = types.ModuleType("pxr")
_FAKE_PXR.Sdf = types.SimpleNamespace(
    ValueTypeNames=types.SimpleNamespace(String="String"),
)


class GeometryAnnotationDeclarationTest(unittest.TestCase):
    """Verify how geometry classes are declared and validated."""

    def setUp(self):
        """Create a job whose annotations section is under test."""
        self.job = CDFJob(domain="pcb-aoi", random_seed=0)

    def test_declares_every_geometry_target(self):
        """All four targets are recorded and compile to one order."""
        self.job.annotations.label_geometry(
            solder="solder",
            terminals="pin",
            pads="pad",
            holes="hole",
        )
        self.assertEqual(
            self.job.annotations.geometry_labels,
            {"solder": "solder", "terminals": "pin", "pads": "pad", "holes": "hole"},
        )

    def test_declares_a_single_target(self):
        """Labeling only solder leaves terminals unset."""
        self.job.annotations.label_geometry(solder="solder")
        self.assertEqual(self.job.annotations.geometry_labels, {"solder": "solder"})

    def test_rejects_unknown_target(self):
        """A misspelled target fails at declaration, not at render time."""
        with self.assertRaises(ValidationError) as caught:
            self.job.annotations.label_geometry(vias="via")
        self.assertIn("unknown target", str(caught.exception))

    def test_rejects_empty_declaration(self):
        """Declaring nothing is a mistake rather than a no-op."""
        with self.assertRaises(ValidationError):
            self.job.annotations.label_geometry()

    def test_rejects_non_string_class(self):
        """A class must be a name."""
        with self.assertRaises(ValidationError):
            self.job.annotations.label_geometry(solder=True)

    def test_rejects_empty_component_allowlist(self):
        """An explicit component filter must select at least one refdes."""
        for components in ([], ()):
            with (
                self.subTest(components=components),
                self.assertRaises(ValidationError),
            ):
                self.job.annotations.label_geometry(
                    terminals="pin",
                    components=components,
                )


class GeometryAnnotationParameterTest(unittest.TestCase):
    """Verify the executable parameter contract."""

    def test_accepts_one_or_both_targets(self):
        """Either target alone is a valid order."""
        self.assertEqual(
            GeometryAnnotationParameters(solder="solder").terminals,
            None,
        )
        self.assertEqual(
            GeometryAnnotationParameters(terminals="pin").solder,
            None,
        )

    def test_rejects_empty_parameters(self):
        """An order with no class would silently label nothing."""
        with self.assertRaises(ValueError):
            GeometryAnnotationParameters()

    def test_handler_is_registered(self):
        """The order id resolves to a runtime handler."""
        registry = create_default_isaac_handler_registry()
        self.assertIn("annotations.geometry.apply", registry.handlers)


class _FakeAttribute:
    """Minimal attribute capturing an authored value."""

    def __init__(self, store, name):
        self._store = store
        self._name = name

    def Set(self, value):
        """Record the authored value."""
        self._store[self._name] = value


class _FakePrim:
    """Minimal prim supporting the instance-proxy resolution under test."""

    def __init__(
        self,
        path,
        *,
        valid=True,
        instance=False,
        instance_proxy=False,
        parent=None,
    ):
        self.path = path
        self._valid = valid
        self._instance = instance
        self._instance_proxy = instance_proxy
        self._parent = parent
        self.authored = {}

    def IsValid(self):
        """Report prim validity."""
        return self._valid

    def IsInstanceProxy(self):
        """Report whether this prim stands in for a prototype."""
        return self._instance_proxy

    def IsInstance(self):
        """Report whether this prim is an instance root."""
        return self._instance

    def GetParent(self):
        """Return the parent prim."""
        return self._parent

    def SetInstanceable(self, instanceable):
        """Record whether the prim remains instanceable."""
        self._instance = bool(instanceable)

    def GetPath(self):
        """Return the prim path."""
        return self.path

    def CreateAttribute(self, name, _type):
        """Return a recording attribute."""
        return _FakeAttribute(self.authored, name)


class _FakeStage:
    """Stage returning preset prims by path."""

    def __init__(self, prims):
        self._prims = prims

    def GetPrimAtPath(self, path):
        """Return the prim at a path, or an invalid stand-in."""
        return self._prims.get(path, _FakePrim(path, valid=False))


class _FakeCommon:
    """Stand-in for the standalone common module."""

    def __init__(self):
        self.refreshed = None

    def refresh_authored_semantics(self, _stage, prim_paths):
        """Record which prims were refreshed."""
        self.refreshed = list(prim_paths)
        return len(prim_paths)


@mock.patch.dict(sys.modules, {"pxr": _FAKE_PXR})
class LabelGeometryPrimsTest(unittest.TestCase):
    """Verify prim-level label authoring."""

    def test_deinstances_parent_and_labels_former_proxy(self):
        """A proxy becomes writable after its instance root is de-instanced."""
        instance_root = _FakePrim("/board/SR_0/C1", instance=True)
        proxy = _FakePrim(
            "/board/SR_0/C1/pkg_lead_1",
            instance_proxy=True,
            parent=instance_root,
        )
        stage = _FakeStage({proxy.path: proxy})
        common = _FakeCommon()

        count = _label_geometry_prims(stage, common, [proxy.path], "pin")

        self.assertEqual(count, 1)
        self.assertFalse(instance_root.IsInstance())
        self.assertEqual(
            proxy.authored["semantic:Semantics:class:semanticData"],
            "pin",
        )
        self.assertEqual(common.refreshed, [proxy.path])

    def test_labels_each_deinstanced_component_instance(self):
        """Each component instance receives its own concrete semantic label."""
        instance_roots = [
            _FakePrim(f"/board/SR_0_{index}/C1", instance=True) for index in range(2)
        ]
        proxies = {}
        for index, instance_root in enumerate(instance_roots):
            path = f"/board/SR_0_{index}/C1/pkg_lead_1"
            proxies[path] = _FakePrim(
                path,
                instance_proxy=True,
                parent=instance_root,
            )
        common = _FakeCommon()

        count = _label_geometry_prims(
            _FakeStage(proxies),
            common,
            sorted(proxies),
            "pin",
        )

        self.assertEqual(count, 2)
        self.assertTrue(all(not root.IsInstance() for root in instance_roots))
        self.assertTrue(all(proxy.authored for proxy in proxies.values()))

    def test_skips_paths_with_no_prim(self):
        """A stale metadata path is skipped rather than raising."""
        common = _FakeCommon()
        count = _label_geometry_prims(_FakeStage({}), common, ["/gone"], "pin")
        self.assertEqual(count, 0)


class GeometryProjectionTest(unittest.TestCase):
    """Verify geometry-only labels activate instanced semantic projection."""

    def test_geometry_initializes_projection_without_component_labels(self):
        """A cohort can project pins and pads without labeling components."""
        cohort = BoardInstanceCohort(
            instance_paths=("/World/Panel/I0", "/World/Panel/I1"),
            representative_instance_path="/World/Panel/I0",
            prototype_path="/__Prototype_1",
            prototype_source_path="/Board",
        )
        context = IsaacExecutionContext(
            runtime=object(),
            board_instance_cohort=cohort,
        )

        count = _project_geometry_class(
            context,
            ("/Board/Pads/P1",),
            "pad",
            "pads",
        )

        self.assertEqual(count, 1)
        self.assertIsNotNone(context.instance_semantic_projection)
        projection = context.instance_semantic_projection
        self.assertIs(projection.cohort, cohort)
        self.assertEqual(projection.components[0].source_path, "/Board/Pads/P1")
        self.assertEqual(projection.components[0].class_name, "pad")


class MultipleStitchDeclarationTest(unittest.TestCase):
    """Verify that one pass can be stitched into several products."""

    def test_image_and_mask_compile_to_separate_orders(self):
        """The RGB and segmentation stitches must not overwrite each other."""
        job = CDFJob(domain="pcb-aoi", random_seed=0)
        job.postprocess.stitch(source_pass="full_board", output_filename="panel.png")
        job.postprocess.stitch(
            source_pass="full_board",
            image_type="semantic_segmentation",
            output_filename="panel_mask.png",
            blend="labels",
        )
        self.assertEqual(len(job.postprocess.declared_stitches), 2)
        self.assertEqual(
            [entry["image_type"] for entry in job.postprocess.declared_stitches],
            ["rgb", "semantic_segmentation"],
        )


class DomainOwnedGeometryVocabularyTest(unittest.TestCase):
    """Verify that the labelable geometry names come from the domain."""

    def test_domain_declares_the_targets(self):
        """CDF holds no board vocabulary of its own."""
        from cdf.domain import get_domain

        self.assertEqual(
            get_domain("pcb-aoi").geometry_targets,
            {"solder", "terminals", "pads", "holes", "barrels", "body", "marking"},
        )

    def test_traversal_targets_name_their_mesh_role(self):
        """Each traversal target labels exactly one ``pcba:mesh:role`` value."""
        from pcb_aoi.geometry_labels import (
            PACKAGE_TRAVERSAL_TARGETS,
            TRAVERSAL_ROLES,
            TRAVERSAL_TARGETS,
            traversal_role,
        )

        self.assertEqual(set(TRAVERSAL_TARGETS), set(TRAVERSAL_ROLES))
        self.assertEqual(traversal_role("barrels"), "board_barrel")
        self.assertEqual(traversal_role("body"), "pkg_body")
        self.assertEqual(traversal_role("marking"), "pkg_marking")
        # package meshes are owned by their enclosing component; board
        # geometry is owned through a relationship
        self.assertEqual(PACKAGE_TRAVERSAL_TARGETS, {"body", "marking"})
        self.assertNotIn("barrels", PACKAGE_TRAVERSAL_TARGETS)
        with self.assertRaises(KeyError):
            traversal_role("solder")

    def test_body_and_marking_reach_the_parameter_model(self):
        """A body/marking declaration survives into the compiled parameters."""
        job = CDFJob(domain="pcb-aoi", random_seed=0)
        job.annotations.label_geometry(body="body", marking="pin1_marker")
        parameters = GeometryAnnotationParameters(**job.annotations.geometry_labels)
        self.assertEqual(parameters.body, "body")
        self.assertEqual(parameters.marking, "pin1_marker")
        self.assertIsNone(parameters.barrels)

    def test_catalog_targets_name_their_metadata_source(self):
        """Each catalog-backed target maps to a capability and an attribute."""
        from pcb_aoi.geometry_labels import CATALOG_TARGETS, catalog_target

        for name in CATALOG_TARGETS:
            capability, attribute = catalog_target(name)
            self.assertTrue(str(capability))
            self.assertTrue(attribute)

    def test_error_names_the_domain_that_rejected_the_target(self):
        """The message points at the vocabulary owner, not at cdf."""
        job = CDFJob(domain="pcb-aoi", random_seed=0)
        with self.assertRaises(ValidationError) as caught:
            job.annotations.label_geometry(vias="via")
        self.assertIn("pcb-aoi supports", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
