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

"""Test through-hole lead discovery against a synthetic board.

Built with plain ``usd-core``: discovery reads authored metadata and mesh
points, so it needs a stage but not Kit.
"""

from __future__ import annotations

import unittest

try:
    from pxr import Sdf, Usd, UsdGeom
except ImportError:  # the CI unit-test image installs numpy only
    Sdf = Usd = UsdGeom = None

from dataclasses import dataclass, field

from pcb_aoi.tht.lead_discovery import (
    discover_through_hole_leads,
    index_catalogs,
    pair_leads,
    scan_board,
)


@dataclass
class _Terminal:
    """Stand-in for a catalog TerminalRecord."""

    prim_path: str
    board_root: str
    component_path: str
    number: str | None = None
    process_class: str | None = None


@dataclass
class _Component:
    """Stand-in for a catalog ComponentRecord."""

    prim_path: str
    board_root: str
    refdes: str | None = None
    package_type: str | None = None


@dataclass
class _Hole:
    """Stand-in for a catalog HoleRecord."""

    prim_path: str
    board_root: str
    component_path: str | None = None
    terminal_number: str | None = None
    plated: bool | None = None


@dataclass
class _Catalog:
    """Stand-in for a BoardMetadataCatalog snapshot."""

    terminals: tuple = ()
    components: tuple = ()
    holes: tuple = ()


@unittest.skipUnless(Sdf is not None, "usd-core is not installed")
class CatalogTest(unittest.TestCase):
    """The catalog is the source of truth wherever it has an answer."""

    def _catalog(self, root="/Board", hole_name="hole"):
        """Build a catalog describing the synthetic board.

        Args:
            root: Board root the catalog's paths sit under.
            hole_name: Hole prim name the association points at.

        Returns:
            A catalog stand-in.
        """
        return _Catalog(
            terminals=(_Terminal(f"{root}/U1/lead", root, f"{root}/U1", "1"),),
            components=(
                _Component(f"{root}/U1", root, refdes="RELAY9",
                           package_type="THConnector"),
            ),
            holes=(
                _Hole(f"{root}/{hole_name}", root, f"{root}/U1", "1", True),
            ),
        )

    def test_identity_comes_from_the_catalog(self) -> None:
        """Prim attributes are double maintenance against a moving schema.

        This module already read the previous `pcb:` namespace and silently
        found nothing, so where the catalog has an answer it wins.
        """
        leads, _holes = scan_board(
            board(), index_catalogs([self._catalog()]),
        )
        self.assertEqual(leads[0].refdes, "RELAY9")
        self.assertEqual(leads[0].package_type, "THConnector")

    def test_the_prim_still_answers_without_a_catalog(self) -> None:
        """A board with no catalog has to keep working."""
        leads, _holes = scan_board(board())
        self.assertEqual(leads[0].refdes, "U1")
        self.assertEqual(leads[0].package_type, "TO92")

    def test_an_authored_association_is_used_and_recorded(self) -> None:
        """Where the board says which barrel a lead belongs to, believe it."""
        paired, _ = discover_through_hole_leads(board(), [self._catalog()])
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0].pairing_source, "catalog")

    def test_geometry_covers_what_the_catalog_does_not(self) -> None:
        """On a panelised THT board most terminals can lack an association.

        Deleting the fallback would silently stop varying those leads.
        """
        catalog = _Catalog(
            terminals=(_Terminal("/Board/U1/lead", "/Board", "/Board/U1", "1"),),
            components=(),
            holes=(),          # no authored association at all
        )
        paired, _ = discover_through_hole_leads(board(), [catalog])
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0].pairing_source, "geometry")

    def test_the_authored_barrel_beats_the_nearest_one(self) -> None:
        """The authored association corrects a lead the nearest bore would misplace.

        A relay's lead centroid sat nearer a neighbouring pin's bore than its
        own, so the geometric guess varied it inside the wrong barrel.
        """
        stage = board()
        far = _mesh(
            stage, "/Board/hole_far",
            _box(0.05, 0.0, 0.75, _BOARD_BOTTOM - _CAP_PROUD,
                 _BOARD_TOP + _CAP_PROUD),
        )
        _role(far, "board_hole")
        far.CreateAttribute(
            "pcba:board:holeRadiusMm", Sdf.ValueTypeNames.Float,
        ).Set(0.5)
        far.CreateAttribute("pcba:board:plated", Sdf.ValueTypeNames.Bool).Set(True)

        # the board says the lead belongs to the SECOND hole
        catalog = self._catalog(hole_name="hole_far")
        paired, _ = discover_through_hole_leads(stage, [catalog])
        self.assertEqual(paired[0].pairing_source, "catalog")
        self.assertTrue(paired[0].hole.path.endswith("hole_far"))

    def test_an_authored_barrel_the_lead_is_nowhere_near_is_refused(
        self,
    ) -> None:
        """The board's answer wins, but not against physics.

        On a panelised THT board, a connector's leads can be authored against
        barrels their own geometry sits millimetres away from -- a pin-numbering
        mismatch, not a pose. Believing it puts the lead in a hole it does not
        occupy; dropping the lead loses a pin the board plainly has. It falls
        back to geometry and the disagreement is reported.
        """
        stage = board()
        far = _mesh(
            stage, "/Board/hole_far",
            _box(20.0, 0.0, 0.75, _BOARD_BOTTOM - _CAP_PROUD,
                 _BOARD_TOP + _CAP_PROUD),
        )
        _role(far, "board_hole")
        far.CreateAttribute(
            "pcba:board:holeRadiusMm", Sdf.ValueTypeNames.Float,
        ).Set(0.5)
        far.CreateAttribute("pcba:board:plated", Sdf.ValueTypeNames.Bool).Set(True)

        paired, _ = discover_through_hole_leads(
            stage, [self._catalog(hole_name="hole_far")],
        )
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0].pairing_source, "geometry")
        self.assertTrue(paired[0].hole.path.endswith("/hole"))
        self.assertEqual(len(pair_leads.disagreements), 1)
        self.assertGreater(pair_leads.disagreements[0][2], 10.0)

    def test_a_panel_catalog_matches_a_prototype_path(self) -> None:
        """The two sides disagree on the prefix, not on the rest.

        The catalog resolves a panel through its instance proxies while
        discovery works on the prototype the placements share; everything
        after the board root is identical.
        """
        catalog = self._catalog(root="/World/Carrier/StepRepeats/SR_0_0")
        leads, _holes = scan_board(board(), index_catalogs([catalog]))
        self.assertEqual(leads[0].refdes, "RELAY9")

_BOARD_TOP, _BOARD_BOTTOM = 0.0, -1.6
_CAP_PROUD = 0.05  # the plated barrel mesh stands proud of each board face
_PROTOTYPE_ROOT = "/Proto"


def _mesh(stage, path, points):
    """Author a mesh carrying only the points a test needs.

    Args:
        stage: Stage to author on.
        path: Prim path.
        points: Vertex list.

    Returns:
        The created mesh's prim.
    """
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set([tuple(map(float, p)) for p in points])
    return mesh.GetPrim()


def _box(cx, cy, half, z_low, z_high):
    """Return a rectangular prism's eight corners, far cap ring first.

    Args:
        cx: Centre X.
        cy: Centre Y.
        half: Half width.
        z_low: Lower Z.
        z_high: Upper Z.

    Returns:
        Eight points.
    """
    corners = [
        (cx - half, cy - half), (cx + half, cy - half),
        (cx + half, cy + half), (cx - half, cy + half),
    ]
    return [(x, y, z_low) for x, y in corners] + [
        (x, y, z_high) for x, y in corners
    ]


def _role(prim, role):
    """Author the discovery role on a prim.

    Args:
        prim: Prim to mark.
        role: Role token.
    """
    prim.CreateAttribute("pcba:mesh:role", Sdf.ValueTypeNames.String).Set(role)


def board(*, lead_half=0.25, hole_radius=0.50, plated=True, abstract=False):
    """Build a one-component, one-lead board.

    Args:
        lead_half: Half width of the square lead.
        hole_radius: Stated bore radius.
        plated: Whether the hole is plated.
        abstract: Author the geometry inside an abstract class prototype, the
            way a panelised export does.

    Returns:
        The stage.
    """
    stage = Usd.Stage.CreateInMemory()
    root = "/Proto" if abstract else "/Board"
    if abstract:
        stage.CreateClassPrim(root)
    else:
        UsdGeom.Xform.Define(stage, root)

    component = UsdGeom.Xform.Define(stage, f"{root}/U1").GetPrim()
    component.CreateAttribute(
        "pcba:board:refdes", Sdf.ValueTypeNames.String,
    ).Set("U1")
    component.CreateAttribute(
        "pcba:component:packageType", Sdf.ValueTypeNames.String,
    ).Set("TO92")

    lead = _mesh(
        stage,
        f"{root}/U1/lead",
        _box(0.0, 0.0, lead_half, _BOARD_BOTTOM - 1.95, _BOARD_TOP + 2.0),
    )
    _role(lead, "pkg_lead")
    lead.CreateAttribute("pcba:pin:number", Sdf.ValueTypeNames.Int).Set(1)

    hole = _mesh(
        stage,
        f"{root}/hole",
        # The barrel mesh is the plating with its annular caps, so it is WIDER
        # than the bore and stands proud of both faces.
        _box(
            0.0, 0.0, hole_radius + 0.25,
            _BOARD_BOTTOM - _CAP_PROUD, _BOARD_TOP + _CAP_PROUD,
        ),
    )
    _role(hole, "board_hole")
    hole.CreateAttribute("pcba:board:holeRadiusMm", Sdf.ValueTypeNames.Float).Set(
        hole_radius,
    )
    hole.CreateAttribute("pcba:board:plated", Sdf.ValueTypeNames.Bool).Set(plated)
    return stage


@unittest.skipUnless(Sdf is not None, "usd-core is not installed")
class ScanTest(unittest.TestCase):
    """What the board declares, and where it hides it."""

    def test_roles_and_identity_are_read_from_metadata(self) -> None:
        """Discovery is metadata-driven, so the metadata must come through."""
        leads, holes = scan_board(board())
        self.assertEqual(len(leads), 1)
        self.assertEqual(len(holes), 1)
        self.assertEqual(leads[0].refdes, "U1")
        self.assertEqual(leads[0].package_type, "TO92")
        self.assertEqual(leads[0].pin, 1)

    def test_geometry_inside_an_abstract_class_is_still_found(self) -> None:
        """A panelised export hides its board in an abstract prototype.

        ``Traverse``'s default predicate skips abstract prims, so a plain
        traversal finds a handful of meshes out of tens of thousands and
        reports success. This is that failure, pinned.
        """
        leads, holes = scan_board(board(abstract=True))
        self.assertEqual(len(leads), 1)
        self.assertEqual(len(holes), 1)

    def test_a_lead_in_both_a_prototype_and_a_placement_counts_once(
        self,
    ) -> None:
        """`TraverseAll` sees the class AND everything composed from it.

        Counting both varies each physical lead twice, and authoring on the
        prototype would give every placement on a panel the SAME variation,
        which is a rig rather than a randomisation. The concrete placement is
        the one that gets varied.
        """
        stage = board(abstract=True)
        placement = UsdGeom.Xform.Define(stage, "/Board/U1_place").GetPrim()
        placement.GetReferences().AddInternalReference(Sdf.Path("/Proto/U1"))

        leads, _holes = scan_board(stage)
        self.assertEqual(len(leads), 1)
        self.assertFalse(
            stage.GetPrimAtPath(leads[0].path).IsAbstract(),
            "the concrete placement should be the one varied",
        )

    def test_a_prototype_only_lead_is_still_found(self) -> None:
        """With no concrete placement the prototype is all there is."""
        leads, _holes = scan_board(board(abstract=True))
        self.assertEqual(len(leads), 1)
        self.assertTrue(leads[0].path.startswith(_PROTOTYPE_ROOT))

    def test_a_default_traversal_would_miss_it(self) -> None:
        """Demonstrate the trap rather than only guarding against it."""
        stage = board(abstract=True)
        seen = [
            prim
            for prim in stage.Traverse()
            if prim.GetAttribute("pcba:mesh:role")
        ]
        self.assertEqual(seen, [])

    def test_the_bore_radius_is_taken_from_metadata_not_the_mesh(self) -> None:
        """The barrel mesh is the plating, so its extent is not the bore."""
        _leads, holes = scan_board(board(hole_radius=0.50))
        self.assertAlmostEqual(holes[0].radius_mm, 0.50, places=5)

    def test_the_legacy_attribute_namespace_still_reads(self) -> None:
        """An earlier export of the same board used `pcb:` without the `a`.

        Reading only the legacy names finds NOTHING on a board authored with
        the canonical namespace while looking exactly like a board that
        carries no schema, so a wrong namespace is easily mistaken for a wrong
        asset. Both are read, canonical first.
        """
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/Board")
        component = UsdGeom.Xform.Define(stage, "/Board/U1").GetPrim()
        component.CreateAttribute(
            "pcb:component:packageType", Sdf.ValueTypeNames.String,
        ).Set("TO92")
        lead = _mesh(
            stage, "/Board/U1/lead",
            _box(0.0, 0.0, 0.25, _BOARD_BOTTOM - 1.95, _BOARD_TOP + 2.0),
        )
        lead.CreateAttribute(
            "pcb:mesh:role", Sdf.ValueTypeNames.String,
        ).Set("pkg_lead")
        lead.CreateAttribute("pcb:pin:number", Sdf.ValueTypeNames.Int).Set(1)
        hole = _mesh(
            stage, "/Board/hole",
            _box(0.0, 0.0, 0.75, _BOARD_BOTTOM - _CAP_PROUD, _BOARD_TOP + _CAP_PROUD),
        )
        hole.CreateAttribute(
            "pcb:mesh:role", Sdf.ValueTypeNames.String,
        ).Set("board_hole")
        hole.CreateAttribute(
            "pcb:hole:radiusMm", Sdf.ValueTypeNames.Float,
        ).Set(0.5)
        hole.CreateAttribute(
            "pcb:hole:plated", Sdf.ValueTypeNames.Bool,
        ).Set(True)

        leads, holes = scan_board(stage)
        self.assertEqual((len(leads), len(holes)), (1, 1))
        self.assertEqual(leads[0].package_type, "TO92")
        self.assertEqual(leads[0].pin, 1)
        self.assertAlmostEqual(holes[0].radius_mm, 0.5, places=5)

    def test_the_canonical_namespace_wins_over_the_legacy_one(self) -> None:
        """A board authoring both must be read as the current schema."""
        stage = board(hole_radius=0.50)
        hole = stage.GetPrimAtPath("/Board/hole")
        hole.CreateAttribute(
            "pcb:hole:radiusMm", Sdf.ValueTypeNames.Float,
        ).Set(9.99)
        _leads, holes = scan_board(stage)
        self.assertAlmostEqual(holes[0].radius_mm, 0.50, places=5)

    def test_a_mesh_without_a_role_is_ignored(self) -> None:
        """Only what the board names is discovered."""
        stage = board()
        _mesh(stage, "/Board/decoration", _box(9.0, 9.0, 1.0, 0.0, 1.0))
        leads, holes = scan_board(stage)
        self.assertEqual((len(leads), len(holes)), (1, 1))


@unittest.skipUnless(Sdf is not None, "usd-core is not installed")
class PairTest(unittest.TestCase):
    """Which barrel a lead belongs to, and when it belongs to none."""

    def test_a_lead_is_paired_with_its_own_barrel(self) -> None:
        """The ordinary case has to work on a realistic barrel mesh."""
        paired, rejected = discover_through_hole_leads(board())
        self.assertEqual(len(paired), 1)
        self.assertEqual(rejected, [])
        self.assertIsNotNone(paired[0].hole)

    def test_coverage_is_a_fraction_not_strict_containment(self) -> None:
        """The barrel's annular caps stand proud of the faces the lead is flush with.

        Demanding the lead exceed the barrel at both ends rejects every genuine
        pin, so the pairing uses an overlap fraction. The synthetic board
        reproduces that geometry exactly.
        """
        _leads, holes = scan_board(board())
        hole = holes[0]
        self.assertLess(hole.z_range[0], _BOARD_BOTTOM)
        self.assertGreater(hole.z_range[1], _BOARD_TOP)
        paired, _rejected = discover_through_hole_leads(board())
        self.assertEqual(len(paired), 1)

    def test_an_unplated_hole_never_claims_a_lead(self) -> None:
        """A via is not a barrel a part is seated in."""
        paired, rejected = discover_through_hole_leads(board(plated=False))
        self.assertEqual(paired, [])
        self.assertEqual(len(rejected), 1)

    def test_a_surface_mount_termination_is_rejected(self) -> None:
        """A short lead beside a via must not be claimed as through-hole.

        Coverage is what separates the two, so a lead that only touches the top
        of the barrel is rejected and its numbers are reported.
        """
        stage = board()
        lead = stage.GetPrimAtPath("/Board/U1/lead")
        UsdGeom.Mesh(lead).GetPointsAttr().Set(
            [tuple(map(float, p)) for p in _box(0.0, 0.0, 0.25, -0.1, 1.0)],
        )
        paired, rejected = discover_through_hole_leads(stage)
        self.assertEqual(paired, [])
        self.assertEqual(len(rejected), 1)
        _lead, _distance, coverage = rejected[0]
        self.assertLess(coverage, 0.9)

    def test_a_lead_far_from_every_hole_is_rejected(self) -> None:
        """Nearest is not near enough on its own."""
        stage = board()
        lead = stage.GetPrimAtPath("/Board/U1/lead")
        UsdGeom.Mesh(lead).GetPointsAttr().Set(
            [
                tuple(map(float, p))
                for p in _box(40.0, 0.0, 0.25, _BOARD_BOTTOM - 1.9, 2.0)
            ],
        )
        paired, rejected = discover_through_hole_leads(stage)
        self.assertEqual(paired, [])
        self.assertGreater(rejected[0][1], 1.0)

    def test_no_holes_rejects_every_lead_without_raising(self) -> None:
        """An empty board is a diagnosis, not a crash."""
        stage = board()
        stage.RemovePrim("/Board/hole")
        paired, rejected = discover_through_hole_leads(stage)
        self.assertEqual(paired, [])
        self.assertEqual(len(rejected), 1)


if __name__ == "__main__":
    unittest.main()
