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

"""A components sidecar read against a board root that is not its own.

A sidecar records paths as authored in the board's OWN USD file, where the
board root is a child of the pseudo-root: ``/Board_X/Components/Top/R1``. A
batch run opens that file as the stage, so they resolve unchanged. A live
session does not -- it referenced the board in under ``/World`` -- and then
every recorded path names a prim that does not exist.

That failed silently end to end: 355 components selected, four chosen, none
of their prims resolved, the defect loop skipped all four, and the job order
reported success with a count of zero.
"""

from __future__ import annotations

import unittest

from pcb_aoi.board_metadata import (
    BoardMetadataRequest,
    SidecarMetadataProvider,
)

_SIDECAR = {
    "schema_version": "v1.1",
    "board_prim_path": "/Board_X",
    "board_name": "Board_X",
    "components": [
        {
            "name": "RES_SMD_0603",
            "family": "RES",
            "prototype_path": "/Board_X/_Prototypes/RES_SMD_0603",
            "prim_path": "/Board_X/Components/Top/RES_SMD_0603",
            "instance_paths": ["/Board_X/Components/Top/RES_SMD_0603/R1"],
        },
    ],
    "component_instances": [
        {
            "path": "/Board_X/Components/Top/RES_SMD_0603/R1",
            "refdes": "R1",
            "type": "RES_SMD_0603",
            "side": "Top",
            "body_prims": ["/Board_X/Components/Top/RES_SMD_0603/R1/Body"],
            "solder_contact_prims": [
                "/Board_X/Components/Top/RES_SMD_0603/R1/Contact_1",
            ],
        },
    ],
}


def _load(board_root=None):
    """Load the fixture, optionally against a different board root."""
    return SidecarMetadataProvider().load(
        BoardMetadataRequest(
            board_root=board_root,
            source_identifier="<test>",
            sidecar_data=_SIDECAR,
        ),
    )


class SidecarRebaseTest(unittest.TestCase):
    """Every path in the document moves, or none of them do."""

    def test_the_document_is_read_as_written_by_default(self):
        component = _load().components[0]
        self.assertEqual(component.prim_path,
                         "/Board_X/Components/Top/RES_SMD_0603/R1")
        self.assertEqual(component.board_root, "/Board_X")

    def test_a_different_board_root_moves_the_component_path(self):
        component = _load("/World/Board_X").components[0]
        self.assertEqual(component.prim_path,
                         "/World/Board_X/Components/Top/RES_SMD_0603/R1")

    def test_the_same_board_root_is_not_a_rewrite(self):
        self.assertEqual(_load("/Board_X").components[0].prim_path,
                         _load().components[0].prim_path)

    def test_nested_paths_move_too(self):
        # The pool is not the only thing keyed by these: the tombstone
        # filter indexes contacts by path, and a half-rebased document
        # turns "moved nothing" into "found nothing".
        component = _load("/World/Board_X").components[0]
        self.assertEqual(
            component.body_prim_paths,
            ("/World/Board_X/Components/Top/RES_SMD_0603/R1/Body",))
        self.assertEqual(
            component.contact_prim_paths,
            ("/World/Board_X/Components/Top/RES_SMD_0603/R1/Contact_1",))

    def test_prototype_and_group_paths_move_too(self):
        component = _load("/World/Board_X").components[0]
        self.assertEqual(component.group_path,
                         "/World/Board_X/Components/Top/RES_SMD_0603")
        prototype = _load("/World/Board_X").prototypes[0]
        self.assertEqual(prototype.prim_path,
                         "/World/Board_X/_Prototypes/RES_SMD_0603")

    def test_the_board_root_the_records_carry_is_the_one_read_against(self):
        # Before this, board_root came from the request while prim_path came
        # from the document, so the record contradicted itself.
        component = _load("/World/Board_X").components[0]
        self.assertEqual(component.board_root, "/World/Board_X")
        self.assertTrue(component.prim_path.startswith(component.board_root))

    def test_a_component_lookup_still_finds_the_rebased_part(self):
        catalog = _load("/World/Board_X")
        # by-type lookup matches both the type and the package name, so
        # the same part comes back twice; what matters here is that it is
        # found at all, and at the rebased path.
        self.assertEqual(
            {c.prim_path for c in catalog.components_by_type("RES_SMD_0603")},
            {"/World/Board_X/Components/Top/RES_SMD_0603/R1"})

    def test_a_board_root_that_merely_shares_a_prefix_is_not_confused(self):
        # "/Board_XY" must not be rewritten by a rule about "/Board_X".
        catalog = _load("/World/Board_X")
        self.assertNotIn("/Board_XY",
                         " ".join(c.prim_path for c in catalog.components))


if __name__ == "__main__":
    unittest.main()
