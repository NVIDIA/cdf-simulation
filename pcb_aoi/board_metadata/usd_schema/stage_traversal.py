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

"""Traverse composed PCBA stages, including referenced instance proxies."""

from __future__ import annotations

from collections.abc import Iterator

from pxr import Usd


def iter_stage_prims(stage: Usd.Stage) -> Iterator[Usd.Prim]:
    """Yield composed stage prims and instance proxies.

    Args:
        stage: Open composed USD stage.

    Yields:
        Prims in composed traversal order, including instance proxies.

    Example:

    .. code-block:: python

        from pxr import Usd

        from pcb_aoi.board_metadata.usd_schema.stage_traversal import (
            iter_stage_prims,
        )

        stage = Usd.Stage.Open("board.usda")  # doctest: +SKIP
        print(sum(1 for _ in iter_stage_prims(stage)))  # doctest: +SKIP

    """
    yield from Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())


def iter_prim_subtree(root: Usd.Prim) -> Iterator[Usd.Prim]:
    """Yield a composed prim subtree and its instance proxies.

    Args:
        root: Root prim of the subtree.

    Yields:
        Root prim and composed descendants in traversal order.

    Example:

    .. code-block:: python

        print([str(prim.GetPath()) for prim in iter_prim_subtree(component)])

    """
    predicate = (
        Usd.PrimAllPrimsPredicate
        if root.IsAbstract() or not root.IsDefined()
        else Usd.PrimDefaultPredicate
    )
    yield from Usd.PrimRange(root, Usd.TraverseInstanceProxies(predicate))


def iter_component_prims(stage: Usd.Stage, board_root: str) -> Iterator[Usd.Prim]:
    """Yield component prims below a board root.

    Args:
        stage: Open composed USD stage.
        board_root: Absolute board-root prim path.

    Yields:
        Prims carrying `pcba:component:refdes` metadata.

    Raises:
        ValueError: If `board_root` does not identify a valid prim.

    Example:

    .. code-block:: python

        components = list(iter_component_prims(stage, "/Board"))
        print(len(components))  # doctest: +SKIP

    """
    root = stage.GetPrimAtPath(board_root)
    if not root or not root.IsValid():
        raise ValueError(f"board root {board_root!r} does not exist")
    predicate = (
        Usd.PrimAllPrimsPredicate
        if root.IsAbstract() or not root.IsDefined()
        else Usd.PrimDefaultPredicate
    )
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(predicate)):
        if prim.HasAttribute("pcba:component:refdes"):
            yield prim
