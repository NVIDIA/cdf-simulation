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

"""Resolve stable identities from composed USD references and prim paths."""

from __future__ import annotations

from collections.abc import Iterator

from pxr import Sdf, Usd


def _reference_items(list_operation: object) -> Iterator[object]:
    """Yield authored reference items from an Sdf list operation.

    Args:
        list_operation: Reference list operation authored on a prim spec.

    Yields:
        Authored references in strongest-to-weakest list order.

    """
    for field_name in (
        "explicitItems",
        "prependedItems",
        "addedItems",
        "appendedItems",
    ):
        yield from getattr(list_operation, field_name, ())


def referenced_asset_path(prim: Usd.Prim) -> str | None:
    """Resolve the first authored external reference for a component prim.

    Args:
        prim: Composed component prim.

    Returns:
        Resolved asset path, or None when the component has no external
        reference.

    Example:

    .. code-block:: python

        print(referenced_asset_path(component))  # doctest: +SKIP
        C:/boards/protos/R_0402.usda

    """
    for prim_spec in prim.GetPrimStack():
        for reference in _reference_items(prim_spec.referenceList):
            asset_path = str(getattr(reference, "assetPath", "") or "")
            if asset_path:
                return str(
                    Sdf.ComputeAssetPathRelativeToLayer(
                        prim_spec.layer,
                        asset_path,
                    ),
                )
    return None


def relative_prim_path(root: Usd.Prim, descendant: Usd.Prim) -> str:
    """Return a descendant path relative to a component root.

    Args:
        root: Component root prim.
        descendant: Descendant prim to express relative to `root`.

    Returns:
        Slash-separated descendant path without the component prefix.

    Raises:
        ValueError: If `descendant` is not below `root`.

    Example:

    .. code-block:: python

        print(relative_prim_path(component, pin))  # doctest: +SKIP
        pin_1

    """
    root_path = str(root.GetPath()).rstrip("/")
    descendant_path = str(descendant.GetPath())
    prefix = f"{root_path}/"
    if not descendant_path.startswith(prefix):
        raise ValueError(
            f"prim {descendant_path!r} is not below component {root_path!r}",
        )
    return descendant_path.removeprefix(prefix)
