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

"""Board geometry that carries a semantic class of its own.

``annotations.label_geometry`` labels geometry that is not a component. Which
geometry a board has -- solder, terminals, pads, holes -- is domain vocabulary,
so the names and their metadata sources live here rather than in ``cdf``.
"""

from __future__ import annotations

from .board_metadata import MetadataCapability

#: Board-level Xform containing annotation-only segmentation proxy geometry.
SEGMENTATION_PROXIES_SCOPE = "SegmentationProxies"

#: Diagnostic display color for annotation-only geometry in ordinary USD views.
SEGMENTATION_MASK_DISPLAY_COLOR = (1.0, 0.0, 1.0)


def ensure_segmentation_proxies_parent(
    stage: object,
    output_prim_path: str,
) -> str | None:
    """Ensure a SegmentationProxies ancestor is a default-hidden Xform.

    Args:
        stage: Open USD stage receiving annotation-only proxy geometry.
        output_prim_path: Absolute path at or below the generated proxy root.

    Returns:
        SegmentationProxies ancestor path, or None when the output path has no
        ancestor with that name.

    Example:

    .. code-block:: python

        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        path = ensure_segmentation_proxies_parent(
            stage,
            "/World/Board/SegmentationProxies/HolePlane",
        )
        print(path)
        # /World/Board/SegmentationProxies

    """
    from pxr import Sdf, UsdGeom

    path = Sdf.Path(output_prim_path)
    while path and path != Sdf.Path.absoluteRootPath:
        if path.name == SEGMENTATION_PROXIES_SCOPE:
            masks = UsdGeom.Xform.Define(stage, path)
            masks.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            return str(path)
        path = path.GetParentPath()
    return None


#: Geometry produced by an operation rather than resolved from the catalog.
OPERATION_TARGETS = ("solder",)

#: Catalog-resolved targets mapped to the capability they need and the catalog
#: attribute holding their records.
CATALOG_TARGETS = {
    "terminals": (MetadataCapability.TERMINALS, "terminals"),
    "pads": (MetadataCapability.PADS, "pads"),
    "holes": (MetadataCapability.HOLES, "holes"),
}

#: Stage-traversal targets resolved by ``pcba:mesh:role`` rather than the
#: catalog, mapped to the role value their prims carry. ``barrels`` is board
#: geometry (the copper inner wall of a drilled hole). ``body`` and ``marking``
#: are package geometry inside a component prototype: the package body and the
#: pin-1 dot / polarity band / notch mesh. Labelling those separately is what
#: lets a mask say where the body ends and where the polarity marker sits;
#: component-level labels collapse both into one class.
TRAVERSAL_ROLES: dict[str, str] = {
    "barrels": "board_barrel",
    "body": "pkg_body",
    "marking": "pkg_marking",
}

#: Traversal targets whose prims live inside a component prototype. Their
#: component is the enclosing component prim, not a relationship target.
PACKAGE_TRAVERSAL_TARGETS: frozenset[str] = frozenset({"body", "marking"})

TRAVERSAL_TARGETS = tuple(TRAVERSAL_ROLES)

#: Every geometry target a pcb-aoi job may label.
GEOMETRY_TARGETS = (
    frozenset(OPERATION_TARGETS)
    | frozenset(CATALOG_TARGETS)
    | frozenset(TRAVERSAL_TARGETS)
)

#: USD mesh-role shortnames (``pcba:mesh:role`` without the ``board_`` prefix)
#: that per-pass visibility may show or hide. Each name maps to the role value
#: ``board_{name}``; matching prims are restored immediately after the pass.
BOARD_GEOMETRY_ROLES: frozenset[str] = frozenset(
    {
        "copper",
        "mask",
        "pad",
        "paste",
        "hole",
        "barrel",
        "substrate",
        "pin_tip",
    }
)


def traversal_role(name: str) -> str:
    """Return the ``pcba:mesh:role`` value a traversal target labels.

    Args:
        name: Traversal-resolved geometry target.

    Returns:
        The role token carried by the prims of that target.

    Raises:
        KeyError: If the target is not traversal-resolved.

    Example:

    .. code-block:: python

        print(traversal_role("marking"))
        # pkg_marking

    """
    return TRAVERSAL_ROLES[name]


def catalog_target(name: str) -> tuple[MetadataCapability, str]:
    """Return the metadata capability and catalog attribute for one target.

    Args:
        name: Catalog-resolved geometry target.

    Returns:
        The capability the catalog must supply and the attribute holding the
        records for that target.

    Raises:
        KeyError: If the target is not catalog-resolved.

    Example:

    .. code-block:: python

        capability, attribute = catalog_target("holes")

    """
    return CATALOG_TARGETS[name]
