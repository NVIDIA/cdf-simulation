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

"""Generate the through-hole coupon that carries the board-metadata schema.

The ``THT_TO254P762H300_U34_coupon.usdc`` coupon (not in the public release) predates
``pcba:mesh:role``/``pcba:board:holeRadiusMm``, so it cannot exercise the discovery
half of ``pin_variation``. This emits a small board that does, **entirely from
this script** -- there is no third-party geometry in it and so nothing to
review for release.

Deliberately faithful in the two ways discovery depends on:

* the barrel mesh is the PLATING, wider than the bore and standing proud of
  both board faces, so a pairing that demanded containment would reject it;
* the component's geometry sits inside an abstract ``class`` prototype behind
  an instanceable reference, the way a panelised export does, so a traversal
  using the default predicate finds nothing.

    python tests/functional/resources/make_tht_schema_coupon.py
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

BOARD_TOP, BOARD_BOTTOM = 0.0, -1.6
BOARD_HALF = 6.0
PIN_HALF = 0.25
BORE_RADIUS = 0.42
PLATING_RADIUS = 0.62
PAD_RADIUS = 0.95
CAP_PROUD = 0.05
PIN_PITCH = 2.54
PROTRUSION = 1.95
BODY_HEIGHT = 4.6

_LEADS = (("U1", -PIN_PITCH), ("U1", 0.0), ("U1", PIN_PITCH))


def _box(stage, path, centre, half_extent):
    """Author an axis-aligned box as a real, renderable mesh.

    Args:
        stage: Stage to author on.
        path: Prim path.
        centre: Box centre.
        half_extent: Half size on each axis.

    Returns:
        The mesh's prim.
    """
    cx, cy, cz = centre
    hx, hy, hz = half_extent
    points = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    faces = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr().Set([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set(
        [index for face in faces for index in face],
    )
    mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(cx - hx, cy - hy, cz - hz),
            Gf.Vec3f(cx + hx, cy + hy, cz + hz),
        ],
    )
    return mesh.GetPrim()


def _tube(stage, path, centre_xy, radius, z_low, z_high, segments=24):
    """Author an open cylinder standing for a plated barrel.

    Args:
        stage: Stage to author on.
        path: Prim path.
        centre_xy: Bore centre.
        radius: Outer radius of the plating.
        z_low: Lower end.
        z_high: Upper end.
        segments: Radial resolution.

    Returns:
        The mesh's prim.
    """
    cx, cy = centre_xy
    points, faces = [], []
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        x, y = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
        points.extend([(x, y, z_low), (x, y, z_high)])
    for index in range(segments):
        a = 2 * index
        b = 2 * ((index + 1) % segments)
        faces.append((a, b, b + 1, a + 1))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr().Set([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set(
        [index for face in faces for index in face],
    )
    mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(cx - radius, cy - radius, z_low),
            Gf.Vec3f(cx + radius, cy + radius, z_high),
        ],
    )
    return mesh.GetPrim()


def _annulus(stage, path, centre_xy, inner, outer, z, segments=24):
    """Author a flat annular ring standing for a solder-side pad.

    A through-hole joint is sized from its pad, so a coupon without one has
    nothing for the solder generator to discover.

    Args:
        stage: Stage to author on.
        path: Prim path.
        centre_xy: Ring centre.
        inner: Inner radius.
        outer: Outer radius.
        z: Height of the ring.
        segments: Radial resolution.

    Returns:
        The mesh's prim.
    """
    cx, cy = centre_xy
    points, faces = [], []
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        points.append((cx + inner * cos_a, cy + inner * sin_a, z))
        points.append((cx + outer * cos_a, cy + outer * sin_a, z))
    for index in range(segments):
        a = 2 * index
        b = 2 * ((index + 1) % segments)
        faces.append((a, a + 1, b + 1, b))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr().Set([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set(
        [index for face in faces for index in face],
    )
    mesh.CreateExtentAttr().Set(
        [Gf.Vec3f(cx - outer, cy - outer, z), Gf.Vec3f(cx + outer, cy + outer, z)],
    )
    return mesh.GetPrim()


def _colour(stage, prim, rgb):
    """Bind a flat preview surface so the coupon renders as something.

    Args:
        stage: Stage to author on.
        prim: Prim to bind.
        rgb: Diffuse colour.
    """
    path = Sdf.Path(f"/Coupon/Looks/{prim.GetName()}_mat")
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*rgb),
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface",
    )
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def build(destination: Path, bore_radius: float = BORE_RADIUS) -> Path:
    """Write the coupon.

    Args:
        destination: Output ``.usdc`` path.
        bore_radius: Drilled radius. The default leaves a square lead roughly
            0.056 mm of clearance, so the tilt clamp holds it near 4 degrees --
            correct, but too small to see in a picture. Widen it to make the
            tilt axis legible in a figure; the clamp is unchanged, it simply
            has more room to give.

    Returns:
        The path written.
    """
    plating_radius = PLATING_RADIUS + (bore_radius - BORE_RADIUS)
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    root = UsdGeom.Xform.Define(stage, "/Coupon")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/Coupon/Board")

    substrate = _box(
        stage,
        "/Coupon/Board/substrate",
        (0.0, 0.0, 0.5 * (BOARD_TOP + BOARD_BOTTOM)),
        (BOARD_HALF, BOARD_HALF, 0.5 * (BOARD_TOP - BOARD_BOTTOM)),
    )
    substrate.CreateAttribute(
        "pcba:mesh:role",
        Sdf.ValueTypeNames.String,
    ).Set("board_substrate")
    _colour(stage, substrate, (0.05, 0.32, 0.16))

    # The component lives in an ABSTRACT CLASS prototype, behind an
    # instanceable reference -- the shape a panelised export has, and the one a
    # default-predicate traversal cannot see.
    stage.CreateClassPrim("/Prototypes")
    prototype = UsdGeom.Xform.Define(stage, "/Prototypes/TO92")
    # The refdes IS authored on a panelised THT export, and it is what
    # `iter_component_prims` keys on -- a component without it is invisible to
    # the metadata catalog.
    prototype.GetPrim().CreateAttribute(
        "pcba:component:refdes", Sdf.ValueTypeNames.String,
    ).Set("U1")
    prototype.GetPrim().CreateAttribute(
        "pcba:component:packageType", Sdf.ValueTypeNames.String,
    ).Set("TO92")
    prototype.GetPrim().CreateAttribute(
        "pcba:component:leadType", Sdf.ValueTypeNames.String,
    ).Set("TH")
    # `leadType` is what lead DISCOVERY reads; the metadata catalog -- and so
    # the solder solver -- reads `processClass` instead, and without it a
    # through-hole component never reaches the through-hole branch at all: the
    # procedural solver reports "no measurable paste or contact sites", which
    # is indistinguishable from a board that genuinely has none.
    prototype.GetPrim().CreateAttribute(
        "pcba:component:processClass", Sdf.ValueTypeNames.String,
    ).Set("through_hole")
    # The side the BODY sits on -- `pcba:component:side`, NOT `pcba:layer:side`
    # (that one is the pad's, and is resolved through ancestors). Solder is
    # placed on the OPPOSITE face, so "top" here is what puts the joints on the
    # same side as the bottom-side pads authored below.
    prototype.GetPrim().CreateAttribute(
        "pcba:component:side", Sdf.ValueTypeNames.String,
    ).Set("top")

    body = _box(
        stage, "/Prototypes/TO92/body",
        (0.0, 0.0, BOARD_TOP + 0.5 * BODY_HEIGHT),
        (2.6, 2.1, 0.5 * BODY_HEIGHT),
    )
    _colour(stage, body, (0.05, 0.05, 0.06))

    for index, (_refdes, offset) in enumerate(_LEADS, start=1):
        lead = _box(
            stage, f"/Prototypes/TO92/pkg_lead_{index}",
            (offset, 0.0, 0.5 * ((BOARD_BOTTOM - PROTRUSION) + BOARD_TOP + 0.2)),
            (
                PIN_HALF, PIN_HALF,
                0.5 * ((BOARD_TOP + 0.2) - (BOARD_BOTTOM - PROTRUSION)),
            ),
        )
        lead.CreateAttribute("pcba:mesh:role", Sdf.ValueTypeNames.String).Set(
            "pkg_lead",
        )
        lead.CreateAttribute("pcba:pin:number", Sdf.ValueTypeNames.Int).Set(index)
        _colour(stage, lead, (0.72, 0.72, 0.75))

        barrel = _tube(
            stage, f"/Coupon/Board/board_hole_{index}",
            (offset, 0.0), plating_radius,
            BOARD_BOTTOM - CAP_PROUD, BOARD_TOP + CAP_PROUD,
        )
        barrel.CreateAttribute("pcba:mesh:role", Sdf.ValueTypeNames.String).Set(
            "board_hole",
        )
        # The DRILLED radius, which the plating mesh's extent is not.
        barrel.CreateAttribute(
            "pcba:board:holeRadiusMm", Sdf.ValueTypeNames.Float,
        ).Set(bore_radius)
        barrel.CreateAttribute(
            "pcba:board:plated", Sdf.ValueTypeNames.Bool,
        ).Set(True)
        # The terminal-to-hole association, which is what unlocks the
        # `terminal_hole_associations` capability `label_geometry(holes=...)`
        # needs. A panelised THT board authors exactly these two.
        barrel.CreateAttribute(
            "pcba:board:refdes", Sdf.ValueTypeNames.String,
        ).Set("U1")
        barrel.CreateAttribute(
            "pcba:board:componentPinNumber", Sdf.ValueTypeNames.String,
        ).Set(str(index))
        _colour(stage, barrel, (0.78, 0.62, 0.25))

        pad = _annulus(
            stage, f"/Coupon/Board/board_pad_{index}",
            (offset, 0.0), bore_radius, PAD_RADIUS, BOARD_BOTTOM - 0.001,
        )
        pad.CreateAttribute("pcba:mesh:role", Sdf.ValueTypeNames.String).Set(
            "board_pad",
        )
        pad.CreateAttribute(
            "pcba:board:refdes", Sdf.ValueTypeNames.String,
        ).Set("U1")
        pad.CreateAttribute(
            "pcba:board:componentPinNumber", Sdf.ValueTypeNames.String,
        ).Set(str(index))
        pad.CreateAttribute(
            "pcba:layer:side", Sdf.ValueTypeNames.String,
        ).Set("bottom")
        _colour(stage, pad, (0.80, 0.66, 0.30))

    placement = UsdGeom.Xform.Define(stage, "/Coupon/Components/U1")
    placement.GetPrim().GetReferences().AddInternalReference(
        Sdf.Path("/Prototypes/TO92"),
    )
    stage.GetRootLayer().Save()
    return destination


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    print("wrote", build(here / "THT_SCHEMA_U1_coupon.usdc"))
