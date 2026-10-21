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

"""Export a solder scene (solder frame + solid parts) to USD with PBR materials.

Writes a standalone .usd(a) with the solder meshes and the chip/board solids,
each bound to a UsdPreviewSurface material (solder = shiny metal, body = ceramic,
leads/pads/terminations = metal, board = FR4 green). Suitable for a photoreal RTX
render (Isaac Sim / Kit).
"""

from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

# name-substring -> (diffuse rgb, metallic, roughness)
_MATERIALS = {
    "solder": ((0.78, 0.78, 0.82), 1.0, 0.28),
    "body":   ((0.32, 0.26, 0.24), 0.0, 0.55),   # ceramic / package
    "term":   ((0.80, 0.80, 0.83), 1.0, 0.30),   # metallized termination
    "lead":   ((0.78, 0.78, 0.80), 1.0, 0.32),   # metal lead
    "pad":    ((0.72, 0.55, 0.35), 1.0, 0.35),   # copper pad
    "board":  ((0.06, 0.22, 0.10), 0.0, 0.45),   # FR4 solder mask green
}
_DEFAULT_MAT = ((0.6, 0.6, 0.6), 0.0, 0.5)


def _mat_for(name: str):
    for key, spec in _MATERIALS.items():
        if key in name:
            return key, spec
    return "default", _DEFAULT_MAT


def _make_material(stage, path, rgb, metallic, roughness):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    if metallic > 0.5:
        sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _add_mesh(stage, path, verts, faces, mat):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in verts])
    m.CreateFaceVertexCountsAttr([3] * len(faces))
    m.CreateFaceVertexIndicesAttr(np.asarray(faces, np.int32).reshape(-1).tolist())
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdShade.MaterialBindingAPI(m).Bind(mat)
    return m


def export_scene(usd_path, frames, solids, *, up="Z", meters_per_unit=0.001):
    """Write solder `frames` (list of [ev.Mesh]) + `solids` {name:(v,idx,nrm)} to USD."""
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up == "Z" else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    looks = "/World/Looks"

    # one material per class, reused
    mats = {}
    for key, spec in list(_MATERIALS.items()) + [("default", _DEFAULT_MAT)]:
        mats[key] = _make_material(stage, f"{looks}/{key}", *spec)

    # solder
    frame = frames[-1] if frames else []
    for j, mesh in enumerate(frame):
        v = np.asarray(mesh.vertices, float)
        f = np.asarray(mesh.triangles, np.int32)
        _add_mesh(stage, f"/World/Solder/solder_{j}", v, f, mats["solder"])

    # solids
    for name, (v, idx, _n) in solids.items():
        key, _ = _mat_for(name)
        faces = np.asarray(idx, np.int32).reshape(-1, 3)
        _add_mesh(stage, f"/World/Solids/{name}", np.asarray(v, float), faces, mats[key])

    stage.GetRootLayer().Save()
    return str(usd_path)
