#!/usr/bin/env python
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

"""Photoreal RTX render of the exported solder-variant USDs, via Isaac Sim.

Run with Isaac Sim's python (NOT the repo venv):

    <isaac>/python.sh scripts/isaac_render.py \
        --manifest <dir>/manifest.json --out <dir>

Boots a headless SimulationApp, then for each scene in the manifest opens the
USD, adds dome + key lights, places a camera per requested view, path-traces, and
writes a PNG per (scene, view). Amortizes the ~90s Kit boot across all images.
"""

import argparse
import json
import os
import sys

from isaacsim import SimulationApp

_ap = argparse.ArgumentParser()
_ap.add_argument("--manifest", required=True)
_ap.add_argument("--out", required=True)
_ap.add_argument("--res", type=int, nargs=2, default=[1200, 900])
_ap.add_argument("--zoom", type=float, default=1.0,
                 help="camera framing: <1 moves the eye closer (object fills more of the frame)")
_ap.add_argument("--spp", type=int, default=64)
_ap.add_argument("--subframes", type=int, default=48)
_ap.add_argument("--focus", default="/World/Solder",
                 help="prim the CAMERA frames on. Framing the whole /World puts "
                      "the board slab and the package in the bbox, and the joint "
                      "-- the only thing being judged -- ends up a few pixels. "
                      "Lights still use the full scene. Falls back to /World if "
                      "this prim is absent or empty.")
args = _ap.parse_args()

sim = SimulationApp({"headless": True})

import carb                                    # noqa: E402
import numpy as np                             # noqa: E402
import omni.usd                                # noqa: E402
import omni.replicator.core as rep             # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux        # noqa: E402

W, H = args.res
_s = carb.settings.get_settings()
_s.set("/rtx/rendermode", "PathTracing")
_s.set("/rtx/pathtracing/spp", int(args.spp))
_s.set("/rtx/pathtracing/totalSpp", int(args.spp * 4))
_s.set("/rtx/pathtracing/maxBounces", 6)


def _png(path, arr):
    from PIL import Image
    a = arr["data"] if isinstance(arr, dict) else arr
    a = np.ascontiguousarray(a)[:, :, :3]
    Image.fromarray(a.astype(np.uint8)).save(path)


def _bbox(stage, root="/World"):
    prim = stage.GetPrimAtPath(root)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    return 0.5 * (lo + hi), max(float(np.linalg.norm(hi - lo)), 1e-4)


def _eye(view, c, span, zoom=1.0):
    d = {
        "persp":   (2.0, -2.0, 1.3),
        "side":    (0.02, -3.2, 0.12),
        "bottom":  (0.02, -0.02, -3.2),
        "profile": (1.6, -2.2, -1.2),   # THT: from the side and below, sees the hanging solder
        "top":     (0.02, -0.02, 3.2),
    }.get(view, (2.0, -2.0, 1.3))
    return c + span * zoom * np.array(d)


def _place_camera(stage, path, eye, target, up=(0, 0, 1)):
    cam = UsdGeom.Camera.Define(stage, path)
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse()
    x = UsdGeom.Xformable(cam)
    x.ClearXformOpOrder()
    x.AddTransformOp().Set(m)
    cam.CreateFocalLengthAttr(35.0)
    cam.CreateHorizontalApertureAttr(24.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.001, 1.0e6))
    return path


def _lights(stage, span, center):
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(600.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))
    for i, (dx, dy, dz, inten) in enumerate([(1, -1, 2, 8000), (-1.5, -0.5, 1.2, 4000),
                                             (0.5, 1.5, 1.0, 3000)]):
        L = UsdLux.DistantLight.Define(stage, f"/World/Lights/Key{i}")
        L.CreateIntensityAttr(float(inten))
        L.CreateAngleAttr(1.5)
        d = -np.array([dx, dy, dz], float); d /= np.linalg.norm(d)
        xf = UsdGeom.Xformable(L)
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0, 0, 0), Gf.Vec3d(*d), Gf.Vec3d(0, 0, 1)).GetInverse()
        xf.AddTransformOp().Set(m)


def main():
    manifest = json.load(open(args.manifest))
    os.makedirs(args.out, exist_ok=True)
    ctx = omni.usd.get_context()
    out_manifest = []

    for entry in manifest:
        ctx.open_stage(entry["usd"])
        for _ in range(8):
            sim.update()
        stage = ctx.get_stage()
        center, span = _bbox(stage)
        _lights(stage, span, center)
        # Frame the SOLDER, light the scene. The joint is a fraction of a
        # millimetre against a board slab several millimetres across, so a
        # whole-world bbox renders it as a speck.
        foc = _bbox(stage, args.focus) or (center, span)
        fc, fs = foc
        # camera + render product (recreated per stage; annotator per stage)
        _place_camera(stage, "/World/RenderCam", _eye("persp", fc, fs, args.zoom), fc)
        rp = rep.create.render_product("/World/RenderCam", (W, H))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([rp])
        for view in entry.get("views", ["persp"]):
            _place_camera(stage, "/World/RenderCam", _eye(view, fc, fs, args.zoom), fc)
            for _ in range(3):
                sim.update()
            rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
            png = os.path.join(args.out, f"{entry['kind']}_{entry['defect']}_{view}.png")
            _png(png, rgb.get_data())
            print(f"[render] {png}", flush=True)
            out_manifest.append({**entry, "view": view, "png": png})
        rgb.detach()
        try:
            rp.destroy()
        except Exception:
            pass

    json.dump(out_manifest, open(os.path.join(args.out, "rendered.json"), "w"), indent=2)
    print(f"[render] wrote {len(out_manifest)} images", flush=True)


try:
    main()
finally:
    sim.close()
