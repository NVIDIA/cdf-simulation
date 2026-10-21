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

"""Bake a triangle mesh to a dense signed-distance grid.

Convention throughout this project: `f > 0` OUTSIDE the solid, `f < 0` inside,
`= 0` on the surface (see geometry/sdf.py). Note that this is the opposite of
trimesh's `signed_distance`, which is positive inside.

This is the slowest step of a cold solve -- every joint bakes an n^3 grid of the
component before Evolver ever starts -- so the default backend is a Warp kernel
that queries the BVH `warp.Mesh` builds for us. `mesh_sdf_grid_trimesh` is the
original CPU implementation, kept because it is the reference the Warp path is
validated against.

Sign, not distance, is the hard part: the asset meshes are not reliably closed.
Both backends therefore use a sign test that degrades gracefully on open meshes
-- trimesh's ray-parity containment on a repaired copy, and Warp's generalised
(fast) winding number. Where the two disagree it is deep inside the solid, which
costs nothing: the keep-out penalty only acts where f < 0 and the solder never
gets there.
"""

from __future__ import annotations

import struct

import numpy as np


# ---------------------------------------------------------------- mesh repair

def _repair(V, F):
    """Weld and consistently orient the mesh, returning plain arrays.

    Both sign tests care about orientation -- ray parity needs a closed surface,
    the winding number needs consistently wound triangles -- and the raw USD
    assets give us neither: duplicated vertices along part seams leave cracks
    that a ray can slip through. trimesh's merge/fix pass is cheap next to the
    bake itself, so both backends pay it and stay comparable.
    """
    import trimesh
    m = trimesh.Trimesh(vertices=np.asarray(V, float),
                        faces=np.asarray(F, int).reshape(-1, 3), process=True)
    m.merge_vertices()
    m.fix_normals()
    return m


def _axes(lo, hi, n):
    """Sample positions per axis, matching np.meshgrid(..., indexing='ij')."""
    return [np.linspace(float(lo[i]), float(hi[i]), int(n[i])) for i in range(3)]


# ------------------------------------------------------------ trimesh (legacy)

def mesh_sdf_grid_trimesh(V, F, lo, hi, n):
    """Reference bake: trimesh's CPU closest-point + containment test.

    Correct but slow -- it is an O(points) query with a Python-level loop inside
    trimesh, and a 64^3 grid is a quarter of a million queries. Kept as the
    accuracy reference for the Warp backend, not as a runtime path.
    """
    import trimesh
    m = _repair(V, F)
    X, Y, Z = np.meshgrid(*_axes(lo, hi, n), indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    # Chunked because trimesh's closest_point broadcasts candidate triangles
    # against the whole query array at once: handed all 128^3 = 2.1M points in
    # one call it is killed by the OOM reaper on a 64 GB box. The query is
    # pointwise, so chunking changes nothing but the peak footprint.
    out = np.empty(len(P))
    step = 1 << 18
    for i in range(0, len(P), step):
        # trimesh returns +inside/-outside; our convention is the opposite
        out[i:i + step] = -trimesh.proximity.signed_distance(m, P[i:i + step])
    return out.reshape(tuple(int(k) for k in n))


# ------------------------------------------------------------------ warp (GPU)

_WARP_READY = False


def _warp():
    """Import and initialise Warp once, returning the module.

    Warp is already a dependency (the viewer uses it) and is Apache-2.0, and
    `wp.Mesh` builds an SAH BVH over the triangles internally -- there is no
    reason to hand-roll one. Kernel compilation is cached by Warp across
    processes, so only the very first bake on a machine pays for it.
    """
    global _WARP_READY
    import warp as wp
    if not _WARP_READY:
        wp.init()
        _WARP_READY = True
    return wp


def _build_kernel():
    """Define the sampling kernel lazily, so importing this module is free."""
    wp = _warp()

    @wp.kernel
    def _sdf_kernel(mesh: wp.uint64,
                    lo: wp.vec3,
                    step: wp.vec3,
                    ny: int,
                    nz: int,
                    max_dist: float,
                    out: wp.array(dtype=float)):
        i, j, k = wp.tid()
        p = wp.vec3(lo[0] + step[0] * float(i),
                    lo[1] + step[1] * float(j),
                    lo[2] + step[2] * float(k))
        # Winding-number sign rather than the (cheaper) pseudo-normal one: the
        # pseudo-normal test assumes a watertight, non-self-intersecting mesh,
        # and ours is neither. The generalised winding number stays meaningful
        # across cracks and self-intersections, which is exactly our failure
        # mode. accuracy=2.0 / threshold=0.5 are Warp's documented defaults.
        q = wp.mesh_query_point_sign_winding_number(mesh, p, max_dist, 2.0, 0.5)
        d = max_dist
        if q.result:
            cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
            d = wp.length(p - cp)
            if q.sign < 0.0:
                d = -d
        # C order, so this matches numpy's reshape of a meshgrid(indexing='ij')
        out[(i * ny + j) * nz + k] = d

    return _sdf_kernel


_KERNEL = None


def mesh_sdf_grid_warp(V, F, lo, hi, n, device=None):
    """Bake on the GPU (or Warp's CPU device) via the BVH inside `wp.Mesh`."""
    global _KERNEL
    wp = _warp()
    if _KERNEL is None:
        _KERNEL = _build_kernel()

    if device is None:
        device = "cuda:0" if wp.is_cuda_available() else "cpu"

    m = _repair(V, F)
    # float32 is what wp.vec3 is; the resulting ~1e-5 mm disagreement with the
    # float64 trimesh path is four orders below the finest grid cell we bake
    # (0.02 mm at 128^3), so it is invisible to the solver.
    Vm = np.ascontiguousarray(m.vertices, dtype=np.float32)
    Fm = np.ascontiguousarray(m.faces, dtype=np.int32).reshape(-1)

    with wp.ScopedDevice(device):
        # support_winding_number=True builds the per-BVH-node solid-angle data
        # the fast winding number needs. It is NOT optional: without it the
        # query silently falls back to a ray-parity test (native/mesh.h), i.e.
        # exactly the sign rule we switched away from, and the call still
        # "succeeds" -- so a missing flag here would look like a correct bake.
        mesh = wp.Mesh(points=wp.array(Vm, dtype=wp.vec3),
                       indices=wp.array(Fm, dtype=wp.int32),
                       support_winding_number=True)

        n = tuple(int(k) for k in n)
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        # linspace endpoints: n==1 degenerates to the low corner, as it does in
        # the trimesh path.
        step = np.array([(hi[i] - lo[i]) / (n[i] - 1) if n[i] > 1 else 0.0
                         for i in range(3)])
        # max_dist prunes the BVH descent, so it must exceed the largest distance
        # any sample can have -- otherwise far-field samples come back with no
        # hit at all. The diagonal of the union of grid and mesh bounds bounds it.
        allo = np.minimum(lo, Vm.min(0))
        alhi = np.maximum(hi, Vm.max(0))
        max_dist = float(np.linalg.norm(alhi - allo)) * 1.01 + 1.0

        out = wp.zeros(n[0] * n[1] * n[2], dtype=float)
        wp.launch(_KERNEL, dim=n,
                  inputs=[mesh.id, wp.vec3(*lo), wp.vec3(*step),
                          n[1], n[2], max_dist],
                  outputs=[out])
        wp.synchronize()
        return out.numpy().astype(np.float64).reshape(n)


# ------------------------------------------------------------------ public API

def mesh_sdf_grid(V, F, lo, hi, n, backend="auto"):
    """Bake a triangle mesh to a signed-distance grid on [lo,hi] at n^3 samples.

    Positive outside the solid, negative inside. `backend` is "auto" (Warp if it
    imports at all -- it falls back to its own CPU device without CUDA), "warp",
    or "trimesh".
    """
    if backend == "trimesh":
        return mesh_sdf_grid_trimesh(V, F, lo, hi, n)
    if backend == "warp":
        return mesh_sdf_grid_warp(V, F, lo, hi, n)
    try:
        return mesh_sdf_grid_warp(V, F, lo, hi, n)
    except Exception as e:                                  # noqa: BLE001
        # A missing/broken Warp must not take the solver down: the bake is the
        # only thing that needs it and the CPU path still gives the right answer.
        print(f"  [sdf] warp bake unavailable ({e}); falling back to trimesh")
        return mesh_sdf_grid_trimesh(V, F, lo, hi, n)


# ---------------------------------------------------- grid file for the .fe



MAGIC = b"SDFGRID2"
VERSION = 2


def read_grid(path):
    """Inverse of ``write_grid``: (values[i,j,k], lo, hi)."""
    with open(path, "rb") as fh:
        magic = fh.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{path}: not a solder-evolver grid (magic {magic!r})")
        (version,) = struct.unpack("<i", fh.read(4))
        if version != VERSION:
            raise ValueError(f"{path}: grid version {version}; "
                             f"this reader expects {VERSION}")
        shape = struct.unpack("<3i", fh.read(12))
        lo = np.asarray(struct.unpack("<3d", fh.read(24)), float)
        hi = np.asarray(struct.unpack("<3d", fh.read(24)), float)
        flat = np.frombuffer(fh.read(), dtype=np.float64)
    if flat.size != int(np.prod(shape)):
        raise ValueError(f"{path}: truncated grid -- {flat.size} values "
                         f"for shape {shape}")
    v = flat.reshape(shape[::-1]).transpose(2, 1, 0)   # undo the x-fastest ravel
    return np.ascontiguousarray(v), lo, hi


def keepout_surface(path):
    """Marching-cubes surface of a baked keep-out grid, at the zero level.

    The visual truth of what the solve is (supposed to be) colliding with --
    grid resolution, inflate and floor folded in, exactly as Evolver samples
    it. Returns (V, F) in world coordinates, or None when the zero level does
    not intersect the box.
    """
    import warp as wp

    values, lo, hi = read_grid(path)
    nx, ny, nz = values.shape
    device = wp.get_device()
    field = wp.array3d(values.astype(np.float32), dtype=wp.float32,
                       device=device)
    mc = wp.MarchingCubes(nx, ny, nz,
                          max_verts=1_000_000, max_tris=2_000_000,
                          device=device)
    mc.surface(field=field, threshold=0.0)
    V = mc.verts.numpy()
    F = mc.indices.numpy().reshape(-1, 3)
    if not len(F):
        return None
    # MC works in GRID INDEX space; scale to the sample-corner bounds.
    scale = (hi - lo) / (np.array([nx, ny, nz], float) - 1.0)
    return V * scale + lo, F.astype(np.int64)


def write_grid(path, values, lo, hi):
    """`values` is [nx,ny,nz] indexed [i,j,k]; lo/hi are the sample corners."""
    v = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if v.ndim != 3:
        raise ValueError(f"values must be 3-D, got {v.shape}")
    if min(v.shape) < 4:
        raise ValueError(f"each axis needs >= 4 samples for cubic interp, got {v.shape}")
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if not np.all(hi > lo):
        raise ValueError(f"bounds must increase: lo={lo} hi={hi}")
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<i", VERSION))
        fh.write(struct.pack("<3i", *v.shape))
        fh.write(struct.pack("<3d", *lo))
        fh.write(struct.pack("<3d", *hi))
        # file order is x fastest -> transpose (i,j,k) to (k,j,i) before ravel
        fh.write(np.ascontiguousarray(v.transpose(2, 1, 0)).tobytes())
    return path


def sample(fn, lo, hi, n):
    """Sample an analytic SDF `fn(P[N,3]) -> [N]` on an n^3 (or [nx,ny,nz]) grid."""
    n = (n, n, n) if np.isscalar(n) else tuple(n)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    ax = [np.linspace(lo[i], hi[i], n[i]) for i in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return fn(P).reshape(n)


# ------------------------------------------------------------- analytic SDFs
# Exact fields, so a test can separate interpolation error from everything else.

def sphere_sdf(center, radius):
    c = np.asarray(center, float)
    return lambda P: np.linalg.norm(np.asarray(P, float) - c, axis=1) - radius


def box_sdf(center, half):
    """Exact SDF of an axis-aligned box (Inigo Quilez)."""
    c = np.asarray(center, float)
    h = np.asarray(half, float)

    def f(P):
        q = np.abs(np.asarray(P, float) - c) - h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    return f


def plane_sdf(z0):
    """Half-space z >= z0, i.e. a floor. Solid is below."""
    return lambda P: np.asarray(P, float)[:, 2] - z0


def bake_keepout_sdf(path, V, F, lo, hi, res=(64, 64, 64), floor_z=None,
                     inflate=0.0):
    """Bake a component keep-out grid and write it beside a datafile.

    The EXPERIMENTAL collision energy: emit_datafile() turns the written grid
    into a penalty on the free surface (one-sided constraints are inert under
    minimisation, so it must be an energy). `floor_z` folds the board plane into
    the same field -- solder may not pass below the pad either; `inflate` grows
    the keep-out by a skin thickness.

    Returns the grid array (the file is the real product; Evolver resolves the
    path against its working directory, so put `path` next to the datafile).
    """
    import numpy as _np

    g = mesh_sdf_grid(_np.asarray(V, float), _np.asarray(F, int).reshape(-1, 3),
                      lo, hi, tuple(res))
    if floor_z is not None:
        zz = _np.linspace(lo[2], hi[2], res[2])[None, None, :]
        g = _np.minimum(g, _np.broadcast_to(zz - float(floor_z), g.shape))
    if inflate:
        g = g - float(inflate)
    write_grid(path, g, lo, hi)
    return g
