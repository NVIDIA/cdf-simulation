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

"""Tombstone defect geometry and eligibility helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import omni.usd
from pxr import UsdGeom

from defects_utils.transform_utils import (
    _component_rotation_frame,
    _min_support_projection_after_rotation,
    _normalized,
    _pivot_rotation_transform,
    _project_point_to_support_end,
)

logger = logging.getLogger(__name__)


def _component_type_tokens(cfg: dict[str, Any]) -> list[str]:
    """Return component-type substring filters from a defect config.

    Args:
        cfg: Per-defect configuration.

    Returns:
        Non-empty substring tokens.

    """
    value = cfg.get("component_types")
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if str(v)]


def _matches_component_types(prim_path: str, tokens: list[str]) -> bool:
    """Return whether a prim path matches any component-type token.

    Args:
        prim_path: Component prim path to test.
        tokens: Substring tokens to match against ``prim_path``.

    Returns:
        True when at least one token is contained in ``prim_path``.

    """
    return any(
        prim_path == token if token.startswith("/") else token in prim_path
        for token in tokens
    )


def _compute_tombstone_transform(
    prim: object,
    M_orig: np.ndarray,
    angle_min: float,
    angle_max: float,
    component_instance: dict[str, Any] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Compute tombstone pivot rotation matrix.

    When *component_instance* supplies two ``solder_paste_prims`` or
    ``solder_contact_prims``, the pivot and rotation axis are derived from the
    actual solder-interface centroids:

    - Long axis: vector between the two contact centroids in component local space.
    - Rotation axis: widest bbox axis perpendicular to the contact vector.
    - Pivot: the fixed solder-interface centroid, projected onto the
      component's board-facing support plane and nearest body end.
    - Board normal: inferred as the remaining local axis from contact and width,
      then oriented from the support plane into the component body.

    Falls back to the bounding-box heuristic when fewer than two valid contact
    prims are available.

    Args:
        prim: Component USD prim (used for bbox fallback).
        M_orig: Row-major 4x4 world transform of the component.
        angle_min: Minimum tilt angle in degrees.
        angle_max: Maximum tilt angle in degrees.
        component_instance: Entry from the ``component_instances`` array in
            ``components.json``; should contain ``side`` plus either
            ``solder_paste_prims`` or ``solder_contact_prims``.

    Returns:
        Tuple containing the flat row-major transform list and diagnostics.

    """
    frame = _component_rotation_frame(prim, M_orig, component_instance)
    vmin = frame["vmin"]
    vmax = frame["vmax"]
    center = frame["center"]
    half = frame["half"]
    contact_source = frame["axis_source"]
    use_contacts = contact_source != "bbox"
    contact_vec_norm = frame["long_axis"]
    rot_axis = frame["width_axis"]
    board_normal_local = frame["board_normal_local"]
    support_projection = frame["support_projection"]
    paste_prim_paths = frame["paste_prim_paths"]
    contact_prim_paths = frame["contact_prim_paths"]

    side = (
        str(component_instance.get("side", "Top"))
        if component_instance is not None
        else "Top"
    )

    if use_contacts:
        c0_local, c1_local = frame["reference_points"]
        c0_pivot_local = _project_point_to_support_end(
            c0_local,
            vmin,
            vmax,
            board_normal_local,
            support_projection,
            contact_vec_norm,
        )
        c1_pivot_local = _project_point_to_support_end(
            c1_local,
            vmin,
            vmax,
            board_normal_local,
            support_projection,
            contact_vec_norm,
        )
        pivot_vec_norm = _normalized(c1_pivot_local - c0_pivot_local)
        if pivot_vec_norm is not None:
            contact_vec_norm = pivot_vec_norm

        # `pivot_idx=0` fixes `c0` and lifts `c1` with a positive angle.
        # `pivot_idx=1` fixes `c1` and lifts `c0` with a negative angle.
        pivot_idx = int(np.random.choice([0, 1]))
        pivot_local = c0_pivot_local if pivot_idx == 0 else c1_pivot_local
        angle_mag = float(np.random.uniform(angle_min, angle_max))
        angle_mag_rad = np.radians(angle_mag)
        pos_clearance = _min_support_projection_after_rotation(
            vmin,
            vmax,
            pivot_local,
            rot_axis,
            angle_mag_rad,
            board_normal_local,
        )
        neg_clearance = _min_support_projection_after_rotation(
            vmin,
            vmax,
            pivot_local,
            rot_axis,
            -angle_mag_rad,
            board_normal_local,
        )
        angle_sign = 1 if pos_clearance >= neg_clearance else -1
        angle_deg = angle_mag * angle_sign

        source_paths = (
            paste_prim_paths if contact_source == "paste" else contact_prim_paths
        )
        source_labels = (
            source_paths if len(source_paths) == 2 else ["contact_0", "contact_1"]
        )
        lifted_path = source_labels[1 - pivot_idx]
        logger.info(
            "Tombstone (%s): side=%s, angle=%.1f deg, pivot=%s, lifted=%s",
            contact_source,
            side,
            angle_deg,
            source_labels[pivot_idx].split("/")[-1],
            lifted_path.split("/")[-1],
        )
        # KNOWN QUIRK (kept for record compatibility): "sign" means
        # different things per branch. Here (contacts) it is the ROTATION
        # sign — identical to the sign of ``angle_deg`` (which contact
        # lifts is carried by ``pivot_idx``). In the bbox branch below,
        # "sign" is the LIFTED-END direction along the long axis and the
        # rotation sign is carried by ``angle_deg`` alone. Stored defect
        # records (``defect_ops`` ``tombstone_params`` metadata written
        # into datasets) already serialize the per-branch meaning, so the
        # key is not re-defined; use ``angle_sign`` — emitted by BOTH
        # branches — as the unambiguous rotation-sign key.
        params: dict[str, Any] = {
            "angle_deg": angle_deg,
            "sign": angle_sign,
            "angle_sign": angle_sign,
            "use_contacts": True,
            "contact_source": contact_source,
            "pivot_idx": pivot_idx,
            "pivot_local": pivot_local.tolist(),
            "rot_axis": rot_axis.tolist(),
            "c0_local": c0_local.tolist(),
            "c1_local": c1_local.tolist(),
            "c0_pivot_local": c0_pivot_local.tolist(),
            "c1_pivot_local": c1_pivot_local.tolist(),
            "support_projection": support_projection,
            "positive_clearance": pos_clearance,
            "negative_clearance": neg_clearance,
            "board_normal_local": board_normal_local.tolist(),
            "contact_vec_norm": contact_vec_norm.tolist(),
            "contact_prim_paths": contact_prim_paths,
            "paste_prim_paths": paste_prim_paths,
        }
    else:
        # Fallback: bbox-based heuristic
        long_axis = int(np.argmax(half))
        long_axis_vec = frame["long_axis"]
        axis_name = ["X", "Y", "Z"][long_axis]

        sign = int(np.random.choice([-1, 1]))
        lifted_vec = long_axis_vec * sign
        lift_dir = np.cross(rot_axis, lifted_vec)
        angle_sign = 1 if float(np.dot(lift_dir, board_normal_local)) >= 0.0 else -1
        angle_deg = float(np.random.uniform(angle_min, angle_max)) * angle_sign

        pivot_vec = np.zeros(3)
        pivot_vec[long_axis] = -sign
        pivot_local = center + half * pivot_vec

        lifted_end = f"{axis_name}+" if sign > 0 else f"{axis_name}-"
        logger.info(
            "Tombstone (bbox): long_axis=%s, rot_axis=%s, angle=%.1f deg, lifted=%s",
            axis_name,
            np.array2string(rot_axis, precision=3),
            angle_deg,
            lifted_end,
        )
        # See the contacts-branch comment: "sign" here is the LIFTED-END
        # direction (+1 lifts the positive long-axis end), NOT the
        # rotation sign — that is ``angle_sign`` / the sign of
        # ``angle_deg``.
        params = {
            "angle_deg": angle_deg,
            "sign": sign,
            "angle_sign": angle_sign,
            "long_axis": long_axis,
            "use_contacts": False,
            "contact_source": contact_source,
            "pivot_local": pivot_local.tolist(),
            "rot_axis": rot_axis.tolist(),
            "board_normal_local": board_normal_local.tolist(),
            "long_axis_vec_norm": long_axis_vec.tolist(),
        }

    final_flat = _pivot_rotation_transform(M_orig, pivot_local, rot_axis, angle_deg)
    return final_flat, params


def _tombstone_auto_filter_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Normalize tombstone auto-filter settings.

    Args:
        cfg: Tombstone defect configuration.

    Returns:
        Auto-filter settings with expected keys populated. Missing or empty
        ``auto_filter`` settings default to enabled.

    """
    raw = cfg.get("auto_filter", True)
    if isinstance(raw, bool):
        auto_filter_cfg: dict[str, Any] = {"enabled": raw}
    elif isinstance(raw, dict):
        auto_filter_cfg = dict(raw)
    else:
        logger.warning("tombstone.auto_filter must be a bool or map; disabling")
        auto_filter_cfg = {"enabled": False}

    auto_filter_cfg.setdefault("enabled", True)
    if "require_solder_contact_count" not in auto_filter_cfg:
        if "require_solder_paste_count" in auto_filter_cfg:
            auto_filter_cfg["require_solder_contact_count"] = auto_filter_cfg[
                "require_solder_paste_count"
            ]
        else:
            auto_filter_cfg["require_solder_contact_count"] = 2
    return auto_filter_cfg


def _tombstone_solder_contact_count(
    prim_path: str,
    defect_context: dict[str, Any] | None,
) -> int | None:
    """Return solder-contact count from components.json or sim_manifest metadata."""
    if defect_context is None:
        return None

    component_defs_by_path = defect_context.get("component_defs_by_path") or {}
    component_def = component_defs_by_path.get(prim_path)
    if component_def is not None and "solder_contact_meshes" in component_def:
        return len(component_def.get("solder_contact_meshes") or [])

    contact_index = defect_context.get("contact_index") or {}
    component_instance = contact_index.get(prim_path)
    if component_instance is not None:
        if "solder_contact_meshes" in component_instance:
            return len(component_instance.get("solder_contact_meshes") or [])
        if "solder_contact_prims" in component_instance:
            return len(component_instance.get("solder_contact_prims") or [])

    manifest_by_path = defect_context.get("manifest_by_path") or {}
    meta = manifest_by_path.get(prim_path)
    if not meta:
        return None

    for key in ("solder_contact_count", "family_contact_count", "solder_paste_count"):
        if meta.get(key) is None:
            continue
        try:
            return int(meta[key])
        except (TypeError, ValueError):
            return None

    return None


def _tombstone_solder_paste_count(
    prim_path: str,
    defect_context: dict[str, Any] | None,
) -> int | None:
    """Return per-instance solder-paste prim count from metadata."""
    if defect_context is None:
        return None

    contact_index = defect_context.get("contact_index") or {}
    component_instance = contact_index.get(prim_path)
    if component_instance is not None and "solder_paste_prims" in component_instance:
        return len(component_instance.get("solder_paste_prims") or [])

    manifest_by_path = defect_context.get("manifest_by_path") or {}
    meta = manifest_by_path.get(prim_path)
    if not meta or meta.get("solder_paste_count") is None:
        return None

    try:
        return int(meta["solder_paste_count"])
    except (TypeError, ValueError):
        return None


def _component_world_dims_mm(
    prim_path: str,
    defect_context: dict[str, Any] | None,
) -> tuple[float, float, float] | None:
    """Return cached world-space component dimensions in millimeters.

    Args:
        prim_path: Component prim path to measure.
        defect_context: Shared defect metadata. Pass None to skip measurement.

    Returns:
        Component dimensions in X, Y, and Z order, or None when unavailable.

    """
    if defect_context is None:
        return None

    cache = defect_context.setdefault("world_bbox_mm_by_path", {})
    if prim_path in cache:
        return cache[prim_path]

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        return None

    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    bbox_range = bbox_cache.ComputeWorldBound(prim).GetRange()
    try:
        if bbox_range.IsEmpty():
            return None
    except AttributeError:
        pass

    vmin = np.array(bbox_range.GetMin(), dtype=float)
    vmax = np.array(bbox_range.GetMax(), dtype=float)
    dims = np.maximum(vmax - vmin, 0.0)
    mm_per_stage_unit = (
        float(defect_context.get("stage_meters_per_unit", 0.001)) * 1000.0
    )
    dims_mm = tuple(float(v) for v in dims * mm_per_stage_unit)
    cache[prim_path] = dims_mm
    return dims_mm


def _passes_tombstone_size_filter(
    prim_path: str,
    auto_cfg: dict[str, Any],
    defect_context: dict[str, Any] | None,
) -> bool:
    """Return whether a component satisfies tombstone size limits.

    Args:
        prim_path: Component prim path to test.
        auto_cfg: Tombstone auto-filter settings.
        defect_context: Shared defect metadata used for measurement.

    Returns:
        True when no size limits are configured or all configured limits pass.

    """
    size_keys = {
        "min_long_axis_mm",
        "max_long_axis_mm",
        "min_short_axis_mm",
        "max_short_axis_mm",
        "min_height_mm",
        "max_height_mm",
    }
    if not any(k in auto_cfg for k in size_keys):
        return True

    dims = _component_world_dims_mm(prim_path, defect_context)
    if dims is None:
        return False

    x_mm, y_mm, z_mm = dims
    long_axis_mm = max(x_mm, y_mm)
    short_axis_mm = min(x_mm, y_mm)

    checks = (
        ("min_long_axis_mm", long_axis_mm, lambda actual, limit: actual >= limit),
        ("max_long_axis_mm", long_axis_mm, lambda actual, limit: actual <= limit),
        ("min_short_axis_mm", short_axis_mm, lambda actual, limit: actual >= limit),
        ("max_short_axis_mm", short_axis_mm, lambda actual, limit: actual <= limit),
        ("min_height_mm", z_mm, lambda actual, limit: actual >= limit),
        ("max_height_mm", z_mm, lambda actual, limit: actual <= limit),
    )
    for key, actual, predicate in checks:
        if key not in auto_cfg:
            continue
        if not predicate(actual, float(auto_cfg[key])):
            return False
    return True


def _tombstone_eligible_components(
    component_pool: list[str],
    used_paths: set[str],
    cfg: dict[str, Any],
    defect_context: dict[str, Any] | None,
    log: bool = True,
) -> list[str]:
    """Return components eligible for tombstone selection.

    Manual ``component_types`` settings are authoritative. When no manual
    allowlist is present, ``auto_filter`` uses ``components.json`` or
    ``sim_manifest`` metadata to require matching component-level solder
    contacts and per-instance solder paste prims.

    Args:
        component_pool: Candidate component prim paths.
        used_paths: Component prim paths already assigned to another defect.
        cfg: Tombstone defect configuration.
        defect_context: Shared defect metadata used by automatic filters.
        log: Whether to emit eligibility logs.

    Returns:
        Component prim paths that may receive a tombstone defect.

    """
    available = [p for p in component_pool if p not in used_paths]

    # Early out: Manual `component_types` in `defects.tombstone.component_types`.
    type_filter = _component_type_tokens(cfg)
    if type_filter:
        if log:
            logger.info(
                "tombstone: component_types specified (%d token(s)); "
                "skipping auto_filter",
                len(type_filter),
            )
        return [p for p in available if _matches_component_types(p, type_filter)]

    # Automatic filters from `defects.tombstone.auto_filter`.
    auto_filter_cfg = _tombstone_auto_filter_cfg(cfg)
    if not auto_filter_cfg["enabled"]:
        if log:
            logger.warning(
                "tombstone: no component_types and auto_filter disabled; "
                "no eligible components, skipping"
            )
        return []

    # `components.json` or `sim_manifest` metadata is required for `auto_filter`.
    if defect_context is None or (
        not defect_context.get("component_defs_by_path")
        and not defect_context.get("contact_index")
        and not defect_context.get("manifest_by_path")
    ):
        if log:
            logger.warning(
                "tombstone: no component_types and no components/sim metadata; "
                "no eligible components, skipping"
            )
        return []

    required_count = auto_filter_cfg["require_solder_contact_count"]
    try:
        required_count = int(required_count)
    except (TypeError, ValueError):
        logger.warning(
            "tombstone.auto_filter.require_solder_contact_count=%r is invalid; using 2",
            required_count,
        )
        required_count = 2

    eligible: list[str] = []
    missing_metadata = 0
    contact_count_rejects = 0
    paste_count_rejects = 0
    size_rejects = 0

    for prim_path in available:
        # Match the component-level solder-contact mesh count and the
        # instance-level solder-paste prim count. Contact prim geometry can be
        # merged in the USD instance, so paste prims verify this placement has
        # the expected number of solder interfaces.
        solder_contact_count = _tombstone_solder_contact_count(
            prim_path,
            defect_context,
        )
        if solder_contact_count is None:
            missing_metadata += 1
            continue
        if solder_contact_count != required_count:
            contact_count_rejects += 1
            continue

        solder_paste_count = _tombstone_solder_paste_count(prim_path, defect_context)
        if solder_paste_count is None:
            missing_metadata += 1
            continue
        if solder_paste_count != required_count:
            paste_count_rejects += 1
            continue

        # Size filter: check whether the component is within the size limits.
        if not _passes_tombstone_size_filter(
            prim_path,
            auto_filter_cfg,
            defect_context,
        ):
            size_rejects += 1
            continue

        eligible.append(prim_path)

    if log:
        logger.info(
            "tombstone auto_filter: %d eligible / %d available "
            "(missing_metadata=%d, contact_count_rejects=%d, "
            "paste_count_rejects=%d, size_rejects=%d)",
            len(eligible),
            len(available),
            missing_metadata,
            contact_count_rejects,
            paste_count_rejects,
            size_rejects,
        )
    return eligible
