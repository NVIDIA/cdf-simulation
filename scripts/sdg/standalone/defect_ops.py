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

"""Defect operations applied through explicit component-aware transforms."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdSemantics

try:
    from omni.replicator.core.scripts.functional.modify import (
        TagCache,
        TransformToken,
        pose_ops,
    )
except ImportError:
    TagCache = None
    TransformToken = None
    pose_ops = None

from defects_utils.api import (
    reverse_polarity,
    shift,
    sideflip,
    tombstone,
)
from defects_utils.selection import (
    describe_targeting,
    selection_count,
)
from defects_utils.tombstone import (
    _tombstone_eligible_components,
)
from defects_utils.transform_utils import _local_transform_matrix

logger = logging.getLogger(__name__)

_pose_ops_context_warmed = False


def _warm_pose_ops_context() -> None:
    """Force warp to bind to the active CUDA context before the first pose_ops.

    Replicator's pose_ops native binding launches a CUDA kernel for the transform
    apply. On multi-GPU hosts (observed on 4xH100 NVL with --gpus all) the first
    launch can race with warp's lazy CUDA module load and print
    `[CUDA LAUNCH ERROR] Transform ops kernel launch failed: invalid resource handle`
    to stderr. The kernel succeeds on Replicator's internal retry, so output
    counts still match, but the log noise is easily mistaken for a failure.

    wp.init() + wp.synchronize_device() forces warp to materialize its CUDA
    stream + module bindings on the active device first, so pose_ops sees a
    valid resource handle on its first launch. Idempotent + cached via the
    module-level flag.
    """
    global _pose_ops_context_warmed
    if _pose_ops_context_warmed:
        return
    if pose_ops is None:
        _pose_ops_context_warmed = True
        return
    try:
        import warp as wp

        wp.init()
        wp.synchronize_device()
        _pose_ops_context_warmed = True
        logger.debug("pose_ops CUDA context warmed via warp.synchronize_device()")
    except Exception as e:  # noqa: BLE001 - optional runtime optimization
        logger.debug("pose_ops CUDA context warm-up skipped: %s", e)


def _clear_tag_cache() -> None:
    """Clear TagCache private caches for Fabric/USD sync."""
    if TagCache is None:
        return
    try:
        TagCache._cache.clear()
        TagCache._path_order_cache.clear()
    except AttributeError:
        logger.warning("TagCache private API changed - cache not cleared")


def _get_prim_and_transform(prim_path: str) -> tuple:
    """Get prim and its original row-major transform."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        logger.error("Prim not found: %s", prim_path)
        return None, None, None
    try:
        M = _local_transform_matrix(prim)
    except ValueError as exc:
        raise ValueError(
            f"Component prim {prim_path!r} has no usable local transform: {exc}"
        ) from exc
    flat = M.flatten().tolist()
    return prim, flat, M


def _apply_semantic(prim: object, defect_type: str) -> None:
    """Apply semantic label + clear TagCache for Fabric/USD sync."""
    labels = UsdSemantics.LabelsAPI.Apply(prim, "defect")
    labels.CreateLabelsAttr().Set([defect_type])
    _clear_tag_cache()


def apply_defect_semantic(prim: object, defect_type: str) -> None:
    """Label a component ``defect=<type>`` without a transform operation.

    The solder-level defects (insufficient/excess/missing_solder/dry_contact)
    change geometry the solver produces, not the component's pose -- but they
    are segmentation classes exactly like the pose operations, so they share
    this labelling path.
    """
    _apply_semantic(prim, defect_type)


def _apply_flat_transforms(
    prim_transforms: list[tuple[object, list[float]]],
) -> None:
    """Apply row-major local matrices as one ordered transform batch."""
    if not prim_transforms:
        return
    if pose_ops is not None and TransformToken is not None:
        prims, flat_transforms = zip(*prim_transforms, strict=True)
        pose_ops(
            list(prims),
            [(TransformToken.TRANSFORM, list(flat_transforms))],
        )
        return
    for prim, flat_transform in prim_transforms:
        matrix = Gf.Matrix4d(*[float(value) for value in flat_transform])
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix)


def build_defect_context(
    stage: object,
    sim_manifest: dict[str, Any] | None = None,
    board_metadata: object | None = None,
) -> dict[str, Any]:
    """Build shared metadata for defect eligibility filters.

    The pipeline loads optional legacy simulation data and a source-neutral
    board metadata catalog once, then passes them here. This function indexes
    per-instance metadata by prim path so ``prepare_defects()`` and
    ``pretag_defect_eligible()`` can reuse it without opening files.

    Args:
        stage: Open USD stage used for unit conversion.
        sim_manifest: Decoded ``sim_manifest.json``. Pass None when not
            configured.
        board_metadata: Source-neutral board metadata catalog.

    Returns:
        Shared context containing manifest metadata, cached sizing data,
        optional ``component_defs_by_path`` mapping prim path to component
        definition, and optional ``contact_index`` mapping prim path to
        component instance.

    Example:

    .. code-block:: python

        from defect_ops import build_defect_context

        context = build_defect_context(stage)  # doctest: +SKIP
        print(context["stage_meters_per_unit"])  # doctest: +SKIP

    """
    try:
        stage_mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    except (RuntimeError, TypeError, ValueError):
        stage_mpu = 0.001

    context: dict[str, Any] = {
        "sim_manifest_present": sim_manifest is not None,
        "manifest_by_path": {},
        "stage_meters_per_unit": stage_mpu,
        "world_bbox_mm_by_path": {},
        "contact_index": {},
        "component_defs_by_path": {},
    }

    if board_metadata is not None:
        component_defs_by_path = context["component_defs_by_path"]
        contact_index = context["contact_index"]
        prototype_by_id = {
            prototype.prototype_id: prototype for prototype in board_metadata.prototypes
        }
        for component in board_metadata.components:
            prototype = prototype_by_id.get(component.prototype_id or "")
            component_defs_by_path[component.prim_path] = {
                "solder_contact_meshes": (
                    prototype.contact_meshes if prototype else ()
                ),
            }
            contact_index[component.prim_path] = {
                "path": component.prim_path,
                "refdes": component.refdes,
                "side": component.side,
                "process_class": component.process_class,
                "body_prims": component.body_prim_paths,
                "solder_contact_prims": component.contact_prim_paths,
                "solder_paste_prims": tuple(
                    paste.prim_path
                    for paste in board_metadata.paste_for_component(component)
                ),
            }
        logger.info(
            "Defect context: indexed %d component instance(s) from board metadata",
            len(contact_index),
        )

    if not sim_manifest:
        return context

    manifest_by_path = context["manifest_by_path"]
    for component_name, entry in sim_manifest.items():
        if not isinstance(entry, dict):
            continue
        if not (
            "contact_method" in entry
            or "instances" in entry
            or "instance_paths" in entry
        ):
            continue

        family_contact_count = len(entry.get("contact_points") or [])
        for prim_path in entry.get("instance_paths") or []:
            if prim_path:
                manifest_by_path.setdefault(
                    str(prim_path),
                    {
                        "component_name": component_name,
                        "family_contact_count": family_contact_count,
                    },
                )

        for instance in entry.get("instances") or []:
            prim_path = instance.get("prim_path")
            if not prim_path:
                continue
            meta = manifest_by_path.setdefault(
                str(prim_path),
                {
                    "component_name": component_name,
                    "family_contact_count": family_contact_count,
                },
            )
            meta["solder_contact_count"] = instance.get("solder_contact_count")
            meta["solder_paste_count"] = instance.get("solder_paste_count")
            meta["ground_clearance_mm"] = instance.get("ground_clearance_mm")

    logger.info(
        "Defect context: indexed %d sim_manifest component instance(s)",
        len(manifest_by_path),
    )
    return context


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


def _eligible_components(
    defect_type: str,
    cfg: dict[str, Any],
    component_pool: list[str],
    used_paths: set[str],
    defect_context: dict[str, Any] | None,
    log: bool = True,
) -> list[str]:
    """Return eligible candidates for a defect.

    Args:
        defect_type: Defect name from the configuration.
        cfg: Per-defect configuration.
        component_pool: Candidate component prim paths.
        used_paths: Component prim paths already assigned to another defect.
        defect_context: Shared defect metadata used by constrained filters.
        log: Whether to emit eligibility logs.

    Returns:
        Component prim paths eligible for ``defect_type``.

    """
    if defect_type == "tombstone":
        return _tombstone_eligible_components(
            component_pool,
            used_paths,
            cfg,
            defect_context,
            log=log,
        )

    available = [p for p in component_pool if p not in used_paths]
    type_filter = _component_type_tokens(cfg)
    if type_filter:
        available = [p for p in available if _matches_component_types(p, type_filter)]
    return available


def _ordered_enabled_defects(
    component_pool: list[str],
    defects_cfg: dict[str, Any],
    defect_context: dict[str, Any] | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return enabled defects ordered from most to least constrained.

    Constrainedness is estimated dynamically from the current eligible
    candidate count for each defect. Defects with fewer eligible components are
    returned first, with original config order used as the tie-breaker.

    Args:
        component_pool: Candidate component prim paths.
        defects_cfg: Defect configuration map.
        defect_context: Shared defect metadata used by constrained filters.

    Returns:
        Enabled defect names and their configurations, sorted by candidate count.

    Example:
        If ``reverse_polarity`` has one eligible component, ``tombstone`` has
        50, and ``shift`` has 200, the returned order is equivalent to::

            [
                ("reverse_polarity", defects_cfg["reverse_polarity"]),
                ("tombstone", defects_cfg["tombstone"]),
                ("shift", defects_cfg["shift"]),
            ]

    """
    ordered: list[tuple[int, int, str, dict[str, Any]]] = []
    for idx, (defect_type, cfg) in enumerate(defects_cfg.items()):
        if not cfg.get("enabled", False):
            continue
        eligible_count = len(
            _eligible_components(
                defect_type,
                cfg,
                component_pool,
                used_paths=set(),
                defect_context=defect_context,
                log=False,
            )
        )
        ordered.append((eligible_count, idx, defect_type, cfg))

    ordered.sort(key=lambda item: (item[0], item[1]))
    return [(defect_type, cfg) for _, _, defect_type, cfg in ordered]


def prepare_defects(
    component_pool: list[str],
    defects_cfg: dict[str, Any],
    defect_context: dict[str, Any] | None = None,
    excluded_paths: set[str] | None = None,
    *,
    apply_semantics: bool = True,
) -> list[dict[str, Any]]:
    """Select components, apply defects, and store reapply parameters.

    Args:
        component_pool: Candidate component prim paths.
        defects_cfg: Defect configuration map.
        defect_context: Shared defect metadata used by constrained filters.
        excluded_paths: Component paths already assigned by earlier operations.
        apply_semantics: Whether to author Replicator defect labels directly.

    Returns:
        Records with selected paths, defect names, and stored transform settings.

    Example:

    .. code-block:: python

        from defect_ops import prepare_defects

        records = prepare_defects(component_paths, defects_cfg)  # doctest: +SKIP
        print(len(records))  # doctest: +SKIP

    """
    _warm_pose_ops_context()
    defect_records: list[dict[str, Any]] = []
    used_paths = set(excluded_paths or ())

    for defect_type, cfg in _ordered_enabled_defects(
        component_pool,
        defects_cfg,
        defect_context,
    ):
        available = _eligible_components(
            defect_type,
            cfg,
            component_pool,
            used_paths,
            defect_context,
        )
        if not available:
            logger.info(
                "%s: no eligible components after filter; skipping",
                defect_type,
            )
            continue
        ratio_base = (
            len(available) if defect_type == "tombstone" else len(component_pool)
        )
        n = selection_count(
            cfg,
            len(available),
            ratio_base_count=ratio_base,
            operation_name=defect_type,
        )
        selected = np.random.choice(available, size=n, replace=False).tolist()
        used_paths.update(selected)
        operation_transforms: list[tuple[object, list[float]]] = []

        for prim_path in selected:
            prim, original_flat, M_orig = _get_prim_and_transform(prim_path)
            if prim is None:
                continue

            if apply_semantics:
                _apply_semantic(prim, defect_type)

            record: dict[str, Any] = {
                "path": prim_path,
                "defect": defect_type,
                "original_flat": original_flat,
            }
            contact_index = (defect_context or {}).get("contact_index") or {}
            component_instance = contact_index.get(prim_path)

            if defect_type == "shift":
                result = shift(
                    prim,
                    M_orig,
                    translate_range=cfg["translate_range"],
                    rotate_z_range=cfg["rotate_z_range"],
                    component_instance=component_instance,
                    variant_index=cfg.get("variant_index"),
                )

            elif defect_type == "tombstone":
                result = tombstone(
                    prim,
                    M_orig,
                    angle_min=cfg["angle_min"],
                    angle_max=cfg["angle_max"],
                    component_instance=component_instance,
                )

            elif defect_type == "sideflip":
                result = sideflip(
                    prim,
                    M_orig,
                    angle_min=cfg["angle_min"],
                    angle_max=cfg["angle_max"],
                    component_instance=component_instance,
                    variant_index=cfg.get("variant_index"),
                    rotate_z_jitter_deg=cfg.get("rotate_z_jitter_deg", 0.0),
                    slide_range_mm=cfg.get("slide_range_mm", 0.0),
                )

            elif defect_type == "reverse_polarity":
                result = reverse_polarity(
                    prim,
                    M_orig,
                    component_instance=component_instance,
                )
            else:
                logger.warning("Unknown defect type %r; skipping", defect_type)
                continue

            record[f"{defect_type}_final_flat"] = result.flat_transform
            record[f"{defect_type}_params"] = result.parameters
            operation_transforms.append((prim, result.flat_transform))
            defect_records.append(record)

        _apply_flat_transforms(operation_transforms)
        logger.info(
            "%s: %d components (%s)",
            defect_type,
            n,
            describe_targeting(cfg),
        )

    logger.info(
        "Total defects: %d / %d components",
        len(defect_records),
        len(component_pool),
    )
    for d in defect_records[:10]:
        logger.info("  %12s -> %s", d["defect"], d["path"].split("/")[-1])
    if len(defect_records) > 10:
        logger.info("  ... and %d more", len(defect_records) - 10)

    return defect_records


def reapply_defects(defect_records: list[dict[str, Any]]) -> None:
    """Re-apply all stored absolute defect transforms.

    Args:
        defect_records: Applied-defect records returned by ``prepare_defects()``.

    Example:

    .. code-block:: python

        from defect_ops import reapply_defects

        reapply_defects(records)  # doctest: +SKIP

    """
    _warm_pose_ops_context()
    stage = omni.usd.get_context().get_stage()
    prim_transforms: list[tuple[object, list[float]]] = []
    for record in defect_records:
        prim = stage.GetPrimAtPath(record["path"])
        if not prim.IsValid():
            continue

        final_key = f"{record['defect']}_final_flat"
        if final_key not in record:
            continue
        prim_transforms.append((prim, record[final_key]))
    _apply_flat_transforms(prim_transforms)


def restore_defects(defect_records: list[dict[str, Any]]) -> None:
    """Reset each previously-flipped prim back to its original xform.

    Used by the per-trigger re-randomization path in sdg_pipeline.py so
    that triggers don't accumulate flips from prior triggers' selections.
    Walks the previous trigger's records and writes the cached
    ``original_flat`` back as a single TRANSFORM op (mirrors the shape
    reapply_defects writes).

    Args:
        defect_records: Applied-defect records returned by ``prepare_defects()``.

    Example:

    .. code-block:: python

        from defect_ops import restore_defects

        restore_defects(records)  # doctest: +SKIP

    """
    stage = omni.usd.get_context().get_stage()
    prim_transforms: list[tuple[object, list[float]]] = []
    for record in defect_records:
        prim = stage.GetPrimAtPath(record["path"])
        if not prim.IsValid():
            continue
        prim_transforms.append((prim, record["original_flat"]))
    _apply_flat_transforms(prim_transforms)


def pretag_defect_eligible(
    component_pool: list[str],
    defects_cfg: dict[str, Any],
    defect_context: dict[str, Any] | None = None,
) -> int:
    """Pre-tag every defect-eligible prim with its defect semantic at init.

    Used by the ``randomize_defects_per_trigger: true`` code path. Without
    this, ``rep_modify.semantics`` calls made inside the trigger loop
    don't propagate to the colorize semseg annotator (which caches its
    prim -> class map when the render product is set up). Tagging all
    candidate prims BEFORE the first render bakes them into that cache,
    so each trigger's colorize PNG shows every eligible prim coloured as
    its defect class. The actual rotation still varies per trigger via
    pose_ops; the bbox annotator (which doesn't cache the map) remains
    the authoritative per-frame "which prims actually got rotated"
    ground truth - read ``bounding_box_2d_tight_prim_paths_*.json``.

    Args:
        component_pool: Candidate component prim paths.
        defects_cfg: Defect configuration map.
        defect_context: Shared defect metadata used by constrained filters.

    Returns:
        Number of prims tagged.

    Example:

    .. code-block:: python

        from defect_ops import pretag_defect_eligible

        count = pretag_defect_eligible(component_paths, defects_cfg)  # doctest: +SKIP
        print(count)  # doctest: +SKIP

    """
    stage = omni.usd.get_context().get_stage()
    tagged = 0
    for defect_type, cfg in _ordered_enabled_defects(
        component_pool,
        defects_cfg,
        defect_context,
    ):
        eligible = _eligible_components(
            defect_type,
            cfg,
            component_pool,
            used_paths=set(),
            defect_context=defect_context,
        )
        for prim_path in eligible:
            prim = stage.GetPrimAtPath(prim_path)
            if prim is None or not prim.IsValid():
                continue
            _apply_semantic(prim, defect_type)
            tagged += 1
    logger.info(
        "Pre-tagged %d defect-eligible component(s) "
        "(randomize_defects_per_trigger mode)",
        tagged,
    )
    return tagged
