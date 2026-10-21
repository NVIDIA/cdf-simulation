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

"""Apply transactional PCB geometry visibility around one capture pass."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from cdf.errors import ValidationError


@dataclass(frozen=True)
class _VisibilitySnapshot:
    """Authored visibility state for one temporarily hidden prim."""

    path: str
    authored: bool
    value: object


def _composed_solder_paths(
    stage: object,
    cohort: object | None,
    source_path: str,
) -> tuple[str, ...]:
    """Expand one prototype-source solder path onto valid composed boards."""
    if cohort is None:
        return ()
    try:
        targets = cohort.expand_source_path(source_path)
    except ValidationError:
        return ()
    return tuple(
        target.composed_path
        for target in targets
        if stage.GetPrimAtPath(target.composed_path).IsValid()
    )


def _solder_visibility_paths(
    stage: object,
    manifest: object,
    cohort: object | None = None,
) -> tuple[str, ...]:
    """Return the smallest valid composed prim set covering generated solder."""
    output_root = str(getattr(manifest, "output_root_path", "") or "")
    composed_roots = _composed_solder_paths(stage, cohort, output_root)
    if composed_roots:
        return composed_roots
    if output_root and stage.GetPrimAtPath(output_root).IsValid():
        return (output_root,)
    paths = []
    for path in getattr(manifest, "generated_prim_paths", ()) or ():
        composed_paths = _composed_solder_paths(stage, cohort, path)
        if composed_paths:
            paths.extend(composed_paths)
        elif stage.GetPrimAtPath(path).IsValid():
            paths.append(path)
    return tuple(dict.fromkeys(paths))


def _hide_solder(
    stage: object,
    manifest: object,
    cohort: object | None = None,
) -> tuple[_VisibilitySnapshot, ...]:
    """Hide generated solder and return the state required to restore it."""
    from pxr import UsdGeom

    paths = _solder_visibility_paths(stage, manifest, cohort)
    if not paths:
        raise ValidationError(
            "capture visibility requested hidden solder, but the solder "
            "placement manifest resolves no generated prims",
        )
    snapshots = []
    try:
        for path in paths:
            imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
            if not imageable:
                raise ValidationError(
                    f"capture visibility cannot hide non-imageable solder prim {path!r}",
                )
            attribute = imageable.GetVisibilityAttr()
            snapshot = _VisibilitySnapshot(
                path=path,
                authored=attribute.HasAuthoredValueOpinion(),
                value=attribute.Get(),
            )
            attribute.Set(UsdGeom.Tokens.invisible)
            snapshots.append(snapshot)
    except Exception:
        _restore_visibility(stage, tuple(snapshots))
        raise
    return tuple(snapshots)


def _hide_board_roles(
    stage: object,
    board_root: str,
    role_names: frozenset[str] | set[str],
) -> tuple[_VisibilitySnapshot, ...]:
    """Hide all prims matching any of the given ``pcba:mesh:role`` shortnames.

    Args:
        stage: Live USD stage.
        board_root: Prim path of the board root to search under.
        role_names: Role shortnames without the ``board_`` prefix (e.g.
            ``{"copper"}``).  Every prim whose ``pcba:mesh:role`` attribute
            equals ``"board_{name}"`` for any name in this set is hidden.

    Returns:
        Snapshots required to restore the authored visibility state.

    Raises:
        ValidationError: If no prims are found for any requested role, or if a
            matched prim is not imageable.

    """
    from pxr import Usd, UsdGeom

    from pcb_aoi.board_metadata.usd_schema_metadata_keys import MESH_ROLE_ATTR

    target_roles = {f"board_{name}" for name in role_names}
    board_prim = stage.GetPrimAtPath(board_root)
    if not board_prim.IsValid():
        raise ValidationError(
            f"capture visibility cannot hide board roles: board root "
            f"{board_root!r} is not a valid prim",
        )
    found_roles: set[str] = set()
    snapshots = []
    try:
        for prim in Usd.PrimRange(board_prim):
            attr = prim.GetAttribute(MESH_ROLE_ATTR)
            if not attr:
                continue
            role_value = attr.Get()
            if role_value not in target_roles:
                continue
            found_roles.add(role_value)
            imageable = UsdGeom.Imageable(prim)
            if not imageable:
                raise ValidationError(
                    f"capture visibility cannot hide non-imageable prim "
                    f"{str(prim.GetPath())!r} with role {role_value!r}",
                )
            vis_attr = imageable.GetVisibilityAttr()
            snapshot = _VisibilitySnapshot(
                path=str(prim.GetPath()),
                authored=vis_attr.HasAuthoredValueOpinion(),
                value=vis_attr.Get(),
            )
            vis_attr.Set(UsdGeom.Tokens.invisible)
            snapshots.append(snapshot)
    except Exception:
        _restore_visibility(stage, tuple(snapshots))
        raise
    missing = sorted(name for name in role_names if f"board_{name}" not in found_roles)
    warn_missing = [name for name in missing if name != "hole"]
    if warn_missing:
        warnings.warn(
            f"capture visibility found no prims for role(s) {warn_missing} under "
            f"{board_root!r}; those targets were skipped",
            stacklevel=3,
        )
    return tuple(snapshots)


def _show_board_roles(
    stage: object,
    board_root: str,
    role_names: frozenset[str] | set[str],
) -> tuple[_VisibilitySnapshot, ...]:
    """Show board-role geometry and return the state required to restore it.

    Generated annotation geometry is hidden at its parent Xform. For each
    matching role, this helper clears authored invisibility from the matching
    prim and its ancestors below the board root. The original opinions are
    restored after the capture pass.

    Args:
        stage: Live USD stage.
        board_root: Prim path of the board root to search under.
        role_names: Role shortnames without the ``board_`` prefix.

    Returns:
        Snapshots required to restore the authored visibility state.

    Raises:
        ValidationError: If the board root is invalid or a matched prim is not
            imageable.

    """
    from pxr import Usd, UsdGeom

    from pcb_aoi.board_metadata.usd_schema_metadata_keys import MESH_ROLE_ATTR

    target_roles = {f"board_{name}" for name in role_names}
    board_prim = stage.GetPrimAtPath(board_root)
    if not board_prim.IsValid():
        raise ValidationError(
            f"capture visibility cannot show board roles: board root "
            f"{board_root!r} is not a valid prim",
        )
    found_roles: set[str] = set()
    matched_prims = []
    for prim in Usd.PrimRange(board_prim):
        attribute = prim.GetAttribute(MESH_ROLE_ATTR)
        if not attribute:
            continue
        role_value = attribute.Get()
        if role_value not in target_roles:
            continue
        found_roles.add(role_value)
        matched_prims.append(prim)

    snapshots_by_path: dict[str, _VisibilitySnapshot] = {}
    try:
        for prim in matched_prims:
            current = prim
            while current.IsValid() and str(current.GetPath()) != board_root:
                imageable = UsdGeom.Imageable(current)
                if not imageable:
                    raise ValidationError(
                        f"capture visibility cannot show non-imageable prim "
                        f"{str(current.GetPath())!r}",
                    )
                visibility = imageable.GetVisibilityAttr()
                path = str(current.GetPath())
                if (
                    visibility.HasAuthoredValueOpinion()
                    and visibility.Get() == UsdGeom.Tokens.invisible
                ):
                    if path not in snapshots_by_path:
                        snapshots_by_path[path] = _VisibilitySnapshot(
                            path=path,
                            authored=True,
                            value=UsdGeom.Tokens.invisible,
                        )
                    visibility.Set(UsdGeom.Tokens.inherited)
                current = current.GetParent()
    except Exception:
        _restore_visibility(stage, tuple(snapshots_by_path.values()))
        raise

    missing = sorted(name for name in role_names if f"board_{name}" not in found_roles)
    warn_missing = [name for name in missing if name != "hole"]
    if warn_missing:
        warnings.warn(
            f"capture visibility found no prims for role(s) {warn_missing} under "
            f"{board_root!r}; those targets were skipped",
            stacklevel=3,
        )
    return tuple(snapshots_by_path.values())


def _restore_visibility(
    stage: object,
    snapshots: tuple[_VisibilitySnapshot, ...],
) -> None:
    """Restore the exact authored visibility state captured before a pass."""
    from pxr import UsdGeom

    for snapshot in snapshots:
        prim = stage.GetPrimAtPath(snapshot.path)
        if not prim.IsValid():
            continue
        attribute = UsdGeom.Imageable(prim).GetVisibilityAttr()
        if snapshot.authored:
            attribute.Set(snapshot.value)
        else:
            attribute.Clear()
