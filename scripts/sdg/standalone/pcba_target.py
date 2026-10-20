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

"""Read the board-scoped keys of a ``--pcba-config`` target.

Two fields of a pcba target need interpreting before the pipeline can use
them, and both used to be interpreted inline at module scope in
the retired legacy pipeline -- which imported Kit, so nothing there
could be unit-tested. The rules live here instead, with no Kit or USD
dependency, so the pipeline and the tests read the same code.

* ``component_types`` may be a keyword (``ALL``, ``0``, or a ``subsets`` key)
  rather than a list. A keyword resolves against ONE board's registry -- the
  ``component_registry:`` path in the target, else a ``components.yaml``
  beside it. There is deliberately no repo-wide default: resolving ``ALL``
  against another board's registry yields a list that matches nothing and
  renders an empty scene.
* ``component_classes`` maps classes to reference designators. An instance
  the map does not name gets ``default_component_class``, which defaults to
  the neutral :data:`DEFAULT_COMPONENT_CLASS` rather than a package family,
  so an unmapped IC cannot be quietly labelled as a passive.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

#: Class given to a component instance the target's ``component_classes`` map
#: does not name. Deliberately not a package family: labelling an unmapped IC
#: or connector as a capacitor reads as a correct dataset right up until a
#: detector is trained on it. A board whose unmapped parts really are one
#: family says so with ``default_component_class`` in its pcba target.
DEFAULT_COMPONENT_CLASS = "component"


class ComponentRegistryError(ValueError):
    """A ``component_types`` keyword could not be resolved for this target."""


def _means_no_components(value: object) -> bool:
    """Return True for the two YAML spellings of "no components".

    An unquoted ``0`` loads as an int and a quoted ``"0"`` as a string; both
    mean an empty selection. ``False`` also compares equal to ``0`` and is
    excluded, since a boolean there is a typo rather than a selection.
    """
    if isinstance(value, bool):
        return False
    return value == 0 or value == "0"


def resolve_component_types(
    cfg: Mapping[str, Any],
    pcba_config_path: str | os.PathLike[str] | None,
) -> list[str] | None:
    """Resolve a ``component_types`` keyword into the list it stands for.

    Args:
        cfg: Merged pipeline configuration -- the ``--config`` flow updated
            with the ``--pcba-config`` target. The registry path is
            environment-expanded here as well, so the result does not depend
            on whether the caller expanded the configuration first.
        pcba_config_path: Path of the ``--pcba-config`` file the keyword came
            from; a registry is looked up relative to it.

    Returns:
        The concrete list, or ``None`` when ``component_types`` is already a
        list (or absent) and nothing needs resolving.

    Raises:
        ComponentRegistryError: If the keyword needs a registry and none can
            be found, if ``component_registry`` is not a path string, or if
            the keyword is not ``ALL``, ``0`` or a ``subsets`` key.

    """
    keyword = cfg.get("component_types")
    if _means_no_components(keyword):
        # Decided before any registry lookup: "0" needs no registry, and an
        # unquoted 0 must never reach component processing as an int.
        return []
    if not isinstance(keyword, str):
        return None
    if not pcba_config_path:
        raise ComponentRegistryError(
            "component_types keyword requires --pcba-config so the component "
            "registry for this board can be found."
        )
    registry_setting = cfg.get("component_registry")
    if registry_setting is not None and not isinstance(registry_setting, str):
        raise ComponentRegistryError(
            f"`component_registry:` must be a path string, got "
            f"{type(registry_setting).__name__} ({registry_setting!r})"
        )
    pcba_dir = Path(pcba_config_path).resolve().parent
    if registry_setting:
        # The loader promises ${VAR} expansion for every configuration
        # string; an environment-backed registry path is one of them.
        registry_path = (pcba_dir / os.path.expandvars(registry_setting)).resolve()
    else:
        registry_path = pcba_dir / "components.yaml"
    if not registry_path.is_file():
        raise ComponentRegistryError(
            f"component_types keyword {keyword!r} needs a component registry; "
            f"looked for {registry_path}. Point `component_registry:` in "
            f"{pcba_config_path} at this board's registry, or inline the "
            "list in `component_types:`."
        )
    with open(registry_path, encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}
    subsets = registry.get("subsets") or {}
    if keyword == "ALL":
        return list(registry.get("all") or [])
    if keyword in subsets:
        return list(subsets[keyword])
    raise ComponentRegistryError(
        f"Unknown component_types keyword: {keyword!r}. "
        f"Allowed: ALL, 0, or a key in {registry_path}'s `subsets`."
    )


def classify_components(
    refdes_list: Iterable[str],
    component_classes: Mapping[str, Iterable[str] | None] | None,
    default_class: str = DEFAULT_COMPONENT_CLASS,
) -> tuple[dict[str, str], list[str]]:
    """Assign a semantic class to every reference designator.

    Args:
        refdes_list: Reference designators of the component instances found
            on the stage, in discovery order.
        component_classes: The target's ``{class: [refdes, ...]}`` map, or
            ``None`` when the board has none.
        default_class: Class for a designator the map does not name.

    Returns:
        ``(classes, unmapped)``: the class per designator, and the designators
        that fell back to ``default_class`` while a map was present. The list
        is empty when there is no map at all -- then every instance is on the
        default by design, not by omission.

    """
    refdes_class: dict[str, str] = {}
    for cls, designators in (component_classes or {}).items():
        for refdes in designators or []:
            refdes_class[refdes] = cls
    classes: dict[str, str] = {}
    unmapped: list[str] = []
    for refdes in refdes_list:
        cls = refdes_class.get(refdes)
        if cls is None:
            cls = default_class
            if refdes_class:
                unmapped.append(refdes)
        classes[refdes] = cls
    return classes, unmapped
