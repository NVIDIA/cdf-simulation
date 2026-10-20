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

"""Shared ``<board>.components.json`` sidecar helpers.

Single source for the sidecar conventions used by
``editor_solder_fillet_board.py`` and the batch pipeline:

  * path derivation — ``<layer stem>.components.json`` next to a USD layer;
  * index load — ``component_instances[]`` keyed by instance ``path``;
  * lookup + prefix remap — sidecar prim paths are authored relative to
    the board layer (``/Board_X/Components/...``) while a composed stage may
    mount the board under a prefix (usually ``/World``); the lookup also
    walks prim-path ancestors so sub-prims resolve to their component.

Deliberately stdlib-only (no omni/pxr/numpy): unit-tested in CI, and
importable from every context the two consumers run in.
"""

from __future__ import annotations

import json
import os

# Instance-record fields carrying board-relative prim paths that must be
# remapped into the composed stage before use (defects_utils / pad-aware
# fillet placement read them).
REMAP_FIELDS = ("solder_paste_prims", "solder_contact_prims", "body_prims")

# Layer-stem suffixes for a DERIVED layer supplied beside
# ``<board>.usd``: the AOI material overlay and geometry/material/physics splits. Only the
# base board carries ``<board>.components.json``, so a derived layer has to
# fall back to the base stem when the AOI overlay is loaded,
# and without this it silently lost every sidecar-backed capability
# (tombstone/sideflip eligibility, defect pivots, pad-aware fillets).
DERIVED_LAYER_SUFFIXES = ("_aoi", "_geo", "_mtl", "_phys")


def sidecar_path_for_layer(layer_real_path) -> str:
    """``<stem>.components.json`` next to ``layer_real_path`` when that
    file exists, else "".

    When ``layer_real_path`` is a derived layer (a
    :data:`DERIVED_LAYER_SUFFIXES` stem such as ``<board>_aoi.usd``), the
    base board's sidecar is accepted as well: the ingest writes ONE
    ``<board>.components.json`` for the whole board, and every derived
    layer describes the same instances.
    """
    real = str(layer_real_path or "")
    if not real:
        return ""
    stem = os.path.splitext(real)[0]
    stems = [stem]
    for suffix in DERIVED_LAYER_SUFFIXES:
        if stem.endswith(suffix):
            stems.append(stem[: -len(suffix)])
            break
    for candidate_stem in stems:
        cand = candidate_stem + ".components.json"
        if os.path.isfile(cand):
            return cand
    return ""


def resolve_sidecar_path(explicit=None, layer_real_paths=()) -> str:
    """Resolve the sidecar path for a board.

    Order: an ``explicit`` path wins (returned only if it exists — an
    explicit-but-missing path never silently falls back); otherwise the
    first entry of ``layer_real_paths`` with a sidecar next to it. Callers
    pass reference-arc target layers or the stage's root-layer realPath.
    "" means no
    sidecar (bbox fallback).
    """
    explicit = str(explicit or "")
    if explicit:
        return explicit if os.path.isfile(explicit) else ""
    for real in layer_real_paths:
        cand = sidecar_path_for_layer(real)
        if cand:
            return cand
    return ""


def load_component_index(path) -> dict | None:
    """Parse ``path`` into ``{instance path: component_instances[] entry}``;
    ``None`` when the file is missing or unreadable."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return {e["path"]: e for e in data.get("component_instances", []) if "path" in e}


def lookup(index: dict, prim_path) -> tuple:
    """Find the sidecar entry for a composed-stage prim path.

    Sidecar paths are authored relative to the board layer (e.g.
    ``/Board_X/Components/Top/FAM/R48``) while the composed stage mounts the
    board under a prefix (e.g. ``/World``). Walk the prim path and its
    ancestors, matching by suffix; return ``(entry, prefix)`` where
    ``prefix + entry["path"] == matched stage path`` so the same prefix
    can remap the entry's pad/body prim paths back into the stage.
    ``(None, "")`` when nothing matches.
    """
    pth = str(prim_path)
    while pth and pth != "/":
        keys = (pth,) + ((pth[len("/World") :],) if pth.startswith("/World/") else ())
        for key in keys:
            entry = index.get(key)
            if entry is not None:
                return entry, pth[: len(pth) - len(key)]
        pth = pth.rsplit("/", 1)[0]
    return None, ""


def remap_record(entry: dict, prefix: str) -> dict:
    """Copy of ``entry`` with its :data:`REMAP_FIELDS` prim-path lists
    remapped into the composed stage by ``prefix``."""
    record = dict(entry)
    for field in REMAP_FIELDS:
        if entry.get(field):
            record[field] = [prefix + p for p in entry[field]]
    return record
