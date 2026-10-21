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

"""Defect strategies: one module per defect, resolved by name.

    from pcb_aoi.solder_evolver.defects import resolve
    strategy = resolve("insufficient")

Each strategy may replace GEOMETRY STEPS, not just supply parameters -- see
``base.DefectStrategy`` for why a parameter table is not sufficient.

Adding one: write a module here, subclass ``DefectStrategy``, override only the
hooks that differ, and register it below. Keeping them in separate modules is
deliberate: the defects are being customised independently, and a single shared
table would couple them.

Unregistered names resolve to NOTHING rather than silently falling back to
nominal -- a defect that quietly produced a good joint would be mislabelled
training data, which is worse than a hard failure.
"""

from __future__ import annotations

from .base import DefectStrategy
from .bright import Bright
from .insufficient import Insufficient
from .interrupted import Interrupted
from .dry_contact import DryContact
from .missing import Missing
from .excess import Excess
from .nominal import Nominal

_REGISTRY = {}


def register(strategy):
    """Register a strategy instance under its ``defect_type``."""
    _REGISTRY[strategy.defect_type] = strategy
    return strategy


register(Nominal())
register(Bright())
register(Insufficient())
register(Interrupted())
register(Excess())
register(Missing())
register(DryContact())

# Pose defects arrive with the component already moved on the stage, so the
# solder strategy is nominal on the geometry that results. Tombstone is a FIRST
# CUT here (a deliberate call): nominal on the MOVED geometry -- the pivot end
# reads right, the lifted end domes/stretches. Its true lifted-joint footprint
# (a pinned pad with a stretched fillet up to the raised terminal) is still to
# be written; until then this at least renders something to iterate against.
for _pose in ("shift", "sideflip", "reverse_polarity", "tombstone"):
    _REGISTRY[_pose] = Nominal()


def known():
    """Registered defect names, sorted."""
    return sorted(_REGISTRY)


def resolve(name):
    """Strategy for `name`, or None if unregistered.

    Returning None rather than raising lets the caller report an INELIGIBLE
    coverage row -- the contract wants an unhandled work item reported, not an
    exception that takes the whole request down.
    """
    if not name:
        return _REGISTRY["nominal"]
    return _REGISTRY.get(str(name))


__all__ = ["DefectStrategy", "Bright", "Insufficient", "Interrupted",
           "DryContact", "Excess", "Missing", "Nominal",
           "known", "register", "resolve"]
