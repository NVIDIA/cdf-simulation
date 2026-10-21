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

"""Excess solder -- an over-filled joint on an otherwise correct footprint.

The mirror of :mod:`insufficient`, and parameter-only for the same reason: the
footprint is unchanged and only the target volume grows. Scaling the blob
instead would move the contact line, conflating "more solder" with "wets
further" -- two different defects the taxonomy needs kept apart.

Opposite solver risk to the starved case. Excess pushes the meniscus AWAY from
the part rather than pressing it against the keep-out, so the collider is not
load-bearing here and Newton is comfortable; the failure mode to watch is the
free surface bulging until it touches a neighbouring joint, which is a BRIDGE
and a different defect. Nothing here detects that yet.
"""

from __future__ import annotations

from .base import DefectStrategy


class Excess(DefectStrategy):
    defect_type = "excess"
    label = "excess"
    severity = "bad"
    description = "over-filled joint: increased volume on the nominal footprint"

    verify_volume = True

    #: Multiple of the printed volume left on the joint.
    #: 2.6, not 1.8: 1.8x on a PINNED footprint barely shows, because the
    #: contact line cannot spread and the extra volume only bulges the free
    #: surface a little. Excess has to be gross to read as excess.
    def __init__(self, fraction=2.6, lead_climb=None, fullness=None,
                 seed=None):
        self.fraction = float(fraction)
        self.lead_climb = lead_climb
        self.fullness = fullness
        self.seed = seed

    def configure(self, record):
        # volume_ratio (> 1, per-component scalar) scales the SMT nominal.
        # On THT the hidden/excess look is lead_climb > 1 (tip buried) with
        # fullness >= 1; either being set bypasses target_volume (see base).
        if not any(record.get(k) is not None for k in
                   ("volume_ratio", "lead_climb", "fullness", "seed")):
            return self
        return Excess(
            fraction=(float(record["volume_ratio"])
                      if record.get("volume_ratio") is not None
                      else self.fraction),
            lead_climb=record.get("lead_climb"),
            fullness=record.get("fullness"),
            seed=record.get("seed"))

    def options(self, base):
        o = dict(base)
        # Same footprint and same minimiser as nominal -- only the volume moves.
        # Newton is safe here (see the module note), so the ladder pass stays on.
        o["blob_scale"] = 1.6
        o["skirt"] = 0.10
        o["hess_stage"] = 10
        o["sdf"] = False
        return o

    def target_volume(self, nominal_volume, joint_index):
        return nominal_volume * self.fraction
