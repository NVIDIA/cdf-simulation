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

"""Insufficient solder -- a starved joint on an otherwise correct footprint.

Parameter-only: the footprint is unchanged and only the target volume shrinks.
That separation is the point. Scaling the BLOB instead would move the contact
line, which conflates "less solder" with "wets less far" -- two different
defects that the taxonomy needs kept apart.

The starved case is also where the solver is least comfortable: the meniscus is
pressed hardest against the keep-out, which is exactly where Hessian
minimisation stalls (docs/hessian_minimisation.md). Gradient descent is left in
place here deliberately.
"""

from __future__ import annotations

from .base import DefectStrategy


class Insufficient(DefectStrategy):
    defect_type = "insufficient"
    label = "insufficient"
    severity = "bad"
    description = "starved joint: reduced volume on the nominal footprint"

    verify_volume = True

    #: Fraction of the printed volume left on the joint.
    #: 0.50, not 0.30: at a third of nominal the joint is barely there and
    #: reads as missing rather than starved. Half is visibly thin against
    #: nominal while still being a JOINT.
    def __init__(self, fraction=0.5, lead_climb=None, fullness=None,
                 seed=None):
        self.fraction = float(fraction)
        self.lead_climb = lead_climb
        self.fullness = fullness
        self.seed = seed

    def configure(self, record):
        # volume_ratio (per-component scalar) scales the SMT nominal -- and
        # the THT one too, when no shape parameter is given. lead_climb /
        # fullness (scalar or per-joint range) shape THT joints directly; when
        # either is set, target_volume() is bypassed on THT (see base).
        if not any(record.get(k) is not None for k in
                   ("volume_ratio", "lead_climb", "fullness", "seed")):
            return self
        return Insufficient(
            fraction=(float(record["volume_ratio"])
                      if record.get("volume_ratio") is not None
                      else self.fraction),
            lead_climb=record.get("lead_climb"),
            fullness=record.get("fullness"),
            seed=record.get("seed"))

    def target_volume(self, nominal_volume, joint_index):
        return nominal_volume * self.fraction
