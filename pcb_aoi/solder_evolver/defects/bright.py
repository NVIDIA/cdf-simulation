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

"""Bright solder -- a steep, starved meniscus that still wets the lead.

The AOI "bright" class: too little solder pulled taut between the pad edge
and a respectable climb up the lead, so the surface necks in and reads shiny
under ring lighting. Distinct from "less solder" (a valley: the lead barely
wets at all) precisely along the two shape axes -- bright climbs, less-solder
does not; both are far below a full fillet.

Parameter-only, and THT-only: the SMT taxonomy has no bright class, and an
SMT component selected into this pool is reported INELIGIBLE rather than
silently soldered nominal. Both parameters follow the shape convention
(scalar or [min, max], sampled per joint against the record seed).
"""

from __future__ import annotations

from .base import DefectStrategy


class Bright(DefectStrategy):
    defect_type = "bright"
    label = "bright"
    severity = "bad"
    description = "steep starved meniscus: climbs the lead but necks in"

    kinds = ("tht",)
    verify_volume = True

    #: The defect's home range: real climb (the lead wets), starved fill
    #: (the silhouette necks). fullness above ~0.6 starts reading nominal;
    #: below ~0.35 the joint reads as less-solder instead.
    def __init__(self, lead_climb=(0.6, 0.9), fullness=(0.35, 0.6),
                 seed=None):
        self.lead_climb = lead_climb
        self.fullness = fullness
        self.seed = seed

    def configure(self, record):
        if not any(record.get(k) is not None for k in
                   ("lead_climb", "fullness", "seed")):
            return self
        return Bright(
            lead_climb=record.get("lead_climb", self.lead_climb),
            fullness=record.get("fullness", self.fullness),
            seed=record.get("seed"))
