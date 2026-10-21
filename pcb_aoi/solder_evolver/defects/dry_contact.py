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

"""DryContact through-hole joint -- the fillet does not ring the barrel.

THT only (``kinds``): the construction is a bite out of the pad
(``tht.interrupt_region``), which has no SMT counterpart -- an SMT work item
asking for this is INELIGIBLE, not silently nominal.

The observable is ``gap_fraction``: the fraction of the barrel circumference
NOT wetted by the fillet. The strategy owns the translation to the cut's
internals: the biting front at ``bite=2.5`` is nearly a straight chord, and a
chord crossing the gap axis at signed distance ``t`` from the hole centre
leaves an unwetted arc of ``arccos(t / r) / pi`` -- so

    front = r * cos(pi * gap_fraction)

(gap 0.5 puts the front through the centre, matching the construction's own
``front=0.0`` cuts-through-the-middle convention). Measured against
``interrupt_region`` directly: gaps 0.15/0.35/0.50/0.75 produce unwetted
fractions 0.84/0.63/0.48/0.23 -- within a few percent, the residual being the
front's slight concave curl. The azimuth is ALWAYS randomized per joint and
never configured:
where the break faces is not part of the defect's definition, and fixing it
would only make datasets less diverse.

The class check that cannot lie is topological, not volumetric: a nominal THT
fillet rings the pin (genus 1); a dry-contact one must not (genus 0). A
"thinner but still ringing" fillet is not dry contact however plausible it
looks -- hence ``expect_genus0``, verified post-solve.
"""

from __future__ import annotations

import numpy as np

from .base import DefectStrategy, sample_scalar


class DryContact(DefectStrategy):
    defect_type = "dry_contact"
    label = "dry_contact"
    severity = "bad"
    description = "THT fillet with an unwetted arc: does not ring the barrel"

    kinds = ("tht",)
    expect_genus0 = True

    #: Default unwetted fraction of the barrel circumference. A third of the
    #: ring reads clearly as a break without degenerating into "barely any
    #: solder", which would be insufficient, a different class.
    def __init__(self, gap_fraction=0.35, seed=None, lead_climb=None,
                 fullness=None):
        self.gap_fraction = gap_fraction
        self.seed = seed
        self.lead_climb = lead_climb
        self.fullness = fullness

    def configure(self, record):
        return DryContact(
            gap_fraction=record.get("gap_fraction", self.gap_fraction),
            seed=record.get("seed", self.seed),
            lead_climb=record.get("lead_climb"),
            fullness=record.get("fullness"),
        )

    def tht_cut_kwargs(self, joint_index, hole_r):
        # Deterministic per (record seed, joint): the same request re-solves to
        # the same geometry, and twin joints on one component still break at
        # independent azimuths. gap_fraction is sampled PER JOINT (the range is
        # a per-joint severity distribution, unlike the per-component volume
        # ratios).
        rng = np.random.default_rng(
            None if self.seed is None else (int(self.seed) + int(joint_index)))
        gap = min(0.999, max(1e-3, sample_scalar(self.gap_fraction, rng)))
        return {
            "interrupt": True,
            "interrupt_angle": float(rng.uniform(0.0, 2.0 * np.pi)),
            "interrupt_front": float(hole_r * np.cos(np.pi * gap)),
        }
