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

"""Interrupted solder -- the pad itself is partially EXPOSED.

The AOI "interrupted" class (formally distinct from dry_contact, which this
codebase once used the name for): the joint is present and, on THT, still
rings the pin -- but the fillet's foot stops short of the pad edge, leaving
bare copper visible. This is why the pad-outline work mattered: the exposed
region is measured against the joint's real pad contour.

Both kinds. The mechanism is a construction-time clip (boolean.exposure_keep):
a noisy dewetting front pushed in from a random azimuth, with a core that
keeps the joint connected; the front depth is bisected so the exposed-area
fraction lands on ``pad_exposure`` exactly, and the pinned contact line
guarantees it survives the solve. On THT the ring must STAY closed (genus 1)
-- a broken ring is dry_contact, a different class.
"""

from __future__ import annotations

import numpy as np

from .base import DefectStrategy, sample_scalar


class Interrupted(DefectStrategy):
    defect_type = "interrupted"
    label = "interrupted"
    severity = "bad"
    description = "pad partially exposed: the foot stops short of the pad edge"

    verify_volume = True
    #: THT: the ring around the pin must survive the bite.
    expect_genus1 = True

    #: A third to half of the pad showing reads clearly as interrupted
    #: without collapsing into missing; below ~0.15 it is invisible.
    def __init__(self, pad_exposure=(0.2, 0.5), seed=None,
                 lead_climb=None, fullness=None):
        self.pad_exposure = pad_exposure
        self.seed = seed
        self.lead_climb = lead_climb
        self.fullness = fullness

    def configure(self, record):
        return Interrupted(
            pad_exposure=record.get("pad_exposure", self.pad_exposure),
            seed=record.get("seed"),
            lead_climb=record.get("lead_climb"),
            fullness=record.get("fullness"))

    def _expose_kwargs(self, joint_index):
        # Deterministic per (record seed, joint), azimuth always randomized
        # -- same contract as dry_contact's gap. pad_exposure samples per
        # JOINT (a range is a per-joint severity distribution).
        rng = np.random.default_rng(
            None if self.seed is None else (int(self.seed) + int(joint_index)))
        return {
            "expose": float(min(0.95, max(1e-3,
                                          sample_scalar(self.pad_exposure,
                                                        rng)))),
            "expose_angle": float(rng.uniform(0.0, 2.0 * np.pi)),
            "expose_seed": int(rng.integers(0, 2**31 - 1)),
        }

    def tht_cut_kwargs(self, joint_index, hole_r):
        return self._expose_kwargs(joint_index)

    def smt_cut_kwargs(self, joint_index):
        return self._expose_kwargs(joint_index)
