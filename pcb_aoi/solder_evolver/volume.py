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

"""How much solder a printed joint ends up with. One definition, shared.

Deliberately here and not in ``cdf/``: how much solder a deposit yields is a
physical question, not a contract question, and the viewer must be able to ask
it without importing ``pcb_aoi``.

    solder volume = paste area x paste thickness x metal fraction

Both factors were previously wrong. The thickness was hardcoded at 0.13 mm while
every deposit on the reference board measures 0.076 -- and the deposits are
authored as real 3D geometry, so it can simply be measured. The metal fraction
was ignored entirely: printed paste is roughly half flux by volume and the flux
boils off during reflow. Together the old nominal was 3.42x too much solder.
"""

from __future__ import annotations

#: Printed paste -> post-reflow solder. NOT a measurement: it depends on paste
#: type, and neither the components sidecar nor the USD spec carries it. Named
#: here so it is one number to argue with rather than a buried assumption.
METAL_FRACTION = 0.5

#: Cut the blob this much larger than the target volume. The footprint comes
#: from the cut, so the solve then shrinks into a well-defined contact line --
#: rather than `match_volume` inflating the blob to reach the target and
#: dragging the wetted patch up the part (the 96%-wrap failure). Larger gives a
#: more stable footprint but a more pulled-in joint at nominal.
BLOB_SCALE = 1.25

#: Fallback thickness (mm) when no geometry is available to measure. Measured
#: on the reference board; it is a per-board stencil choice, not a constant.
STENCIL_MM = 0.076


def volume_plan(paste_thickness=None, metal_fraction=None, blob_scale=None):
    """Return ``(blob_stencil, target_fraction)``.

    * ``blob_stencil`` -> ``cut_joint(stencil=..., fill=1.0)``; sets the blob,
      and therefore the footprint.
    * ``target_fraction`` -> multiply ``res["target_volume"]`` by this (and by
      the caller's fill) to get the volume to solve to.

    Split because the blob is cut deliberately larger than the target. Callers
    that compute either half themselves will drift from each other; that is the
    entire reason this function exists.
    """
    thick = float(paste_thickness if paste_thickness else STENCIL_MM)
    mf = float(METAL_FRACTION if metal_fraction is None else metal_fraction)
    bs = float(BLOB_SCALE if blob_scale is None else blob_scale)
    return thick * mf * bs, 1.0 / bs
