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

"""Nominal ("good") joint -- the reference every defect is a deviation from.

Two things about the pinned formulation that decide what is tunable here.

CONTACT ANGLE IS NOT A KNOB. The wetted patches are emitted `fixed`, so their
`-cos(theta)` tension is a constant in the energy and cannot influence the
minimisation. Measured: theta 25/35, 80/80 and 5/5 give identical geometry to
5e-13 mm. `theta_board` / `theta_comp` survive only for a future free-contact-
line formulation; changing them today does nothing.

WHAT DOES SET THE ANGLE is the footprint size against the volume. Measured on an
0805 at nominal volume:

    blob_scale  1.00 -> 86 deg   1.25 -> 55 deg   1.60 -> 35 deg   2.20 -> 11 deg
    fill        0.50 -> 32 deg   0.75 -> 41 deg   1.00 -> 55 deg   1.50 -> 70 deg

`fill` must stay at nominal (it IS the volume), so the footprint is the free
knob. 1.25 gave a bulbous ~55 deg joint; 1.6 gives ~35 deg, a concave fillet of
roughly the shape a reflowed chip termination should show.

That this is a fudge factor is worth stating plainly: physically the footprint
should come from the PAD (bounded by the solder-mask opening) while the volume
comes from the PASTE, and real pads are larger than their paste apertures. We
read paste for both, so `blob_scale` is standing in for the pad. The USD spec
carries `board_pad` + `padContour`; reading it is the real fix.

NO COLLIDER by default. A nominal joint has enough solder to sit on its own
footprint without being pulled through the part, and the keep-out costs roughly
8x per iteration. Starved joints are the ones that need it.
"""

from __future__ import annotations

from .base import DefectStrategy


class Nominal(DefectStrategy):
    defect_type = "nominal"
    label = "good"
    severity = "good"
    description = "nominal joint: printed volume on a pad-sized footprint"

    #: Footprint inflation. Sets the meniscus angle; see the table above.
    blob_scale = 1.6

    #: Pad overhang beyond the paste aperture, in mm. Makes the joint FLOOD its
    #: pad instead of sitting on a footprint pinched in from the aperture. Small
    #: on purpose: real paste apertures run close to 1:1 with the pad, and on an
    #: 0805 a 0.25 mm skirt spreads the nominal volume into a flat pancake with
    #: no fillet at all. Still a stand-in for the real pad -- see the note above.
    skirt = 0.10

    #: Newton steps per level of the refine ladder. The fixed recipe on its own
    #: leaves a faceted golfball; a short hessian_seek burst at each level snaps
    #: it into shape, and it is nearly free because factorisation cost goes with
    #: vertex count, so the coarse levels -- where the faceting is -- are cheap.
    #: Measured on SWITCH_DIP j0 at nominal fill:
    #:
    #:     plain                    energy 0.95961   11.0 s
    #:     hess_stage 10            energy 0.22682   13.0 s
    #:     convergence tail only    energy 0.22806   57.9 s
    #:
    #: Safe HERE specifically because nominal runs with no collider. Newton is
    #: the wrong choice for a starved joint pressed along a keep-out.
    hess_stage = 10

    def options(self, base):
        o = dict(base)
        o["blob_scale"] = self.blob_scale
        o["skirt"] = self.skirt
        o["hess_stage"] = self.hess_stage
        o["sdf"] = False
        return o
