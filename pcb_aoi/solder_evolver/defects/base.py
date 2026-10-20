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

"""One strategy per defect type, each able to replace geometry steps.

A flat parameter table is not enough, because defects do not all differ in the
same way:

  * **parameter only** -- insufficient / excessive are a target-volume scale;
  * **different geometry** -- a bridge fuses two pads into one joint, which is
    the inverse of the clustering a nominal joint wants; a tombstoned part has
    left the pad, so the footprint is derived differently;
  * **no geometry at all** -- "missing" produces nothing to solve.

So a strategy is an object with hooks at the points where the pipeline actually
diverges, not a dict of knobs. The base class implements the NOMINAL behaviour,
so a subclass overrides only what it changes and everything else stays on the
one shared path.

Deliberately outside ``cdf/``: these manipulate geometry, so the viewer must
be able to use them without importing ``pcb_aoi``. (The volume rule learned this
the hard way -- see geometry/solder_volume.py.)
"""

from __future__ import annotations


def sample_scalar(spec, rng):
    """A configured observable: a value, or a ``[min, max]`` range to sample.

    The configuration allows a range on every scalar parameter; whoever holds
    the rng (per component for volume ratios, per joint for the dry-contact
    gap) collapses it to a number here.
    """
    if isinstance(spec, (list, tuple)):
        lo, hi = float(spec[0]), float(spec[1])
        return float(rng.uniform(lo, hi)) if hi > lo else lo
    return float(spec)


class DefectStrategy:
    """Nominal behaviour. Subclass and override only the differing hooks."""

    #: Matched against ``SolderDefect.defect_type`` (or an explicit request).
    defect_type = "nominal"

    #: Class reported for the produced joint, for downstream labelling.
    label = "good"

    #: Severity in the AOI sense: good | warning | bad.
    severity = "good"

    #: Human-readable, surfaced in coverage messages.
    description = "nominal joint: printed volume, derived footprint"

    #: Work-item kinds the strategy can honour. A construction that only
    #: exists on one side (dry contact is a THT cut) narrows this, so a work
    #: item of the other kind is INELIGIBLE coverage rather than a silently
    #: nominal -- and mislabelled -- joint.
    kinds = ("smt", "tht")

    #: Post-solve observable checks (see the solder-defects proposal's
    #: verification contract): compare each solved joint's mesh volume to its
    #: requested target, warn beyond the tolerance, then DISCARD the number.
    verify_volume = False
    volume_tolerance = 0.10
    #: THT topology check: the fillet must NOT ring the barrel (genus 0).
    expect_genus0 = False

    #: THT SHAPE PARAMETERS (the two documented ones): scalar or [min, max],
    #: sampled per joint by the THT solve path. None defers to the solver
    #: options (nominal defaults). A strategy that sets either one DRIVES the
    #: joint's target volume itself -- target_volume() is then not applied on
    #: THT joints, so a ratio default cannot double-scale a shaped defect.
    lead_climb = None
    fullness = None
    #: Per-record rng seed (the runtime stamps one per selected component);
    #: per-joint draws key off (seed + joint index).
    seed = None

    # ------------------------------------------------------------- configure

    def configure(self, record):
        """Return a strategy carrying the record's observable parameters.

        `record` is the defect record the configuration produced (the solver
        copies it into ``SolderDefect.metadata`` verbatim). The registry keeps
        one shared instance per defect type, so a strategy with parameters
        must return a NEW instance here -- never mutate self. The base
        strategy has no parameters and returns itself.
        """
        return self

    # ---------------------------------------------------------------- options

    def options(self, base):
        """Return the solver options for this defect.

        `base` is the resolved default set. Return a NEW dict; do not mutate.
        """
        return dict(base)

    # --------------------------------------------------------------- geometry

    def cluster(self, rects, gap):
        """Group paste deposits into joints. Returns a list of index lists.

        The nominal rule is proximity: deposits closer than `gap` are one joint
        (a QFN thermal pad), anything further apart is a separate joint (the two
        ends of an 0805). A bridge strategy overrides this to fuse instead.
        """
        from ..boolean import cluster_rects

        return cluster_rects([tuple(r) for r in rects], gap=gap)

    def joint_ids(self, all_ids):
        """Which of the clustered joints to actually solve. Nominal: all."""
        return list(all_ids)

    def target_volume(self, nominal_volume, joint_index):
        """Solder volume for one joint. Nominal: as printed.

        `nominal_volume` already accounts for measured paste thickness and metal
        fraction (geometry/solder_volume.py). Returning 0 or None skips the
        joint, which is how "no solder here" is expressed.
        """
        return nominal_volume

    def tht_cut_kwargs(self, joint_index, hole_r):
        """Extra kwargs for ``tht.cut_tht_joint`` on one joint. Nominal: none.

        This is where a strategy replaces the THT cut step -- dry contact
        translates its observable (gap_fraction) into the interrupt-cut
        construction here. `hole_r` is the joint's barrel radius, needed for
        any translation between circumference fractions and distances.
        """
        return {}

    def smt_cut_kwargs(self, joint_index):
        """Extra kwargs for ``boolean.cut_joint`` on one SMT joint.

        The SMT counterpart of ``tht_cut_kwargs`` -- interrupted translates
        its pad_exposure observable into the exposure-clip construction
        here. Nominal: none.
        """
        return {}

    # ------------------------------------------------------------- post-solve

    def postprocess(self, meshes):
        """Last look at the solved meshes before they are returned."""
        return meshes

    def __repr__(self):                                  # pragma: no cover
        return "<%s %s>" % (type(self).__name__, self.defect_type)
