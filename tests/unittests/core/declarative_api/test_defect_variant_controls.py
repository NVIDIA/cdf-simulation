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

"""The stratified-variant controls are reachable from the declarative API.

The defect maths gained deterministic variant selection and extra sideflip
perturbation, but those knobs could only be set by the standalone capture
script. A capability the Python job API cannot ask for does not exist as far as
the declarative surface is concerned, so this covers the whole path: declare it
on the job, compile it, and validate it against the executor contract.
"""

import sys
import unittest

from cdf.errors import ValidationError

from pydantic import ValidationError as PydanticValidationError
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: F401,E402 -- registers the `pcb-aoi` CDF domain
from cdf import CDFJob  # noqa: E402
from cdf.execution.isaac.job_order_parameters import (  # noqa: E402
    MissingParameters,
    ShiftParameters,
    SideflipParameters,
)


def _order(job, order_type):
    for order in job.compile_orders().orders:
        if order.order_type == order_type:
            return order
    raise AssertionError("no %s order was compiled" % order_type)


def _job():
    job = CDFJob(domain="pcb-aoi", random_seed=11)
    job.scene.configure(usd_path="/tmp/board.usd", root_path="/Board")
    job.scene.select_components(types=["PASSIVE_SMD_0402"])
    job.capture.renderer(mode="pathtracing", resolution=(64, 64), spp=1,
                         total_spp=1, max_bounces=2)
    job.capture.camera(rotation_range={"x": (0, 0), "y": (0, 0), "z": 0})
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.outputs(rgb=True)
    return job


class ShiftVariantTest(unittest.TestCase):
    """``variant_index`` reaches the shift contract."""

    def test_variant_index_is_carried_to_the_executor(self):
        job = _job()
        job.operations.add_shift(
            ratio=0.2, translate_range=0.2, rotate_z_range=15.0, variant_index=7)
        order = _order(job, "operation.shift.apply")
        self.assertEqual(
            ShiftParameters.model_validate(dict(order.parameters)).variant_index, 7)

    def test_variant_index_defaults_to_the_random_draw(self):
        job = _job()
        job.operations.add_shift(ratio=0.2, translate_range=0.2, rotate_z_range=15.0)
        order = _order(job, "operation.shift.apply")
        self.assertIsNone(
            ShiftParameters.model_validate(dict(order.parameters)).variant_index)

    def test_a_negative_variant_index_is_rejected(self):
        with self.assertRaises(PydanticValidationError):
            ShiftParameters.model_validate(
                {"ratio": 0.2, "translate_range": 0.2,
                 "rotate_z_range": 15.0, "variant_index": -1})


class SideflipVariantTest(unittest.TestCase):
    """The sideflip perturbation controls reach the contract."""

    def test_all_three_controls_are_carried_to_the_executor(self):
        job = _job()
        job.operations.add_sideflip(
            ratio=0.3, angle_range=(70, 90), variant_index=3,
            rotate_z_jitter_deg=2.5, slide_range_mm=0.4)
        validated = SideflipParameters.model_validate(
            dict(_order(job, "operation.sideflip.apply").parameters))
        self.assertEqual(validated.variant_index, 3)
        self.assertEqual(validated.rotate_z_jitter_deg, 2.5)
        self.assertEqual(validated.slide_range_mm, 0.4)

    def test_perturbation_defaults_to_no_extra_motion(self):
        job = _job()
        job.operations.add_sideflip(ratio=0.3, angle_range=(70, 90))
        validated = SideflipParameters.model_validate(
            dict(_order(job, "operation.sideflip.apply").parameters))
        self.assertEqual(validated.rotate_z_jitter_deg, 0.0)
        self.assertEqual(validated.slide_range_mm, 0.0)

    def test_negative_perturbation_is_rejected_at_the_boundary(self):
        for field in ("rotate_z_jitter_deg", "slide_range_mm"):
            with self.subTest(field=field):
                with self.assertRaises(PydanticValidationError):
                    SideflipParameters.model_validate(
                        {"ratio": 0.3, "angle_range": (70, 90), field: -1.0})


class MissingHasNoMechanismKnobTest(unittest.TestCase):
    """``missing`` exposes intent only; the hide mechanism is derived."""

    def test_add_missing_takes_no_mechanism_argument(self):
        # There are two hides -- author visibility, or prune the subtree -- and
        # which one is correct depends on where the semantic labels live and on
        # whether the Fabric Scene Delegate is on. Neither is the caller's
        # business, and an earlier revision of this branch did expose it.
        job = _job()
        with self.assertRaises(TypeError):
            job.operations.add_missing(ratio=0.1, render_effective=True)

    def test_missing_parameters_carry_no_mechanism_field(self):
        job = _job()
        job.operations.add_missing(ratio=0.1)
        parameters = dict(_order(job, "operation.missing.apply").parameters)
        self.assertNotIn("render_effective", parameters)
        self.assertFalse(
            hasattr(MissingParameters.model_validate(parameters),
                    "render_effective"))


class ShiftVariantPlanIndependenceTest(unittest.TestCase):
    """Every stratification axis must vary independently of every other.

    Regression lock for a defect found in review: reusing strides made each
    direction sector permanently pair with one rotation bin and one rotation
    sign, so only 72 of the 3 x 8 x 3 x 2 x 2 = 288 combinations were reachable
    and a component shifted into sector 0 could never receive a large rotation.
    That is precisely the clumping stratification exists to remove.
    """

    _PERIOD = 3 * 8 * 3 * 2 * 2

    def _plans(self):
        # component_rotation reaches omni.usd through transform_utils at module
        # scope, which is why this plan function had no off-GPU test until now.
        # The plan itself is pure integer arithmetic, so stubbing the Kit-only
        # names is enough to reach it.
        import importlib
        # pxr has to be stubbed as well as omni: the CI unit-test image has no
        # USD, and transform_utils imports UsdGeom at module scope. Stubbing
        # only omni passes on a developer machine and errors in CI.
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import kitless

        module = kitless.import_standalone("defects_utils.component_rotation")
        plans = [module._shift_variant_plan(v) for v in range(self._PERIOD)]
        kitless.forget_standalone()
        return plans

    def test_every_combination_is_reachable(self):
        plans = self._plans()
        combos = {
            (p["mode"], p["sector"], p["translate_bin"], p["rotate_bin"],
             p["rotate_sign"])
            for p in plans
        }
        self.assertEqual(len(combos), self._PERIOD)

    def test_no_axis_is_locked_to_the_direction_sector(self):
        plans = self._plans()
        for axis, expected in (("rotate_bin", 2), ("rotate_sign", 2),
                               ("translate_bin", 3), ("mode", 3)):
            by_sector = {}
            for plan in plans:
                by_sector.setdefault(plan["sector"], set()).add(plan[axis])
            for sector, values in sorted(by_sector.items()):
                with self.subTest(axis=axis, sector=sector):
                    self.assertEqual(
                        len(values), expected,
                        "sector %d only ever sees %s=%s" % (sector, axis,
                                                            sorted(values)))


class ApiBoundaryValidationTest(unittest.TestCase):
    """Bad stratification inputs fail at the call, not somewhere downstream."""

    def test_variant_index_rejects_bool_negative_and_non_integer(self):
        for bad in (True, -1, 1.5, "3"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    _job().operations.add_shift(
                        count=1, translate_range=0.2, rotate_z_range=5.0,
                        variant_index=bad)

    def test_sideflip_magnitudes_reject_bool_and_negative(self):
        for field in ("rotate_z_jitter_deg", "slide_range_mm"):
            for bad in (True, -1.0, "2"):
                with self.subTest(field=field, bad=bad):
                    with self.assertRaises(ValidationError):
                        _job().operations.add_sideflip(
                            count=1, angle_range=(60.0, 80.0), **{field: bad})

    def test_valid_values_still_pass(self):
        job = _job()
        job.operations.add_shift(count=1, translate_range=0.2,
                                 rotate_z_range=5.0, variant_index=7)
        job.operations.add_sideflip(count=1, angle_range=(60.0, 80.0),
                                    variant_index=3, rotate_z_jitter_deg=2.0,
                                    slide_range_mm=0.1)


if __name__ == "__main__":
    unittest.main()
