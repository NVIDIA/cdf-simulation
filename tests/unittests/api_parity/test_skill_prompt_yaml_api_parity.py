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

"""YAML-versus-Python-API parity for the cdf-simulation SQA twins.

One YAML file per DIG Flow Catalog row, plus the default-coverage tombstone
variant N07, lives under ``tests/functional/skill_prompts/``. Each builder
below is a hand-written Python API build of that job on the repository coupon
the prompt attaches, and the test asserts that the YAML and the builder compile
to the same ``JobOrderPlan``.

What this does and does not prove: it proves the YAML interpreter and the
Python API agree on one job, so the YAML SQA executes is the plan the Python
side would hand to Airflow. It does **not** read ``skills/cdf-simulation/
SKILL.md`` -- the builders are written by hand, not imported from the
templates -- so a template can drift while this stays green. The evidence
that the templates, and an agent following them, reach these plans requires
reviewing the templates alongside their twins. A drift caught here is still a
defect in one of the two artefacts:
either the YAML no longer runs what the prompt promises, or the Python
spelling of the job moved and the plan SQA tests against is stale.

The hand-off representation is checked too: the skill emits
``JobOrderCodec.dumps(plan)`` (a bare ``plan.model_dump()`` carries
``mappingproxy`` and tuple values and is not JSON-serializable), and the
receiving side must get an equal plan back from ``JobOrderCodec.loads``.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi domain
from cdf import JobOrderCodec, JobOrderPlan, CDFJob, load_job
from pcb_aoi.solder_generation import registered_solver_ids

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPTS_DIR = _REPO_ROOT / "tests" / "functional" / "skill_prompts"
_RESOURCES = _REPO_ROOT / "tests" / "functional" / "resources"

# The attachments the SQA prompts name, and the one component family each
# coupon carries. Embedded USD metadata, so ``source="usd"`` and no sidecar.
_THT_COUPON = _RESOURCES / "ARTIX_PINHEADER_J13_coupon.usdc"
_THT_TYPE = "PinHeader_1x3_P2.54mm_Drill1.1mm"
_THT_REFDES = "J13"
_SMT_COUPON = _RESOURCES / "ICT_C0805_C22_coupon.usdc"
_SMT_TYPE = "C_0805_2012Metric"

# SQA harness arguments carried in every prompt, replacing the 1920x1080 / 32
# spp catalog defaults so a case renders in CI time.
_RESOLUTION = (640, 360)
_TOTAL_SPP = 4

# The skill's fixed defaults for rows 4-5: "full board coverage" and the
# no-phrase default. The executing side selects ``max(1, int(pool * ratio))``,
# so on the one-component coupon both yield exactly one defect.
_FULL_COVERAGE = 1.0
_DEFAULT_COVERAGE = 0.2
_TOMBSTONE_ANGLE = (70.0, 90.0)


def _intent(plan: JobOrderPlan) -> JobOrderPlan:
    """Return the plan with authoring provenance stripped.

    ``load_job()`` stamps ``metadata["source"]`` (the YAML's path and sha256);
    a plan built in Python has none and is not thereby a different plan. Same
    rule as ``test_job_config_parity._intent``: parity is a claim about intent.
    """
    metadata = {key: value for key, value in plan.metadata.items() if key != "source"}
    return plan.model_copy(update={"metadata": metadata})


def _lighting(job: CDFJob) -> None:
    """Declare the skill's fixed white-light default (every row)."""
    job.appearance.lighting(
        mode="white_light",
        exposure_range=(0.0, 0.0),
        cone_angle_range=(120.0, 120.0),
        cone_softness_range=(1.0, 1.0),
        dome={"intensity": (1000.0, 1000.0), "visible": True},
    )


def _renderer(job: CDFJob) -> None:
    """Declare the pathtracing renderer at the SQA harness resolution.

    ``output_start(0)`` is the API default; it is declared because the YAML
    interpreter always emits it, and the plans are compared whole.
    """
    job.capture.output_start(0)
    job.capture.renderer(
        mode="pathtracing",
        resolution=_RESOLUTION,
        spp=1,
        total_spp=_TOTAL_SPP,
        max_bounces=4,
    )


def _default_limits(job: CDFJob) -> None:
    """Declare the API's default limits, which the YAML interpreter emits."""
    job.capture.limits(triggers=1, max_images=-1)


def _annotated_outputs(job: CDFJob) -> None:
    """Declare rgb + tight boxes + colorized semantics (rows 1, 3, 4, 5)."""
    job.capture.outputs(
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        colorize_semantic_segmentation=True,
    )


def _component_crop(job: CDFJob) -> None:
    """Declare the per-component crop shared by P04 and every row 4-5 record.

    ``class_filter=None`` on purpose: a component-type string there is the
    documented silent-zero-files failure.
    """
    job.postprocess.crop(
        method="bbox_tight",
        types=["rgb", "semantic_segmentation", "component_instance"],
        class_filter=None,
        offset=10,
        xform_depth=7,
        output_subdir="cropped",
    )


def build_row1_golden_tht() -> CDFJob:
    """P01: golden THT board, procedural solder, no defect, no crop."""
    job = CDFJob(domain="pcb-aoi", random_seed=0)
    job.scene.configure(usd_path=_THT_COUPON, root_path="/Board")
    job.scene.select_components(types=[_THT_TYPE], source="usd")
    _lighting(job)
    job.operations.solder_joints(
        solver="procedural", component_types=[_THT_TYPE], source="usd"
    )
    job.annotations.label_components(source="usd")
    _renderer(job)
    job.capture.camera(side="bottom", rotation_range={"z": -90.0})
    job.capture.scan_grid(x_count=1, y_count=1)
    _default_limits(job)
    _annotated_outputs(job)
    return job


def build_row2_pin_tilt() -> CDFJob:
    """P02: THT pin tilt variation at the fixed 5-15 degree band."""
    job = CDFJob(domain="pcb-aoi", random_seed=0)
    job.scene.configure(usd_path=_THT_COUPON, root_path="/Board")
    _lighting(job)
    job.operations.pin_variation(components=[_THT_REFDES], tilt_range=(5.0, 15.0))
    _renderer(job)
    job.capture.camera(side="bottom", rotation_range={"z": -90.0})
    job.capture.scan_grid(x_count=1, y_count=1)
    _default_limits(job)
    job.capture.outputs(rgb=True)
    return job


def _smt_scene(job: CDFJob) -> None:
    """Configure the SMT coupon and narrow the selection to its one family."""
    job.scene.configure(usd_path=_SMT_COUPON, root_path="/Board")
    job.scene.select_components(types=[_SMT_TYPE], source="usd")
    _lighting(job)


def _smt_capture(job: CDFJob, *, max_images: int = -1) -> None:
    """Declare the top-side, one-frame SMT capture shared by P04 and rows 4-5."""
    _renderer(job)
    job.capture.camera(side="top", rotation_range={"z": 0.0})
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.limits(triggers=1, max_images=max_images)
    _annotated_outputs(job)


def _defect_annotations_and_crop(job: CDFJob) -> None:
    """Declare the Row 4/5 output contract after the operations.

    Every structural or solder defect record is reported as per-component
    defect crops: component labels, the ``defect`` operation namespace in the
    segmentation output, the annotated capture, then the crop.
    """
    job.annotations.label_components(source="usd")
    job.annotations.enable_operation_labels(namespace="defect")
    _smt_capture(job)
    _component_crop(job)


def build_row3_good_board() -> CDFJob:
    """P03: four good board images -> a 2x2 scan grid, no crop, no defect."""
    job = CDFJob(domain="pcb-aoi", random_seed=0)
    _smt_scene(job)
    job.annotations.label_components(source="usd")
    _renderer(job)
    job.capture.camera(side="top", rotation_range={"z": 0.0})
    job.capture.scan_grid(x_count=2, y_count=2)
    _default_limits(job)
    _annotated_outputs(job)
    return job


def build_row3_good_component_crop() -> CDFJob:
    """P04: good images of one family, max 3, bbox_tight crop, class_filter=None."""
    job = CDFJob(domain="pcb-aoi", random_seed=0)
    _smt_scene(job)
    job.annotations.label_components(source="usd")
    _smt_capture(job, max_images=3)
    _component_crop(job)
    return job


def _tombstone(job: CDFJob, *, ratio: float) -> None:
    """Declare a tombstone scoped by the narrowed selection.

    ``eligibility_source`` names the scene's metadata source; on an
    embedded-metadata board that is "usd" (the YAML interpreter derives it).
    No ``component_types`` here: the narrowed selection is the scope. On an
    embedded-metadata board a tombstone-level type list resolved to prim
    paths the stage did not carry and the operation applied nothing.
    """
    job.operations.add_tombstone(
        ratio=ratio,
        angle_range=_TOMBSTONE_ANGLE,
        eligibility_source="usd",
    )


def build_row4_tombstone() -> CDFJob:
    """P05: tombstone at full coverage, fixed 70-90 degree band."""
    job = CDFJob(domain="pcb-aoi", random_seed=0)
    _smt_scene(job)
    _tombstone(job, ratio=_FULL_COVERAGE)
    _defect_annotations_and_crop(job)
    return job


def build_row4_tombstone_default_coverage() -> CDFJob:
    """N07: P05 without "full board coverage" -- the default ratio applies.

    On the one-component coupon the executing side selects
    ``max(1, int(1 * 0.2)) = 1``: the default never resolves a non-empty
    selection to zero defects, and a clean board on this plan is a failure.
    """
    job = CDFJob(domain="pcb-aoi", random_seed=0)
    _smt_scene(job)
    _tombstone(job, ratio=_DEFAULT_COVERAGE)
    _defect_annotations_and_crop(job)
    return job


def build_row4_shift() -> CDFJob:
    """P06: shift at full coverage, fixed 0.2 mm / 15 degree magnitudes."""
    job = CDFJob(domain="pcb-aoi", random_seed=0)
    _smt_scene(job)
    job.operations.add_shift(
        ratio=_FULL_COVERAGE, translate_range=0.2, rotate_z_range=15.0
    )
    _defect_annotations_and_crop(job)
    return job


def build_row4_sideflip() -> CDFJob:
    """P07: sideflip at full coverage, fixed 70-90 degree band."""
    job = CDFJob(domain="pcb-aoi", random_seed=0)
    _smt_scene(job)
    job.operations.add_sideflip(ratio=_FULL_COVERAGE, angle_range=(70.0, 90.0))
    _defect_annotations_and_crop(job)
    return job


def build_row5_excess_solder() -> CDFJob:
    """P08: excess solder at full coverage on a surface_evolver solve."""
    job = CDFJob(domain="pcb-aoi", random_seed=0)
    _smt_scene(job)
    job.operations.solder_joints(
        solver="surface_evolver", component_types=[_SMT_TYPE], source="usd"
    )
    job.operations.add_excess(ratio=_FULL_COVERAGE, component_types=[_SMT_TYPE])
    _defect_annotations_and_crop(job)
    return job


#: One entry per SQA twin: the YAML SQA runs, and the Python API build.
CASES = (
    ("row1_golden_tht.yaml", build_row1_golden_tht),
    ("row2_pin_tilt.yaml", build_row2_pin_tilt),
    ("row3_good_board.yaml", build_row3_good_board),
    ("row3_good_component_crop.yaml", build_row3_good_component_crop),
    ("row4_tombstone.yaml", build_row4_tombstone),
    ("row4_tombstone_default_coverage.yaml", build_row4_tombstone_default_coverage),
    ("row4_shift.yaml", build_row4_shift),
    ("row4_sideflip.yaml", build_row4_sideflip),
    ("row5_excess_solder.yaml", build_row5_excess_solder),
)

_NEEDS_SIMULATED_SOLVER = frozenset({"row5_excess_solder.yaml"})

#: Rows 4-5 and P04 are reported as per-component crops; every one of their
#: plans must carry the crop order.
_CROPPED = frozenset(
    name
    for name, _ in CASES
    if name.startswith(("row4_", "row5_")) or name == "row3_good_component_crop.yaml"
)


class SkillPromptYamlApiParityTest(unittest.TestCase):
    """Each SQA YAML compiles to the plan its Python API build compiles to."""

    def _skip_without_solver(self, filename: str) -> None:
        if (
            filename in _NEEDS_SIMULATED_SOLVER
            and "surface_evolver" not in registered_solver_ids()
        ):
            self.skipTest("surface_evolver is not registered in this environment")

    def test_every_prompt_yaml_is_covered(self) -> None:
        """A YAML added for SQA without a Python API twin is untested."""
        on_disk = sorted(path.name for path in _PROMPTS_DIR.glob("*.yaml"))
        self.assertEqual(on_disk, sorted(name for name, _ in CASES))

    def test_prompt_yaml_matches_python_api_build(self) -> None:
        for filename, build_python_job in CASES:
            with self.subTest(prompt=filename):
                self._skip_without_solver(filename)
                config, yaml_job = load_job(_PROMPTS_DIR / filename)
                output_dir = Path(config.output_dir)
                self.assertEqual(
                    _intent(yaml_job.compile_orders(output_dir=output_dir)),
                    _intent(build_python_job().compile_orders(output_dir=output_dir)),
                )

    def test_defect_rows_carry_operation_labels_and_crop(self) -> None:
        """The Row 4/5 contract: defect records are per-component crops."""
        for filename, build_python_job in CASES:
            with self.subTest(prompt=filename):
                self._skip_without_solver(filename)
                plan = build_python_job().compile_orders(output_dir=Path("out"))
                order_types = {order.order_type for order in plan.orders}
                self.assertEqual(
                    "postprocess.crop" in order_types, filename in _CROPPED
                )
                self.assertEqual(
                    "annotations.operations.configure" in order_types,
                    filename.startswith(("row4_", "row5_")),
                )

    def test_emitted_plan_round_trips_through_codec(self) -> None:
        """The hand-off is ``JobOrderCodec.dumps(plan)``; Airflow loads it back.

        The codec is the representation the executor's own process boundary
        uses. Dumping, parsing as JSON, loading and dumping again must give
        back the same plan and the same bytes.
        """
        for filename, build_python_job in CASES:
            with self.subTest(prompt=filename):
                self._skip_without_solver(filename)
                plan = build_python_job().compile_orders(output_dir=Path("out"))
                payload = JobOrderCodec.dumps(plan)
                self.assertIsInstance(json.loads(payload), dict)
                restored = JobOrderCodec.loads(payload)
                self.assertEqual(restored, plan)
                self.assertEqual(JobOrderCodec.dumps(restored), payload)


if __name__ == "__main__":
    unittest.main()
