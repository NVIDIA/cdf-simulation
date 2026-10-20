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

"""Tests for the declarative CDF API.

CI-safe: stdlib ``unittest`` only; no omni/USD import, and PyYAML-dependent
tests skip themselves when PyYAML is absent (the CI unit-test job installs
numpy only).
"""

import contextlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: F401 -- registers the "pcb-aoi" domain
from cdf import CDFError, CDFJob, RenderResult, ValidationError
from pcb_aoi.examples.allegro_defect_job import board_dir, build_job

try:
    import yaml
except ImportError:  # CI unit-test image installs numpy only
    yaml = None

#: Component names the fixture sidecar lists, sorted as compile() emits them.
_SIDECAR_TYPES = ["CAP_SMD_0603", "PASSIVE_SMD_0805", "SOIC8"]
_SIDECAR_JSON = (
    '{"schema_version": "v1.1", "board_prim_path": "/Board", "components": ['
    '{"name": "SOIC8"}, {"name": "CAP_SMD_0603"}, {"name": "PASSIVE_SMD_0805"}, '
    '{"name": "CAP_SMD_0603"}], "component_instances": []}'
)
_FIXTURE_DIR = tempfile.TemporaryDirectory(prefix="cdf_sidecar_")


def tearDownModule():
    _FIXTURE_DIR.cleanup()


def write_sidecar(directory):
    """Write the fixture pcb.components sidecar into ``directory``; return its path."""
    sidecar = Path(directory) / "CAD_BOARD.components.json"
    sidecar.write_text(_SIDECAR_JSON, encoding="utf-8")
    return str(sidecar)


def build_job_with_sidecar():
    """The example job with its pcb.components sidecar readable on disk.

    The example points at ``_boards/``, which is not in the repository.
    ``compile()`` refuses to emit an unresolved ``component_types`` keyword,
    so a test that compiles the example needs the sidecar it selects from.
    """
    job = build_job()
    job.scene.sidecars["components"]["path"] = write_sidecar(_FIXTURE_DIR.name)
    return job


class JobModelTest(unittest.TestCase):
    """Verify the declarative model exposes six public areas."""

    def test_six_public_areas(self):
        job = CDFJob(domain="pcb-aoi")
        for area in (
            "scene",
            "appearance",
            "operations",
            "annotations",
            "capture",
            "postprocess",
        ):
            self.assertTrue(hasattr(job, area), f"missing area: {area}")

    def test_domain_supplies_its_operations_section(self):
        job = CDFJob(domain="pcb-aoi")

        self.assertEqual(
            type(job.operations).__module__,
            "pcb_aoi.sections",
        )
        self.assertTrue(hasattr(job.operations, "add_tombstone"))
        self.assertTrue(hasattr(job.operations, "solder_joints"))
        self.assertTrue(hasattr(job.operations, "get_solder_joint_solvers"))

    def test_unknown_domain_names_registered_and_hints_import(self):
        with self.assertRaises(CDFError) as ctx:
            CDFJob(domain="warehouse")
        self.assertIn("pcb-aoi", str(ctx.exception))
        self.assertIn("import", str(ctx.exception))

    def test_description_calls_do_not_touch_stage_or_disk(self):
        # Asserted in a FRESH interpreter, not this one. `omni`/`pxr` in
        # sys.modules is a process-wide fact: any earlier test in the run
        # that legitimately imports usd-core (tht/, board_metadata/,
        # geometry/, editor/) makes this assertion about THAT import
        # instead of about compile(). It therefore passed only on the
        # numpy-only CI image and failed on every machine with usd-core --
        # exactly backwards, since a dev box is where the regression would
        # be introduced.
        sidecar = write_sidecar(_FIXTURE_DIR.name)
        probe = "\n".join(
            (
                f"import sys; sys.path.insert(0, {str(_REPO_ROOT)!r})",
                "import pcb_aoi",
                "from pcb_aoi.examples.allegro_defect_job import build_job",
                "job = build_job()",
                f"job.scene.sidecars['components']['path'] = {sidecar!r}",
                "job.compile()",
                "leaked = [m for m in ('omni', 'pxr') if m in sys.modules]",
                "print('LEAKED:' + ','.join(leaked))",
            )
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertIn("LEAKED:", completed.stdout)
        leaked = completed.stdout.split("LEAKED:")[-1].strip()
        self.assertEqual(leaked, "", f"describing a job imported {leaked}")


class CompileEquivalenceTest(unittest.TestCase):
    """Verify Python descriptions and YAML configurations compile equally."""

    maxDiff = None

    def test_defect_flow_compiles_to_pipeline_schema(self):
        job = build_job_with_sidecar()
        plan = job.compile()
        self.assertEqual(plan.pipeline_type, "defect")
        flow = plan.flow
        self.assertEqual(flow["pipeline_type"], "defect")
        self.assertEqual(flow["lighting"]["mode"], "rgb_light")
        self.assertFalse(flow["lighting"]["preserve_scene_light_color"])
        self.assertFalse(flow["vantablack_components"])
        self.assertEqual(
            flow["components_json"],
            job.scene.sidecars["components"]["path"],
        )
        self.assertEqual(
            {
                k: flow[k]
                for k in ("seed", "random_seed", "max_image_count", "num_triggers")
            },
            {"seed": 0, "random_seed": 0, "max_image_count": -1, "num_triggers": 1},
        )
        self.assertEqual(flow["resolution"], [1920, 1080])
        self.assertEqual(
            flow["pathtracing"], {"spp": 1, "total_spp": 32, "max_bounces": 4}
        )
        self.assertEqual(flow["scan_grid"], {"x_num": 10, "y_num": 10})
        self.assertEqual(
            flow["defects"],
            {
                "shift": {
                    "enabled": True,
                    "ratio": 0.2,
                    "translate_range": 0.2,
                    "rotate_z_range": 15,
                },
                "tombstone": {
                    "enabled": True,
                    "ratio": 0.2,
                    "angle_min": 70,
                    "angle_max": 90,
                    "component_types": None,
                    "auto_filter": True,
                },
                "sideflip": {
                    "enabled": True,
                    "ratio": 0.2,
                    "angle_min": 70,
                    "angle_max": 90,
                },
                "reverse_polarity": {"enabled": False},
            },
        )
        # required-schema default: pipeline indexes augmentation.motion_blur
        self.assertIn("motion_blur", flow["augmentation"])
        self.assertEqual(
            flow["camera_rotation"],
            {"x_range": [0, 0], "y_range": [0, 0], "z_fixed": -90},
        )
        self.assertEqual(flow["camera_side"], "top")
        self.assertEqual(
            flow["augmentation"]["motion_blur"],
            {"probability": 0.8, "alpha_range": [0.01, 0.3], "kernel_choices": [5, 7]},
        )
        self.assertEqual(
            flow["lighting"]["layers"]["Inner_Red"]["intensity"], [5000, 8000]
        )
        self.assertEqual(flow["writer"]["semantic_types"], ["defect"])
        self.assertEqual(flow["writer"]["frame_padding"], 4)
        self.assertTrue(flow["writer"]["bounding_box_2d_tight"])

    def test_grid_index_rename_compiles_as_pipeline_option(self):
        job = build_job_with_sidecar()
        job.capture.outputs(rename_to_grid_index=True)

        flow = job.compile().flow

        self.assertTrue(flow["rename_to_grid_index"])
        self.assertNotIn("rename_to_grid_index", flow["writer"])

    def test_pcba_config_keeps_scene_bound_keys(self):
        job = build_job_with_sidecar()
        plan = job.compile()
        self.assertEqual(
            plan.pcba["scene"], str(board_dir / "CAD_BOARD_aoi.usd")
        )
        self.assertEqual(plan.pcba["pcba_root"], "/Board_CAD_BOARD")
        self.assertEqual(plan.pcba["camera_path"], "/World/camera_light/Camera")
        self.assertEqual(
            plan.pcba["ring_light_root"], "/World/camera_light/aoi_ring_light"
        )
        # Concrete names from the sidecar -- never a keyword, which the
        # generated target has no registry to resolve.
        self.assertEqual(plan.pcba["component_types"], _SIDECAR_TYPES)
        # strict --config / --pcba-config key boundary
        self.assertFalse(set(plan.pcba) & set(plan.flow))

    def test_compile_refuses_a_selection_it_cannot_make_concrete(self):
        # `ALL` used to be emitted here even though a generated target has no
        # sibling registry. Refuse at compile time, and say what to do.
        def never_selected(job):
            job.scene.selection = None

        def no_types_no_source(job):
            job.scene.select_components()

        def sidecar_not_on_disk(job):
            pass  # the example's own sidecar path, absent from the repository

        def embedded_usd_source(job):
            job.scene.select_components(source="usd")

        for arrange in (
            never_selected,
            no_types_no_source,
            sidecar_not_on_disk,
            embedded_usd_source,
        ):
            with self.subTest(case=arrange.__name__):
                job = build_job()
                job.operations.ops.pop("tombstone", None)  # its own sidecar check
                arrange(job)
                with self.assertRaises(ValidationError) as ctx:
                    job.compile()
                message = str(ctx.exception)
                self.assertIn("select_components", message)
                self.assertIn("types=[...]", message)

    def test_the_unreadable_sidecar_is_named(self):
        job = build_job()
        job.operations.ops.pop("tombstone", None)
        with self.assertRaises(ValidationError) as ctx:
            job.compile()
        self.assertIn(
            str(board_dir / "CAD_BOARD.components.json"),
            str(ctx.exception),
        )

    def test_pcba_component_types_expand_from_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "components.json"
            sidecar.write_text(
                '{"schema_version": "v1.1", '
                '"board_prim_path": "/Board", "components": ['
                '{"name": "PASSIVE_SMD_0805"}, '
                '{"name": "CAP_SMD_0603"}, '
                '{"name": "PASSIVE_SMD_0805"}], '
                '"component_instances": []}',
                encoding="utf-8",
            )
            job = build_job()
            job.scene.sidecars["components"]["path"] = str(sidecar)

            self.assertEqual(
                job.compile().pcba["component_types"],
                ["CAP_SMD_0603", "PASSIVE_SMD_0805"],
            )

    def test_pcba_component_types_expand_from_remote_sidecar(self):
        job = build_job()
        remote_path = "s3://bucket/boards/components.json"
        job.scene.sidecars["components"]["path"] = remote_path

        with (
            mock.patch("cdf.plan.file_utils.is_file", return_value=True),
            mock.patch(
                "cdf.plan.file_utils.open",
                mock.mock_open(
                    read_data=(
                        '{"schema_version": "v1.1", '
                        '"board_prim_path": "/Board", "components": '
                        '[{"name": "RESISTOR"}, {"name": "CAPACITOR"}], '
                        '"component_instances": []}'
                    ),
                ),
            ) as open_file,
        ):
            component_types = job.compile().pcba["component_types"]

        self.assertEqual(component_types, ["CAPACITOR", "RESISTOR"])
        open_file.assert_called_once_with(remote_path, encoding="utf-8")

    def test_pipeline_type_inference(self):
        job = build_job_with_sidecar()
        job.operations.add_missing(ratio=0.3)
        self.assertEqual(job.compile().pipeline_type, "missing")

        good = CDFJob(domain="pcb-aoi")
        good.scene.configure(usd_path="/tmp/x.usd", root_path="/Board")
        good.scene.select_components(types=["PASSIVE_SMD_0805"])
        good.capture.renderer(
            mode="pathtracing", resolution=(64, 64), spp=1, total_spp=1, max_bounces=1
        )
        good.capture.scan_grid(x_count=1, y_count=1)
        good.capture.camera(rotation_range={"x": (0, 0), "y": (0, 0), "z": 0})
        good.capture.outputs(rgb=True)
        plan = good.compile()
        self.assertEqual(plan.pipeline_type, "good")
        self.assertEqual(plan.flow["writer"]["semantic_types"], ["class"])

class ValidationTest(unittest.TestCase):
    def test_rejections(self):
        job = CDFJob(domain="pcb-aoi")
        cases = [
            (
                "ratio out of range",
                lambda: job.operations.add_shift(
                    ratio=1.5, translate_range=0.1, rotate_z_range=5
                ),
            ),
            (
                "reversed angle range",
                lambda: job.operations.add_tombstone(ratio=0.1, angle_range=(90, 70)),
            ),
            (
                "unsupported renderer",
                lambda: job.capture.renderer(
                    mode="raster",
                    resolution=(64, 64),
                    spp=1,
                    total_spp=1,
                    max_bounces=1,
                ),
            ),
            (
                "unsupported crop method",
                lambda: job.postprocess.crop(method="label_cc"),
            ),
            (
                "rgb-only crop types (no semantic => zero crops)",
                lambda: job.postprocess.crop(types=["rgb"]),
            ),
            (
                "unknown sidecar kind",
                lambda: job.scene.attach_sidecar(
                    name="x", kind="warehouse.racks", path="/tmp/x.json"
                ),
            ),
            (
                "relative root path",
                lambda: job.scene.configure(usd_path="/tmp/x.usd", root_path="Board"),
            ),
            (
                "instances before scene configuration",
                lambda: job.scene.instances(prim_paths=("/World/Panel/Board_0",)),
            ),
            (
                "unsupported camera side",
                lambda: job.capture.camera(
                    side="back",
                    rotation_range={"x": (0, 0), "y": (0, 0), "z": 0},
                ),
            ),
        ]
        for label, call in cases:
            with self.subTest(label), self.assertRaises(ValidationError):
                call()

    def test_scene_instances_reject_invalid_or_duplicate_paths(self):
        for prim_paths in (
            (),
            ("World/Panel/Board_0",),
            ("/World/Panel/Board_0", "/World/Panel/Board_0"),
            ("/World/Panel/Board_0", 1),
        ):
            with self.subTest(prim_paths=prim_paths):
                job = CDFJob(domain="pcb-aoi")
                job.scene.configure(usd_path="panel.usda", root_path="/World/Panel")
                with self.assertRaises(ValidationError):
                    job.scene.instances(prim_paths=prim_paths)

    def test_scene_instances_reject_component_sidecars_in_either_order(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="panel.usda", root_path="/World/Panel")
        job.scene.attach_sidecar(
            name="components",
            kind="pcb.components",
            path="components.json",
        )
        with self.assertRaisesRegex(ValidationError, "embedded USD"):
            job.scene.instances(prim_paths=("/World/Panel/Board_0",))

        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="panel.usda", root_path="/World/Panel")
        job.scene.instances(prim_paths=("/World/Panel/Board_0",))
        with self.assertRaisesRegex(ValidationError, "embedded USD"):
            job.scene.attach_sidecar(
                name="components",
                kind="pcb.components",
                path="components.json",
            )

    def test_duplicate_operation_rejected(self):
        job = CDFJob(domain="pcb-aoi")
        job.operations.add_sideflip(ratio=0.1, angle_range=(70, 90))
        with self.assertRaises(ValidationError):
            job.operations.add_sideflip(ratio=0.2, angle_range=(70, 90))

    def test_exact_count_targeting_is_recorded_and_mutually_exclusive(self):
        job = CDFJob(domain="pcb-aoi")
        job.operations.add_shift(
            count=2,
            translate_range=0.1,
            rotate_z_range=5,
        )

        self.assertEqual(job.operations.ops["shift"]["count"], 2)
        self.assertNotIn("ratio", job.operations.ops["shift"])
        other = CDFJob(domain="pcb-aoi")
        with self.assertRaisesRegex(ValidationError, "exactly one"):
            other.operations.add_sideflip(
                ratio=0.5,
                count=2,
                angle_range=(60, 80),
            )

    def test_compile_reports_all_missing_requirements(self):
        with self.assertRaises(ValidationError) as ctx:
            CDFJob(domain="pcb-aoi").compile()
        message = str(ctx.exception)
        for expected in (
            "scene.configure",
            "capture.renderer",
            "capture.scan_grid",
            "capture.outputs",
            "capture.camera",
        ):
            self.assertIn(expected, message)

    def test_tombstone_eligibility_requires_components_sidecar(self):
        job = build_job()
        job.scene.sidecars.clear()
        with self.assertRaises(ValidationError) as ctx:
            job.compile()
        self.assertIn("eligibility_source", str(ctx.exception))


class ExecutionTest(unittest.TestCase):
    def test_postprocess_crop_builds_pipeline_cfg(self):
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            job = build_job_with_sidecar()
            render_result = RenderResult(
                output_dir=tmp,
                pipeline_type=job.compile().pipeline_type,
            )
            crop_result = job.postprocess.crop(
                source=render_result,
                method="bbox_tight",
                output_subdir="cropped",
                types=["rgb", "semantic_segmentation", "component_instance"],
                offset=10,
                min_coverage=0.2,
                xform_depth=7,
                runner=captured.update,
            )
            self.assertTrue(crop_result.executed)
            self.assertEqual(captured["pipeline_type"], "defect")
            self.assertEqual(captured["output"], tmp)
            self.assertEqual(captured["crop"]["method"], "bbox_tight")
            self.assertTrue(captured["crop"]["enabled"])
            self.assertEqual(crop_result.output_dir, str(Path(tmp) / "cropped"))

    def test_postprocess_crop_stages_and_republishes_s3_output(self):
        captured = {}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            localized = root / "localized"
            staged = root / "staged"
            localized.mkdir()
            staged.mkdir()
            (localized / "trigger_0000").mkdir()
            (localized / "trigger_0000" / "rgb.png").write_bytes(b"rgb")

            render_result = RenderResult(
                output_dir="s3://bucket/runs/job-1",
                pipeline_type=build_job_with_sidecar().compile().pipeline_type,
            )

            def run_crop(cfg):
                captured.update(cfg)
                crop_dir = Path(cfg["output"]) / cfg["crop"]["output_subdir"]
                crop_dir.mkdir()
                (crop_dir / "crop.png").write_bytes(b"crop")

            with (
                mock.patch(
                    "cdf.plan.file_utils.localize_tree",
                    return_value=contextlib.nullcontext(str(localized)),
                ),
                mock.patch(
                    "cdf.plan.file_utils.output_tree",
                    return_value=contextlib.nullcontext(str(staged)),
                ),
            ):
                crop_result = build_job().postprocess.crop(
                    source=render_result,
                    method="bbox_tight",
                    output_subdir="cropped",
                    types=["rgb", "semantic_segmentation"],
                    runner=run_crop,
                )

            self.assertEqual(captured["output"], str(staged))
            self.assertTrue((staged / "trigger_0000" / "rgb.png").is_file())
            self.assertTrue((staged / "cropped" / "crop.png").is_file())
            self.assertEqual(
                crop_result.output_dir,
                "s3://bucket/runs/job-1/cropped",
            )
            self.assertEqual(crop_result.config["output"], "s3://bucket/runs/job-1")

    def test_crop_rejects_non_render_source(self):
        job = build_job()
        with self.assertRaises(ValidationError):
            job.postprocess.crop(source="/some/dir")


if __name__ == "__main__":
    unittest.main()
