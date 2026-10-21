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

"""Kit-free tests for lowering CDF declarations into JobOrders."""

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi CDF domain
from cdf import (
    CDF_API_VERSION,
    ComponentFilter,
    JobOrderCodec,
    CDFError,
    CDFJob,
    ValidationError,
)
from cdf.execution.compiler_adapters import compiler_adapter


def _configure_capture(job: CDFJob) -> None:
    """Add the declarations required by the initial capture compiler."""
    job.capture.output_start(0)
    job.capture.renderer(
        mode="pathtracing",
        resolution=(1920, 1080),
        spp=1,
        total_spp=32,
        max_bounces=4,
    )
    job.capture.camera(
        rotation_range={"x": (0, 0), "y": (0, 0), "z": -90},
    )
    job.capture.scan_grid(x_count=1, y_count=1)
    job.capture.limits(triggers=1, max_images=1)
    job.capture.outputs(
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        colorize_semantic_segmentation=True,
        semantic_types=["defect"],
    )


def _build_tombstone_job(
    *,
    usd_path: str = "board.usd",
    archive_member: str | None = None,
    root_path: str = "/Board",
) -> CDFJob:
    """Build the representative two-tombstone compiler vertical slice."""
    job = CDFJob(domain="pcb-aoi", random_seed=7)
    job.scene.configure(
        usd_path=usd_path,
        archive_member=archive_member,
        root_path=root_path,
        camera_path="/World/camera_light/Camera",
        camera_xform_path="/World/camera_light",
        ring_light_root="/World/camera_light/aoi_ring_light",
    )
    job.scene.attach_sidecar(
        name="components",
        kind="pcb.components",
        path="components.json",
    )
    job.scene.select_components(types=["PASSIVE_SMD_0805"])
    job.appearance.lighting(
        mode="rgb_light",
        exposure_range=(0.25, 0.25),
        layers={"Inner_Red": {"intensity": (6500, 6500)}},
    )
    job.appearance.components(vantablack=False)
    job.annotations.label_components(source="components")
    job.annotations.enable_operation_labels(namespace="defect")
    job.operations.add_tombstone(
        ratio=0.25,
        angle_range=(70, 90),
        component_types=["PASSIVE_SMD_0805"],
        auto_filter=True,
        eligibility_source="components",
    )
    _configure_capture(job)
    job.postprocess.crop(
        method="bbox_tight",
        types=["rgb", "semantic_segmentation", "component_instance"],
        offset=32,
        min_coverage=0.2,
        class_filter=["tombstone"],
        xform_depth=7,
        output_subdir="tombstone_closeups",
    )
    return job


class JobOrderCompilerTest(unittest.TestCase):
    """Verify declaration recording and the initial compiler vertical slice."""

    def test_compiler_requires_a_registered_domain_adapter(self):
        """Never compile domain declarations through silent no-op hooks."""
        with self.assertRaisesRegex(
            CDFError,
            "domain 'missing-adapter' has no registered compiler adapter",
        ):
            compiler_adapter("missing-adapter")

    def test_pin_variation_lands_between_the_scene_and_the_render(self):
        """The order has to run after the board loads and before it is captured."""
        # the compiler needs no special knowledge of pin_variation, but the
        # ordering it produces by default is load-bearing: varying leads after
        # the render would capture the nominal board and report success
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.operations.pin_variation(
            package_types=("TO92",),
            tilt_range=(0, 15),
        ).pin_variation(
            components=("C1001",),
            cut_angle=(0, 60),
        )
        _configure_capture(job)
        plan = job.compile_orders()

        by_id = {order.order_id: order for order in plan.orders}
        self.assertIn("operation.pin_variation", by_id)
        variation = by_id["operation.pin_variation"]
        self.assertEqual(variation.order_type, "operation.pin_variation.apply")
        self.assertIn("scene.load", variation.depends_on)
        # The two calls vary different axes, so each reaches the order its
        # axis belongs to: TO92 leans, C1001 is trimmed.
        self.assertEqual(len(variation.parameters["groups"]), 1)
        self.assertEqual(len(by_id["operation.pin_cut"].parameters["groups"]), 1)

        render = by_id["capture.render"]
        self.assertIn("operation.pin_variation", render.depends_on)
        self.assertIn("operation.pin_cut", render.depends_on)

        order_ids = [order.order_id for order in plan.orders]
        self.assertLess(
            order_ids.index("operation.pin_variation"),
            order_ids.index("capture.render"),
        )
        self.assertLess(
            order_ids.index("operation.pin_cut"),
            order_ids.index("capture.render"),
        )

    def test_a_trim_lands_after_the_joint_it_has_to_cut_through(self):
        """The whole reason the operation is split in two.

        Leaning a lead has to precede the solder, because the joint is sized
        from where the lead ended up. Trimming has to follow it, because the
        blade takes lead and joint together and leaves one section across both.
        A caller declares one `pin_variation(...)`; the compiler is what knows
        the two halves belong on opposite sides of the solder.
        """
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.operations.pin_variation(
            package_types=("TO92",),
            tilt_range=(0, 15),
            cut_angle=(0, 60),
        )
        job.operations.solder_joints(solver="procedural")
        _configure_capture(job)
        order_ids = [order.order_id for order in job.compile_orders().orders]

        self.assertLess(
            order_ids.index("operation.pin_variation"),
            order_ids.index("solder_joints.collect"),
        )
        self.assertLess(
            order_ids.index("solder_joints.attach"),
            order_ids.index("operation.pin_cut"),
        )

    def test_pin_tip_lands_after_final_pin_and_solder_geometry(self):
        """The proxy must copy the final cut face rather than the source mesh."""
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.operations.pin_variation(
            components=("X205",),
            cut_angle=(10, 20),
        )
        job.operations.solder_joints(solver="procedural")
        job.operations.pin_tip(side="bottom", components=("X205",))
        _configure_capture(job)

        plan = job.compile_orders()
        by_id = {order.order_id: order for order in plan.orders}
        order_ids = [order.order_id for order in plan.orders]

        self.assertLess(
            order_ids.index("solder_joints.attach"),
            order_ids.index("operation.pin_tip"),
        )
        self.assertLess(
            order_ids.index("operation.pin_cut"),
            order_ids.index("operation.pin_tip"),
        )
        self.assertIn(
            "operation.pin_cut",
            by_id["operation.pin_tip"].depends_on,
        )

    def test_pin_variation_survives_a_codec_round_trip(self):
        """Every field must come back from encode-then-decode unchanged."""
        # groups are nested mappings; a transport that flattened or dropped
        # them would only be caught at execution
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.operations.pin_variation(
            components=("C1001", "C1002"),
            tilt_range=(0, 15),
            height_variation=(0.0, 1.5),
        )
        _configure_capture(job)
        plan = job.compile_orders()
        restored = JobOrderCodec.loads(JobOrderCodec.dumps(plan))
        original = next(
            o for o in plan.orders if o.order_id == "operation.pin_variation"
        )
        round_tripped = next(
            o for o in restored.orders if o.order_id == "operation.pin_variation"
        )
        self.assertEqual(round_tripped.parameters, original.parameters)

    def test_scene_instances_extend_scene_load_without_reordering_it(self):
        job = CDFJob(domain="pcb-aoi")
        instance_paths = (
            "/World/Panel/StepRepeats/SR_0_0",
            "/World/Panel/StepRepeats/SR_0_1",
        )
        job.scene.configure(usd_path="panel.usda", root_path="/World/Panel")
        job.scene.instances(prim_paths=instance_paths)
        job.scene.select_components(source="usd")

        plan = job.compile_orders(serialization_path="panel.usda")

        self.assertEqual(plan.orders[0].order_id, "scene.load")
        self.assertEqual(plan.orders[0].parameters["instance_paths"], instance_paths)
        self.assertEqual(plan.orders[1].order_id, "scene.components.select")

    def test_stitch_depends_only_on_its_named_capture_pass(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="panel.usda", root_path="/World/Panel")
        for name in ("full_board", "component_rois"):
            with job.capture.render_pass(name) as render_pass:
                render_pass.renderer(
                    mode="pathtracing",
                    resolution=(2000, 2000),
                    spp=1,
                    total_spp=1,
                    max_bounces=1,
                )
                render_pass.camera(side="bottom", rotation_range={"z": -90})
                render_pass.scan_grid(x_count=1, y_count=1)
                render_pass.outputs(rgb=True)
        job.postprocess.stitch(source_pass="full_board")

        plan = job.compile_orders(output_dir="output")
        stitch = next(
            order for order in plan.orders if order.order_id == "postprocess.stitch"
        )

        self.assertEqual(stitch.depends_on, ("capture.full_board.render",))

    def test_crop_depends_on_the_named_capture_pass_it_reads(self):
        # The runtime crop handler resolves its source from this dependency
        # shape: one named pass and no implicit default pass.
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usda", root_path="/Board")
        with job.capture.render_pass("component_focus") as render_pass:
            render_pass.renderer(
                mode="pathtracing",
                resolution=(512, 512),
                spp=1,
                total_spp=1,
                max_bounces=1,
            )
            render_pass.camera(side="top", rotation_range={"z": -90})
            render_pass.scan_grid(x_count=1, y_count=1)
            render_pass.outputs(rgb=True)
        job.postprocess.crop(
            types=["rgb", "semantic_segmentation", "component_instance"],
            offset=10,
        )

        plan = job.compile_orders(output_dir="output")
        order_ids = [order.order_id for order in plan.orders]
        crop = next(
            order for order in plan.orders if order.order_id == "postprocess.crop"
        )

        self.assertNotIn("capture.render", order_ids)
        self.assertEqual(crop.depends_on, ("capture.component_focus.render",))

    def test_stitch_rejects_unknown_capture_pass(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="panel.usda", root_path="/World/Panel")
        with job.capture.render_pass("full_board") as render_pass:
            render_pass.renderer(
                mode="pathtracing",
                resolution=(64, 64),
                spp=1,
                total_spp=1,
                max_bounces=1,
            )
            render_pass.camera(side="bottom", rotation_range={"z": -90})
            render_pass.scan_grid(x_count=1, y_count=1)
            render_pass.outputs(rgb=True)
        job.postprocess.stitch(source_pass="missing")

        with self.assertRaisesRegex(ValidationError, "unknown capture pass"):
            job.compile_orders(output_dir="output")

    def test_tombstone_plan_preserves_calls_and_places_crop_after_capture(self):
        job = _build_tombstone_job()

        plan = job.compile_orders(output_dir=Path("output"))

        self.assertEqual(
            [order.order_id for order in plan.orders],
            [
                "scene.load",
                "scene.sidecar.components",
                "scene.components.select",
                "appearance.lighting",
                "appearance.components",
                "annotations.components",
                "annotations.operations",
                "operation.tombstone",
                "capture.output",
                "capture.renderer",
                "capture.camera",
                "capture.scan_grid",
                "capture.limits",
                "capture.outputs",
                "appearance.lighting.default.apply",
                "capture.render",
                "postprocess.crop",
            ],
        )

        self.assertEqual(plan.ordered(), plan.orders)
        self.assertEqual(
            plan.metadata,
            {
                "api_version": CDF_API_VERSION,
                "domain": "pcb-aoi",
                "random_seed": 7,
                # The run root travels on the plan so provenance can refuse to
                # enumerate artifacts outside it.
                "output_root": "output",
            },
        )

        by_id = {order.order_id: order for order in plan.orders}
        tombstone = by_id["operation.tombstone"]
        self.assertEqual(tombstone.parameters["angle_range"], (70, 90))
        self.assertNotIn("angle_min", tombstone.parameters)
        self.assertEqual(
            tombstone.depends_on,
            (
                "scene.load",
                "scene.sidecar.components",
                "scene.components.select",
            ),
        )
        self.assertEqual(
            by_id["annotations.components"].depends_on,
            (
                "scene.load",
                "scene.sidecar.components",
                "scene.components.select",
            ),
        )
        self.assertEqual(
            by_id["capture.camera"].parameters["rotation_range"]["z"],
            -90,
        )
        self.assertEqual(by_id["capture.camera"].parameters["side"], "top")
        self.assertEqual(
            by_id["capture.scan_grid"].parameters,
            {
                "x_count": 1,
                "y_count": 1,
                "scan_area_mm": None,
                "centered_on": None,
                "scan_center_mm": None,
            },
        )
        self.assertEqual(
            by_id["capture.render"].parameters["output_dir"],
            "output",
        )
        self.assertEqual(
            by_id["postprocess.crop"].depends_on,
            ("capture.render",),
        )

    def test_exact_count_targeting_compiles_to_job_order(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.scene.select_components(types=["PASSIVE_SMD_0805"])
        job.operations.add_shift(
            count=2,
            translate_range=0.2,
            rotate_z_range=15,
        )
        _configure_capture(job)

        order = next(
            order
            for order in job.compile_orders().orders
            if order.order_id == "operation.shift"
        )
        self.assertEqual(order.parameters["count"], 2)
        self.assertNotIn("ratio", order.parameters)

    def test_component_roi_capture_compiles_to_job_order(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.scene.select_components(source="usd")
        job.capture.roi_selection(refdes=("J1", "J2"), padding=0.25)
        job.capture.roi_selection(
            pins=(("J1", "1"), ("J1", "2")),
            padding=0.1,
        )
        job.capture.roi_selection(
            component_types=("CAPACITOR",),
            count=2,
            padding=0.5,
        )
        job.capture.roi_selection(
            process_classes=("through_hole",),
            focus="pin",
        )
        job.capture.roi_selection(all_components=True, focus="pin")
        job.capture.roi_selection(defects=("shift", "tombstone"), padding=0.75)
        _configure_capture(job)

        plan = job.compile_orders()
        order = next(
            order for order in plan.orders if order.order_id == "capture.roi_selection"
        )

        self.assertEqual(
            order.parameters["selections"],
            (
                {
                    "refdes": ("J1", "J2"),
                    "count": None,
                    "padding": 0.25,
                },
                {
                    "pins": (("J1", "1"), ("J1", "2")),
                    "count": None,
                    "padding": 0.1,
                },
                {
                    "component_types": ("CAPACITOR",),
                    "count": 2,
                    "padding": 0.5,
                },
                {
                    "process_classes": ("through_hole",),
                    "count": None,
                    "padding": 0.5,
                    "focus": "pin",
                },
                {
                    "all_components": True,
                    "count": None,
                    "padding": 0.5,
                    "focus": "pin",
                },
                {
                    "defects": ("shift", "tombstone"),
                    "count": None,
                    "padding": 0.75,
                },
            ),
        )
        self.assertLess(
            [item.order_id for item in plan.orders].index(order.order_id),
            [item.order_id for item in plan.orders].index("capture.render"),
        )

    def test_roi_selection_rejects_ambiguous_or_invalid_selectors(self):
        job = CDFJob(domain="pcb-aoi")

        with self.assertRaisesRegex(ValidationError, "exactly one"):
            job.capture.roi_selection()
        with self.assertRaisesRegex(ValidationError, "exactly one"):
            job.capture.roi_selection(
                refdes=("J1",),
                pins=(("J1", "1"),),
                component_types=("CAPACITOR",),
            )
        with self.assertRaisesRegex(ValidationError, "not valid with explicit refdes"):
            job.capture.roi_selection(refdes=("J1",), count=1)
        with self.assertRaisesRegex(ValidationError, "unsupported defects"):
            job.capture.roi_selection(defects=("bridge",))
        with self.assertRaisesRegex(ValidationError, "pairs"):
            job.capture.roi_selection(pins=(("J1",),))
        with self.assertRaisesRegex(ValidationError, "explicit pins"):
            job.capture.roi_selection(pins=(("J1", "1"),), count=1)
        with self.assertRaisesRegex(ValidationError, "exact pins"):
            job.capture.roi_selection(
                pins=(("J1", "1"),),
                focus="component",
            )
        with self.assertRaisesRegex(ValidationError, "focus"):
            job.capture.roi_selection(refdes=("J1",), focus="pad")
        with self.assertRaisesRegex(ValidationError, "all_components"):
            job.capture.roi_selection(all_components=True, count=1)

    def test_named_render_passes_compile_with_scoped_outputs(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")

        with job.capture.render_pass("full_board") as full_board:
            full_board.renderer(
                mode="pathtracing",
                resolution=(1920, 1080),
                spp=1,
                total_spp=32,
                max_bounces=4,
            )
            full_board.camera(rotation_range={})
            full_board.scan_grid(x_count=1, y_count=1)
            full_board.outputs(rgb=True)

        with job.capture.render_pass("component_rois") as component_rois:
            component_rois.renderer(
                mode="pathtracing",
                resolution=(1920, 1080),
                spp=1,
                total_spp=32,
                max_bounces=4,
            )
            component_rois.camera(side="bottom", rotation_range={"z": -90})
            component_rois.scan_grid(x_count=1, y_count=1)
            component_rois.roi_selection(refdes=("J1", "J2"))
            component_rois.outputs(rgb=True)

        plan = job.compile_orders(output_dir=Path("output"))
        by_id = {order.order_id: order for order in plan.orders}

        self.assertEqual(
            by_id["capture.full_board.render"].parameters["output_dir"],
            str(Path("output") / "full_board"),
        )
        self.assertEqual(
            by_id["capture.component_rois.render"].parameters["output_dir"],
            str(Path("output") / "component_rois"),
        )
        self.assertLess(
            [order.order_id for order in plan.orders].index(
                "capture.full_board.render"
            ),
            [order.order_id for order in plan.orders].index(
                "capture.component_rois.renderer"
            ),
        )

    def test_render_pass_context_does_not_redirect_default_capture_calls(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")

        with (
            self.assertRaisesRegex(ValidationError, "missing: renderer"),
            job.capture.render_pass("focused") as focused,
        ):
            job.capture.renderer(
                mode="pathtracing",
                resolution=(64, 64),
                spp=1,
                total_spp=1,
                max_bounces=1,
            )
            focused.camera(rotation_range={})
            focused.scan_grid(x_count=1, y_count=1)
            focused.outputs(rgb=True)

    def test_scene_archive_member_compiles_to_both_execution_paths(self):
        job = _build_tombstone_job(
            usd_path="boards.zip",
            archive_member="panel.usda",
            root_path="/World",
        )

        orders = job.compile_orders(output_dir="output")
        scene_order = next(
            order for order in orders.orders if order.order_id == "scene.load"
        )
        self.assertEqual(scene_order.parameters["archive_member"], "panel.usda")
        self.assertEqual(job.compile().pcba["archive_member"], "panel.usda")

    def test_replacement_declaration_moves_to_latest_call_position(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.appearance.lighting(mode="usd_light")
        job.scene.select_components(types=["PASSIVE_SMD_0805"])
        _configure_capture(job)

        job.appearance.lighting(
            mode="rgb_light",
            layers={"Inner_Red": {"intensity": (6500, 6500)}},
        )
        plan = job.compile_orders()

        lighting_orders = [
            order for order in plan.orders if order.order_id == "appearance.lighting"
        ]
        self.assertEqual(len(lighting_orders), 1)
        self.assertEqual(lighting_orders[0].parameters["mode"], "rgb_light")
        self.assertEqual(plan.orders[-3].order_id, "appearance.lighting")
        self.assertEqual(
            plan.orders[-2].order_id,
            "appearance.lighting.default.apply",
        )
        self.assertEqual(plan.orders[-1].order_id, "capture.render")

    def test_missing_or_late_sidecar_is_rejected(self):
        for attach_after_operation in (False, True):
            with self.subTest(attach_after_operation=attach_after_operation):
                job = CDFJob(domain="pcb-aoi")
                job.scene.configure(usd_path="board.usd", root_path="/Board")
                job.operations.add_tombstone(
                    ratio=0.25,
                    angle_range=(70, 90),
                    eligibility_source="components",
                )
                if attach_after_operation:
                    job.scene.attach_sidecar(
                        name="components",
                        kind="pcb.components",
                        path="components.json",
                    )
                _configure_capture(job)

                with self.assertRaisesRegex(
                    ValidationError,
                    "sidecar 'components'.*attached",
                ):
                    job.compile_orders()

    def test_scene_must_precede_stage_declarations(self):
        job = CDFJob(domain="pcb-aoi")
        job.appearance.lighting(mode="usd_light")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        _configure_capture(job)

        with self.assertRaisesRegex(
            ValidationError,
            r"appearance\.lighting\.configure must be declared after",
        ):
            job.compile_orders()

    def test_component_selection_must_precede_operations(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.operations.add_tombstone(
            ratio=0.25,
            angle_range=(70, 90),
        )
        job.scene.select_components(types=["PASSIVE_SMD_0805"])
        _configure_capture(job)

        with self.assertRaisesRegex(
            ValidationError,
            "component selection to be declared first",
        ):
            job.compile_orders()

    def test_component_consumers_depend_on_a_later_component_selection(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.annotations.label_geometry(holes="hole")
        job.capture.roi_selection(all_components=True, focus="pin")
        job.scene.select_components(types=["PASSIVE_SMD_0805"])
        _configure_capture(job)

        plan = job.compile_orders()
        ordered = [order.order_id for order in plan.ordered()]
        selection = ordered.index("scene.components.select")
        for order_type in (
            "annotations.geometry.apply",
            "capture.roi_selection.configure",
        ):
            order = next(
                order for order in plan.orders if order.order_type == order_type
            )
            self.assertIn("scene.components.select", order.depends_on)
            self.assertLess(selection, ordered.index(order.order_id))

    def test_operations_synthesize_unrestricted_component_selection(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.operations.add_shift(
            ratio=0.25,
            translate_range=0.2,
            rotate_z_range=15,
        )
        _configure_capture(job)

        plan = job.compile_orders()

        selection = next(
            order
            for order in plan.orders
            if order.order_id == "scene.components.select"
        )
        shift = next(
            order for order in plan.orders if order.order_id == "operation.shift"
        )
        self.assertEqual(
            selection.parameters,
            {"source": "usd", "types": None, "component_filter": {}},
        )
        self.assertIn("scene.components.select", shift.depends_on)

    def test_component_filters_lower_at_scene_and_operation_levels(self):
        """Preserve normalized parent and child filters in JobOrders."""
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.scene.select_components(
            component_filter=ComponentFilter(
                side=("Top",),
                process_classes=("SMT",),
            ),
        )
        job.operations.add_shift(
            ratio=0.25,
            translate_range=0.2,
            rotate_z_range=15,
            component_filter={"package_type": ["0603"]},
        )
        _configure_capture(job)

        plan = job.compile_orders()
        by_id = {order.order_id: order for order in plan.orders}

        self.assertEqual(
            dict(by_id["scene.components.select"].parameters["component_filter"]),
            {"side": ("Top",), "process_classes": ("SMT",)},
        )
        self.assertEqual(
            dict(by_id["operation.shift"].parameters["component_filter"]),
            {"package_type": ("0603",)},
        )
        self.assertIn(
            "scene.components.select",
            by_id["operation.shift"].depends_on,
        )

    def test_all_transform_operations_compile_in_python_call_order(self):
        job = CDFJob(domain="pcb-aoi", random_seed=11)
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.scene.select_components(types=["PASSIVE_SMD_0805"])
        job.operations.add_shift(
            ratio=0.1,
            translate_range=0.2,
            rotate_z_range=15,
        )
        job.operations.add_tombstone(
            ratio=0.2,
            angle_range=(70, 90),
        )
        job.operations.add_sideflip(
            ratio=0.3,
            angle_range=(60, 80),
        )
        job.operations.add_reverse_polarity(
            ratio=0.4,
            component_types=["IC_SMD"],
        )
        _configure_capture(job)

        plan = job.compile_orders(output_dir="output")
        operations = [
            order for order in plan.orders if order.order_type.startswith("operation.")
        ]

        self.assertEqual(
            [order.order_type for order in operations],
            [
                "operation.shift.apply",
                "operation.tombstone.apply",
                "operation.sideflip.apply",
                "operation.reverse_polarity.apply",
            ],
        )
        self.assertEqual(operations[0].parameters["translate_range"], 0.2)
        self.assertEqual(operations[2].parameters["angle_range"], (60, 80))
        self.assertEqual(operations[3].parameters["component_types"], ("IC_SMD",))

    def test_missing_compiles_alone_and_rejects_transform_operations(self):
        missing_job = CDFJob(domain="pcb-aoi")
        missing_job.scene.configure(usd_path="board.usd", root_path="/Board")
        missing_job.scene.select_components(types=["PASSIVE_SMD_0805"])
        missing_job.operations.add_missing(ratio=0.25)
        _configure_capture(missing_job)

        plan = missing_job.compile_orders(output_dir="output")

        self.assertIn(
            "operation.missing.apply",
            [order.order_type for order in plan.orders],
        )

        mixed_job = CDFJob(domain="pcb-aoi")
        mixed_job.scene.configure(usd_path="board.usd", root_path="/Board")
        mixed_job.scene.select_components(types=["PASSIVE_SMD_0805"])
        mixed_job.operations.add_missing(ratio=0.25)
        mixed_job.operations.add_shift(
            ratio=0.25,
            translate_range=0.2,
            rotate_z_range=15,
        )
        _configure_capture(mixed_job)

        with self.assertRaisesRegex(
            ValidationError,
            "missing.*cannot be combined",
        ):
            mixed_job.compile_orders()

    def test_compilation_and_transport_do_not_write_yaml(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _build_tombstone_job().compile_orders(root / "output")

            payload = JobOrderCodec.dumps(plan)

            self.assertEqual(JobOrderCodec.loads(payload), plan)
            self.assertEqual(list(root.iterdir()), [])

    def test_solder_collect_waits_for_pin_variation(self):
        """Solder is sized from the leads, so moving them must land first.

        Without the dependency the two orders were only ordered by the accident
        of declaration order, and a job that declared `solder_joints` before
        `pin_variation` would have measured leads that had not moved yet.
        """
        job = CDFJob(domain="pcb-aoi", random_seed=11)
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        # Declared AFTER the solder joints on purpose: the dependency, not the
        # declaration order, is what has to put it first.
        job.operations.solder_joints(solver="procedural")
        job.operations.pin_variation(
            package_types=("TO92",),
            tilt_range=(10.0, 15.0),
        )
        _configure_capture(job)

        plan = job.compile_orders(output_dir="output")
        by_id = {order.order_id: order for order in plan.orders}

        self.assertIn(
            "operation.pin_variation", by_id["solder_joints.collect"].depends_on
        )
        order_ids = [order.order_id for order in plan.orders]
        self.assertLess(
            order_ids.index("operation.pin_variation"),
            order_ids.index("solder_joints.collect"),
        )

    def test_solder_collect_has_no_pin_variation_dependency_when_undeclared(self):
        """A job without the operation must not gain a dangling dependency."""
        job = CDFJob(domain="pcb-aoi", random_seed=11)
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.operations.solder_joints(solver="procedural")
        _configure_capture(job)

        plan = job.compile_orders(output_dir="output")
        by_id = {order.order_id: order for order in plan.orders}

        self.assertEqual(
            by_id["solder_joints.collect"].depends_on,
            ("solder_joints.discover",),
        )

    def test_solder_configuration_lowers_to_internal_pre_capture_orders(self):
        job = CDFJob(domain="pcb-aoi", random_seed=11)
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.scene.attach_sidecar(
            name="components",
            kind="pcb.components",
            path="components.json",
        )
        job.scene.select_components(types=["PASSIVE_SMD_0805"])
        job.operations.add_shift(
            ratio=0.1,
            translate_range=0.2,
            rotate_z_range=15,
        )
        job.operations.solder_joints(
            solver="procedural",
            component_types=["PASSIVE_SMD_0805"],
            solver_options={"lead_local_sizing": True},
        )
        _configure_capture(job)

        plan = job.compile_orders(output_dir="output")
        by_id = {order.order_id: order for order in plan.orders}

        self.assertNotIn("operation.solder_joints", by_id)
        self.assertEqual(
            by_id["solder_joints.generate"].parameters["solver"],
            "procedural",
        )
        self.assertEqual(
            by_id["solder_joints.generate"].parameters["solver_options"],
            {"lead_local_sizing": True},
        )
        self.assertEqual(
            [
                order.order_id
                for order in plan.orders
                if order.order_id.startswith("solder_joints.")
            ],
            [
                "solder_joints.discover",
                "solder_joints.collect",
                "solder_joints.generate",
                "solder_joints.attach",
            ],
        )
        self.assertEqual(
            by_id["solder_joints.discover"].depends_on,
            (
                "scene.load",
                "scene.sidecar.components",
                "scene.components.select",
            ),
        )
        self.assertEqual(
            by_id["solder_joints.collect"].depends_on,
            ("solder_joints.discover", "operation.shift"),
        )
        self.assertEqual(
            by_id["solder_joints.attach"].depends_on,
            ("solder_joints.generate",),
        )
        for capture_order_id in (
            "capture.output",
            "capture.renderer",
            "capture.camera",
            "capture.scan_grid",
            "capture.limits",
            "capture.outputs",
        ):
            self.assertIn(
                "solder_joints.attach",
                by_id[capture_order_id].depends_on,
            )
        self.assertIn(
            "solder_joints.attach",
            by_id["capture.render"].depends_on,
        )

    def test_solder_dependencies_override_declarative_capture_order(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.scene.attach_sidecar(
            name="components",
            kind="pcb.components",
            path="components.json",
        )
        _configure_capture(job)
        job.operations.solder_joints(solver="procedural")
        job.scene.select_components(types=["PASSIVE_SMD_0805"])
        job.operations.add_shift(
            ratio=0.1,
            translate_range=0.2,
            rotate_z_range=15,
        )

        plan = job.compile_orders(output_dir="output")
        declared_ids = [order.order_id for order in plan.orders]
        ordered_ids = [order.order_id for order in plan.ordered()]

        self.assertLess(
            declared_ids.index("capture.renderer"),
            declared_ids.index("solder_joints.attach"),
        )
        self.assertLess(
            ordered_ids.index("operation.shift"),
            ordered_ids.index("solder_joints.collect"),
        )
        self.assertLess(
            ordered_ids.index("solder_joints.attach"),
            ordered_ids.index("capture.output"),
        )
        self.assertLess(
            ordered_ids.index("solder_joints.attach"),
            ordered_ids.index("capture.renderer"),
        )
        self.assertLess(
            ordered_ids.index("solder_joints.attach"),
            ordered_ids.index("capture.render"),
        )

    def test_solder_configuration_carries_extension_and_solver_options(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.scene.attach_sidecar(
            name="components",
            kind="pcb.components",
            path="components.json",
        )
        job.operations.solder_joints(
            solver="procedural",
            solver_options={"height_ratio": 0.35, "tht_solder_bath": True},
            output_root_path="/Board/GeneratedSolder",
            serialization_path="output/solder.usda",
        )
        _configure_capture(job)

        plan = job.compile_orders(output_dir="output")
        parameters = {order.order_id: order.parameters for order in plan.orders}[
            "solder_joints.generate"
        ]

        self.assertEqual(
            parameters["solver_options"],
            {"height_ratio": 0.35, "tht_solder_bath": True},
        )
        self.assertEqual(
            parameters["output_root_path"],
            "/Board/GeneratedSolder",
        )
        self.assertEqual(parameters["serialization"], "solder-joints-only")
        self.assertEqual(parameters["serialization_path"], "output/solder.usda")

    def test_legacy_all_serialization_compiles_without_capture(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.operations.solder_joints(
            solver="procedural",
            source="usd",
            serialization="all",
            serialization_path="output/solder.usda",
        )

        plan = job.compile_orders()

        self.assertEqual(plan.orders[-1].order_id, "solder_joints.attach")
        self.assertNotIn("capture.render", {order.order_id for order in plan.orders})

    def test_unserialized_solder_job_still_requires_capture(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.operations.solder_joints(solver="procedural", source="usd")

        with self.assertRaisesRegex(
            ValidationError,
            "no capture render pass or serialization output",
        ):
            job.compile_orders()

    def test_final_stage_serialization_precedes_capture(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        _configure_capture(job)

        plan = job.compile_orders(
            output_dir="renders",
            serialization_path="final_stage.usda",
        )
        order_ids = [order.order_id for order in plan.orders]
        serialization = plan.orders[order_ids.index("stage.serialize")]

        self.assertEqual(
            serialization.parameters,
            {"serialization_path": "final_stage.usda"},
        )
        self.assertLess(
            order_ids.index("stage.serialize"),
            order_ids.index("capture.render"),
        )

    def test_static_lighting_rig_is_applied_before_final_stage_serialization(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.appearance.lighting(mode="modus_light", modus={})
        with job.capture.render_pass("component_focus") as render_pass:
            render_pass.renderer(
                mode="pathtracing",
                resolution=(512, 512),
                spp=1,
                total_spp=32,
                max_bounces=4,
            )
            render_pass.camera(side="top", rotation_range={"z": -90})
            render_pass.scan_grid(x_count=1, y_count=1)
            render_pass.outputs(rgb=True)

        plan = job.compile_orders(
            output_dir="renders",
            serialization_path="final_stage.usda",
        )
        order_ids = [order.order_id for order in plan.orders]
        lighting = plan.orders[
            order_ids.index("appearance.lighting.component_focus.apply")
        ]

        self.assertEqual(lighting.order_type, "appearance.lighting.apply")
        self.assertEqual(lighting.parameters, {"camera_side": "top"})
        self.assertLess(
            order_ids.index("appearance.lighting.component_focus.apply"),
            order_ids.index("stage.serialize"),
        )
        self.assertLess(
            order_ids.index("stage.serialize"),
            order_ids.index("capture.component_focus.render"),
        )
        self.assertIn(
            "appearance.lighting.component_focus.apply",
            plan.orders[order_ids.index("stage.serialize")].depends_on,
        )

    def test_static_lighting_is_applied_for_each_capture_side(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.appearance.lighting(mode="modus_light", modus={})
        for name, side in (("top", "top"), ("bottom", "bottom")):
            with job.capture.render_pass(name) as render_pass:
                render_pass.renderer(
                    mode="pathtracing",
                    resolution=(64, 64),
                    spp=1,
                    total_spp=1,
                    max_bounces=1,
                )
                render_pass.camera(side=side, rotation_range={})
                render_pass.scan_grid(x_count=1, y_count=1)
                render_pass.outputs(rgb=True)

        plan = job.compile_orders(
            output_dir="renders",
            serialization_path="final_stage.usda",
        )
        by_id = {order.order_id: order for order in plan.orders}
        order_ids = [order.order_id for order in plan.orders]

        self.assertEqual(
            by_id["appearance.lighting.top.apply"].parameters,
            {"camera_side": "top"},
        )
        self.assertEqual(
            by_id["appearance.lighting.bottom.apply"].parameters,
            {"camera_side": "bottom"},
        )
        self.assertLess(
            order_ids.index("appearance.lighting.top.apply"),
            order_ids.index("stage.serialize"),
        )
        self.assertLess(
            order_ids.index("capture.top.render"),
            order_ids.index("appearance.lighting.bottom.apply"),
        )
        self.assertLess(
            order_ids.index("appearance.lighting.bottom.apply"),
            order_ids.index("capture.bottom.render"),
        )

    def test_incomplete_capture_still_fails_with_serialization(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.capture.camera(rotation_range={})

        with self.assertRaisesRegex(ValidationError, "missing"):
            job.compile_orders(serialization_path="final_stage.usda")

    def test_solder_defaults_to_pass_through_when_operation_is_invoked(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.scene.attach_sidecar(
            name="components",
            kind="pcb.components",
            path="components.json",
        )

        job.operations.solder_joints()

        self.assertEqual(job.operations.solder_joints_spec["solver"], "none")

    def test_omitting_solder_adds_no_internal_work(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        _configure_capture(job)

        plan = job.compile_orders(output_dir="output")

        self.assertFalse(
            any(order.order_id.startswith("solder_joints.") for order in plan.orders),
        )

    def test_enabled_solder_defaults_to_embedded_usd_metadata(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")

        job.operations.solder_joints(solver="procedural")
        _configure_capture(job)

        plan = job.compile_orders(output_dir="output")
        discover = next(
            order for order in plan.orders if order.order_id == "solder_joints.discover"
        )
        self.assertEqual(discover.parameters["source"], "usd")
        self.assertEqual(
            discover.depends_on,
            ("scene.load", "scene.components.select"),
        )

    def test_solder_requires_registered_solver(self):
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="board.usd", root_path="/Board")
        job.scene.attach_sidecar(
            name="components",
            kind="pcb.components",
            path="components.json",
        )

        with self.assertRaisesRegex(ValidationError, "unknown solder solver"):
            job.operations.solder_joints(
                solver="missing-solver",
            )

        with self.assertRaisesRegex(ValidationError, "unknown solder solver"):
            job.operations.solder_joints(
                solver="surface-evolver",
            )

    def test_solder_solver_listing_supports_type_filter(self):
        job = CDFJob(domain="pcb-aoi")

        # 'none' and 'procedural' are always listed. The simulated solver
        # (surface_evolver) is environment-gated -- it lists only where its heavy
        # deps import (a full dev machine, not the CI numpy-only image) -- so the
        # simulated filter is either empty or exactly that solver, and it sorts
        # last in the full listing.
        simulated = job.operations.get_solder_joint_solvers(mode_filter="simulated")
        self.assertIn([entry["id"] for entry in simulated], ([], ["surface_evolver"]))
        for entry in simulated:
            self.assertEqual(entry["type"], "simulated")

        self.assertEqual(
            job.operations.get_solder_joint_solvers(),
            [
                {"id": "none", "type": "none", "available": True},
                {
                    "id": "procedural",
                    "type": "procedural",
                    "available": True,
                },
                *simulated,
            ],
        )
        self.assertEqual(
            job.operations.get_solder_joint_solvers(
                mode_filter="procedural",
            ),
            [
                {
                    "id": "procedural",
                    "type": "procedural",
                    "available": True,
                },
            ],
        )
        self.assertEqual(
            job.operations.get_solder_joint_solvers(
                mode_filter="none",
            ),
            [{"id": "none", "type": "none", "available": True}],
        )
        with self.assertRaisesRegex(ValidationError, "filter"):
            job.operations.get_solder_joint_solvers(
                mode_filter="unsupported",
            )


if __name__ == "__main__":
    unittest.main()
