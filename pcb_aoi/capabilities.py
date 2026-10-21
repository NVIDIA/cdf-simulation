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

"""Bind every PCB-AOI workflow capability across its batch surfaces.

The registry covers config, Python API, example user intent, preview behavior,
and the repository documentation that explains each capability. The generated
``cdf/CAPABILITIES.md`` matrix is the public inventory.
"""

from __future__ import annotations

from cdf.capability_registry import Capability, register_capabilities

_FLOWS_DOC = "pcb_aoi/config/README.md"
_API_DOC = "cdf/README.md"
_VARIANTS_DOC = "docs/defect-variants.md"
_DOCUMENTATION = (
    f"Docs: `{_FLOWS_DOC}` (config keys) and `{_API_DOC}` (Python API mapping)."
)

_PREVIEW_BATCH_OUT = "open the written pngs under `<output>/trigger_NNNN`"
_PREVIEW_BOARD = _PREVIEW_BATCH_OUT
_PREVIEW_REFDES = _PREVIEW_BATCH_OUT


CAPABILITIES = (
    Capability(
        id="scene.board",
        title="Load/compose the board USD",
        config="pcba_target.yaml: scene, pcba_root, camera_*, ring_light_root",
        api="scene.configure",
        skill="load the <board> in the scene",
        preview=_PREVIEW_BATCH_OUT,
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="scene.instances",
        title="Register a shared-prototype board-instance cohort",
        config="scene.instances",
        api="scene.instances",
        skill="apply identical operations to every board instance in a panel",
        preview="render the panel or component ROIs and compare corresponding boards",
        docs=("cdf/API_SPEC.md", "pcb_aoi/config/README.md"),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="scene.component_filter",
        title="Filter components by explicit metadata fields with nested intersection",
        config=(
            "component_filter under scene.components, operations, annotations.geometry, "
            "or capture ROI selectors"
        ),
        api="scene.select_components",
        skill=(
            "select components by refdes, component type, package type, footprint, "
            "side, or process class"
        ),
        preview="inspect the compiled component-selection and child-operation orders",
        docs=("pcb_aoi/config/README.md",),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="scene.sidecars",
        title="Attach component/simulation sidecars",
        config="components_json, sim_manifest",
        api="scene.attach_sidecar",
        skill="use the components json for eligibility",
        preview="indirect: eligible-component counts in the pipeline log; "
        "pad-aware fillet shapes change once the sidecar attaches",
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="appearance.lighting",
        title="Lighting rigs (mode: rgb/white/dome/modus/usd)",
        config="lighting.mode selects the rig: rgb_light (layers, plus "
        "optional dome_fill), white_light (dome pinned to white), "
        "dome_light (dome with an authored colour), modus_light "
        "(modus, plus optional batch dome_fill and camera-side-aware "
        "top/bottom placement), "
        "usd_light (reuse the board USD's lights). Two rigs "
        "are never active at once, so mode is the only selector -- the "
        "pre-enum use_scene_lights / ring_light booleans are gone. "
        "lighting.preserve_scene_light_color applies to usd_light; "
        "exposure_range/cone_angle_range/cone_softness_range control "
        "the authored lights. The aoi_rgb preset provides "
        "(the blessed AOI rig: R4000/G1000/B1200 rings, no dome "
        "fill, specular 5.0, sphere_radius 26.353)",
        api="appearance.lighting",
        skill="AOI RGB ring light (blessed: preset aoi_rgb); white light; "
        "dome fill; modus light (2D-dome linear AOI illuminator)",
        preview=_PREVIEW_BOARD,
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="appearance.vantablack",
        title="Vantablack component override",
        config="vantablack_components",
        api="appearance.components",
        skill="make components black (vantablack)",
        preview=_PREVIEW_BOARD,
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="appearance.material_override",
        title="OmniPBR material appearance override",
        config="appearance.materials",
        api="appearance.material_override",
        skill=(
            "apply color, opacity, roughness, and optional textures to an "
            "existing material"
        ),
        preview=_PREVIEW_BATCH_OUT,
        docs=(_API_DOC,),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="appearance.solder_fillets",
        title="Solder fillet menisci on component pads (good flows only)",
        config="solder_fillet (enabled, shape/noise/material keys, "
        "adaptive_scale); randomize_fillet",
        api="",
        skill="add solder fillets (good flows only — the pipeline rejects "
        "fillets in defect/missing flows)",
        preview=_PREVIEW_BATCH_OUT,
        docs=(_FLOWS_DOC, "COOKBOOK.md"),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="appearance.component_material",
        title="Component body colour override",
        config="component_material (body colour so authored colours don't "
        "compete with the ring light)",
        api="",
        skill="set component body colour (a colour, not black — for black "
        "use vantablack)",
        preview=_PREVIEW_BATCH_OUT,
        docs=(_FLOWS_DOC,),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="appearance.board_material",
        title="PCB board-layer blackout",
        config="board_material (forces PCB layers to vantablack); "
        "extra_blackout_prims, extra_blackout_enabled",
        api="",
        skill="black out the PCB board layers (substrate, not components)",
        preview=_PREVIEW_BATCH_OUT,
        docs=(_FLOWS_DOC,),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="appearance.tin_noise",
        title="Tin pad perlin normal-map",
        config="tin_noise (+ randomize_tin_noise)",
        api="",
        skill="add tin pad surface noise / roughen the solder pads",
        preview=_PREVIEW_BATCH_OUT,
        docs=(_FLOWS_DOC,),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.shift",
        title="Shift defect",
        config="defects.shift (count or ratio, translate_range, rotate_z_range, variant_index)",
        api="operations.add_shift",
        skill="nudge / offset a component's placement (shift)",
        preview=_PREVIEW_REFDES,
        docs=(_FLOWS_DOC, _API_DOC, _VARIANTS_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.pin_variation",
        title="Through-hole pin variation",
        config="pin_variation (list of groups: components XOR package_types, "
        "tilt, cut_angle, height, offset)",
        api="operations.pin_variation",
        skill="tilt / bend / trim the through-hole pins",
        preview=(
            "render the board BOTTOM-side: tilt, protrusion and the cut plane "
            "are all below the far face and a top-down pass shows none of them"
        ),
        docs=(_API_DOC, "COOKBOOK.md"),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.tombstone",
        title="Tombstone defect",
        config="defects.tombstone (count or ratio, angle_min/max, auto_filter)",
        api="operations.add_tombstone",
        skill="stand a part up on one end (tombstone)",
        preview=_PREVIEW_REFDES,
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.sideflip",
        title="Sideflip defect",
        config="defects.sideflip (count or ratio, angle_min/max, variant_index, rotate_z_jitter_deg, slide_range_mm)",
        api="operations.add_sideflip",
        skill="flip a component onto its side (sideflip)",
        preview=_PREVIEW_REFDES,
        docs=(_FLOWS_DOC, _API_DOC, _VARIANTS_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.reverse_polarity",
        title="Reverse-polarity defect",
        config="defects.reverse_polarity (count or ratio, component_types)",
        api="operations.add_reverse_polarity",
        skill="rotate a polarized part 180 degrees / wrong polarity",
        preview=_PREVIEW_REFDES,
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.missing",
        title="Missing-component defect",
        config="missing (count or ratio); pipeline_type: missing",
        api="operations.add_missing",
        skill="remove a component / absent part (missing)",
        preview=_PREVIEW_REFDES,
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.solder_joints",
        title="Solder-joint generation (procedural or simulated)",
        config="solder_joints (solver: none/procedural/surface_evolver, "
        "component_types, serialization, options incl. exact refdes/prim-path "
        "components allowlist, workers = total core cap, and procedural THT "
        "options tht_solder_bath, tht_profile_exponent, and "
        "tht_crease_angle_deg; solder bath extends each joint to envelop the "
        "exposed solder-side pin, profile exponent controls intermediate fillet "
        "fullness, and crease angle controls normal splitting); simulated "
        "solvers register only where their runtime deps import",
        api="operations.solder_joints",
        skill="generate solder joints / simulate solder with surface evolver; "
        "extend THT solder over the exposed pin, make the fillet fuller, or "
        "sharpen strong shading creases",
        preview=_PREVIEW_BATCH_OUT,
        docs=(
            "pcb_aoi/config/README.md",
            "pcb_aoi/solder_generation/README.md",
            _API_DOC,
        ),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.hole_plane",
        title="Hole-reveal plane generation",
        config="hole_plane (class_name, side: top/bottom/both, "
        "z_offset_mm — offset inward from the chosen board_substrate FR4 face, "
        "output_prim_path, components — optional refdes allowlist); without "
        "components: generates one board-sized quad mesh; substrate geometry "
        "occludes it everywhere except at through-hole voids; with components: "
        "generates one small plane per selected-component pad/hole site instead; "
        "grouped under SegmentationProxies with magenta USD display color; hidden "
        "by default and revealed in annotation passes with "
        "visibility.show: [hole] for hole segmentation",
        api="operations.hole_plane",
        skill="add a hole-reveal plane for through-hole segmentation",
        preview=_PREVIEW_BATCH_OUT,
        docs=("pcb_aoi/config/README.md", _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.pin_tip",
        title="Through-hole pin-tip annotation proxies",
        config="pin_tip (class_name, side: top/bottom, offset_mm, components — "
        "optional refdes allowlist); copies each selected terminal's connected "
        "outward tip surface, including curved and steeply angled tips, after pin "
        "cutting and solder attachment; traversal stops at vertical lead walls and "
        "beyond 30 degrees of tilt from the tip normal, so a rounded lead end keeps "
        "its chamfer in the pin class; grouped under "
        "SegmentationProxies with magenta USD display color; hidden by default and "
        "revealed in annotation passes with visibility.show: [pin_tip]",
        api="operations.pin_tip",
        skill="label through-hole pin tips separately from the rest of each pin",
        preview="open the semantic output and labels JSON under the pin-tip capture pass",
        docs=("pcb_aoi/config/README.md", _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.insufficient",
        title="Insufficient-solder defect",
        config="insufficient (ratio, volume_ratio value or [min, max], "
        "lead_climb value or [min, max], fullness value or [min, max], "
        "component_types); requires solder_joints with a simulated solver",
        api="operations.add_insufficient",
        skill="starved / insufficient solder on some joints",
        preview=_PREVIEW_BATCH_OUT,
        docs=("pcb_aoi/config/README.md", _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.excess",
        title="Excess-solder defect",
        config="excess (ratio, volume_ratio value or [min, max], "
        "lead_climb value or [min, max], fullness value or [min, max], "
        "component_types); requires solder_joints with a simulated solver",
        api="operations.add_excess",
        skill="excess / too much solder on some joints",
        preview=_PREVIEW_BATCH_OUT,
        docs=("pcb_aoi/config/README.md", _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.bright",
        title="Bright-solder defect",
        config="bright (ratio, lead_climb value or [min, max], fullness "
        "value or [min, max], component_types); THT-only; requires "
        "solder_joints with a simulated solver",
        api="operations.add_bright",
        skill="bright solder / steep starved meniscus on THT joints",
        preview=_PREVIEW_BATCH_OUT,
        docs=("pcb_aoi/config/README.md", _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.interrupted",
        title="Interrupted-solder defect",
        config="interrupted (ratio, pad_exposure value or [min, max], "
        "lead_climb value or [min, max], fullness value or [min, max], "
        "component_types); requires solder_joints with a simulated solver",
        api="operations.add_interrupted",
        skill="interrupted solder / partially exposed pad under the joint",
        preview=_PREVIEW_BATCH_OUT,
        docs=("pcb_aoi/config/README.md", _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.missing_solder",
        title="Missing-solder defect",
        config="missing_solder (ratio, component_types); requires "
        "solder_joints with a simulated solver; suppresses the printed paste "
        "so nothing renders where solder should be",
        api="operations.add_missing_solder",
        skill="missing solder / unsoldered joints on placed parts",
        preview=_PREVIEW_BATCH_OUT,
        docs=("pcb_aoi/config/README.md", _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="operations.dry_contact",
        title="Dry-contact solder defect",
        config="dry_contact (ratio, gap_fraction value or [min, max], "
        "lead_climb value or [min, max], fullness value or [min, max], "
        "component_types); through-hole only; requires solder_joints "
        "with a simulated solver",
        api="operations.add_dry_contact",
        skill="dry contact / unwetted arc, solder does not ring the pin",
        preview=_PREVIEW_BATCH_OUT,
        docs=("pcb_aoi/config/README.md", _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="annotations.labels",
        title="Semantic labels and ground-truth classes",
        config="writer.semantic_types; pcba_target.yaml: component_classes",
        api="annotations.label_components",
        skill="label components by class / defect labels",
        preview="open the semantic outputs (png + labels json) under "
        "`<output>/trigger_NNNN`",
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="scene.companion_assets",
        title="Textures shipped with the scene",
        config="pcba.scene (a directory or .zip); pcba.archive_member",
        api="appearance.material_override",
        skill="ship a board and its textures together",
        preview="address an asset as `scene://textures/albedo.png`, or give a "
        "relative path that falls back to the scene; the resolved path is "
        "reported as `textures` by the override order",
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="annotations.geometry",
        title="Solder, terminal (pin), pad, barrel, hole, package body and "
        "polarity-marking semantic classes",
        config="annotations.geometry (solder, terminals, pads, holes, barrels, body, "
        "marking — each maps a target to its semantic class name; barrels labels "
        "the cylindrical copper inner wall of drilled holes (board_barrel role "
        "prims) and is typically set to the same class as pads to eliminate the "
        "unlabeled ring visible between pad and hole in semantic segmentation; "
        "body labels the package body mesh (pkg_body role) and marking the pin-1 "
        "dot / polarity band mesh (pkg_marking role) so a mask can place the "
        "polarity marker on the body; components — optional refdes allowlist that "
        "restricts pads/terminals/holes/barrels/body/marking to the listed "
        "components; solder is not filtered); YAML uses AnnotationsGeometryConfig",
        api="annotations.label_geometry",
        skill="label solder joints, pins, pads, barrels, drilled holes, package "
        "bodies and pin-1 / polarity markings separately from their component; a "
        "pad or a hole is labellable because the board has one, not because a "
        "terminal claims it, so unassociated holes are reachable too; pass "
        "components= to restrict labeling to selected refdes components only",
        preview="open the semantic outputs (png + labels json) under "
        "`<output>/trigger_NNNN`",
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="capture.plan",
        title="Renderer, scan grid, visibility, limits, augmentation, writer outputs",
        config="passes, resolution, pathtracing, scan_grid, roi_selection "
        "(focused frame metadata includes component type, package, footprint, "
        "side, and process fields), visibility, num_triggers, max_image_count, "
        "augmentation, writer",
        api="capture.render_pass",
        skill="render 10x10 scan grid at 1080p with motion blur, or hide solder "
        "only for an annotation pass; counted "
        "asks ('generate 5 defect images') are ambiguous — resolve "
        "what N counts first (scan-grid cells: x_count x y_count "
        "images per pass, vs N components vs N triggers vs N "
        "viewpoints)",
        preview=_PREVIEW_BATCH_OUT,
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="capture.fov",
        title="State the capture as a scan rectangle at a scanner resolution",
        config="capture.fov: scan_area_mm or corners_mm, dpi / mm_per_px / "
        "image_size_px, max_tile_px, centered_on, guard_px (replaces "
        "renderer.resolution and scan_grid, which it derives; corners_mm also "
        "places the rectangle; guard_px renders each cell wider and crops the "
        "band before the tile is written)",
        api="capture.fov",
        skill="scan 300 x 200 mm at 600 dpi / image the board at 0.0423 mm per "
        "pixel / a 1200 dpi scan of the product area",
        preview=_PREVIEW_BATCH_OUT
        + "; the stitched image's `_to_usd.json` sidecar carries `fov.dpi` "
        "and `fov.mm_per_px`",
        docs=(
            _API_DOC,
            "pcb_aoi/config/examples/README.md",
            "docs/image-to-usd-sidecar.md",
        ),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="capture.render",
        title="Execute capture and optional composed-stage export",
        config="output, execution.serialization_path",
        api="capture.render",
        skill="render the job, export the composed stage, or do both in one job",
        preview=_PREVIEW_BATCH_OUT,
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="postprocess.crop",
        title="Component cropping of captured artifacts",
        config="crop (method, types, offset, min_coverage, xform_depth)",
        api="postprocess.crop",
        skill="crop the components out of the render / component crops "
        "from the dataset (labeled, all instances, from existing frames)",
        preview="open the per-class crop output folders next to the render output",
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="postprocess.stitch",
        title="Stitch orthographic scan-grid frames",
        config=(
            "stitch (source_pass, image_type, output_subdir, output_filename, "
            "blend, write_labels, recolor_tiles, color_map; supports sequential and "
            "grid-index-renamed tiles; blend=labels unifies per-tile "
            "segmentation colors; recolor_tiles applies them to source tiles; "
            "color_map overrides class colors). Also "
            "writes image_<name>_to_usd.json beside the stitched image and "
            "returns image_to_usd_path -- schema pcb_aoi.image_to_usd/1.0.0, "
            "USD stage world coordinates in stage units, raster top-left "
            "pixel-edge origin. Emitted ONLY for an orthographic camera on an "
            "auto-derived scan grid with a cardinal camera yaw: explicit legacy "
            "x_start/x_end grids, perspective or tilted cameras, ROI captures, "
            "and any camera_rotation.z_fixed that is not a multiple of 90 "
            "degrees have no derivable world placement, and the sidecar and "
            "image_to_usd_path are both omitted. Format, conventions and worked "
            "pixel-to-USD examples: docs/image-to-usd-sidecar.md"
        ),
        api="postprocess.stitch",
        skill=(
            "stitch scan-grid tiles into one high-resolution board image; "
            "assign specific colors to segmentation classes"
        ),
        preview="open the stitched image beneath the selected capture-pass output",
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="postprocess.close_openings",
        title="Recover the opening a ring-shaped label encloses",
        config="close_openings (source_pass, class_name, output_subdir, "
        "output_filename, max_diameter_px, min_circularity)",
        api="postprocess.close_openings",
        skill="fill in the middle of a drilled hole whose label came back as "
        "a rim; runs on the STITCHED mask, and finds that stitch by the "
        "pass, directory and filename the two of them share",
        preview="open the stitched mask beneath the selected capture-pass "
        "output and compare it with the same mask before closure",
        docs=(_FLOWS_DOC, _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="postprocess.format_roi_output",
        title="Format paired AOI ROI capture outputs",
        config="format_roi_output (rgb_pass, roi_pass); both passes require "
        "rename_to_grid_index; renames semantic masks, merges labels, writes "
        "pixel-layout metadata, and validates RGB/ROI grid alignment",
        api="postprocess.format_roi_output",
        skill="format paired full-board RGB and semantic tiles for the AOI ROI output contract",
        preview="inspect metadata.json and semantic_segmentation_labels.json under each trigger directory",
        docs=("pcb_aoi/config/README.md", _API_DOC),
        modes=frozenset({"batch"}),
    ),
    Capability(
        id="postprocess.generate_joint_rois",
        title="Generate stitched full-board solder-joint ROIs",
        config="generate_joint_rois (roi_pass, rgb_pass); consumes grid-indexed "
        "tight 2D boxes and pixel layout; writes trigger-local and stitched "
        "joint_rois.json manifests to both passes in full-board pixel coordinates "
        "with pin number, source prim paths, component type, package, footprint, "
        "and process metadata; unavailable catalog values remain null; "
        "bbox_xyxy is the pin, joint_bbox_xyxy adds the catalog-associated pads "
        "labeled in the ROI pass",
        api="postprocess.generate_joint_rois",
        skill="generate full-board pixel bounding boxes for solder joints from tiled "
        "tight-box annotations and publish them with both RGB and ROI outputs",
        preview="inspect matching joint_rois.json beside the stitched RGB and ROI images",
        docs=("pcb_aoi/config/README.md", _API_DOC),
        modes=frozenset({"batch"}),
    ),
)


def register() -> None:
    """Register the PCB-AOI capability catalog with CDF.

    Example:
        >>> register()

    """
    register_capabilities(
        "pcb-aoi",
        CAPABILITIES,
        documentation=_DOCUMENTATION,
    )
