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

"""Postprocess configuration models and Python API interpretation."""

from cdf.config._common import ConfigModel
from cdf.job import CDFJob

POSTPROCESS_API_CONFIG_FIELDS = {
    "crop": "crop",
    "stitch": "stitch",
    "close_openings": "close_openings",
    "format_roi_output": "format_roi_output",
    "generate_joint_rois": "generate_joint_rois",
}


class CropConfig(ConfigModel):
    """Component-centered crop settings for captured artifacts."""

    method: str = "bbox_tight"
    types: tuple[str, ...] | None = None
    offset: int = 10
    min_coverage: float = 0.2
    class_filter: tuple[str, ...] | None = None
    xform_depth: int = 7
    output_subdir: str = "cropped"
    mode: str | None = None
    input_subdir: str | None = None
    reference_subdir: str | None = None


class StitchConfig(ConfigModel):
    """Scan-grid image stitching settings."""

    source_pass: str = "default"
    image_type: str = "rgb"
    output_subdir: str = "stitched"
    output_filename: str = "rgb.png"
    blend: str = "none"
    #: Overlap policy: none, feather, or labels for a segmentation pass.
    write_labels: bool = False
    #: When ``True`` and ``blend`` is not ``labels``, write a ``*_labels.json`` sidecar.
    recolor_tiles: bool = False
    #: Apply the unified label colors to source tiles and their label sidecars.
    color_map: dict[str, tuple[int, int, int, int]] | None = None
    #: Optional per-class RGBA color overrides for ``blend='labels'`` output.


class CloseOpeningsConfig(ConfigModel):
    """Ring-closure settings for one segmentation class."""

    source_pass: str = "default"
    class_name: str
    output_subdir: str = "stitched"
    output_filename: str = "rgb.png"
    max_diameter_px: int = 48
    min_circularity: float = 0.55


class FormatRoiOutputConfig(ConfigModel):
    """AOI ROI output formatting settings.

    Both named passes must have ``rename_to_grid_index: true`` in their
    ``outputs`` block.  The postprocessor identifies tiles by the
    ``x<col>_y<row>`` suffix that setting produces; without it no files
    are matched and a ``ValidationError`` is raised.
    """

    rgb_pass: str = "full_board"
    roi_pass: str = "full_board_roi"


class GenerateJointRoisConfig(ConfigModel):
    """Joint ROI bounding-box generation settings.

    The ROI pass must have ``rename_to_grid_index: true`` in its
    ``outputs`` block.  The postprocessor locates bounding-box ``.npy``
    files by the ``x<col>_y<row>`` suffix that setting produces and uses
    those indices to look up per-tile canvas offsets from the pixel
    layout. Each output item includes an explicit pin number plus component
    type, type source, native package type, native package name, footprint
    identity, and manufacturing process class from the selected board metadata
    catalog. Missing or ambiguous component metadata is represented by null
    values. Exact contributing USD prim paths are retained, and the manifest is
    written beneath the trigger and stitched directories of both the ROI and RGB
    capture passes.
    """

    roi_pass: str = "full_board_roi"
    rgb_pass: str = "full_board"


class PostprocessConfig(ConfigModel):
    """Post-capture artifact-processing settings for a job."""

    crop: CropConfig | None = None
    close_openings: CloseOpeningsConfig | tuple[CloseOpeningsConfig, ...] | None = None
    #: Classes whose ring-shaped labels are closed before stitching.
    stitch: StitchConfig | tuple[StitchConfig, ...] | None = None
    #: One stitch, or several products of one pass -- typically the image and
    #: its segmentation, which need different overlap policies.
    format_roi_output: FormatRoiOutputConfig | None = None
    #: Rename semantic masks, merge label JSONs, preserve the dataset manifest,
    #: and run RGB/ROI acceptance checks.
    generate_joint_rois: GenerateJointRoisConfig | None = None
    #: Derive per-joint full-board pixel bboxes and component metadata.


def apply_postprocess(job: CDFJob, config: PostprocessConfig) -> None:
    """Apply postprocess declarations."""
    if config.crop is not None:
        job.postprocess.crop(**config.crop.model_dump())
    if config.close_openings is not None:
        closures = (
            config.close_openings
            if isinstance(config.close_openings, tuple)
            else (config.close_openings,)
        )
        for closure in closures:
            job.postprocess.close_openings(**closure.model_dump())
    if config.stitch is not None:
        entries = (
            config.stitch if isinstance(config.stitch, tuple) else (config.stitch,)
        )
        for entry in entries:
            job.postprocess.stitch(**entry.model_dump())
    if config.format_roi_output is not None:
        job.postprocess.format_roi_output(**config.format_roi_output.model_dump())
    if config.generate_joint_rois is not None:
        job.postprocess.generate_joint_rois(**config.generate_joint_rois.model_dump())
