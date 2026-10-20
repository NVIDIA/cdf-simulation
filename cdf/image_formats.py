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

"""Accepted ``image_format`` values and how the capture writer produces them.

One allow-list serves the YAML surface (``OutputsConfig``), the Python surface
(``capture.outputs(...)``) and the compiled ``capture.outputs.configure``
parameters, so the three cannot drift apart.

Replicator's BasicWriter uses the requested format as the filename extension
and hands the frame to its disk backend, which only encodes ``png``, ``jpeg``
and ``jpg``. Any other extension raises inside the writer's io thread, where
nothing reports it: the run exits 0 and no rgb frame reaches disk. A format
outside :data:`WRITER_NATIVE_IMAGE_FORMATS` is therefore rendered as PNG and
re-encoded after the writer flushes -- rgb frames only; masks, bounding boxes
and label sidecars keep the writer's native formats.
"""

from __future__ import annotations

from typing import Literal, get_args

from .errors import ValidationError

ImageFormat = Literal["png", "jpeg", "jpg", "bmp"]

#: Every value ``image_format`` accepts, on all three authoring surfaces.
IMAGE_FORMATS: tuple[str, ...] = get_args(ImageFormat)

#: Formats BasicWriter's disk backend encodes itself. Anything else in
#: :data:`IMAGE_FORMATS` is produced by converting the rgb frames after capture.
WRITER_NATIVE_IMAGE_FORMATS = frozenset({"png", "jpeg", "jpg"})


def validate_image_format(value: object, *, name: str = "image_format") -> str:
    """Return ``value`` when it is an accepted image format.

    Args:
        value: The format as the caller wrote it.
        name: Parameter name used in the error message.

    Returns:
        The validated format, unchanged.

    Raises:
        ValidationError: If ``value`` is not one of :data:`IMAGE_FORMATS`.

    """
    if not isinstance(value, str) or value not in IMAGE_FORMATS:
        raise ValidationError(
            f"{name} must be one of {', '.join(IMAGE_FORMATS)}, got {value!r}",
        )
    return value
