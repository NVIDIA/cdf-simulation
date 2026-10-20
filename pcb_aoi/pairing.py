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

"""Provide Isaac-free maths for aligned PCB defect pairs.

Everything a pair record needs that does not require a stage: the union of the
clean and defect boxes, the severity magnitude and bucket for a defect type, and
how an occlusion ratio resolves. Imports nothing from Isaac, which is why it can
be unit tested off-GPU and why it lives here rather than beside the capture
script that currently calls it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Defect-type -> (DefectTransform.parameters field carrying its continuous
# magnitude, units).  Types with no continuous magnitude map to ``None`` (the
# defect is categorical): ``reverse_polarity`` is a fixed 180deg spin,
# ``missing`` is a binary presence flip, ``solder_defect`` is fillets present or
# absent.  Field names are single-sourced from ``defects_utils/api.py`` /
# ``defects_utils/component_rotation.py`` / ``defects_utils/tombstone.py``.
_SEVERITY_MAGNITUDE_FIELD: dict[str, tuple[str, str] | None] = {
    "shift": ("translate_magnitude_mm", "mm"),
    "tombstone": ("angle_deg", "deg"),
    "sideflip": ("angle_deg", "deg"),
    "reverse_polarity": None,  # binary: fixed 180deg, no continuous magnitude
    "missing": None,  # binary: presence flip
    "solder_defect": None,  # binary: fillets present / absent
}


def bbox_union(*boxes: list[int] | None) -> list[int] | None:
    """Union of the non-null ``[x0, y0, x1, y1]`` boxes (the inpaint region).

    Args:
        *boxes: Zero or more inclusive bboxes; ``None`` entries are ignored.

    Returns:
        The enclosing bbox, or ``None`` when every argument is ``None``.

    """
    present = [b for b in boxes if b is not None]
    if not present:
        return None
    return [
        int(min(b[0] for b in present)),
        int(min(b[1] for b in present)),
        int(max(b[2] for b in present)),
        int(max(b[3] for b in present)),
    ]


#: The occlusion field's authoritative meaning (echoed into every pair record).
OCCLUSION_SEMANTIC = "replicator_occlusion_ratio_defect_pass"
#: The area-based fallback meaning, used only when the native annotator is absent.


def resolve_occlusion(
    defect_occlusion_ratio: float | None, *, is_missing: bool
) -> float | None:
    """Occlusion of the target component in the DEFECT pass, in ``[0, 1]``.

    The value is Replicator's native per-instance ``occlusionRatio`` (a
    pixel-accurate fraction of the target that is occluded), measured on the
    defect-pass render.  ``missing`` is a special case: the component is gone, so
    occlusion is ``1.0`` by convention.  This is a real definition, not a
    placeholder — see :data:`OCCLUSION_SEMANTIC`.

    Args:
        defect_occlusion_ratio: The target's ``occlusionRatio`` from the defect
            pass (``None`` when the annotator reported nothing, e.g. the instance
            is absent).
        is_missing: Whether this scene's defect type is ``missing``.

    Returns:
        The occlusion fraction in ``[0, 1]``, or ``None`` when it is genuinely
        unavailable for a present component (never ``None`` for ``missing``).

    """
    if is_missing:
        return 1.0
    if defect_occlusion_ratio is None:
        return None
    return float(np.clip(float(defect_occlusion_ratio), 0.0, 1.0))


def severity_magnitude(
    defect_type: str | None, parameters: dict[str, Any] | None
) -> tuple[float | None, str | None, str | None]:
    """Continuous defect magnitude sampled for this scene, or nulls.

    Reads the magnitude straight from ``DefectTransform.parameters`` using
    single-sourced field names.  Returns ``(None, None, None)`` for categorical
    defects (``missing``, ``reverse_polarity``, ``solder_defect``) which have no
    continuous magnitude, and also when the expected field is absent.

    Args:
        defect_type: Internal defect key (``shift``/``tombstone``/``sideflip``/
            ``reverse_polarity``/``missing``/``solder_defect``).
        parameters: ``DefectTransform.parameters`` dict (may be ``None``).

    Returns:
        ``(magnitude, field_name, units)`` — all ``None`` when unavailable.

    """
    spec = _SEVERITY_MAGNITUDE_FIELD.get((defect_type or "").lower())
    if spec is None:
        return None, None, None
    field, units = spec
    value = (parameters or {}).get(field)
    if value is None:
        return None, None, None
    return float(value), field, units


# TODO(SDGW-cosmos3): severity BUCKET thresholds are DOMAIN INPUT.
def severity_bucket(defect_type: str | None, magnitude: float | None) -> str | None:
    """Map a magnitude to a coarse severity bucket.

    Always ``None`` today: the cut points between minor / moderate / severe are
    the requester's domain input and we do not have them. The field exists in
    the record schema so consumers can rely on its presence, and inventing
    thresholds here would ship a guess as if it were truth.

    Args:
        defect_type: Internal defect key.
        magnitude: Continuous magnitude from :func:`severity_magnitude`.

    Returns:
        ``None`` until real thresholds are supplied.

    """
    return None
