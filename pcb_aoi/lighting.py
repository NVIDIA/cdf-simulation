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

"""The lighting rig selector, resolved from a pipeline config.

Kept free of ``carb`` / ``pxr`` imports so the resolution rule can be unit
tested outside Isaac -- ``common`` cannot, and this rule is exactly the part
worth testing: which rig a config asks for, and what to tell someone whose
config still uses the retired boolean spelling.
"""

from __future__ import annotations

from typing import Any

#: The rigs the pipeline can build. Two are never active at once, so a config
#: names exactly one.
LIGHTING_MODES = (
    "rgb_light",
    "white_light",
    "dome_light",
    "modus_light",
    "usd_light",
)

#: The pre-enum spelling of the same choice, retained for compatibility with
#: previously serialized lighting payloads.
RETIRED_LIGHTING_KEYS = {
    "use_scene_lights": "lighting.mode: usd_light",
    "ring_light": "lighting.mode: rgb_light (true) / dome_light (false)",
}


def _mode_from_retired_booleans(cfg, light_cfg):
    """Resolve the legacy boolean spelling, or return None if it is absent.

    Resolving the booleans in ONE place is the point: before the enum, the rig
    builder read ``ring_light`` as False-by-default while the per-trigger
    randomizer read it as True-by-default, so a config that omitted the key got
    opposite answers from the two steps. Both now call this function.
    """
    if cfg.get("use_scene_lights") or light_cfg.get("use_scene_lights"):
        return "usd_light"
    if "ring_light" in light_cfg:
        return "rgb_light" if light_cfg["ring_light"] else "dome_light"
    if "use_scene_lights" in cfg or "use_scene_lights" in light_cfg:
        return "dome_light"
    return None


def lighting_mode(cfg: dict[str, Any]) -> str:
    """Resolve which lighting rig a config asks for.

    ``lighting.mode`` wins. A legacy flow config that still carries the retired
    ``use_scene_lights`` / ``lighting.ring_light`` booleans resolves through
    them instead, so the compat surface keeps running unchanged.

    Args:
        cfg: Merged pipeline config.

    Returns:
        One of :data:`LIGHTING_MODES`.

    Raises:
        ValueError: If the config names no rig at all, or names an unknown one.

    Example:

    .. code-block:: python

        print(lighting_mode({"lighting": {"mode": "dome_light"}}))
        # dome_light

    """
    light_cfg = cfg.get("lighting") or {}
    mode = light_cfg.get("mode")
    if not mode:
        mode = _mode_from_retired_booleans(cfg, light_cfg)
    if not mode:
        raise ValueError(
            "lighting.mode is required, one of %s" % ", ".join(LIGHTING_MODES)
        )
    if mode not in LIGHTING_MODES:
        raise ValueError(
            "lighting.mode must be one of %s, got %r"
            % (", ".join(LIGHTING_MODES), mode)
        )
    return mode


def sample_ring_lighting(
    rng: Any,
    base_intensity: dict[str, float],
    *,
    exposure_range: tuple[float, float],
    cone_angle_range: tuple[float, float],
    cone_softness_range: tuple[float, float],
    intensity_jitter: float = 0.0,
) -> dict[str, Any]:
    """Sample one scene's AOI ring-light parameters (photometric randomization).

    Pure math (no Isaac): the caller applies the returned values to the
    existing ring rig's DiskLights.  Sampling once per scene — and holding it
    constant across a pair's clean and defect passes — keeps the two frames
    pixel-aligned while varying the illumination across scenes so a downstream
    model sees many lighting conditions.  Draw order is fixed (exposure, cone
    angle, cone softness, then layers in ``base_intensity`` order) so a given
    ``rng`` seed is reproducible.

    Args:
        rng: A ``numpy.random.Generator`` (seed per scene for reproducibility).
        base_intensity: Per-layer un-exposed base intensity, e.g.
            ``{"Inner_Red": 2500.0, "Middle_Green": 1000.0, "Outer_Blue": 1200.0}``.
        exposure_range: ``(lo, hi)`` global exposure multiplier applied to every
            layer's base intensity.
        cone_angle_range: ``(lo, hi)`` spotlight cone angle in degrees.
        cone_softness_range: ``(lo, hi)`` spotlight cone softness.
        intensity_jitter: Per-layer fractional jitter ``j``; each layer is scaled
            by ``1 + U(-j, +j)`` independently.  ``0`` disables per-layer jitter.

    Returns:
        A dict with ``exposure``, ``cone_angle``, ``cone_softness`` and a
        ``layer_intensity`` map (``layer -> intensity``), plus ``randomized: True``.
        Each intensity is ``max(0, base * exposure * (1 + jitter))``.

    """
    exposure = float(rng.uniform(*exposure_range))
    cone_angle = float(rng.uniform(*cone_angle_range))
    cone_softness = float(rng.uniform(*cone_softness_range))
    j = max(0.0, float(intensity_jitter))
    layer_intensity: dict[str, float] = {}
    for layer, base in base_intensity.items():
        jitter = float(rng.uniform(-j, j)) if j > 0.0 else 0.0
        layer_intensity[layer] = max(0.0, float(base) * exposure * (1.0 + jitter))
    return {
        "randomized": True,
        "exposure": exposure,
        "cone_angle": cone_angle,
        "cone_softness": cone_softness,
        "layer_intensity": layer_intensity,
    }
