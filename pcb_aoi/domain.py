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

"""PCB-AOI domain registration for the domain-neutral CDF registry."""

from cdf.domain import DomainSpec, register_domain

from .geometry_labels import BOARD_GEOMETRY_ROLES, GEOMETRY_TARGETS
from .sections import PcbAoiCaptureSection, PcbAoiOperationsSection

DOMAIN = DomainSpec(
    name="pcb-aoi",
    operations=frozenset(
        {
            "shift",
            "tombstone",
            "sideflip",
            "reverse_polarity",
            "missing",
        },
    ),
    job_operations=frozenset(
        {
            "solder_joints",
            "solder_defects",
            "pin_variation",
            "hole_plane",
            "pin_tip",
        },
    ),
    geometry_targets=GEOMETRY_TARGETS,
    board_geometry_roles=BOARD_GEOMETRY_ROLES,
    sidecar_kinds=frozenset({"pcb.components", "pcb.sim_manifest"}),
    operations_section_factory=PcbAoiOperationsSection,
    capture_section_factory=PcbAoiCaptureSection,
)

register_domain(DOMAIN)
