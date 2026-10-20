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

"""Through-hole lead discovery and variation."""

from .lead_discovery import (
    DiscoveredLead,
    Hole,
    discover_through_hole_leads,
    pair_leads,
    scan_board,
)
from .pin_variation_apply import (
    LeadOutcome,
    plan_board_variation,
    select_leads,
    summarise,
    write_lead_points,
)

__all__ = [
    "DiscoveredLead",
    "Hole",
    "LeadOutcome",
    "discover_through_hole_leads",
    "pair_leads",
    "plan_board_variation",
    "scan_board",
    "select_leads",
    "summarise",
    "write_lead_points",
]
