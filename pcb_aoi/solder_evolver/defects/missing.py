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

"""Missing solder -- the site is pasted and populated, but no joint formed.

Produces NO geometry, which is why this needs a strategy rather than a
parameter: a volume of zero is not a very small joint, it is the absence of one,
and asking a fixed-volume minimiser for zero is meaningless.

Distinct from a component that is absent (an unoccupied or intentionally
unpopulated site, which the USD contract marks with `occupied` /
`intentionallyUnpopulated`). Here the part is present and the solder is not.
"""

from __future__ import annotations

from .base import DefectStrategy


class Missing(DefectStrategy):
    defect_type = "missing_solder"
    label = "missing"
    severity = "bad"
    description = "no solder formed at a populated, pasted site"

    def joint_ids(self, all_ids):
        return []                       # nothing to solve

    def target_volume(self, nominal_volume, joint_index):
        return 0.0
