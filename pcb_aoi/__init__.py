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

"""pcb_aoi — PCB-AOI domain plugin for the declarative CDF API.

Importing this module registers the ``pcb-aoi`` domain with :mod:`cdf`:
the defect-operation vocabulary implemented by
``scripts/sdg/standalone/defects_utils/api.py`` plus the ``missing`` flow,
and the sidecar kinds the pipeline consumes (``<board>.components.json``
and the simulation manifest).
"""

from .capabilities import register as _register_capabilities
from .config import register as _register_config_adapter
from .domain import DOMAIN
from .execution.compiler_adapter import register as _register_compiler_adapter

__all__ = ["DOMAIN"]

# Contribute this domain's Isaac handlers to CDF's executor. Importing the
# domain is what makes its orders executable, the same way importing it is
# what registers the domain itself.
from .execution import register as _register_isaac_adapter

_register_isaac_adapter()
_register_capabilities()
_register_config_adapter()
_register_compiler_adapter()
