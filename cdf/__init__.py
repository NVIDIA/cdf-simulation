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

"""cdf — declarative Cosmos Data Factory job API.

Declarative Python description layer over the existing configuration-driven
pipeline. A job built here and its YAML counterpart resolve to the same
normalized job plan; see ``cdf/README.md`` and the SRD/SDD
(Declarative CDF API).

    import pcb_aoi                      # registers the "pcb-aoi" domain
    from cdf import CDFJob

    job = CDFJob(domain="pcb-aoi", random_seed=0)
    job.scene.configure(usd_path=..., root_path=...)
    ...
    plan = job.compile_orders(output_dir=...)
    result = job.execute(output_dir=..., serialization_path=...)
"""

from .api_version import CDF_API_STATUS, CDF_API_VERSION
from .batch import BatchJob, render_batch
from .component_filter import ComponentFilter
from .config import (
    JobConfig,
    build_job_from_config,
    dump_job_config_schema,
    load_job,
    load_job_config,
)
from .domain import DomainSpec, register_domain
from .errors import CDFError, JobExecutionError, ValidationError
from .execution import (
    JobExecutor,
    JobOrder,
    JobOrderCodec,
    JobOrderHandler,
    JobOrderHandlerRegistry,
    JobOrderPlan,
    JobOrderResult,
    JobResult,
)
from .job import CDFJob
from .results import CropResult, RenderResult

__all__ = [
    "CDF_API_STATUS",
    "CDF_API_VERSION",
    "BatchJob",
    "CDFError",
    "CDFJob",
    "ComponentFilter",
    "CropResult",
    "DomainSpec",
    "IsaacExecutionContext",
    "IsaacInProcessExecutor",
    "IsaacSdgRuntime",
    "JobConfig",
    "JobExecutionError",
    "JobExecutor",
    "JobOrder",
    "JobOrderCodec",
    "JobOrderHandler",
    "JobOrderHandlerRegistry",
    "JobOrderPlan",
    "JobOrderResult",
    "JobResult",
    "RenderResult",
    "ValidationError",
    "build_job_from_config",
    "create_default_isaac_executor",
    "create_default_isaac_handler_registry",
    "dump_job_config_schema",
    "load_job",
    "load_job_config",
    "register_domain",
    "render_batch",
]


def __getattr__(name):
    """Resolve backend-specific public exports on first use.

    Keeping the Isaac backend lazy lets the domain-neutral package import
    without importing any installed domain plugin.
    """
    isaac_exports = {
        "IsaacExecutionContext",
        "IsaacInProcessExecutor",
        "IsaacSdgRuntime",
        "create_default_isaac_executor",
        "create_default_isaac_handler_registry",
    }
    if name in isaac_exports:
        from .execution import isaac

        value = getattr(isaac, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
