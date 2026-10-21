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

"""Expose domain-neutral CDF job-order execution infrastructure."""

from .dataset_manifest import (
    DatasetManifestObserver,
    EntityIdentity,
    ManifestArtifact,
    ManifestDefect,
    ManifestFrame,
    ManifestROISelection,
    ManifestSerializer,
    ManifestView,
)
from .executor_base import JobExecutor
from .handler_registry import JobOrderHandler, JobOrderHandlerRegistry
from .job_order_codec import JobOrderCodec
from .job_order_types import JobOrder, JobOrderPlan
from .job_result_types import JobOrderResult, JobResult

__all__ = [
    "DatasetManifestObserver",
    "EntityIdentity",
    "JobExecutor",
    "JobOrder",
    "JobOrderCodec",
    "JobOrderHandler",
    "JobOrderHandlerRegistry",
    "JobOrderPlan",
    "JobOrderResult",
    "JobResult",
    "ManifestArtifact",
    "ManifestDefect",
    "ManifestFrame",
    "ManifestROISelection",
    "ManifestSerializer",
    "ManifestView",
]
