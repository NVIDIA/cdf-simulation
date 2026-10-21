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

"""Versioning, loading, and orchestration for YAML-authored CDF jobs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from scripts import file_utils

from ..api_version import CDF_API_VERSION
from ..errors import CDFError
from ..job import CDFJob
from ._common import ConfigModel
from .domain_adapters import (
    get_domain_config,
    load_installed_domain_configs,
    registered_domain_configs,
)
from .job_execution import ExecutionConfig

ApiVersion = Literal[CDF_API_VERSION]
_DEFINITION_REF_PREFIX = "#/$defs/"


def _qualify_definition_name(domain: str, name: str) -> str:
    """Return a domain-qualified JSON Schema definition name."""
    return f"{domain}.{name}"


def _qualify_schema_references(value: object, domain: str) -> None:
    """Qualify local JSON Schema definition references in place."""
    if isinstance(value, dict):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith(
            _DEFINITION_REF_PREFIX,
        ):
            name = reference.removeprefix(_DEFINITION_REF_PREFIX)
            qualified_name = _qualify_definition_name(domain, name)
            value["$ref"] = f"{_DEFINITION_REF_PREFIX}{qualified_name}"
        for nested_value in value.values():
            _qualify_schema_references(nested_value, domain)
    elif isinstance(value, list):
        for nested_value in value:
            _qualify_schema_references(nested_value, domain)


class JobConfig(ConfigModel):
    """Versioned, human-authored job configuration."""

    api_version: ApiVersion
    domain: str
    random_seed: int = 0
    output_dir: str = Field(default="output")
    scene: object
    appearance: object = Field(default_factory=dict)
    operations: object = Field(default_factory=dict)
    annotations: object = Field(default_factory=dict)
    capture: object | None = None
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    postprocess: object = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def validate_domain_config(cls, value: object) -> object:
        """Validate domain-owned configuration blocks with their plugin.

        Args:
            value: Raw configuration payload.

        Returns:
            Payload with its operations block replaced by the domain model.

        Raises:
            ValueError: If the payload or domain field is invalid.

        """
        if not isinstance(value, Mapping):
            # Pydantic v2 does not wrap `TypeError` from model validators.
            raise ValueError("JobConfig must be a mapping")  # noqa: TRY004
        domain = value.get("domain")
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("JobConfig domain must be a non-empty string")
        adapter = get_domain_config(domain)
        payload = dict(value)
        for field_name, model, default in (
            ("scene", adapter.scene_model, None),
            ("appearance", adapter.appearance_model, {}),
            ("operations", adapter.operations_model, {}),
            ("annotations", adapter.annotations_model, {}),
            ("postprocess", adapter.postprocess_model, {}),
        ):
            payload[field_name] = model.model_validate(
                payload.get(field_name, default),
            )
        capture = payload.get("capture")
        payload["capture"] = (
            adapter.capture_model.model_validate(capture)
            if capture is not None
            else None
        )
        return payload

    @classmethod
    def model_json_schema(cls, *args: object, **kwargs: object) -> dict[str, object]:
        """Return JSON Schema including registered domain operation models.

        Args:
            *args: Positional arguments accepted by Pydantic.
            **kwargs: Keyword arguments accepted by Pydantic.

        Returns:
            JSON Schema with one registered operation schema per domain.

        """
        schema = super().model_json_schema(*args, **kwargs)
        adapters = registered_domain_configs()
        if not adapters:
            return schema
        definitions = schema.setdefault("$defs", {})
        field_names = (
            "scene",
            "appearance",
            "operations",
            "annotations",
            "capture",
            "postprocess",
        )
        references: dict[str, list[dict[str, str]]] = {
            field_name: [] for field_name in field_names
        }
        for domain, adapter in sorted(adapters.items()):
            for field_name, model in (
                ("scene", adapter.scene_model),
                ("appearance", adapter.appearance_model),
                ("annotations", adapter.annotations_model),
                ("operations", adapter.operations_model),
                ("capture", adapter.capture_model),
                ("postprocess", adapter.postprocess_model),
            ):
                field_schema = model.model_json_schema(
                    ref_template="#/$defs/{model}",
                )
                nested_definitions = field_schema.pop("$defs", {})
                _qualify_schema_references(field_schema, domain)
                for name, definition in nested_definitions.items():
                    _qualify_schema_references(definition, domain)
                    qualified_name = _qualify_definition_name(domain, name)
                    definitions[qualified_name] = definition
                model_name = _qualify_definition_name(domain, model.__name__)
                definitions[model_name] = field_schema
                references[field_name].append(
                    {
                        "$ref": f"#/$defs/{model_name}",
                        "title": f"{domain} {field_name}",
                    },
                )
        for field_name, field_references in references.items():
            domain_schema = (
                field_references[0]
                if len(field_references) == 1
                else {"oneOf": field_references}
            )
            if field_name == "capture":
                original_schema = schema["properties"][field_name]
                domain_schema = {
                    "anyOf": [domain_schema, {"type": "null"}],
                    "default": original_schema.get("default"),
                    "title": original_schema.get("title", "Capture"),
                }
            schema["properties"][field_name] = domain_schema
        return schema

    @field_validator("api_version", mode="before")
    @classmethod
    def validate_api_version(cls, value: object) -> object:
        """Report the supplied and active API versions on incompatibility."""
        if value != CDF_API_VERSION:
            raise ValueError(
                f"unsupported CDF API version {value!r}; "
                f"this build accepts {CDF_API_VERSION!r}"
            )
        return value


def _validate_job_config(text: str, location: str) -> JobConfig:
    """Validate one YAML document that has already been read into memory."""
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exception:
        raise ValueError(f"invalid YAML in {location}: {exception}") from exception
    return JobConfig.model_validate(payload)


def load_job_config(
    path: str | os.PathLike[str],
    *,
    storage_client: object | None = None,
) -> JobConfig:
    """Read and validate one local or S3-hosted YAML job configuration."""
    location = os.fspath(path)
    if not file_utils.is_remote(location):
        location = str(Path(location).expanduser().resolve())
    with file_utils.open(
        location,
        encoding="utf-8",
        storage_client=storage_client,
    ) as handle:
        text = handle.read()
    return _validate_job_config(text, location)


def build_job_from_config(
    config: JobConfig,
    *,
    base_dir: str | os.PathLike[str] = ".",
) -> CDFJob:
    """Interpret validated configuration in public Python API order."""
    root = os.fspath(base_dir)
    if not file_utils.is_remote(root):
        root = str(Path(root).expanduser().resolve())
    adapter = get_domain_config(config.domain)
    job = CDFJob(domain=config.domain, random_seed=config.random_seed)

    applied_scene = adapter.apply_scene(job, config.scene, root)
    adapter.apply_appearance(job, config.appearance)
    adapter.apply_operations(
        job,
        config.operations,
        base_dir=root,
        component_metadata_source=applied_scene.component_metadata_source,
        selected_types=applied_scene.selected_types,
    )
    adapter.apply_annotations(
        job,
        config.annotations,
        component_metadata_source=applied_scene.component_metadata_source,
    )
    if config.capture is not None:
        adapter.apply_capture(job, config.capture)
    adapter.apply_postprocess(job, config.postprocess)
    return job


def load_job(
    path: str | os.PathLike[str],
    *,
    storage_client: object | None = None,
) -> tuple[JobConfig, CDFJob]:
    """Load a config and interpret it relative to its containing directory."""
    location = os.fspath(path)
    if file_utils.is_remote(location):
        base_dir = location.rsplit("/", 1)[0]
    else:
        config_path = Path(location).expanduser().resolve()
        location = str(config_path)
        base_dir = str(config_path.parent)
    # Hash the bytes actually read, not a re-serialization: a reader who holds
    # the file can check it, and the path alone does not survive a later edit
    # to the same file. Those same bytes are then parsed, rather than the
    # location being opened a second time -- otherwise an edit landing between
    # the two reads leaves config_sha256 describing a config that was never the
    # one interpreted here. Local and remote go through the same read so a
    # config served from S3 is pinned exactly like one on disk.
    with file_utils.open(location, "rb", storage_client=storage_client) as handle:
        raw = handle.read()
    config_sha256 = hashlib.sha256(raw).hexdigest()
    config = _validate_job_config(raw.decode("utf-8"), location)
    job = build_job_from_config(config, base_dir=base_dir)
    job.source = {"config_path": location, "config_sha256": config_sha256}
    return config, job


def dump_job_config_schema(
    *,
    indent: int = 2,
    domains: tuple[str, ...] = (),
) -> str:
    """Return JSON Schema for the current CDF API configuration contract.

    Args:
        indent: JSON indentation width.
        domains: Explicit domain identifiers to discover before aggregation.

    Returns:
        Generated JSON Schema.

    """
    for domain in domains:
        get_domain_config(domain)
    load_installed_domain_configs()
    if not registered_domain_configs():
        raise CDFError(
            "no CDF domain configuration plugins are registered; pass "
            "--domain <name> or install a cdf.domains plugin",
        )
    schema = JobConfig.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"urn:cdf:job-config:api-version:{CDF_API_VERSION}"
    return json.dumps(schema, indent=indent)
