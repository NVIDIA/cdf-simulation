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

"""Environment overrides that record themselves as they are read.

A render is reproducible from its config only if the config holds every input.
It does not: an environment variable can override a constant compiled into the
package, change the geometry that gets generated, and leave no trace in the YAML,
the job summary, or the commit. Two runs from an identical commit and an
identical config are then not the same picture, and nothing in the record says
why.

The fix is to make reading an override the same act as declaring it. Call sites
use :func:`env_float` (or a sibling) instead of ``os.environ.get``; every read is
appended to a process-local registry with the name, the default, the raw string
and the value actually used. :func:`effective_overrides` then reports exactly the
overrides that influenced this run -- not the ones someone remembered to write
down.

Only reads that *changed* something are reported as overrides. A variable that is
unset, or set to its default, is recorded as consulted so the record can
distinguish "not overridden" from "never looked at".
"""

from __future__ import annotations

import os
import threading

_LOCK = threading.Lock()
_READS: dict[str, dict[str, object]] = {}


def _record(name: str, default: object, raw: str | None, value: object) -> None:
    """Append one override read to the process-local registry."""
    with _LOCK:
        _READS[name] = {
            "name": name,
            "default": default,
            "raw": raw,
            "value": value,
            "overridden": raw is not None and value != default,
        }


def record_override(name: str, default: object, raw: str | None, value: object) -> None:
    """Record an override whose effective value had to be validated first.

    The ``env_*`` helpers record the raw string as the effective value, which
    is right for a number or a flag and wrong for anything that must be checked
    before it can be used -- a path that has to exist, a binary that has to
    run. A caller that validates and falls back records what it actually used
    through this function, so the registry says what happened rather than what
    was asked for. ``default`` is what would have been used had the variable
    been unset, so ``overridden`` stays true only when the variable changed the
    outcome; ``raw`` keeps the attempt visible beside it.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or unusable.
        raw: The raw string read from the environment, or ``None`` if unset.
        value: The value actually used, after validation and any fallback.

    Example:

    .. code-block:: python

        reset_overrides()
        record_override("CDF_EXAMPLE_BIN", "/opt/evolver", "/missing", "/opt/evolver")
        print(effective_overrides()["CDF_EXAMPLE_BIN"]["overridden"])
        # False

    """
    _record(name, default, raw, value)


def env_str(name: str, default: str) -> str:
    """Read a string override and record the read.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or unparsable.

    Returns:
        The effective value.

    Example:

    .. code-block:: python

        root = env_str("CDF_ASSET_ROOT", "/data")
        print(isinstance(root, str))
        # True

    """
    raw = os.environ.get(name)
    value = default if raw is None else raw
    _record(name, default, raw, value)
    return value


def env_float(name: str, default: float) -> float:
    """Read a float override and record the read.

    An unparsable value falls back to the default rather than raising: a render
    should not die because a stray shell export was malformed. The raw string is
    kept in the record, so the fingerprint still shows what was attempted.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or unparsable.

    Returns:
        The effective value.

    Example:

    .. code-block:: python

        ratio = env_float("PCB_AOI_THT_HEIGHT_RATIO", 0.9)
        print(ratio)
        # 0.9

    """
    raw = os.environ.get(name)
    value = default
    if raw is not None:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = default
    _record(name, default, raw, value)
    return value


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean override and record the read.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.

    Returns:
        The effective value. ``1``, ``true``, ``yes`` and ``on`` are true;
        ``0``, ``false``, ``no`` and ``off`` are false; anything else is the
        default.

    Example:

    .. code-block:: python

        print(env_bool("CDF_STRICT", False))
        # False

    """
    raw = os.environ.get(name)
    value = default
    if raw is not None:
        lowered = raw.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            value = True
        elif lowered in ("0", "false", "no", "off"):
            value = False
    _record(name, default, raw, value)
    return value


def effective_overrides() -> dict[str, dict[str, object]]:
    """Return every override read so far, keyed by variable name.

    Returns:
        Mapping of variable name to a record carrying ``default``, ``raw``,
        ``value`` and ``overridden``.

    Example:

    .. code-block:: python

        reset_overrides()
        env_float("CDF_EXAMPLE_RATIO", 0.5)
        print(effective_overrides()["CDF_EXAMPLE_RATIO"]["overridden"])
        # False

    """
    with _LOCK:
        return {name: dict(record) for name, record in _READS.items()}


def reset_overrides() -> None:
    """Clear the registry.

    Tests need a clean registry, and a long-lived process that runs several
    jobs should not report a previous job's overrides as this job's.

    Example:

    .. code-block:: python

        reset_overrides()
        print(effective_overrides())
        # {}

    """
    with _LOCK:
        _READS.clear()
