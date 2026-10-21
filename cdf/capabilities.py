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

"""Expose capabilities contributed by registered CDF domain plugins."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator, Sequence

from cdf.capability_registry import (
    Capability,
    get_capabilities,
    get_capability_documentation,
)
from cdf.domain import get_domain
from cdf.errors import CDFError

__all__ = ["CAPABILITIES", "Capability", "to_markdown"]


class _RegisteredCapabilities(Sequence[Capability]):
    """Provide a live view of capabilities registered by domain plugins."""

    def __getitem__(self, index: int | slice) -> Capability | tuple[Capability, ...]:
        """Return one capability or a slice from the current registry."""
        return get_capabilities()[index]

    def __iter__(self) -> Iterator[Capability]:
        """Iterate over a snapshot of the current registry."""
        return iter(get_capabilities())

    def __len__(self) -> int:
        """Return the number of currently registered capabilities."""
        return len(get_capabilities())


CAPABILITIES: Sequence[Capability] = _RegisteredCapabilities()


def to_markdown(*, domains: Iterable[str] | None = None) -> str:
    """Render the registered capabilities as a Markdown matrix.

    Args:
        domains: Domain identifiers to include, or all installed domains.

    Returns:
        Generated capability documentation.

    Example:
        >>> to_markdown().startswith("# CDF capability matrix")
        True

    """
    selected_domains = None if domains is None else tuple(dict.fromkeys(domains))
    if selected_domains is not None:
        for domain in selected_domains:
            get_domain(domain)
    records = get_capabilities(
        domains=selected_domains,
        load_plugins=selected_domains is None,
    )
    if not records:
        raise CDFError(
            "no CDF capabilities are registered; pass --domain <name> or "
            "install a cdf.domains plugin",
        )
    documentation = get_capability_documentation(
        domains=selected_domains,
        load_plugins=False,
    )
    lines = [
        "# CDF capability matrix",
        "",
        (
            "Generated from registered `cdf.domains` plugins "
            "(`python -m cdf.capabilities --domain pcb-aoi > "
            "cdf/CAPABILITIES.md`) — do not "
            "edit by hand; unit tests enforce freshness and binding validity."
        ),
        "",
        (
            "Five surfaces per capability: YAML config, Python API (`cdf`), "
            "skill intent (natural language), preview guidance, and the "
            "repository documentation that defines the behavior. Batch "
            "execution is authoritative for rendered datasets."
        ),
        "",
        (
            "| Capability | Config (YAML) | Python API | Skill trigger | "
            "Preview (verify) | Modes |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for capability in records:
        lines.append(
            "| {} (`{}`) | {} | {} | {} | {} | {} |".format(
                capability.title,
                capability.id,
                f"`{capability.config}`" if capability.config else "—",
                f"`job.{capability.api}(...)`" if capability.api else "—",
                f"“{capability.skill}”" if capability.skill else "—",
                capability.preview or "—",
                ", ".join(sorted(capability.modes)),
            )
        )
    lines.append("")
    for block in documentation:
        lines.extend((block, ""))
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse capability-documentation command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="domain to include in the matrix; may be repeated",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate a capability matrix for installed or explicit domains."""
    args = _parse_args(argv)
    domains = tuple(args.domain) if args.domain else None
    try:
        print(to_markdown(domains=domains), end="")
        return 0
    except CDFError as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
