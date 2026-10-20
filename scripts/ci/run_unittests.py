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

"""Run a unittest lane and enforce ownership of every skipped test."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

_DEFAULT_POLICY = _REPOSITORY_ROOT / "tests" / "skip_policy.json"
_GROUP_DIRECTORIES = {
    "api-parity": _REPOSITORY_ROOT / "tests" / "unittests" / "api_parity",
    "core": _REPOSITORY_ROOT / "tests" / "unittests" / "core",
    "imaging": _REPOSITORY_ROOT / "tests" / "unittests" / "imaging",
    "solver": _REPOSITORY_ROOT / "tests" / "unittests" / "solver",
    "usd": _REPOSITORY_ROOT / "tests" / "unittests" / "usd",
}
_VALID_STATUSES = frozenset({"active", "covered-external", "deferred"})


@dataclass(frozen=True)
class _SkipRule:
    """One skip-reason classification rule."""

    pattern: str
    owner: str
    status: str
    rationale: str
    allowed_lanes: frozenset[str]
    follow_up: str | None


def _load_policy(path: Path) -> tuple[frozenset[str], tuple[_SkipRule, ...]]:
    """Load and validate a skip policy file."""
    document = json.loads(path.read_text(encoding="utf-8"))
    lanes = frozenset(document["lanes"])
    rules: list[_SkipRule] = []
    for index, raw_rule in enumerate(document["rules"]):
        rule = _SkipRule(
            pattern=raw_rule["pattern"],
            owner=raw_rule["owner"],
            status=raw_rule["status"],
            rationale=raw_rule["rationale"],
            allowed_lanes=frozenset(raw_rule["allowed_lanes"]),
            follow_up=raw_rule.get("follow_up"),
        )
        if rule.status not in _VALID_STATUSES:
            msg = f"rule {index} has invalid status {rule.status!r}"
            raise ValueError(msg)
        if rule.status == "active" and rule.owner not in lanes:
            msg = f"rule {index} has undeclared active owner {rule.owner!r}"
            raise ValueError(msg)
        unknown_lanes = rule.allowed_lanes - lanes
        if unknown_lanes:
            msg = f"rule {index} allows undeclared lanes {sorted(unknown_lanes)!r}"
            raise ValueError(msg)
        re.compile(rule.pattern)
        rules.append(rule)
    return lanes, tuple(rules)


def _classify_skips(
    skipped: list[tuple[str, str]],
    rules: tuple[_SkipRule, ...],
    lane: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Classify skips and return their policy violations."""
    classified: list[dict[str, Any]] = []
    violations: list[str] = []
    for test_id, reason in skipped:
        matches = [rule for rule in rules if re.fullmatch(rule.pattern, reason)]
        if not matches:
            violations.append(f"unclassified skip: {test_id}: {reason}")
            classified.append(
                {
                    "test": test_id,
                    "reason": reason,
                    "owner": None,
                    "status": "unclassified",
                }
            )
            continue
        if len(matches) > 1:
            patterns = ", ".join(rule.pattern for rule in matches)
            violations.append(
                f"ambiguous skip: {test_id}: {reason} (matches: {patterns})"
            )
            classified.append(
                {
                    "test": test_id,
                    "reason": reason,
                    "owner": None,
                    "status": "ambiguous",
                    "matching_patterns": [rule.pattern for rule in matches],
                }
            )
            continue

        rule = matches[0]
        record = {
            "test": test_id,
            "reason": reason,
            "owner": rule.owner,
            "status": rule.status,
            "rationale": rule.rationale,
            "allowed_lanes": sorted(rule.allowed_lanes),
        }
        if rule.follow_up is not None:
            record["follow_up"] = rule.follow_up
        classified.append(record)
        if lane not in rule.allowed_lanes:
            violations.append(f"unexpected skip in lane {lane!r}: {test_id}: {reason}")
    return classified, violations


def _test_id(test: unittest.case.TestCase) -> str:
    """Return a stable identifier for a unittest result entry."""
    identifier = getattr(test, "id", None)
    return identifier() if identifier is not None else str(test)


def _build_report(
    *,
    groups: tuple[str, ...],
    lane: str,
    result: unittest.TestResult,
    classified_skips: list[dict[str, Any]],
    violations: list[str],
) -> dict[str, Any]:
    """Build the machine-readable report for a completed test lane."""
    failure_ids = [_test_id(test) for test, _ in result.failures]
    error_ids = [_test_id(test) for test, _ in result.errors]
    expected_failure_ids = [_test_id(test) for test, _ in result.expectedFailures]
    unexpected_success_ids = [_test_id(test) for test in result.unexpectedSuccesses]
    non_passing = (
        len(failure_ids)
        + len(error_ids)
        + len(classified_skips)
        + len(expected_failure_ids)
        + len(unexpected_success_ids)
    )
    owners = Counter(record["owner"] or "unclassified" for record in classified_skips)
    reasons = Counter(record["reason"] for record in classified_skips)
    successful = result.wasSuccessful() and not violations and result.testsRun > 0
    return {
        "schema_version": 1,
        "lane": lane,
        "groups": list(groups),
        "successful": successful,
        "tests": {
            "total": result.testsRun,
            "passed": max(0, result.testsRun - non_passing),
            "failures": failure_ids,
            "errors": error_ids,
            "expected_failures": expected_failure_ids,
            "unexpected_successes": unexpected_success_ids,
        },
        "skips": {
            "total": len(classified_skips),
            "by_owner": dict(sorted(owners.items())),
            "by_reason": dict(sorted(reasons.items())),
            "tests": classified_skips,
        },
        "policy_violations": violations,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, help="Owning CI lane name")
    parser.add_argument(
        "--groups",
        default="core",
        help="Comma-separated capability groups to discover",
    )
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--policy", type=Path, default=_DEFAULT_POLICY)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--junit-directory", type=Path)
    parser.add_argument("--verbosity", type=int, default=2)
    return parser.parse_args(argv)


def _parse_groups(value: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated capability-group list."""
    groups = tuple(
        dict.fromkeys(part.strip() for part in value.split(",") if part.strip())
    )
    if not groups:
        raise ValueError("at least one test group is required")
    unknown = set(groups) - _GROUP_DIRECTORIES.keys()
    if unknown:
        raise ValueError(f"unknown test groups {sorted(unknown)!r}")
    return groups


def _discover_groups(groups: tuple[str, ...], pattern: str) -> unittest.TestSuite:
    """Discover every test module in the selected capability groups."""
    suite = unittest.TestSuite()
    for group in groups:
        discovered = unittest.defaultTestLoader.discover(
            str(_GROUP_DIRECTORIES[group]),
            pattern=pattern,
            top_level_dir=str(_REPOSITORY_ROOT),
        )
        suite.addTests(discovered)
    return suite


def _run_suite(
    suite: unittest.TestSuite,
    *,
    junit_directory: Path | None,
    lane: str,
    verbosity: int,
) -> unittest.TestResult:
    """Run a test suite with optional JUnit XML output."""
    if junit_directory is None:
        return unittest.TextTestRunner(verbosity=verbosity).run(suite)

    try:
        import xmlrunner
    except ImportError as exc:
        msg = "JUnit output requires the pinned unittest-xml-reporting test dependency"
        raise RuntimeError(msg) from exc
    junit_directory.mkdir(parents=True, exist_ok=True)
    return xmlrunner.XMLTestRunner(
        output=str(junit_directory),
        outsuffix=lane,
        verbosity=verbosity,
    ).run(suite)


def main(argv: list[str] | None = None) -> int:
    """Run one unittest lane, write its skip report, and return an exit code."""
    args = _parse_args(argv)
    try:
        lanes, rules = _load_policy(args.policy)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Invalid skip policy {args.policy}: {exc}", file=sys.stderr)
        return 2
    if args.lane not in lanes:
        print(f"Unknown test lane {args.lane!r}", file=sys.stderr)
        return 2

    try:
        groups = _parse_groups(args.groups)
    except ValueError as exc:
        print(f"Invalid test groups: {exc}", file=sys.stderr)
        return 2

    suite = _discover_groups(groups, args.pattern)
    try:
        result = _run_suite(
            suite,
            junit_directory=args.junit_directory,
            lane=args.lane,
            verbosity=args.verbosity,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    skipped = [(_test_id(test), reason) for test, reason in result.skipped]
    classified, violations = _classify_skips(skipped, rules, args.lane)
    if result.testsRun == 0:
        violations.append("test discovery returned zero tests")
    report = _build_report(
        groups=groups,
        lane=args.lane,
        result=result,
        classified_skips=classified,
        violations=violations,
    )

    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\nSkip ownership summary:")
    if report["skips"]["by_owner"]:
        for owner, count in report["skips"]["by_owner"].items():
            print(f"  {owner}: {count}")
    else:
        print("  no skips")
    for violation in violations:
        print(f"POLICY VIOLATION: {violation}", file=sys.stderr)
    print(f"Skip report: {args.report_file}")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
