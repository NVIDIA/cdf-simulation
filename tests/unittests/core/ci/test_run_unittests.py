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

"""Tests for lane-aware unittest skip reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.ci import run_unittests


class SkipPolicyTest(unittest.TestCase):
    """Verify skip ownership and policy enforcement."""

    def setUp(self) -> None:
        """Load the repository skip policy."""
        self.lanes, self.rules = run_unittests._load_policy(
            Path("tests/skip_policy.json")
        )

    def test_policy_declares_expected_active_lanes(self) -> None:
        """The first milestone lanes are declared explicitly."""
        self.assertTrue(
            {"core-linux", "usd", "imaging", "solver-linux", "windows"}.issubset(
                self.lanes,
            )
        )

    def test_group_selection_is_module_level_and_ordered(self) -> None:
        """Keep lane composition explicit while removing duplicate groups."""
        self.assertEqual(
            ("core", "usd", "imaging"),
            run_unittests._parse_groups("core,usd,core,imaging"),
        )

    def test_unknown_group_is_rejected(self) -> None:
        """Reject a lane configuration that names no test directory."""
        with self.assertRaisesRegex(ValueError, "unknown test groups"):
            run_unittests._parse_groups("core,imaginary")

    def test_cross_lane_skip_is_reported_but_allowed(self) -> None:
        """A core run may defer an owned USD skip to the USD lane."""
        classified, violations = run_unittests._classify_skips(
            [("example.test", "usd-core is not installed")],
            self.rules,
            "core-linux",
        )

        self.assertEqual([], violations)
        self.assertEqual("usd", classified[0]["owner"])
        self.assertEqual("active", classified[0]["status"])

    def test_skip_fails_in_a_disallowed_lane(self) -> None:
        """Reject a dependency skip in every lane that installs it."""
        _, violations = run_unittests._classify_skips(
            [("example.test", "usd-core is not installed")],
            self.rules,
            "windows",
        )

        self.assertEqual(1, len(violations))
        self.assertIn("unexpected skip in lane 'windows'", violations[0])

    def test_legacy_usd_reason_uses_the_same_lane_policy(self) -> None:
        """Classify the older USD reason without maintaining a duplicate rule."""
        reason = "usd-core not installed (CI installs numpy only)"
        classified, allowed_violations = run_unittests._classify_skips(
            [("example.test", reason)],
            self.rules,
            "core-linux",
        )
        _, rejected_violations = run_unittests._classify_skips(
            [("example.test", reason)],
            self.rules,
            "windows",
        )

        self.assertEqual("usd", classified[0]["owner"])
        self.assertEqual([], allowed_violations)
        self.assertEqual(1, len(rejected_violations))

    def test_installed_baseline_dependency_skips_are_unclassified(self) -> None:
        """Do not excuse missing dependencies installed in every full lane."""
        for reason in ("Pillow not installed", "PyYAML not installed"):
            with self.subTest(reason=reason):
                classified, violations = run_unittests._classify_skips(
                    [("example.test", reason)],
                    self.rules,
                    "core-linux",
                )

                self.assertEqual("unclassified", classified[0]["status"])
                self.assertEqual(1, len(violations))

    def test_unknown_skip_fails_every_lane(self) -> None:
        """An unclassified skip cannot silently enter the test suite."""
        classified, violations = run_unittests._classify_skips(
            [("example.test", "new unexplained dependency")],
            self.rules,
            "core-linux",
        )

        self.assertEqual("unclassified", classified[0]["status"])
        self.assertEqual(1, len(violations))

    def test_ambiguous_skip_is_reported_and_fails(self) -> None:
        """Overlapping policy rules cannot hide an unclear owner."""
        duplicate_rules = (self.rules[0], self.rules[0])

        classified, violations = run_unittests._classify_skips(
            [("example.test", "usd-core is not installed")],
            duplicate_rules,
            "core-linux",
        )

        self.assertEqual("ambiguous", classified[0]["status"])
        self.assertEqual(2, len(classified[0]["matching_patterns"]))
        self.assertEqual(1, len(violations))

    def test_deferred_skip_records_follow_up(self) -> None:
        """Approved fixture gaps remain visible with an explicit follow-up."""
        deferred_rules = (
            run_unittests._SkipRule(
                pattern="optional fixture absent",
                owner="developer-fixture",
                status="deferred",
                rationale="Exercise deferred policy classification.",
                allowed_lanes=frozenset({"solver-linux"}),
                follow_up="Add a repository-owned fixture.",
            ),
        )
        classified, violations = run_unittests._classify_skips(
            [("example.test", "optional fixture absent")],
            deferred_rules,
            "solver-linux",
        )

        self.assertEqual([], violations)
        self.assertEqual("developer-fixture", classified[0]["owner"])
        self.assertIn("follow_up", classified[0])

    def test_release_fixture_skips_are_deferred_in_affected_lanes(self) -> None:
        """Classify intentionally excluded release coupons without hiding gaps."""
        cases = (
            ("approved SMT coupon fixture is not shipped in the public release", "usd"),
            (
                "approved THT coupon fixture is not shipped in the public release",
                "solver-linux",
            ),
            (
                "approved polarized coupon fixture is not shipped in the public release",
                "windows",
            ),
        )

        for reason, lane in cases:
            with self.subTest(reason=reason, lane=lane):
                classified, violations = run_unittests._classify_skips(
                    [("example.test", reason)],
                    self.rules,
                    lane,
                )

                self.assertEqual([], violations)
                self.assertEqual("release-fixture-clearance", classified[0]["owner"])
                self.assertEqual("deferred", classified[0]["status"])
                self.assertIn("follow_up", classified[0])

    def test_deferred_skip_is_limited_to_relevant_lanes(self) -> None:
        """Do not let a deferred fixture excuse an unrelated lane skip."""
        deferred_rules = (
            run_unittests._SkipRule(
                pattern="optional fixture absent",
                owner="developer-fixture",
                status="deferred",
                rationale="Exercise deferred policy classification.",
                allowed_lanes=frozenset({"solver-linux"}),
                follow_up="Add a repository-owned fixture.",
            ),
        )
        _, violations = run_unittests._classify_skips(
            [("example.test", "optional fixture absent")],
            deferred_rules,
            "windows",
        )

        self.assertEqual(1, len(violations))
        self.assertIn("unexpected skip in lane 'windows'", violations[0])

    def test_invalid_active_owner_is_rejected(self) -> None:
        """An active rule must refer to a lane that CI can run."""
        policy = {
            "lanes": ["core-linux"],
            "rules": [
                {
                    "pattern": "missing tool",
                    "owner": "unknown-lane",
                    "status": "active",
                    "rationale": "invalid fixture",
                    "allowed_lanes": ["core-linux"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "undeclared active owner"):
                run_unittests._load_policy(path)

    def test_undeclared_allowed_lane_is_rejected(self) -> None:
        """Reject policy permissions for CI lanes that do not exist."""
        policy = {
            "lanes": ["core-linux"],
            "rules": [
                {
                    "pattern": "missing tool",
                    "owner": "core-linux",
                    "status": "active",
                    "rationale": "invalid lane permission",
                    "allowed_lanes": ["imaginary"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "allows undeclared lanes"):
                run_unittests._load_policy(path)

    def test_junit_dependency_is_required_only_when_requested(self) -> None:
        """A missing XML reporter produces an actionable JUnit error."""
        suite = unittest.TestSuite()
        real_import = __import__

        def _import(name, *args, **kwargs):
            if name == "xmlrunner":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("builtins.__import__", side_effect=_import),
            self.assertRaisesRegex(RuntimeError, "unittest-xml-reporting"),
        ):
            run_unittests._run_suite(
                suite,
                junit_directory=Path(directory),
                lane="core-linux",
                verbosity=0,
            )


if __name__ == "__main__":
    unittest.main()
