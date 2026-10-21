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

"""Tests for the render provenance record."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import pcb_aoi  # noqa: F401 -- registers the pcb-aoi domain
from cdf.execution.dataset_manifest import DatasetManifestObserver
from cdf.execution.job_result_types import JobOrderResult, JobResult
from cdf.provenance import env as env_module
from cdf.provenance import fingerprint


class EnvOverrideRegistryTest(unittest.TestCase):
    """A read of an override is also a declaration of it."""

    def setUp(self):
        env_module.reset_overrides()
        self.addCleanup(env_module.reset_overrides)

    def test_unset_variable_records_the_default_and_is_not_an_override(self):
        """A variable nobody set is consulted, not overridden."""
        os.environ.pop("CDF_TEST_RATIO", None)
        self.assertEqual(env_module.env_float("CDF_TEST_RATIO", 0.55), 0.55)
        record = env_module.effective_overrides()["CDF_TEST_RATIO"]
        self.assertFalse(record["overridden"])
        self.assertIsNone(record["raw"])

    def test_set_variable_is_reported_with_default_and_effective_value(self):
        """The record carries both numbers, so the delta is visible."""
        os.environ["CDF_TEST_RATIO"] = "1.5"
        self.addCleanup(os.environ.pop, "CDF_TEST_RATIO", None)
        self.assertEqual(env_module.env_float("CDF_TEST_RATIO", 0.55), 1.5)
        record = env_module.effective_overrides()["CDF_TEST_RATIO"]
        self.assertTrue(record["overridden"])
        self.assertEqual(record["default"], 0.55)
        self.assertEqual(record["value"], 1.5)

    def test_unparsable_value_falls_back_and_keeps_the_raw_string(self):
        """A malformed export must not kill a render, and must not be hidden."""
        os.environ["CDF_TEST_RATIO"] = "not-a-number"
        self.addCleanup(os.environ.pop, "CDF_TEST_RATIO", None)
        self.assertEqual(env_module.env_float("CDF_TEST_RATIO", 0.55), 0.55)
        record = env_module.effective_overrides()["CDF_TEST_RATIO"]
        self.assertEqual(record["raw"], "not-a-number")
        self.assertEqual(record["value"], 0.55)

    def test_a_validated_override_records_what_was_used_not_what_was_asked(self):
        """A path override that fails validation is recorded as the fallback.

        ``default`` is what would have run without the variable, so a variable
        that changed nothing is not reported as an override, while the raw
        string keeps the attempt visible.
        """
        env_module.record_override(
            "CDF_TEST_BIN", "/opt/fallback", "/missing", "/opt/fallback"
        )
        record = env_module.effective_overrides()["CDF_TEST_BIN"]
        self.assertEqual(record["value"], "/opt/fallback")
        self.assertEqual(record["raw"], "/missing")
        self.assertFalse(record["overridden"])
        env_module.record_override(
            "CDF_TEST_BIN", "/opt/fallback", "/opt/mine", "/opt/mine"
        )
        self.assertTrue(env_module.effective_overrides()["CDF_TEST_BIN"]["overridden"])

    def test_overrides_reach_the_record(self):
        """The whole point: the fingerprint carries what the config cannot."""
        os.environ["CDF_TEST_RATIO"] = "2.0"
        self.addCleanup(os.environ.pop, "CDF_TEST_RATIO", None)
        env_module.env_float("CDF_TEST_RATIO", 0.55)
        record = fingerprint.build(result=None, output_dir=None)
        self.assertEqual(record["environment"]["CDF_TEST_RATIO"]["value"], 2.0)


class SolverBinaryProvenanceTest(unittest.TestCase):
    """The recorded solver binary is the one that ran, not the one requested.

    ``find_evolver()`` falls back to the vendored build or ``$PATH`` when
    ``$SOLDER_EVOLVER_BIN`` names a missing file. The first version recorded
    the raw override as the effective value and only warned, so the manifest
    named a binary that never solved anything.
    """

    def setUp(self):
        try:
            from pcb_aoi.solder_evolver import backend
        except (Exception, SystemExit) as error:  # noqa: BLE001
            # The package imports manifold3d at module scope and exits when it
            # is absent (the plain unit-test image has no solver deps).
            self.skipTest(f"solder_evolver deps unavailable: {error}")
        self.backend = backend
        env_module.reset_overrides()
        self.addCleanup(env_module.reset_overrides)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.vendored = Path(self.tmp) / "evolver"
        self.vendored.write_bytes(b"")
        original = backend.vendored_evolver_dir
        backend.vendored_evolver_dir = lambda: Path(self.tmp)
        self.addCleanup(setattr, backend, "vendored_evolver_dir", original)
        prior = os.environ.pop("SOLDER_EVOLVER_BIN", None)
        if prior is not None:
            self.addCleanup(os.environ.__setitem__, "SOLDER_EVOLVER_BIN", prior)

    def test_missing_override_records_the_fallback_that_actually_ran(self):
        os.environ["SOLDER_EVOLVER_BIN"] = os.path.join(self.tmp, "not-here")
        self.addCleanup(os.environ.pop, "SOLDER_EVOLVER_BIN", None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = self.backend.find_evolver()
        self.assertEqual(found, str(self.vendored))
        self.assertTrue(any("does not exist" in str(w.message) for w in caught))
        record = env_module.effective_overrides()["SOLDER_EVOLVER_BIN"]
        self.assertEqual(record["value"], str(self.vendored))
        self.assertEqual(record["raw"], os.path.join(self.tmp, "not-here"))
        # The variable was set but changed nothing: not an override.
        self.assertFalse(record["overridden"])

    def test_valid_override_is_recorded_as_the_binary_used(self):
        override = Path(self.tmp) / "custom-evolver"
        override.write_bytes(b"")
        os.environ["SOLDER_EVOLVER_BIN"] = str(override)
        self.addCleanup(os.environ.pop, "SOLDER_EVOLVER_BIN", None)
        self.assertEqual(self.backend.find_evolver(), str(override))
        record = env_module.effective_overrides()["SOLDER_EVOLVER_BIN"]
        self.assertEqual(record["value"], str(override))
        self.assertEqual(record["default"], str(self.vendored))
        self.assertTrue(record["overridden"])

    def test_unset_variable_records_the_fallback_and_is_not_an_override(self):
        self.assertEqual(self.backend.find_evolver(), str(self.vendored))
        record = env_module.effective_overrides()["SOLDER_EVOLVER_BIN"]
        self.assertIsNone(record["raw"])
        self.assertEqual(record["value"], str(self.vendored))
        self.assertFalse(record["overridden"])


class ContainerIdentityTest(unittest.TestCase):
    """Inside a container the launcher's two variables are the only image identity."""

    def setUp(self):
        env_module.reset_overrides()
        self.addCleanup(env_module.reset_overrides)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (Path(self.tmp) / "GIT_COMMIT").write_text("0123abcd\n", encoding="utf-8")
        for name in ("CDF_CONTAINER_IMAGE", "CDF_IMAGE", "CDF_IMAGE_DIGEST"):
            prior = os.environ.pop(name, None)
            if prior is not None:
                self.addCleanup(os.environ.__setitem__, name, prior)
        original = fingerprint._package_root
        fingerprint._package_root = lambda: Path(self.tmp)
        self.addCleanup(setattr, fingerprint, "_package_root", original)
        self.digest = "sha256:" + "f" * 64
        self.tag = "nvcr.io/x/cdf-simulation:1.0.0-0123abcd.mr137"

    def test_exported_image_and_digest_make_the_identity_verified(self):
        os.environ["CDF_CONTAINER_IMAGE"] = self.tag
        os.environ["CDF_IMAGE_DIGEST"] = self.digest
        self.addCleanup(os.environ.pop, "CDF_CONTAINER_IMAGE", None)
        self.addCleanup(os.environ.pop, "CDF_IMAGE_DIGEST", None)
        record = fingerprint.build(result=None, output_dir=None)
        code = record["code"]
        self.assertEqual(code["source"], "container")
        self.assertEqual(code["commit"], "0123abcd")
        self.assertEqual(code["image"], self.tag)
        self.assertEqual(code["image_digest"], self.digest)
        self.assertFalse(
            [item for item in record["unverified"] if "image" in item.lower()],
            record["unverified"],
        )
        summary = fingerprint.summary(record)
        self.assertEqual(summary["image_digest"], self.digest)
        self.assertEqual(summary["code_source"], "container")

    def test_a_missing_digest_is_declared_not_assumed(self):
        os.environ["CDF_CONTAINER_IMAGE"] = self.tag
        self.addCleanup(os.environ.pop, "CDF_CONTAINER_IMAGE", None)
        record = fingerprint.build(result=None, output_dir=None)
        self.assertEqual(record["code"]["image"], self.tag)
        self.assertNotIn("image_digest", record["code"])
        self.assertTrue(
            any("Image digest not recorded" in item for item in record["unverified"]),
            record["unverified"],
        )


class RemoteUrlSanitizingTest(unittest.TestCase):
    """A remote URL may carry a token; the manifest ships with the dataset."""

    def test_userinfo_and_query_are_stripped_from_https_remotes(self):
        self.assertEqual(
            fingerprint._sanitize_remote_url(
                "https://user:token@gitlab.example.com/group/repo.git?x=y#frag",
            ),
            "https://gitlab.example.com/group/repo.git",
        )

    def test_a_clean_url_is_unchanged_and_the_port_survives(self):
        self.assertEqual(
            fingerprint._sanitize_remote_url("https://gitlab.example.com:8443/g/r.git"),
            "https://gitlab.example.com:8443/g/r.git",
        )

    def test_scp_like_remotes_lose_their_login(self):
        self.assertEqual(
            fingerprint._sanitize_remote_url("git@gitlab.example.com:group/repo.git"),
            "gitlab.example.com:group/repo.git",
        )
        self.assertIsNone(fingerprint._sanitize_remote_url(None))


class DirtyTreeUntrackedFilesTest(unittest.TestCase):
    """``git diff HEAD`` omits untracked files; the record must not.

    git is driven through a fake so the test runs in every CI lane (none of
    the starved images ship a git binary) and pins the collector's handling of
    the answers, not the availability of the tool.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = Path(self.tmp)
        (self.repo / ".git").mkdir()
        self.answers = {
            ("status", "--porcelain"): "?? new.py\n",
            ("rev-parse", "HEAD"): "0123abcd",
            ("rev-parse", "--abbrev-ref", "HEAD"): "main",
            ("log", "-1", "--format=%cI"): "2026-09-15T12:00:00+00:00",
            ("log", "-1", "--format=%s"): "init",
            ("remote",): "origin",
            (
                "remote",
                "get-url",
                "origin",
            ): "https://me:secret@git.example.com/g/r.git?x=1",
            ("diff", "HEAD"): "",
            ("ls-files", "--others", "--exclude-standard", "-z"): "new.py\0",
        }
        original = fingerprint._git
        fingerprint._git = lambda repo, *args: self.answers.get(tuple(args))
        self.addCleanup(setattr, fingerprint, "_git", original)

    def test_untracked_content_changes_the_fingerprint(self):
        (self.repo / "new.py").write_text("a = 1\n", encoding="utf-8")
        first = fingerprint.collect_code(self.repo)
        (self.repo / "new.py").write_text("a = 2\n", encoding="utf-8")
        second = fingerprint.collect_code(self.repo)
        self.assertTrue(first["dirty"])
        self.assertEqual(first["untracked_files"], ["new.py"])
        # The tracked diff is identical (empty) both times; only the untracked
        # digest tells the two working states apart.
        self.assertEqual(first["diff_sha256"], second["diff_sha256"])
        self.assertNotEqual(first["untracked_sha256"], second["untracked_sha256"])

    def test_over_the_cap_is_listed_and_declared_unverified(self):
        self.answers[("ls-files", "--others", "--exclude-standard", "-z")] = "big.bin\0"
        (self.repo / "big.bin").write_bytes(b"x" * 64)
        original = fingerprint._UNTRACKED_HASH_LIMIT_BYTES
        fingerprint._UNTRACKED_HASH_LIMIT_BYTES = 16
        self.addCleanup(setattr, fingerprint, "_UNTRACKED_HASH_LIMIT_BYTES", original)
        code = fingerprint.collect_code(self.repo)
        self.assertIsNone(code["untracked_sha256"])
        self.assertIn("not hashed", code["untracked_skipped"])
        self.assertEqual(code["untracked_files"], ["big.bin"])
        original_root = fingerprint._package_root
        fingerprint._package_root = lambda: self.repo
        self.addCleanup(setattr, fingerprint, "_package_root", original_root)
        record = fingerprint.build(result=None, output_dir=None)
        self.assertTrue(
            any(
                "Untracked files were not hashed" in item
                for item in record["unverified"]
            ),
            record["unverified"],
        )

    def test_remote_urls_are_recorded_without_credentials(self):
        code = fingerprint.collect_code(self.repo)
        self.assertEqual(code["remotes"], {"origin": "https://git.example.com/g/r.git"})

    def test_git_ls_files_failure_is_declared_not_ignored(self):
        del self.answers[("ls-files", "--others", "--exclude-standard", "-z")]
        code = fingerprint.collect_code(self.repo)
        self.assertIsNone(code["untracked_sha256"])
        self.assertIn("ls-files failed", code["untracked_skipped"])


class RecordContentTest(unittest.TestCase):
    """What the record must say, including what it cannot say."""

    def setUp(self):
        env_module.reset_overrides()
        self.addCleanup(env_module.reset_overrides)

    def test_missing_scene_is_reported_as_unverified_not_omitted(self):
        """Absence of a caveat must not read as absence of a problem."""
        record = fingerprint.build(result=None, output_dir=None)
        self.assertTrue(
            any("scene path" in item for item in record["unverified"]),
            record["unverified"],
        )

    def test_artifacts_are_hashed_and_the_manifest_documents_are_not(self):
        """Every delivered file is hashed; the manifest's own documents are not.

        Recording the provenance republishes ``dataset_index.json``, so a hash
        of it taken here would describe a version that no longer exists by the
        time anyone reads the record.
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "rgb_0000.png").write_bytes(b"pixels")
            (out / "dataset_index.json").write_text("{}", encoding="utf-8")
            (out / "trigger_0000").mkdir()
            (out / "trigger_0000" / "metadata.json").write_text("{}", encoding="utf-8")
            artifacts = fingerprint.collect_artifacts(out)
        names = [item["path"] for item in artifacts["files"]]
        self.assertEqual(names, ["rgb_0000.png"])
        self.assertEqual(len(artifacts["files"][0]["sha256"]), 64)

    def test_truncation_is_declared(self):
        """A silent cap reads as full coverage; say what was dropped."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for index in range(4):
                (out / f"f{index}.png").write_bytes(b"x")
            artifacts = fingerprint.collect_artifacts(out, limit=2)
        self.assertEqual(artifacts["truncated_at"], 2)
        self.assertEqual(artifacts["count"], 2)

    def test_scene_chain_surfaces_the_vendor_stamp(self):
        """createdDate is the only field that identifies a source-export revision."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "board.usda").write_text(
                "#usda 1.0\n(\n    customLayerData = {\n"
                '        string creator = "ExampleExporter"\n'
                '        string creatorVersion = "1.0.0"\n'
                '        string createdDate = "2026-01-01T00:00:00Z"\n'
                "    }\n)\n",
                encoding="utf-8",
            )
            (root / "scene.usda").write_text(
                "#usda 1.0\n(\n    subLayers = [\n        @board.usda@\n    ]\n)\n",
                encoding="utf-8",
            )
            scene = fingerprint.collect_scene(root / "scene.usda")
        self.assertEqual(len(scene["layers"]), 2)
        self.assertEqual(
            scene["source_export"]["createdDate"],
            "2026-01-01T00:00:00Z",
        )

    def test_references_and_payloads_are_followed_and_hashed(self):
        """A scene layer composes the board by reference, not only by sublayer."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "board.usdc").write_bytes(b"PXR-USDC\x00" + b"\x01" * 32)
            (root / "lib.usda").write_text("#usda 1.0\n", encoding="utf-8")
            (root / "scene.usda").write_text(
                "#usda 1.0\n(\n    subLayers = [\n        @lib.usda@\n    ]\n)\n"
                'def Xform "Board" (\n'
                "    prepend references = @board.usdc@</Board>\n"
                "    payload = [\n        @lib.usda@\n    ]\n"
                ")\n{\n}\n",
                encoding="utf-8",
            )
            scene = fingerprint.collect_scene(root / "scene.usda")
            record = fingerprint.build(
                result=None, output_dir=None, usd_path=root / "scene.usda"
            )
        by_role = {layer["role"]: layer for layer in scene["layers"]}
        self.assertEqual(set(by_role), {"entry", "sublayer", "reference"})
        self.assertEqual(len(by_role["reference"]["sha256"]), 64)
        self.assertEqual(
            by_role["reference"]["referenced_from"], str(root / "scene.usda")
        )
        # lib.usda is reached first as a sublayer; the payload arc to the same
        # file does not record it twice.
        self.assertEqual(len(scene["layers"]), 3)
        # Only the crate layer leaves a gap; nothing is missing or remote.
        gaps = [item for item in record["unverified"] if item.startswith("Layer ")]
        self.assertEqual(len(gaps), 1, gaps)
        self.assertIn("board.usdc", gaps[0])

    def test_a_missing_referenced_layer_is_declared_unverified(self):
        """An incomplete chain must not read as a verified one."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scene.usda").write_text(
                '#usda 1.0\ndef "B" (\n    references = @gone.usda@\n)\n{\n}\n',
                encoding="utf-8",
            )
            scene = fingerprint.collect_scene(root / "scene.usda")
            record = fingerprint.build(
                result=None, output_dir=None, usd_path=root / "scene.usda"
            )
        missing = [layer for layer in scene["layers"] if layer.get("missing")]
        self.assertEqual([layer["role"] for layer in missing], ["reference"])
        self.assertTrue(
            any(
                "gone.usda" in item and "missing on disk" in item
                for item in record["unverified"]
            ),
            record["unverified"],
        )

    def test_a_remote_scene_root_is_declared_not_opened(self):
        """`s3://` is not a local path; the record says so instead of probing it."""
        scene = fingerprint.collect_scene("s3://bucket/boards/scene.usda")
        record = fingerprint.build(
            result=None, output_dir=None, usd_path="s3://bucket/boards/scene.usda"
        )
        self.assertEqual(len(scene["layers"]), 1)
        self.assertTrue(scene["layers"][0]["remote"])
        self.assertNotIn("sha256", scene["layers"][0])
        self.assertTrue(
            any(
                "s3://bucket/boards/scene.usda" in item and "remote" in item
                for item in record["unverified"]
            ),
            record["unverified"],
        )

    def test_a_remote_reference_inside_a_local_layer_is_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scene.usda").write_text(
                '#usda 1.0\ndef "B" (\n    references = @s3://assets/board.usdc@</B>\n)\n{\n}\n',
                encoding="utf-8",
            )
            record = fingerprint.build(
                result=None, output_dir=None, usd_path=root / "scene.usda"
            )
        remote = [layer for layer in record["scene"]["layers"] if layer.get("remote")]
        self.assertEqual(remote[0]["path"], "s3://assets/board.usdc")
        self.assertEqual(remote[0]["role"], "reference")
        self.assertTrue(
            any("s3://assets/board.usdc" in item for item in record["unverified"]),
            record["unverified"],
        )

    def test_isaac_version_is_found_from_the_interpreter_location(self):
        """`ISAAC_SIM_PATH` is usually unset when launching via `python.bat`.

        A real verification run came back with no Isaac version at all because
        the collector only consulted the environment variable. The interpreter
        lives at ``<install>/kit/python/python.exe``, so walking up from it
        recovers the install.
        """
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            install = Path(tmp) / "isaac-sim-6.0"
            interpreter = install / "kit" / "python" / "python.exe"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_bytes(b"")
            (install / "VERSION").write_text("6.0.0-rc.20", encoding="utf-8")
            original = sys.executable
            sys.executable = str(interpreter)
            self.addCleanup(setattr, sys, "executable", original)
            # Restored, or a runner that starts with ISAAC_SIM_PATH set makes
            # later tests in the same process fail by ordering.
            _prior_isaac = os.environ.pop("ISAAC_SIM_PATH", None)
            if _prior_isaac is not None:
                self.addCleanup(os.environ.__setitem__, "ISAAC_SIM_PATH", _prior_isaac)
            record = fingerprint.collect_runtime()

        self.assertEqual(record["isaac_version"], "6.0.0-rc.20")
        self.assertEqual(record["isaac_root"], str(install))

    def test_an_unhashed_layer_is_surfaced_where_gaps_are_read(self):
        """A skipped hash is a gap, so it belongs in `unverified` too.

        It is already recorded on the layer, but a reader scanning `unverified`
        for what is not pinned would otherwise miss that a large board
        export's bytes are unpinned.
        """
        with tempfile.TemporaryDirectory() as tmp:
            board = Path(tmp) / "big.usda"
            board.write_text("#usda 1.0\n", encoding="utf-8")
            original = fingerprint._HASH_LIMIT_BYTES
            fingerprint._HASH_LIMIT_BYTES = 1
            self.addCleanup(setattr, fingerprint, "_HASH_LIMIT_BYTES", original)
            record = fingerprint.build(result=None, output_dir=None, usd_path=board)

        self.assertTrue(
            any("not hashed" in item for item in record["unverified"]),
            record["unverified"],
        )

    def test_order_outputs_are_summarised_not_dropped_and_not_dumped(self):
        """The record must answer "how many joints" without inlining every path.

        A real run left this record unable to say how much solder was generated,
        which is the number that decides whether a board is usable as training
        data. Inlining thousands of prim paths would have been the other failure.
        """
        summary = fingerprint._summarise_output(
            {
                "solver": "procedural",
                "artifact_count": 7,
                "generated_prim_paths": [f"/World/J/{i}" for i in range(400)],
            }
        )
        self.assertEqual(summary["solver"], "procedural")
        self.assertEqual(summary["artifact_count"], 7)
        self.assertEqual(summary["generated_prim_paths"]["count"], 400)
        self.assertEqual(len(summary["generated_prim_paths"]["sample"]), 3)

    def test_deeply_nested_output_collapses_instead_of_being_walked(self):
        """A handler may return something arbitrarily deep; the record may not.

        Past the depth limit each level summarises to an empty mapping, which
        reports as None rather than as an empty shell -- so a pathological
        output leaves nothing in the record instead of a misleading skeleton.
        """
        deep = {"a": {"b": {"c": {"d": {"e": 1}}}}}
        self.assertIsNone(fingerprint._summarise_output(deep))

        # A useful value at a shallow depth still survives beside a deep one.
        mixed = {"artifact_count": 7, "deep": {"a": {"b": {"c": 1}}}}
        summary = fingerprint._summarise_output(mixed)
        self.assertEqual(summary["artifact_count"], 7)
        self.assertNotIn("deep", summary)

    def test_a_failing_collector_is_reported_and_does_not_raise(self):
        """The record degrades; it does not explode."""
        original = fingerprint.collect_runtime
        fingerprint.collect_runtime = lambda: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        self.addCleanup(setattr, fingerprint, "collect_runtime", original)
        record = fingerprint.build(result=None, output_dir=None)
        self.assertIn("runtime", record["errors"])
        self.assertTrue(any("runtime" in item for item in record["unverified"]))

    def test_versions_are_recorded_beside_the_commit(self):
        """A version is what QA quotes; a commit is what reproduces. Both."""
        from cdf import CDF_API_VERSION
        from cdf.execution.dataset_manifest import SCHEMA_VERSION

        record = fingerprint.build(result=None, output_dir=None)
        versions = record["versions"]
        self.assertEqual(versions["api_version"], CDF_API_VERSION)
        self.assertEqual(versions["manifest_schema"], SCHEMA_VERSION)
        self.assertEqual(versions["fingerprint"], fingerprint.FINGERPRINT_VERSION)
        # pyproject.toml is in this checkout, so the package version resolves.
        self.assertTrue(versions["cdf"])
        self.assertIn("commit", record["code"])

    def test_a_python_built_job_says_it_has_no_config_file(self):
        """Absence is stated, not omitted: the field is present and explains."""
        record = fingerprint.build(result=None, output_dir=None, plan=None)
        self.assertEqual(record["config"]["source"], "python")

    def test_a_loaded_config_is_named_by_path_and_hash(self):
        """load_job() stamps the plan; the record reads it back from there."""
        plan = _FakePlan(
            [],
            metadata={
                "source": {
                    "config_path": "C:/jobs/board.yaml",
                    "config_sha256": "ab" * 32,
                }
            },
        )
        record = fingerprint.build(result=None, output_dir=None, plan=plan)
        self.assertEqual(record["config"]["source"], "yaml")
        self.assertEqual(record["config"]["path"], "C:/jobs/board.yaml")
        self.assertEqual(record["config"]["sha256"], "ab" * 32)

    def test_a_config_without_a_hash_is_declared_unverified(self):
        """A path names the file; only the hash pins its content."""
        plan = _FakePlan(
            [],
            metadata={
                "source": {"config_path": "s3://b/job.yaml", "config_sha256": None}
            },
        )
        record = fingerprint.build(result=None, output_dir=None, plan=plan)
        self.assertEqual(record["config"]["source"], "yaml")
        self.assertTrue(
            any("bytes were not hashed" in item for item in record["unverified"]),
            record["unverified"],
        )

    def test_summary_is_compact_and_names_the_commit(self):
        """The per-trigger summary answers which code/versions/config/board."""
        plan = _FakePlan(
            [],
            metadata={
                "source": {
                    "config_path": "job.yaml",
                    "config_sha256": "cd" * 32,
                }
            },
        )
        record = fingerprint.build(result=None, output_dir=None, plan=plan)
        summary = fingerprint.summary(record)
        self.assertEqual(summary["config_sha256"], "cd" * 32)
        self.assertEqual(summary["versions"], record["versions"])
        self.assertIn("unverified_count", summary)
        # Nothing bulky rides along with every trigger.
        self.assertNotIn("artifacts", summary)
        self.assertNotIn("environment", summary)
        self.assertNotIn("orders", summary)

    def test_static_record_is_marked_as_the_prepare_phase(self):
        """Taken before the first handler runs; artifacts deliberately empty."""
        record = fingerprint.static_record(None)
        self.assertEqual(record["phase"], "prepare")
        self.assertEqual(record["artifacts"]["count"], 0)
        self.assertEqual(record["orders"], [])

    def test_binary_layer_is_declared_not_traversed_and_unverified(self):
        """A crate layer is hashed but cannot be walked; the record must say so.

        Without this, a `.usdc` scene produced a one-entry chain that read as a
        flat scene with no source-export stamp, weaker than the record claims to be.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "board.usdc").write_bytes(b"PXR-USDC\x00" + b"\x01" * 64)
            scene = fingerprint.collect_scene(root / "board.usdc")
            record = fingerprint.build(
                result=None,
                output_dir=None,
                usd_path=root / "board.usdc",
            )
        self.assertEqual(len(scene["layers"]), 1)
        layer = scene["layers"][0]
        self.assertEqual(len(layer["sha256"]), 64)
        self.assertIn("not_traversed", layer)
        self.assertNotIn("source_export", scene)
        self.assertTrue(
            any(
                "binary USD layer" in item and "board.usdc" in item
                for item in record["unverified"]
            ),
            record["unverified"],
        )

    def test_summary_identity_fields_are_stable_across_phases(self):
        """Trigger manifests are never rewritten, so their snapshot must agree
        with the final index on everything that identifies the run."""
        plan = _FakePlan(
            [],
            metadata={
                "source": {
                    "config_path": "job.yaml",
                    "config_sha256": "cd" * 32,
                }
            },
        )
        static = fingerprint.summary(fingerprint.static_record(plan))
        final = fingerprint.summary(
            fingerprint.build(result=None, output_dir=None, plan=plan),
        )
        self.assertEqual(static.pop("phase"), "prepare")
        self.assertEqual(final.pop("phase"), "complete")
        static.pop("unverified_count", None)
        final.pop("unverified_count", None)
        self.assertEqual(static, final)


class _FakeOrder:
    """Minimal stand-in for a compiled order, carrying only what is read."""

    def __init__(self, order_type, parameters):
        self.order_type = order_type
        self.order_id = order_type
        self.parameters = parameters


class _FakePlan:
    """Minimal stand-in for a compiled plan."""

    def __init__(self, orders, metadata=None):
        self.orders = tuple(orders)
        self.metadata = dict(metadata or {})


class _FakeContext:
    """Minimal stand-in for IsaacExecutionContext: only the manifest is read."""

    def __init__(self):
        self.manifest = DatasetManifestObserver()


class PlanDerivationTest(unittest.TestCase):
    """The plan, not the job, is what survives into an Isaac child process."""

    def test_scene_and_artifact_root_come_from_the_plan(self):
        """Both facts are transport-safe, so a subprocess run can record them."""
        plan = _FakePlan(
            [
                _FakeOrder("scene.load", {"usd_path": "board.usda"}),
                _FakeOrder("capture.render", {"output_dir": os.path.join("OUT", "a")}),
                _FakeOrder("capture.render", {"output_dir": os.path.join("OUT", "b")}),
            ],
        )
        usd_path, output_root = fingerprint.plan_paths(plan)
        self.assertEqual(usd_path, "board.usda")
        self.assertEqual(output_root, "OUT")

    def test_a_single_pass_files_inside_its_own_directory(self):
        """The root returned is never above a directory the caller named.

        Taking the parent of a lone output directory was tried and reverted: a
        caller who passes their final destination straight through would have
        had the record written outside it. Filing one level deeper than a
        multi-pass run is the accepted cost.
        """
        plan = _FakePlan(
            [_FakeOrder("capture.render", {"output_dir": os.path.join("OUT", "only")})],
        )
        self.assertEqual(
            fingerprint.plan_paths(plan)[1],
            os.path.join("OUT", "only"),
        )

    def test_passes_sharing_only_the_filesystem_root_do_not_widen_the_walk(self):
        """`/alpha/run` and `/beta/run` have `/` in common; walking `/` would
        hash the whole machine into this render's record."""
        first = os.path.join(os.sep, "alpha", "run")
        plan = _FakePlan(
            [
                _FakeOrder("capture.rgb.render", {"output_dir": first}),
                _FakeOrder(
                    "capture.depth.render",
                    {"output_dir": os.path.join(os.sep, "beta", "run")},
                ),
            ],
        )
        _, root, notes = fingerprint.plan_paths_detail(plan)
        self.assertEqual(root, first)
        self.assertEqual(len(notes), 1)
        self.assertIn("filesystem root", notes[0])
        record = fingerprint.record_for_plan(plan, None)
        self.assertTrue(
            any(
                "filesystem root" in item and "beta" in item
                for item in record["unverified"]
            ),
            record["unverified"],
        )

    def test_passes_under_the_declared_output_root_are_accepted(self):
        plan = _FakePlan(
            [
                _FakeOrder(
                    "capture.rgb.render", {"output_dir": os.path.join("OUT", "rgb")}
                ),
                _FakeOrder(
                    "capture.depth.render", {"output_dir": os.path.join("OUT", "depth")}
                ),
            ],
            metadata={"output_root": "OUT"},
        )
        self.assertEqual(fingerprint.plan_paths_detail(plan), (None, "OUT", []))

    def test_a_common_path_outside_the_declared_root_is_rejected(self):
        first = os.path.join("ELSEWHERE", "rgb")
        plan = _FakePlan(
            [
                _FakeOrder("capture.rgb.render", {"output_dir": first}),
                _FakeOrder(
                    "capture.depth.render",
                    {"output_dir": os.path.join("ELSEWHERE", "depth")},
                ),
            ],
            metadata={"output_root": "OUT"},
        )
        _, root, notes = fingerprint.plan_paths_detail(plan)
        self.assertEqual(root, first)
        self.assertIn("outside the run output root", notes[0])
        # The static record, taken before the run, carries the same caveat.
        static = fingerprint.static_record(plan)
        self.assertTrue(
            any("outside the run output root" in item for item in static["unverified"])
        )

    def test_a_plan_with_no_capture_still_gets_a_record(self):
        """Serialization-only runs have no artifacts, but they have a code and a scene.

        With no manifest to file it in, the record is returned and nothing is
        written anywhere -- provenance never creates a file of its own.
        """
        plan = _FakePlan([_FakeOrder("scene.load", {"usd_path": "board.usda"})])
        self.assertEqual(fingerprint.plan_paths(plan), ("board.usda", None))
        record = fingerprint.record_for_plan(plan, None)
        self.assertEqual(record["artifacts"]["count"], 0)
        self.assertEqual(record["scene"]["entry"], "board.usda")
        self.assertEqual(record["phase"], "complete")


class ExecutorHookTest(unittest.TestCase):
    """Every execution path converges on the executor, so the hook lives there."""

    def setUp(self):
        env_module.reset_overrides()
        self.addCleanup(env_module.reset_overrides)

    def test_both_entry_points_reach_the_same_executor(self):
        """`render` serializes into a child that runs this same executor.

        Hooking the authoring section would have covered `render_async` only,
        and missed the synchronous `render` used by every shipped example and by
        the YAML config path. This pins the convergence that makes one hook
        sufficient.
        """
        from cdf.execution.isaac import app_runner, in_process_executor

        self.assertTrue(hasattr(in_process_executor, "_record_provenance"))
        source = Path(app_runner.__file__).read_text(encoding="utf-8")
        self.assertIn("create_default_isaac_executor().execute_async", source)

    def _executor_plan(self, tmp):
        """A one-order plan whose handler writes a file, plus its output dir."""
        from cdf.execution.job_order_types import JobOrder, JobOrderPlan

        out = Path(tmp) / "out"
        out.mkdir()
        (out / "rgb_0000.png").write_bytes(b"pixels")
        plan = JobOrderPlan(
            orders=(
                JobOrder(
                    order_id="capture.render",
                    order_type="capture.render",
                    parameters={"output_dir": str(out)},
                ),
            ),
        )
        return plan, out

    def test_the_synchronous_execute_path_records(self):
        """`execute` and `execute_async` are separate methods with separate returns.

        The first version of this hook was added to one of them only. Every unit
        test still passed, because they called the collector directly, and a real
        render then produced no record at all. Both paths are pinned here.
        """
        from cdf.execution.isaac.in_process_executor import IsaacInProcessExecutor

        with tempfile.TemporaryDirectory() as tmp:
            plan, out = self._executor_plan(tmp)
            context = _FakeContext()
            executor = IsaacInProcessExecutor(
                {"capture.render": lambda order, context: {"ok": True}},
                context=context,
            )
            executor.execute(plan)
            provenance = context.manifest.index_document()["provenance"]
            # No sibling file: the manifest is the only record.
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["rgb_0000.png"])
        self.assertEqual(provenance["phase"], "complete")
        self.assertTrue(provenance["success"])
        self.assertEqual(provenance["orders"][0]["order_id"], "capture.render")

    def test_the_asynchronous_execute_path_records(self):
        """The path a real Isaac render actually takes."""
        from cdf.execution.isaac.in_process_executor import IsaacInProcessExecutor

        with tempfile.TemporaryDirectory() as tmp:
            plan, out = self._executor_plan(tmp)
            context = _FakeContext()
            executor = IsaacInProcessExecutor(
                {"capture.render": lambda order, context: {"ok": True}},
                context=context,
            )
            asyncio.run(executor.execute_async(plan))
            provenance = context.manifest.index_document()["provenance"]
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["rgb_0000.png"])
        self.assertEqual(provenance["phase"], "complete")
        self.assertTrue(provenance["success"])

    def test_a_context_without_a_manifest_does_not_break_the_run(self):
        """A foreign context is tolerated: the run completes, nothing is written."""
        from cdf.execution.isaac.in_process_executor import IsaacInProcessExecutor

        with tempfile.TemporaryDirectory() as tmp:
            plan, out = self._executor_plan(tmp)
            executor = IsaacInProcessExecutor(
                {"capture.render": lambda order, context: {"ok": True}},
            )
            result = executor.execute(plan)
            self.assertTrue(result.success)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["rgb_0000.png"])

    def test_record_for_plan_files_the_record_in_the_manifest(self):
        """One executed plan, one record, filed where the run's facts live."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "rgb_0000.png").write_bytes(b"pixels")
            board = Path(tmp) / "board.usda"
            board.write_text("#usda 1.0\n", encoding="utf-8")
            plan = _FakePlan(
                [
                    _FakeOrder("scene.load", {"usd_path": str(board)}),
                    _FakeOrder("capture.render", {"output_dir": str(out)}),
                ],
            )
            result = JobResult(
                success=True,
                order_results=(
                    JobOrderResult(
                        order_id="capture.render",
                        order_type="capture.render",
                        status="succeeded",
                    ),
                ),
            )
            observer = DatasetManifestObserver()
            record = fingerprint.record_for_plan(plan, result, manifest=observer)
            # Nothing beside the artifacts: the manifest is the sink.
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["rgb_0000.png"])

        self.assertEqual(record["fingerprint_version"], fingerprint.FINGERPRINT_VERSION)
        self.assertTrue(record["success"])
        self.assertEqual(record["orders"][0]["order_id"], "capture.render")
        self.assertEqual(record["scene"]["entry"], str(board))
        self.assertEqual(record["artifacts"]["count"], 1)
        filed = observer.index_document()["provenance"]
        self.assertEqual(filed["artifacts"]["count"], 1)
        # In the CI image there is no git binary, so a checkout's commit reads as
        # None and the summary omits the key; compare with .get on both sides.
        self.assertEqual(
            observer.provenance_summary().get("commit"),
            record["code"].get("commit"),
        )

    def test_a_failed_run_is_still_recorded(self):
        """A failed render is when provenance is most wanted."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plan = _FakePlan([_FakeOrder("capture.render", {"output_dir": str(out)})])
            partial = JobResult(
                success=False,
                order_results=(
                    JobOrderResult(
                        order_id="capture.render",
                        order_type="capture.render",
                        status="failed",
                        error="RuntimeError: boom",
                    ),
                ),
            )
            record = fingerprint.record_for_plan(plan, partial)
        self.assertFalse(record["success"])
        self.assertEqual(record["orders"][0]["status"], "failed")


class RemoteOutputTest(unittest.TestCase):
    """A remote destination must not be reported as an empty local directory."""

    def test_remote_output_is_declared_unverified_not_empty(self):
        # Path("s3://bucket/run") is a relative local path that does not exist,
        # so collect_artifacts() returned count 0 -- which reads as "the render
        # wrote nothing" rather than "this was not checked".
        record = fingerprint.build(result=None, output_dir="s3://bucket/run")

        self.assertEqual(record["artifacts"], {"count": 0, "files": []})
        self.assertTrue(
            any("remote" in line for line in record["unverified"]),
            record["unverified"],
        )

    def test_two_passes_under_one_s3_root_stay_remote(self):
        """os.path.commonpath turned `s3://b/run/{a,b}` into `s3:/b/run`.

        That is no longer a URI, so is_remote() said no and the record reported
        an empty LOCAL artifact directory for a render that wrote to S3.
        """
        plan = _FakePlan(
            [
                _FakeOrder("capture.render", {"output_dir": "s3://bucket/run/rgb"}),
                _FakeOrder("capture.render", {"output_dir": "s3://bucket/run/depth"}),
            ],
        )
        _, output_root = fingerprint.plan_paths(plan)
        self.assertEqual(output_root, "s3://bucket/run")
        record = fingerprint.record_for_plan(plan, None)
        self.assertEqual(record["artifacts"], {"count": 0, "files": []})
        self.assertTrue(
            any(
                "remote" in line and "s3://bucket/run" in line
                for line in record["unverified"]
            ),
            record["unverified"],
        )

    def test_passes_in_different_buckets_fall_back_to_the_first(self):
        plan = _FakePlan(
            [
                _FakeOrder("capture.render", {"output_dir": "s3://one/run"}),
                _FakeOrder("capture.render", {"output_dir": "s3://two/run"}),
            ],
        )
        self.assertEqual(fingerprint.plan_paths(plan)[1], "s3://one/run")

    def test_local_output_is_still_enumerated(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "rgb_0000.png").write_bytes(b"x")

            record = fingerprint.build(result=None, output_dir=directory)

            self.assertEqual(record["artifacts"]["count"], 1)
            self.assertFalse(
                any("remote" in line for line in record["unverified"]),
                record["unverified"],
            )


if __name__ == "__main__":
    unittest.main()
