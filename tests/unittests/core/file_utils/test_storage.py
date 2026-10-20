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

"""Test local and S3-backed file utility behavior."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from scripts import file_utils
from scripts.file_utils import s3_storage


class _S3Error(Exception):
    """Fake boto3 error carrying an AWS-style response.

    Args:
        code: AWS error code exposed by the fake response.
    """

    def __init__(self, code):
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


class _FakeS3Client:
    """In-memory implementation of the required boto3 S3 operations.

    Args:
        objects: Initial mapping of bucket/key pairs to object bytes.
        page_size: Maximum number of keys returned by each listing request.
    """

    def __init__(self, objects=None, page_size=None):
        self.objects = dict(objects or {})
        self.page_size = page_size
        self.download_calls = []
        self.upload_calls = []

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _S3Error("404")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def download_fileobj(self, bucket, key, handle):
        try:
            handle.write(self.objects[(bucket, key)])
        except KeyError as exc:
            raise _S3Error("NoSuchKey") from exc

    def upload_fileobj(self, handle, bucket, key):
        self.objects[(bucket, key)] = handle.read()
        self.upload_calls.append((bucket, key))

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None):
        keys = sorted(
            key
            for bucket, key in self.objects
            if bucket == Bucket and key.startswith(Prefix)
        )
        start = int(ContinuationToken or 0)
        page_size = self.page_size or len(keys) or 1
        page = keys[start : start + page_size]
        next_index = start + page_size
        response = {
            "Contents": [{"Key": key} for key in page],
            "IsTruncated": next_index < len(keys),
        }
        if response["IsTruncated"]:
            response["NextContinuationToken"] = str(next_index)
        return response

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[(bucket, key)])
        self.download_calls.append((bucket, key))

    def upload_file(self, filename, bucket, key):
        self.objects[(bucket, key)] = Path(filename).read_bytes()
        self.upload_calls.append((bucket, key))


class LocalFileTests(unittest.TestCase):
    def test_remote_location_detection_is_backend_neutral(self):
        self.assertFalse(file_utils.is_remote(Path("local") / "value.txt"))
        self.assertTrue(file_utils.is_remote("s3://bucket/value.txt"))

    def test_remote_location_detection_rejects_other_locations(self):
        for value in (
            "/absolute/value.txt",
            "relative/value.txt",
            "omniverse://server/value.usd",
            "https://example.com/value.txt",
            "S3://bucket/value.txt",
            "s3:/bucket/value.txt",
            "",
        ):
            with self.subTest(value=value):
                self.assertFalse(file_utils.is_remote(value))

    def test_remote_location_detection_rejects_unsupported_types(self):
        for value in (None, 123, ["s3://bucket/value.txt"]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    file_utils.is_remote(value)

    def test_open_and_exists_delegate_to_local_filesystem(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "value.txt"
            with file_utils.open(path, "w", encoding="utf-8") as handle:
                handle.write("hello")
            self.assertTrue(file_utils.exists(path))
            self.assertTrue(file_utils.is_file(path))
            self.assertTrue(file_utils.exists(temporary_directory))
            self.assertFalse(file_utils.is_file(temporary_directory))
            with file_utils.open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "hello")

    def test_tree_contexts_pass_local_paths_through(self):
        path = os.path.join("relative", "path")
        with file_utils.localize_tree(path) as localized:
            self.assertEqual(localized, path)
        with file_utils.output_tree(path) as output:
            self.assertEqual(output, path)


class S3UriTests(unittest.TestCase):
    def test_parser_preserves_object_and_prefix_keys(self):
        self.assertEqual(
            s3_storage._parse_uri("s3://bucket/boards/board.usd"),
            ("bucket", "boards/board.usd"),
        )
        self.assertEqual(
            s3_storage._parse_uri("s3://bucket/boards/"),
            ("bucket", "boards/"),
        )
        self.assertEqual(s3_storage._parse_uri("s3://bucket"), ("bucket", ""))

    def test_parser_rejects_malformed_uris(self):
        for uri in (
            "/local/value.txt",
            "s3://",
            "s3:///value.txt",
            "s3://bucket/value.txt?versionId=1",
            "s3://bucket/value.txt#fragment",
        ):
            with self.subTest(uri=uri):
                with self.assertRaises(ValueError):
                    s3_storage._parse_uri(uri)

    def test_parser_requires_an_object_key_when_requested(self):
        for uri in ("s3://bucket", "s3://bucket/"):
            with self.subTest(uri=uri):
                with self.assertRaises(IsADirectoryError):
                    s3_storage._parse_uri(uri, require_key=True)


class S3DependencyTests(unittest.TestCase):
    def test_missing_boto3_has_an_actionable_error(self):
        with mock.patch.dict(sys.modules, {"boto3": None}):
            with self.assertRaisesRegex(RuntimeError, "boto3 is required"):
                s3_storage._client()


class S3OpenTests(unittest.TestCase):
    def test_text_read_and_write(self):
        client = _FakeS3Client({("bucket", "in.txt"): b"hello"})
        with file_utils.open(
            "s3://bucket/in.txt",
            encoding="utf-8",
            storage_client=client,
        ) as handle:
            self.assertEqual(handle.read(), "hello")
        with file_utils.open(
            "s3://bucket/out.txt",
            "w",
            encoding="utf-8",
            storage_client=client,
        ) as handle:
            handle.write("world")
        self.assertEqual(client.objects[("bucket", "out.txt")], b"world")

    def test_binary_append_downloads_then_uploads(self):
        client = _FakeS3Client({("bucket", "value.bin"): b"a"})
        with file_utils.open(
            "s3://bucket/value.bin",
            "ab",
            storage_client=client,
        ) as handle:
            handle.write(b"b")
        self.assertEqual(client.objects[("bucket", "value.bin")], b"ab")

    def test_failed_context_does_not_publish(self):
        client = _FakeS3Client()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with file_utils.open(
                "s3://bucket/out.txt",
                "w",
                storage_client=client,
            ) as handle:
                handle.write("partial")
                raise RuntimeError("failed")
        self.assertNotIn(("bucket", "out.txt"), client.objects)

    def test_exists_and_exclusive_create(self):
        client = _FakeS3Client({("bucket", "present"): b"x"})
        self.assertTrue(file_utils.exists("s3://bucket/present", storage_client=client))
        self.assertFalse(file_utils.exists("s3://bucket/missing", storage_client=client))
        with self.assertRaises(FileExistsError):
            with file_utils.open("s3://bucket/present", "x", storage_client=client):
                pass

    def test_is_file_accepts_objects_but_rejects_prefixes(self):
        client = _FakeS3Client(
            {
                ("bucket", "present.txt"): b"value",
                ("bucket", "prefix/"): b"",
            }
        )
        self.assertTrue(
            file_utils.is_file(
                "s3://bucket/present.txt",
                storage_client=client,
            )
        )
        self.assertFalse(
            file_utils.is_file(
                "s3://bucket/missing.txt",
                storage_client=client,
            )
        )
        self.assertFalse(
            file_utils.is_file(
                "s3://bucket/prefix/",
                storage_client=client,
            )
        )
        self.assertFalse(
            file_utils.is_file(
                "s3://bucket",
                storage_client=client,
            )
        )

    def test_rejects_invalid_repeated_mode(self):
        with self.assertRaises(ValueError):
            with file_utils.open(
                "s3://bucket/value",
                "rr",
                storage_client=_FakeS3Client(),
            ):
                pass


class LocalizeTreeTests(unittest.TestCase):
    def test_object_downloads_only_exact_key_and_cleans_up(self):
        client = _FakeS3Client(
            {
                ("bucket", "boards/a/board.usd"): b"usd",
                ("bucket", "boards/a/board.components.json"): b"{}",
                ("bucket", "boards/a/payload/mesh.usd"): b"mesh",
            },
            page_size=1,
        )
        with file_utils.localize_tree(
            "s3://bucket/boards/a/board.usd",
            storage_client=client,
        ) as scene:
            root = Path(scene).parent
            self.assertEqual(Path(scene).read_bytes(), b"usd")
            self.assertFalse((root / "payload" / "mesh.usd").exists())
            self.assertFalse((root / "board.components.json").exists())
            self.assertEqual(client.download_calls, [("bucket", "boards/a/board.usd")])
        self.assertFalse(root.exists())

    def test_prefix_uri_returns_the_localized_directory(self):
        client = _FakeS3Client(
            {
                ("bucket", "boards/a/board.usd"): b"usd",
                ("bucket", "boards/a/payload/mesh.usd"): b"mesh",
            }
        )
        with file_utils.localize_tree(
            "s3://bucket/boards/a/",
            storage_client=client,
        ) as localized:
            root = Path(localized)
            self.assertTrue(root.is_dir())
            self.assertEqual((root / "board.usd").read_bytes(), b"usd")
            self.assertEqual((root / "payload" / "mesh.usd").read_bytes(), b"mesh")
        self.assertFalse(root.exists())

    def test_prefix_without_trailing_slash_downloads_only_its_descendants(self):
        client = _FakeS3Client(
            {
                ("bucket", "uploads/job-123/board.usdc"): b"board",
                ("bucket", "uploads/job-123/payloads/mesh.usdc"): b"mesh",
                ("bucket", "uploads/job-1234/adjacent.usdc"): b"adjacent",
                ("bucket", "uploads/other.txt"): b"other",
            }
        )

        with file_utils.localize_tree(
            "s3://bucket/uploads/job-123",
            storage_client=client,
        ) as localized:
            root = Path(localized)
            self.assertEqual((root / "board.usdc").read_bytes(), b"board")
            self.assertEqual(
                (root / "payloads" / "mesh.usdc").read_bytes(),
                b"mesh",
            )
            self.assertFalse((root / "adjacent.usdc").exists())
            self.assertFalse((root / "other.txt").exists())

        self.assertEqual(
            client.download_calls,
            [
                ("bucket", "uploads/job-123/board.usdc"),
                ("bucket", "uploads/job-123/payloads/mesh.usdc"),
            ],
        )

    def test_exact_object_wins_when_same_name_prefix_also_exists(self):
        client = _FakeS3Client(
            {
                ("bucket", "uploads/job-123"): b"exact",
                ("bucket", "uploads/job-123/board.usdc"): b"descendant",
            }
        )

        with file_utils.localize_tree(
            "s3://bucket/uploads/job-123",
            storage_client=client,
        ) as localized:
            self.assertEqual(Path(localized).read_bytes(), b"exact")

        self.assertEqual(
            client.download_calls,
            [("bucket", "uploads/job-123")],
        )

    def test_empty_prefix_raises_file_not_found(self):
        client = _FakeS3Client({("bucket", "other/value.usd"): b"usd"})
        with self.assertRaises(FileNotFoundError):
            with file_utils.localize_tree(
                "s3://bucket/missing/",
                storage_client=client,
            ):
                pass

    def test_missing_exact_object_does_not_accept_a_sibling(self):
        client = _FakeS3Client({("bucket", "boards/a/sibling.usd"): b"usd"})
        with self.assertRaises(FileNotFoundError):
            with file_utils.localize_tree(
                "s3://bucket/boards/a/missing.usd",
                storage_client=client,
            ):
                pass

    def test_rejects_object_key_that_escapes_staging_root(self):
        client = _FakeS3Client(
            {
                ("bucket", "boards/a/board.usd"): b"usd",
                ("bucket", "boards/a/../../escape.txt"): b"bad",
            }
        )
        with self.assertRaisesRegex(ValueError, "escapes"):
            with file_utils.localize_tree(
                "s3://bucket/boards/a/",
                storage_client=client,
            ):
                pass


class ZipExtractionTests(unittest.TestCase):
    def test_s3_zip_downloads_exact_object_then_extracts(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("board.usda", "#usda 1.0")
        client = _FakeS3Client(
            {
                ("bucket", "uploads/job-123/board.zip"): buffer.getvalue(),
                ("bucket", "uploads/job-123/adjacent.txt"): b"do not download",
            }
        )

        with file_utils.localize_tree(
            "s3://bucket/uploads/job-123/board.zip",
            storage_client=client,
        ) as localized_zip:
            with file_utils.extract_zip(localized_zip) as extracted:
                self.assertEqual(
                    (Path(extracted) / "board.usda").read_text(),
                    "#usda 1.0",
                )

        self.assertEqual(
            client.download_calls,
            [("bucket", "uploads/job-123/board.zip")],
        )

    def test_extracts_tree_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "board.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("board.usda", "#usda 1.0")
                archive.writestr("payloads/mesh.usdc", b"mesh")

            with file_utils.extract_zip(archive_path) as extracted:
                root = Path(extracted)
                self.assertEqual((root / "board.usda").read_text(), "#usda 1.0")
                self.assertEqual(
                    (root / "payloads" / "mesh.usdc").read_bytes(), b"mesh"
                )
            self.assertFalse(root.exists())

    def test_rejects_path_traversal_before_extracting(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("board.usda", "safe")
                archive.writestr("../escape.usda", "unsafe")

            with self.assertRaisesRegex(ValueError, "escapes"):
                with file_utils.extract_zip(archive_path):
                    pass
            self.assertFalse((Path(temporary_directory) / "escape.usda").exists())

    def test_rejects_invalid_zip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "invalid.zip"
            archive_path.write_bytes(b"not a zip")
            with self.assertRaisesRegex(ValueError, "invalid ZIP"):
                with file_utils.extract_zip(archive_path):
                    pass


class OutputTreeTests(unittest.TestCase):
    def test_uploads_tree_on_success_and_cleans_up(self):
        client = _FakeS3Client()
        with file_utils.output_tree(
            "s3://bucket/results/run",
            storage_client=client,
        ) as output:
            output_path = Path(output)
            (output_path / "nested").mkdir()
            (output_path / "metadata.json").write_text("{}", encoding="utf-8")
            (output_path / "nested" / "image.png").write_bytes(b"png")
        self.assertEqual(client.objects[("bucket", "results/run/metadata.json")], b"{}")
        self.assertEqual(client.objects[("bucket", "results/run/nested/image.png")], b"png")
        self.assertFalse(output_path.exists())

    def test_trailing_slash_output_prefix_is_normalized(self):
        client = _FakeS3Client()
        with file_utils.output_tree(
            "s3://bucket/results/run/",
            storage_client=client,
        ) as output:
            Path(output, "artifact.txt").write_text("done", encoding="utf-8")

        self.assertEqual(
            client.objects[("bucket", "results/run/artifact.txt")],
            b"done",
        )
        self.assertNotIn(("bucket", "results/run//artifact.txt"), client.objects)

    def test_failed_context_does_not_upload(self):
        client = _FakeS3Client()
        with self.assertRaisesRegex(RuntimeError, "render failed"):
            with file_utils.output_tree(
                "s3://bucket/results",
                storage_client=client,
            ) as output:
                (Path(output) / "partial.txt").write_text("partial", encoding="utf-8")
                raise RuntimeError("render failed")
        self.assertEqual(client.upload_calls, [])


if __name__ == "__main__":
    unittest.main()
