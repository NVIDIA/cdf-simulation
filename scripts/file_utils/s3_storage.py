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

"""Implement S3-backed file and tree operations."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlparse

_SCHEME = "s3://"


def is_s3_uri(value: object) -> bool:
    """Return whether a value is an S3 URI.

    Args:
        value: Candidate path or URI.

    Returns:
        True when the value is a string beginning with ``s3://``.
    """
    return isinstance(value, str) and value.startswith(_SCHEME)


def _parse_uri(uri: str, *, require_key: bool = False) -> tuple[str, str]:
    """Split an S3 URI into its bucket and object key.

    Args:
        uri: S3 URI to parse.
        require_key: Require the URI to identify an object rather than a bucket.

    Returns:
        Bucket name and object key.

    Raises:
        ValueError: If the URI is malformed or includes a query or fragment.
        IsADirectoryError: If an object key is required but absent.
    """
    if not is_s3_uri(uri):
        raise ValueError("not an s3:// URI: %r" % uri)
    parsed = urlparse(uri)
    if not parsed.netloc:
        raise ValueError("S3 URI is missing a bucket: %r" % uri)
    if parsed.query or parsed.fragment:
        raise ValueError("S3 URI must not contain a query or fragment: %r" % uri)
    key = parsed.path.lstrip("/")
    if require_key and not key:
        raise IsADirectoryError("S3 URI identifies a bucket, not an object: %s" % uri)
    return parsed.netloc, key


def _client(client: Any = None) -> Any:
    """Return an injected or default boto3 S3 client.

    Args:
        client: Existing client to reuse, or None to create one.

    Returns:
        Client implementing the boto3 S3 operations used by this module.

    Raises:
        RuntimeError: If boto3 is unavailable when a client must be created.
    """
    if client is not None:
        return client
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for s3:// paths; install project requirements "
            "or use a local filesystem path"
        ) from exc
    return boto3.client("s3")


def _error_code(exc: BaseException) -> str | None:
    """Extract an AWS error code from an exception when available.

    Args:
        exc: Exception raised by an AWS client operation.

    Returns:
        Error code, or None when the exception has no AWS error response.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    if not isinstance(error, dict):
        return None
    code = error.get("Code")
    return str(code) if code is not None else None


def _is_not_found(exc: BaseException) -> bool:
    """Return whether an exception represents a missing S3 resource.

    Args:
        exc: Exception raised by an S3 client operation.

    Returns:
        True when the exception carries a recognized not-found error code.
    """
    return _error_code(exc) in {"404", "NoSuchBucket", "NoSuchKey", "NotFound"}


def _validate_mode(mode: str) -> tuple[bool, str]:
    """Validate an S3 object open mode.

    Args:
        mode: File mode using the built-in ``open`` syntax.

    Returns:
        Whether the mode permits writes and its primary action character.

    Raises:
        TypeError: If ``mode`` is not a string.
        ValueError: If ``mode`` contains an unsupported combination.
    """
    if not isinstance(mode, str):
        raise TypeError("mode must be a string")
    if set(mode) - set("rwaxbt+"):
        raise ValueError("invalid mode: %r" % mode)
    actions = [action for action in "rwax" if action in mode]
    if (
        len(actions) != 1
        or sum(mode.count(action) for action in "rwax") != 1
        or mode.count("+") > 1
        or mode.count("b") > 1
        or mode.count("t") > 1
    ):
        raise ValueError("invalid mode: %r" % mode)
    if "b" in mode and "t" in mode:
        raise ValueError("can't have text and binary mode at once")
    action = actions[0]
    writable = action in "wax" or "+" in mode
    return writable, action


@contextlib.contextmanager
def open_object(
    uri: str,
    mode: str = "r",
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
    closefd: bool = True,
    opener: Any = None,
    *,
    client: Any = None,
) -> Iterator[IO[Any]]:
    """Open an S3 object as a context-managed file object.

    Writes are buffered locally and uploaded only after the context exits
    successfully. Append and update modes download the existing object first.

    Args:
        uri: S3 object URI.
        mode: File mode using the built-in ``open`` syntax.
        encoding: Text encoding, or None to use the platform default.
        errors: Text encoding error policy, or None to use the default.
        newline: Text newline translation policy.
        closefd: Accepted for compatibility with the backend-neutral API.
        opener: Accepted for compatibility with the backend-neutral API.
        client: Existing S3 client to reuse, or None to create one lazily.

    Yields:
        File object backed by a temporary local buffer.

    Raises:
        ValueError: If a mode or option combination is unsupported.
        FileNotFoundError: If an object opened for reading does not exist.
        FileExistsError: If an object opened exclusively already exists.
    """
    writable, action = _validate_mode(mode)
    binary = "b" in mode
    if binary and any(value is not None for value in (encoding, errors, newline)):
        raise ValueError("binary mode doesn't take an encoding, errors, or newline argument")
    if not closefd or opener is not None:
        raise ValueError("closefd=False and opener are not supported for S3 objects")

    s3_client = _client(client)
    bucket, key = _parse_uri(uri, require_key=True)
    if action == "x" and object_exists(uri, client=s3_client):
        raise FileExistsError(uri)

    # `TemporaryFile` implements the complete buffered binary IO interface on
    # every supported Python version, including the methods `TextIOWrapper`
    # requires. It also keeps large S3 objects out of process memory.
    buffer = tempfile.TemporaryFile(mode="w+b")
    text_handle = None
    try:
        if action in {"r", "a"}:
            try:
                s3_client.download_fileobj(bucket, key, buffer)
            except Exception as exc:
                if action == "a" and _is_not_found(exc):
                    pass
                elif _is_not_found(exc):
                    raise FileNotFoundError(uri) from exc
                else:
                    raise
        buffer.seek(0, os.SEEK_END if action == "a" else os.SEEK_SET)

        handle: IO[Any]
        if binary:
            handle = buffer
        else:
            text_handle = io.TextIOWrapper(
                buffer,
                encoding=encoding,
                errors=errors,
                newline=newline,
            )
            handle = text_handle

        try:
            yield handle
        except BaseException:
            raise
        else:
            handle.flush()
            if writable:
                buffer.seek(0)
                s3_client.upload_fileobj(buffer, bucket, key)
    finally:
        if text_handle is not None:
            try:
                text_handle.detach()
            except (ValueError, OSError):
                pass
        buffer.close()


def object_exists(uri: str, *, client: Any = None) -> bool:
    """Return whether an exact S3 object exists.

    Args:
        uri: Exact S3 object URI.
        client: Existing S3 client to reuse, or None to create one lazily.

    Returns:
        True when the object exists.
    """
    s3_client = _client(client)
    bucket, key = _parse_uri(uri, require_key=True)
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _is_not_found(exc):
            return False
        raise
    return True


def object_is_file(uri: str, *, client: Any = None) -> bool:
    """Return whether an S3 URI identifies an exact non-prefix object.

    Args:
        uri: S3 object or prefix URI.
        client: Existing S3 client to reuse, or None to create one lazily.

    Returns:
        True when the URI identifies an existing non-prefix object.

    Raises:
        ValueError: If the URI is malformed.
        RuntimeError: If boto3 is unavailable when a client must be created.
    """
    _bucket, key = _parse_uri(uri)
    if not key or key.endswith("/"):
        return False
    return object_exists(uri, client=client)


def _list_keys(client: Any, bucket: str, prefix: str) -> list[str]:
    """List non-directory S3 object keys under a prefix.

    Args:
        client: S3 client used to issue listing requests.
        bucket: Bucket containing the objects.
        prefix: Object-key prefix to list.

    Returns:
        Object keys collected from all result pages.
    """
    keys: list[str] = []
    token = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token is not None:
            request["ContinuationToken"] = token
        response = client.list_objects_v2(**request)
        for entry in response.get("Contents") or ():
            key = entry.get("Key")
            if isinstance(key, str) and key and not key.endswith("/"):
                keys.append(key)
        token = response.get("NextContinuationToken") if response.get("IsTruncated") else None
        if token is None:
            return keys


def _safe_destination(root: Path, relative_key: str) -> Path:
    """Resolve an object key beneath a local staging root.

    Args:
        root: Local staging directory.
        relative_key: Object key relative to the staged S3 prefix.

    Returns:
        Resolved destination beneath ``root``.

    Raises:
        ValueError: If the key resolves outside the staging directory.
    """
    root = root.resolve()
    destination = root.joinpath(*relative_key.split("/")).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "S3 object key escapes the local staging directory: %r" % relative_key
        ) from exc
    return destination


@contextlib.contextmanager
def localize_tree(uri: str, *, client: Any = None) -> Iterator[str]:
    """Download an S3 object or prefix tree and yield its local path.

    An existing object URI downloads only that exact object. A URI ending in
    ``/`` downloads every object beneath that prefix. If a URI without a
    trailing slash is not an exact object, it is treated as a prefix when keys
    exist beneath ``<uri>/``.

    Args:
        uri: S3 object or prefix URI.
        client: Existing S3 client to reuse, or None to create one lazily.

    Yields:
        Temporary local object path or prefix directory.

    Raises:
        FileNotFoundError: If the object or prefix does not exist.
        ValueError: If an object key would escape the staging directory.
    """
    s3_client = _client(client)
    bucket, key = _parse_uri(uri)
    explicit_prefix = not key or key.endswith("/")

    with tempfile.TemporaryDirectory(prefix="pcb_aoi_s3_input_") as temporary_directory:
        root = Path(temporary_directory)
        if explicit_prefix:
            prefix = key
            keys = _list_keys(s3_client, bucket, prefix)
            if not keys:
                raise FileNotFoundError(uri)
            for object_key in keys:
                relative_key = object_key[len(prefix) :] if prefix else object_key
                destination = _safe_destination(root, relative_key)
                destination.parent.mkdir(parents=True, exist_ok=True)
                s3_client.download_file(bucket, object_key, str(destination))
            yield str(root)
        elif object_exists(uri, client=s3_client):
            destination = _safe_destination(root, key.rsplit("/", 1)[-1])
            s3_client.download_file(bucket, key, str(destination))
            yield str(destination)
        else:
            prefix = key.rstrip("/") + "/"
            keys = _list_keys(s3_client, bucket, prefix)
            if not keys:
                raise FileNotFoundError(uri)
            for object_key in keys:
                relative_key = object_key[len(prefix) :]
                destination = _safe_destination(root, relative_key)
                destination.parent.mkdir(parents=True, exist_ok=True)
                s3_client.download_file(bucket, object_key, str(destination))
            yield str(root)


def _upload_tree(client: Any, local_directory: str, bucket: str, prefix: str) -> None:
    """Upload every regular file beneath a local directory.

    Args:
        client: S3 client used to upload files.
        local_directory: Root directory containing output artifacts.
        bucket: Destination S3 bucket.
        prefix: Destination object-key prefix.
    """
    for root, _directories, files in os.walk(local_directory):
        for filename in sorted(files):
            local_path = os.path.join(root, filename)
            relative = os.path.relpath(local_path, local_directory).replace(os.sep, "/")
            object_key = "%s/%s" % (prefix, relative) if prefix else relative
            client.upload_file(local_path, bucket, object_key)


@contextlib.contextmanager
def output_tree(uri: str, *, client: Any = None) -> Iterator[str]:
    """Yield a temporary output directory and publish it to S3 on success.

    Args:
        uri: Destination S3 prefix URI.
        client: Existing S3 client to reuse, or None to create one lazily.

    Yields:
        Temporary local directory for output artifacts.
    """
    s3_client = _client(client)
    bucket, key = _parse_uri(uri)
    prefix = key.rstrip("/")
    with tempfile.TemporaryDirectory(prefix="pcb_aoi_s3_output_") as temporary_directory:
        yield temporary_directory
        _upload_tree(s3_client, temporary_directory, bucket, prefix)
