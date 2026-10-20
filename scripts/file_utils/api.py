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

"""Backend-neutral file access API."""

from __future__ import annotations

import builtins
import contextlib
import os
from collections.abc import Iterator
from typing import IO, Any

from . import s3_storage


def is_remote(uri: str | os.PathLike[str]) -> bool:
    """Return whether a location uses a supported remote backend.

    Args:
        uri: Local path or candidate remote URI.

    Returns:
        True when the location is handled by a remote storage backend.

    Example:

    .. code-block:: python

        from scripts import file_utils

        print(file_utils.is_remote("s3://bucket/object.json"))
        # True
    """
    return s3_storage.is_s3_uri(os.fspath(uri))


@contextlib.contextmanager
def open(  # noqa: A001 - intentionally mirrors builtins.open
    uri: str | os.PathLike[str],
    mode: str = "r",
    buffering: int = -1,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
    closefd: bool = True,
    opener: Any = None,
    *,
    storage_client: Any = None,
) -> Iterator[IO[Any]]:
    """Open a local path or remote object as a managed file object.

    Remote writes are published only after the ``with`` block exits
    successfully. Local paths retain the behavior of the built-in ``open``.

    Args:
        uri: Local path or supported remote object URI.
        mode: File mode using the built-in ``open`` syntax.
        buffering: Local filesystem buffering policy.
        encoding: Text encoding, or None to use the platform default.
        errors: Text encoding error policy, or None to use the default.
        newline: Text newline translation policy.
        closefd: Whether to close the local file descriptor with the handle.
        opener: Custom local filesystem opener.
        storage_client: Optional backend client override.

    Yields:
        Open file object backed by the selected storage implementation.

    Raises:
        ValueError: If a mode or option combination is unsupported.
        FileNotFoundError: If an object opened for reading does not exist.
        FileExistsError: If an object opened exclusively already exists.
        RuntimeError: If a remote backend dependency is unavailable.

    Example:

    .. code-block:: python

        from scripts import file_utils

        with file_utils.open("metadata.json", encoding="utf-8") as handle:
            print(handle.read())
    """
    uri_text = os.fspath(uri)
    if is_remote(uri_text):
        with s3_storage.open_object(
            uri_text,
            mode=mode,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
            client=storage_client,
        ) as handle:
            yield handle
        return

    with builtins.open(
        uri_text,
        mode,
        buffering=buffering,
        encoding=encoding,
        errors=errors,
        newline=newline,
        closefd=closefd,
        opener=opener,
    ) as handle:
        yield handle


def exists(
    uri: str | os.PathLike[str],
    *,
    storage_client: Any = None,
) -> bool:
    """Return whether a local path or exact remote object exists.

    Args:
        uri: Local path or supported remote object URI.
        storage_client: Optional backend client override.

    Returns:
        True when the path or object exists.

    Raises:
        RuntimeError: If a remote backend dependency is unavailable.

    Example:

    .. code-block:: python

        from scripts import file_utils

        print(file_utils.exists("metadata.json"))
    """
    uri_text = os.fspath(uri)
    if is_remote(uri_text):
        return s3_storage.object_exists(uri_text, client=storage_client)
    return os.path.exists(uri_text)


def is_file(
    uri: str | os.PathLike[str],
    *,
    storage_client: Any = None,
) -> bool:
    """Return whether a location identifies an existing file object.

    Args:
        uri: Local path or supported remote object URI.
        storage_client: Optional backend client override.

    Returns:
        True for a local regular file or exact remote non-prefix object.

    Raises:
        ValueError: If a remote URI is malformed.
        RuntimeError: If a remote backend dependency is unavailable.

    Example:

    .. code-block:: python

        from scripts import file_utils

        print(file_utils.is_file("metadata.json"))
    """
    uri_text = os.fspath(uri)
    if is_remote(uri_text):
        return s3_storage.object_is_file(uri_text, client=storage_client)
    return os.path.isfile(uri_text)


@contextlib.contextmanager
def localize_tree(
    uri: str | os.PathLike[str],
    *,
    storage_client: Any = None,
) -> Iterator[str]:
    """Yield a local path for a local or remote input tree.

    Args:
        uri: Local path or supported remote object or prefix URI.
        storage_client: Optional backend client override.

    Yields:
        Local path corresponding to the input path or URI.

    Raises:
        FileNotFoundError: If the remote object or prefix does not exist.
        ValueError: If remote content cannot be staged safely.
        RuntimeError: If a remote backend dependency is unavailable.

    Example:

    .. code-block:: python

        from scripts import file_utils

        with file_utils.localize_tree("board.usd") as local_scene:
            print(local_scene)
    """
    uri_text = os.fspath(uri)
    if is_remote(uri_text):
        with s3_storage.localize_tree(
            uri_text,
            client=storage_client,
        ) as localized:
            yield localized
        return
    yield uri_text


@contextlib.contextmanager
def output_tree(
    uri: str | os.PathLike[str],
    *,
    storage_client: Any = None,
) -> Iterator[str]:
    """Yield a local directory and publish remote output on success.

    Args:
        uri: Local output path or supported remote prefix URI.
        storage_client: Optional backend client override.

    Yields:
        Local directory into which callers write output artifacts.

    Raises:
        RuntimeError: If a remote backend dependency is unavailable.

    Example:

    .. code-block:: python

        from pathlib import Path

        from scripts import file_utils

        with file_utils.output_tree("output") as local_output:
            Path(local_output, "result.txt").write_text("done", encoding="utf-8")
    """
    uri_text = os.fspath(uri)
    if is_remote(uri_text):
        with s3_storage.output_tree(
            uri_text,
            client=storage_client,
        ) as local_output:
            yield local_output
        return
    yield uri_text
