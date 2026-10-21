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

"""Safe, backend-neutral archive extraction helpers."""

from __future__ import annotations

import contextlib
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

_MAX_ZIP_ENTRIES = 10_000
_MAX_ZIP_UNCOMPRESSED_BYTES = 4 * 1024**3
_MAX_ZIP_COMPRESSION_RATIO = 200


def _member_destination(root: Path, name: str) -> Path:
    """Return a ZIP member destination after checking for path traversal.

    Args:
        root: Extraction directory that must contain the member.
        name: Member name recorded in the archive.

    Returns:
        Resolved destination beneath the extraction directory.

    Raises:
        ValueError: If the member would escape the extraction directory.

    """
    normalized = PurePosixPath(name.replace("\\", "/"))
    if (
        not name
        or normalized.is_absolute()
        or ".." in normalized.parts
        or (normalized.parts and ":" in normalized.parts[0])
    ):
        raise ValueError("ZIP member escapes the extraction directory: %r" % name)
    destination = root.joinpath(*normalized.parts).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            "ZIP member escapes the extraction directory: %r" % name
        ) from exc
    return destination


def _validate_members(
    root: Path,
    members: list[zipfile.ZipInfo],
) -> list[tuple[zipfile.ZipInfo, Path]]:
    """Validate archive members and return their extraction destinations.

    Args:
        root: Extraction directory that must contain every member.
        members: ZIP metadata records to validate.

    Returns:
        Pairs containing each member and its validated destination.

    Raises:
        ValueError: If the archive violates a safety or resource limit.

    """
    if len(members) > _MAX_ZIP_ENTRIES:
        raise ValueError(
            "ZIP archive contains too many entries (%d; maximum %d)"
            % (len(members), _MAX_ZIP_ENTRIES)
        )

    total_size = 0
    destinations: set[Path] = set()
    validated = []
    for member in members:
        if member.flag_bits & 0x1:
            raise ValueError(
                "encrypted ZIP members are not supported: %r" % member.filename
            )
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(
                "symbolic links are not supported in ZIP files: %r" % member.filename
            )
        total_size += member.file_size
        if total_size > _MAX_ZIP_UNCOMPRESSED_BYTES:
            raise ValueError(
                "ZIP archive expands beyond the %d-byte limit"
                % _MAX_ZIP_UNCOMPRESSED_BYTES
            )
        if (
            member.file_size
            and member.compress_size
            and member.file_size / member.compress_size > _MAX_ZIP_COMPRESSION_RATIO
        ):
            raise ValueError(
                "ZIP member exceeds the maximum compression ratio: %r" % member.filename
            )
        destination = _member_destination(root, member.filename)
        if destination in destinations:
            raise ValueError("ZIP contains duplicate member path: %r" % member.filename)
        destinations.add(destination)
        validated.append((member, destination))
    return validated


@contextlib.contextmanager
def extract_zip(path: str | Path) -> Iterator[str]:
    """Safely extract a local ZIP file and yield its temporary root directory.

    The extracted tree is removed when the context exits. Encrypted members,
    links, path traversal, excessive expansion, and suspicious compression
    ratios are rejected before any member is written.

    Args:
        path: Local ZIP file to extract.

    Yields:
        Temporary directory containing the extracted archive tree.

    Raises:
        ValueError: If the ZIP is invalid, unsafe, encrypted, or exceeds a
            resource limit.

    Example:

    .. code-block:: python

        from scripts import file_utils

        with file_utils.extract_zip("board.zip") as extracted:
            print(extracted)

    """
    archive_path = Path(path)
    with tempfile.TemporaryDirectory(
        prefix="pcb_aoi_zip_input_"
    ) as temporary_directory:
        root = Path(temporary_directory)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = _validate_members(root, archive.infolist())
                for member, destination in members:
                    if member.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        archive.open(member) as source,
                        destination.open("xb") as output,
                    ):
                        shutil.copyfileobj(source, output)
        except zipfile.BadZipFile as exc:
            raise ValueError("invalid ZIP archive: %s" % archive_path) from exc
        yield str(root)
