# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and verify source archives for the shipped Python runtime.

The lock file is deliberately independent of pip's wheel cache.  Release
images install wheels for runtime use and carry the corresponding, immutable
source archives under ``/usr/share/oss-sources/python`` for OSS compliance.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
from pathlib import Path
import re
import shutil
import time
from urllib.request import Request, urlopen


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lock(path: Path) -> list[tuple[str, str, str, str, str, bool]]:
    rows: list[tuple[str, str, str, str, str, bool]] = []
    filenames: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) not in (5, 6):
            raise ValueError(f"{path}:{line_number}: expected five or six tab-separated fields")
        component, version, filename, expected_hash, url = fields[:5]
        supplemental = len(fields) == 6 and fields[5] == "supplemental"
        if len(fields) == 6 and not supplemental:
            raise ValueError(f"{path}:{line_number}: unknown source-row qualifier {fields[5]!r}")
        if Path(filename).name != filename or filename in filenames:
            raise ValueError(f"{path}:{line_number}: unsafe or duplicate filename {filename!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"{path}:{line_number}: invalid SHA-256 for {filename}")
        filenames.add(filename)
        rows.append((component, version, filename, expected_hash, url, supplemental))
    if not rows:
        raise ValueError(f"{path}: no source archives listed")
    return rows


def download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "paidf-simulation-source-collector/1"})
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            partial.replace(destination)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = read_lock(args.lock)
    installed = {canonical_name(dist.metadata["Name"]): dist.version for dist in metadata.distributions()}
    mismatches = []
    for component, version, *_, supplemental in rows:
        if supplemental:
            continue
        actual = installed.get(canonical_name(component))
        if actual != version:
            mismatches.append(f"{component}: expected {version}, installed {actual or 'ABSENT'}")
    if mismatches:
        raise SystemExit("Python source lock does not match the runtime:\n  " + "\n  ".join(mismatches))

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = ["component\tversion\tfilename\tsha256\tsource_url\tkind"]
    for component, version, filename, expected_hash, url, supplemental in rows:
        destination = args.output / filename
        if not destination.exists() or sha256(destination) != expected_hash:
            destination.unlink(missing_ok=True)
            print(f"Downloading source for {component}=={version}: {filename}", flush=True)
            download(url, destination)
        actual_hash = sha256(destination)
        if actual_hash != expected_hash:
            destination.unlink(missing_ok=True)
            raise SystemExit(
                f"SHA-256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )
        kind = "supplemental" if supplemental else "distribution"
        manifest.append("\t".join((component, version, filename, actual_hash, url, kind)))

    (args.output / "MANIFEST.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    (args.output / "README.txt").write_text(
        "Corresponding source archives for Python distributions added by "
        "paidf-simulation, including supplemental source components required "
        "to build them. See MANIFEST.tsv for versions, checksums, and provenance.\n",
        encoding="utf-8",
    )
    print(f"Collected {len(rows)} verified Python source archives in {args.output}")


if __name__ == "__main__":
    main()
