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

"""Provenance record of a render, kept in the dataset manifest.

A rendered image that reaches a partner outlives the session that made it. When
feedback comes back weeks later -- "the solder overruns the pad", "these bright
areas read as pins" -- the first question is which board, which config, which
code and which runtime produced that exact picture. Reconstructing that
afterwards is archaeology, and it silently fails: a checkout's HEAD moves, a prep
layer is rewritten in place, a container tag is retagged.

So the record is taken at render time, by the run itself, and it lives where the
run's other facts live: ``DatasetManifestObserver`` puts the full record into
``dataset_index.json`` and a compact summary into every trigger's
``metadata.json``. There is no sibling ``fingerprint.json`` -- two records of one
run would be two things that can disagree.

What it holds, and why each field earned its place:

``versions``
    the ``cdf`` package version, ``api_version`` and the manifest schema
    version. These are what a QA report can quote; they are not unique, which
    is why ``code`` is kept beside them.
``config``
    the source YAML's path and the sha256 of the bytes that were loaded, when
    the job came from ``load_job()``. A job built in Python has none, and says
    so rather than leaving the field out.

``artifacts``
    sha256 and size of every file the render wrote. Answers "is the file you are
    looking at the file I sent" -- size and filename do not.
``scene``
    the USD layers a text scan of the entry layer can reach -- sublayers,
    references and payloads -- each hashed, with the vendor export's own
    ``customLayerData``. On Process Preparation exports ``creatorVersion`` has
    been constant across every drop received, so ``createdDate`` is the only
    field that identifies a revision. A layer that is remote, missing or
    binary is recorded as such and repeated under ``unverified``; the record
    does not open a USD runtime, so arcs a text scan cannot see are not
    claimed.
``code``
    commit, branch, dirty flag and a hash of the uncommitted diff. A dirty tree
    is common and a clean-looking sha would be a lie. When the package is baked
    into a container the tag is recorded instead, because there is no checkout.
``environment``
    every override actually read, from :mod:`cdf.provenance.env`. This is the
    field configs cannot supply.
``orders``
    the executed plan: order id, type, status. What ran, not what was intended.
``runtime``
    Isaac build, platform, host.
``unverified``
    everything this record could not establish. A fingerprint that quietly
    invents a commit is worse than no fingerprint.

Collection never fails a job. Any collector that raises is reported inside the
record and the render still succeeds.
"""

from __future__ import annotations

import hashlib
import os
import platform
import posixpath
import re
import socket
import subprocess
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from scripts import file_utils

from .env import effective_overrides

#: 1.1: the record moved from a sibling fingerprint.json into the dataset
#: manifest, and gained ``versions``, ``config`` and ``phase``.
FINGERPRINT_VERSION = "1.1"

#: Files that are bookkeeping rather than delivered artifacts. The manifest's own
#: documents are rewritten after this record is taken (recording the record
#: republishes the index), so hashing them would pin a version that no longer
#: exists by the time anyone reads it.
_SKIP_ARTIFACTS = ("dataset_index.json", "metadata.json", ".lock")

#: What the container image bakes beside the package when there is no checkout:
#: the pyproject version, and the commit the CI build was made from
#: (``--build-arg GIT_COMMIT=$CI_COMMIT_SHA`` in the Dockerfile).
_BAKED_VERSION_FILE = "VERSION"
_BAKED_COMMIT_FILE = "GIT_COMMIT"

#: Keys lifted into the per-trigger summary. Everything else stays in the index.
_SUMMARY_KEYS = ("fingerprint_version", "phase", "versions", "config", "code", "image")

#: Hashing every tile of an 80-tile panel is cheap; hashing a 400 MB USD on every
#: render is not, so scene layers above this size are recorded by size and mtime
#: with the hash left explicitly unknown rather than silently skipped.
_HASH_LIMIT_BYTES = 256 << 20

#: ``git diff HEAD`` does not see untracked files, so two dirty trees that
#: differ only in a new file would share a ``diff_sha256``. Untracked files are
#: therefore hashed into the record as well -- up to these caps, past which a
#: run that left a large unignored output tree in the checkout would spend its
#: render time hashing it. Beyond the caps the files are listed and declared
#: unverified instead of silently dropped.
_UNTRACKED_HASH_LIMIT_FILES = 200
_UNTRACKED_HASH_LIMIT_BYTES = 64 << 20

#: A ``.usda`` up to this size is read whole so references and payloads deep
#: in the prim tree are found; above it only the header is scanned and the
#: layer is marked ``not_traversed`` so the gap is declared, not hidden.
_ARC_SCAN_LIMIT_BYTES = 64 << 20
_LAYER_HEADER_BYTES = 8192


def _sha256(path: Path) -> str:
    """Return the hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_file(path: Path, *, hash_limit: int | None = None) -> dict:
    """Hash and stat one file, tolerating absence.

    ``hash_limit`` defaults to the module constant, resolved at call time rather
    than bound as a default argument, so the threshold can be adjusted (and
    exercised) without reimporting.
    """
    if hash_limit is None:
        hash_limit = _HASH_LIMIT_BYTES
    entry: dict[str, object] = {"path": str(path)}
    if not path.exists():
        entry["missing"] = True
        return entry
    stat = path.stat()
    entry["bytes"] = stat.st_size
    entry["mtime_utc"] = datetime.fromtimestamp(
        stat.st_mtime,
        UTC,
    ).isoformat(timespec="seconds")
    if stat.st_size > hash_limit:
        entry["sha256"] = None
        entry["sha256_skipped"] = f"larger than {hash_limit} bytes"
    else:
        entry["sha256"] = _sha256(path)
    return entry


def _git(repo: Path, *args: str) -> str | None:
    """Run one git command in a checkout, returning stdout or None."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            # Git emits UTF-8; a dirty tree's diff carries file content, which
            # the Windows locale codec cannot decode and would turn the whole
            # code record into None.
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _sanitize_remote_url(url: str | None) -> str | None:
    """Reduce a git remote URL to its host and path.

    ``git remote get-url`` returns whatever was configured, and an HTTPS remote
    set up by CI or a token-based clone carries ``user:token@`` in it. The
    manifest ships with the dataset, so userinfo, query and fragment are
    dropped: enough survives to name the repository, nothing that
    authenticates to it. The scp-like form (``[user@]host:path``) loses its
    login the same way.

    Args:
        url: A remote URL as git reports it, or ``None``.

    Returns:
        The sanitized URL, or ``None`` when there was none.

    Example:

    .. code-block:: python

        print(_sanitize_remote_url("https://me:tok@git.example.com/g/r.git?x=1"))
        # https://git.example.com/g/r.git

    """
    if not url:
        return None
    if "://" in url:
        parts = urlsplit(url)
        host = parts.netloc.rpartition("@")[2]
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    head, sep, tail = url.partition(":")
    if sep and "@" in head:
        head = head.rpartition("@")[2]
    return f"{head}{sep}{tail}"


def _describe_untracked(repo: Path) -> dict:
    """Account for the untracked files ``git diff HEAD`` leaves out.

    Each untracked path is hashed with its content into one digest when the set
    is within :data:`_UNTRACKED_HASH_LIMIT_FILES` and
    :data:`_UNTRACKED_HASH_LIMIT_BYTES`; otherwise the digest is left ``None``
    with the reason spelled out, and :func:`build` repeats that under
    ``unverified``.

    Args:
        repo: The checkout.

    Returns:
        ``untracked_files``, ``untracked_count``, ``untracked_bytes`` and
        ``untracked_sha256`` (``None`` with ``untracked_skipped`` when capped).

    Example:

    .. code-block:: python

        print(sorted(_describe_untracked(Path(".")))[:2])
        # ['untracked_bytes', 'untracked_count']

    """
    listing = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    if listing is None:
        return {
            "untracked_files": None,
            "untracked_count": None,
            "untracked_bytes": None,
            "untracked_sha256": None,
            "untracked_skipped": "git ls-files failed; untracked files are unknown",
        }
    paths = sorted(item for item in listing.split("\0") if item)
    total = 0
    for rel in paths:
        try:
            total += (repo / rel).stat().st_size
        except OSError:
            continue
    record: dict[str, object] = {
        "untracked_files": paths[:_UNTRACKED_HASH_LIMIT_FILES],
        "untracked_count": len(paths),
        "untracked_bytes": total,
    }
    if len(paths) > _UNTRACKED_HASH_LIMIT_FILES or total > _UNTRACKED_HASH_LIMIT_BYTES:
        record["untracked_sha256"] = None
        record["untracked_skipped"] = (
            f"{len(paths)} files, {total} bytes; not hashed above "
            f"{_UNTRACKED_HASH_LIMIT_FILES} files or {_UNTRACKED_HASH_LIMIT_BYTES} bytes"
        )
        return record
    digest = hashlib.sha256()
    for rel in paths:
        try:
            content = _sha256(repo / rel)
        except OSError:
            content = "unreadable"
        digest.update(f"{rel}\0{content}\n".encode())
    record["untracked_sha256"] = digest.hexdigest()
    return record


def collect_code(package_root: Path) -> dict:
    """Describe the code that is executing.

    Args:
        package_root: Directory to interrogate as a git checkout.

    Returns:
        A record naming the commit, or naming why there is no commit to give.

    Example:

    .. code-block:: python

        record = collect_code(Path("."))
        print("source" in record)
        # True

    """
    # A container bakes the package and ships no .git, so the image reference is
    # the only commit-bearing identifier available inside it. It is recorded
    # verbatim; resolving a tag to a sha is the reader's job, and a tag is
    # mutable, which the reader needs to know.
    image = os.environ.get("CDF_CONTAINER_IMAGE") or os.environ.get("CDF_IMAGE")
    image_digest = os.environ.get("CDF_IMAGE_DIGEST")
    if not (package_root / ".git").exists():
        # The CI build bakes the commit it was made from beside the package
        # (Dockerfile: GIT_COMMIT from --build-arg), so a container can name
        # its commit without a checkout and without asking the daemon.
        baked = _read_baked(package_root / _BAKED_COMMIT_FILE)
        commit = baked if baked and baked != "unknown" else None
        if commit or image:
            record: dict[str, object] = {"source": "container", "commit": commit}
            if image:
                record["image"] = image
            if image_digest:
                record["image_digest"] = image_digest
            if commit is None:
                record["note"] = (
                    "No checkout and no baked GIT_COMMIT. A tag is mutable, so "
                    "this identifies the build only as strongly as the tag is "
                    "protected."
                )
            return record
        return {
            "source": "unknown",
            "commit": None,
            "note": "No git checkout, no baked GIT_COMMIT and no container image reference.",
        }

    status = _git(package_root, "status", "--porcelain") or ""
    record: dict[str, object] = {
        "source": "checkout",
        "path": str(package_root),
        "commit": _git(package_root, "rev-parse", "HEAD"),
        "branch": _git(package_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "committed_utc": _git(package_root, "log", "-1", "--format=%cI"),
        "subject": _git(package_root, "log", "-1", "--format=%s"),
        "dirty": bool(status.strip()),
        "remotes": {},
    }
    for name in (_git(package_root, "remote") or "").split():
        # Sanitized: a remote URL can carry a token, and this record ships.
        record["remotes"][name] = _sanitize_remote_url(
            _git(package_root, "remote", "get-url", name),
        )
    if record["dirty"]:
        # The commit alone does not reproduce a dirty tree. Recording the file
        # list and a hash of the diff at least makes the gap measurable, and
        # lets a reader confirm they hold the same working state. The diff
        # covers tracked files only; untracked ones are hashed separately.
        record["dirty_files"] = [line[3:] for line in status.splitlines()][:200]
        record["diff_sha256"] = hashlib.sha256(
            (_git(package_root, "diff", "HEAD") or "").encode("utf-8"),
        ).hexdigest()
        record.update(_describe_untracked(package_root))
    if image:
        record["image"] = image
    return record


#: Composition arcs a usda declares in text. ``subLayers`` lives in the layer
#: header, ``references`` and ``payload`` in prim metadata; all three take one
#: ``@asset@`` or a list of them, optionally followed by a ``</prim>`` target.
_ARC_BLOCK = re.compile(
    r"\b(subLayers|references|payload)\b\s*=\s*(\[[^\]]*\]|@[^@\n]*@)",
    re.DOTALL,
)
_ASSET_REF = re.compile(r"@([^@\n]+)@")
_CUSTOM_STR = re.compile(r'string\s+(\w+)\s*=\s*"([^"]*)"')
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_ARC_ROLE = {"subLayers": "sublayer", "references": "reference", "payload": "payload"}


def _is_remote_layer(target: str) -> bool:
    """Whether a layer path is something this module must not try to open."""
    return file_utils.is_remote(target) or bool(_URI_SCHEME.match(target))


def _is_text_layer(path: Path) -> bool:
    """Whether a layer is ASCII usda, by suffix or by its magic header."""
    suffix = path.suffix.lower()
    if suffix == ".usda":
        return True
    if suffix == ".usdc":
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"#usda"
    except OSError:
        return False


def _read_layer_text(path: Path, size: int) -> tuple[str, bool]:
    """Return a text layer's content and whether all of it was read."""
    complete = size <= _ARC_SCAN_LIMIT_BYTES
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(None if complete else _LAYER_HEADER_BYTES), complete
    except OSError:
        return "", False


def _layer_arcs(text: str) -> list[tuple[str, str]]:
    """List ``(role, asset_path)`` for every arc the text declares."""
    arcs: list[tuple[str, str]] = []
    for block in _ARC_BLOCK.finditer(text):
        role = _ARC_ROLE[block.group(1)]
        arcs.extend((role, ref) for ref in _ASSET_REF.findall(block.group(2)))
    return arcs


def collect_scene(usd_path: str | os.PathLike[str] | None) -> dict:
    """Describe the USD layers behind a render, as far as a text scan reaches.

    Walks the composition arcs a ``.usda`` declares in plain text -- ``subLayers``,
    ``references`` and ``payload`` -- from the entry layer down, hashing each
    layer it can open and surfacing the deepest vendor stamp it finds. What it
    cannot establish is recorded on the layer and repeated by :func:`build`
    under ``unverified``:

    * a **remote** layer (``s3://`` or any other scheme) is named but never
      opened: this module has no storage client and no USD runtime;
    * a **missing** layer is named with the layer that referenced it;
    * a **binary** (crate) layer is hashed but its arcs are not read;
    * a text layer above :data:`_ARC_SCAN_LIMIT_BYTES` has only its header
      scanned.

    Arcs that only a composed stage knows about -- variant selections,
    inherits, arcs inside binary layers -- are outside what this records.

    Args:
        usd_path: Entry USD, typically a small prep layer.

    Returns:
        A record with ``layers`` and, when present, ``source_export``.

    Example:

    .. code-block:: python

        print(collect_scene(None)["layers"])
        # []

    """
    if not usd_path:
        return {"layers": [], "note": "No scene path recorded for this run."}

    entry_text = os.fspath(usd_path)
    layers: list[dict] = []
    seen: set[object] = set()
    queue: list[tuple[str, str, str | None]] = [(entry_text, "entry", None)]
    while queue:
        target, role, parent = queue.pop(0)
        if _is_remote_layer(target):
            if target in seen:
                continue
            seen.add(target)
            entry: dict[str, object] = {
                "path": target,
                "role": role,
                "remote": True,
                "not_traversed": "remote layer; not opened",
            }
            if parent:
                entry["referenced_from"] = parent
            layers.append(entry)
            continue
        path = Path(target)
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        entry = _describe_file(path)
        entry["role"] = role
        if parent:
            entry["referenced_from"] = parent
        if entry.get("missing"):
            layers.append(entry)
            continue
        if _is_text_layer(path):
            text, complete = _read_layer_text(path, int(entry.get("bytes") or 0))
            head = text[:_LAYER_HEADER_BYTES]
            start = head.find("customLayerData")
            if start >= 0:
                custom = dict(_CUSTOM_STR.findall(head[start : start + 1024]))
                if custom:
                    entry["customLayerData"] = custom
            for kind, ref in _layer_arcs(text):
                child = ref if _is_remote_layer(ref) else str(path.parent / ref)
                queue.append((child, kind, str(path)))
            if not complete:
                entry["not_traversed"] = (
                    f"text layer larger than {_ARC_SCAN_LIMIT_BYTES} bytes; only the "
                    f"first {_LAYER_HEADER_BYTES} bytes were scanned for arcs"
                )
        else:
            # A crate (binary) layer cannot be read without a USD runtime, and
            # this module deliberately has none. Its bytes are hashed above;
            # what is NOT known is whether it composes further layers or
            # carries a vendor stamp. Say so, or a one-entry chain reads as a
            # flat scene.
            entry["not_traversed"] = (
                "binary USD layer; subLayers, references and payloads were not read"
            )
        layers.append(entry)

    scene: dict[str, object] = {"entry": entry_text, "layers": layers}
    for entry in reversed(layers):
        custom = entry.get("customLayerData") or {}
        if custom.get("creator"):
            scene["source_export"] = {
                "path": entry["path"],
                "sha256": entry.get("sha256"),
                **custom,
            }
            break
    return scene


def collect_artifacts(output_dir: Path, *, limit: int = 4096) -> dict:
    """Hash every file the render wrote under ``output_dir``.

    Args:
        output_dir: Directory holding the render's outputs.
        limit: Maximum number of files to record.

    Returns:
        A record with ``files``, ``count`` and ``bytes``.

    Example:

    .. code-block:: python

        record = collect_artifacts(Path("does-not-exist"))
        print(record["count"])
        # 0

    """
    files: list[dict] = []
    total = 0
    truncated = False
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file() or path.name.endswith(_SKIP_ARTIFACTS):
                continue
            if len(files) >= limit:
                truncated = True
                break
            entry = _describe_file(path)
            entry["path"] = str(path.relative_to(output_dir))
            files.append(entry)
            total += entry.get("bytes", 0) or 0
    record: dict[str, object] = {"count": len(files), "bytes": total, "files": files}
    if truncated:
        # A silent cap reads as "everything was covered". Say what was dropped.
        record["truncated_at"] = limit
    return record


def collect_runtime() -> dict:
    """Describe the machine and the Isaac build.

    Returns:
        A record naming the platform, the host and the Isaac version when the
        install can be located.

    Example:

    .. code-block:: python

        print("platform" in collect_runtime())
        # True

    """
    record: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    # ISAAC_SIM_PATH is often unset: launching through `python.bat` puts the
    # interpreter inside the install without exporting anything, which is how a
    # real verification run came back with no Isaac version at all. Walk up from
    # the interpreter as a fallback.
    candidates: list[Path] = []
    isaac_root = os.environ.get("ISAAC_SIM_PATH")
    if isaac_root:
        candidates.append(Path(isaac_root))
    import sys

    candidates.extend(Path(sys.executable).resolve().parents[:6])
    for candidate in candidates:
        version_file = candidate / "VERSION"
        if not version_file.is_file():
            continue
        try:
            record["isaac_version"] = version_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        record["isaac_root"] = str(candidate)
        break
    return record


def collect_orders(result: object) -> list[dict]:
    """Summarise what the executor actually ran.

    Args:
        result: A ``JobResult``.

    Returns:
        One entry per order with its id, type and status.

    Example:

    .. code-block:: python

        print(collect_orders(None))
        # []

    """
    orders = getattr(result, "order_results", None) or ()
    entries = []
    for item in orders:
        entry = {
            "order_id": getattr(item, "order_id", None),
            "order_type": getattr(item, "order_type", None),
            "status": getattr(item, "status", None),
            "error": getattr(item, "error", "") or None,
        }
        summary = _summarise_output(getattr(item, "output", None))
        if summary:
            entry["output"] = summary
        entries.append(entry)
    return entries


#: Order outputs can carry thousands of prim paths. Recording none of them left
#: the record unable to answer "how many joints were generated" -- which is the
#: number that decides whether a board is usable as training data. Recording all
#: of them would make the file useless. So: scalars verbatim, collections by
#: length and a short sample.
_OUTPUT_SAMPLE = 3
_OUTPUT_MAX_KEYS = 40


def _summarise_output(output: object, depth: int = 0) -> object:
    """Reduce a handler output to something worth keeping in a record."""
    if output is None or depth > 2:
        return None
    if isinstance(output, (bool, int, float, str)):
        return output
    if isinstance(output, Mapping):
        summary = {}
        for key in list(output)[:_OUTPUT_MAX_KEYS]:
            value = _summarise_output(output[key], depth + 1)
            if value is not None:
                summary[str(key)] = value
        return summary or None
    if isinstance(output, (list, tuple, set)):
        items = list(output)
        summary: dict[str, object] = {"count": len(items)}
        if items and depth < 2:
            sample = [
                _summarise_output(item, depth + 1) for item in items[:_OUTPUT_SAMPLE]
            ]
            sample = [item for item in sample if item is not None]
            if sample:
                summary["sample"] = sample
        return summary
    return None


def _package_root() -> Path:
    """Return the repository root that contains this package."""
    return Path(__file__).resolve().parents[2]


def _read_baked(path: Path) -> str | None:
    """Return the stripped content of a baked marker file, or None."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def collect_versions(package_root: Path) -> dict:
    """Name the versions a reader can quote.

    The package version comes from ``pyproject.toml`` in a checkout, or from
    the ``VERSION`` file the image bakes beside the package. Neither is unique
    across commits, which is why ``code`` is recorded beside it; but a QA
    report or a bug filed against ``1.0.0 / api 1.x`` is something a human can
    act on, and a bare sha is not.

    Args:
        package_root: Repository or installed package root.

    Returns:
        ``cdf``, ``api_version``, ``manifest_schema`` and ``fingerprint``.

    Example:

    .. code-block:: python

        print("api_version" in collect_versions(Path(".")))
        # True

    """
    from ..api_version import CDF_API_VERSION
    from ..execution.dataset_manifest import SCHEMA_VERSION

    cdf_version: str | None = None
    pyproject = package_root / "pyproject.toml"
    if pyproject.is_file():
        try:
            for line in pyproject.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("version") and "=" in stripped:
                    cdf_version = (
                        stripped.split("=", 1)[1].strip().strip('"').strip("'")
                    )
                    break
        except OSError:
            cdf_version = None
    if cdf_version is None:
        cdf_version = _read_baked(package_root / _BAKED_VERSION_FILE)
    return {
        "cdf": cdf_version,
        "api_version": CDF_API_VERSION,
        "manifest_schema": SCHEMA_VERSION,
        "fingerprint": FINGERPRINT_VERSION,
    }


def collect_config(plan: object) -> dict:
    """Name the config file a plan was authored from, if there was one.

    ``load_job()`` stamps ``JobOrderPlan.metadata["source"]`` with the path and
    the sha256 of the bytes it loaded; that survives into the Isaac child
    process, where the ``CDFJob`` does not. A job built directly in Python
    has no source file, and the record says so instead of omitting the field.

    Args:
        plan: A ``JobOrderPlan``.

    Returns:
        ``path`` and ``sha256`` when known, with ``sha256_now`` when the file
        can be re-read at record time and differs from what was loaded.

    Example:

    .. code-block:: python

        print(collect_config(None)["source"])
        # python

    """
    metadata = getattr(plan, "metadata", None) or {}
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    if not isinstance(source, Mapping) or not source.get("config_path"):
        return {"source": "python", "note": "Job was built in Python; no config file."}
    record: dict[str, object] = {
        "source": "yaml",
        "path": str(source["config_path"]),
        "sha256": source.get("config_sha256"),
    }
    path = Path(str(source["config_path"]))
    if path.is_file():
        try:
            now = _sha256(path)
        except OSError:
            now = None
        if now and record["sha256"] and now != record["sha256"]:
            # The file changed between load and render. Both hashes are kept so
            # a reader knows which bytes the plan came from and which are on disk.
            record["sha256_now"] = now
    return record


def build(
    *,
    result: object,
    output_dir: str | os.PathLike[str] | None,
    usd_path: str | os.PathLike[str] | None = None,
    plan: object = None,
    phase: str = "complete",
    extra: dict | None = None,
    notes: list[str] | None = None,
) -> dict:
    """Assemble the provenance record for one render.

    Every collector is isolated: one that raises is reported in ``errors`` and
    the rest still run, because a provenance record is worth having incomplete
    and worthless if it can fail a render.

    Args:
        result: The ``JobResult`` returned by the executor, or ``None`` before
            the run has one.
        output_dir: Directory holding the render's artifacts.
        usd_path: Entry USD for the scene, when known.
        plan: The ``JobOrderPlan``; supplies the config identity.
        phase: ``"prepare"`` for the static record taken before the first
            handler runs, ``"complete"`` for the one taken at the end.
        extra: Additional caller-supplied facts to merge in.
        notes: ``unverified`` entries the caller established before building,
            such as why the artifact root is narrower than the plan's passes.

    Returns:
        The provenance record.

    Example:

    .. code-block:: python

        record = build(result=None, output_dir=None)
        print(record["fingerprint_version"])
        # 1.1

    """
    record: dict[str, object] = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "phase": phase,
        "unverified": [],
        "errors": {},
    }
    # A remote destination is not a filesystem path: Path("s3://bucket/x")
    # is a relative local path that does not exist, and collect_artifacts()
    # would report count 0 -- indistinguishable from a render that wrote
    # nothing. Say it could not be established instead.
    out: Path | None = None
    remote_output: str | None = None
    if output_dir:
        if file_utils.is_remote(output_dir):
            remote_output = str(output_dir)
        else:
            out = Path(output_dir)
    root = _package_root()

    collectors = {
        "versions": lambda: collect_versions(root),
        "code": lambda: collect_code(root),
        "config": lambda: collect_config(plan),
        "scene": lambda: collect_scene(usd_path),
        "environment": effective_overrides,
        "runtime": collect_runtime,
        "orders": lambda: collect_orders(result),
        "artifacts": lambda: (
            collect_artifacts(out) if out else {"count": 0, "files": []}
        ),
    }
    for key, collector in collectors.items():
        try:
            record[key] = collector()
        except Exception as error:  # noqa: BLE001 -- never fail a render
            record[key] = None
            record["errors"][key] = f"{type(error).__name__}: {error}"
            record["unverified"].append(f"Collector {key!r} failed: {error}")

    if remote_output is not None:
        record["unverified"].append(
            f"Artifacts were not hashed: the output location {remote_output!r} is "
            "remote and this collector enumerates local filesystem paths only."
        )
    record["unverified"].extend(notes or [])

    record["success"] = bool(getattr(result, "success", False))

    code = record.get("code") or {}
    if code.get("source") == "unknown":
        record["unverified"].append(
            "The code that produced these artifacts is not established: no git "
            "checkout, no baked GIT_COMMIT and no container image reference.",
        )
    if code.get("source") == "checkout" and not code.get("commit"):
        record["unverified"].append(
            "A checkout is present but its commit could not be read (git "
            "unavailable or the repository unreadable); the code is not pinned.",
        )
    if code.get("dirty"):
        record["unverified"].append(
            "The working tree was dirty, so the commit alone does not reproduce "
            "this render; see code.diff_sha256 and code.untracked_sha256.",
        )
    if code.get("dirty") and code.get("untracked_skipped"):
        record["unverified"].append(
            "Untracked files were not hashed ({}); the working state is not "
            "fully pinned. The first {} are listed under code.untracked_files.".format(
                code["untracked_skipped"],
                _UNTRACKED_HASH_LIMIT_FILES,
            ),
        )
    if code.get("source") == "container" and not code.get("commit"):
        record["unverified"].append(
            "Identified by container tag only. Tags are mutable; this image was "
            "built without GIT_COMMIT baked in.",
        )
    if code.get("source") == "container" and not code.get("image_digest"):
        record["unverified"].append(
            "Image digest not recorded (CDF_IMAGE_DIGEST unset); the commit "
            "identifies the source, not the exact image bytes.",
        )
    config = record.get("config") or {}
    if config.get("source") == "yaml" and not config.get("sha256"):
        record["unverified"].append(
            "The config was loaded from a file whose bytes were not hashed; "
            "config.path names it but nothing pins its content.",
        )
    if config.get("sha256_now"):
        record["unverified"].append(
            "The config file on disk differs from the bytes this plan was "
            "loaded from; see config.sha256 vs config.sha256_now.",
        )
    if not usd_path:
        record["unverified"].append("No scene path was recorded for this run.")
    for layer in (record.get("scene") or {}).get("layers") or []:
        role = layer.get("role", "layer")
        name = layer["path"] if layer.get("remote") else Path(layer["path"]).name
        if layer.get("sha256_skipped"):
            # Recorded on the layer, but a reader scanning `unverified` for what
            # is NOT pinned would otherwise miss it.
            record["unverified"].append(
                "Layer {} is not hashed ({}); its bytes are not pinned by this "
                "record.".format(name, layer["sha256_skipped"]),
            )
        if layer.get("remote"):
            record["unverified"].append(
                f"Layer {name} ({role}) is remote; it was not opened, so neither "
                "its bytes nor any layer it composes is recorded.",
            )
        elif layer.get("missing"):
            record["unverified"].append(
                "Layer {} ({}, referenced from {}) is missing on disk; the "
                "composition this record describes is incomplete.".format(
                    name,
                    role,
                    layer.get("referenced_from") or "the run",
                ),
            )
        elif layer.get("not_traversed"):
            record["unverified"].append(
                "Layer {} was not traversed ({}); deeper layers and any vendor "
                "stamp are not recorded.".format(name, layer["not_traversed"]),
            )

    if extra:
        record.update(extra)
    return record


def summary(record: Mapping[str, object]) -> dict:
    """Reduce a record to what every trigger manifest should carry.

    ``dataset_index.json`` holds the whole record once; repeating it in every
    trigger's ``metadata.json`` would repeat thousands of artifact hashes per
    trigger. What a trigger needs is enough to answer "which code, which
    versions, which config, which board" without opening the index.

    A trigger keeps the summary it was written with; trigger manifests are not
    rewritten when the final record lands. That is safe because every identity
    field here --
    ``versions``, ``commit``, ``dirty``, ``code_source``, ``image``,
    ``image_digest``, ``config_path``, ``config_sha256``, ``board_created`` --
    is the same in the ``prepare`` record and the final one. Only ``phase``
    and ``unverified_count`` describe the moment of writing, and a reader who
    needs the run's final state reads ``dataset_index.json``.

    Args:
        record: Provenance record from :func:`build`.

    Returns:
        A compact mapping. Never raises on a partial record.

    Example:

    .. code-block:: python

        print(summary({"fingerprint_version": "1.1", "code": {"commit": "abc"}})["commit"])
        # abc

    """
    code = record.get("code") or {}
    config = record.get("config") or {}
    scene = record.get("scene") or {}
    source_export = (
        scene.get("source_export") or {} if isinstance(scene, Mapping) else {}
    )
    out: dict[str, object] = {
        "fingerprint_version": record.get("fingerprint_version"),
        "phase": record.get("phase"),
        "versions": record.get("versions"),
        "commit": code.get("commit") if isinstance(code, Mapping) else None,
        "dirty": code.get("dirty") if isinstance(code, Mapping) else None,
        "code_source": code.get("source") if isinstance(code, Mapping) else None,
        "image": code.get("image") if isinstance(code, Mapping) else None,
        "image_digest": code.get("image_digest") if isinstance(code, Mapping) else None,
        "config_path": config.get("path") if isinstance(config, Mapping) else None,
        "config_sha256": config.get("sha256") if isinstance(config, Mapping) else None,
        "board_created": source_export.get("createdDate"),
        "unverified_count": len(record.get("unverified") or []),
    }
    return {key: value for key, value in out.items() if value is not None}


def static_record(plan: object) -> dict:
    """Build the part of the record that is knowable before the run starts.

    Called from ``IsaacExecutionContext.prepare()``. Everything except orders,
    artifacts and ``success`` is established here, so a run that dies half-way
    still leaves its trigger manifests attributable. ``artifacts`` is left
    empty on purpose: there are none yet, and walking the output root now would
    hash whatever a previous run left there.

    Args:
        plan: The ``JobOrderPlan`` about to execute.

    Returns:
        A record with ``phase == "prepare"``.

    Example:

    .. code-block:: python

        print(static_record(None)["phase"])
        # prepare

    """
    usd_path, _, notes = plan_paths_detail(plan)
    return build(
        result=None,
        output_dir=None,
        usd_path=usd_path,
        plan=plan,
        phase="prepare",
        notes=notes,
    )


def plan_paths(plan: object) -> tuple[str | None, str | None]:
    """Derive the scene and the artifact root from a compiled plan.

    See :func:`plan_paths_detail`; this drops the explanatory notes.

    Args:
        plan: A ``JobOrderPlan``.

    Returns:
        ``(usd_path, output_root)``, either of which may be ``None``.

    Example:

    .. code-block:: python

        print(plan_paths(None))
        # (None, None)

    """
    usd_path, output_root, _ = plan_paths_detail(plan)
    return usd_path, output_root


def plan_paths_detail(plan: object) -> tuple[str | None, str | None, list[str]]:
    """Derive the scene, the artifact root and why the root is what it is.

    The plan is the transport-safe truth: it survives serialization into an
    Isaac child process, where no ``CDFJob`` exists to interrogate. Capture
    orders carry a per-pass ``output_dir``, so the artifact root is their common
    parent -- but only when that parent is a place ``collect_artifacts`` may
    walk. Two passes whose only common ancestor is the filesystem root, or a
    common path outside the run's declared ``output_root``, would send the
    collector across unrelated files and record their hashes as this render's.
    Such a root is rejected in favour of the first pass's directory, and the
    reason is returned so :func:`build` can file it under ``unverified``.

    Args:
        plan: A ``JobOrderPlan``.

    Returns:
        ``(usd_path, output_root, notes)``.

    Example:

    .. code-block:: python

        print(plan_paths_detail(None))
        # (None, None, [])

    """
    usd_path: str | None = None
    output_dirs: list[str] = []
    notes: list[str] = []
    for order in getattr(plan, "orders", None) or ():
        params = getattr(order, "parameters", None) or {}
        if getattr(order, "order_type", None) == "scene.load" and not usd_path:
            candidate = params.get("usd_path")
            if candidate:
                usd_path = str(candidate)
        candidate_dir = params.get("output_dir")
        if candidate_dir:
            output_dirs.append(str(candidate_dir))
    if not output_dirs:
        return usd_path, None, notes
    # The record sits with the artifacts it describes, and `collect_artifacts`
    # walks whatever root is returned here -- so a one-pass run files inside
    # that pass's directory and a multi-pass run files at their shared parent.
    # That reads as an inconsistency and is deliberately kept: taking the parent
    # of a single directory would, for a caller who passed their final
    # destination straight through, write the record OUTSIDE the directory they
    # asked us to fill. Filing one level deep is cosmetic; escaping the output
    # directory is not.
    if len(output_dirs) == 1:
        return usd_path, output_dirs[0], notes
    remote = [file_utils.is_remote(candidate) for candidate in output_dirs]
    if any(remote):
        # os.path.commonpath would turn `s3://bucket/run/a` and `.../b` into
        # `s3:/bucket/run` -- no longer a URI, so is_remote() said no and the
        # record reported an empty LOCAL artifact directory. Common prefix on
        # the URI text instead; a local/remote mix has no shared root at all.
        if all(remote):
            return usd_path, _common_remote_root(output_dirs), notes
        notes.append(
            "Capture passes mix local and remote output locations; artifacts "
            f"were enumerated under the first pass only ({output_dirs[0]!r})."
        )
        return usd_path, output_dirs[0], notes
    root, note = _local_artifact_root(output_dirs, _declared_output_root(plan))
    if note:
        notes.append(note)
    return usd_path, root, notes


def _declared_output_root(plan: object) -> str | None:
    """Return the run output root the compiler stamped on the plan, if any."""
    metadata = getattr(plan, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    declared = metadata.get("output_root")
    return str(declared) if declared else None


def _is_filesystem_root(path: str) -> bool:
    """Whether ``path`` is ``/`` or a drive root once made absolute."""
    absolute = os.path.abspath(path)
    return os.path.dirname(absolute) == absolute


def _is_within(child: str, root: str) -> bool:
    """Whether ``child`` is ``root`` or lies beneath it."""
    child_abs = os.path.abspath(child)
    root_abs = os.path.abspath(root)
    try:
        return os.path.commonpath([child_abs, root_abs]) == root_abs
    except ValueError:
        return False


def _local_artifact_root(
    output_dirs: list[str], declared: str | None
) -> tuple[str, str | None]:
    """Choose the directory ``collect_artifacts`` may walk for local passes.

    Args:
        output_dirs: Two or more local per-pass output directories.
        declared: The run's ``output_root`` from the plan metadata, when known.

    Returns:
        ``(root, note)``: the common parent and ``None`` when it is safe; the
        first pass and the reason when the common parent was rejected.

    Example:

    .. code-block:: python

        print(_local_artifact_root(["run/a", "run/b"], "run"))
        # ('run', None)

    """
    first = output_dirs[0]
    others = ", ".join(repr(item) for item in output_dirs[1:])
    try:
        common = os.path.commonpath(output_dirs)
    except ValueError:
        # Different drives on Windows: no common root, so fall back to the first
        # rather than dropping the record entirely.
        return first, (
            "Capture passes share no common root (different drives); artifacts "
            f"were enumerated under the first pass only ({first!r}), not under {others}."
        )
    if _is_filesystem_root(common):
        return first, (
            f"Capture passes share only the filesystem root {common!r}; walking it "
            "would hash files that are not this render's, so artifacts were "
            f"enumerated under the first pass only ({first!r}), not under {others}."
        )
    if (
        declared
        and not file_utils.is_remote(declared)
        and not _is_within(common, declared)
    ):
        return first, (
            f"The passes' common path {common!r} lies outside the run output root "
            f"{declared!r}; artifacts were enumerated under the first pass only "
            f"({first!r}), not under {others}."
        )
    return common, None


def _common_remote_root(uris: list[str]) -> str:
    """Return the deepest URI all ``uris`` share, or the first when none.

    Args:
        uris: Remote URIs, all with a scheme (``s3://bucket/key``).

    Returns:
        ``scheme://common/prefix``; the first URI when schemes differ or the
        paths share nothing.

    Example:

    .. code-block:: python

        print(_common_remote_root(["s3://b/run/a", "s3://b/run/b"]))
        # s3://b/run

    """
    split = [uri.split("://", 1) for uri in uris]
    schemes = {scheme for scheme, _ in split}
    if len(schemes) != 1:
        return uris[0]
    try:
        common = posixpath.commonpath([tail for _, tail in split])
    except ValueError:
        return uris[0]
    if not common:
        return uris[0]
    return f"{schemes.pop()}://{common}"


def record_for_plan(plan: object, result: object, *, manifest: object = None) -> dict:
    """Build the full provenance record for an executed plan and file it.

    This is the single point every execution path reaches: ``render_async``
    calls the executor directly, and the synchronous ``render`` serializes the
    plan into an Isaac child that runs the same executor. Hooking the executor
    therefore covers the Python API, the YAML config path and the examples,
    where hooking the authoring section covered only the first.

    The sink is the dataset manifest: ``manifest.record_provenance(record)``
    puts the record into ``dataset_index.json`` and a summary into every
    trigger ``metadata.json``, replacing the static record taken at
    ``prepare()``. Without a manifest (tests, foreign executors) the record is
    still built and returned, and nothing is written anywhere.

    Args:
        plan: The executed ``JobOrderPlan``.
        result: The ``JobResult``, including a partial one from a failure.
        manifest: The run's ``DatasetManifestObserver``, when there is one.

    Returns:
        The record.

    Example:

    .. code-block:: python

        print(record_for_plan(None, None)["phase"])
        # complete

    """
    try:
        usd_path, output_root, notes = plan_paths_detail(plan)
        record = build(
            result=result,
            output_dir=output_root,
            usd_path=usd_path,
            plan=plan,
            notes=notes,
        )
        recorder = getattr(manifest, "record_provenance", None)
        if callable(recorder):
            recorder(record)
        return record
    except Exception as error:
        # A provenance system that silently stops producing provenance is worse
        # than one that fails loudly; the executor hook warns and carries on.
        warnings.warn(
            f"provenance record could not be built or recorded: {type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        raise
