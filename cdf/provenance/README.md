# `cdf.provenance` — what produced a render, recorded by the render

Every declarative render records its provenance into the dataset manifest: the
full record in `dataset_index.json` under `provenance`, and a compact summary in
every trigger's `metadata.json`. Nothing has to be remembered, declared, or
reconstructed afterwards, and there is no separate `fingerprint.json` — two
records of one run would be two things that can disagree.

The record is taken twice. `IsaacExecutionContext.prepare()` records the static
part (code, versions, scene, config, runtime) before the first handler runs, so a
run that dies half-way still leaves its trigger manifests attributable; the
executor replaces it with the full record (orders, artifacts, `success`) when the
plan finishes.

## Why this is not a caller's job

Reconstructing provenance after the fact fails quietly, and has:

* a checkout's HEAD moves, so reading it later reports a commit **authored after
  the images it claims to explain**;
* a prep layer regenerated per variant leaves a file whose mtime **postdates**
  the render being discussed;
* a container tag is mutable, so it is not an identity;
* an environment override that changed the geometry appears in **no** YAML, no
  job summary and no commit, so two runs from an identical commit and an
  identical config are not the same picture and nothing says why.

All four occur in practice. The record is therefore taken at render time, by
the run itself.

## What is recorded

`dataset_index.json` → `provenance`:

```jsonc
{
  "fingerprint_version": "1.1",
  "phase": "complete",         // "prepare" until the executor finishes the plan
  "success": true,
  "versions": {                // what a QA report can quote
    "cdf": "1.0.0", "api_version": "...", "manifest_schema": "1.0.0"
  },
  "code": {                    // commit + branch + dirty + diff hash from a
    "source": "checkout",      // checkout, or the GIT_COMMIT the image bakes
    "commit": "...", "dirty": false
  },
  "config": {                  // the YAML this plan was loaded from, if any
    "source": "yaml", "path": "...", "sha256": "..."
  },
  "scene": {                   // sublayers/references/payloads a text scan
                               // reaches, hashed; remote/missing/binary declared
    "layers": [...],
    "source_export": {"creator": "...", "createdDate": "..."}
  },
  "environment": {...},        // every override actually read (see below)
  "orders": [...],             // what the executor ran, with status
  "artifacts": {...},          // sha256 + size of every file written
  "runtime": {...},            // Isaac build, platform, host
  "unverified": [...],         // everything this record could NOT establish
  "errors": {}                 // any collector that failed
}
```

Every trigger `metadata.json` → `provenance`: the compact summary —
`commit`, `dirty`, `versions`, `config_path`, `config_sha256`, `board_created`
(the vendor export's `createdDate`), `image`/`image_digest` when in a container,
and `unverified_count`.

`unverified` is the load-bearing field. A record that quietly invents a commit is
worse than no record.

### Versions *and* the commit, not one or the other

A version number is what a human quotes and a bug is filed against; it is also
not unique — two branches can both say `1.0.0`. The commit is unique and is what
reproduces a build; it is also not something QA can act on. So both are kept.
Inside the container there is no checkout: the Dockerfile bakes `GIT_COMMIT`
(from `--build-arg GIT_COMMIT=$CI_COMMIT_SHA`) beside `VERSION`, and `code`
reads it. A tag alone is recorded as exactly that, with an `unverified` entry.

### Runtime contract for containers

The process inside a container cannot ask which image it is running from, so
the **launcher** exports it:

| Variable | Set to | Recorded as |
|---|---|---|
| `CDF_CONTAINER_IMAGE` | the image tag that was started (`repo:tag`) | `code.image` |
| `CDF_IMAGE_DIGEST` | the image's `sha256:...` digest | `code.image_digest` |

The CI functional jobs pass both from the `docker-build-push` dotenv
(`--env CDF_CONTAINER_IMAGE="$CANDIDATE_IMAGE_TAG"
--env CDF_IMAGE_DIGEST="$CANDIDATE_IMAGE_DIGEST"`, see
`pipelines/stage-functional-test.yml`). Any other launcher -- a skill emitting a
`docker run`, a lab box, an Airflow task -- owes the same two `--env` flags. When
they are absent the record does not guess: `code.image` is missing or the tag
alone is recorded, and `unverified` carries an explicit entry saying the image
identity is not established.

### Artifact root never leaves the run

`collect_artifacts` walks the common parent of the capture passes' output
directories. When that parent is the filesystem root, or lies outside the
`output_root` the compiler stamped on the plan, the collector would hash files
that are not this render's; the root is then narrowed to the first pass and
`unverified` says why and which passes were not enumerated.

> An exporter's `creatorVersion` may not change between revisions;
> `createdDate` identifies them. The record surfaces both so a reader cannot
> mistake one for the other.

## Environment overrides declare themselves

Do not call `os.environ.get` for anything that can change an output. Use the
registry, so reading an override *is* declaring it:

```python
from cdf.provenance import env_float

ratio = env_float("PCB_AOI_THT_HEIGHT_RATIO", 0.9)
```

Every read is recorded with its name, default, raw string and effective value,
and `overridden` says whether it actually changed anything. Unset variables are
recorded too, so the record distinguishes *not overridden* from *never consulted*.

`env_str`, `env_float` and `env_bool` all fall back to the default on a malformed
value rather than raising — a stray shell export should not kill a render — while
keeping the raw string so the attempt is still visible.

## Guarantees

* **It never fails a render.** Each collector is isolated; one that raises is
  reported in `errors` and the rest still run.
* **It never writes a file of its own.** The record travels inside the manifest
  documents the run already writes; a render that wrote nothing gains no
  directory and no file from provenance, and `render_async` is observably
  expected not to conjure a directory its executor never used.
* **It never blocks on huge inputs.** Scene layers above 256 MB record size and
  mtime with `sha256: null` and an explicit `sha256_skipped` reason, rather than
  silently omitting the layer; text layers above 64 MB have only their header
  scanned for arcs and say so.
* **It never opens a USD runtime or a remote store.** The layer chain is what a
  text scan of `.usda` files can see (`subLayers`, `references`, `payload`).
  A remote (`s3://`) or missing layer is named and declared `unverified`; a
  binary layer is hashed but marked `not_traversed`. Arcs only a composed stage
  knows (variants, inherits, arcs inside crate files) are not claimed.
* **It declares truncation.** An artifact list capped at `limit` says so; a
  silent cap reads as full coverage.

## Reading one

Open `dataset_index.json` in the pass root and read `provenance`. Start with
`versions` and `config` (what was run, from which file), then
`scene.source_export.createdDate` (which board), then `code` (which commit, and
whether the tree was clean), then `environment` (what the config could not tell
you), and finish with `unverified`. A single trigger's `metadata.json` carries
the summary of the same record, so a frame handed over on its own still names
its commit, versions, config and board.
