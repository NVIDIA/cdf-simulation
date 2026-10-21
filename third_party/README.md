# Third-party software

This directory records third-party software added by `cdf-simulation` on
top of the NVIDIA Isaac Sim base container. It separates shipped runtime
dependencies from CI-only tools, base-image components, optional packages,
and vendored source so the OSRB scope is unambiguous.

## Shipped Python runtime

[`requirements.txt`](../requirements.txt) is the source of truth for the
direct Python packages installed into the runtime image. The Dockerfile copies
that file and installs it with `pip3 install -r`.
[`python-runtime-constraints.txt`](python-runtime-constraints.txt) freezes the
transitive resolution used for a release image. `manifold3d` and `warp-lang`
are additionally installed from the direct-requirements file into Isaac Sim's
bundled Python interpreter for in-job solder solving.

Direct packages:

| Package | Version | License | Use |
|---|---:|---|---|
| numpy | 2.4.6 | BSD-3-Clause and bundled notices | Numerical arrays and image-registration math |
| Pillow | 12.3.0 | MIT-CMU | Image loading and conversion |
| pydantic | 2.13.4 | MIT | Configuration and job-model validation |
| PyYAML | 6.0.3 | MIT | YAML configuration loading |
| usd-core | 26.5 | LicenseRef-TOST-1.0 | OpenUSD Python APIs outside Isaac Sim |
| boto3 | 1.40.61 | Apache-2.0 | S3/object-storage integration |
| manifold3d | 3.5.2 | Apache-2.0 | Mesh boolean operations for solder geometry |
| trimesh | 4.12.2 | MIT | Mesh I/O, repair, and primitives |
| scipy | 1.18.0 | BSD-3-Clause and bundled notices | Convex hulls and numerical fitting |
| coacd | 1.0.11 | MIT; bundles CDT under MPL-2.0 | Approximate convex decomposition |
| scikit-image | 0.26.0 | BSD-3-Clause and bundled notices | Marching-cubes mesh generation |
| warp-lang | 1.15.0 | Apache-2.0 and bundled notices | SDF geometry kernels with CPU/CUDA devices |

The transitive closure below was resolved from those pins for CPython 3.12 on
Linux x86_64 on 2026-09-28 and is frozen by
`python-runtime-constraints.txt`. The build-time CycloneDX SBOM verifies the
inventory installed in a particular image.

| Package | Resolved version | License |
|---|---:|---|
| annotated-types | 0.8.0 | MIT |
| botocore | 1.40.76 | Apache-2.0 |
| ImageIO | 2.38.0 | BSD-2-Clause |
| jmespath | 1.1.0 | MIT |
| lazy-loader | 0.6 | BSD-3-Clause |
| networkx | 3.7 | BSD-3-Clause |
| pydantic-core | 2.46.4 | MIT |
| python-dateutil | 2.9.0.post0 | Apache-2.0 OR BSD-3-Clause |
| s3transfer | 0.14.0 | Apache-2.0 |
| six | 1.17.0 | MIT |
| tifffile | 2026.9.20 | BSD-3-Clause |
| typing-extensions | 4.16.0 | PSF-2.0 |
| typing-inspection | 0.4.4 | MIT |
| urllib3 | 2.8.0 | MIT |

[`LICENSE-3rd-party.txt`](LICENSE-3rd-party.txt) contains the license and
attribution notices extracted from the exact 26 wheels/source distributions
in that resolution, including notices for software bundled into binary
wheels.

### Corresponding Python source

The release image carries a checksum-verified source archive for every direct
and transitive Python distribution in the constrained runtime resolution at:

```text
/usr/share/oss-sources/python/
```

[`python-source-archives.lock`](python-source-archives.lock) records each
distribution, version, archive filename, SHA-256 digest, and source URL. The
Docker build verifies that the lock matches the installed runtime before it
downloads the archives. `MANIFEST.tsv` in the image records the same provenance
for the files delivered to users.

Most packages publish a source distribution on PyPI. `usd-core`, `coacd`, and
`warp-lang` do not publish an sdist for these versions, so their entries use
checksum-pinned source snapshots of the corresponding upstream release tags.
CoACD's release archive records CDT as a Git submodule rather than embedding
its files, so the lock also carries the exact CDT commit used by CoACD as a
supplemental source archive. CoACD compiles CDT into its native extension; the
CDT sources are governed by MPL-2.0, and the complete license is included in
[`LICENSE-3rd-party.txt`](LICENSE-3rd-party.txt).

## Source-code locations in the release image

| Scope | Location |
|---|---|
| CDF-Simulation application source and documentation | `/workspace/cdf-simulation/` |
| Vendored Surface Evolver source, NVIDIA patches, and build integration | `/workspace/cdf-simulation/pcb_aoi/solder_evolver/third_party/evolver/` |
| Python runtime corresponding-source archives | `/usr/share/oss-sources/python/` |
| Debian source packages added above the base image | `/usr/share/oss-sources/debian/` |
| Isaac Sim base-image license | `/workspace/cdf-simulation/BASE_IMAGE_LICENSE.txt` |

The applicable Isaac Sim base-container approval and base source-code reference
are recorded in the release's OSRB request; the repository does not duplicate
the base container's source payload.

## Dependencies outside the pinned runtime set

### Isaac Sim base image

The shipped image is based on `nvcr.io/nvidia/isaac-sim:6.1.0`. Isaac Sim and
packages it already supplies, including the runtime `cv2` module, are governed
by the base image's licensing and OSRB coverage. The base image is obtained
from NVIDIA's official distribution channel rather than redistributed by this
repository. Its `LICENSE.txt` is surfaced in the final image as
`/workspace/cdf-simulation/BASE_IMAGE_LICENSE.txt`.

### Build-only Surface Evolver stage

The multi-stage Docker build uses `debian:bookworm-slim` to compile the
Surface Evolver executable. The builder image and its filesystem are not part
of the final release image; only the resulting executable is copied into the
Isaac Sim runtime image. The release OSRB request records this build-only stage
and asks OSRB to confirm that it does not require separate base-image approval.

### CI-only Python packages

[`requirements-test.txt`](../requirements-test.txt) contains packages used by
GitLab CI and tests, not installed into the runtime image:

| Package | Version | Role |
|---|---:|---|
| opencv-python-headless | 5.0.0.93 | Host-side image tests; runtime OpenCV comes from Isaac Sim |
| pip-audit | 2.10.1 | Dependency audit and CycloneDX SBOM generation |
| unittest-xml-reporting | 4.0.0 | JUnit-compatible unit-test reports |

Their transitive packages belong to the CI environment and are recorded in
`python-test-sbom.cdx.json`, not in the runtime notice bundle.

### Other third-party source and system packages

#### Surface Evolver

This work includes code from Surface Evolver, developed by Kenneth A. Brakke.

- Project: [Surface Evolver](https://kenbrakke.com/evolver/evolver.html)
- Version/source: Surface Evolver 2.70 (August 2013), from the upstream
  `evolver-2.70.tar.gz` release
- Files incorporated: the complete upstream `src/` tree, `README.unix.txt`,
  and `evolver.1`; repository changes are maintained separately as patches

The core Surface Evolver code is public domain. Debian records the author's
permission to copy, modify, and redistribute it freely. The original code is
provided without warranty.

The incorporated source includes `src/f2c.h`, the Fortran-to-C header:

> Copyright 1990–1997 AT&T, Lucent Technologies, and Bellcore.
>
> Permission to use, copy, modify, and distribute this software and its
> documentation for any purpose and without fee is hereby granted, provided
> that the above copyright notice appear in all copies and that both that the
> copyright notice and this permission notice and warranty disclaimer appear
> in supporting documentation, and that the names of AT&T, Bell Laboratories,
> Lucent or Bellcore or any of their entities not be used in advertising or
> publicity pertaining to distribution of the software without specific,
> written prior permission.
>
> AT&T, Lucent and Bellcore disclaim all warranties with regard to this
> software, including all implied warranties of merchantability and fitness.
> In no event shall AT&T, Lucent or Bellcore be liable for any special,
> indirect or consequential damages or any damages whatsoever resulting from
> loss of use, data or profits, whether in an action of contract, negligence
> or other tortious action, arising out of or in connection with the use or
> performance of this software.

The incorporated `src/sdrv.c` also contains modified routines from the Yale
Sparse Matrix Package (YSMP). The file retains the following copyright:

> Copyright (c) 1990 by Tamar Schlick.

The YSMP routines are licensed under the MIT License. The complete attribution
and license text are included in
[`LICENSE-3rd-party.txt`](LICENSE-3rd-party.txt).

The exact source URL and checksum, licensing findings, file inventory, and
local changes are documented in
[`pcb_aoi/solder_evolver/third_party/evolver/upstream/PROVENANCE.md`](../pcb_aoi/solder_evolver/third_party/evolver/upstream/PROVENANCE.md).

Debian packages added after the base-image snapshot are handled separately.
The Docker build collects their corresponding source packages under
`/usr/share/oss-sources/debian/` in the final image.

`aiohttp` is not installed by this repository. It was previously carried as
an Isaac Sim CVE-remediation override and is not part of the current
application dependency inventory.

## Regeneration and verification

When `requirements.txt`, `requirements-test.txt`, or the base image changes:

1. Resolve both requirement sets for CPython 3.12/Linux, matching the runtime
   container platform.
2. Update `python-runtime-constraints.txt` and
   `python-source-archives.lock`, including checksums for every corresponding
   source archive.
3. Regenerate `LICENSE-3rd-party.txt` from the selected runtime wheels and
   source archives, preserving bundled-component notices.
4. Run the `python-dependency-audit` CI job. It audits both requirement sets
   and publishes `python-sbom.cdx.json` and `python-test-sbom.cdx.json`.
5. Compare the runtime SBOM, source-archive manifest, this document, and the
   notice bundle.

Do not generate the release inventory from a developer's host environment;
PEP 508 markers and available binary wheels can produce a different closure.
