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

ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.1.0

# ---------------------------------------------------------------------------
# Stage 1 (evolver-builder): compile the DLL-capable Surface Evolver binary.
#
# The final image must ship a ready-to-run Evolver (users never compile in the
# container) and must ship NO compiler, so the build happens here in a
# throwaway stage and only the ~2MB binary is copied across. The build is
# hermetic: the 2.70 source is vendored under third_party/evolver/upstream/,
# so no network access is needed beyond apt.
#
# The COPY preserves the pcb_aoi/solder_evolver/third_party/ path layout
# because build.sh derives its install prefix from its own location: the
# binary lands at /evolver/pcb_aoi/solder_evolver/vendor/surface-evolver/.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS evolver-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY pcb_aoi/solder_evolver/third_party/ /evolver/pcb_aoi/solder_evolver/third_party/

# build.sh self-verifies dynamic-load support (compiles and loads a probe .so);
# --check re-asserts the installed binary is present and DLL-capable.
RUN bash /evolver/pcb_aoi/solder_evolver/third_party/evolver/build.sh && \
    bash /evolver/pcb_aoi/solder_evolver/third_party/evolver/build.sh --check

# ---------------------------------------------------------------------------
# Stage 2: the shipped image.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE}

USER root

# Snapshot the base image's installed-package set before our first apt-get
# install. The final OSS-sources layer diffs against this to report the full
# new-package set (including transitive deps), so we can widen the source
# collection if OSS later asks for the full closure.
RUN mkdir -p /tmp/oss && \
    dpkg-query -W -f='${Package}\n' | sort > /tmp/oss/pkgs-base && \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Host-Python runtime dependencies. Direct pins live in requirements.txt; the
# constraints file freezes their transitive resolution so installed wheels,
# notices, SBOMs, and corresponding source archives stay aligned.
COPY requirements.txt /tmp/cdf-requirements.txt
COPY third_party/python-runtime-constraints.txt /tmp/cdf-constraints.txt
COPY third_party/python-source-archives.lock /tmp/python-source-archives.lock
COPY scripts/docker/collect_python_sources.py /tmp/collect_python_sources.py

RUN pip3 install --no-cache-dir --break-system-packages \
        -r /tmp/cdf-requirements.txt \
        -c /tmp/cdf-constraints.txt 2>/dev/null \
    || pip3 install --no-cache-dir \
        -r /tmp/cdf-requirements.txt \
        -c /tmp/cdf-constraints.txt

# The simulated solder solver (solver: surface_evolver) runs IN-JOB, inside
# Isaac's own bundled interpreter -- the one python.sh launches -- so its two
# compiled geometry wheels must live in THAT site-packages, not the host
# python above (Isaac already bundles numpy/trimesh/scipy; these two are the
# only gaps). Pins come from the same requirements lock.
#
# NOTE: this is a stopgap IN PLACE OF a full Isaac Sim extension. The intended
# end state is a dedicated extension that bundles the solver and its
# dependencies into the runtime properly; until that exists, baking the wheels
# into the image is what makes `solver: surface_evolver` work out of the box.
RUN /isaac-sim/kit/python/bin/python3 -m pip install --no-cache-dir \
        "$(grep -E '^manifold3d==' /tmp/cdf-requirements.txt)" \
        "$(grep -E '^warp-lang==' /tmp/cdf-requirements.txt)"

# Carry the exact source corresponding to every Python distribution added by
# requirements.txt. Most entries are PyPI sdists; packages that publish wheels
# without an sdist use a checksum-pinned upstream release archive. The
# collector also fails the build if the installed versions drift from the
# source lock.
RUN python3 /tmp/collect_python_sources.py \
        --lock /tmp/python-source-archives.lock \
        --output /usr/share/oss-sources/python && \
    rm -f /tmp/collect_python_sources.py \
        /tmp/python-source-archives.lock

# OSS compliance: ship source code of every Debian package the final image
# has on top of the base container at /usr/share/oss-sources/debian/. Python
# package sources are handled separately above because several runtime wheels
# contain native binaries and installing a wheel does not provide corresponding
# source for those binaries.
#
# We compute the new-package set by diffing dpkg-query output against the
# baseline snapshot from the first RUN layer. NEW_PACKAGES.txt records the
# full diff (including transitive deps apt pulled in). For each new binary we
# resolve its source package via `dpkg-query -f '${source:Package}'`, dedupe,
# and fetch the .dsc + tarballs with `apt-get source`. No hand-curated list,
# so any future transitive-dep change in the base picks up source coverage
# automatically.
RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's/^# *deb-src/deb-src/' /etc/apt/sources.list; \
    fi; \
    for f in /etc/apt/sources.list.d/*.sources; do \
        [ -f "$f" ] || continue; \
        if ! grep -q '^Types:.*deb-src' "$f"; then \
            sed -i 's/^Types: deb$/Types: deb deb-src/' "$f"; \
        fi; \
    done; \
    apt-get update; \
    mkdir -p /usr/share/oss-sources/debian; \
    dpkg-query -W -f='${Package}\n' | sort > /tmp/oss/pkgs-after; \
    comm -13 /tmp/oss/pkgs-base /tmp/oss/pkgs-after \
        > /usr/share/oss-sources/debian/NEW_PACKAGES.txt; \
    echo "=== OSS: new Debian packages in final image (vs base) ==="; \
    cat /usr/share/oss-sources/debian/NEW_PACKAGES.txt; \
    src_pkgs=$(while read -r p; do \
        [ -z "$p" ] && continue; \
        dpkg-query -W -f='${source:Package}\n' "$p"; \
    done < /usr/share/oss-sources/debian/NEW_PACKAGES.txt | sort -u); \
    echo "=== OSS: unique source packages to collect ==="; \
    echo "$src_pkgs" | tee /usr/share/oss-sources/debian/SOURCE_PACKAGES.txt; \
    cd /usr/share/oss-sources/debian; \
    for src in $src_pkgs; do \
        apt-get source --download-only "$src"; \
    done; \
    cd /; \
    rm -rf /tmp/oss; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Stage repo into /workspace owned by the base image's default user (uid 1234).
# /home/isaac-sim doesn't exist in this base — isaac-sim's HOME is /isaac-sim
# (the install dir itself), so we keep the repo under a clean /workspace path.
# Copy only the supported runtime, shipped documentation, and the retained
# functional harness. Dockerfile.dockerignore mirrors this allowlist and filters
# local caches and generated files inside the selected directories.
COPY --chown=isaac-sim:isaac-sim cdf/ /workspace/cdf-simulation/cdf/
COPY --chown=isaac-sim:isaac-sim pcb_aoi/ /workspace/cdf-simulation/pcb_aoi/
COPY --chown=isaac-sim:isaac-sim scripts/ /workspace/cdf-simulation/scripts/
COPY --chown=isaac-sim:isaac-sim apps/ /workspace/cdf-simulation/apps/
COPY --chown=isaac-sim:isaac-sim third_party/ /workspace/cdf-simulation/third_party/
COPY --chown=isaac-sim:isaac-sim docs/ /workspace/cdf-simulation/docs/
COPY --chown=isaac-sim:isaac-sim skills/ /workspace/cdf-simulation/skills/
COPY --chown=isaac-sim:isaac-sim tests/functional/ /workspace/cdf-simulation/tests/functional/
COPY --chown=isaac-sim:isaac-sim pyproject.toml requirements.txt LICENSE README.md COOKBOOK.md /workspace/cdf-simulation/

# Install only the Surface Evolver binary built in the builder stage.
# Host vendor/ and .build/ products are excluded by Dockerfile.dockerignore.
# backend.find_evolver() uses this package-relative path at runtime.
COPY --from=evolver-builder --chown=isaac-sim:isaac-sim \
    /evolver/pcb_aoi/solder_evolver/vendor/ \
    /workspace/cdf-simulation/pcb_aoi/solder_evolver/vendor/

# Gate the image build on the binary actually being present and DLL-capable.
# --check never compiles; it exits non-zero unless the vendored binary is
# ready, so a broken builder stage is a loud build failure, not a runtime one.
RUN bash /workspace/cdf-simulation/pcb_aoi/solder_evolver/third_party/evolver/build.sh --check

# The end-to-end gate: the simulated solver must REGISTER inside Isaac's
# interpreter AS THE RUNTIME LAUNCHES IT -- via python.sh, which layers Kit's
# bundled packages (yaml etc.) onto the kit python the wheels above landed in;
# the raw kit binary lacks those and is not what jobs run under. One assertion
# covers the wheels, the vendored binary, and the whole import chain: if a
# base-image bump ever drops a dependency, the IMAGE BUILD fails here instead
# of users meeting an unknown-solver error at runtime.
RUN cd /workspace/cdf-simulation && /isaac-sim/python.sh -c "\
from pcb_aoi.solder_generation import registered_solver_ids; \
ids = registered_solver_ids(available_only=True); \
assert 'surface_evolver' in ids, f'surface_evolver missing from {ids}'; \
print('solder solvers available in-job:', ids)"

RUN chmod 755 /workspace/cdf-simulation/scripts/docker/entrypoint.sh

# Surface base-image LICENSE, a VERSION marker and the source commit at the WORKDIR.
# VERSION is derived from pyproject.toml. GIT_COMMIT is passed by the CI build
# (--build-arg GIT_COMMIT=$CI_COMMIT_SHA); the image ships no .git, and a tag is
# mutable, so this file is the only immutable identity a run inside the
# container can record (cdf.provenance reads it).
ARG GIT_COMMIT=unknown
RUN cp /isaac-sim/LICENSE.txt /workspace/cdf-simulation/BASE_IMAGE_LICENSE.txt && \
    awk -F'"' '/^version *=/ {print $2; exit}' /workspace/cdf-simulation/pyproject.toml \
        > /workspace/cdf-simulation/VERSION && \
    printf '%s\n' "${GIT_COMMIT}" > /workspace/cdf-simulation/GIT_COMMIT && \
    chown isaac-sim:isaac-sim \
        /workspace/cdf-simulation/BASE_IMAGE_LICENSE.txt \
        /workspace/cdf-simulation/VERSION \
        /workspace/cdf-simulation/GIT_COMMIT

USER isaac-sim

# Share Kit's cv2 with host-python for --entrypoint python3 stages.
ENV PYTHONPATH=/isaac-sim/exts/omni.pip.compute/pip_prebundle:${PYTHONPATH}

WORKDIR /workspace/cdf-simulation

# Runtime options:
# - default: legacy Kit `--exec` command
# - `job-script`: authored Python job
# - `config`: auto-routed versioned or legacy YAML job
# - `--entrypoint python3`: host-only stages such as register/crop
ENTRYPOINT ["/workspace/cdf-simulation/scripts/docker/entrypoint.sh"]
