#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build Ken Brakke's Surface Evolver 2.70 from source WITH dynamic-load support
# (-DENABLE_DLL), which the Debian/Ubuntu `evolver` package does NOT ship.
#
# Why we need our own build
# -------------------------
# We want to hand Evolver a component keep-out surface as a dense-grid SDF
# evaluated by compiled C, instead of a giant `min-of-max` formula string
# (see pcb_aoi/solder_evolver/docs/keepout.md). Evolver supports exactly
# that via `load_library "foo.so"` + `constraint N formula: my_func`, but only
# when compiled with -DENABLE_DLL. The packaged binary is not:
#
#   $ echo q | evolver bad.fe        # (Debian 2.70+ds-8)
#   SYNTAX ERROR 1106: Illegal token 'dynamic_load_func'.
#   $ strings $(which evolver) | grep 'not compiled for dynamic'
#   This Evolver not compiled for dynamic load libraries.
#
# Output
# ------
# Installs the binary to  pcb_aoi/solder_evolver/vendor/surface-evolver/evolver,
# which is where `pcb_aoi.solder_evolver.backend.find_evolver()` looks.
#
# Source
# ------
# The upstream 2.70 source is VENDORED, pristine, under upstream/ -- see
# upstream/PROVENANCE.md for origin, checksum and licensing. The build is
# hermetic: no network, no tarball, no mirror variables. Local modifications
# live in patches/ and are applied into .build/ here, never to upstream/.
#
# Usage
# -----
#   ./build.sh                 # patch, build, install (idempotent)
#   ./build.sh --check         # report status only; never builds. exit 0 = ready
#   ./build.sh --clean         # rebuild from scratch
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$(cd "$HERE/../.." && pwd)"           # pcb_aoi/solder_evolver
WORK="$HERE/.build"
PREFIX="$ENGINE/vendor/surface-evolver"
UPSTREAM="$HERE/upstream"

# Evolver 2.70 is 2013-era C. Modern gcc defaults reject a lot of it, so we pin
# gnu89 and demote the K&R-isms to warnings. -fcommon is required: the source
# relies on tentative definitions in headers included by many .c files.
CFLAGS_EV="-O2 -DLINUX -DOOGL -DENABLE_DLL -fcommon -std=gnu89"
CFLAGS_EV="$CFLAGS_EV -Wno-implicit-int -Wno-implicit-function-declaration"
CFLAGS_EV="$CFLAGS_EV -Wno-return-type -Wno-int-conversion"

log() { printf '\033[1m[evolver-build]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[evolver-build] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

for arg in "$@"; do
  case "$arg" in
    --clean) log "cleaning $WORK"; rm -rf "$WORK" ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) die "unknown option: $arg (try --help)" ;;
  esac
done

# --check: report what's on this machine and whether the solver can use it.
# Exits 0 only when a DLL-capable binary is installed and ready.
if [[ -n "${CHECK_ONLY:-}" ]]; then
  rc=0
  if [[ -x "$PREFIX/evolver" ]]; then
    if grep -qa "not compiled for dynamic load libraries" "$PREFIX/evolver"; then
      printf 'vendored build : %s (NO dynamic-load support -- rebuild)\n' "$PREFIX/evolver"; rc=1
    else
      printf 'vendored build : %s (dynamic-load OK)\n' "$PREFIX/evolver"
    fi
  else
    printf 'vendored build : ABSENT -- run %s\n' "${BASH_SOURCE[0]}"; rc=1
  fi
  sys="$(command -v evolver || true)"
  if [[ -n "$sys" ]]; then
    if grep -qa "not compiled for dynamic load libraries" "$sys"; then
      printf 'system evolver : %s (no dynamic-load -- expected; cannot run grid-SDF datafiles)\n' "$sys"
    else
      printf 'system evolver : %s (dynamic-load OK)\n' "$sys"
    fi
  else
    printf 'system evolver : none on $PATH\n'
  fi
  printf 'override       : SOLDER_EVOLVER_BIN=%s\n' "${SOLDER_EVOLVER_BIN:-<unset>}"
  exit $rc
fi

command -v gcc >/dev/null || die "gcc not found. apt-get install build-essential"
[[ -f /usr/include/dlfcn.h ]] || die "/usr/include/dlfcn.h missing; -DENABLE_DLL needs it (apt-get install libc6-dev)"

mkdir -p "$WORK"

# ---------------------------------------------------------------- stage + patch
[[ -d "$UPSTREAM/src" ]] || die "vendored source missing: $UPSTREAM/src (see upstream/PROVENANCE.md)"
SRC="$WORK/src"
if [[ ! -d "$SRC" ]]; then
  log "staging vendored source"
  cp -r "$UPSTREAM/src" "$SRC"

  log "applying patches"
  for p in "$HERE"/patches/*.patch; do
    [[ -e "$p" ]] || continue
    # patches are written against a/src/... b/src/...; apply from $WORK root
    ( cd "$WORK" && patch -p1 -d . --forward -i "$p" ) \
      || die "failed to apply $(basename "$p")"
    log "  applied $(basename "$p")"
  done
fi

# Copy OUR sources in on EVERY build, not just the first unpack. They are ours,
# they are tiny, and patches never touch them -- whereas leaving them behind a
# "only if .build is missing" guard means an edit to sdf_collider.c silently
# does not reach the compiler until the next --clean. That cost me a debug cycle.
if compgen -G "$HERE/src/*.c" >/dev/null; then
  log "syncing local sources"
  for f in "$HERE"/src/*.c; do
    cp "$f" "$SRC/" && log "  + $(basename "$f")"
  done
  # ...and force the TU that #includes them to recompile. The Makefile has no
  # rule saying userfunc.o depends on sdf_collider.c, so make happily keeps a
  # stale object and the build "succeeds" with the old code silently linked in.
  rm -f "$SRC/userfunc.o"
fi

# ---------------------------------------------------------------- build
# NOTE: pass GRAPH/GRAPHLIB/CFLAGS on the make COMMAND LINE, not by editing the
# Makefile. The `evolver:` rule lists $(GRAPH) as a prerequisite, and make
# expands prerequisite lists immediately when the rule is read -- a GRAPH=
# assignment appended to the Makefile comes too late and nulgraph.o silently
# never gets built ("cannot find nulgraph.o" at link).
#
# nulgraph = headless. We render via our own USD/Isaac viewers, never Evolver's
# built-in graphics, so there is no reason to pull in X11/GLUT.
log "building (this takes ~1 min)"
make -C "$SRC" -j"$(nproc)" \
  CC="gcc" \
  CFLAGS="$CFLAGS_EV" \
  GRAPH="nulgraph.o" \
  GRAPHLIB="-ldl" \
  evolver >"$WORK/build.log" 2>&1 \
  || { tail -30 "$WORK/build.log"; die "build failed; full log at $WORK/build.log"; }

[[ -x "$SRC/evolver" ]] || die "build reported success but no binary at $SRC/evolver"

# ---------------------------------------------------------------- install
log "installing to $PREFIX"
mkdir -p "$PREFIX"
install -m 0755 "$SRC/evolver" "$PREFIX/evolver"
# Upstream's `fe` sample datafiles are not vendored (see PROVENANCE.md); the
# solver only ever runs datafiles we generate.

# ---------------------------------------------------------------- verify
log "verifying dynamic-load support"
TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT
cat >"$TMPD/probe.c" <<'EOF'
#define FUNC_VALUE 1
#define MAXCOORD 6
#define REAL double
struct dstack { REAL value; REAL deriv[2*MAXCOORD]; REAL second[2*MAXCOORD][2*MAXCOORD]; };
void probe_fn(int mode, REAL *x, struct dstack *s)
{ s->value = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] - 1.0;
  if (mode == FUNC_VALUE) return;
  s->deriv[0] = 2*x[0]; s->deriv[1] = 2*x[1]; s->deriv[2] = 2*x[2]; }
EOF
gcc -O1 -fPIC -shared "$TMPD/probe.c" -o "$TMPD/probe.so" -lm
cat >"$TMPD/probe.fe" <<'EOF'
load_library "probe.so"
constraint 1
formula: probe_fn
vertices
1  3.0  0.0  0.0  constraint 1
edges
faces
read
printf "PROBE_RADIUS %.6f\n", sqrt(vertex[1].x^2 + vertex[1].y^2 + vertex[1].z^2);
quit 0
EOF
OUT="$(cd "$TMPD" && echo "" | "$PREFIX/evolver" probe.fe 2>&1 || true)"
if ! grep -q "PROBE_RADIUS 1.000000" <<<"$OUT"; then
  printf '%s\n' "$OUT" >&2
  die "dynamic-load verification FAILED (vertex was not projected onto the .so's level set)"
fi

log "OK -- $PREFIX/evolver built with dynamic-load support"
log "     point the solver at it with: export SOLDER_EVOLVER_BIN=$PREFIX/evolver"
