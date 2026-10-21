# Surface Evolver, built from source with dynamic-load support

`build.sh` builds Ken Brakke's Surface Evolver 2.70 from the VENDORED source
under `upstream/` (pristine; see `upstream/PROVENANCE.md`), patches it, builds
with `-DENABLE_DLL`, and installs to `pcb_aoi/solder_evolver/vendor/surface-evolver/`.
The build is hermetic — no network.

```bash
pcb_aoi/solder_evolver/third_party/evolver/build.sh          # ~1 min
pcb_aoi/solder_evolver/third_party/evolver/build.sh --clean  # from scratch
```

Neither `.build/` (staged sources, objects) nor `vendor/` (the binary) is
committed; `upstream/` is.

## Why we can't use the distro package

We want the component keep-out surface handed to Evolver as a **compiled**
function — a dense-grid SDF — instead of a multi-kilobyte `min-of-max` formula
string (see `docs/analytic_representation_alternatives.md`). Evolver supports
exactly this:

```
load_library "component_sdf.so"

constraint 1 nonnegative
formula: component_sdf
```

but only when compiled with `-DENABLE_DLL`. The Debian/Ubuntu `evolver` package
is not:

```
$ strings $(which evolver) | grep 'not compiled for dynamic'
This Evolver not compiled for dynamic load libraries.
$ echo q | evolver uses_dll.fe
SYNTAX ERROR 1106: Illegal token 'dynamic_load_func'.
```

`solder_evolver.backends.evolver.evolver_has_dll()` probes for that string;
`find_evolver()` prefers the vendored build over `$PATH` for this reason.
`tests/test_evolver_dll.py` gates the capability.

## The `.so` interface

From `doc/misc.htm` ("Dynamic load libraries") and `src/extern.h`. The function
takes a mode flag and returns value / first partials / second partials through a
`dstack`. **`MAXCOORD` and `REAL` must match the Evolver build** (6 and `double`
for ours) or the struct layout disagrees and you get silent garbage.

```c
#define FUNC_VALUE 1
#define FUNC_DERIV 2
#define FUNC_SECOND 3
#define MAXCOORD 6      /* src/model.h */
#define REAL double     /* src/include.h -- long double if -DLONGDOUBLE */

struct dstack { REAL value;
                REAL deriv[2*MAXCOORD];
                REAL second[2*MAXCOORD][2*MAXCOORD]; };

void component_sdf(int mode, REAL *x, struct dstack *s)
{ s->value = /* f(x[0], x[1], x[2]) */;
  if (mode == FUNC_VALUE) return;
  s->deriv[0] = /* df/dx */; ...
  if (mode == FUNC_DERIV) return;
  s->second[0][0] = /* d2f/dx2 */; ... }
```

```bash
gcc -O2 -fPIC -shared component_sdf.c -o component_sdf.so -lm
```

Evolver searches the current directory and `$EVOLVERPATH` for the library.
Arguments are **implicit** — the function is called with the point coordinates
and is only legal where Evolver expects a function of position (constraint
`formula:`, boundary `x1:`, named-method integrands, metrics). It is *not* a
general expression you can `printf`; `vertex[1].my_func` is a syntax error.

Note that we supply analytic derivatives. This is a real advantage over a
formula string: Evolver gets an exact gradient of a true distance function
(`|grad f| = 1`) rather than differentiating a kinky `maximum()` tree.

Upstream labels dynamic loading "experimental ... the interface described here
may change" — but 2.70 (2013) is the last release, so there is nothing to drift.

## Patches

- `patches/0001-nulgraph-set_graphics_title.patch` — upstream build fix.
  `evalmore.c` references `set_graphics_title()` unconditionally, but only
  `glutgraph.c` defines it, so any headless (`GRAPH=nulgraph.o`) build fails to
  link. Adds a no-op stub. Not related to dynamic loading; it just isn't
  possible to build a graphics-free Evolver upstream without it.

## Build notes (things that will bite you)

- **Pass `GRAPH`/`GRAPHLIB`/`CFLAGS` on the `make` command line**, never by
  editing the Makefile. The `evolver:` rule lists `$(GRAPH)` as a prerequisite,
  and make expands prerequisite lists when the rule is *read* — a `GRAPH=` line
  appended to the Makefile arrives too late, `nulgraph.o` is never built, and
  you get a confusing `cannot find nulgraph.o` at link time.
- **`-std=gnu89 -fcommon`** are required. This is 2013-era C: implicit `int`,
  K&R declarations, and tentative definitions in shared headers.
- `-DOOGL` is harmless and kept (geomview output support); graphics are
  `nulgraph` because we render through USD/Isaac, never Evolver's own viewer.
- The upstream tree is vendored pristine under `upstream/`; never edit it in
  place — local changes are patches in `patches/`, applied into `.build/`.

## Provenance

See `upstream/PROVENANCE.md` — origin, the sha256 pin the ingest was verified
against, the licensing determination (Debian distributes this source as public
domain per the copyright holder's declaration; file-level notices are
enumerated there), and exactly which parts of the release were vendored.
