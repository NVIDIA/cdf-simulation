# Surface Evolver 2.70 — vendored upstream source

**Do not edit the vendored release files under `upstream/`.** This
`PROVENANCE.md` file is repository-maintained metadata; the remaining files
form a pristine, unmodified subset of the upstream release. Local source
changes belong in `../patches/`, applied at build time by `../build.sh`.
Keeping the source tree pristine is what keeps the patches auditable.

## Origin

| | |
|---|---|
| Project | Surface Evolver, Kenneth A. Brakke |
| Version | 2.70 (August 2013 — the final upstream release; the project is inactive) |
| Source | `https://kenbrakke.com/evolver/downloads/evolver-2.70.tar.gz` |
| SHA-256 | `be7d60d9eb690f1fb124e2ecc12237a7a35be1935bfa5e92be3fe912207a3b6b` |
| Retrieved | 2026-07-31 (by the previous download-at-build flow; verified against the pin above at ingest, 2026-08-12) |

## What is vendored (and what is not)

Vendored: `src/` (the complete C source, 118 files), `README.unix.txt`
(upstream build instructions), `evolver.1` (man page).

Omitted, retrievable from the pinned tarball: `doc/` (HTML manual, 3.2 MB),
`fe/` (114 sample datafiles), `manual270.pdf`, `EvolverDoc.html`. None are
needed to build or run the solver.

## Required attribution

This work includes code from Surface Evolver, developed by Kenneth A. Brakke.

- Project: [Surface Evolver](https://kenbrakke.com/evolver/evolver.html)
- Version/source: Surface Evolver 2.70 (August 2013), from the upstream
  [`evolver-2.70.tar.gz`](https://kenbrakke.com/evolver/downloads/evolver-2.70.tar.gz)
  release
- Files incorporated: the complete upstream `src/` tree (118 files),
  `README.unix.txt`, and `evolver.1`; repository changes are maintained
  separately in `../patches/`

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

## Licensing

The tarball ships **no standalone license file**, but individual source files
contain component-specific notices. Debian's `debian/copyright` for the
`evolver` package records the copyright holder's declaration for the core:

> "These files are public domain and come with NO WARRANTY of any kind."

Debian marks the main source **public domain** and separately identifies:

- `src/f2c.h` — ATT license (AT&T / Lucent / Bellcore, 1990-1997): free use,
  copying and modification for any purpose provided copyright notices are
  preserved in supporting documentation.

Reference: `https://metadata.ftp-master.debian.org/changelogs/main/e/evolver/stable_copyright`

That package-level description does not supersede the component-specific
notices below. In particular, it is not a determination that the embedded
Mersenne Twister code is public domain.

File-level notices an auditor will find in the tree, stated here so none of
them is a surprise:

- `src/ytab.c`, `src/ytab.h` — Bison-generated parser carrying the FSF
  skeleton notice **including the special exception**: "when this file is
  copied by Bison into a Bison output file, you may use that output file
  without restriction."
- `src/utility.c` — embeds MT19937B, attributed to Makoto Matsumoto and
  Takuji Nishimura and citing their January 1998 paper. Lines 4265-4268
  contain the notice "Matsumoto and Nishimura consent to GNU General Public
  Licence for this code." The notice specifies no GPL version, and the
  archive contains no accompanying GPL license text. The notice remains
  unchanged in the upstream file and is reproduced in
  `third_party/LICENSE-3rd-party.txt`. We do not infer that Debian's core
  public-domain description or the licensing of a later Mersenne Twister
  implementation supersedes this notice. Applicable licensing treatment
  requires an OSS/legal determination; this provenance record does not
  establish one.
- `src/sdrv.c` — f2c-translated Yale Sparse Matrix Package (YSMP) routines,
  including modifications marked "copyright (c) 1990 by Tamar Schlick". The
  YSMP routines are MIT-licensed. The copyright remains in the pristine source,
  and the complete attribution and MIT text are included in
  `third_party/LICENSE-3rd-party.txt`.

## Why vendored

The previous flow downloaded the tarball from a personal website at build
time (SHA-pinned, with a manual mirror variable for offline runners). Upstream
is permanently frozen at 2.70, the archive is small, and a hermetic build
removes both the network dependency and the download-and-execute pattern from
the build documentation. See `../build.sh` for the build itself.
