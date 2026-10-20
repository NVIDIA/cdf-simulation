/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SDF collider for Surface Evolver's user-function slots.
 * ======================================================
 *
 * Gives Evolver a dense-grid signed distance field it can use as a level-set
 * constraint, so a keep-out obstacle of ARBITRARY shape costs O(1) per
 * evaluation instead of a multi-kilobyte `min-of-max` formula string.
 *
 *     constraint 1 nonnegative
 *     formula: usr3            // usr3 = collider slot 0 (see userfunc.c)
 *
 * Signs follow geometry/sdf.py:  f > 0 outside the solid, f = 0 on its surface,
 * f < 0 inside. So "solder stays out of the component" is exactly f >= 0, which
 * is what `nonnegative` enforces.
 *
 * This file is #included from userfunc.c ON PURPOSE. It inherits REAL and SDIM
 * from Evolver's own headers, which guarantees the float type matches the
 * binary; a separately-compiled unit could disagree (e.g. -DLONGDOUBLE) and the
 * mismatch would be silent numerical garbage, not a link error.
 *
 *
 * ON THE STATIC STATE
 * -------------------
 * Evolver's slots are `REAL (*)(REAL*)` -- a bare function pointer with no
 * context argument. There is nowhere to thread an SdfGrid* through, and C has
 * no closures, so the grid MUST live in static storage: the wrapper functions
 * are a partial application whose captured environment is a file-scope static.
 *
 * That is forced, but its blast radius is not. Everything here is contained in
 * ONE `static SdfCollider g_slot[]` table that nothing outside this file can
 * name, and every operation on it goes through a function taking an explicit
 * `SdfCollider*`. The evaluator (`sdf_eval` and below it `sdf_interp`) is pure
 * and testable with no globals at all; only the thin usrN wrappers touch the
 * table. tests/sdf_probe.c takes advantage of that: it #includes THIS FILE with
 * a shim for Evolver's symbols and exports the evaluator to ctypes, which is
 * the only way to test the second derivatives -- Evolver's datafile language
 * cannot ask for one.
 *
 *
 * GRID FILE FORMAT  (little-endian; write it with tools/write_sdf_grid.py)
 * -----------------------------------------------------------------------
 *   char    magic[8]   "SDFGRID2"
 *   int32   version    2
 *   int32   nx, ny, nz          samples per axis (>= 4 for cubic interpolation)
 *   float64 lo[3]               world position of sample (0,0,0)
 *   float64 hi[3]               world position of sample (nx-1,ny-1,nz-1)
 *   float64 data[nx*ny*nz]      x fastest, then y, then z
 *
 * The BOUNDS ARE EXPLICIT: both `lo` and `hi` are stored, not `lo` plus a
 * spacing. Spacing is derived as (hi-lo)/(n-1). Storing the far corner makes
 * the covered region self-describing and lets the loader sanity-check it, at
 * the cost of six doubles.
 */

/* ------------------------------------------------------------------ types */

typedef struct {
  int     n[3];             /* samples per axis                             */
  double  lo[3], hi[3];     /* world bounds of the SAMPLED region           */
  double  h[3];             /* derived spacing (hi-lo)/(n-1)                */
  double *d;                /* n[0]*n[1]*n[2] samples, x fastest            */
  char    path[512];        /* where it came from, for error messages       */
  int     ready;            /* 0 = untried, 1 = loaded, -1 = failed         */
} SdfCollider;

#define SDF_MAX_SLOTS 4

/* THE one piece of static state (see header comment). Nothing outside this
   file can name it; everything below takes an explicit SdfCollider*. */
static SdfCollider g_slot[SDF_MAX_SLOTS];

/* --------------------------------------------------------------- indexing */

static double sdf_at(const SdfCollider *g, int i, int j, int k)
{
  /* Clamp to the sampled block. Sampling the border rather than wrapping or
     erroring means the field continues flat outside, which combined with the
     distance-to-box term in sdf_value() keeps the result continuous. */
  if (i < 0) i = 0; else if (i >= g->n[0]) i = g->n[0] - 1;
  if (j < 0) j = 0; else if (j >= g->n[1]) j = g->n[1] - 1;
  if (k < 0) k = 0; else if (k >= g->n[2]) k = g->n[2] - 1;
  return g->d[((size_t)k * g->n[1] + j) * g->n[0] + i];
}

/* Cubic B-spline basis and its derivatives on [0,1).
   Cubic and not trilinear DELIBERATELY: trilinear is only C0, so its gradient
   jumps across every cell face, and a kinked gradient is precisely what makes
   Evolver's one-sided constraint projection misbehave (docs/solver_paradigms.md
   records keep-outs coming out inert against a kinky min-of-max formula). The
   cubic B-spline is C2, so the gradient this returns is continuous everywhere. */
static void sdf_basis(double t, double w[4], double dw[4], double ddw[4])
{
  const double t2 = t * t, t3 = t2 * t;
  w[0]  = (1 - 3*t + 3*t2 -   t3) / 6.0;
  w[1]  = (4        - 6*t2 + 3*t3) / 6.0;
  w[2]  = (1 + 3*t + 3*t2 - 3*t3) / 6.0;
  w[3]  = t3 / 6.0;
  /* `ddw` IMPLIES `dw`. The mixed second derivatives d2f/dx dy are built from
     the FIRST-derivative basis on both of those axes, so a caller that wants
     curvature needs dw filled even if it does not want the gradient itself.
     Guarding on `!dw` alone returned early here and left BOTH dw[] and ddw[] as
     uninitialised stack garbage -- silently, in the one path (Hessian without
     gradient) that nothing exercised until now. */
  if (!dw && !ddw) return;
  if (dw) {
    dw[0] = (-3 + 6*t - 3*t2) / 6.0;
    dw[1] = (     -12*t + 9*t2) / 6.0;
    dw[2] = ( 3 + 6*t - 9*t2) / 6.0;
    dw[3] = (3*t2) / 6.0;
  }
  if (!ddw) return;
  ddw[0] = ( 6 - 6*t) / 6.0;
  ddw[1] = (-12 + 18*t) / 6.0;
  ddw[2] = ( 6 - 18*t) / 6.0;
  ddw[3] = ( 6*t) / 6.0;
}

/* ------------------------------------------------------------- evaluation */

/* Interpolate inside the sampled block. `p` is in GRID coordinates. */
static double sdf_interp(const SdfCollider *g, const double p[3],
                         double grad[3], double hess[3][3])
{
  double wx[4], wy[4], wz[4], dx[4], dy[4], dz[4], ex[4], ey[4], ez[4];
  int b[3], a, c, e, i, j;
  double v = 0.0;

  for (i = 0; i < 3; i++)
    b[i] = (int)floor(p[i]);
  sdf_basis(p[0] - b[0], wx, (grad || hess) ? dx : 0, hess ? ex : 0);
  sdf_basis(p[1] - b[1], wy, (grad || hess) ? dy : 0, hess ? ey : 0);
  sdf_basis(p[2] - b[2], wz, (grad || hess) ? dz : 0, hess ? ez : 0);

  if (grad) grad[0] = grad[1] = grad[2] = 0.0;
  if (hess) for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) hess[i][j] = 0.0;

  for (e = 0; e < 4; e++)
    for (c = 0; c < 4; c++)
      for (a = 0; a < 4; a++) {
        const double s = sdf_at(g, b[0] + a - 1, b[1] + c - 1, b[2] + e - 1);
        v += s * wx[a] * wy[c] * wz[e];
        if (grad) {
          grad[0] += s * dx[a] * wy[c] * wz[e];
          grad[1] += s * wx[a] * dy[c] * wz[e];
          grad[2] += s * wx[a] * wy[c] * dz[e];
        }
        if (hess) {
          hess[0][0] += s * ex[a] * wy[c] * wz[e];
          hess[1][1] += s * wx[a] * ey[c] * wz[e];
          hess[2][2] += s * wx[a] * wy[c] * ez[e];
          hess[0][1] += s * dx[a] * dy[c] * wz[e];
          hess[0][2] += s * dx[a] * wy[c] * dz[e];
          hess[1][2] += s * wx[a] * dy[c] * dz[e];
        }
      }
  if (hess) {                       /* symmetric */
    hess[1][0] = hess[0][1];
    hess[2][0] = hess[0][2];
    hess[2][1] = hess[1][2];
  }
  return v;
}

/* sdf_interp works in grid units; the chain rule to world units is one factor
   of 1/h per differentiation. Kept separate so each path below is obvious. */
static void sdf_to_world(const SdfCollider *g, double grad[3], double hess[3][3])
{
  int i, j;
  if (hess) for (i = 0; i < 3; i++) for (j = 0; j < 3; j++)
    hess[i][j] /= (g->h[i] * g->h[j]);
  if (grad) for (i = 0; i < 3; i++) grad[i] /= g->h[i];
}

/* f(x) with optional gradient/Hessian, in WORLD coordinates.
 *
 * Outside the sampled box the field is extended as
 *
 *     f(x) = f(clamp(x)) + |x - clamp(x)|
 *
 * i.e. the interpolated value at the nearest sampled point plus the distance
 * you had to travel to get there. This is exact when the obstacle sits well
 * inside the grid (the true SDF grows at unit rate away from it), it is
 * CONTINUOUS across the box face, and it gives |grad f| = 1 out there -- so a
 * vertex that wanders off the grid is pushed sensibly rather than seeing a
 * cliff or a NaN. It does NOT invent an obstacle outside the box, which is the
 * correct default: the grid says where the solid is, and beyond it there is
 * none.
 */
static double sdf_eval(const SdfCollider *g, const REAL *x,
                       double grad[3], double hess[3][3])
{
  double p[3], q[3], out[3], dist2 = 0.0;
  int i, j;

  for (i = 0; i < 3; i++) {
    p[i] = (x[i] - g->lo[i]) / g->h[i];             /* grid coords          */
    q[i] = p[i];
    if (q[i] < 0.0) q[i] = 0.0;
    else if (q[i] > g->n[i] - 1) q[i] = (double)(g->n[i] - 1);
    out[i] = (p[i] - q[i]) * g->h[i];               /* world-space overhang */
    dist2 += out[i] * out[i];
  }

  {
    double v = sdf_interp(g, q, grad, hess);
    sdf_to_world(g, grad, hess);                    /* grid units -> world  */

    if (dist2 <= 0.0)
      return v;                                     /* inside the box       */

    {
      const double dist = sqrt(dist2);
      if (grad) {
        /* d/dx [ f(clamp(x)) + |x-clamp(x)| ]. A CLAMPED axis contributes only
           through the distance term; a free axis keeps the interpolated
           gradient (already converted to world units above). */
        for (i = 0; i < 3; i++)
          if (out[i] != 0.0) grad[i] = out[i] / dist;
      }
      if (hess) {
        /* The extension SEPARATES: with A = the axes still inside the box and
         * B = the clamped ones,
         *
         *     f(x) = g(x_A, const_B) + |o_B|,     o = x - clamp(x)
         *
         * so the Hessian is block diagonal -- interpolated curvature on A, the
         * curvature of a distance-to-a-point on B, and exactly zero coupling
         * between them (an x_A never appears in the |o_B| term and vice versa).
         *
         *     d2|o|/do_i do_j = (delta_ij - n_i n_j) / |o|,   n = o/|o|
         *
         * which is the familiar 1/r transverse curvature of a distance field:
         * it goes to zero along the overhang direction (the extension really is
         * linear THAT way, which is what the old code assumed of every
         * direction) but NOT across it.
         *
         * Zeroing the whole block, as this used to, is not merely inexact -- it
         * hands Newton a singular direction for any vertex off the grid, and a
         * Newton step against a zero curvature is unbounded. Getting it right
         * costs a dozen flops on a path that is already the rare one. */
        for (i = 0; i < 3; i++)
          for (j = 0; j < 3; j++) {
            if (out[i] == 0.0 && out[j] == 0.0)
              continue;                     /* A-A: keep interpolated value  */
            if (out[i] != 0.0 && out[j] != 0.0) {
              const double ni = out[i] / dist, nj = out[j] / dist;
              hess[i][j] = ((i == j ? 1.0 : 0.0) - ni * nj) / dist;   /* B-B  */
            } else {
              hess[i][j] = 0.0;             /* A-B: no coupling              */
            }
          }
      }
      return v + dist;
    }
  }
}

/* ----------------------------------------------------------------- loading */

/* `loud` is 0 while we are only guessing at the default path (the datafile may
   still name a different one), 1 once someone has explicitly asked for this
   file and it had better exist. */
static void sdf_fail(SdfCollider *g, const char *why, int loud)
{
  g->ready = -1;
  if (!loud) return;
  sprintf(errmsg, "sdf_collider: %s (%s)\n", why, g->path);
  kb_error(4901, errmsg, RECOVERABLE);
}

static void sdf_load(SdfCollider *g, const char *path, int loud)
{
  FILE *fp;
  char magic[8];
  int ver, i;
  size_t want;

  strncpy(g->path, path, sizeof(g->path) - 1);
  g->path[sizeof(g->path) - 1] = 0;
  g->ready = -1;                                    /* mark attempted       */

  fp = fopen(path, "rb");
  if (!fp) { sdf_fail(g, "cannot open grid file", loud); return; }

  if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "SDFGRID2", 8)) {
    fclose(fp); sdf_fail(g, "bad magic (not an SDFGRID2 file)", 1); return;
  }
  if (fread(&ver, sizeof(int), 1, fp) != 1 || ver != 2) {
    fclose(fp); sdf_fail(g, "unsupported version", 1); return;
  }
  if (fread(g->n, sizeof(int), 3, fp) != 3) {
    fclose(fp); sdf_fail(g, "truncated header (dims)", 1); return;
  }
  for (i = 0; i < 3; i++)
    if (g->n[i] < 4) {                              /* cubic needs 4 taps   */
      fclose(fp); sdf_fail(g, "each axis needs >= 4 samples", 1); return;
    }
  if (fread(g->lo, sizeof(double), 3, fp) != 3 ||
      fread(g->hi, sizeof(double), 3, fp) != 3) {
    fclose(fp); sdf_fail(g, "truncated header (bounds)", 1); return;
  }
  for (i = 0; i < 3; i++) {
    if (!(g->hi[i] > g->lo[i])) {
      fclose(fp); sdf_fail(g, "bounds not increasing", 1); return;
    }
    g->h[i] = (g->hi[i] - g->lo[i]) / (g->n[i] - 1);
  }

  want = (size_t)g->n[0] * g->n[1] * g->n[2];
  /* mycalloc, not temp_calloc: the grid has to outlive the load and stay valid
     for every constraint evaluation of the whole solve. */
  if (g->d) { myfree((char *)g->d); g->d = NULL; }
  g->d = (double *)mycalloc(want, sizeof(double));
  if (fread(g->d, sizeof(double), want, fp) != want) {
    fclose(fp); sdf_fail(g, "truncated sample data", 1); return;
  }
  fclose(fp);
  g->ready = 1;
}

/* Where slot k's grid comes from, most specific first:
 *
 *   1. a STRING VARIABLE `sdf<k+1>_file` set in the datafile -- so the
 *      reference lives in the .fe, next to the constraint that uses it:
 *
 *          read
 *          define sdf1_file string
 *          sdf1_file := "joint_17.bin"
 *
 *   2. $EVOLVER_SDF_<k+1>
 *   3. ./sdf_<k+1>.bin in the run directory
 *
 * Option 1 is why loading is LAZY rather than done in userfunc_init(): init
 * runs from reset_web() BEFORE the datafile's read section, so the variable
 * does not exist yet. Deferring to the first evaluation means the datafile has
 * had its say by the time we look. Paths are resolved relative to the process
 * cwd, which run_evolver() already sets per solve.
 */
static const char *sdf_datafile_path(int k)
{
  char var[32];
  int gid;
  sprintf(var, "sdf%d_file", k + 1);
  gid = lookup_global(var);
  if (gid < 0)
    return NULL;
  return globals(gid)->value.string;      /* NULL if it is not a string */
}

static SdfCollider *sdf_slot(int k)
{
  char env[32], def[32];
  SdfCollider *g = &g_slot[k];
  const char *p;
  int explicitly_set = 1;

  if (g->ready == 1)
    return g;                              /* already loaded               */

  p = sdf_datafile_path(k);
  if (!(p && *p)) {
    sprintf(env, "EVOLVER_SDF_%d", k + 1);
    p = getenv(env);
  }
  if (!(p && *p)) {
    sprintf(def, "./sdf_%d.bin", k + 1);
    p = def;
    explicitly_set = 0;
  }

  /* RETRY when the resolved path changes. A constraint is evaluated once at
     datafile-read time to project its vertices, which happens BEFORE the read
     section runs -- so on that first call `sdf1_file` does not exist yet and we
     resolve to the default. Re-resolving each time lets the datafile's own
     setting take effect as soon as it is made, instead of latching the failure.
     Failing on the implicit default is therefore not (yet) an error; failing on
     a path someone explicitly asked for is. */
  if (g->ready == -1 && strcmp(p, g->path) == 0)
    return g;                              /* same path, already failed    */

  sdf_load(g, p, explicitly_set);
  return g;
}

/* ------------------------------------------------ the usrN slot wrappers */
/* Thin partial applications: bind a slot, forward to the pure evaluator.
   A failed load returns +HUGE so the obstacle reads as "infinitely far away"
   rather than silently as "you are inside me"; sdf_load has already raised a
   kb_error, so this is belt-and-braces against a RECOVERABLE error being
   ignored. */

#define SDF_ABSENT 1.0e30

static REAL sdf_slot_value(int k, REAL *x)
{
  SdfCollider *g = sdf_slot(k);
  if (g->ready != 1) return (REAL)SDF_ABSENT;
  return (REAL)sdf_eval(g, x, 0, 0);
}

static REAL sdf_slot_deriv(int k, REAL *x, REAL *partials)
{
  SdfCollider *g = sdf_slot(k);
  double grad[3], v;
  int i;
  if (g->ready != 1) {
    for (i = 0; i < SDIM; i++) partials[i] = 0.0;
    return (REAL)SDF_ABSENT;
  }
  v = sdf_eval(g, x, grad, 0);
  /* sdf_eval returns WORLD units on both paths (it calls sdf_to_world itself),
     so this is a straight copy plus the SDIM padding. */
  for (i = 0; i < 3 && i < SDIM; i++) partials[i] = (REAL)grad[i];
  for (i = 3; i < SDIM; i++) partials[i] = 0.0;
  return (REAL)v;
}

static REAL sdf_slot_seconds(int k, REAL *x, REAL *partials, REAL **seconds)
{
  SdfCollider *g = sdf_slot(k);
  double grad[3], hess[3][3], v;
  int i, j;
  if (g->ready != 1) {
    for (i = 0; i < SDIM; i++) {
      partials[i] = 0.0;
      for (j = 0; j < SDIM; j++) seconds[i][j] = 0.0;
    }
    return (REAL)SDF_ABSENT;
  }
  v = sdf_eval(g, x, grad, hess);
  for (i = 0; i < SDIM; i++) {
    partials[i] = (i < 3) ? (REAL)grad[i] : 0.0;
    for (j = 0; j < SDIM; j++)
      seconds[i][j] = (i < 3 && j < 3) ? (REAL)hess[i][j] : 0.0;
  }
  return (REAL)v;
}

/* One trio per slot. Adding a slot is three one-line functions plus three
   entries in userfunc.c's arrays. */
REAL sdf1(REAL *x)                              { return sdf_slot_value(0, x); }
REAL sdf1_deriv(REAL *x, REAL *p)               { return sdf_slot_deriv(0, x, p); }
REAL sdf1_seconds(REAL *x, REAL *p, REAL **s)   { return sdf_slot_seconds(0, x, p, s); }

REAL sdf2(REAL *x)                              { return sdf_slot_value(1, x); }
REAL sdf2_deriv(REAL *x, REAL *p)               { return sdf_slot_deriv(1, x, p); }
REAL sdf2_seconds(REAL *x, REAL *p, REAL **s)   { return sdf_slot_seconds(1, x, p, s); }

/* Called from userfunc_init(), which Evolver runs from reset_web() once per
   surface load. Dropping `ready` forces a reload, so `load`ing a second
   datafile in one session re-reads the grid instead of inheriting the old one. */
static void sdf_reset_all(void)
{
  int k;
  for (k = 0; k < SDF_MAX_SLOTS; k++) {
    g_slot[k].ready = 0;
    g_slot[k].d = NULL;
  }
}
