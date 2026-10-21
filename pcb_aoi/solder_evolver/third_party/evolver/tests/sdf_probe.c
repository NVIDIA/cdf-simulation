/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * A TEST harness for src/sdf_collider.c -- not part of the Evolver build.
 * =====================================================================
 *
 * sdf_collider.c is written to be #included from Evolver's userfunc.c, so that
 * it inherits REAL/SDIM from the binary it is linked into (see its header
 * comment). That is the right call for shipping, but it means the evaluator is
 * only reachable through Evolver's datafile language -- and the datafile
 * language has no way at all to ASK for a second derivative. `usr3` in a
 * constraint gets you the value; the Hessian is consumed silently, deep inside
 * `hessian`, where a wrong answer shows up only as a solve that misbehaves.
 *
 * So: #include the same file into a shim that fakes the half-dozen Evolver
 * symbols it touches, and export the pure evaluator to ctypes. The test then
 * differences the analytic gradient and compares. This tests the SAME SOURCE
 * TEXT the binary compiles -- not a reimplementation -- which is the only
 * version of this that is worth having.
 *
 * The one thing it cannot catch is a REAL/SDIM mismatch, since we pick those
 * here. The existing end-to-end datafile tests cover that.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------- the Evolver shim layer */
/* Everything sdf_collider.c reaches for that normally comes from Evolver's
   headers. Matching the real signatures matters: a mismatch would compile here
   and diverge from the shipped build, which is exactly the failure this file
   exists to rule out. */

typedef double REAL;            /* Evolver's default; see build.sh CFLAGS     */
#define SDIM 3
#define RECOVERABLE 1

char errmsg[2048];

void kb_error(int errnum, char *msg, int level)
{
  (void)errnum; (void)level;
  fprintf(stderr, "kb_error: %s", msg ? msg : "(null)");
}

char *mycalloc(size_t n, size_t size) { return (char *)calloc(n, size); }
void  myfree(char *p)                 { free(p); }

/* The datafile-variable lookup. Returning -1 ("no such global") makes
   sdf_datafile_path() fall through to $EVOLVER_SDF_1, which is how the Python
   test points the collider at its grid. globals() is therefore never actually
   called, but it must still typecheck. */
struct sdf_probe_global { struct { char *string; } value; };
static struct sdf_probe_global sdf_probe_dummy_global;

int lookup_global(char *name) { (void)name; return -1; }
#define globals(k) (&sdf_probe_dummy_global)

#include "../src/sdf_collider.c"

/* ------------------------------------------------------------- the exports */

/* Force slot 0 to (re)load from `path`. Returns 1 on success. Explicit rather
   than relying on the lazy path so a test can assert the load itself failed or
   succeeded, instead of misreading SDF_ABSENT as a numerical answer. */
int probe_load(const char *path)
{
  sdf_reset_all();
  sdf_load(&g_slot[0], path, 1);
  return g_slot[0].ready == 1;
}

/* value + optional gradient + optional Hessian, all in WORLD units.
   `hess` is 9 doubles, row major. Either pointer may be NULL. */
double probe_eval(const double *x, double *grad, double *hess)
{
  double gr[3], hs[3][3], v;
  int i, j;

  v = sdf_eval(&g_slot[0], x, grad ? gr : 0, hess ? hs : 0);
  if (grad) for (i = 0; i < 3; i++) grad[i] = gr[i];
  if (hess) for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) hess[3*i + j] = hs[i][j];
  return v;
}

/* Go through the ACTUAL Evolver slot entry points, so the test also covers the
   row-pointer marshalling in sdf_slot_seconds() and not just sdf_eval itself.
   `seconds` is Evolver's array-of-row-pointers layout. */
double probe_slot_seconds(double *x, double *partials, double *seconds_flat)
{
  REAL *rows[SDIM];
  int i;
  for (i = 0; i < SDIM; i++) rows[i] = seconds_flat + (size_t)i * SDIM;
  return sdf1_seconds(x, partials, rows);
}

double probe_slot_deriv(double *x, double *partials)
{
  return sdf1_deriv(x, partials);
}

double probe_slot_value(double *x) { return sdf1(x); }
