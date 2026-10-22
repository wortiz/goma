/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2026 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

/*
 * Runs Goma several times in one process through the goma.h interface
 *
 * usage: goma_lib_sequence <dir> [--stop-after=create|setup] <goma args>
 *                          [-- <dir> ...] ...
 *
 * Each run changes to <dir> and runs Goma with the given arguments, for
 * example:
 *
 *   goma_lib_sequence problemA -i input -- problemB -i input
 *
 * --stop-after destroys the context after goma_create or goma_setup instead
 * of solving, to test teardown of a partly set up problem.
 *
 * Requires GOMA_EXPERIMENTAL_MULTIPLE_RUNS=1 in the environment until
 * goma_destroy fully resets Goma's global state.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* RTLD_DEFAULT */
#endif
#include <mpi.h>
#include <stdio.h>
#ifdef __GLIBC__
#include <malloc.h>
#endif
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "goma.h"

enum stop_after { STOP_AFTER_CREATE, STOP_AFTER_SETUP, STOP_AFTER_SOLVE };

static int run_goma(int argc, char **argv, enum stop_after stop) {
  goma_context *ctx = NULL;
  int error = goma_create(&ctx, MPI_COMM_WORLD, argc, argv);
  if (error == 0 && stop >= STOP_AFTER_SETUP) {
    error = goma_setup(ctx);
  }
  if (error == 0 && stop >= STOP_AFTER_SOLVE) {
    error = goma_solve(ctx);
  }
  if (ctx != NULL) {
    goma_destroy(&ctx);
  }
  return error;
}

int main(int argc, char **argv) {
  int status = 0;
  MPI_Init(&argc, &argv);

  char start_dir[4096];
  if (getcwd(start_dir, sizeof(start_dir)) == NULL || argc < 2) {
    fprintf(stderr, "usage: %s <dir> [--stop-after=create|setup] <goma args> [-- <dir> ...] ...\n",
            argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  goma_library_initialize(MPI_COMM_WORLD);

  int run = 0;
  int first = 1;
  while (first < argc) {
    int last = first;
    while (last < argc && strcmp(argv[last], "--") != 0) {
      last++;
    }
    const char *dir = argv[first];
    enum stop_after stop = STOP_AFTER_SOLVE;
    if (first + 1 < last && strncmp(argv[first + 1], "--stop-after=", 13) == 0) {
      const char *stage = argv[first + 1] + 13;
      if (strcmp(stage, "create") == 0) {
        stop = STOP_AFTER_CREATE;
      } else if (strcmp(stage, "setup") == 0) {
        stop = STOP_AFTER_SETUP;
      } else {
        fprintf(stderr, "unknown stage %s\n", stage);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      first++;
    }
    /* goma expects argv[0] to be the program name */
    char *run_argv[last - first + 1];
    run_argv[0] = argv[0];
    for (int i = first + 1; i < last; i++) {
      run_argv[i - first] = argv[i];
    }
    run_argv[last - first] = NULL;

    if (chdir(dir) != 0) {
      fprintf(stderr, "could not change to directory %s\n", dir);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /*
     * GOMA_LIB_SEQUENCE_TRACE_RUN=<n> asks a preloaded allocation tracer that
     * defines leaktrace_enable(int) to record run n, allocations still live
     * when it is disabled were leaked by that run
     */
    void (*leaktrace_enable)(int) = (void (*)(int))dlsym(RTLD_DEFAULT, "leaktrace_enable");
    const char *trace_run = getenv("GOMA_LIB_SEQUENCE_TRACE_RUN");
    int trace = leaktrace_enable != NULL && trace_run != NULL && atoi(trace_run) == run;
    if (trace)
      leaktrace_enable(1);
    int error = run_goma(last - first, run_argv, stop);
    if (trace)
      leaktrace_enable(0);
    fprintf(stderr, "goma_lib_sequence: run %d in %s returned %d\n", run, dir, error);
#ifdef __GLIBC__
    /* heap still in use after goma_destroy, growth between runs is leaked memory */
    struct mallinfo2 mi = mallinfo2();
    fprintf(stderr, "goma_lib_sequence: run %d heap in use %zu bytes\n", run, mi.uordblks);
#endif
    if (error != 0) {
      status = 1;
    }
    if (chdir(start_dir) != 0) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    run++;
    first = last + 1;
  }

  goma_library_finalize();
  MPI_Finalize();
  return status;
}
