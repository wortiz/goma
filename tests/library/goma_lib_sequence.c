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
 * usage: goma_lib_sequence [--ranks=<n>] <dir> [--stop-after=create|setup]
 *                          [--log=<file>] <goma args> [-- <dir> ...] ...
 *
 * Each run changes to <dir> and runs Goma with the given arguments, for
 * example:
 *
 *   goma_lib_sequence problemA -i input -- problemB -i input
 *
 * --stop-after destroys the context after goma_create or goma_setup instead
 * of solving, to test teardown of a partly set up problem.  --log writes
 * the standard output of rank 0 during the run to <file> (relative to the
 * starting directory).
 *
 * --ranks runs Goma on a communicator split from MPI_COMM_WORLD that holds
 * only its first <n> ranks, the other ranks wait, to check that Goma does not
 * communicate outside the communicator it is given.
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
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "goma.h"

enum stop_after { STOP_AFTER_CREATE, STOP_AFTER_SETUP, STOP_AFTER_SOLVE };

static int run_goma(MPI_Comm comm, int argc, char **argv, enum stop_after stop) {
  goma_context *ctx = NULL;
  int error = goma_create(&ctx, comm, argc, argv);
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
    fprintf(stderr,
            "usage: %s [--ranks=<n>] <dir> [--stop-after=create|setup] [--log=<file>] <goma args> "
            "[-- <dir> ...] ...\n",
            argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int run = 0;
  int first = 1;
  MPI_Comm comm = MPI_COMM_WORLD;
  if (strncmp(argv[1], "--ranks=", 8) == 0) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int color = world_rank < atoi(argv[1] + 8) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &comm);
    first = 2;
  }
  if (comm == MPI_COMM_NULL) {
    /* not one of the ranks running Goma */
    MPI_Finalize();
    return 0;
  }

  int rank;
  MPI_Comm_rank(comm, &rank);
  goma_library_initialize(comm);

  while (first < argc) {
    int last = first;
    while (last < argc && strcmp(argv[last], "--") != 0) {
      last++;
    }
    const char *dir = argv[first];
    enum stop_after stop = STOP_AFTER_SOLVE;
    const char *log = NULL;
    while (first + 1 < last && strncmp(argv[first + 1], "--", 2) == 0) {
      const char *opt = argv[first + 1];
      if (strncmp(opt, "--stop-after=", 13) == 0) {
        const char *stage = opt + 13;
        if (strcmp(stage, "create") == 0) {
          stop = STOP_AFTER_CREATE;
        } else if (strcmp(stage, "setup") == 0) {
          stop = STOP_AFTER_SETUP;
        } else {
          fprintf(stderr, "unknown stage %s\n", stage);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      } else if (strncmp(opt, "--log=", 6) == 0) {
        log = opt + 6;
      } else {
        fprintf(stderr, "unknown option %s\n", opt);
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

    int saved_stdout = -1;
    if (log != NULL && rank == 0) {
      int fd = open(log, O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd < 0) {
        fprintf(stderr, "could not open %s\n", log);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      fflush(stdout);
      saved_stdout = dup(STDOUT_FILENO);
      dup2(fd, STDOUT_FILENO);
      close(fd);
    }
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
    int error = run_goma(comm, last - first, run_argv, stop);
    if (trace)
      leaktrace_enable(0);
    if (saved_stdout >= 0) {
      fflush(stdout);
      dup2(saved_stdout, STDOUT_FILENO);
      close(saved_stdout);
    }
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
  if (comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&comm);
  }
  MPI_Finalize();
  return status;
}
