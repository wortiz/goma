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
 * Library interface to Goma
 *
 * Typical use, with MPI initialized and finalized by the caller:
 *
 *   goma_library_initialize(MPI_COMM_WORLD);
 *   goma_context *ctx;
 *   goma_create(&ctx, MPI_COMM_WORLD, argc, argv);  // command line + input deck
 *   goma_setup(ctx);                                 // mesh and problem setup
 *   goma_solve(ctx);
 *   goma_destroy(&ctx);
 *   goma_library_finalize();
 *
 * Only one goma_context may exist at a time.  goma_destroy frees the problem
 * and resets Goma's global state, so a later goma_create can run another
 * (independent) problem.  While this is experimental a second goma_create in
 * a process returns an error unless GOMA_EXPERIMENTAL_MULTIPLE_RUNS is set in
 * the environment.
 *
 * Goma runs on a duplicate of the communicator given to goma_create, which
 * may hold any subset of the processes.  Fatal errors still terminate the
 * process.
 *
 * All functions return 0 on success and a negative value on failure.
 */

#ifndef GOMA_H
#define GOMA_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct goma_context goma_context;

/* Call once per process after MPI_Init, before any other goma_* function */
int goma_library_initialize(MPI_Comm comm);

/* Finalizes third-party libraries initialized by Goma (never MPI) */
int goma_library_finalize(void);

/* Parse the command line (argv[0] is the program name) and input deck,
 * collective over comm */
int goma_create(goma_context **ctx, MPI_Comm comm, int argc, char **argv);

/* Read the mesh and set up the problem */
int goma_setup(goma_context *ctx);

/* Solve the problem, writing results to the output files */
int goma_solve(goma_context *ctx);

/*
 * Free the context and memory allocated for the problem and reset global
 * state, sets *ctx to NULL.  May be called after any stage, including a
 * failed one.
 */
int goma_destroy(goma_context **ctx);

#ifdef __cplusplus
}
#endif

#endif /* GOMA_H */
