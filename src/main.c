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
 * Main driver for the goma executable, see goma_api.c for the solver stages
 */

#include <mpi.h>

#include "goma.h"
#include "mm_eh.h"

int main(int argc, char **argv) {
  goma_context *ctx = NULL;
  int error;

#ifdef PARALLEL
  MPI_Init(&argc, &argv);
#endif

  goma_library_initialize(MPI_COMM_WORLD);
  /* the executable owns MPI, so fatal errors may finalize it */
  Goma_Owns_MPI = TRUE;

  error = goma_create(&ctx, MPI_COMM_WORLD, argc, argv);
  if (error == 0) {
    error = goma_setup(ctx);
  }

  /*
   * Missing files on any processor are detected at a lower level
   * forcing a return to the higher level, shutdown now
   */
  if (error < 0) {
    goma_destroy(&ctx);
    goma_library_finalize();
#ifdef PARALLEL
    MPI_Finalize();
#endif
    return (-1);
  }

  goma_solve(ctx);

  goma_destroy(&ctx);

  goma_library_finalize();

#ifdef PARALLEL
  MPI_Finalize();
#endif
  return (0);
}
