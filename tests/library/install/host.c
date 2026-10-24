/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) Goma Developers, National Technology & Engineering        *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

/*
 * Host program using an installed Goma (find_package(Goma), Goma::goma)
 *
 * usage: goma_install_host <goma args>
 *
 * Solves the problem of the current directory, for example
 * goma_install_host -i input
 */

#include <goma.h>
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int err = goma_library_initialize(MPI_COMM_WORLD);
  goma_context *ctx = NULL;
  if (!err)
    err = goma_create(&ctx, MPI_COMM_WORLD, argc, argv);
  if (!err)
    err = goma_setup(ctx);
  if (!err)
    err = goma_solve(ctx);
  if (ctx != NULL && goma_destroy(&ctx))
    err = 1;
  if (goma_library_finalize())
    err = 1;
  if (err)
    fprintf(stderr, "goma_install_host: failed\n");
  MPI_Finalize();
  return err ? 1 : 0;
}
