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

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <mpi.h>

#ifdef GOMA_KOKKOS_UNIT_TESTS
#include "mm_kokkos_assembly.h"
#endif

extern "C" {
#ifdef GOMA_UNIT_TESTS_LINK_GOMA
/* defined by libgoma */
extern int ProcID;
extern int Num_Proc;
#else
int ProcID = 0;
int parallel_err = 0;
int Num_Proc = 1;
#endif
}

int main(int argc, char *argv[]) {
  // global setup...
  MPI_Init(&argc, &argv);

  auto error = MPI_Comm_size(MPI_COMM_WORLD, &Num_Proc);
  if (error) {
    return error;
  }
  error = MPI_Comm_rank(MPI_COMM_WORLD, &ProcID);
  if (error) {
    return error;
  }

#ifndef GOMA_MPI_UNIT_TESTS
  if (Num_Proc > 1) {
    if (ProcID == 0) {
      std::cerr << "Unit tests are currently expected to be run serially, found " << Num_Proc
                << " MPI Processes\n";
    }
    return -1;
  }
#endif

#ifdef GOMA_KOKKOS_UNIT_TESTS
  if (goma_kokkos_runtime_initialize() != GOMA_KOKKOS_SUCCESS) {
    MPI_Finalize();
    return -1;
  }
#endif

  int result = Catch::Session().run(argc, argv);

#ifdef GOMA_KOKKOS_UNIT_TESTS
  if (goma_kokkos_runtime_finalize() != GOMA_KOKKOS_SUCCESS) {
    result = -1;
  }
#endif

  // global clean-up...
  MPI_Finalize();

  return result;
}
