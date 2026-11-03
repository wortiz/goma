#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <mpi.h>

extern "C" {
#include "util/goma_context.h"
}

extern "C" {
int parallel_err = 0;
}

int main(int argc, char *argv[]) {
  // global setup...
  MPI_Init(&argc, &argv);

  auto error = MPI_Comm_size(MPI_COMM_WORLD, &goma_ctx->num_proc);
  if (error) {
    return error;
  }
  error = MPI_Comm_rank(MPI_COMM_WORLD, &goma_ctx->proc_id);
  if (error) {
    return error;
  }

  if (goma_ctx->num_proc > 1) {
    if (goma_ctx->proc_id == 0) {
      std::cerr << "Unit tests are currently expected to be run serially, found " << goma_ctx->num_proc
                << " MPI Processes\n";
    }
    return -1;
  }

  int result = Catch::Session().run(argc, argv);

  // global clean-up...
  MPI_Finalize();

  return result;
}
