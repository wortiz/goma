#include <catch2/catch_test_macros.hpp>

#include <climits>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "util/goma_context.h"
}

TEST_CASE("goma context create has startup defaults", "[util][goma_context]") {
  GomaContext *ctx = goma_context_create();
  REQUIRE(ctx != nullptr);
  REQUIRE(ctx->num_proc == 1);
  REQUIRE(ctx->proc_id == 0);
  REQUIRE(ctx->dim == -1);
  REQUIRE(ctx->num_requests == 6);
  REQUIRE(ctx->unlimited_output == 1);
  REQUIRE(ctx->exo_time_plane == INT_MAX);
  REQUIRE(ctx->decompose_flag == 1);
  REQUIRE(std::strcmp(ctx->front_scratch_directory, FRONT_SCRATCH_DIRECTORY) == 0);
  REQUIRE(ctx->particles.xi_boundary_tolerances[0] == XI_BOUNDARY_TOLERANCE0);
  REQUIRE(ctx->particles.xi_boundary_tolerances[1] == XI_BOUNDARY_TOLERANCE1);
  REQUIRE(ctx->particles.xi_boundary_tolerances[2] == XI_BOUNDARY_TOLERANCE2);
  REQUIRE(ctx->mesh.coor == nullptr);
  REQUIRE(ctx->particles.element_info == nullptr);
  goma_context_destroy(ctx);
}

TEST_CASE("goma context default instance matches goma_context_init", "[util][goma_context]") {
  GomaContext *saved = goma_context_set_current(nullptr);
  GomaContext *fresh = goma_context_create();
  REQUIRE(goma_ctx->dim == fresh->dim);
  REQUIRE(goma_ctx->num_requests == fresh->num_requests);
  REQUIRE(goma_ctx->exo_time_plane == fresh->exo_time_plane);
  REQUIRE(goma_ctx->decompose_flag == fresh->decompose_flag);
  REQUIRE(std::strcmp(goma_ctx->front_scratch_directory, fresh->front_scratch_directory) == 0);
  for (int i = 0; i < 3; i++) {
    REQUIRE(goma_ctx->particles.xi_boundary_tolerances[i] ==
            fresh->particles.xi_boundary_tolerances[i]);
  }
  goma_context_destroy(fresh);
  goma_context_set_current(saved);
}

TEST_CASE("goma context release frees owned memory and keeps settings", "[util][goma_context]") {
  GomaContext *ctx = goma_context_create();
  ctx->debug_flag = 2;
  ctx->request = static_cast<MPI_Request *>(std::malloc(4 * sizeof(MPI_Request)));
  ctx->status = static_cast<MPI_Status *>(std::malloc(4 * sizeof(MPI_Status)));
  ctx->petsc_options = static_cast<char *>(std::calloc(8, 1));
  ctx->petsc_options_str_len = 8;

  ctx->spec_source_inventory_rows = 2;
  ctx->spec_source_inventory = static_cast<double **>(std::calloc(2, sizeof(double *)));
  ctx->spec_source_inventory[0] = static_cast<double *>(std::calloc(3, sizeof(double)));
  ctx->spec_source_inventory[1] = static_cast<double *>(std::calloc(3, sizeof(double)));
  ctx->spec_source_lumped_mass = static_cast<double *>(std::calloc(5, sizeof(double)));

  ctx->mesh.num_dim = 2;
  ctx->mesh.coor = static_cast<double **>(std::calloc(2, sizeof(double *)));
  ctx->mesh.proc_ss_node_count = static_cast<int *>(std::calloc(3, sizeof(int)));

  GomaParticles *p = &ctx->particles;
  p->number_sample_types = 2;
  p->number_samples = static_cast<int *>(std::calloc(2, sizeof(int)));
  p->number_samples_existing = static_cast<int *>(std::calloc(2, sizeof(int)));
  p->number_output_variables = static_cast<int *>(std::calloc(2, sizeof(int)));
  p->output_variables =
      static_cast<particle_variable_s **>(std::calloc(2, sizeof(particle_variable_s *)));
  p->output_variables[0] =
      static_cast<particle_variable_s *>(std::calloc(3, sizeof(particle_variable_s)));
  p->output_variables[1] = nullptr;
  p->filename_template =
      static_cast<particle_filename_s *>(std::calloc(2, sizeof(particle_filename_s)));
  p->num_pbcs = 1;
  p->pbcs = static_cast<PBC_t *>(std::calloc(1, sizeof(PBC_t)));
  p->num_element_info = 2;
  p->element_info =
      static_cast<element_particle_info_t *>(std::calloc(2, sizeof(element_particle_info_t)));
  p->element_info[0].PBC_side_id = static_cast<int *>(std::calloc(6, sizeof(int)));
  p->element_info[1].source_term = static_cast<double *>(std::calloc(4, sizeof(double)));

  goma_context_release(ctx);

  REQUIRE(ctx->request == nullptr);
  REQUIRE(ctx->status == nullptr);
  REQUIRE(ctx->petsc_options == nullptr);
  REQUIRE(ctx->petsc_options_str_len == 0);
  REQUIRE(ctx->spec_source_inventory == nullptr);
  REQUIRE(ctx->spec_source_inventory_rows == 0);
  REQUIRE(ctx->spec_source_lumped_mass == nullptr);
  REQUIRE(ctx->mesh.coor == nullptr);
  REQUIRE(ctx->mesh.proc_ss_node_count == nullptr);
  REQUIRE(ctx->mesh.num_dim == 0);
  REQUIRE(p->number_samples == nullptr);
  REQUIRE(p->number_samples_existing == nullptr);
  REQUIRE(p->number_output_variables == nullptr);
  REQUIRE(p->output_variables == nullptr);
  REQUIRE(p->filename_template == nullptr);
  REQUIRE(p->pbcs == nullptr);
  REQUIRE(p->element_info == nullptr);
  REQUIRE(p->num_element_info == 0);

  // Settings that are not owned allocations survive a release.
  REQUIRE(ctx->debug_flag == 2);

  // Releasing twice is safe.
  goma_context_release(ctx);
  goma_context_destroy(ctx);
}

TEST_CASE("goma context reset restores defaults", "[util][goma_context]") {
  GomaContext *ctx = goma_context_create();
  ctx->debug_flag = 3;
  ctx->decompose_flag = 0;
  ctx->petsc_options = static_cast<char *>(std::calloc(8, 1));
  goma_context_reset(ctx);
  REQUIRE(ctx->debug_flag == 0);
  REQUIRE(ctx->decompose_flag == 1);
  REQUIRE(ctx->petsc_options == nullptr);
  goma_context_destroy(ctx);
}

TEST_CASE("goma context set current and destroy", "[util][goma_context]") {
  GomaContext *original = goma_ctx;
  GomaContext *ctx = goma_context_create();
  ctx->proc_id = 7;

  GomaContext *previous = goma_context_set_current(ctx);
  REQUIRE(previous == original);
  REQUIRE(goma_ctx == ctx);
  REQUIRE(goma_ctx->proc_id == 7);

  // Destroying the current context falls back to the built-in default.
  goma_context_destroy(ctx);
  REQUIRE(goma_ctx != nullptr);
  REQUIRE(goma_ctx != ctx);

  goma_context_set_current(original);
  REQUIRE(goma_ctx == original);

  goma_context_destroy(nullptr);
}
