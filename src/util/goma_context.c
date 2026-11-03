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

#include "util/goma_context.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* Static storage with the same defaults goma_context_init() applies. */
static GomaContext goma_context_storage = {
    .num_proc = 1,
    .dim = -1,
    .num_requests = 6,
    .unlimited_output = 1,
    .exo_time_plane = INT_MAX,
    .decompose_flag = 1,
    .front_scratch_directory = FRONT_SCRATCH_DIRECTORY,
    .particles = {.xi_boundary_tolerances = {XI_BOUNDARY_TOLERANCE0, XI_BOUNDARY_TOLERANCE1,
                                             XI_BOUNDARY_TOLERANCE2}}};

GomaContext *goma_ctx = &goma_context_storage;

void goma_context_init(GomaContext *ctx) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->num_proc = 1;
  ctx->proc_id = 0;
  ctx->dim = -1;
  ctx->request = NULL;
  ctx->status = NULL;
  ctx->num_requests = 6;
  ctx->unlimited_output = 1;
  ctx->exo_time_plane = INT_MAX;
  ctx->decompose_flag = 1;
  strncpy(ctx->front_scratch_directory, FRONT_SCRATCH_DIRECTORY, MAX_FNL - 1);
  ctx->particles.xi_boundary_tolerances[0] = XI_BOUNDARY_TOLERANCE0;
  ctx->particles.xi_boundary_tolerances[1] = XI_BOUNDARY_TOLERANCE1;
  ctx->particles.xi_boundary_tolerances[2] = XI_BOUNDARY_TOLERANCE2;
}

GomaContext *goma_context_create(void) {
  GomaContext *ctx = malloc(sizeof(GomaContext));
  if (ctx != NULL) {
    goma_context_init(ctx);
  }
  return ctx;
}

void goma_mesh_release(GomaMesh *mesh) {
  free(mesh->coor);
  free(mesh->proc_ss_node_count);
  /* Everything else aliases Exo_DB storage, which is owned and freed elsewhere. */
  memset(mesh, 0, sizeof(*mesh));
}

void goma_particles_release_element_info(GomaParticles *particles) {
  if (particles->element_info != NULL) {
    for (int i = 0; i < particles->num_element_info; i++) {
      element_particle_info_t *info = &particles->element_info[i];
      free(info->PBC_side_id);
#ifdef PARALLEL
      free(info->owner_local_element_id);
      free(info->ghost_proc);
      free(info->ghost_local_elem_id);
#endif
      free(info->source_term);
      free(info->list_PBC_IMPERMEABLE);
    }
  }
  free(particles->element_info);
  particles->element_info = NULL;
  particles->num_element_info = 0;
}

void goma_particles_release(GomaParticles *particles) {
  if (particles->output_variables != NULL) {
    for (int i = 0; i < particles->number_sample_types; i++) {
      free(particles->output_variables[i]);
    }
  }
  free(particles->output_variables);
  particles->output_variables = NULL;
  free(particles->number_samples);
  particles->number_samples = NULL;
  free(particles->number_samples_existing);
  particles->number_samples_existing = NULL;
  free(particles->number_output_variables);
  particles->number_output_variables = NULL;
  free(particles->filename_template);
  particles->filename_template = NULL;

  free(particles->pbcs);
  particles->pbcs = NULL;

  goma_particles_release_element_info(particles);
}

void goma_fem_release(GomaFem *fem) {
  if (fem->release != NULL) {
    fem->release(fem);
  }
  memset(fem, 0, sizeof(*fem));
}

void goma_context_release(GomaContext *ctx) {
  free(ctx->request);
  ctx->request = NULL;
  free(ctx->status);
  ctx->status = NULL;

  free(ctx->petsc_options);
  ctx->petsc_options = NULL;
  ctx->petsc_options_str_len = 0;

  if (ctx->spec_source_inventory != NULL) {
    for (int i = 0; i < ctx->spec_source_inventory_rows; i++) {
      free(ctx->spec_source_inventory[i]);
    }
  }
  free(ctx->spec_source_inventory);
  ctx->spec_source_inventory = NULL;
  ctx->spec_source_inventory_rows = 0;
  free(ctx->spec_source_lumped_mass);
  ctx->spec_source_lumped_mass = NULL;

  goma_fem_release(&ctx->fem);
  goma_mesh_release(&ctx->mesh);
  goma_particles_release(&ctx->particles);
}

void goma_context_reset(GomaContext *ctx) {
  goma_context_release(ctx);
  goma_context_init(ctx);
}

void goma_context_destroy(GomaContext *ctx) {
  if (ctx == NULL) {
    return;
  }
  goma_context_release(ctx);
  if (goma_ctx == ctx) {
    goma_ctx = &goma_context_storage;
  }
  if (ctx != &goma_context_storage) {
    free(ctx);
  }
}

GomaContext *goma_context_set_current(GomaContext *ctx) {
  GomaContext *previous = goma_ctx;
  goma_ctx = (ctx != NULL) ? ctx : &goma_context_storage;
  return previous;
}
