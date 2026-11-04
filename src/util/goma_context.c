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

/*
 * Startup defaults for every context field; anything not listed is zero/NULL.
 * Used for the built-in default context and by goma_context_init().
 */
#define GOMA_CONTEXT_DEFAULTS                                                                  \
  {                                                                                            \
      .num_proc = 1,                                                                           \
      .dim = -1,                                                                               \
      .num_requests = 6,                                                                       \
      .unlimited_output = 1,                                                                   \
      .exo_time_plane = INT_MAX,                                                               \
      .decompose_flag = 1,                                                                     \
      .front_scratch_directory = FRONT_SCRATCH_DIRECTORY,                                      \
      .particles = {.xi_boundary_tolerances = {XI_BOUNDARY_TOLERANCE0, XI_BOUNDARY_TOLERANCE1, \
                                               XI_BOUNDARY_TOLERANCE2}},                       \
      .post = {GOMA_POST_DEFAULTS},                                                            \
      .fill = {.ca_boundary_num_ss = -1, .dofptr_elem_blk_id = -123},                          \
      .problem = {.lsa_3d_of_2d_wave_number = -1.0},                                           \
      .solver = {.first_linear_solve = 1,                                                      \
                 .stratimikos_tpetra_param_echo = {1},                                         \
                 .loca_stab_first_linear_solve = 1,                                            \
                 .loca_stab_umf_id = -1,                                                       \
                 .loca_sv_init = 1,                                                            \
                 .loca_eig_old_step = -1,                                                      \
                 .loca_eig_nprint = -1},                                                       \
  }

/* Post-processing variable flags are off (-1) by default. */
#define GOMA_POST_DEFAULTS                                                                       \
  .capillary_pressure = -1, .conc_cont = -1, .conduction_vectors = -1, .curl_v = -1,             \
  .darcy_velocity_gas = -1, .darcy_velocity_liq = -1, .density = -1, .polymer_viscosity = -1,    \
  .polymer_time_const = -1, .mobility_parameter = -1, .ptt_xi = -1, .ptt_epsilon = -1,           \
  .dielectrophoretic_field = -1, .dielectrophoretic_field_norm = -1, .diffusion_vectors = -1,    \
  .diffusion_vectors_por_liq_gphase = -1, .diffusion_vectors_por_air_gphase = -1,                \
  .div_pvelocity = -1, .div_total = -1, .div_velocity = -1, .electric_field = -1,                \
  .electric_field_mag = -1, .energy_fluxlines = -1, .error_zz_p = -1, .error_zz_p_elsize = -1,   \
  .error_zz_q = -1, .error_zz_q_elsize = -1, .error_zz_vel = -1, .error_zz_vel_elsize = -1,      \
  .evp_def_grad_tensor = -1, .external_post = -1, .fill_cont = -1, .first_invar_strain = -1,     \
  .fluxlines = -1, .lagrange_convection = -1, .mean_shear = -1, .mm_residuals = -1,              \
  .ns_residuals = -1, .porous_rho_gas_solvents = -1, .porous_rho_lphase = -1,                    \
  .porous_rho_total_solvents = -1, .porous_saturation = -1, .porous_gridpeclet = -1,             \
  .porous_supgvelocity = -1, .porous_liquid_accum_rate = -1, .rel_liq_perm = -1,                 \
  .pressure_cont = -1, .sh_div_s_v_cont = -1, .sh_curv_cont = -1, .real_stress_tensor = -1,      \
  .sec_invar_strain = -1, .strain_tensor = -1, .stream = -1, .stream_normal_stress = -1,         \
  .stream_shear_stress = -1, .stream_tension = -1, .stress_cont = -1, .stress_tensor = -1,       \
  .surface_vectors = -1, .shell_normals = -1, .third_invar_strain = -1, .time_derivatives = -1,  \
  .total_stress11 = -1, .total_stress12 = -1, .total_stress13 = -1, .total_stress22 = -1,        \
  .total_stress23 = -1, .total_stress33 = -1, .user_post = -1, .pp_viscosity = -1,               \
  .pp_flowingliquid_viscosity = -1, .pp_volumefractiongas = -1, .sat_curve_type = -1,            \
  .cap_press_switch = -1, .sat_qp_switch = -1, .num_curve_switch = -1, .sat_hyst_min = -1,       \
  .sat_hyst_max = -1, .acoustic_pressure = -1, .acoustic_phase_angle = -1,                       \
  .acoustic_energy_density = -1, .light_intensity = -1, .principal_stress = -1,                  \
  .principal_real_stress = -1, .lub_height = -1, .lub_height_2 = -1, .lub_velo_upper = -1,       \
  .lub_velo_lower = -1, .lub_velo_field = -1, .lub_velo_field_2 = -1, .disj_press = -1,          \
  .sh_sat_open = -1, .sh_sat_open_2 = -1, .sh_cap_pres = -1, .sh_pore_flux = -1,                 \
  .sh_stress_tensor = -1, .sh_tang = -1, .pp_lame_mu = -1, .pp_lame_lambda = -1,                 \
  .von_mises_strain = -1, .von_mises_stress = -1, .conf_map = -1, .j_flux = -1, .eig = -1,       \
  .eig1 = -1, .eig2 = -1, .eig3 = -1, .heaviside = -1, .rho_dot = -1, .moment_sources = -1,      \
  .yzbeta = -1, .grad_y = -1, .grad_sh = -1, .untracked_spec = -1, .tfmp_gas_velo = -1,          \
  .tfmp_liq_velo = -1, .tfmp_inv_peclet = -1, .tfmp_krg = -1, .velo_speed = -1, .gies_crit = -1, \
  .helicity = -1, .lamb_vector = -1, .q_fcn = -1, .poynting_vectors = -1, .pspg_pp = -1,         \
  .saramito_yield = -1, .stress_norm = -1, .species_sources = -1, .viscous_stress = -1,          \
  .pp_velocity_gradients = -1, .viscous_stress_norm = -1, .viscous_von_mises_stress = -1,        \
  .em_contours = -1, .total_em_contours = -1, .scattered_em_contours = -1,                       \
  .orientation_vectors = -1, .first_strainrate_invar = -1, .sec_strainrate_invar = -1,           \
  .third_strainrate_invar = -1, .wall_distance = -1, .contact_distance = -1,                     \
  .pp_fluid_stress = -1, .lub_convection = -1, .pp_mesh_velocity = -1, .lub_fluid_source = -1,   \
  .mean_vorticity = -1, .pp_diffusivity = -1

static const GomaContext goma_context_defaults = GOMA_CONTEXT_DEFAULTS;

static GomaContext goma_context_storage = GOMA_CONTEXT_DEFAULTS;

GomaContext *goma_ctx = &goma_context_storage;

void goma_context_init(GomaContext *ctx) { *ctx = goma_context_defaults; }

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

void goma_fill_release(GomaFillState *fill) {
  if (fill->ls_surf_log != NULL) {
    fclose(fill->ls_surf_log);
  }
  *fill = goma_context_defaults.fill;
}

void goma_echo_release(GomaEcho *echo) {
  for (int i = 0; i < echo->num_files; i++) {
    if (echo->files[i] != NULL) {
      fclose(echo->files[i]);
    }
    free(echo->names[i]);
  }
  memset(echo, 0, sizeof(*echo));
}

void goma_context_release(GomaContext *ctx) {
  if (ctx->release_problem != NULL) {
    /* the solver library reaches its data through goma_ctx */
    GomaContext *previous = goma_context_set_current(ctx);
    ctx->release_problem(ctx);
    ctx->release_problem = NULL;
    goma_context_set_current(previous);
  }

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

  memset(&ctx->fem, 0, sizeof(ctx->fem));
  goma_fill_release(&ctx->fill);
  goma_echo_release(&ctx->echo);
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
