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
 * Teardown of everything a Goma run hangs off a GomaContext: the input deck
 * (materials, boundary conditions, data tables, continuation and augmenting
 * conditions, level set and post-processing requests), the problem setup
 * (unknown maps, communication maps, nodes, element blocks, BC lists), the
 * mesh and decomposition databases, and the problem description structures
 * from mm_as_alloc.c.
 *
 * goma_problem_release() is installed as the context's release_problem hook by
 * pd_alloc() and runs from goma_context_release() with the context current.
 * Every routine here is safe on partially initialized state and clears what
 * it frees, so a release after a failed or partial setup is fine.
 */

#include "rf_release.h"

#include <stdlib.h>

#include "ac_particles.h"
#include "bc/rotate_coordinates.h"
#include "exo_struct.h"
#include "mm_as.h"
#include "mm_as_alloc.h"
#include "mm_as_structs.h"
#include "mm_bc.h"
#include "mm_fill_ls.h"
#include "mm_mp_structs.h"
#include "mm_post_def.h"
#include "rd_dpi.h"
#include "rd_exo.h"
#include "rf_bc_const.h"
#include "rf_element_storage_const.h"
#include "rf_node_const.h"
#include "rf_solve.h"
#include "rf_vars_const.h"
#include "sl_amesos_interface.h"
#include "sl_aux.h"
#include "sl_lu.h"
#include "sl_petsc.h"
#include "sl_umf.h"
#include "util/goma_context.h"

#define FREE_NULL(p) \
  do {               \
    free(p);         \
    (p) = NULL;      \
  } while (0)

/* Number of matrices the per-matrix arrays were sized with. */
static int num_matrices(const GomaContext *ctx) {
  return (ctx->fem.upd != NULL) ? ctx->fem.upd->Total_Num_Matrices : 0;
}

/* Nodes known to this processor (internal + border + external). */
static int num_local_nodes(const GomaContext *ctx) {
  return ctx->mesh.num_internal_nodes + ctx->mesh.num_border_nodes + ctx->mesh.num_external_nodes;
}

/* Free an array of num_matrices() per-matrix arrays. */
static void free_per_matrix(void ***list, int n) {
  if (*list != NULL) {
    for (int i = 0; i < n; i++) {
      free((*list)[i]);
    }
  }
  FREE_NULL(*list);
}

/****************************************************************************/
/* Post-processing requests                                                 */
/****************************************************************************/

static void free_pp_list(void ***list, int *count) {
  if (*list != NULL) {
    for (int i = 0; i < *count; i++) {
      free((*list)[i]);
    }
  }
  FREE_NULL(*list);
  *count = 0;
}

void post_processing_release(GomaPost *post) {
  if (post->pp_volume != NULL) {
    for (int i = 0; i < post->nn_volume; i++) {
      if (post->pp_volume[i] != NULL) {
        free(post->pp_volume[i]->params);
      }
    }
  }
  free_pp_list((void ***)&post->pp_fluxes, &post->nn_post_fluxes);
  free_pp_list((void ***)&post->pp_fluxes_sens, &post->nn_post_fluxes_sens);
  free_pp_list((void ***)&post->pp_data, &post->nn_post_data);
  free_pp_list((void ***)&post->pp_data_sens, &post->nn_post_data_sens);
  free_pp_list((void ***)&post->pp_particles, &post->nn_particles);
  free_pp_list((void ***)&post->pp_volume, &post->nn_volume);
  free_pp_list((void ***)&post->pp_global, &post->nn_global);
  free_pp_list((void ***)&post->pp_average, &post->nn_average);
  FREE_NULL(post->pp_error_data);
  post->nn_error_metrics = 0;
  FREE_NULL(post->u_post_proc);
  post->len_u_post_proc = 0;
}

/****************************************************************************/
/* Boundary conditions                                                      */
/****************************************************************************/

static void free_data_table(struct Data_Table *table) {
  if (table == NULL) {
    return;
  }
  /* f_name points at a string literal or a caller's buffer */
  free(table->t);
  free(table->t2);
  free(table->t3);
  free(table->f);
  free(table);
}

static void free_table_list(struct Data_Table **tables, int *count, int max) {
  for (int i = 0; i < max; i++) {
    free_data_table(tables[i]);
    tables[i] = NULL;
  }
  *count = 0;
}

/*
 * bc_dup_list[imtrx][i] holds the per-unknown BC lists of the node
 * bc_dup_nodes[imtrx][i] (see alloc_bc_unk_list_node() in mm_bc_conflict.c),
 * whose length is the node's unknown count, or 1 for an unknown-less node in a
 * multi-matrix problem.
 */
static void free_bc_dup_lists(GomaContext *ctx) {
  GomaBC *bc = &ctx->bc;
  const int nm = num_matrices(ctx);

  if (bc->bc_dup_list != NULL) {
    for (int imtrx = 0; imtrx < nm; imtrx++) {
      int ***lists = bc->bc_dup_list[imtrx];
      if (lists == NULL) {
        continue;
      }
      const int ndup = (bc->bc_dup_ptr != NULL) ? bc->bc_dup_ptr[imtrx] : 0;
      for (int i = 0; i < ndup; i++) {
        int **unk_lists = lists[i];
        if (unk_lists == NULL) {
          continue;
        }
        int num = 0;
        if (ctx->fem.nodes != NULL && bc->bc_dup_nodes != NULL && bc->bc_dup_nodes[imtrx]) {
          NODE_INFO_STRUCT *node = ctx->fem.nodes[bc->bc_dup_nodes[imtrx][i]];
          num = node->Nodal_Vars_Info[imtrx]->Num_Unknowns;
        }
        if (num <= 0 && nm > 1) {
          num = 1;
        }
        for (int unk = 0; unk < num; unk++) {
          free(unk_lists[unk]);
        }
        free(unk_lists);
      }
      free(lists);
    }
  }
  FREE_NULL(bc->bc_dup_list);
  free_per_matrix((void ***)&bc->bc_dup_nodes, nm);
  FREE_NULL(bc->bc_dup_ptr);
}

static void free_node_lists(int ***list, int n) {
  if (*list != NULL) {
    for (int i = 0; i < n; i++) {
      free((*list)[i]);
    }
  }
  FREE_NULL(*list);
}

/* rotation[node][vector equation][0..dim-1], allocated in bc/rotate.c */
static void free_rotation_vectors(GomaContext *ctx) {
  GomaFem *fem = &ctx->fem;
  const int nn = num_local_nodes(ctx);
  const int dim = (fem->pd_glob != NULL && fem->pd_glob[0] != NULL) ? fem->pd_glob[0]->Num_Dim : 0;

  if (fem->rotation != NULL) {
    for (int inode = 0; inode < nn; inode++) {
      struct Rotation_Vectors ***node_rot = fem->rotation[inode];
      if (node_rot == NULL) {
        continue;
      }
      for (int eq = 0; eq < NUM_VECTOR_EQUATIONS; eq++) {
        if (node_rot[eq] == NULL) {
          continue;
        }
        for (int k = 0; k < dim; k++) {
          free(node_rot[eq][k]);
        }
        free(node_rot[eq]);
      }
      free(node_rot);
    }
  }
  FREE_NULL(fem->rotation);
  free_node_lists(&fem->local_rot_list, nn);
  fem->rotation_allocated = FALSE;
}

void boundary_conditions_release(GomaContext *ctx) {
  GomaBC *bc = &ctx->bc;
  const int nm = num_matrices(ctx);
  const int nn = num_local_nodes(ctx);

  free_Surf_BC(&bc->first_elem_side_bc_array, ctx->mesh.num_internal_elems);
  free_Edge_BC(&bc->first_elem_edge_bc_array, ctx->mesh.num_internal_elems);
  free_bc_dup_lists(ctx);
  free_rotation_vectors(ctx);

  if (goma_automatic_rotations.rotation_nodes != NULL && ctx->exo != NULL) {
    free_rotations(ctx->exo, &goma_automatic_rotations.rotation_nodes);
    goma_automatic_rotations.automatic_rotations = false;
  }

  free_node_lists(&bc->ss_list, nn);
  free_node_lists(&bc->rot_list, nn);

  free_per_matrix((void ***)&bc->mesh_rotate_node, nm);
  free_per_matrix((void ***)&bc->mesh_rotate_ss, nm);
  FREE_NULL(bc->num_mesh_rotate);
  free_per_matrix((void ***)&bc->mom_rotate_node, nm);
  free_per_matrix((void ***)&bc->mom_rotate_ss, nm);
  FREE_NULL(bc->num_mom_rotate);

  for (int i = 0; i < MAX_MAT_PER_SS + 1; i++) {
    FREE_NULL(bc->ss_to_blks[i]);
  }
  FREE_NULL(bc->ss_internal_boundary);

  if (bc->rot_types != NULL) {
    for (int i = 0; i < ctx->problem.num_rot; i++) {
      free(bc->rot_types[i].elems);
    }
  }
  FREE_NULL(bc->rot_types);

  /* BC_Types[].table aliases bc_tables[] and .desc aliases BC_Desc[] or new_bc_desc[] */
  if (bc->bc_types != NULL) {
    for (int i = 0; i < ctx->problem.num_bc; i++) {
      free(bc->bc_types[i].u_BC);
    }
  }
  FREE_NULL(bc->bc_types);

  /* new descriptions take their names from BC_Desc[] */
  if (bc->new_bc_desc != NULL) {
    for (int i = 0; i < bc->num_new_bc_desc; i++) {
      free(bc->new_bc_desc[i]);
    }
  }
  FREE_NULL(bc->new_bc_desc);
  bc->num_new_bc_desc = 0;

  free_table_list(bc->bc_tables, &bc->num_bc_tables, MAX_BC_TABLES);
  free_table_list(bc->mp_tables, &bc->num_mp_tables, MAX_MP_TABLES);
  free_table_list(bc->ext_tables, &bc->num_ext_tables, MAX_EXT_TABLES);
  free_table_list(bc->ac_tables, &bc->num_ac_tables, MAX_AC_TABLES);
}

/****************************************************************************/
/* Problem setup: unknown maps, communication maps, nodes, element blocks   */
/****************************************************************************/

void problem_setup_release(GomaContext *ctx) {
  GomaProblem *p = &ctx->problem;
  GomaFem *fem = &ctx->fem;
  const int nm = num_matrices(ctx);
  const int nn = num_local_nodes(ctx);

  if (p->dolphin != NULL) {
    for (int imtrx = 0; imtrx < nm; imtrx++) {
      free_node_lists(&p->dolphin[imtrx], nn);
    }
  }
  FREE_NULL(p->dolphin);
  if (p->local_offset != NULL) {
    for (int imtrx = 0; imtrx < nm; imtrx++) {
      free_node_lists(&p->local_offset[imtrx], nn);
    }
  }
  FREE_NULL(p->local_offset);
  free_per_matrix((void ***)&p->first_unknown, nm);
  free_per_matrix((void ***)&p->idv, nm);
  free_per_matrix((void ***)&p->dofname, nm);
  free_per_matrix((void ***)&p->resname, nm);

  FREE_NULL(p->num_internal_dofs);
  FREE_NULL(p->num_boundary_dofs);
  FREE_NULL(p->num_external_dofs);
  FREE_NULL(p->num_universe_dofs);
  FREE_NULL(p->num_unknowns);
  FREE_NULL(p->num_ext_unknowns);

  /* cx[][].local_nodeces_send point into list_node_send */
  FREE_NULL(p->ptr_node_recv);
  FREE_NULL(p->ptr_node_send);
  FREE_NULL(p->list_node_send);
  FREE_NULL(p->ptr_fill_node_recv);
  FREE_NULL(p->ptr_fill_node_send);
  FREE_NULL(p->list_fill_node_send);
  free_per_matrix((void ***)&p->ptr_dof_send, nm);
  free_per_matrix((void ***)&p->list_dof_send, nm);
  FREE_NULL(p->node_index_fill);
  FREE_NULL(p->fill_node_list);
  free_per_matrix((void ***)&ctx->cx, nm);

  if (p->var_info != NULL) {
    for (int i = 0; i < p->num_var_info_records; i++) {
      variable_description_destroy(&p->var_info[i]);
    }
  }
  FREE_NULL(p->var_info);
  p->num_var_info_records = 0;
  if (p->nodal_vars_list != NULL) {
    for (int i = 0; i < p->nodal_vars_list_length; i++) {
      nodal_vars_destroy(&p->nodal_vars_list[i]);
    }
  }
  FREE_NULL(p->nodal_vars_list);
  p->nodal_vars_list_length = 0;

  for (int i = 0; i < ctx->num_var_init; i++) {
    FREE_NULL(p->var_init[i].u_pars);
  }
  for (int mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    for (int i = 0; i < p->num_var_init_mat[mn]; i++) {
      FREE_NULL(p->var_init_mat[mn][i].u_pars);
    }
  }

  FREE_NULL(p->matilda);
  FREE_NULL(p->lsa_wave_numbers);
  p->lsa_number_wave_numbers = 0;

  if (fem->shell_blocks != NULL) {
    for (int i = 0; i < fem->num_shell_blocks; i++) {
      struct Shell_Block *sb = fem->shell_blocks[i];
      if (sb != NULL) {
        for (int n = 0; n < sb->num_nbr_blocks; n++) {
          free(sb->nbr_elem_ids[n]);
        }
        free(sb);
      }
    }
  }
  FREE_NULL(fem->shell_blocks);
  fem->num_shell_blocks = 0;
  if (fem->elem_friends != NULL && ctx->exo != NULL) {
    for (int i = 0; i < ctx->exo->num_elems; i++) {
      free(fem->elem_friends[i]);
    }
  }
  FREE_NULL(fem->elem_friends);
  FREE_NULL(fem->num_elem_friends);

  if (fem->element_blocks != NULL && ctx->exo != NULL) {
    free_element_blocks(ctx->exo);
  }
  FREE_NULL(fem->element_blocks);
  if (fem->nodes != NULL) {
    free_nodes();
  }
}

/****************************************************************************/
/* Materials                                                                */
/****************************************************************************/

static void free_dbl_vectors(dbl **v, int n) {
  for (int i = 0; i < n; i++) {
    FREE_NULL(v[i]);
  }
}

#define FREE_VECTORS(arr) free_dbl_vectors((arr), (int)(sizeof(arr) / sizeof((arr)[0])))

/*
 * Free every pointer member of a material read from the material file.
 * Material tables live in bc.mp_tables[]; Material_Properties::table is unused.
 */
static void material_properties_free(struct Material_Properties *m) {
  free(m->Matrl_Elem_Blk_Ids);
  free(m->Species_Names);
  free(m->Porous_Names);
  if (m->DensityJac != NULL) {
    propertyJac_destroy(&m->DensityJac);
  }
  free(m->mp2nd);

  free(m->u_thermal_conductivity);
  free(m->u_electrical_conductivity);
  free(m->u_permittivity);
  free(m->u_elect_surf_diffusivity);
  free(m->u_magnetic_permeability);
  free(m->u_shell_user_par);
  free(m->u_acoustic_impedance);
  free(m->u_wave_number);
  free(m->u_acoustic_absorption);
  free(m->u_refractive_index);
  free(m->u_light_absorption);
  free(m->u_extinction_index);
  free(m->u_viscosity);
  free(m->u_dilationalViscosity);
  free(m->u_surface_tension);
  free(m->u_heat_capacity);
  free(m->u_Volume_Expansion);
  free(m->u_density);
  free(m->u_reaction_rate);
  free(m->u_solution_temperature);
  free(m->u_thermodynamic_potential);
  free(m->u_interfacial_area);
  free(m->u_porosity);
  free(m->u_porous_compressibility);
  free(m->u_permeability);
  free(m->u_rel_gas_perm);
  free(m->u_rel_liq_perm);
  free(m->u_saturation);
  free(m->u_cap_pres);
  free(m->u_porous_gas_constants);
  free(m->u_porous_sink_constants);
  free(m->u_momentum_source);
  free(m->u_heat_source);
  free(m->u_mass_source);
  free(m->u_mesh_source);
  free(m->u_current_source);
  free(m->u_moment_source);
  free(m->u_heightU_function_constants);
  free(m->u_heightL_function_constants);
  free(m->u_veloU_function_constants);
  free(m->u_veloL_function_constants);
  free(m->u_dcaU_function_constants);
  free(m->u_dcaL_function_constants);
  free(m->u_lubsource_function_constants);
  free(m->u_FilmEvap_function_constants);
  free(m->u_DisjPress_function_constants);
  free(m->u_SlipCoeff_function_constants);
  free(m->u_DiffCoeff_function_constants);
  free(m->u_PorousShellClosedPorosity_function_constants);
  free(m->u_PorousShellClosedHeight_function_constants);
  free(m->u_PorousShellClosedRadius_function_constants);
  free(m->u_PorousShellClosedP0_function_constants);
  free(m->u_PorousShellPatm_function_constants);
  free(m->u_PorousShellPref_function_constants);
  free(m->u_PorousShellCrossKappa_function_constants);
  free(m->u_PorousShellInitPorePres_function_constants);
  free(m->u_PorousShellDiffusivity_function_constants);
  free(m->u_PorousShellRT_function_constants);
  free(m->u_PorousShellHenry_function_constants);
  free(m->u_incident_wave);
  free(m->u_FlowingLiquid_viscosity);
  free(m->tfmp_density_const);
  free(m->tfmp_viscosity_const);
  free(m->tfmp_diff_const);
  free(m->tfmp_rel_perm_const);
  free(m->tfmp_dissolution_const);
  free(m->tfmp_drop_lattice_const);
  free(m->shell_tangent_seed_vec_const);

  FREE_VECTORS(m->u_diffusivity);
  FREE_VECTORS(m->u_gadiffusivity);
  FREE_VECTORS(m->u_mdiffusivity);
  FREE_VECTORS(m->u_fdiffusivity);
  FREE_VECTORS(m->u_nscoeff);
  FREE_VECTORS(m->u_gdiffusivity);
  FREE_VECTORS(m->SBM_Lengths2);
  FREE_VECTORS(m->u_cdiffusivity);
  FREE_VECTORS(m->u_qdiffusivity);
  FREE_VECTORS(m->u_settling);
  FREE_VECTORS(m->u_reference_concn);
  FREE_VECTORS(m->u_porous_diffusivity);
  FREE_VECTORS(m->u_porous_vapor_pressure);
  FREE_VECTORS(m->u_porous_vol_expansion);
  FREE_VECTORS(m->u_species_source);
  FREE_VECTORS(m->u_species_vol_expansion);
  FREE_VECTORS(m->u_PorousShellPorosity);
  FREE_VECTORS(m->u_PorousShellHeight);
  FREE_VECTORS(m->u_PorousShellPermeability);
  FREE_VECTORS(m->u_PorousShellCrossPermeability);
  FREE_VECTORS(m->u_PorousShellRelPerm);
  FREE_VECTORS(m->u_PorousShellCapPres);
  FREE_VECTORS(m->u_vapor_pressure);
}

static void generalized_newtonian_free(struct Generalized_Newtonian *gn) {
  if (gn == NULL) {
    return;
  }
  free(gn->u_mu0);
  free(gn->u_nexp);
  free(gn->u_muinf);
  free(gn->u_lam);
  free(gn->u_aexp);
  free(gn->u_atexp);
  free(gn->u_T_shift);
  free(gn->u_wlfc2);
  free(gn->u_tau_y);
  free(gn->u_thixo_factor);
}

static void elastic_constitutive_free(struct Elastic_Constitutive *e) {
  if (e == NULL) {
    return;
  }
  free(e->u_mu);
  free(e->u_mu_ns);
  free(e->multi_contact_line_distances);
  free(e->u_lambda);
  free(e->u_lame_TempShift);
  free(e->u_bend_stiffness);
  free(e->u_exten_stiffness);
  free(e->u_poisson);
  free(e->u_v_mesh_sfs);
  free(e->u_fxi);
  free(e->u_geta);
  free(e->u_hzeta);
  free(e->u_thermal_expansion);
  free(e->u_solid_viscosity);
  free(e->u_solid_dil_viscosity);
  free(e->u_solid_retardation);
}

/*
 * Free what reading the material files hung off the per-material structures.
 * The structures themselves belong to fem_structures_free().
 */
void materials_release(GomaFem *fem) {
  for (int mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    if (fem->mp_glob != NULL && fem->mp_glob[mn] != NULL) {
      material_properties_free(fem->mp_glob[mn]);
    }
    if (fem->gn_glob != NULL) {
      generalized_newtonian_free(fem->gn_glob[mn]);
    }
    if (fem->elc_glob != NULL) {
      elastic_constitutive_free(fem->elc_glob[mn]);
    }
    if (fem->elc_rs_glob != NULL) {
      elastic_constitutive_free(fem->elc_rs_glob[mn]);
    }
    if (fem->evpl_glob != NULL && fem->evpl_glob[mn] != NULL) {
      free(fem->evpl_glob[mn]->u_plastic_mu);
      free(fem->evpl_glob[mn]->u_yield);
    }
    if (fem->vn_glob != NULL && fem->vn_glob[mn] != NULL) {
      free(fem->vn_glob[mn]->dg_J_model_wt);
      free(fem->vn_glob[mn]->shift);
    }
    if (fem->ve_glob != NULL) {
      for (int mode = 0; mode < MAX_MODES; mode++) {
        if (fem->ve_glob[mn][mode] != NULL) {
          generalized_newtonian_free(fem->ve_glob[mn][mode]->gn);
        }
      }
    }
  }
}

/****************************************************************************/
/* Level set, continuation, augmenting conditions, external fields          */
/****************************************************************************/

static void level_set_free(struct Level_Set_Data *ls, int owns_output_file) {
  if (ls == NULL) {
    return;
  }
  if (owns_output_file) {
    free(ls->output_file);
  }
  free(ls->sm_object_name);
  free(ls->sm_object_type);
  while (ls->embedded_bc != NULL) {
    struct LS_Embedded_BC *next = ls->embedded_bc->next;
    free(ls->embedded_bc);
    ls->embedded_bc = next;
  }
  free_surf_list(&ls->init_surf_list);
  free_surf_list(&ls->last_surf_list);
  free_surf_list(&ls->Ext_Vel_Surf_List);
  free(ls);
}

void fem_input_release(GomaContext *ctx) {
  GomaFem *fem = &ctx->fem;
  GomaProblem *p = &ctx->problem;

  /* fem.ls may temporarily point at a phase function; it owns output_file otherwise */
  int ls_is_phase_function = 0;
  if (fem->pfd != NULL) {
    if (fem->pfd->ls != NULL) {
      for (int i = 0; i < fem->pfd->num_phase_funcs; i++) {
        if (fem->pfd->ls[i] == fem->ls) {
          ls_is_phase_function = 1;
        }
        level_set_free(fem->pfd->ls[i], 0);
      }
    }
    free(fem->pfd->ls);
    free(fem->pfd->jac_info);
    FREE_NULL(fem->pfd);
  }
  if (!ls_is_phase_function) {
    level_set_free(fem->ls, 1);
  }
  fem->ls = NULL;
  FREE_NULL(fem->lsi);

  if (fem->augc != NULL) {
    for (int i = 0; i < p->n_ac; i++) {
      free(fem->augc[i].DataFlt);
      free(fem->augc[i].d_evol_dx);
      free(fem->augc[i].d_lsvel_dx);
      free(fem->augc[i].d_lsvol_dx);
      free(fem->augc[i].Aprepro_lib_string);
    }
  }
  FREE_NULL(fem->augc);
  FREE_NULL(fem->hunt);
  FREE_NULL(fem->cpcc);
  FREE_NULL(fem->tpcc);
  FREE_NULL(fem->cpuc);
  FREE_NULL(fem->tpuc);

  if (fem->upd != NULL) {
    struct Uniform_Problem_Description *upd = fem->upd;
    if (upd->turbulent_info != NULL) {
      free(upd->turbulent_info->wall_distances);
      free(upd->turbulent_info->side_set_ids);
      free(upd->turbulent_info->node_set_ids);
    }
    FREE_NULL(upd->turbulent_info);
    FREE_NULL(upd->solver_info);
    FREE_NULL(upd->mesh_correction_damping);
    FREE_NULL(upd->mesh_correction_tolerances);
  }

  if (fem->efv != NULL) {
    struct External_Field_Variables *efv = fem->efv;
    for (int w = 0; w < MAX_EXTERNAL_FIELD; w++) {
      FREE_NULL(efv->ext_fld_ndl_val[w]);
      FREE_NULL(efv->ext_fld_ndl_val_old[w]);
      FREE_NULL(efv->ext_fld_ndl_val_older[w]);
    }
    for (int a = 0; a < 2 * DIM; a++) {
      FREE_NULL(efv->init_displacement_ndl_val[a]);
    }
  }

  if (fem->pg != NULL) {
    /* the Matrix_Data entries point at solver-owned vectors */
    FREE_NULL(fem->pg->matrices);
  }
  if (fem->tran != NULL) {
    FREE_NULL(fem->tran->xdbl_dot);
    FREE_NULL(fem->tran->xdbl_dot_old);
  }
  if (fem->lec != NULL) {
    FREE_NULL(fem->lec->R);
    FREE_NULL(fem->lec->J);
    FREE_NULL(fem->lec->J_stress_neighbor);
  }
}

/****************************************************************************/

/*
 * Solver state kept between calls goes back to its defaults, so a released
 * context starts its next solve like a new one.
 */
static void solver_state_release(GomaContext *ctx) {
  GomaSolver *sv = &ctx->solver;
  sv->first_linear_solve = TRUE;
  sv->nl_prev_matrix = 0;
  sv->nl_dofs_hidden = FALSE;
  sv->nl_umf_system_id = 0;
  sv->loca_umf_id = 0;
  sv->nl_factor_flag = 0;
  sv->eggroll_umf_system_id = 0;
  sv->loca_first_linear_solve = FALSE;
  sv->loca_factor_flag = 0;
  sv->loca_stab_first_linear_solve = TRUE;
  sv->loca_stab_umf_id = -1;
  sv->loca_stab_factor_flag = 0;
  sv->loca_sv_init = TRUE;
  sv->loca_step_print = 0;
  sv->loca_n_print = 0;
  sv->loca_eig_old_step = -1;
  sv->loca_eig_nprint = -1;
  sv->sl_init_calls = 0;
  sv->matrix_dump_calls = 0;
  sv->matrix_dump_msr_index = 0;
  sv->matrix_dump_vbr_index = 0;
  sv->sparse_calls = 0;
  for (int i = 0; i < MAX_NUM_MATRICES; i++) {
    sv->stratimikos_tpetra_param_echo[i] = (i == 0);
    sv->stratimikos_param_echo[i] = FALSE;
  }
}

void goma_problem_release(GomaContext *ctx) {
  /* solution history and solver data kept between solve_problem() calls */
  solve_problem_release();
  particle_tracker_release();

  /* factorizations the direct solvers keep between calls */
  sl_umf_release();
  lu_release();
  lu_decomp_release();
#if defined(GOMA_ENABLE_AMESOS) && defined(TRILINOS)
  amesos_release();
  trilinos_solve_ls_release();
#endif
#if defined(GOMA_ENABLE_PETSC) && !(PETSC_USE_COMPLEX)
  if (ctx->fem.upd != NULL) {
    goma_petsc_free_post_proc();
  }
#endif

  /* subgrid integration points kept from the last Courant time step estimate */
  FREE_NULL(Subgrid_Int.s);
  FREE_NULL(Subgrid_Int.wt);
  FREE_NULL(Subgrid_Int.ip_sign);
  Subgrid_Int.ip_total = 0;

  post_processing_release(&ctx->post);
  boundary_conditions_release(ctx);
  problem_setup_release(ctx);
  materials_release(&ctx->fem);
  fem_input_release(ctx);

  if (ctx->exo != NULL) {
    free_exo(ctx->exo);
    FREE_NULL(ctx->exo);
  }
  if (ctx->dpi != NULL) {
    if (ctx->num_proc > 1) {
      free_dpi(ctx->dpi);
    } else {
      free_dpi_uni(ctx->dpi);
    }
    FREE_NULL(ctx->dpi);
  }

  fem_structures_free(&ctx->fem);
  solver_state_release(ctx);
}
