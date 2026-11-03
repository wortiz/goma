/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2023 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/
#include "mm_fill_ls_capillary_bcs.h"
#include "ac_particles.h"
#ifdef GOMA_ENABLE_AZTEC
#include "az_aztec.h"
#endif
#include "bc_colloc.h"
#include "bc_contact.h"
#include "density.h"
#include "el_elm.h"
#include "el_elm_info.h"
#include "el_geom.h"
#include "exo_struct.h"
#include "load_field_variables.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_dil_viscosity.h"
#include "mm_eh.h"
#include "mm_fill_aux.h"
#include "mm_fill_common.h"
#include "mm_fill_fill.h"
#include "mm_fill_ls.h"
#include "mm_fill_momentum.h"
#include "mm_fill_population.h"
#include "mm_fill_ptrs.h"
#include "mm_fill_rs.h"
#include "mm_fill_shell.h"
#include "mm_fill_solid.h"
#include "mm_fill_species.h"
#include "mm_fill_stabilization.h"
#include "mm_fill_stress.h"
#include "mm_fill_terms.h"
#include "mm_fill_util.h"
#include "mm_flux.h"
#include "mm_mp.h"
#include "mm_mp_const.h"
#include "mm_mp_structs.h"
#include "mm_ns_bc.h"
#include "mm_post_def.h"
#include "mm_qtensor_model.h"
#include "mm_shell_util.h"
#include "mm_species.h"
#include "mm_unknown_map.h"
#include "mm_viscosity.h"
#include "rf_allo.h"
#include "rf_bc.h"
#include "rf_bc_const.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_solver.h"
#include "rf_vars_const.h"
#include "sl_util.h"
#include "sl_util_structs.h"
#include "std.h"
#include "stdbool.h"
#include "user_mp.h"
#include "user_mp_gen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int continuous_surface_tension_old(double st,
                                          double csf[DIM][DIM],
                                          struct Level_Set_Interface *lsi_old) {
  int a, b;
  int status = 0;

  int var;

  var = goma_ctx->fem.ls->var;

  if (var != FILL) {
    GOMA_EH(GOMA_ERROR, "Unknown fill variable");
  }

  /* Fetch the level set interface functions. */
  load_lsi_old(goma_ctx->fem.ls->Length_Scale, lsi_old);

  /* If we're not near the zero level set, then csf is a zero tensor. */
  if (!lsi_old->near) {
    for (a = 0; a < goma_ctx->problem.vim; a++) {
      for (b = 0; b < goma_ctx->problem.vim; b++) {
        csf[a][b] = 0;
      }
    }

    return (status);
  }

  /* Calculate the CSF tensor. */
  for (a = 0; a < goma_ctx->problem.vim; a++) {
    for (b = 0; b < goma_ctx->problem.vim; b++) {
      csf[a][b] =
          st * lsi_old->delta * (((double)delta(a, b)) - lsi_old->normal[a] * lsi_old->normal[b]);
    }
  }

  return status;
}
int assemble_csf_tensor(void) {
  int i, j, a, b, p, q, ii, ledof;
  int eqn, peqn, var, pvar;

  struct Basis_Functions *bfm;
  dbl(*grad_phi_i_e_a)[DIM] = NULL;

  double wt, det_J, h3, phi_j, d_area;

  double csf[DIM][DIM];
  double d_csf_dF[DIM][DIM][MDE];
  double d_csf_dX[DIM][DIM][DIM][MDE];

  double source, source_etm;

  eqn = R_MOMENTUM1;
  if (!goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn]) {
    return (0);
  }

  wt = goma_ctx->fem.fv->wt;
  h3 = goma_ctx->fem.fv->h3;
  if (goma_ctx->fem.ls->on_sharp_surf) /* sharp interface */
  {
    det_J = goma_ctx->fem.fv->sdet;
  } else /* diffuse interface */
  {
    det_J = goma_ctx->fem.bf[eqn]->detJ;
  }

  d_area = wt * det_J * h3;

  memset(csf, 0, sizeof(double) * DIM * DIM);
  memset(d_csf_dF, 0, sizeof(double) * DIM * DIM * MDE);
  memset(d_csf_dX, 0, sizeof(double) * DIM * DIM * DIM * MDE);

  continuous_surface_tension(goma_ctx->fem.mp->surface_tension, csf, d_csf_dF, d_csf_dX);

  /* finite difference calculation of path dependencies for
          subelement integration
          */
  if (goma_ctx->fem.ls->CalcSurfDependencies) {
    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];
      bfm = goma_ctx->fem.bf[eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          grad_phi_i_e_a = bfm->grad_phi_e[i][a];

          source = 0.;

          for (p = 0; p < goma_ctx->problem.vim; p++) {
            for (q = 0; q < goma_ctx->problem.vim; q++) {
              source += grad_phi_i_e_a[p][q] * csf[q][p];
            }
          }

          source *= -det_J * wt * h3;
          source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][LOG2_SOURCE];

          /* J_m_F
           */
          var = LS;
          pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            phi_j = goma_ctx->fem.bf[var]->phi[j];
            goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += source * phi_j;
          }
        }
      }
    }
    return (0);
  }

  /*
   * Residuals ____________________________________________________________________________
   */

  if (goma_ctx->fem.af->Assemble_Residual) {

    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];
      bfm = goma_ctx->fem.bf[eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          grad_phi_i_e_a = bfm->grad_phi_e[i][a];

          source = 0.;

#ifdef DO_NO_UNROLL

          for (p = 0; p < goma_ctx->problem.vim; p++) {
            for (q = 0; q < goma_ctx->problem.vim; q++) {
              source += grad_phi_i_e_a[p][q] * csf[q][p];
            }
          }
#else
          source += grad_phi_i_e_a[0][0] * csf[0][0];
          source += grad_phi_i_e_a[1][1] * csf[1][1];
          source += grad_phi_i_e_a[0][1] * csf[1][0];
          source += grad_phi_i_e_a[1][0] * csf[0][1];
          if (goma_ctx->problem.vim == 3) {
            source += grad_phi_i_e_a[2][2] * csf[2][2];
            source += grad_phi_i_e_a[2][1] * csf[1][2];
            source += grad_phi_i_e_a[2][0] * csf[0][2];
            source += grad_phi_i_e_a[1][2] * csf[2][1];
            source += grad_phi_i_e_a[0][2] * csf[2][0];
          }
#endif

          source *= -det_J * wt * h3;
          source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][LOG2_SOURCE];

          goma_ctx->fem.lec->R[LEC_R_INDEX(peqn, ii)] += source;
        }
      }
    }
  }

  /*
   * Yacobian terms...
   */

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];
      bfm = goma_ctx->fem.bf[eqn];
      source_etm = goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][LOG2_SOURCE];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          grad_phi_i_e_a = bfm->grad_phi_e[i][a];

          /* J_m_F
           */
          var = LS;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
              phi_j = goma_ctx->fem.bf[var]->phi[j];

              source = 0.;

#ifdef DO_NO_UNROLL

              for (p = 0; p < goma_ctx->problem.vim; p++) {
                for (q = 0; q < goma_ctx->problem.vim; q++) {
                  source += grad_phi_i_e_a[p][q] * d_csf_dF[q][p][j];
                }
              }

#else
              source += grad_phi_i_e_a[0][0] * d_csf_dF[0][0][j];
              source += grad_phi_i_e_a[0][1] * d_csf_dF[1][0][j];
              source += grad_phi_i_e_a[1][0] * d_csf_dF[0][1][j];
              source += grad_phi_i_e_a[1][1] * d_csf_dF[1][1][j];

              if (goma_ctx->problem.vim == 3) {
                source += grad_phi_i_e_a[2][0] * d_csf_dF[0][2][j];
                source += grad_phi_i_e_a[2][1] * d_csf_dF[1][2][j];
                source += grad_phi_i_e_a[2][2] * d_csf_dF[2][2][j];
                source += grad_phi_i_e_a[0][2] * d_csf_dF[2][0][j];
                source += grad_phi_i_e_a[1][2] * d_csf_dF[2][1][j];
              }
#endif

              source *= -d_area;
              source *= source_etm;

              goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += source;
            }
          }
          var = MESH_DISPLACEMENT1;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (b = 0; b < goma_ctx->problem.vim; b++) {
              var = MESH_DISPLACEMENT1 + b;
              pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

              for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
                phi_j = goma_ctx->fem.bf[var]->phi[j];

                source = 0.;
#ifdef DO_NO_UNROLL
                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  for (q = 0; q < goma_ctx->problem.vim; q++) {
                    source += grad_phi_i_e_a[p][q] * d_csf_dX[q][p][b][j];
                  }
                }
#else

                source += grad_phi_i_e_a[0][0] * d_csf_dX[0][0][b][j];
                source += grad_phi_i_e_a[0][1] * d_csf_dX[1][0][b][j];
                source += grad_phi_i_e_a[1][0] * d_csf_dX[0][1][b][j];
                source += grad_phi_i_e_a[1][1] * d_csf_dX[1][1][b][j];

                if (goma_ctx->problem.vim == 3) {
                  source += grad_phi_i_e_a[2][0] * d_csf_dX[0][2][b][j];
                  source += grad_phi_i_e_a[2][1] * d_csf_dX[1][2][b][j];
                  source += grad_phi_i_e_a[2][2] * d_csf_dX[2][2][b][j];
                  source += grad_phi_i_e_a[0][2] * d_csf_dX[2][0][b][j];
                  source += grad_phi_i_e_a[1][2] * d_csf_dX[2][1][b][j];
                }
#endif

                source *= -d_area;
                source *= source_etm;

                goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += source;
              }
            }
          }
        }
      }
    }
  }

  return (1);
}
int assemble_div_n_source(void) {
  int i, j, a, b, p, ii, ledof;
  int eqn, peqn, var, pvar;
  int dim;

  struct Basis_Functions *bfm;

  double wt, det_J, h3, phi_i, phi_j;
  double source, source1;

  eqn = R_MOMENTUM1;
  if (!goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn]) {
    return (-1);
  }

  wt = goma_ctx->fem.fv->wt;
  h3 = goma_ctx->fem.fv->h3;

  if (goma_ctx->fem.ls->on_sharp_surf) /* sharp interface */
  {
    det_J = goma_ctx->fem.fv->sdet;
  } else /* diffuse interface */
  {
    det_J = goma_ctx->fem.bf[eqn]->detJ;
  }

  dim = goma_ctx->fem.pd->Num_Dim;

  /* finite difference calculation of path dependencies for
     subelement integration
  */
  if (goma_ctx->fem.ls->CalcSurfDependencies) {
    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];
      bfm = goma_ctx->fem.bf[eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          phi_i = bfm->phi[i];

          source = -goma_ctx->fem.mp->surface_tension * goma_ctx->fem.fv->div_n *
                   goma_ctx->fem.fv->n[a] * goma_ctx->fem.lsi->delta;

          source *= phi_i * wt * det_J * h3;
          source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_SOURCE)];

          /* J_m_F
           */
          var = LS;
          pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            phi_j = goma_ctx->fem.bf[var]->phi[j];
            goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += source * phi_j;
          }
        }
      }
    }
    return (0);
  }

  /*
   * Residuals
   * ________________________________________________________________________________
   */

  if (goma_ctx->fem.af->Assemble_Residual) {
    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];
      bfm = goma_ctx->fem.bf[eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          phi_i = bfm->phi[i];

          source = 0.0;

          source = -goma_ctx->fem.mp->surface_tension * goma_ctx->fem.fv->div_n *
                   goma_ctx->fem.fv->n[a] * goma_ctx->fem.lsi->delta;

          source *= phi_i * wt * det_J * h3;
          source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_SOURCE)];

          goma_ctx->fem.lec->R[LEC_R_INDEX(peqn, ii)] += source;
        }
      }
    }
  }

  /*
   * Jacobian terms...
   */

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];
      bfm = goma_ctx->fem.bf[eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          phi_i = bfm->phi[i];

          /*
           * J_m_n
           */

          for (b = 0; b < dim; b++) {
            var = NORMAL1 + b;
            if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {

              pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

              for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {

                source = goma_ctx->fem.fv->div_n * goma_ctx->fem.bf[var]->phi[j] * delta(a, b);

                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  source += goma_ctx->fem.fv->n[a] * goma_ctx->fem.bf[var]->grad_phi_e[j][b][p][p];
                }
                source *= -goma_ctx->fem.mp->surface_tension * goma_ctx->fem.lsi->delta;
                source *= phi_i * wt * det_J * h3;
                source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_SOURCE)];

                goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += source;
              }
            }
          }

          /*
           * J_m_F
           */

          var = LS;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
              source = 0.0;

              source = -goma_ctx->fem.mp->surface_tension * goma_ctx->fem.fv->div_n *
                       goma_ctx->fem.fv->n[a];
              source *= goma_ctx->fem.lsi->d_delta_dF[j];
              source *= phi_i * wt * det_J * h3;
              source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_SOURCE)];

              goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += source;
            }
          }

          /*
           * J_m_d
           */

          var = MESH_DISPLACEMENT1;

          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (b = 0; b < dim; b++) {
              var = MESH_DISPLACEMENT1 + b;
              pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

              for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
                source = -goma_ctx->fem.mp->surface_tension * goma_ctx->fem.fv->div_n *
                         goma_ctx->fem.fv->n[a] * goma_ctx->fem.lsi->delta;

                source *= goma_ctx->fem.bf[var]->d_det_J_dm[b][j] * h3 +
                          goma_ctx->fem.fv->dh3dmesh[b][j] * det_J;
                source *= wt * phi_i;
                source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_SOURCE)];

                source1 = -goma_ctx->fem.mp->surface_tension *
                          goma_ctx->fem.fv->d_div_n_dmesh[b][j] * goma_ctx->fem.fv->n[a] *
                          goma_ctx->fem.lsi->delta;
                source1 *= wt * phi_i * det_J * h3;
                source1 *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_SOURCE)];

                goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += source + source1;
              }
            }
          }
        }
      }
    }
  }
  return (0);
}
int assemble_div_s_n_source(void) {
  int i, j, a, b, p, q, ii, ledof;
  int eqn, peqn, var, pvar;
  int dim;

  struct Basis_Functions *bfm;

  double wt, det_J, h3, phi_i, phi_j;
  double source, source1, source2, source3;

  eqn = R_MOMENTUM1;
  if (!goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn]) {
    return (-1);
  }

  wt = goma_ctx->fem.fv->wt;
  h3 = goma_ctx->fem.fv->h3;

  if (goma_ctx->fem.ls->on_sharp_surf) /* sharp interface */
  {
    det_J = goma_ctx->fem.fv->sdet;
  } else /* diffuse interface */
  {
    det_J = goma_ctx->fem.bf[eqn]->detJ;
  }

  dim = goma_ctx->fem.pd->Num_Dim;

  /* finite difference calculation of path dependencies for
     subelement integration
   */
  if (goma_ctx->fem.ls->CalcSurfDependencies) {
    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];
      bfm = goma_ctx->fem.bf[eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          phi_i = bfm->phi[i];

          source = -goma_ctx->fem.mp->surface_tension * goma_ctx->fem.fv->div_s_n *
                   goma_ctx->fem.fv->n[a] * goma_ctx->fem.lsi->delta;

          source *= phi_i * wt * det_J * h3;
          source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_SOURCE)];

          /* J_m_F
           */
          var = LS;
          pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            phi_j = goma_ctx->fem.bf[var]->phi[j];
            goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += source * phi_j;
          }
        }
      }
    }
    return (0);
  }

  /*
   * Wesiduals
   * ________________________________________________________________________________
   */

  if (goma_ctx->fem.af->Assemble_Residual) {
    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];
      bfm = goma_ctx->fem.bf[eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          phi_i = bfm->phi[i];

          source = 0.0;

          source = -goma_ctx->fem.mp->surface_tension * goma_ctx->fem.fv->div_s_n *
                   goma_ctx->fem.fv->n[a] * goma_ctx->fem.lsi->delta;

          source *= phi_i * wt * det_J * h3;
          source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_SOURCE)];

          goma_ctx->fem.lec->R[LEC_R_INDEX(peqn, ii)] += source;
        }
      }
    }
  }

  /*
   * Yacobian terms...
   */

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];
      bfm = goma_ctx->fem.bf[eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          phi_i = bfm->phi[i];

          /*
           * J_m_n
           */

          for (b = 0; b < dim; b++) {
            var = NORMAL1 + b;
            if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {

              pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

              for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
                source1 = 0.0;
                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  source1 -= goma_ctx->fem.fv->n[p] * goma_ctx->fem.fv->grad_n[p][b] +
                             goma_ctx->fem.fv->n[p] * goma_ctx->fem.fv->grad_n[b][p];
                }
                source1 *= goma_ctx->fem.fv->n[a] * goma_ctx->fem.bf[var]->phi[j];

                source2 = 0.0;
                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  for (q = 0; q < goma_ctx->problem.vim; q++) {
                    source2 += (delta(p, q) - goma_ctx->fem.fv->n[p] * goma_ctx->fem.fv->n[q]) *
                               goma_ctx->fem.bf[var]->grad_phi_e[j][b][p][q];
                  }
                }
                source2 *= goma_ctx->fem.fv->n[a];

                source3 = delta(a, b) * goma_ctx->fem.fv->div_s_n * goma_ctx->fem.bf[var]->phi[j];

                source = source1 + source2 + source3;
                source *= -goma_ctx->fem.mp->surface_tension * goma_ctx->fem.lsi->delta;
                source *= phi_i * wt * det_J * h3;
                source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_SOURCE)];

                goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += source;
              }
            }
          }
          /*
           * J_m_F
           */
          var = FILL;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
              source = 0.0;

              source = -goma_ctx->fem.mp->surface_tension * goma_ctx->fem.fv->div_s_n *
                       goma_ctx->fem.fv->n[a];
              source *= goma_ctx->fem.lsi->d_delta_dF[j];
              source *= phi_i * wt * det_J * h3;
              source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_SOURCE)];

              goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += source;
            }
          }
        }
      }
    }
  }
  return (0);
}
int assemble_cap_hysing(double dt, double scale) {
  int i, j, a, b, p, q, k, ii, ledof;
  int eqn, peqn, var, pvar;

  struct Level_Set_Interface lsi_old_struct;
  struct Level_Set_Interface *lsi_old = &lsi_old_struct;

  struct Basis_Functions *bfm;
  dbl(*grad_phi_i_e_a)[DIM] = NULL;

  double wt, det_J, h3;

  double csf[DIM][DIM];

  double grad_s_v[DIM][DIM];
  //  double grad_s_phi_i_e_a[DIM][DIM];
  double d_grad_s_v_dv[DIM][DIM];

  double source;
  double diffusion;

  eqn = R_MOMENTUM1;
  if (!goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn]) {
    return (0);
  }

  wt = goma_ctx->fem.fv->wt;
  h3 = goma_ctx->fem.fv->h3;
  if (goma_ctx->fem.ls->on_sharp_surf) /* sharp interface */
  {
    det_J = goma_ctx->fem.fv->sdet;
  } else /* diffuse interface */
  {
    det_J = goma_ctx->fem.bf[eqn]->detJ;
  }

  memset(csf, 0, sizeof(double) * DIM * DIM);

  continuous_surface_tension_old(goma_ctx->fem.mp->surface_tension, csf, lsi_old);

  for (p = 0; p < goma_ctx->problem.vim; p++) {
    for (q = 0; q < goma_ctx->problem.vim; q++) {
      grad_s_v[p][q] = goma_ctx->fem.fv->grad_v[p][q];
      for (k = 0; k < goma_ctx->problem.vim; k++) {
        grad_s_v[p][q] -=
            (lsi_old->normal[p] * lsi_old->normal[k]) * goma_ctx->fem.fv->grad_v[k][q];
      }
    }
  }

  /* finite difference calculation of path dependencies for
     subelement integration
  */
  if (goma_ctx->fem.ls->CalcSurfDependencies) {
    GOMA_EH(GOMA_ERROR, "Calc surf dependencies not implemented");
  }

  /*
   * Residuals ____________________________________________________________________________
   */

  if (goma_ctx->fem.af->Assemble_Residual) {

    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];
      bfm = goma_ctx->fem.bf[eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          grad_phi_i_e_a = bfm->grad_phi_e[i][a];

          source = 0.;

          for (p = 0; p < goma_ctx->problem.vim; p++) {
            for (q = 0; q < goma_ctx->problem.vim; q++) {
              source += grad_phi_i_e_a[p][q] * csf[q][p];
            }
          }

          source *= -det_J * wt * h3;
          source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][LOG2_SOURCE];

          diffusion = 0;
          /*
           * 			grad_v[a][b] = d v_b
           *				       -----
           *				       d x_a
           */
          for (p = 0; p < goma_ctx->problem.vim; p++) {
            for (q = 0; q < goma_ctx->problem.vim; q++) {
              diffusion += goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q] * grad_s_v[p][q];
            }
          }

          diffusion *=
              -det_J * wt * h3 * (dt * goma_ctx->fem.mp->surface_tension * lsi_old->delta * scale);

          goma_ctx->fem.lec->R[LEC_R_INDEX(peqn, ii)] += source + diffusion;
        }
      }
    }
  }

  /*
   * Yacobian terms...
   */

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];
      bfm = goma_ctx->fem.bf[eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          grad_phi_i_e_a = bfm->grad_phi_e[i][a];

          var = VELOCITY1;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (b = 0; b < goma_ctx->problem.vim; b++) {
              var = VELOCITY1 + b;
              pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

              for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  for (q = 0; q < goma_ctx->problem.vim; q++) {
                    d_grad_s_v_dv[p][q] = goma_ctx->fem.bf[var]->grad_phi_e[j][b][p][q];
                    for (k = 0; k < goma_ctx->problem.vim; k++) {
                      d_grad_s_v_dv[p][q] -= lsi_old->normal[p] * lsi_old->normal[k] *
                                             goma_ctx->fem.bf[var]->grad_phi_e[j][b][k][q];
                    }
                  }
                }

                diffusion = 0;

                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  for (q = 0; q < goma_ctx->problem.vim; q++) {
                    diffusion +=
                        goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q] * d_grad_s_v_dv[p][q];
                  }
                }

                diffusion *= -det_J * wt * h3 *
                             (dt * goma_ctx->fem.mp->surface_tension * lsi_old->delta * scale);
                goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += diffusion;
              }
            }
          }

          var = MESH_DISPLACEMENT1;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            GOMA_EH(GOMA_ERROR, "Jacobian terms for hysing capillary wrt mesh not implemented");
          }
        }
      }
    }
  }

  return 0;
}
int assemble_cap_denner_diffusion(double dt, double scale) {
  int i, j, a, b, p, q, k, ii, ledof;
  int eqn, peqn, var, pvar;

  double wt, det_J, h3;

  double csf[DIM][DIM];

  double grad_s_v[DIM][DIM];
  //  double grad_s_phi_i_e_a[DIM][DIM];
  double d_grad_s_v_dv[DIM][DIM];
  double d_grad_s_v_dF[DIM][DIM];
  //  double d_grad_s_v_dn[DIM][DIM];

  double diffusion;

  eqn = R_MOMENTUM1;
  if (!goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn]) {
    return (0);
  }

  load_lsi(goma_ctx->fem.ls->Length_Scale);

  wt = goma_ctx->fem.fv->wt;
  h3 = goma_ctx->fem.fv->h3;
  if (goma_ctx->fem.ls->on_sharp_surf) /* sharp interface */
  {
    det_J = goma_ctx->fem.fv->sdet;
  } else /* diffuse interface */
  {
    det_J = goma_ctx->fem.bf[eqn]->detJ;
  }

  memset(csf, 0, sizeof(double) * DIM * DIM);

  for (p = 0; p < goma_ctx->problem.vim; p++) {
    for (q = 0; q < goma_ctx->problem.vim; q++) {
      grad_s_v[p][q] = goma_ctx->fem.fv->grad_v[p][q];
      for (k = 0; k < goma_ctx->problem.vim; k++) {
        grad_s_v[p][q] -= (goma_ctx->fem.lsi->normal[p] * goma_ctx->fem.lsi->normal[k]) *
                          goma_ctx->fem.fv->grad_v[k][q];
      }
    }
  }

  /* finite difference calculation of path dependencies for
     subelement integration
  */
  if (goma_ctx->fem.ls->CalcSurfDependencies) {
    GOMA_EH(GOMA_ERROR, "Calc surf dependencies not implemented");
  }

  /*
   * Residuals ____________________________________________________________________________
   */

  if (goma_ctx->fem.af->Assemble_Residual) {

    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          diffusion = 0;

          for (p = 0; p < goma_ctx->problem.vim; p++) {
            for (q = 0; q < goma_ctx->problem.vim; q++) {
              diffusion += goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q] * grad_s_v[p][q];
            }
          }

          diffusion *= -det_J * wt * h3 *
                       (dt * goma_ctx->fem.mp->surface_tension * goma_ctx->fem.lsi->delta * scale);

          goma_ctx->fem.lec->R[LEC_R_INDEX(peqn, ii)] += diffusion;
        }
      }
    }
  }

  /*
   * Yacobian terms...
   */

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          var = LS;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {

              /* grouping lsi->delta into this now */
              for (p = 0; p < goma_ctx->problem.vim; p++) {
                for (q = 0; q < goma_ctx->problem.vim; q++) {
                  d_grad_s_v_dF[p][q] =
                      goma_ctx->fem.lsi->d_delta_dF[j] * goma_ctx->fem.fv->grad_v[p][q] +
                      goma_ctx->fem.lsi->delta * goma_ctx->fem.fv->grad_v[p][q];
                  for (k = 0; k < goma_ctx->problem.vim; k++) {
                    d_grad_s_v_dF[p][q] -=
                        goma_ctx->fem.lsi->d_delta_dF[j] *
                        (goma_ctx->fem.lsi->normal[p] * goma_ctx->fem.lsi->normal[k]) *
                        goma_ctx->fem.fv->grad_v[k][q];
                    d_grad_s_v_dF[p][q] -=
                        goma_ctx->fem.lsi->delta *
                        (goma_ctx->fem.lsi->d_normal_dF[p][j] * goma_ctx->fem.lsi->normal[k]) *
                        goma_ctx->fem.fv->grad_v[k][q];
                    d_grad_s_v_dF[p][q] -=
                        goma_ctx->fem.lsi->delta *
                        (goma_ctx->fem.lsi->normal[p] * goma_ctx->fem.lsi->d_normal_dF[k][j]) *
                        goma_ctx->fem.fv->grad_v[k][q];
                  }
                }
              }

              diffusion = 0;

              for (p = 0; p < goma_ctx->problem.vim; p++) {
                for (q = 0; q < goma_ctx->problem.vim; q++) {
                  diffusion += goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q] * d_grad_s_v_dF[p][q];
                }
              }

              diffusion *= -det_J * wt * h3 * (dt * goma_ctx->fem.mp->surface_tension * scale);
              goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += diffusion;
            }
          }

          var = VELOCITY1;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (b = 0; b < goma_ctx->problem.vim; b++) {
              var = VELOCITY1 + b;
              pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

              for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {

                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  for (q = 0; q < goma_ctx->problem.vim; q++) {
                    d_grad_s_v_dv[p][q] = goma_ctx->fem.bf[var]->grad_phi_e[j][b][p][q];
                    for (k = 0; k < goma_ctx->problem.vim; k++) {
                      d_grad_s_v_dv[p][q] -= goma_ctx->fem.lsi->normal[p] *
                                             goma_ctx->fem.lsi->normal[k] *
                                             goma_ctx->fem.bf[var]->grad_phi_e[j][b][k][q];
                    }
                  }
                }

                diffusion = 0;

                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  for (q = 0; q < goma_ctx->problem.vim; q++) {
                    diffusion +=
                        goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q] * d_grad_s_v_dv[p][q];
                  }
                }

                diffusion *=
                    -det_J * wt * h3 *
                    (dt * goma_ctx->fem.mp->surface_tension * goma_ctx->fem.lsi->delta * scale);
                goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += diffusion;
              }
            }
          }

          var = MESH_DISPLACEMENT1;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            GOMA_EH(GOMA_ERROR,
                    "Jacobian terms for denner capillary diffusion wrt mesh not implemented");
          }
        }
      }
    }
  }

  return 0;
}
int assemble_cap_denner_diffusion_n(double dt, double scale) {
  int i, j, a, b, p, q, k, ii, ledof;
  int eqn, peqn, var, pvar;

  double wt, det_J, h3;

  double csf[DIM][DIM];

  double grad_s_v[DIM][DIM];
  //  double grad_s_phi_i_e_a[DIM][DIM];
  double d_grad_s_v_dv[DIM][DIM];
  double d_grad_s_v_dn[DIM][DIM];
  //  double d_grad_s_v_dn[DIM][DIM];

  double diffusion;

  eqn = R_MOMENTUM1;
  if (!goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn]) {
    return (0);
  }

  load_lsi(goma_ctx->fem.ls->Length_Scale);

  wt = goma_ctx->fem.fv->wt;
  h3 = goma_ctx->fem.fv->h3;
  if (goma_ctx->fem.ls->on_sharp_surf) /* sharp interface */
  {
    det_J = goma_ctx->fem.fv->sdet;
  } else /* diffuse interface */
  {
    det_J = goma_ctx->fem.bf[eqn]->detJ;
  }

  memset(csf, 0, sizeof(double) * DIM * DIM);

  for (p = 0; p < goma_ctx->problem.vim; p++) {
    for (q = 0; q < goma_ctx->problem.vim; q++) {
      grad_s_v[p][q] = goma_ctx->fem.fv->grad_v[p][q];
      for (k = 0; k < goma_ctx->problem.vim; k++) {
        grad_s_v[p][q] -=
            (goma_ctx->fem.fv->n[p] * goma_ctx->fem.fv->n[k]) * goma_ctx->fem.fv->grad_v[k][q];
      }
    }
  }

  /* finite difference calculation of path dependencies for
     subelement integration
  */
  if (goma_ctx->fem.ls->CalcSurfDependencies) {
    GOMA_EH(GOMA_ERROR, "Calc surf dependencies not implemented");
  }

  /*
   * Residuals ____________________________________________________________________________
   */

  if (goma_ctx->fem.af->Assemble_Residual) {

    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          diffusion = 0;

          for (p = 0; p < goma_ctx->problem.vim; p++) {
            for (q = 0; q < goma_ctx->problem.vim; q++) {
              diffusion += goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q] * grad_s_v[p][q];
            }
          }

          diffusion *= -det_J * wt * h3 *
                       (dt * goma_ctx->fem.mp->surface_tension * goma_ctx->fem.lsi->delta * scale);

          goma_ctx->fem.lec->R[LEC_R_INDEX(peqn, ii)] += diffusion;
        }
      }
    }
  }

  /*
   * Yacobian terms...
   */

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (a = 0; a < goma_ctx->problem.wim; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];

        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {

          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          var = LS;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
              diffusion = 0;

              for (p = 0; p < goma_ctx->problem.vim; p++) {
                for (q = 0; q < goma_ctx->problem.vim; q++) {
                  diffusion += goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q] * grad_s_v[p][q];
                }
              }

              diffusion *= goma_ctx->fem.lsi->d_delta_dF[j];

              diffusion *= -det_J * wt * h3 * (dt * goma_ctx->fem.mp->surface_tension * scale);
              goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += diffusion;
            }
          }

          var = VELOCITY1;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (b = 0; b < goma_ctx->problem.wim; b++) {
              var = VELOCITY1 + b;
              pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

              for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {

                for (p = 0; p < goma_ctx->problem.wim; p++) {
                  for (q = 0; q < goma_ctx->problem.wim; q++) {
                    d_grad_s_v_dv[p][q] = goma_ctx->fem.bf[var]->grad_phi_e[j][b][p][q];
                    for (k = 0; k < goma_ctx->problem.wim; k++) {
                      d_grad_s_v_dv[p][q] -= goma_ctx->fem.fv->n[p] * goma_ctx->fem.fv->n[k] *
                                             goma_ctx->fem.bf[var]->grad_phi_e[j][b][k][q];
                    }
                  }
                }

                diffusion = 0;

                for (p = 0; p < goma_ctx->problem.wim; p++) {
                  for (q = 0; q < goma_ctx->problem.wim; q++) {
                    diffusion +=
                        goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q] * d_grad_s_v_dv[p][q];
                  }
                }

                diffusion *=
                    -det_J * wt * h3 *
                    (dt * goma_ctx->fem.mp->surface_tension * goma_ctx->fem.lsi->delta * scale);
                goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += diffusion;
              }
            }
          }

          var = R_NORMAL1;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (b = 0; b < goma_ctx->problem.vim; b++) {
              var = R_NORMAL1 + b;
              pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

              for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  for (q = 0; q < goma_ctx->problem.vim; q++) {
                    d_grad_s_v_dn[p][q] = 0;
                    for (k = 0; k < goma_ctx->problem.vim; k++) {
                      d_grad_s_v_dn[p][q] -= goma_ctx->fem.bf[var]->phi[j] *
                                             goma_ctx->fem.fv->n[k] *
                                             goma_ctx->fem.fv->grad_v[k][q];
                      d_grad_s_v_dn[p][q] -= goma_ctx->fem.fv->n[p] *
                                             goma_ctx->fem.bf[var]->phi[j] *
                                             goma_ctx->fem.fv->grad_v[k][q];
                    }
                  }
                }

                diffusion = 0;

                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  for (q = 0; q < goma_ctx->problem.vim; q++) {
                    diffusion +=
                        goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q] * d_grad_s_v_dn[p][q];
                  }
                }

                diffusion *=
                    -det_J * wt * h3 *
                    (dt * goma_ctx->fem.mp->surface_tension * goma_ctx->fem.lsi->delta * scale);
                goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += diffusion;
              }
            }
          }

          var = MESH_DISPLACEMENT1;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            GOMA_EH(GOMA_ERROR,
                    "Jacobian terms for denner capillary diffusion wrt mesh not implemented");
          }
        }
      }
    }
  }

  return 0;
}