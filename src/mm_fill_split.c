/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2022 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

#include <stdio.h>

/* GOMA include files */
#define GOMA_MM_FILL_SPLIT_C
#include "density.h"
#include "el_elm.h"
#include "mm_as.h"
#include "mm_as_structs.h"
#include "mm_fill_ls.h"
#include "mm_fill_momentum.h"
#include "mm_fill_split.h"
#include "mm_fill_terms.h"
#include "mm_mp.h"
#include "mm_viscosity.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "std.h"

int assemble_ustar(dbl time_value, /* current time */
                   dbl tt,         /* parameter to vary time integration from
                                      explicit (tt = 1) to implicit (tt = 0)    */
                   dbl dt,         /* current time step size                    */
                   const PG_DATA *pg_data) {
#ifdef DEBUG_MOMENTUM_JAC
  int adx;
#endif
  //! WIM is the length of the velocity vector
  int i, j, a, b;
  int ledof, eqn, var, ii, peqn, pvar;
  int *pdv = goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx];

  int status = 0;

  eqn = USTAR;
  double d_area = goma_ctx->fem.fv->wt * goma_ctx->fem.bf[eqn]->detJ * goma_ctx->fem.fv->h3;
  double gamma[DIM][DIM];

  double rho;
  DENSITY_DEPENDENCE_STRUCT d_rho_struct;
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;

  double mu;
  VISCOSITY_DEPENDENCE_STRUCT d_mu_struct;
  VISCOSITY_DEPENDENCE_STRUCT *d_mu = &d_mu_struct;

  dbl f[DIM];                                  /* Body force. */
  MOMENTUM_SOURCE_DEPENDENCE_STRUCT df_struct; /* Body force dependence */
  MOMENTUM_SOURCE_DEPENDENCE_STRUCT *df = &df_struct;

  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      gamma[a][b] = goma_ctx->fem.fv->grad_v[a][b] + goma_ctx->fem.fv->grad_v[b][a];
    }
  }

  rho = density(d_rho, time_value);
  mu = viscosity(goma_ctx->fem.gn, gamma, d_mu);

  momentum_source_term(f, df, time_value);

  /*
   * Residuals_________________________________________________________________
   */

  if (goma_ctx->fem.af->Assemble_Residual) {
    /*
     * Assemble each component "a" of the momentum equation...
     */
    for (a = 0; a < WIM; a++) {
      eqn = USTAR + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {
        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];
        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {
          /*
           *  Here is where we figure out whether the row is to placed in
           *  the normal spot (e.g., ii = i), or whether a boundary condition
           *  require that the volumetric contribution be stuck in another
           *  ldof pertaining to the same variable type.
           */
          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];
          double resid = rho * (goma_ctx->fem.fv->v_star[a] - goma_ctx->fem.fv->v[a]) / dt;
          resid *= -goma_ctx->fem.bf[eqn]->phi[i] * d_area;

          double adv = 0;
          for (int p = 0; p < VIM; p++) {
            adv += rho * goma_ctx->fem.fv_old->v_star[p] * goma_ctx->fem.fv->grad_v_star[p][a];
          }

          // adv += 0.5 * fv->div_v * fv->v[a];

          adv *= -goma_ctx->fem.bf[eqn]->phi[i] * d_area;

          double pres = goma_ctx->fem.fv->grad_P_star[a] * goma_ctx->fem.bf[eqn]->phi[i];
          pres *= -d_area;

          double diff = 0;
          for (int p = 0; p < VIM; p++) {
            for (int q = 0; q < VIM; q++) {
              diff += mu * (goma_ctx->fem.fv->grad_v_star[p][q]) *
                      goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q];
            }
          }

          diff *= -d_area;

          double source = 0;

          source += f[a] * goma_ctx->fem.bf[eqn]->phi[i] * d_area;

          goma_ctx->fem.lec->R[LEC_R_INDEX(peqn, ii)] += resid + adv + pres + diff + source;
        } /*end if (active_dofs) */
      } /* end of for (i=0,ei[pg->imtrx]->dofs...) */
    }
  }

  /*
   * Jacobian terms...
   */

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (a = 0; a < WIM; a++) {
      eqn = USTAR + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {
        ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];
        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {
          /*
           *  Here is where we figure out whether the row is to placed in
           *  the normal spot (e.g., ii = i), or whether a boundary condition
           *  require that the volumetric contribution be stuck in another
           *  ldof pertaining to the same variable type.
           */
          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          for (b = 0; b < WIM; b++) {
            var = USTAR + b;
            if (pdv[var]) {
              pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];
              for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
                double resid = rho * goma_ctx->fem.bf[var]->phi[j] / dt;
                resid *= -delta(a, b) * goma_ctx->fem.bf[eqn]->phi[i] * d_area;

                double adv = 0; // bf[var]->phi[j] * fv->grad_v[b][a];
                for (int p = 0; p < VIM; p++) {
                  adv += rho * goma_ctx->fem.fv_old->v_star[p] *
                         goma_ctx->fem.bf[var]->grad_phi_e[j][b][p][a];
                }
                // double div_phi_j_e_b = 0.;
                // for (int p = 0; p < VIM; p++) {
                //   div_phi_j_e_b += bf[var]->grad_phi_e[j][b][p][p];
                // }

                // adv += 0.5 * (div_phi_j_e_b * fv->v[a] +  fv->div_v * bf[var]->phi[j]);

                adv *= -goma_ctx->fem.bf[eqn]->phi[i] * d_area;

                double diff = 0;
                for (int p = 0; p < VIM; p++) {
                  for (int q = 0; q < VIM; q++) {
                    diff += mu * (goma_ctx->fem.bf[VELOCITY1 + q]->grad_phi_e[j][b][p][q]) *
                            goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q];
                  }
                }

                diff *= -d_area;

                double source = 0;

                source += df->v[a][b][j] * goma_ctx->fem.bf[eqn]->phi[i] * -d_area;

                goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += resid + adv + diff + source;
              }
            }
          }
        }
      }
    }
  }
  return (status);
}

int assemble_pstar(dbl time_value, /* current time */
                   dbl tt,         /* parameter to vary time integration from
                                      explicit (tt = 1) to implicit (tt = 0)    */
                   dbl dt,         /* current time step size                    */
                   const PG_DATA *pg_data) {
  int a;

  int eqn, var;
  int peqn, pvar;

  int i, j;
  int status;

  dbl det_J;
  dbl h3;
  dbl wt;
  dbl d_area;

  /*
   * Galerkin weighting functions...
   */

  /*
   * Variables for Pressure Stabilization Petrov-Galerkin...
   */

  int *pdv = goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx];

  dbl mass;

  dbl rho = 0;
  DENSITY_DEPENDENCE_STRUCT d_rho_struct; /* density dependence */
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;

  status = 0;

  /*
   * Unpack variables from structures for local convenience...
   */

  eqn = R_PSTAR;
  peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];

  /*
   * Bail out fast if there's nothing to do...
   */

  if (!goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn]) {
    return (status);
  }

  wt = goma_ctx->fem.fv->wt;
  det_J = goma_ctx->fem.bf[eqn]->detJ; /* Really, ought to be mesh eqn. */
  h3 = goma_ctx->fem.fv->h3;           /* Differential volume element (scales). */

  d_area = wt * det_J * h3;
  /*
   * Get the deformation gradients and tensors if needed
   */

  rho = density(d_rho, time_value);

  if (goma_ctx->fem.af->Assemble_Residual) {
    for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

#if 1
      /* this is an optimization for xfem */

      if (xfem != NULL) {
        int xfem_active, extended_dof, base_interp, base_dof;

        xfem_dof_state(i, goma_ctx->fem.pd->i[goma_ctx->fem.pg->imtrx][eqn],
                       goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->ielem_shape, &xfem_active,
                       &extended_dof, &base_interp, &base_dof);

        if (extended_dof && !xfem_active)
          continue;
      }
#endif

      /*
       *  Mass Terms: drhodt terms (usually though problem dependent)
       */
      mass = 0;
      for (a = 0; a < WIM; a++) {
        mass += (goma_ctx->fem.fv->grad_P_star[a] - goma_ctx->fem.fv_old->grad_P_star[a]) / rho *
                goma_ctx->fem.bf[eqn]->grad_phi[i][a];
      }

      double tmp = 0;
      tmp += (1 / dt) * (goma_ctx->fem.fv->div_v_star * goma_ctx->fem.bf[eqn]->phi[i]);

      mass += tmp;
      mass *= -d_area;

      /*
       *  Add up the individual contributions and sum them into the local element
       *  contribution for the total continuity equation for the ith local unknown
       */
      goma_ctx->fem.lec->R[LEC_R_INDEX(peqn, i)] += mass;
    }
  }

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {

      var = PSTAR;
      if (pdv[var]) {
        pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          mass = 0;
          for (a = 0; a < WIM; a++) {
            mass += (goma_ctx->fem.bf[var]->grad_phi[j][a] / rho) *
                    goma_ctx->fem.bf[eqn]->grad_phi[i][a];
          }

          mass *= -d_area;

          goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += mass;
        }
      }
    }
  }

  return (status);
}

int assemble_continuity_segregated(dbl time_value, /* current time */
                                   dbl tt,         /* parameter to vary time integration from
                                                      explicit (tt = 1) to implicit (tt = 0)    */
                                   dbl dt,         /* current time step size                    */
                                   const PG_DATA *pg_data) {
  int i, j;
  int ledof, eqn, var, ii, peqn, pvar;
  int *pdv = goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx];

  int status = 0;

  eqn = R_PRESSURE;

  double d_area = goma_ctx->fem.fv->wt * goma_ctx->fem.bf[eqn]->detJ * goma_ctx->fem.fv->h3;

  /*
   * Residuals_________________________________________________________________
   */

  if (goma_ctx->fem.af->Assemble_Residual) {
    /*
     * Assemble each component "a" of the momentum equation...
     */
    peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];

    for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {
      ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];
      if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {
        /*
         *  Here is where we figure out whether the row is to placed in
         *  the normal spot (e.g., ii = i), or whether a boundary condition
         *  require that the volumetric contribution be stuck in another
         *  ldof pertaining to the same variable type.
         */
        ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

        double resid = goma_ctx->fem.fv->P - goma_ctx->fem.fv->P_star -
                       goma_ctx->fem.fv_old->P; // + 0.1 * fv->div_v;
        resid *= -goma_ctx->fem.bf[eqn]->phi[i] * d_area;
        /*lec->R[peqn][ii] += mass + advection + porous + diffusion + source;*/
        goma_ctx->fem.lec->R[LEC_R_INDEX(peqn, ii)] += resid;

#ifdef DEBUG_MOMENTUM_RES
        printf("R_m[%d][%d] += %10f %10f %10f %10f %10f\n", a, i, mass, advection, porous,
               diffusion, source);
#endif /* DEBUG_MOMENTUM_RES */

      } /*end if (active_dofs) */
    } /* end of for (i=0,ei[pg->imtrx]->dofs...) */
  }

  /*
   * Jacobian terms...
   */

  if (goma_ctx->fem.af->Assemble_Jacobian) {

    eqn = PRESSURE;
    peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];

    for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {
      ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

      ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];
      if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {
        /*
         *  Here is where we figure out whether the row is to placed in
         *  the normal spot (e.g., ii = i), or whether a boundary condition
         *  require that the volumetric contribution be stuck in another
         *  ldof pertaining to the same variable type.
         */
        ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

        var = PRESSURE;
        if (pdv[var]) {
          pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            double resid = goma_ctx->fem.bf[var]->phi[j] * goma_ctx->fem.bf[eqn]->phi[i];
            resid *= -d_area;
            goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += resid;
          }
        }
      }
    }
  }
  return (status);
}

int assemble_momentum_segregated(dbl time, /* current time */
                                 dbl tt,   /* parameter to vary time integration from
                                              explicit (tt = 1) to implicit (tt = 0) */
                                 dbl dt,   /* current time step size */
                                 const PG_DATA *pg_data) {
#ifdef DEBUG_MOMENTUM_JAC
  int adx;
#endif
  //! WIM is the length of the velocity vector
  int i, j, a, b;
  int ledof, eqn, var, ii, peqn, pvar;
  int *pdv = goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx];

  int status = 0;

  eqn = R_MOMENTUM1;
  double d_area = goma_ctx->fem.fv->wt * goma_ctx->fem.bf[eqn]->detJ * goma_ctx->fem.fv->h3;

  double rho;
  DENSITY_DEPENDENCE_STRUCT d_rho_struct;
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;

  rho = density(d_rho, time);

  /*
   * Residuals_________________________________________________________________
   */

  if (goma_ctx->fem.af->Assemble_Residual) {
    /*
     * Assemble each component "a" of the momentum equation...
     */
    for (a = 0; a < WIM; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {
        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];
        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {
          /*
           *  Here is where we figure out whether the row is to placed in
           *  the normal spot (e.g., ii = i), or whether a boundary condition
           *  require that the volumetric contribution be stuck in another
           *  ldof pertaining to the same variable type.
           */
          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          double resid =
              goma_ctx->fem.fv->v[a] - goma_ctx->fem.fv->v_star[a] +
              dt * (goma_ctx->fem.fv->grad_P_star[a] - goma_ctx->fem.fv_old->grad_P_star[a]) / rho;
          resid *= -goma_ctx->fem.bf[eqn]->phi[i] * d_area;

          /*lec->R[peqn][ii] += mass + advection + porous + diffusion + source;*/
          goma_ctx->fem.lec->R[LEC_R_INDEX(peqn, ii)] += resid;

#ifdef DEBUG_MOMENTUM_RES
          printf("R_m[%d][%d] += %10f %10f %10f %10f %10f\n", a, i, mass, advection, porous,
                 diffusion, source);
#endif /* DEBUG_MOMENTUM_RES */

        } /*end if (active_dofs) */
      } /* end of for (i=0,ei[pg->imtrx]->dofs...) */
    }
  }

  /*
   * Jacobian terms...
   */

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (a = 0; a < WIM; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];
      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {
        ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

        ledof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_ledof[eqn][i];
        if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->active_interp_ledof[ledof]) {
          /*
           *  Here is where we figure out whether the row is to placed in
           *  the normal spot (e.g., ii = i), or whether a boundary condition
           *  require that the volumetric contribution be stuck in another
           *  ldof pertaining to the same variable type.
           */
          ii = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          for (b = 0; b < WIM; b++) {
            var = VELOCITY1 + b;
            if (pdv[var]) {
              pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];

              for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {

                double resid = goma_ctx->fem.bf[var]->phi[j];
                resid *= -delta(a, b) * goma_ctx->fem.bf[eqn]->phi[i] * d_area;

                goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += resid;
              }
            }
          }
        }
      }
    }
  }
  return (status);
}
