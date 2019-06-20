/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2014 Sandia Corporation.                                  *
*                                                                         *
* Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,  *
* the U.S. Government retains certain rights in this software.            *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
\************************************************************************/

/*
 *$Id: mm_fill_fill.c,v 5.5 2009-05-19 23:07:03 hkmoffa Exp $
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "std.h" /* This needs to be here. */

#include "el_elm.h"
#include "el_geom.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_mp_const.h"
#include "rf_allo.h"
#include "rf_bc.h"
#include "rf_bc_const.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_fill_const.h"
#include "rf_io.h"
#include "rf_io_const.h"
#include "rf_masks.h"
#include "rf_mp.h"
#include "rf_solver.h"
#include "rf_solver_const.h"
#include "rf_vars_const.h"

#include "mm_eh.h"

#include "mm_mp.h"
#include "mm_mp_structs.h"

#include "mm_shell_util.h"

#include "sl_util.h"

#define MM_FILL_FILL_CONSERVATIVE_C
#include "goma.h"
#include "mm_fill_fill_conservative.h"

double heaviside_smooth(double field, double *d_heaviside_dfield,
                        double alpha) {
  double heaviside;

  if (fabs(field) < alpha) {
    heaviside = 0.5 * (1. + field / alpha + sin(M_PIE * field / alpha) / M_PIE);
    if (d_heaviside_dfield != NULL) {
      *d_heaviside_dfield = 0.5 * (1 + cos(M_PIE * field / alpha)) / alpha;
    }
  } else if (field > alpha) {
    heaviside = 1;
    if (d_heaviside_dfield != NULL) {
      *d_heaviside_dfield = 0;
    }
  } else {
    heaviside = 0;
    if (d_heaviside_dfield != NULL) {
      *d_heaviside_dfield = 0;
    }
  }

  return heaviside;
}

double d_heaviside_smooth(double field, double alpha) {
  double d_heaviside;

  if (fabs(field) < alpha) {
    d_heaviside = 0.5 * (1. + cos(M_PIE * field / alpha)) / alpha;
  } else {
    d_heaviside = 0;
  }

  return d_heaviside;
}

int assemble_eikonal(int dim, double tt, double dt, PG_DATA *pg_data)

{
  dbl h3 = fv->h3; /* Volume element (scale factors). */
  dbl wt = fv->wt; /* Gauss point weight. */
  int status = 0;

  /*
   * Bail out fast if there's nothing to do...
   */
  int eqn = R_EIKONAL;

  if (!pd->e[pg->imtrx][eqn]) {
    return (status);
  }

  double detJ = bf[eqn]->detJ;

  double G[DIM][DIM];

  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      G[i][j] = 0;
      for (int k = 0; k < DIM; k++) {
        G[i][j] += bf[eqn]->B[k][i] * bf[eqn]->B[k][j];
      }
    }
  }

  double v_d_gv = 0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      v_d_gv += fv->grad_eikonal[i] * G[i][j] * fv->grad_eikonal[j];
    }
  }

  /**********************************************************************
   **********************************************************************
   ** Residuals
   **********************************************************************
   **********************************************************************/

  double eikonal_norm = 0;
  double eikonal_norm_old = 0;
  for (int i = 0; i < dim; i++) {
    eikonal_norm += fv->grad_eikonal[i] * fv->grad_eikonal[i];
    eikonal_norm_old += fv_old->grad_eikonal[i] * fv_old->grad_eikonal[i];
  }
  eikonal_norm = sqrt(eikonal_norm);
  eikonal_norm_old = sqrt(eikonal_norm_old);
  double inv_eikonal_norm = 1 / eikonal_norm;
  double inv_eikonal_norm_old = 1 / eikonal_norm_old;

  double d_eikonal_norm[MDE];
  for (int j = 0; j < ei[pg->imtrx]->dof[EIKONAL]; j++) {
    d_eikonal_norm[j] = 0;
    for (int i = 0; i < dim; i++) {
      d_eikonal_norm[j] +=
          fv->grad_eikonal[i] * bf[EIKONAL]->grad_phi[j][i] * inv_eikonal_norm;
    }
  }

  double h_elem = 0.;
  for (int a = 0; a < ei[pg->imtrx]->ielem_dim; a++)
    h_elem += pg_data->hsquared[a];
  /* This is the size of the element */
  h_elem = sqrt(h_elem / ((double)ei[pg->imtrx]->ielem_dim));
  double alpha = 2 * h_elem;
  double sign = 2 * heaviside_smooth(fv->F, NULL, alpha) - 1;
  double d_hs = d_heaviside_smooth(fv->F, alpha);
  double w[DIM];
  double w_old[DIM];
  double d_w[DIM][MDE];
  for (int i = 0; i < dim; i++) {
    w[i] = sign * fv->grad_eikonal[i] * inv_eikonal_norm;
    w_old[i] = sign * fv_old->grad_eikonal[i] * inv_eikonal_norm_old;

    for (int j = 0; j < ei[pg->imtrx]->dof[eqn]; j++) {
      d_w[i][j] = sign * (-fv->grad_eikonal[i] * d_eikonal_norm[j] *
                              inv_eikonal_norm * inv_eikonal_norm +
                          bf[eqn]->grad_phi[j][i] * inv_eikonal_norm);
    }
  }

  double aGa = 0;
  double d_aGa[MDE];
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      aGa += w[i] * G[i][j] * w[j];
    }
  }

  for (int k = 0; k < ei[pg->imtrx]->dof[eqn]; k++) {
    d_aGa[k] = 0;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        d_aGa[k] += d_w[i][k] * G[i][j] * w[j] + w[i] * G[i][j] * d_w[j][k];
      }
    }
  }

  double invdetJ = 1 / detJ;
  //    double delta = 1 / (sqrt(4.0 / (dt*dt) + aGa));
  double delta = 1 / (1e-8 + aGa);
  //    delta = h_elem;
  double d_delta[MDE];
  for (int j = 0; j < ei[pg->imtrx]->dof[eqn]; j++) {
    d_delta[j] = d_aGa[j] / 2 * pow(sqrt(1e-8 + aGa), 1.5);
    //      d_delta[j] = 0;
  }

  double residual_strong = fv_dot->eikonal;
  for (int a = 0; a < dim; a++) {
    residual_strong += w[a] * fv->grad_eikonal[a];
  }
  residual_strong -= sign;

  double d_residual_strong[MDE];

  for (int j = 0; j < ei[pg->imtrx]->dof[eqn]; j++) {
    double phi_j = bf[eqn]->phi[j];
    d_residual_strong[j] = phi_j * (1. + 2. * tt) / dt;

    for (int i = 0; i < dim; i++) {
      d_residual_strong[j] +=
          w[i] * bf[eqn]->grad_phi[j][i] + d_w[i][j] * fv->grad_eikonal[i];
    }
  }

  double gpGgp = 0;
  double gpGgp_old = 0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      gpGgp += fv->grad_eikonal[i] * G[i][j] * fv->grad_eikonal[j];
      gpGgp_old += fv_old->grad_eikonal[i] * G[i][j] * fv_old->grad_eikonal[j];
    }
  }
  double invsqrt_gpGgp = 1 / sqrt(gpGgp);
  double invsqrt_gpGgp_old = 1 / sqrt(gpGgp_old);

  double d_gpGgp[MDE];
  for (int k = 0; k < ei[pg->imtrx]->dof[eqn]; k++) {
    d_gpGgp[k] = 0;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        d_gpGgp[k] += bf[eqn]->grad_phi[k][i] * G[i][j] * fv->grad_eikonal[j] +
                      fv->grad_eikonal[i] * G[i][j] * bf[eqn]->grad_phi[k][j];
      }
    }
  }

#define LAG_KAPPA 1
  int lagkappa = LAG_KAPPA;

  double kappa = 0.25 * fabs(residual_strong) * invsqrt_gpGgp;
  if (lagkappa) {
    double residual_strong_old = fv_dot_old->eikonal;
    for (int a = 0; a < dim; a++) {
      residual_strong_old += w_old[a] * fv_old->grad_eikonal[a];
    }
    residual_strong_old -= sign;

    kappa = 0.25 * fabs(residual_strong) * invsqrt_gpGgp_old;
  }
  double d_kappa[MDE];

  for (int j = 0; j < ei[pg->imtrx]->dof[eqn]; j++) {
    if (lagkappa) {
      d_kappa[j] = 0;
    } else {
      d_kappa[j] =
          0.25 *
          ((residual_strong * d_residual_strong[j] / fabs(residual_strong)) *
               invsqrt_gpGgp +
           fabs(residual_strong) *
               (-0.5 * d_gpGgp[j] * (1 / (gpGgp)) * invsqrt_gpGgp));
    }
  }

  double kappa_bg = 0;
  if (fv->F > 2 * alpha) {
    kappa_bg = 0.01 * h_elem;
  }

  double lambda = 10 * h_elem;

  if (af->Assemble_Residual) {
    int peqn = upd->ep[pg->imtrx][eqn];

    double residual = fv_dot->eikonal;
#ifdef ADV_EIKONAL
    for (int a = 0; a < dim; a++) {
      residual += w[a] * fv->grad_eikonal[a];
    }
    residual -= sign;
#else
    residual += sign * eikonal_norm - sign;
#endif

    double penalty = lambda * d_hs * (fv->eikonal - fv->F);

    for (int i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      double wt_func = bf[eqn]->phi[i];
      //      for (int a = 0; a < dim; a++) {
      //        wt_func += delta * w[a] * bf[eqn]->grad_phi[i][a];
      //      }

      double shock_capturing = 0;
#define EIK_SHOCK_CAPTURING
#ifdef EIK_SHOCK_CAPTURING
      for (int a = 0; a < dim; a++) {
        shock_capturing +=
            (kappa + kappa_bg) * fv->grad_eikonal[a] * bf[eqn]->grad_phi[i][a];
        //        shock_capturing +=
        //            (kappa_bg)*fv->grad_eikonal[a] * bf[eqn]->grad_phi[i][a];
      }
#endif

      double supg = 0;
#define EIK_SUPG
#ifdef EIK_SUPG
      for (int a = 0; a < dim; a++) {
        supg += delta * residual_strong * w[a] * bf[eqn]->grad_phi[i][a];
        //        supg += w[a] * bf[eqn]->grad_phi[i][a];
      }
#endif

      lec->R[peqn][i] += (residual * wt_func + penalty * bf[eqn]->phi[i] +
                          shock_capturing + supg) *
                         wt * detJ * h3;
    }
  }

  /**********************************************************************
   **********************************************************************
   * Jacobian terms...
   **********************************************************************
   **********************************************************************/

  if (af->Assemble_Jacobian) {
    int peqn = upd->ep[pg->imtrx][eqn];
    for (int i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

      double residual = fv_dot->eikonal;
#ifdef ADV_EIKONAL
      for (int a = 0; a < dim; a++) {
        residual += w[a] * fv->grad_eikonal[a];
      }
      residual -= sign;
#else
      residual += sign * eikonal_norm - sign;
#endif

      double wt_func = bf[eqn]->phi[i];
      //      for (int a = 0; a < dim; a++) {
      //        wt_func += delta * w[a] * bf[eqn]->grad_phi[i][a];
      //      }

      /*************************************************************
       *
       * Derivatives of fill equation w.r.t. to FILL variable
       *
       *************************************************************/

      int var = EIKONAL;
      if (pd->v[pg->imtrx][var]) {
        int pvar = upd->vp[pg->imtrx][var];
        for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          double phi_j = bf[eqn]->phi[j];
          double d_wt_func = 0;
          //          for (int a = 0; a < dim; a++) {
          //            d_wt_func += (d_delta[j] * w[a] + delta * d_w[a][j]) *
          //                         bf[eqn]->grad_phi[i][a];
          //          }

          double d_residual = phi_j * (1. + 2. * tt) / dt;
#ifdef ADV_EIKONAL
          for (int a = 0; a < dim; a++) {
            d_residual += d_w[a][j] * fv->grad_eikonal[a] +
                          w[a] * bf[eqn]->grad_phi[j][a];
          }
#else
          d_residual += sign * d_eikonal_norm[j];
#endif
          double d_penalty = lambda * d_hs * (phi_j);

          double d_shock_capturing = 0;
#ifdef EIK_SHOCK_CAPTURING
          for (int a = 0; a < dim; a++) {
            d_shock_capturing +=
                (d_kappa[j]) * fv->grad_eikonal[a] * bf[eqn]->grad_phi[i][a];
            //            d_shock_capturing +=
            //                (kappa_bg)*bf[var]->grad_phi[j][a] *
            //                bf[eqn]->grad_phi[i][a];
            d_shock_capturing += (kappa + kappa_bg) * bf[var]->grad_phi[j][a] *
                                 bf[eqn]->grad_phi[i][a];
          }
#endif

          double d_supg = 0;
#ifdef EIK_SUPG
          for (int a = 0; a < dim; a++) {
            d_supg +=
                d_delta[j] * residual_strong * w[a] * bf[eqn]->grad_phi[i][a];
            d_supg +=
                delta * d_residual_strong[j] * w[a] * bf[eqn]->grad_phi[i][a];
            d_supg +=
                delta * residual_strong * d_w[a][j] * bf[eqn]->grad_phi[i][a];
            //            d_supg += d_delta[j]  * w[a] *
            //            bf[eqn]->grad_phi[i][a]; d_supg += delta *
            //            d_residual_strong[j] * w[a] * bf[eqn]->grad_phi[i][a];
            //            d_supg +=  d_w[a][j] * bf[eqn]->grad_phi[i][a];
          }
#endif

          lec->J[peqn][pvar][i][j] +=
              (residual * d_wt_func + d_residual * wt_func +
               d_penalty * bf[eqn]->phi[i] + d_shock_capturing + d_supg) *
              wt * h3 * detJ;

        } /* for: FILL DoFs */

      } /* if: FILL exisits */

    } /* for 'i': FILL DoFs */

  } /* if: af->Assemble_Jacobian */

  return (status);

} /* end of assemble_eikonal */

int assemble_heaviside_smooth(int dim, double tt, double dt, PG_DATA *pg_data)

{
  dbl h3 = fv->h3; /* Volume element (scale factors). */
  dbl wt = fv->wt; /* Gauss point weight. */
  int status = 0;

  /*
   * Bail out fast if there's nothing to do...
   */
  int eqn = R_HEAVISIDE_SMOOTH;

  if (!pd->e[pg->imtrx][eqn]) {
    return (status);
  }

  double detJ = bf[eqn]->detJ;

  int a;
  double vnorm = 0;
  double supg_tau = 0;

  for (a = 0; a < VIM; a++) {
    vnorm += fv->v[a] * fv->v[a];
  }
  vnorm = sqrt(vnorm);
  double h_elem = 0.;
  for (int a = 0; a < ei[pg->imtrx]->ielem_dim; a++)
    h_elem += pg_data->hsquared[a];
  /* This is the size of the element */
  h_elem = sqrt(h_elem / ((double)ei[pg->imtrx]->ielem_dim));

  double D = pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

  if (D == 0) {
    // if numerical diffusion is off use 1 for Peclet number
    D = 1e-6;
  }

  double Pek = 0.5 * vnorm * h_elem / D;

  double eta = Pek;
  if (Pek > 1) {
    eta = 1;
  }
  eta = 1;

  if (vnorm > 0) {
    supg_tau = 0.5 * h_elem * eta / vnorm;

  } else {
    supg_tau = 0;
  }
  /**********************************************************************
   **********************************************************************
   ** Residuals
   **********************************************************************
   **********************************************************************/

  if (af->Assemble_Residual) {
    int peqn = upd->ep[pg->imtrx][eqn];
    for (int i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      double wt_func = bf[eqn]->phi[i];
      for (int a = 0; a < dim; a++) {
        wt_func += supg_tau * fv->v[a] * bf[eqn]->grad_phi[i][a];
      }

      double mass = fv_dot->heaviside_smooth * wt_func;
      mass *= -pd->etm[pg->imtrx][eqn][(LOG2_MASS)];

      double advection = 0;
      for (int a = 0; a < dim; a++) {
        advection += fv->v[a] * fv->grad_heaviside_smooth[a];
      }

      advection *= wt_func;
      advection *= -pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

      double diffusion = 0;
      for (int a = 0; a < dim; a++) {
        diffusion += fv->grad_heaviside_smooth[a] * bf[eqn]->grad_phi[i][a];
      }
      diffusion *= -pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

      double source = 0;
      source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

      lec->R[peqn][i] +=
          (mass + advection + diffusion + source) * wt * detJ * h3;
    }
  }

  /**********************************************************************
   **********************************************************************
   * Jacobian terms...
   **********************************************************************
   **********************************************************************/

  if (af->Assemble_Jacobian) {
    int peqn = upd->ep[pg->imtrx][eqn];
    for (int i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      double phi_i = bf[eqn]->phi[i];

      double wt_func = bf[eqn]->phi[i];
      for (int a = 0; a < dim; a++) {
        wt_func += supg_tau * fv->v[a] * bf[eqn]->grad_phi[i][a];
      }

      /*************************************************************
       *
       * Derivatives of fill equation w.r.t. to FILL variable
       *
       *************************************************************/

      int var = HEAVISIDE_SMOOTH;
      if (pd->v[pg->imtrx][var]) {
        int pvar = upd->vp[pg->imtrx][var];
        for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          double phi_j = bf[eqn]->phi[j];

          double d_wt_func = 0;
          //		  for (int a = 0; a < dim; a++)
          //		    {
          //		      d_wt_func += delta * d_w_d_eikonal[a][j] *
          // bf[eqn]->grad_phi[i][a];
          //		    }

          double mass = phi_j * (1. + 2. * tt) / dt * wt_func;
          mass += fv_dot->eikonal * d_wt_func;
          mass *= -pd->etm[pg->imtrx][eqn][(LOG2_MASS)];

          double advection = 0;

          for (int a = 0; a < dim; a++) {
            advection += fv->v[a] * bf[var]->grad_phi[j][a];
            //		      advection_b += w_old[a] * bf[var]->grad_phi[j][a];
          }
          //		  advection *= d_wt_func;
          advection *= wt_func;

          advection *= -pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

          double diffusion = 0;
          for (int a = 0; a < dim; a++) {
            diffusion += bf[var]->grad_phi[j][a] * bf[eqn]->grad_phi[i][a];
          }

          diffusion *= -pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          double source = 0;
          source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

          lec->J[peqn][pvar][i][j] +=
              (mass + advection + diffusion + source) * wt * h3 * detJ;

        } /* for: FILL DoFs */

      } /* if: FILL exisits */

    } /* for 'i': FILL DoFs */

  } /* if: af->Assemble_Jacobian */

  return (status);

} /* end of assemble_heaviside_smooth */

int assemble_fill_prime(int dim, double tt, double dt, PG_DATA *pg_data)

{
  dbl h3 = fv->h3; /* Volume element (scale factors). */
  dbl wt = fv->wt; /* Gauss point weight. */
  int status = 0;

  /*
   * Bail out fast if there's nothing to do...
   */
  int eqn = R_FILL_PRIME;

  if (!pd->e[pg->imtrx][eqn]) {
    return (status);
  }

  double detJ = bf[eqn]->detJ;

  int a;
  double vnorm = 0;
  double supg_tau = 0;

  for (a = 0; a < VIM; a++) {
    vnorm += fv->v[a] * fv->v[a];
  }
  vnorm = sqrt(vnorm);
  double h_elem = 0.;
  for (int a = 0; a < ei[pg->imtrx]->ielem_dim; a++)
    h_elem += pg_data->hsquared[a];
  /* This is the size of the element */
  h_elem = sqrt(h_elem / ((double)ei[pg->imtrx]->ielem_dim));

  double D = pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

  if (D == 0) {
    // if numerical diffusion is off use 1 for Peclet number
    D = 1e-6;
  }

  double Pek = 0.5 * vnorm * h_elem / D;

  double eta = Pek;
  if (Pek > 1) {
    eta = 1;
  }
  eta = 1;

  if (vnorm > 0) {
    supg_tau = 0.5 * h_elem * eta / vnorm;

  } else {
    supg_tau = 0;
  }
  /**********************************************************************
   **********************************************************************
   ** Residuals
   **********************************************************************
   **********************************************************************/
  load_lsi(ls->Length_Scale);
  double alpha = lsi->alpha;
  double heaviside;
  double d_heaviside_dFprime;
  if (fabs(fv->eikonal) < alpha) {
    heaviside =
        heaviside_smooth(fv->F + fv->F_prime, &d_heaviside_dFprime, alpha);
  } else if (fv->eikonal > alpha) {
    heaviside = 1;
    d_heaviside_dFprime = 0;
  } else {
    heaviside = 0;
    d_heaviside_dFprime = 0;
  }
  double kappa = 100;

  if (af->Assemble_Residual) {
    int peqn = upd->ep[pg->imtrx][eqn];
    for (int i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      double wt_func = bf[eqn]->phi[i];
      //          for (int a = 0; a < dim; a++)
      //            {
      //              wt_func += supg_tau * fv->v[a] * bf[eqn]->grad_phi[i][a];
      //            }

      double mass = 0;
      //	  double mass = fv_dot->heaviside_sharp  * wt_func ;
      //	  mass *= -pd->etm[pg->imtrx][eqn][(LOG2_MASS)];

      double advection = 0;
      //	  for (int a = 0; a < dim; a++)
      //	    {
      //	      advection += fv->v[a] * fv->grad_heaviside_sharp[a];
      //	    }

      //	  advection *= wt_func;
      //	  advection *= -pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

      double diffusion = 0;
      for (int a = 0; a < dim; a++) {
        diffusion +=
            kappa * h_elem * fv->grad_F_prime[a] * bf[eqn]->grad_phi[i][a];
      }
      diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

      double source = heaviside - fv->heaviside_smooth;
      source *= wt_func;
      source *= -pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

      lec->R[peqn][i] +=
          (mass + advection + diffusion + source) * wt * detJ * h3;
    }
  }

  /**********************************************************************
   **********************************************************************
   * Jacobian terms...
   **********************************************************************
   **********************************************************************/

  if (af->Assemble_Jacobian) {
    int peqn = upd->ep[pg->imtrx][eqn];
    for (int i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      double phi_i = bf[eqn]->phi[i];

      double wt_func = bf[eqn]->phi[i];
      //	  for (int a = 0; a < dim; a++)
      //	    {
      //	      wt_func += supg_tau * fv->v[a] * bf[eqn]->grad_phi[i][a];
      //	    }

      /*************************************************************
       *
       * Derivatives of fill equation w.r.t. to FILL variable
       *
       *************************************************************/

      int var = R_FILL_PRIME;
      if (pd->v[pg->imtrx][var]) {
        int pvar = upd->vp[pg->imtrx][var];
        for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          double phi_j = bf[eqn]->phi[j];

          double d_wt_func = 0;
          //		  for (int a = 0; a < dim; a++)
          //		    {
          //		      d_wt_func += delta * d_w_d_eikonal[a][j] *
          // bf[eqn]->grad_phi[i][a];
          //		    }

          double mass = 0;

          //		  double mass = phi_j * (1. + 2. * tt) / dt * wt_func;
          //		  mass += fv_dot->eikonal * d_wt_func;
          //		  mass *= -pd->etm[pg->imtrx][eqn][(LOG2_MASS)];

          double advection = 0;
          //		  double advection_b = 0;
          //		  for (int a = 0; a < dim; a++)
          //		    {
          //		      advection += fv->v[a] * bf[var]->grad_phi[j][a];
          ////		      advection_b += w_old[a] * bf[var]->grad_phi[j][a];

          //		    }
          ////		  advection *= d_wt_func;
          //		  advection_b *= wt_func;
          //		  advection += advection_b;
          //		  advection *=
          //-pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

          double diffusion = 0;
          for (int a = 0; a < dim; a++) {
            diffusion += kappa * h_elem * bf[var]->grad_phi[j][a] *
                         bf[eqn]->grad_phi[i][a];
          }

          diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          double source = phi_j * d_heaviside_dFprime;
          source *= wt_func;

          source *= -pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

          lec->J[peqn][pvar][i][j] +=
              (mass + advection + diffusion + source) * wt * h3 * detJ;

        } /* for: FILL DoFs */

      } /* if: FILL exisits */

    } /* for 'i': FILL DoFs */

  } /* if: af->Assemble_Jacobian */

  return (status);

} /* end of assemble_fill_prime */

int assemble_heaviside_projection(int dim, double tt, double dt,
                                  PG_DATA *pg_data)

{
  dbl h3 = fv->h3; /* Volume element (scale factors). */
  dbl wt = fv->wt; /* Gauss point weight. */
  int status = 0;

  /*
   * Bail out fast if there's nothing to do...
   */
  int eqn = R_HEAVISIDE_PROJECTION;

  if (!pd->e[pg->imtrx][eqn]) {
    return (status);
  }

  double detJ = bf[eqn]->detJ;

  int a;
  double vnorm = 0;
  double supg_tau = 0;

  for (a = 0; a < VIM; a++) {
    vnorm += fv->v[a] * fv->v[a];
  }
  vnorm = sqrt(vnorm);
  double h_elem = 0.;
  for (int a = 0; a < ei[pg->imtrx]->ielem_dim; a++)
    h_elem += pg_data->hsquared[a];
  /* This is the size of the element */
  h_elem = sqrt(h_elem / ((double)ei[pg->imtrx]->ielem_dim));

  double D = pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

  if (D == 0) {
    // if numerical diffusion is off use 1 for Peclet number
    D = 1e-6;
  }

  double Pek = 0.5 * vnorm * h_elem / D;

  double eta = Pek;
  if (Pek > 1) {
    eta = 1;
  }
  eta = 1;

  if (vnorm > 0) {
    supg_tau = 0.5 * h_elem * eta / vnorm;

  } else {
    supg_tau = 0;
  }
  /**********************************************************************
   **********************************************************************
   ** Residuals
   **********************************************************************
   **********************************************************************/

  load_lsi(ls->Length_Scale);
  if (af->Assemble_Residual) {
    int peqn = upd->ep[pg->imtrx][eqn];
    for (int i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      double wt_func = bf[eqn]->phi[i];
      //          for (int a = 0; a < dim; a++)
      //            {
      //              wt_func += supg_tau * fv->v[a] * bf[eqn]->grad_phi[i][a];
      //            }

      double mass = 0;
      //	  double mass = fv_dot->heaviside_smooth  * wt_func ;
      //	  mass *= -pd->etm[pg->imtrx][eqn][(LOG2_MASS)];

      double advection = 0;
      //	  for (int a = 0; a < dim; a++)
      //	    {
      //	      advection += fv->v[a] * fv->grad_heaviside_smooth[a];
      //	    }

      //	  advection *= wt_func;
      //	  advection *= -pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

      double diffusion = 0;
      //	  for (int a = 0; a < dim; a++)
      //	    {
      //	      diffusion += fv->grad_heaviside_smooth[a]  *
      // bf[eqn]->grad_phi[i][a];
      //	    }
      //	  diffusion *= -pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

      double source =
          fv->heaviside_projection -
          heaviside_smooth(fv->eikonal + fv->F_prime, NULL, lsi->alpha);
      source *= wt_func;
      source *= -pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

      lec->R[peqn][i] +=
          (mass + advection + diffusion + source) * wt * detJ * h3;
    }
  }

  /**********************************************************************
   **********************************************************************
   * Jacobian terms...
   **********************************************************************
   **********************************************************************/

  if (af->Assemble_Jacobian) {
    int peqn = upd->ep[pg->imtrx][eqn];
    for (int i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      double phi_i = bf[eqn]->phi[i];

      double wt_func = bf[eqn]->phi[i];
      //	  for (int a = 0; a < dim; a++)
      //	    {
      //	      wt_func += supg_tau * fv->v[a] * bf[eqn]->grad_phi[i][a];
      //	    }

      /*************************************************************
       *
       * Derivatives of fill equation w.r.t. to FILL variable
       *
       *************************************************************/

      int var = R_HEAVISIDE_PROJECTION;
      if (pd->v[pg->imtrx][var]) {
        int pvar = upd->vp[pg->imtrx][var];
        for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          double phi_j = bf[eqn]->phi[j];

          double d_wt_func = 0;
          //		  for (int a = 0; a < dim; a++)
          //		    {
          //		      d_wt_func += delta * d_w_d_eikonal[a][j] *
          // bf[eqn]->grad_phi[i][a];
          //		    }

          double mass = 0;

          //		  double mass = phi_j * (1. + 2. * tt) / dt * wt_func;
          //		  mass += fv_dot->eikonal * d_wt_func;
          //		  mass *= -pd->etm[pg->imtrx][eqn][(LOG2_MASS)];

          double advection = 0;
          //		  double advection_b = 0;
          //		  for (int a = 0; a < dim; a++)
          //		    {
          //		      advection += fv->v[a] * bf[var]->grad_phi[j][a];
          ////		      advection_b += w_old[a] * bf[var]->grad_phi[j][a];

          //		    }
          ////		  advection *= d_wt_func;
          //		  advection_b *= wt_func;
          //		  advection += advection_b;
          //		  advection *=
          //-pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

          double diffusion = 0;
          //		  for (int a = 0; a < dim; a++)
          //		    {
          //		      diffusion += bf[var]->grad_phi[j][a] *
          // bf[eqn]->grad_phi[i][a];
          //		    }

          //		  diffusion *=
          //-pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          double source = phi_j;
          source *= wt_func;

          source *= -pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

          lec->J[peqn][pvar][i][j] +=
              (mass + advection + diffusion + source) * wt * h3 * detJ;

        } /* for: FILL DoFs */

      } /* if: FILL exisits */

    } /* for 'i': FILL DoFs */

  } /* if: af->Assemble_Jacobian */

  return (status);

} /* end of assemble_heaviside_projection */
