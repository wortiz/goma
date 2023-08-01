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

#include "density.h"
#include "mm_fill_stabilization.h"
#include "rf_bc_const.h"
#include <stdio.h>

/* GOMA include files */
#define GOMA_MM_FILL_TURBULENT_C
#include "el_elm.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_fill_terms.h"
#include "mm_fill_turbulent.h"
#include "mm_fill_util.h"
#include "mm_mp.h"
#include "mm_viscosity.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "std.h"

/*  _______________________________________________________________________  */

static int calc_vort_mag(dbl *vort_mag,
                         dbl omega[DIM][DIM],
                         dbl d_vort_mag_dv[DIM][MDE],
                         dbl d_vort_mag_dmesh[DIM][MDE]) {
  int mdofs = 0;
  int p, q, a, b;
  int vdofs, i, j, v;

  dbl grad_phi_e_omega[MDE][DIM][DIM][DIM]; /* transpose of grad(phi_i ea) tensor
                                             + grad(phi_i ea) tensor */
  dbl d_omega_dmesh[DIM][DIM][DIM][MDE];    /* d/dmesh(grad_v)T */

  int status = 1;

  /* Zero out sensitivities */

  if (d_vort_mag_dv != NULL)
    memset(d_vort_mag_dv, 0, sizeof(dbl) * DIM * MDE);
  if (d_vort_mag_dmesh != NULL)
    memset(d_vort_mag_dmesh, 0, sizeof(dbl) * DIM * MDE);

  *vort_mag = 0.;
  /* get gamma_dot invariant for viscosity calculations */
  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      *vort_mag += omega[a][b] * omega[a][b];
    }
  }

  *vort_mag = sqrt(0.5 * (*vort_mag));

  /* get stuff for Jacobian entries */
  v = VELOCITY1;
  vdofs = ei[pg->imtrx]->dof[v];

  if (d_vort_mag_dmesh != NULL || d_vort_mag_dv != NULL) {
    if (pd->v[pg->imtrx][R_MESH1]) {
      mdofs = ei[pg->imtrx]->dof[R_MESH1];
    }

    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        for (a = 0; a < VIM; a++) {
          for (i = 0; i < vdofs; i++) {
            grad_phi_e_omega[i][a][p][q] =
                bf[v]->grad_phi_e[i][a][p][q] - bf[v]->grad_phi_e[i][a][q][p];
          }
        }
      }
    }
  }

  /*
   * d( gamma_dot )/dmesh
   */

  if (pd->v[pg->imtrx][R_MESH1] && d_vort_mag_dmesh != NULL) {

    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        for (b = 0; b < VIM; b++) {
          for (j = 0; j < mdofs; j++) {

            d_omega_dmesh[p][q][b][j] =
                fv->d_grad_v_dmesh[p][q][b][j] - fv->d_grad_v_dmesh[q][p][b][j];
          }
        }
      }
    }

    /*
     * d( gammadot )/dmesh
     */

    if (*vort_mag != 0.) {
      for (b = 0; b < VIM; b++) {
        for (j = 0; j < mdofs; j++) {
          d_vort_mag_dmesh[b][j] = 0.;
          for (p = 0; p < VIM; p++) {
            for (q = 0; q < VIM; q++) {
              d_vort_mag_dmesh[b][j] += 0.5 * d_omega_dmesh[p][q][b][j] * omega[p][q] / *vort_mag;
            }
          }
        }
      }
    }
  }

  /*
   * d( gammadot )/dv
   */

  if (*vort_mag != 0. && d_vort_mag_dv != NULL) {
    for (a = 0; a < VIM; a++) {
      for (i = 0; i < vdofs; i++) {
        d_vort_mag_dv[a][i] = 0.;
        for (p = 0; p < VIM; p++) {
          for (q = 0; q < VIM; q++) {
            d_vort_mag_dv[a][i] += 0.5 * grad_phi_e_omega[i][a][p][q] * omega[p][q] / *vort_mag;
          }
        }
      }
    }
  }
  return (status);
}

/* assemble_spalart_allmaras -- assemble terms (Residual & Jacobian) for conservation
 *                              of eddy viscosity for Spalart Allmaras turbulent flow model
 *
 *  Kessels, P. C. J. "Finite element discretization of the Spalart-Allmaras
 *  turbulence model." (2016).
 *
 *  Spalart, Philippe, and Steven Allmaras. "A one-equation turbulence model for
 *  aerodynamic flows." 30th aerospace sciences meeting and exhibit. 1992.
 *
 * in:
 *      time value
 *      theta (time stepping parameter, 0 for BE, 0.5 for CN)
 *      time step size
 *      Streamline Upwind Petrov Galerkin (PG) data structure
 *
 * out:
 *      lec -- gets loaded up with local contributions to resid, Jacobian
 *
 * Created:     August 2022 kristianto.tjiptowidjojo@averydennison.com
 * Modified:    June 2023 Weston Ortiz
 *
 */
int assemble_spalart_allmaras(dbl time_value, /* current time */
                              dbl tt,         /* parameter to vary time integration from
                                                 explicit (tt = 1) to implicit (tt = 0)    */
                              dbl dt,         /* current time step size                    */
                              const PG_DATA *pg_data) {

  //! WIM is the length of the velocity vector
  int i, j, a, b;
  int eqn, var, peqn, pvar;
  int *pdv = pd->v[pg->imtrx];

  dbl mass, adv, src, diff;
  dbl src_1, src_2, diff_1, diff_2;

  int status = 0;

  eqn = EDDY_NU;
  dbl d_area = fv->wt * bf[eqn]->detJ * fv->h3;

  /* Get Eddy viscosity at Gauss point */
  dbl mu_e = fv->eddy_nu;

  int negative_sa = false;
  // Previous workaround, see comment for negative_Se below
  //
  //   int transient_run = FALSE;
  //   if (pd->TimeIntegration != STEADY) {
  //     transient_run = true;
  //   }
  //   // Use old values for equation switching for transient runs
  //   // Seems to work reasonably well.
  //   if (transient_run && (fv_old->eddy_nu < 0)) {
  //     negative_sa = true;
  //   } else if (!transient_run && (mu_e < 0)) {
  //     // Kris thinks it might work with switching equations in steady state
  //     negative_sa = true;
  //   }
  if (mu_e < 0) {
    negative_sa = true;
  }

  /* Get fluid viscosity */
  dbl mu_newt = mp->viscosity;

  /* Rate of rotation tensor  */
  dbl omega[DIM][DIM];
  dbl omega_old[DIM][DIM];
  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      omega[a][b] = (fv->grad_v[a][b] - fv->grad_v[b][a]);
      omega_old[a][b] = (fv_old->grad_v[a][b] - fv_old->grad_v[b][a]);
    }
  }

  /* Vorticity */
  dbl S = 0.0;
  dbl S_old = 0;
  dbl dS_dvelo[DIM][MDE];
  dbl dS_dmesh[DIM][MDE];
  calc_vort_mag(&S, omega, dS_dvelo, dS_dmesh);
  calc_vort_mag(&S_old, omega_old, NULL, NULL);

  dbl d = fv->wall_distance;
  /* Get distance from nearest wall */
  if (d < 1.0e-6)
    d = 1.0e-6;

  /* Model coefficients (constants) */
  dbl cb1 = 0.1355;
  dbl cb2 = 0.622;
  dbl cv1 = 7.1;
  dbl cv2 = 0.7;
  dbl cv3 = 0.9;
  dbl sigma = (2.0 / 3.0);
  dbl cw2 = 0.3;
  dbl cw3 = 2.0;
  dbl cn1 = 16;
  dbl kappa = 0.41;
  dbl cw1 = (cb1 / kappa / kappa) + (1.0 + cb2) / sigma;

  /* More model coefficients (depends on mu_e) */
  dbl chi = mu_e / mu_newt;
  dbl fv1 = pow(chi, 3) / (pow(chi, 3) + pow(cv1, 3));
  dbl fv2 = 1.0 - (chi) / (1.0 + chi * fv1);
  dbl fn = 1.0;
  if (negative_sa) {
    fn = (cn1 + pow(chi, 3.0)) / (cn1 - pow(chi, 3));
  }
  dbl Sbar = (mu_e * fv2) / (kappa * kappa * d * d);
  int negative_Se = false;
  // I tried to use the old values for equation switching for transient runs but
  // I end up getting floating point errors. because of Se going very far below
  // zero I'm trying to use current values instead and hope that Newton's method
  // will converge with the switching equations
  // previously:
  // . dbl Sbar_old = (fv_old->eddy_nu * fv2) / (kappa * kappa * d * d);
  //   if (transient_run && (Sbar_old < -cv2 * S_old)) {
  //     negative_Se = true;
  //   } else if (!transient_run && (Sbar < -cv2 * S)) {
  // .   negative_Se = true;
  //   }
  if (Sbar < -cv2 * S) {
    negative_Se = true;
  }
  dbl S_e = S + Sbar;
  if (negative_Se) {
    S_e = S + S * (cv2 * cv2 * S + cv3 * Sbar) / ((cv3 - 2 * cv2) * S - Sbar);
  }
  dbl r_max = 10.0;
  dbl r = 0.0;
  if (fabs(S_e) > 1.0e-6) {
    r = mu_e / (kappa * kappa * d * d * S_e);
  } else {
    r = r_max;
  }
  if (r >= r_max) {
    r = r_max;
  }
  // Arbitrary limit to avoid floating point errors should only hit this when
  // S_e is very small and either mu_e or S_e are negative.  Which means we are
  // already trying to alleviate the issue.
  if (r < -100) {
    r = -100;
  }
  dbl g = r + cw2 * (pow(r, 6) - r);
  dbl fw_inside = (1.0 + pow(cw3, 6)) / (pow(g, 6) + pow(cw3, 6));
  dbl fw = g * pow(fw_inside, (1.0 / 6.0));

  /* Model coefficients sensitivity w.r.t. mu_e*/
  dbl dchi_dmu_e = 1.0 / mu_newt;
  dbl dfv1_dchi = 3 * pow(cv1, 3) * pow(chi, 2) / (pow((pow(chi, 3) + pow(cv1, 3)), 2));
  dbl dfv1_dmu_e = dfv1_dchi * dchi_dmu_e;
  dbl dfv2_dmu_e = (chi * chi * dfv1_dmu_e - dchi_dmu_e) / (pow(1 + fv1 * chi, 2));
  dbl dfn_dmu_e = 0.0;
  if (negative_sa) {
    dfn_dmu_e = 6.0 * (cn1 * chi * chi * dchi_dmu_e) / (pow(cn1 - pow(chi, 3.0), 2.0));
  }
  dbl dSbar_dmu_e = (fv2 + mu_e * dfv2_dmu_e) / (kappa * kappa * d * d);
  dbl dS_e_dmu_e = dSbar_dmu_e;
  if (negative_Se) {
    dS_e_dmu_e =
        (pow(cv2 - cv3, 2.0) * S * S * dSbar_dmu_e) / (pow(2 * cv2 * S - cv3 * S + Sbar, 2.0));
  }
  dbl dr_dmu_e = 0.0;
  if (r < r_max) {
    dr_dmu_e = 1.0 / (kappa * kappa * d * d * S_e) - (mu_e * kappa * kappa * d * d * dS_e_dmu_e) /
                                                         (kappa * kappa * d * d * S_e) /
                                                         (kappa * kappa * d * d * S_e);
  }
  dbl dg_dr = 1.0 + cw2 * (6.0 * pow(r, 5) - 1.0);
  dbl dg_dmu_e = dg_dr * dr_dmu_e;
  dbl dfw_inside_dg = -6. * (pow(g, 5) * (pow(cw3, 6) + 1)) / pow((pow(cw3, 6) + pow(g, 6)), 2);
  dbl dfw_dg = pow(fw_inside, (1.0 / 6.0)) +
               g * dfw_inside_dg * (1.0 / 6.0) * pow(fw_inside, (1.0 / 6.0) - 1.0);
  dbl dfw_dmu_e = dfw_dg * dg_dmu_e;

  /* Model coefficients sensitivity w.r.t. velocity*/
  dbl dr_dS = 0;
  dbl dg_dS = 0;
  dbl dfw_dS = 0;
  if (fabs(S_e) > 1.0e-6) {
    dr_dS = -(mu_e * kappa * kappa * d * d) / (kappa * kappa * d * d * S_e) /
            (kappa * kappa * d * d * S_e);
  }
  if (r == r_max)
    dr_dS = 0.0;
  if (r < -100)
    dr_dS = 0.0;
  dg_dS = dg_dr * dr_dS;
  dfw_dS = dfw_dg * dg_dS;
  dbl dS_e_dS = 1.0;
  if (negative_Se) {
    S_e = S + S * (cv2 * cv2 * S + cv3 * Sbar) / ((cv3 - 2 * cv2) * S - Sbar);
    dS_e_dS += (cv2 * cv2 * S) / ((cv3 - 2 * cv2) * S - Sbar) +
               (cv2 * cv2 * S + cv3 * Sbar) / ((cv3 - 2 * cv2) * S - Sbar) -
               (cv3 - 2 * cv2) * S * (cv2 * cv2 * S + cv3 * Sbar) /
                   (pow(((cv3 - 2 * cv2) * S - Sbar), 2.0));
  }

  dbl supg = 1.;
  SUPG_terms supg_terms;
  if (mp->SAwt_funcModel == GALERKIN) {
    supg = 0.;
  } else if (mp->SAwt_funcModel == SUPG || mp->SAwt_funcModel == SUPG_GP ||
             mp->SAwt_funcModel == SUPG_SHAKIB) {
    supg = mp->SAwt_func;
    supg_tau_shakib(&supg_terms, pd->Num_Dim, dt, mu_newt, EDDY_NU);
  }

  /*
   * Residuals_________________________________________________________________
   */

  if (af->Assemble_Residual) {
    /*
     * Assemble residual for eddy viscosity
     */
    eqn = EDDY_NU;
    peqn = upd->ep[pg->imtrx][eqn];

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      dbl wt_func = bf[eqn]->phi[i];

      if (supg > 0) {
        if (supg != 0.0) {
          for (int p = 0; p < VIM; p++) {
            wt_func += supg * supg_terms.supg_tau * fv->v[p] * bf[eqn]->grad_phi[i][p];
          }
        }
      }

      /* Assemble mass term */
      mass = 0.0;
      if (pd->TimeIntegration != STEADY) {
        if (pd->e[pg->imtrx][eqn] & T_MASS) {
          mass += fv_dot->eddy_nu * wt_func * d_area;
          mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
        }
      }

      /* Assemble advection term */
      adv = 0;
      for (int p = 0; p < VIM; p++) {
        adv += fv->v[p] * fv->grad_eddy_nu[p];
      }
      adv *= wt_func * d_area;
      adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

      dbl neg_c = 1.0;
      if (negative_sa) {
        neg_c = -1.0;
      }

      /* Assemble source terms */
      src_1 = cb1 * S_e * mu_e;
      src_2 = neg_c * cw1 * fw * (mu_e * mu_e) / (d * d);
      src = -src_1 + src_2;
      src *= wt_func * d_area;
      src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

      /* Assemble diffusion terms */
      diff_1 = 0.0;
      diff_2 = 0.0;
      for (int p = 0; p < VIM; p++) {
        diff_1 += bf[eqn]->grad_phi[i][p] * (mu_newt + mu_e * fn) * fv->grad_eddy_nu[p];
        diff_2 += wt_func * cb2 * fv->grad_eddy_nu[p] * fv->grad_eddy_nu[p];
      }
      diff = (1.0 / sigma) * (diff_1 - diff_2);
      diff *= d_area;
      diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

      lec->R[LEC_R_INDEX(peqn, i)] += mass + adv + src + diff;
    } /* end of for (i=0,ei[pg->imtrx]->dofs...) */
  }   /* end of if assemble residual */

  /*
   * Jacobian terms...
   */

  if (af->Assemble_Jacobian) {
    eqn = EDDY_NU;
    peqn = upd->ep[pg->imtrx][eqn];

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

      dbl wt_func = bf[eqn]->phi[i];

      dbl r_mass = 0;
      dbl r_adv = 0;
      dbl r_src = 0;
      dbl r_diff = 0;
      if (supg > 0) {
        if (supg != 0.0) {
          for (int p = 0; p < VIM; p++) {
            wt_func += supg * supg_terms.supg_tau * fv->v[p] * bf[eqn]->grad_phi[i][p];
          }
        }

        /* Assemble mass term */
        r_mass = 0.0;
        if (pd->TimeIntegration != STEADY) {
          if (pd->e[pg->imtrx][eqn] & T_MASS) {
            r_mass += fv_dot->eddy_nu;
            mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
          }
        }

        /* Assemble advection term */
        r_adv = 0;
        for (int p = 0; p < VIM; p++) {
          r_adv += fv->v[p] * fv->grad_eddy_nu[p];
        }
        r_adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

        dbl neg_c = 1.0;
        if (negative_sa) {
          neg_c = -1.0;
        }

        /* Assemble source terms */
        src_1 = cb1 * S_e * mu_e;
        src_2 = neg_c * cw1 * fw * (mu_e * mu_e) / (d * d);
        r_src = -src_1 + src_2;
        r_src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

        /* Assemble diffusion terms */
        diff_1 = 0.0;
        diff_2 = 0.0;
        for (int p = 0; p < VIM; p++) {
          diff_2 += wt_func * cb2 * fv->grad_eddy_nu[p] * fv->grad_eddy_nu[p];
        }
        r_diff = (1.0 / sigma) * (diff_1 - diff_2);
        r_diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
      }

      /* Sensitivity w.r.t. eddy viscosity */
      var = EDDY_NU;
      if (pdv[var]) {
        pvar = upd->vp[pg->imtrx][var];

        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

          /* Assemble mass term */
          mass = 0.0;
          if (pd->TimeIntegration != STEADY) {
            if (pd->e[pg->imtrx][eqn] & T_MASS) {
              mass += (1.0 + 2.0 * tt) / dt * bf[eqn]->phi[j] * wt_func * d_area;
              mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
            }
          }

          /* Assemble advection term */
          adv = 0.0;
          for (int p = 0; p < VIM; p++) {
            adv += fv->v[p] * bf[eqn]->grad_phi[j][p];
          }
          adv *= wt_func * d_area;
          adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

          dbl neg_c = 1.0;
          if (negative_sa) {
            neg_c = -1.0;
          }
          /* Assemble source term */
          src_1 = cb1 * (dS_e_dmu_e * bf[eqn]->phi[j] * mu_e + S_e * bf[eqn]->phi[j]);
          src_2 = neg_c *
                  ((cw1 / d / d) * bf[var]->phi[j] * (dfw_dmu_e * mu_e * mu_e + fw * 2.0 * mu_e));
          src = -src_1 + src_2;
          src *= wt_func * d_area;
          src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

          /* Assemble diffusion terms */
          diff_1 = 0.0;
          diff_2 = 0.0;
          for (int p = 0; p < VIM; p++) {
            diff_1 += bf[eqn]->grad_phi[i][p] *
                      ((fn + mu_e * dfn_dmu_e) * bf[eqn]->phi[j] * fv->grad_eddy_nu[p] +
                       (mu_newt + mu_e * fn) * bf[eqn]->grad_phi[j][p]);
            diff_2 += wt_func * cb2 * 2.0 * fv->grad_eddy_nu[p] * bf[var]->grad_phi[j][p];
          }
          diff = (1.0 / sigma) * (diff_1 - diff_2);
          diff *= d_area;
          diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += mass + adv + src + diff;
        } /* End of loop over j */
      }   /* End of if the variable is active */

      /* Sensitivity w.r.t. velocity */
      for (b = 0; b < VIM; b++) {
        var = VELOCITY1 + b;
        if (pdv[var]) {
          pvar = upd->vp[pg->imtrx][var];

          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            if (supg > 0) {
              dbl d_wt_func =
                  supg * bf[var]->phi[j] * supg_terms.supg_tau * bf[eqn]->grad_phi[i][b];
              for (int p = 0; p < VIM; p++) {
                d_wt_func +=
                    supg * fv->v[p] * supg_terms.d_supg_tau_dv[p][j] * bf[eqn]->grad_phi[i][p];
              }
              lec->J[LEC_J_INDEX(peqn, pvar, i, j)] +=
                  (r_mass + r_adv + r_src + r_diff) * d_area * d_wt_func;
            }

            /* Assemble advection term */
            adv = bf[var]->phi[j] * fv->grad_eddy_nu[b];
            adv *= wt_func * d_area;
            adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

            dbl neg_c = 1.0;
            if (negative_sa) {
              neg_c = -1.0;
            }
            /* Assemble source term */
            src_1 = cb1 * dS_e_dS * dS_dvelo[b][j] * mu_e;
            src_2 = neg_c * cw1 * dfw_dS * dS_dvelo[b][j] * (mu_e / d) * (mu_e / d);
            src = -src_1 + src_2;
            src *= wt_func * d_area;
            src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

            lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += adv + src;

          } /* End of loop over j */
        }   /* End of if the variale is active */
      }     /* End of loop over velocity components */

      /* Sensistivity w.r.t. mesh */
      for (b = 0; b < pd->Num_Dim; b++) {
        var = MESH_DISPLACEMENT1 + b;
        if (pdv[var]) {
          pvar = upd->vp[pg->imtrx][var];
          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            dbl d_area_dmesh = +fv->wt * bf[eqn]->d_det_J_dm[b][j] * fv->h3 +
                               fv->wt * bf[eqn]->detJ * fv->dh3dmesh[b][j];
            if (supg > 0) {
              dbl d_wt_func = 0;
              for (int p = 0; p < VIM; p++) {
                d_wt_func +=
                    supg * fv->v[p] * supg_terms.d_supg_tau_dX[p][j] * bf[eqn]->grad_phi[i][p];
                d_wt_func +=
                    supg * fv->v[p] * supg_terms.supg_tau * bf[eqn]->d_grad_phi_dmesh[i][p][b][j];
              }
              lec->J[LEC_J_INDEX(peqn, pvar, i, j)] +=
                  (r_mass + r_adv + r_src + r_diff) * d_area * d_wt_func +
                  (r_mass + r_adv + r_src + r_diff) * d_area_dmesh * wt_func;
            }

            /* Assemble mass term */
            mass = 0.0;
            if (pd->TimeIntegration != STEADY) {
              if (pd->e[pg->imtrx][eqn] & T_MASS) {
                mass += fv_dot->eddy_nu * wt_func * d_area_dmesh;
                mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
              }
            }

            /* Assemble advection term */
            adv = 0;
            for (int p = 0; p < VIM; p++) {
              adv += fv->v[p] * fv->grad_eddy_nu[p];
            }
            dbl d_adv = 0;
            for (int p = 0; p < VIM; p++) {
              d_adv += fv->v[p] * fv->d_grad_eddy_nu_dmesh[p][b][j];
            }
            adv *= wt_func * d_area_dmesh;
            adv += d_adv * wt_func * d_area;
            adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

            dbl neg_c = 1.0;
            if (negative_sa) {
              neg_c = -1.0;
            }

            /* Assemble source terms */
            dbl d_src_1 = cb1 * dS_e_dS * dS_dmesh[b][j] * mu_e;
            dbl d_src_2 = neg_c * cw1 * dfw_dS * dS_dmesh[b][j] * (mu_e * mu_e) / (d * d);
            src_1 = cb1 * S_e * mu_e;
            src_2 = neg_c * cw1 * fw * (mu_e * mu_e) / (d * d);
            src = -src_1 + src_2;
            dbl d_src = -d_src_1 + d_src_2;
            src *= wt_func * d_area_dmesh;
            src += d_src * wt_func * d_area;
            src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

            /* Assemble diffusion terms */
            dbl d_diff_1 = 0;
            dbl d_diff_2 = 0;
            diff_1 = 0.0;
            diff_2 = 0.0;
            for (int p = 0; p < VIM; p++) {
              diff_1 += bf[eqn]->grad_phi[i][p] * (mu_newt + mu_e * fn) * fv->grad_eddy_nu[p];
              diff_2 += wt_func * cb2 * fv->grad_eddy_nu[p] * fv->grad_eddy_nu[p];

              d_diff_1 += bf[eqn]->d_grad_phi_dmesh[i][p][b][j] * (mu_newt + mu_e * fn) *
                          fv->grad_eddy_nu[p];
              d_diff_1 += bf[eqn]->grad_phi[i][p] * (mu_newt + mu_e * fn) *
                          fv->d_grad_eddy_nu_dmesh[p][b][j];
              d_diff_2 +=
                  wt_func * cb2 * 2.0 * fv->d_grad_eddy_nu_dmesh[p][b][j] * fv->grad_eddy_nu[p];
            }
            diff = (1.0 / sigma) * (diff_1 - diff_2);
            dbl d_diff = (1.0 / sigma) * (d_diff_1 - d_diff_2);
            diff *= d_area_dmesh;
            diff += d_diff * d_area;
            diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

            lec->R[LEC_R_INDEX(peqn, i)] += mass + adv + src + diff;
          } /* End of loop over j */
        }
      }

    } /* End of loop over i */
  }   /* End of if assemble Jacobian */
  return (status);
}

typedef struct {
  dbl dk[MDE];
  dbl domega[MDE];
} blending_dependence_struct;

static void calc_blending_functions(dbl rho,
                                    DENSITY_DEPENDENCE_STRUCT *d_rho,
                                    dbl *F1,
                                    dbl *F2,
                                    blending_dependence_struct *d_F1,
                                    blending_dependence_struct *d_F2) {
  dbl d = fv->wall_distance;
  dbl k = fv_old->turb_k;
  dbl omega = fv_old->turb_omega;
  dbl nu = mp->viscosity / rho;
  dbl beta_star = 0.09;
  dbl sigma_omega2 = 0.856;
  dbl ddkdomega = 0;
  for (int i = 0; i < pd->Num_Dim; i++) {
    ddkdomega += fv_old->grad_turb_k[i] * fv_old->grad_turb_omega[i];
  }
  dbl CD_komega = fmax(2 * rho * sigma_omega2 * ddkdomega / omega, 1e-10);
  dbl C500 = 500 * nu / (d * d * omega);

  dbl arg1 = fmin(fmax(sqrt(k) / (beta_star * omega * d), C500),
                  4 * rho * sigma_omega2 * k / (CD_komega * d * d));
  dbl arg2 = fmax(2 * sqrt(k) / (beta_star * omega * d), C500);

  *F1 = tanh(arg1 * arg1 * arg1 * arg1);
  *F2 = tanh(arg2 * arg2);
#if 0
  if (d_F1 != NULL) {
    dbl d_dot_k_omega_dk[MDE];
    dbl d_dot_k_omega_domega[MDE];
    for (int i = 0; i < pd->Num_Dim; i++) {
      for (int j = 0; j < ei[pg->imtrx]->dof[TURB_K]; j++) {
        d_dot_k_omega_dk[j] = bf[TURB_K]->grad_phi[j][i] * fv->grad_turb_omega[i];
      }
      for (int j = 0; j < ei[pg->imtrx]->dof[TURB_OMEGA]; j++) {
        d_dot_k_omega_domega[j] = bf[TURB_OMEGA]->grad_phi[j][i] * fv->grad_turb_k[i];
      }
    }

    dbl d_CD_komega_dk[MDE] = {0.};
    dbl d_CD_komega_domega[MDE] = {0.};
    if (CD_komega > 1e-10) {
      for (int j = 0; j < ei[pg->imtrx]->dof[TURB_K]; j++) {
        d_CD_komega_dk[j] = 2 * rho * sigma_omega2 * d_dot_k_omega_dk[j] / omega;
      }
      for (int j = 0; j < ei[pg->imtrx]->dof[TURB_K]; j++) {
        d_CD_komega_domega[j] =
            2 * rho * sigma_omega2 * d_dot_k_omega_domega[j] / omega -
            2 * rho * sigma_omega2 * dot_k_omega * bf[TURB_OMEGA]->phi[j] / (omega * omega);
      }
    }

    
  }
#endif
}

/* assemble_turb_k -- assemble terms (Residual & Jacobian) for conservation
 *
 * SST-2003m turbulence model
 *
 * in:
 *      time value
 *      theta (time stepping parameter, 0 for BE, 0.5 for CN)
 *      time step size
 *      Streamline Upwind Petrov Galerkin (PG) data structure
 *
 * out:
 *      lec -- gets loaded up with local contributions to resid, Jacobian
 *
 * Created:    July 2023 Weston Ortiz
 *
 */
int assemble_turb_k(dbl time_value, /* current time */
                    dbl tt,         /* parameter to vary time integration from
                                       explicit (tt = 1) to implicit (tt = 0)    */
                    dbl dt,         /* current time step size                    */
                    const PG_DATA *pg_data) {

  //! WIM is the length of the velocity vector
  int i, j, b;
  int eqn, var, peqn, pvar;
  int *pdv = pd->v[pg->imtrx];

  dbl mass, adv, src, diff;
  dbl diff_1, diff_2;

  int status = 0;

  eqn = TURB_K;

  dbl d_area = fv->wt * bf[eqn]->detJ * fv->h3;

  dbl mu = mp->viscosity;
  DENSITY_DEPENDENCE_STRUCT d_rho_struct;
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;
  dbl rho = density(d_rho, time_value);
  dbl F1 = 0;
  dbl F2 = 0;
  calc_blending_functions(rho, d_rho, &F1, &F2, NULL, NULL);

  dbl SI;
  dbl gamma_dot[DIM][DIM];
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      gamma_dot[i][j] = (fv->grad_v[i][j] + fv->grad_v[j][i]);
    }
  }
  dbl d_SI_dv[DIM][MDE];
  dbl d_SI_dmesh[DIM][MDE];
  calc_shearrate(&SI, gamma_dot, d_SI_dv, d_SI_dmesh);
  dbl a1 = 0.31;

  dbl x_dot[DIM] = {0.};
  if (pd->gv[R_MESH1]) {
    for (int i = 0; i < DIM; i++) {
      x_dot[i] = fv_dot->x[i];
    }
  }

  dbl supg = 1.;
  SUPG_terms supg_terms;
  if (mp->SAwt_funcModel == GALERKIN) {
    supg = 0.;
  } else if (mp->SAwt_funcModel == SUPG || mp->SAwt_funcModel == SUPG_GP ||
             mp->SAwt_funcModel == SUPG_SHAKIB) {
    supg = mp->SAwt_func;
    supg_tau_shakib(&supg_terms, pd->Num_Dim, dt, mu, EDDY_NU);
  }

  dbl beta_star = 0.09;
  dbl sigma_k1 = 0.85;
  dbl sigma_k2 = 1.0;

  // blended values
  dbl sigma_k = F1 * sigma_k1 + (1 - F1) * sigma_k2;

  dbl mu_t = rho * a1 * fv->turb_k / (fmax(a1 * fv->turb_omega, SI * F2));
  dbl d_mu_t_dk = rho * a1 / (fmax(a1 * fv->turb_omega, SI * F2));
  dbl d_mu_t_domega = 0;
  dbl d_mu_t_SI = 0;
  if (a1 * fv->turb_omega > SI * F2) {
    d_mu_t_domega = -rho * a1 * fv->turb_k / ((a1 * fv->turb_omega) * (a1 * fv->turb_omega));
  } else {
    d_mu_t_SI = -rho * a1 * fv->turb_k / ((SI * F2) * (SI * F2));
  }
  dbl P = mu_t * SI * SI;
  dbl Plim = fmin(P, 10 * beta_star * rho * fv->turb_omega * fv->turb_k);

  dbl d_Plim_dk = 0;
  dbl d_Plim_domega = 0;
  dbl d_Plim_dSI = 0;
  if (P < 10 * beta_star * rho * fv->turb_omega * fv->turb_k) {
    d_Plim_dk = 10 * beta_star * rho * fv->turb_omega;
    d_Plim_domega = 10 * beta_star * rho * fv->turb_k;
  } else {
    d_Plim_dSI = d_mu_t_SI * SI * SI + 2 * mu_t * SI;
    d_Plim_dk = d_mu_t_dk * SI * SI;
    d_Plim_domega = d_mu_t_domega * SI * SI;
  }

  /*
   * Residuals_________________________________________________________________
   */
  if (af->Assemble_Residual) {
    /*
     * Assemble residual for eddy viscosity
     */
    eqn = TURB_K;
    peqn = upd->ep[pg->imtrx][eqn];

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      dbl wt_func = bf[eqn]->phi[i];

      if (supg > 0) {
        if (supg != 0.0) {
          for (int p = 0; p < VIM; p++) {
            wt_func += supg * supg_terms.supg_tau * fv->v[p] * bf[eqn]->grad_phi[i][p];
          }
        }
      }

      /* Assemble mass term */
      mass = 0.0;
      if (pd->TimeIntegration != STEADY) {
        if (pd->e[pg->imtrx][eqn] & T_MASS) {
          mass += rho * fv_dot->turb_k * wt_func * d_area;
          mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
        }
      }

      /* Assemble advection term */
      adv = 0;
      for (int p = 0; p < VIM; p++) {
        adv += rho * (fv->v[p] - x_dot[p]) * fv->grad_turb_k[p];
      }
      adv *= wt_func * d_area;
      adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

      /* Assemble source terms */
      src = Plim - beta_star * rho * fv->turb_omega * fv->turb_k;
      src *= -wt_func * d_area;
      src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

      /* Assemble diffusion terms */
      diff = 0.0;
      for (int p = 0; p < VIM; p++) {
        diff += bf[eqn]->grad_phi[i][p] * (mu + mu_t * sigma_k) * fv->grad_turb_k[p];
      }
      diff *= d_area;
      diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

      lec->R[LEC_R_INDEX(peqn, i)] += mass + adv + src + diff;
    } /* end of for (i=0,ei[pg->imtrx]->dofs...) */
  }   /* end of if assemble residual */

  /*
   * Jacobian terms...
   */

  if (af->Assemble_Jacobian) {
    eqn = TURB_K;
    peqn = upd->ep[pg->imtrx][eqn];

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

      dbl wt_func = bf[eqn]->phi[i];

      dbl r_mass = 0;
      dbl r_adv = 0;
      dbl r_src = 0;
      if (supg > 0) {
        if (supg != 0.0) {
          for (int p = 0; p < VIM; p++) {
            wt_func += supg * supg_terms.supg_tau * fv->v[p] * bf[eqn]->grad_phi[i][p];
          }
        }

        /* Assemble mass term */
        r_mass = 0.0;
        if (pd->TimeIntegration != STEADY) {
          if (pd->e[pg->imtrx][eqn] & T_MASS) {
            r_mass += rho * fv_dot->turb_k;
            r_mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
          }
        }

        /* Assemble advection term */
        r_adv = 0;
        for (int p = 0; p < VIM; p++) {
          r_adv += rho * (fv->v[p] - x_dot[p]) * fv->grad_turb_k[p];
        }
        r_adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

        r_src = Plim - beta_star * rho * fv->turb_omega * fv->turb_k;
        r_src *= -pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
      }

      /* Sensitivity w.r.t. k */
      var = TURB_K;
      if (pdv[var]) {
        pvar = upd->vp[pg->imtrx][var];

        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

          /* Assemble mass term */
          mass = 0.0;
          if (pd->TimeIntegration != STEADY) {
            if (pd->e[pg->imtrx][eqn] & T_MASS) {
              mass += rho * (1.0 + 2.0 * tt) / dt * bf[eqn]->phi[j] * wt_func * d_area;
              mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
            }
          }

          /* Assemble advection term */
          adv = 0.0;
          for (int p = 0; p < VIM; p++) {
            adv += rho * (fv->v[p] - x_dot[p]) * bf[eqn]->grad_phi[j][p];
          }
          adv *= wt_func * d_area;
          adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

          /* Assemble source term */
          src = bf[var]->phi[j] * (d_Plim_dk - beta_star * rho * fv->turb_omega);
          src *= -wt_func * d_area;
          src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

          /* Assemble diffusion terms */
          diff_1 = 0.0;
          diff_2 = 0.0;
          for (int p = 0; p < VIM; p++) {
            diff_1 += bf[var]->phi[j] * (d_mu_t_dk * sigma_k) * bf[eqn]->grad_phi[i][p] *
                      fv->grad_turb_k[p];
            diff_2 += bf[eqn]->grad_phi[i][p] * (mu + mu_t * sigma_k) * bf[var]->grad_phi[j][p];
          }
          diff = diff_1 + diff_2;
          diff *= d_area;
          diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += mass + adv + src + diff;
        } /* End of loop over j */
      }   /* End of if the variable is active */

      /* Sensitivity w.r.t. omega */
      var = TURB_OMEGA;
      if (pdv[var]) {
        pvar = upd->vp[pg->imtrx][var];

        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

          /* Assemble mass term */
          mass = 0.0;

          /* Assemble advection term */
          adv = 0.0;

          /* Assemble source term */
          src = bf[var]->phi[j] * (d_Plim_domega - beta_star * rho * fv->turb_k);
          src *= -wt_func * d_area;
          src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

          /* Assemble diffusion terms */
          diff_1 = 0.0;
          diff_2 = 0.0;
          for (int p = 0; p < VIM; p++) {
            diff_1 += bf[var]->phi[j] * (d_mu_t_domega * sigma_k) * bf[eqn]->grad_phi[i][p] *
                      fv->grad_turb_k[p];
          }
          diff = diff_1 + diff_2;
          diff *= d_area;
          diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += mass + adv + src + diff;
        } /* End of loop over j */
      }   /* End of if the variable is active */

      /* Sensitivity w.r.t. velocity */
      for (b = 0; b < VIM; b++) {
        var = VELOCITY1 + b;
        if (pdv[var]) {
          pvar = upd->vp[pg->imtrx][var];

          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            if (supg > 0) {
              dbl d_wt_func =
                  supg * bf[var]->phi[j] * supg_terms.supg_tau * bf[eqn]->grad_phi[i][b];
              for (int p = 0; p < VIM; p++) {
                d_wt_func +=
                    supg * fv->v[p] * supg_terms.d_supg_tau_dv[p][j] * bf[eqn]->grad_phi[i][p];
              }
              lec->J[LEC_J_INDEX(peqn, pvar, i, j)] +=
                  (r_mass + r_adv + r_src) * d_area * d_wt_func;
            }

            /* Assemble advection term */
            adv = bf[var]->phi[j] * fv->grad_turb_k[b];
            adv *= wt_func * d_area;
            adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

            /* Assemble source term */
            src = d_Plim_dSI * d_SI_dv[b][j];
            src *= -wt_func * d_area;
            src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

            diff = 0.0;
            for (int p = 0; p < VIM; p++) {
              diff += bf[eqn]->grad_phi[i][p] * (d_SI_dv[b][j] * d_mu_t_SI * sigma_k) *
                      fv->grad_turb_k[p];
            }
            diff *= d_area;
            diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

            lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += adv + src + diff;

          } /* End of loop over j */
        }   /* End of if the variale is active */
      }     /* End of loop over velocity components */

      /* Sensistivity w.r.t. mesh */
      for (b = pd->Num_Dim; b < pd->Num_Dim; b++) {
        var = MESH_DISPLACEMENT1 + b;
        if (pdv[var]) {
          pvar = upd->vp[pg->imtrx][var];
          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            dbl d_area_dmesh = +fv->wt * bf[eqn]->d_det_J_dm[b][j] * fv->h3 +
                               fv->wt * bf[eqn]->detJ * fv->dh3dmesh[b][j];
            if (supg > 0) {
              dbl d_wt_func = 0;
              for (int p = 0; p < VIM; p++) {
                d_wt_func +=
                    supg * fv->v[p] * supg_terms.d_supg_tau_dX[p][j] * bf[eqn]->grad_phi[i][p];
                d_wt_func +=
                    supg * fv->v[p] * supg_terms.supg_tau * bf[eqn]->d_grad_phi_dmesh[i][p][b][j];
              }
              lec->J[LEC_J_INDEX(peqn, pvar, i, j)] +=
                  (r_mass + r_adv + r_src) * d_area * d_wt_func +
                  (r_mass + r_adv + r_src) * d_area_dmesh * wt_func;
            }

            lec->R[LEC_R_INDEX(peqn, i)] += mass + adv + src + diff;
          } /* End of loop over j */
        }
      }

    } /* End of loop over i */
  }   /* End of if assemble Jacobian */
  return (status);
}

/* assemble_turb_omega -- assemble terms (Residual & Jacobian) for conservation
 *
 * k-omega SST
 *
 * in:
 *      time value
 *      theta (time stepping parameter, 0 for BE, 0.5 for CN)
 *      time step size
 *      Streamline Upwind Petrov Galerkin (PG) data structure
 *
 * out:
 *      lec -- gets loaded up with local contributions to resid, Jacobian
 *
 * Created:    July 2023 Weston Ortiz
 *
 */
int assemble_turb_omega(dbl time_value, /* current time */
                        dbl tt,         /* parameter to vary time integration from
                                           explicit (tt = 1) to implicit (tt = 0)    */
                        dbl dt,         /* current time step size                    */
                        const PG_DATA *pg_data) {

  //! WIM is the length of the velocity vector
  int i, j, b;
  int eqn, var, peqn, pvar;
  int *pdv = pd->v[pg->imtrx];

  dbl mass, adv, src, diff;
  dbl diff_1, diff_2;

  int status = 0;

  eqn = TURB_OMEGA;

  dbl d_area = fv->wt * bf[eqn]->detJ * fv->h3;

  dbl mu = mp->viscosity;
  DENSITY_DEPENDENCE_STRUCT d_rho_struct;
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;
  dbl rho = density(d_rho, time_value);
  dbl F1 = 0;
  dbl F2 = 0;
  calc_blending_functions(rho, d_rho, &F1, &F2, NULL, NULL);

  dbl SI;
  dbl gamma_dot[DIM][DIM];
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      gamma_dot[i][j] = (fv->grad_v[i][j] + fv->grad_v[j][i]);
    }
  }
  dbl d_SI_dv[DIM][MDE];
  dbl d_SI_dmesh[DIM][MDE];
  calc_shearrate(&SI, gamma_dot, d_SI_dv, d_SI_dmesh);
  dbl a1 = 0.31;

  dbl x_dot[DIM] = {0.};
  if (pd->gv[R_MESH1]) {
    for (int i = 0; i < DIM; i++) {
      x_dot[i] = fv_dot->x[i];
    }
  }

  dbl supg = 1.;
  SUPG_terms supg_terms;
  if (mp->SAwt_funcModel == GALERKIN) {
    supg = 0.;
  } else if (mp->SAwt_funcModel == SUPG || mp->SAwt_funcModel == SUPG_GP ||
             mp->SAwt_funcModel == SUPG_SHAKIB) {
    supg = mp->SAwt_func;
    supg_tau_shakib(&supg_terms, pd->Num_Dim, dt, mu, EDDY_NU);
  }

  dbl beta_star = 0.09;
  dbl sigma_k1 = 0.85;
  dbl sigma_k2 = 1.0;
  dbl sigma_omega1 = 0.5;
  dbl sigma_omega2 = 0.856;
  dbl beta1 = 0.075;
  dbl beta2 = 0.0828;
  dbl gamma1 = 5.0 / 9.0;
  dbl gamma2 = 0.44;

  // blended values
  dbl gamma = F1 * gamma1 + (1 - F1) * gamma2;
  dbl sigma_k = F1 * sigma_k1 + (1 - F1) * sigma_k2;
  dbl sigma_omega = F1 * sigma_omega1 + (1 - F1) * sigma_omega2;
  dbl beta = F1 * beta1 + (1 - F1) * beta2;

  dbl mu_t = rho * a1 * fv->turb_k / (fmax(a1 * fv->turb_omega, SI * F2));
  dbl d_mu_t_dk = rho * a1 / (fmax(a1 * fv->turb_omega, SI * F2));
  dbl d_mu_t_domega = 0;
  dbl d_mu_t_SI = 0;
  if (a1 * fv->turb_omega > SI * F2) {
    d_mu_t_domega = -rho * a1 * fv->turb_k / ((a1 * fv->turb_omega) * (a1 * fv->turb_omega));
  } else {
    d_mu_t_SI = -rho * a1 * fv->turb_k / ((SI * F2) * (SI * F2));
  }
  dbl P = mu_t * SI * SI;
  dbl Plim = fmin(P, 10 * beta_star * rho * fv->turb_omega * fv->turb_k);

  dbl d_Plim_dk = 0;
  dbl d_Plim_domega = 0;
  dbl d_Plim_dSI = 0;
  if (P < 10 * beta_star * rho * fv->turb_omega * fv->turb_k) {
    d_Plim_dk = 10 * beta_star * rho * fv->turb_omega;
    d_Plim_domega = 10 * beta_star * rho * fv->turb_k;
  } else {
    d_Plim_dSI = d_mu_t_SI * SI * SI + 2 * mu_t * SI;
    d_Plim_dk = d_mu_t_dk * SI * SI;
    d_Plim_domega = d_mu_t_domega * SI * SI;
  }

  /*
   * Residuals_________________________________________________________________
   */
  if (af->Assemble_Residual) {
    /*
     * Assemble residual for eddy viscosity
     */
    eqn = TURB_K;
    peqn = upd->ep[pg->imtrx][eqn];

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      dbl wt_func = bf[eqn]->phi[i];

      if (supg > 0) {
        if (supg != 0.0) {
          for (int p = 0; p < VIM; p++) {
            wt_func += supg * supg_terms.supg_tau * fv->v[p] * bf[eqn]->grad_phi[i][p];
          }
        }
      }

      /* Assemble mass term */
      mass = 0.0;
      if (pd->TimeIntegration != STEADY) {
        if (pd->e[pg->imtrx][eqn] & T_MASS) {
          mass += rho * fv_dot->turb_omega * wt_func * d_area;
          mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
        }
      }

      /* Assemble advection term */
      adv = 0;
      for (int p = 0; p < VIM; p++) {
        adv += rho * (fv->v[p] - x_dot[p]) * fv->grad_turb_omega[p];
      }
      adv *= wt_func * d_area;
      adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

      /* Assemble source terms */
      dbl src1 = (gamma * rho / mu_t) * Plim - beta * rho * fv->turb_omega * fv->turb_omega;
      dbl src2 = 0;
      for (int p = 0; p < pd->Num_Dim; p++) {
        src2 += fv->grad_turb_k[p] * fv->grad_turb_omega[p];
      }
      src2 *= 2 * (1 - F1) * rho * sigma_omega2 / fv->turb_omega;
      src = src1 + src2;
      src *= -wt_func * d_area;
      src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

      /* Assemble diffusion terms */
      diff = 0.0;
      for (int p = 0; p < VIM; p++) {
        diff += bf[eqn]->grad_phi[i][p] * (mu + mu_t * sigma_omega) * fv->grad_turb_omega[p];
      }
      diff *= d_area;
      diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

      lec->R[LEC_R_INDEX(peqn, i)] += mass + adv + src + diff;
    } /* end of for (i=0,ei[pg->imtrx]->dofs...) */
  }   /* end of if assemble residual */

  /*
   * Jacobian terms...
   */

  if (af->Assemble_Jacobian) {
    eqn = TURB_K;
    peqn = upd->ep[pg->imtrx][eqn];

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

      dbl wt_func = bf[eqn]->phi[i];

      dbl r_mass = 0;
      dbl r_adv = 0;
      dbl r_src = 0;
      if (supg > 0) {
        if (supg != 0.0) {
          for (int p = 0; p < VIM; p++) {
            wt_func += supg * supg_terms.supg_tau * fv->v[p] * bf[eqn]->grad_phi[i][p];
          }
        }

        /* Assemble mass term */
        r_mass = 0.0;
        if (pd->TimeIntegration != STEADY) {
          if (pd->e[pg->imtrx][eqn] & T_MASS) {
            r_mass += rho * fv_dot->turb_omega;
            r_mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
          }
        }

        /* Assemble advection term */
        r_adv = 0;
        for (int p = 0; p < VIM; p++) {
          r_adv += rho * (fv->v[p] - x_dot[p]) * fv->grad_turb_omega[p];
        }
        r_adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

        dbl src1 = (gamma * rho / mu_t) * Plim - beta * rho * fv->turb_omega * fv->turb_omega;
        dbl src2 = 0;
        for (int p = 0; p < pd->Num_Dim; p++) {
          src2 += fv->grad_turb_k[p] * fv->grad_turb_omega[p];
        }
        src2 *= 2 * (1 - F1) * rho * sigma_omega2 / fv->turb_omega;
        r_src = src1 + src2;
        r_src *= -pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
      }

      /* Sensitivity w.r.t. k */
      var = TURB_OMEGA;
      if (pdv[var]) {
        pvar = upd->vp[pg->imtrx][var];

        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

          /* Assemble mass term */
          mass = 0.0;
          if (pd->TimeIntegration != STEADY) {
            if (pd->e[pg->imtrx][eqn] & T_MASS) {
              mass += rho * (1.0 + 2.0 * tt) / dt * bf[eqn]->phi[j] * wt_func * d_area;
              mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
            }
          }

          /* Assemble advection term */
          adv = 0.0;
          for (int p = 0; p < VIM; p++) {
            adv += rho * (fv->v[p] - x_dot[p]) * bf[eqn]->grad_phi[j][p];
          }
          adv *= wt_func * d_area;
          adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

          /* Assemble source term */
          dbl src_1 = bf[var]->phi[j] * ((gamma * rho / mu_t) * d_Plim_domega +
                                         (-d_mu_t_domega * gamma * rho / (mu_t * mu_t)) * Plim -
                                         2.0 * beta * rho * fv->turb_omega);
          dbl src_2a = 0;
          dbl src_2b = 0;
          for (int p = 0; p < pd->Num_Dim; p++) {
            src_2a += fv->grad_turb_k[p] * fv->grad_turb_omega[p];
            src_2b += fv->grad_turb_k[p] * bf[var]->grad_phi[j][p];
          }
          dbl src_2 =
              (2 * (1 - F1) * rho * sigma_omega2 / fv->turb_omega) * (src_2b) +
              (bf[var]->phi[j] * -2 * (1 - F1) * rho * sigma_omega2 / (SQUARE(fv->turb_omega))) *
                  src_2a;
          src = src_1 + src_2;
          src *= -wt_func * d_area;
          src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

          /* Assemble diffusion terms */
          diff_1 = 0.0;
          diff_2 = 0.0;
          for (int p = 0; p < VIM; p++) {
            diff_1 += bf[var]->phi[j] * (d_mu_t_domega * sigma_omega) * bf[eqn]->grad_phi[i][p] *
                      fv->grad_turb_omega[p];
            diff_2 += bf[eqn]->grad_phi[i][p] * (mu + mu_t * sigma_omega) * bf[var]->grad_phi[j][p];
          }
          diff = diff_1 + diff_2;
          diff *= d_area;
          diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += mass + adv + src + diff;
        } /* End of loop over j */
      }   /* End of if the variable is active */

      /* Sensitivity w.r.t. omega */
      var = TURB_K;
      if (pdv[var]) {
        pvar = upd->vp[pg->imtrx][var];

        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

          /* Assemble mass term */
          mass = 0.0;

          /* Assemble advection term */
          adv = 0.0;

          /* Assemble source term */
          dbl src1 = bf[var]->phi[j] * (gamma * rho / mu_t) * (d_Plim_dk) +
                     (-gamma * rho * d_mu_t_dk / (SQUARE(mu_t)) * Plim);
          dbl src2 = 0;
          for (int p = 0; p < pd->Num_Dim; p++) {
            src2 += bf[var]->grad_phi[j][p] * fv->grad_turb_omega[p];
          }
          src = src1 + src2;
          src *= -wt_func * d_area;
          src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

          /* Assemble diffusion terms */
          diff_1 = 0.0;
          diff_2 = 0.0;
          for (int p = 0; p < VIM; p++) {
            diff_1 += bf[var]->phi[j] * (d_mu_t_dk * sigma_k) * bf[eqn]->grad_phi[i][p] *
                      fv->grad_turb_k[p];
          }
          diff = diff_1 + diff_2;
          diff *= d_area;
          diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += mass + adv + src + diff;
        } /* End of loop over j */
      }   /* End of if the variable is active */

      /* Sensitivity w.r.t. velocity */
      for (b = 0; b < VIM; b++) {
        var = VELOCITY1 + b;
        if (pdv[var]) {
          pvar = upd->vp[pg->imtrx][var];

          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            if (supg > 0) {
              dbl d_wt_func =
                  supg * bf[var]->phi[j] * supg_terms.supg_tau * bf[eqn]->grad_phi[i][b];
              for (int p = 0; p < VIM; p++) {
                d_wt_func +=
                    supg * fv->v[p] * supg_terms.d_supg_tau_dv[p][j] * bf[eqn]->grad_phi[i][p];
              }
              lec->J[LEC_J_INDEX(peqn, pvar, i, j)] +=
                  (r_mass + r_adv + r_src) * d_area * d_wt_func;
            }

            /* Assemble advection term */
            adv = bf[var]->phi[j] * fv->grad_turb_omega[b];
            adv *= wt_func * d_area;
            adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

            /* Assemble source term */
            dbl src = ((gamma * rho / mu_t) * d_Plim_dSI +
                       (-gamma * rho * d_mu_t_SI / (SQUARE(mu_t)) * Plim)) *
                      d_SI_dv[b][j];
            src *= -wt_func * d_area;
            src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

            diff = 0.0;
            for (int p = 0; p < VIM; p++) {
              diff += bf[eqn]->grad_phi[i][p] * (d_SI_dv[b][j] * d_mu_t_SI * sigma_omega) *
                      fv->grad_turb_omega[p];
            }
            diff *= d_area;
            diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

            lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += adv + src + diff;

          } /* End of loop over j */
        }   /* End of if the variale is active */
      }     /* End of loop over velocity components */

      /* Sensistivity w.r.t. mesh */
      for (b = pd->Num_Dim; b < pd->Num_Dim; b++) {
        var = MESH_DISPLACEMENT1 + b;
        if (pdv[var]) {
          pvar = upd->vp[pg->imtrx][var];
          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            dbl d_area_dmesh = +fv->wt * bf[eqn]->d_det_J_dm[b][j] * fv->h3 +
                               fv->wt * bf[eqn]->detJ * fv->dh3dmesh[b][j];
            if (supg > 0) {
              dbl d_wt_func = 0;
              for (int p = 0; p < VIM; p++) {
                d_wt_func +=
                    supg * fv->v[p] * supg_terms.d_supg_tau_dX[p][j] * bf[eqn]->grad_phi[i][p];
                d_wt_func +=
                    supg * fv->v[p] * supg_terms.supg_tau * bf[eqn]->d_grad_phi_dmesh[i][p][b][j];
              }
              lec->J[LEC_J_INDEX(peqn, pvar, i, j)] +=
                  (r_mass + r_adv + r_src) * d_area * d_wt_func +
                  (r_mass + r_adv + r_src) * d_area_dmesh * wt_func;
            }

            lec->R[LEC_R_INDEX(peqn, i)] += mass + adv + src + diff;
          } /* End of loop over j */
        }
      }

    } /* End of loop over i */
  }   /* End of if assemble Jacobian */
  return (status);
}