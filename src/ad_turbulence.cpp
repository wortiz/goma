
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

#ifdef GOMA_ENABLE_SACADO
#include "Sacado.hpp"
#include "Sacado_Fad_DFad.hpp"

extern "C" {
#include "mm_fill_stabilization.h"
/* GOMA include files */
#define GOMA_AD_TURBULENCE_CPP
#include "density.h"
#include "el_elm.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_fill_terms.h"
#include "mm_fill_turbulent.h"
#include "mm_fill_util.h"
#include "mm_mp.h"
#include "mm_mp_structs.h"
#include "mm_viscosity.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "std.h"
}

using ADType = Sacado::Fad::DFad<double>;

ADType ad_sa_viscosity(struct Generalized_Newtonian *gn_local);

/*  _______________________________________________________________________  */

static int calc_vort_mag(ADType &vort_mag, ADType omega[DIM][DIM]) {

  vort_mag = 0.;
  /* get gamma_dot invariant for viscosity calculations */
  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      vort_mag += omega[a][b] * omega[a][b];
    }
  }

  vort_mag = sqrt(0.5 * (vort_mag) + 1e-14);
  return 0;
}

template <typename scalar>
static int ad_calc_sa_S(scalar &S,                /* strain rate invariant */
                        scalar omega[DIM][DIM]) { /* strain rate tensor */
  S = 0.;
  /* get gamma_dot invariant for viscosity calculations */
  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      S += omega[a][b] * omega[a][b];
    }
  }

  S = sqrt(0.5 * (S) + 1e-16);

  return 0;
}

struct AD_Field_Variables {
  int ielem;
  int v_offset[DIM];
  ADType v[DIM];
  ADType v_dot[DIM];
  ADType grad_v[DIM][DIM];
  int eddy_nu_offset;
  ADType eddy_nu;
  ADType eddy_nu_dot;
  ADType grad_eddy_nu[DIM];
  int turb_k_offset;
  ADType turb_k;
  ADType turb_k_dot;
  ADType grad_turb_k[DIM];
  int turb_omega_offset;
  ADType turb_omega;
  ADType turb_omega_dot;
  ADType grad_turb_omega[DIM];
  int total_ad_variables;
};

static int ad_calc_shearrate(ADType &gammadot,             /* strain rate invariant */
                             ADType gamma_dot[DIM][DIM]) { /* strain rate tensor */
  gammadot = 0.;
  /* get gamma_dot invariant for viscosity calculations */
  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      gammadot += gamma_dot[a][b] * gamma_dot[b][a];
    }
  }

  gammadot = sqrt(0.5 * fabs(gammadot) + 1e-14);
  return 0;
}

static AD_Field_Variables ad_fv;
extern "C" void fill_ad_field_variables() {
  ad_fv.ielem = ei[pg->imtrx]->ielem;
  int num_ad_variables = 0;
  ad_fv.v_offset[0] = num_ad_variables;
  if (pd->gv[VELOCITY1]) {
    num_ad_variables += ei[upd->matrix_index[VELOCITY1]]->dof[VELOCITY1];
    ad_fv.v_offset[1] = num_ad_variables;
    num_ad_variables += ei[upd->matrix_index[VELOCITY2]]->dof[VELOCITY2];
    ad_fv.v_offset[2] = num_ad_variables;
    if (WIM == 3) {
      num_ad_variables += ei[upd->matrix_index[VELOCITY3]]->dof[VELOCITY3];
    }
  }

  if (pd->gv[EDDY_NU]) {
    ad_fv.eddy_nu_offset = num_ad_variables;
    num_ad_variables += ei[upd->matrix_index[EDDY_NU]]->dof[EDDY_NU];
  }

  if (pd->gv[TURB_K]) {
    ad_fv.turb_k_offset = num_ad_variables;
    num_ad_variables += ei[upd->matrix_index[TURB_K]]->dof[TURB_K];
  }

  if (pd->gv[TURB_OMEGA]) {
    ad_fv.turb_omega_offset = num_ad_variables;
    num_ad_variables += ei[upd->matrix_index[TURB_OMEGA]]->dof[TURB_OMEGA];
  }

  ad_fv.total_ad_variables = num_ad_variables;
  if (pd->gv[VELOCITY1]) {
    for (int p = 0; p < WIM; p++) {
      ad_fv.v[p] = 0;
      for (int i = 0; i < ei[upd->matrix_index[VELOCITY1 + p]]->dof[VELOCITY1 + p]; i++) {
        ad_fv.v[p] += ADType(num_ad_variables, ad_fv.v_offset[p] + i, *esp->v[p][i]) *
                      bf[VELOCITY1 + p]->phi[i];
      }
    }
    for (int p = 0; p < VIM; p++) {
      for (int q = 0; q < VIM; q++) {
        ad_fv.grad_v[p][q] = 0;

        for (int r = 0; r < WIM; r++) {
          for (int i = 0; i < ei[upd->matrix_index[VELOCITY1 + r]]->dof[VELOCITY1 + r]; i++) {
            ad_fv.grad_v[p][q] += ADType(num_ad_variables, ad_fv.v_offset[r] + i, *esp->v[r][i]) *
                                  bf[VELOCITY1 + r]->grad_phi_e[i][r][p][q];
          }
        }
      }
    }
  }

  if (pd->gv[EDDY_NU]) {
    ad_fv.eddy_nu = 0;
    ad_fv.eddy_nu_dot = 0;
    for (int i = 0; i < ei[upd->matrix_index[EDDY_NU]]->dof[EDDY_NU]; i++) {
      ad_fv.eddy_nu += ADType(num_ad_variables, ad_fv.eddy_nu_offset + i, *esp->eddy_nu[i]) *
                       bf[EDDY_NU]->phi[i];

      ADType ednudot = ADType(num_ad_variables, ad_fv.eddy_nu_offset + i, *esp_dot->eddy_nu[i]);
      ednudot.fastAccessDx(ad_fv.eddy_nu_offset + i) = (1. + 2. * tran->theta) / tran->delta_t;
      ad_fv.eddy_nu_dot += ednudot * bf[EDDY_NU]->phi[i];
    }

    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv.grad_eddy_nu[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[EDDY_NU]]->dof[EDDY_NU]; i++) {
        ad_fv.grad_eddy_nu[q] +=
            ADType(num_ad_variables, ad_fv.eddy_nu_offset + i, *esp->eddy_nu[i]) *
            bf[EDDY_NU]->grad_phi[i][q];
      }
    }
  }

  if (pd->gv[TURB_K]) {
    ad_fv.turb_k = 0;
    ad_fv.turb_k_dot = 0;
    for (int i = 0; i < ei[upd->matrix_index[TURB_K]]->dof[TURB_K]; i++) {
      ad_fv.turb_k +=
          ADType(num_ad_variables, ad_fv.turb_k_offset + i, *esp->turb_k[i]) * bf[TURB_K]->phi[i];

      ADType ednudot = ADType(num_ad_variables, ad_fv.turb_k_offset + i, *esp_dot->turb_k[i]);
      ednudot.fastAccessDx(ad_fv.turb_k_offset + i) = (1. + 2. * tran->theta) / tran->delta_t;
      ad_fv.turb_k_dot += ednudot * bf[TURB_K]->phi[i];
    }

    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv.grad_turb_k[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[TURB_K]]->dof[TURB_K]; i++) {
        ad_fv.grad_turb_k[q] += ADType(num_ad_variables, ad_fv.turb_k_offset + i, *esp->turb_k[i]) *
                                bf[TURB_K]->grad_phi[i][q];
      }
    }
  }

  if (pd->gv[TURB_OMEGA]) {
    ad_fv.turb_omega = 0;
    ad_fv.turb_omega_dot = 0;
    for (int i = 0; i < ei[upd->matrix_index[TURB_OMEGA]]->dof[TURB_OMEGA]; i++) {
      ad_fv.turb_omega +=
          ADType(num_ad_variables, ad_fv.turb_omega_offset + i, *esp->turb_omega[i]) *
          bf[TURB_OMEGA]->phi[i];

      ADType ednudot =
          ADType(num_ad_variables, ad_fv.turb_omega_offset + i, *esp_dot->turb_omega[i]);
      ednudot.fastAccessDx(ad_fv.turb_omega_offset + i) = (1. + 2. * tran->theta) / tran->delta_t;
      ad_fv.turb_omega_dot += ednudot * bf[TURB_OMEGA]->phi[i];
    }

    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv.grad_turb_omega[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[TURB_OMEGA]]->dof[TURB_OMEGA]; i++) {
        ad_fv.grad_turb_omega[q] +=
            ADType(num_ad_variables, ad_fv.turb_omega_offset + i, *esp->turb_omega[i]) *
            bf[TURB_OMEGA]->grad_phi[i][q];
      }
    }
  }

#if 0
  // check field variables
  for (int p = 0; p < VIM; p++) {
    if (fabs(ad_fv.v[p].val() - fv->v[p]) > 1e-14) {
      printf("diff in fv->v[%d] %.12f != %.12f\n", p, ad_fv.v[p].val(), fv->v[p]);
    }
    for (int q = 0; q < VIM; q++) {
      if (fabs(ad_fv.grad_v[p][q].val() - fv->grad_v[p][q]) > 1e-12) {
        printf("diff in fv->grad_v[%d][%d] %.12f != %.12f\n", p, q, ad_fv.grad_v[p][q].val(),
               fv->grad_v[p][q]);
      }
    }
  }
  if (fabs(ad_fv.eddy_nu.val() - fv->eddy_nu) > 1e-14) {
    printf("diff in fv->eddy_nu %.12f != %.12f\n", ad_fv.eddy_nu.val(), fv->eddy_nu);
  }
  if (fabs(ad_fv.eddy_nu_dot.val() - fv_dot->eddy_nu) > 1e-14) {
    printf("diff in fv->eddy_nu_dot %.12f != %.12f\n", ad_fv.eddy_nu_dot.val(), fv_dot->eddy_nu);
  }
  for (int p = 0; p < pd->Num_Dim; p++) {
    if (fabs(ad_fv.grad_eddy_nu[p].val() - fv->grad_eddy_nu[p]) > 1e-14) {
      printf("diff in fv->grad_eddy_nu[%d] %.12f != %.12f\n", p, ad_fv.grad_eddy_nu[p].val(),
             fv->grad_eddy_nu[p]);
    }
  }
#endif
}

static void ad_supg_tau_shakib(ADType &supg_tau, int dim, dbl dt, dbl diffusivity, int interp_eqn) {
  dbl G[DIM][DIM];

  get_metric_tensor(bf[interp_eqn]->B, dim, ei[pg->imtrx]->ielem_type, G);

  supg_tau = 0;

  ADType v_d_gv = 0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      v_d_gv += fabs(ad_fv.v[i] * G[i][j] * ad_fv.v[j]);
    }
  }

  dbl diff_g_g = 0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      diff_g_g += G[i][j] * G[i][j];
    }
  }
  diff_g_g *= 9 * diffusivity * diffusivity;

  if (dt > 0) {
    supg_tau = 1.0 / (sqrt(4 / (dt * dt) + v_d_gv + diff_g_g));
  } else {
    supg_tau = 1.0 / (sqrt(v_d_gv + diff_g_g) + 1e-14);
  }
}

extern "C" void
ad_tau_momentum_shakib(momentum_tau_terms *tau_terms, int dim, dbl dt, int pspg_scale) {
  dbl G[DIM][DIM];
  dbl inv_rho = 1.0;
  DENSITY_DEPENDENCE_STRUCT d_rho_struct;
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;

  if (pspg_scale) {
    dbl rho = density(d_rho, dt);
    inv_rho = 1.0 / rho;
  }

  int interp_eqn = VELOCITY1;
  get_metric_tensor(bf[interp_eqn]->B, dim, ei[pg->imtrx]->ielem_type, G);

  ADType v_d_gv = 0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      v_d_gv += fabs(ad_fv.v[i] * G[i][j] * ad_fv.v[j]);
    }
  }

  ADType mu = ad_sa_viscosity(gn);

  ADType coeff = (12.0 * mu * mu);

  ADType diff_g_g = 0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      diff_g_g += coeff * G[i][j] * G[i][j];
    }
  }

  dbl d_v_d_gv[DIM][MDE];
  for (int a = 0; a < dim; a++) {
    for (int k = 0; k < ei[pg->imtrx]->dof[VELOCITY1]; k++) {
      d_v_d_gv[a][k] = 0.0;
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          d_v_d_gv[a][k] += delta(a, i) * bf[VELOCITY1 + a]->phi[k] * G[i][j] * fv->v[j] +
                            delta(a, j) * fv->v[i] * G[i][j] * bf[VELOCITY1 + a]->phi[k];
        }
      }
    }
  }
  ADType tau;

  if (pd->TimeIntegration != STEADY) {
    tau = inv_rho / (sqrt(4 / (dt * dt) + v_d_gv + diff_g_g));
    tau_terms->tau = tau.val();
  } else {
    tau = inv_rho / (sqrt(v_d_gv + diff_g_g) + 1e-14);
    tau_terms->tau = tau.val();
  }

  for (int a = 0; a < dim; a++) {
    for (int k = 0; k < ei[pg->imtrx]->dof[VELOCITY1]; k++) {
      tau_terms->d_tau_dv[a][k] = tau.dx(ad_fv.v_offset[a] + k);
    }
  }

#if 0
  if (pd->e[pg->imtrx][MESH_DISPLACEMENT1]) {
    dbl dG[DIM][DIM][DIM][MDE];
    get_metric_tensor_deriv(bf[MESH_DISPLACEMENT1]->B, bf[MESH_DISPLACEMENT1]->dB, dim,
                            MESH_DISPLACEMENT1, ei[pg->imtrx]->ielem_type, dG);
    for (int a = 0; a < dim; a++) {
      for (int k = 0; k < ei[pg->imtrx]->dof[MESH_DISPLACEMENT1 + a]; k++) {
        dbl v_d_gv_dx = 0;
        for (int i = 0; i < dim; i++) {
          for (int j = 0; j < dim; j++) {
            v_d_gv_dx += fv->v[i] * dG[i][j][a][k] * fv->v[j];
          }
        }

        dbl diff_g_g_dx = 0;
        for (int i = 0; i < dim; i++) {
          for (int j = 0; j < dim; j++) {
            diff_g_g_dx += 2 * coeff * dG[i][j][a][k] * G[i][j];
          }
        }
        tau_terms->d_tau_dX[a][k] = inv_rho * -0.5 *
                                    (v_d_gv_dx + diff_g_g_dx + d_mu->X[a][k] * d_diff_g_g_dmu) *
                                    supg_tau_cubed;
      }
    }
  }
  if (pd->e[pg->imtrx][TEMPERATURE]) {
    for (int k = 0; k < ei[pg->imtrx]->dof[TEMPERATURE]; k++) {
      tau_terms->d_tau_dT[k] = inv_rho * -0.5 * (d_mu->T[k] * d_diff_g_g_dmu) * supg_tau_cubed;
    }
  }
  if (pd->e[pg->imtrx][PRESSURE]) {
    for (int k = 0; k < ei[pg->imtrx]->dof[PRESSURE]; k++) {
      tau_terms->d_tau_dP[k] = inv_rho * -0.5 * (d_mu->P[k] * d_diff_g_g_dmu) * supg_tau_cubed;
    }
  }
  if (pd->e[pg->imtrx][FILL]) {
    for (int k = 0; k < ei[pg->imtrx]->dof[FILL]; k++) {
      tau_terms->d_tau_dF[k] = inv_rho * -0.5 * (d_mu->F[k] * d_diff_g_g_dmu) * supg_tau_cubed;
    }
  }
  if (pd->e[pg->imtrx][BOND_EVOLUTION]) {
    for (int k = 0; k < ei[pg->imtrx]->dof[BOND_EVOLUTION]; k++) {
      tau_terms->d_tau_dnn[k] = inv_rho * -0.5 * (d_mu->nn[k] * d_diff_g_g_dmu) * supg_tau_cubed;
    }
  }
#endif
  if (pd->e[pg->imtrx][EDDY_NU]) {
    for (int k = 0; k < ei[pg->imtrx]->dof[EDDY_NU]; k++) {
      tau_terms->d_tau_dEDDY_NU[k] = tau.dx(ad_fv.eddy_nu_offset + k);
    }
  }
#if 0
  if (pd->e[pg->imtrx][TURB_K]) {
    for (int k = 0; k < ei[pg->imtrx]->dof[TURB_K]; k++) {
      tau_terms->d_tau_dturb_k[k] =
          inv_rho * -0.5 * (d_mu->turb_k[k] * d_diff_g_g_dmu) * supg_tau_cubed;
    }
  }
  if (pd->e[pg->imtrx][TURB_OMEGA]) {
    for (int k = 0; k < ei[pg->imtrx]->dof[TURB_OMEGA]; k++) {
      tau_terms->d_tau_dturb_omega[k] =
          inv_rho * -0.5 * (d_mu->turb_omega[k] * d_diff_g_g_dmu) * supg_tau_cubed;
    }
  }
  if (pd->e[pg->imtrx][MASS_FRACTION]) {
    for (int w = 0; w < pd->Num_Species_Eqn; w++) {
      for (int k = 0; k < ei[pg->imtrx]->dof[MASS_FRACTION]; k++) {
        tau_terms->d_tau_dC[w][k] =
            inv_rho * -0.5 * (d_mu->C[w][k] * d_diff_g_g_dmu) * supg_tau_cubed;
      }
    }
  }
#endif
}

extern "C" void ad_sa_wall_func(double func[DIM],
                                 double d_func[DIM][MAX_VARIABLE_TYPES + MAX_CONC][MDE]) {
  // kind of hacky, near wall velocity is velocity at central node
  ADType unw[DIM];
  ADType eddy_nw;
  unw[0] = ADType(ad_fv.total_ad_variables, ad_fv.v_offset[0] + 8, *esp->v[0][8]);
  unw[1] = ADType(ad_fv.total_ad_variables, ad_fv.v_offset[1] + 8, *esp->v[1][8]);
  eddy_nw = ADType(ad_fv.total_ad_variables, ad_fv.eddy_nu_offset + 8, *esp->eddy_nu[8]);

  ADType mu = 0;
  dbl scale = 1.0;
  DENSITY_DEPENDENCE_STRUCT d_rho;
  if (gn->ConstitutiveEquation == TURBULENT_SA_DYNAMIC) {
    scale = density(&d_rho, tran->time_value);
  }
  int negative_mu_e = FALSE;
  if (fv_old->eddy_nu < 0) {
    negative_mu_e = TRUE;
  }

  double mu_newt = mp->viscosity;
  ADType fv1 = 1.0;
  if (negative_mu_e) {
    mu = 0;
  } else {

    ADType mu_e = eddy_nw;
    ADType cv1 = 7.1;
    ADType chi = mu_e / mu_newt;
    fv1 = pow(chi, 3) / (pow(chi, 3) + pow(cv1, 3));

    mu = scale * (mu_e * fv1);
    if (mu > 1e3 * mu_newt) {
      mu = 1e3 * mu_newt;
    }
  }

  ADType ut = 0;
  for (int i = 0; i < 2; i++) {
    ut += unw[i] * fv->stangent[0][i];
  }

  ADType normgv = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      normgv += ad_fv.grad_v[i][j] * ad_fv.grad_v[i][j];
    }
  }
  normgv = std::sqrt(normgv);

  ADType mu_t = scale * ut * ut / (normgv + 1e-12);

  ADType mu_tt = std::max(mu_t - mu, 0);


  ADType omega[DIM][DIM];
  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      omega[a][b] = fv_old->grad_v[a][b] + fv_old->grad_v[b][a];
    }
  }


  ADType S = 0.;
  /* get gamma_dot invariant for viscosity calculations */
  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      S += omega[a][b] * omega[a][b];
    }
  }

  S = sqrt(0.5 * (S) + 1e-16);

  ADType eddy_nu = std::min(std::min(5 , 1e-1 * scale * S * mp->viscosity), fv_old->eddy_nu * 1.5);

  func[0] = fv->eddy_nu - eddy_nu.val();

  // printf("eddynu = %g %g %g\n", eddy_nu.val(), fv->eddy_nu, func[0]);

  // for (int p = 0; p < WIM; p++) {
  //   for (int i = 0; i < ei[pg->imtrx]->dof[VELOCITY1 + p]; i++) {
  //     d_func[0][VELOCITY1 + p][i] = eddy_nu.dx(ad_fv.v_offset[p] + i);
  //   }
  // }

  // for (int i = 0; i < ei[pg->imtrx]->dof[EDDY_NU]; i++) {
  //   d_func[0][EDDY_NU][i] = eddy_nu.dx(ad_fv.eddy_nu_offset + i);
  // }
}

ADType ad_sa_viscosity(struct Generalized_Newtonian *gn_local) {
  ADType mu = 0;
  dbl scale = 1.0;
  DENSITY_DEPENDENCE_STRUCT d_rho;
  if (gn_local->ConstitutiveEquation == TURBULENT_SA_DYNAMIC) {
    scale = density(&d_rho, tran->time_value);
  }
  int negative_mu_e = FALSE;
  if (fv_old->eddy_nu < 0) {
    negative_mu_e = TRUE;
  }

  double mu_newt = mp->viscosity;
  if (negative_mu_e) {
    mu = mu_newt;
  } else {

    ADType mu_e = ad_fv.eddy_nu;
    ADType cv1 = 7.1;
    ADType chi = mu_e / mu_newt;
    ADType fv1 = pow(chi, 3) / (pow(chi, 3) + pow(cv1, 3));

    mu = scale * (mu_newt + (mu_e * fv1));
    if (mu > 1e3 * mu_newt) {
      mu = 1e3 * mu_newt;
    }
  }

  return mu;
}

extern "C" dbl ad_sa_viscosity(struct Generalized_Newtonian *gn_local,
                               VISCOSITY_DEPENDENCE_STRUCT *d_mu) {
  ADType mu = 0;
  dbl scale = 1.0;
  DENSITY_DEPENDENCE_STRUCT d_rho;
  if (gn_local->ConstitutiveEquation == TURBULENT_SA_DYNAMIC) {
    scale = density(&d_rho, tran->time_value);
  }
  int negative_mu_e = FALSE;
  if (fv_old->eddy_nu < 0) {
    negative_mu_e = TRUE;
  }

  double mu_newt = mp->viscosity;
  if (negative_mu_e) {
    mu = mu_newt;
  } else {

    ADType mu_e = ad_fv.eddy_nu;
    ADType cv1 = 7.1;
    ADType chi = mu_e / mu_newt;
    ADType fv1 = pow(chi, 3) / (pow(chi, 3) + pow(cv1, 3));

    mu = scale * (mu_newt + (mu_e * fv1));
    if (mu > 1e3 * mu_newt) {
      mu = 1e3 * mu_newt;
    }

    if (d_mu != NULL) {
      for (int j = 0; j < ei[pg->imtrx]->dof[EDDY_NU]; j++) {
        d_mu->eddy_nu[j] = mu.dx(ad_fv.eddy_nu_offset + j);
      }
    }
  }

  return mu.val();
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

extern "C" int ad_assemble_spalart_allmaras(dbl time_value, /* current time */
                                            dbl tt, /* parameter to vary time integration from
                                                       explicit (tt = 1) to implicit (tt = 0)    */
                                            dbl dt, /* current time step size                    */
                                            const PG_DATA *pg_data) {

  //! WIM is the length of the velocity vector
  int i, j, a, b;
  int eqn, var, peqn, pvar;
  int *pdv = pd->v[pg->imtrx];

  int status = 0;

  eqn = EDDY_NU;
  double d_area = fv->wt * bf[eqn]->detJ * fv->h3;

  /* Get Eddy viscosity at Gauss point */
  ADType mu_e = ad_fv.eddy_nu;

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
  double mu_newt = mp->viscosity;

  /* Rate of rotation tensor  */
  ADType omega[DIM][DIM];
  double omega_old[DIM][DIM];
  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      omega[a][b] = (ad_fv.grad_v[a][b] - ad_fv.grad_v[b][a]);
      omega_old[a][b] = (fv_old->grad_v[a][b] - fv_old->grad_v[b][a]);
    }
  }

  /* Vorticity */
  ADType S = 0.0;
  double S_old = 0;
  ad_calc_sa_S(S, omega);
  ad_calc_sa_S(S_old, omega_old);

  double d = fv->wall_distance;
  /* Get distance from nearest wall */
  if (d < 1.0e-6)
    d = 1.0e-6;

  /* Model coefficients (constants) */
  double cb1 = 0.1355;
  double cb2 = 0.622;
  double cv1 = 7.1;
  double cv2 = 0.7;
  double cv3 = 0.9;
  double sigma = (2.0 / 3.0);
  double cw2 = 0.3;
  double cw3 = 2.0;
  double cn1 = 16;
  double kappa = 0.41;
  double cw1 = (cb1 / kappa / kappa) + (1.0 + cb2) / sigma;

  /* More model coefficients (depends on mu_e) */
  ADType chi = mu_e / mu_newt;
  ADType fv1 = pow(chi, 3) / (pow(chi, 3) + pow(cv1, 3));
  ADType fv2 = 1.0 - (chi) / (1.0 + chi * fv1);
  ADType fn = 1.0;
  if (negative_sa) {
    fn = (cn1 + pow(chi, 3.0)) / (cn1 - pow(chi, 3));
  }
  ADType Sbar = (mu_e * fv2) / (kappa * kappa * d * d);
  int negative_Se = false;
  // I tried to use the old values for equation switching for transient runs but
  // I end up getting floating point errors. because of Se going very far below
  // zero I'm trying to use current values instead and hope that Newton's method
  // will converge with the switching equations
  // previously:
  // . double Sbar_old = (fv_old->eddy_nu * fv2) / (kappa * kappa * d * d);
  //   if (transient_run && (Sbar_old < -cv2 * S_old)) {
  //     negative_Se = true;
  //   } else if (!transient_run && (Sbar < -cv2 * S)) {
  // .   negative_Se = true;
  //   }
  if (Sbar < -cv2 * S) {
    negative_Se = true;
  }
  ADType S_e = S + Sbar;
  if (negative_Se) {
    S_e = S + S * (cv2 * cv2 * S + cv3 * Sbar) / ((cv3 - 2 * cv2) * S - Sbar);
  }
  double r_max = 10.0;
  ADType r = 0.0;
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
  ADType g = r + cw2 * (pow(r, 6) - r);
  ADType fw_inside = (1.0 + pow(cw3, 6)) / (pow(g, 6) + pow(cw3, 6));
  ADType fw = g * pow(fw_inside, (1.0 / 6.0));

  dbl supg = 1.;
  ADType supg_tau = 0;
  if (mp->Mwt_funcModel == GALERKIN) {
    supg = 0.;
  } else if (mp->Mwt_funcModel == SUPG || mp->Mwt_funcModel == SUPG_GP ||
             mp->Mwt_funcModel == SUPG_SHAKIB) {
    supg = mp->Mwt_func;
    ad_supg_tau_shakib(supg_tau, pd->Num_Dim, dt, mu_newt, EDDY_NU);
  }

  /*
   * Residuals_________________________________________________________________
   */

  std::vector<ADType> resid(ei[pg->imtrx]->dof[eqn]);
  for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
    resid[i] = 0;
  }

  if (af->Assemble_Residual) {
    /*
     * Assemble residual for eddy viscosity
     */
    eqn = EDDY_NU;
    peqn = upd->ep[pg->imtrx][eqn];

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      ADType wt_func = bf[eqn]->phi[i];

      if (supg > 0) {
        if (supg != 0.0) {
          for (int p = 0; p < VIM; p++) {
            wt_func += supg * supg_tau * ad_fv.v[p] * bf[eqn]->grad_phi[i][p];
          }
        }
      }

      /* Assemble mass term */
      ADType mass = 0.0;
      if (pd->TimeIntegration != STEADY) {
        if (pd->e[pg->imtrx][eqn] & T_MASS) {
          mass += ad_fv.eddy_nu_dot * wt_func * d_area;
          mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
        }
      }

      /* Assemble advection term */
      ADType adv = 0;
      for (int p = 0; p < VIM; p++) {
        adv += ad_fv.v[p] * ad_fv.grad_eddy_nu[p];
      }
      adv *= wt_func * d_area;
      adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

      double neg_c = 1.0;
      if (negative_sa) {
        neg_c = -1.0;
      }

      /* Assemble source terms */
      ADType src_1 = cb1 * S_e * mu_e;
      ADType src_2 = neg_c * cw1 * fw * (mu_e * mu_e) / (d * d);
      ADType src = -src_1 + src_2;
      src *= wt_func * d_area;
      src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

      /* Assemble diffusion terms */
      ADType diff_1 = 0.0;
      ADType diff_2 = 0.0;
      for (int p = 0; p < VIM; p++) {
        diff_1 += bf[eqn]->grad_phi[i][p] * (mu_newt + mu_e * fn) * ad_fv.grad_eddy_nu[p];
        diff_2 += wt_func * cb2 * ad_fv.grad_eddy_nu[p] * ad_fv.grad_eddy_nu[p];
      }
      ADType diff = (1.0 / sigma) * (diff_1 - diff_2);
      diff *= d_area;
      diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

      resid[i] += mass + adv + src + diff;
      lec->R[LEC_R_INDEX(peqn, i)] += mass.val() + adv.val() + src.val() + diff.val();
    } /* end of for (i=0,ei[pg->imtrx]->dofs...) */
  }   /* end of if assemble residual */

  /*
   * Jacobian terms...
   */

  if (af->Assemble_Jacobian) {
    eqn = EDDY_NU;
    peqn = upd->ep[pg->imtrx][eqn];

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

      /* Sensitivity w.r.t. eddy viscosity */
      var = EDDY_NU;
      if (pdv[var]) {
        pvar = upd->vp[pg->imtrx][var];

        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += resid[i].dx(ad_fv.eddy_nu_offset + j);
        } /* End of loop over j */
      }   /* End of if the variable is active */

      /* Sensitivity w.r.t. velocity */
      for (b = 0; b < VIM; b++) {
        var = VELOCITY1 + b;
        if (pdv[var]) {
          pvar = upd->vp[pg->imtrx][var];

          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += resid[i].dx(ad_fv.v_offset[b] + j);

          } /* End of loop over j */
        }   /* End of if the variale is active */
      }     /* End of loop over velocity components */

    } /* End of loop over i */
  }   /* End of if assemble Jacobian */
  return (status);
}

ADType turb_omega_wall_bc(void) {
  double rho = density(NULL, tran->time_value);
  double beta1 = 0.075;
  double nu = mp->viscosity / rho;
  double Dy = 0.0;

  if (upd->turbulent_info->use_internal_wall_distance) {
    fv->wall_distance = 0.;
    if (pd->gv[pd->ShapeVar]) {
      int dofs = ei[upd->matrix_index[pd->ShapeVar]]->dof[pd->ShapeVar];
      for (int i = 0; i < dofs; i++) {
        dbl d =
            upd->turbulent_info
                ->wall_distances[ei[upd->matrix_index[pd->ShapeVar]]->gnn_list[pd->ShapeVar][i]];
        if (d > Dy) {
          Dy = d;
        }
      }
    }
  } else {
    GOMA_EH(GOMA_ERROR, "Unimplemented wall distance for turb_omega_wall_bc\n");
  }
  double omega_wall = 10.0 * (6 * nu) / (beta1 * Dy * Dy);
  return omega_wall;
}

static void
calc_blending_functions(dbl rho, DENSITY_DEPENDENCE_STRUCT *d_rho, ADType &F1, ADType &F2) {
  dbl d = fv->wall_distance;
  ADType k = ad_fv.turb_k;
  ADType omega = ad_fv.turb_omega;
  dbl nu = mp->viscosity / rho;
  dbl beta_star = 0.09;
  dbl sigma_omega2 = 0.856;
  ADType ddkdomega = 0;
  for (int i = 0; i < pd->Num_Dim; i++) {
    ddkdomega += ad_fv.grad_turb_k[i] * ad_fv.grad_turb_omega[i];
  }
  ADType CD_komega = std::max(2 * rho * sigma_omega2 * ddkdomega / (omega + 1e-20), 1e-20);
  ADType C500 = 500 * nu / (d * d * omega + 1e-20);

  ADType arg11 = 0;
  if (k > 0) {
    ADType arg11 = sqrt(k) / (beta_star * omega * d + 1e-20);
  }
  ADType arg12 = 4 * rho * sigma_omega2 * k / (CD_komega * d * d + 1e-20);
  ADType arg1 = std::min(std::max(arg11, C500), arg12);
  // ADType arg1 = std::min(std::max(sqrt(std::max(k+1e-20, 0)) / (beta_star * omega * d + 1e-20),
  // C500),
  //                        4 * rho * sigma_omega2 * k / (CD_komega * d * d + 1e-20));
  ADType arg2 = std::max(2 * arg11, C500);

  F1 = tanh(arg1 * arg1 * arg1 * arg1);
  F2 = tanh(arg2 * arg2);
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
extern "C" int ad_assemble_turb_k(dbl time_value, /* current time */
                                  dbl tt,         /* parameter to vary time integration from
                                                     explicit (tt = 1) to implicit (tt = 0)    */
                                  dbl dt,         /* current time step size                    */
                                  const PG_DATA *pg_data) {

  //! WIM is the length of the velocity vector
  int i, j, b;
  int eqn, var, peqn, pvar;
  int *pdv = pd->v[pg->imtrx];

  int status = 0;

  eqn = TURB_K;

  dbl d_area = fv->wt * bf[eqn]->detJ * fv->h3;

  dbl mu = mp->viscosity;
  DENSITY_DEPENDENCE_STRUCT d_rho_struct;
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;
  dbl rho = density(d_rho, time_value);
  ADType F1 = 0;
  ADType F2 = 0;
  calc_blending_functions(rho, d_rho, F1, F2);

  ADType SI;
  ADType gamma_dot[DIM][DIM];
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      gamma_dot[i][j] = (ad_fv.grad_v[i][j] + ad_fv.grad_v[j][i]);
    }
  }
  ad_calc_shearrate(SI, gamma_dot);
  dbl a1 = 0.31;

  /* Rate of rotation tensor  */
  ADType omega[DIM][DIM];
  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      omega[a][b] = (ad_fv.grad_v[a][b] - ad_fv.grad_v[b][a]);
    }
  }

  /* Vorticity */
  ADType Omega = 0.0;
  calc_vort_mag(Omega, omega);

  dbl x_dot[DIM] = {0.};
  if (pd->gv[R_MESH1]) {
    for (int i = 0; i < DIM; i++) {
      x_dot[i] = fv_dot->x[i];
    }
  }

  dbl supg = 1.;
  ADType supg_tau = 0;
  if (mp->SAwt_funcModel == GALERKIN) {
    supg = 0.;
  } else if (mp->SAwt_funcModel == SUPG || mp->SAwt_funcModel == SUPG_GP ||
             mp->SAwt_funcModel == SUPG_SHAKIB) {
    supg = mp->SAwt_func;
    ad_supg_tau_shakib(supg_tau, pd->Num_Dim, dt, mu, TURB_K);
  }

  dbl beta_star = 0.09;
  dbl sigma_k1 = 0.85;
  dbl sigma_k2 = 1.0;

  // blended values
  ADType sigma_k = F1 * sigma_k1 + (1 - F1) * sigma_k2;

  ADType mu_t = rho * a1 * fv_old->turb_k / (std::max(a1 * ad_fv.turb_omega, Omega * F2) + 1e-16);
  ADType P = mu_t * SI * SI;
  ADType Plim = std::min(P, 20 * beta_star * rho * ad_fv.turb_omega * fv_old->turb_k);

  std::vector<ADType> resid(ei[pg->imtrx]->dof[eqn]);
  for (int i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
    resid[i] = 0;
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
      ADType wt_func = bf[eqn]->phi[i];

      if (supg > 0) {
        if (supg != 0.0) {
          for (int p = 0; p < VIM; p++) {
            wt_func += supg * supg_tau * ad_fv.v[p] * bf[eqn]->grad_phi[i][p];
          }
        }
      }

      /* Assemble mass term */
      ADType mass = 0.0;
      if (pd->TimeIntegration != STEADY) {
        if (pd->e[pg->imtrx][eqn] & T_MASS) {
          mass += rho * ad_fv.turb_k_dot * wt_func * d_area;
          mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
        }
      }

      /* Assemble advection term */
      ADType adv = 0;
      for (int p = 0; p < VIM; p++) {
        adv += rho * (ad_fv.v[p] - x_dot[p]) * ad_fv.grad_turb_k[p];
      }
      adv *= wt_func * d_area;
      adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

      /* Assemble source terms */
      ADType src = Plim - beta_star * rho * ad_fv.turb_omega * fv_old->turb_k;
      src *= -wt_func * d_area;
      src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

      /* Assemble diffusion terms */
      ADType diff = 0.0;
      for (int p = 0; p < VIM; p++) {
        diff += bf[eqn]->grad_phi[i][p] * (mu + mu_t * sigma_k) * ad_fv.grad_turb_k[p];
      }
      diff *= d_area;
      diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

      resid[i] += mass + adv + src + diff;
      lec->R[LEC_R_INDEX(peqn, i)] += mass.val() + adv.val() + src.val() + diff.val();
    } /* end of for (i=0,ei[pg->imtrx]->dofs...) */
  }   /* end of if assemble residual */

  /*
   * Jacobian terms...
   */

  if (af->Assemble_Jacobian) {
    eqn = TURB_K;
    peqn = upd->ep[pg->imtrx][eqn];

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

      /* Sensitivity w.r.t. k */
      var = TURB_K;
      if (pdv[var]) {
        pvar = upd->vp[pg->imtrx][var];

        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += resid[i].dx(ad_fv.turb_k_offset + j);
        } /* End of loop over j */
      }   /* End of if the variable is active */

      /* Sensitivity w.r.t. omega */
      var = TURB_OMEGA;
      if (pdv[var]) {
        pvar = upd->vp[pg->imtrx][var];

        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += resid[i].dx(ad_fv.turb_omega_offset + j);
        } /* End of loop over j */
      }   /* End of if the variable is active */

      /* Sensitivity w.r.t. velocity */
      for (b = 0; b < VIM; b++) {
        var = VELOCITY1 + b;
        if (pdv[var]) {
          pvar = upd->vp[pg->imtrx][var];

          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += resid[i].dx(ad_fv.v_offset[b] + j);
          } /* End of loop over j */
        }   /* End of if the variale is active */
      }     /* End of loop over velocity components */

    } /* End of loop over i */
  }   /* End of if assemble Jacobian */
  return (status);
}

extern "C" dbl ad_turb_k_omega_sst_viscosity(VISCOSITY_DEPENDENCE_STRUCT *d_mu) {
  ADType mu = 0;
  double mu_newt = mp->viscosity;
  dbl rho;
  DENSITY_DEPENDENCE_STRUCT d_rho_struct;
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;
  rho = density(d_rho, tran->time_value);
  ADType F1 = 0;
  ADType F2 = 0;
  calc_blending_functions(rho, d_rho, F1, F2);

  ADType omega[DIM][DIM];
  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      omega[a][b] = (ad_fv.grad_v[a][b] - ad_fv.grad_v[b][a]);
    }
  }
  /* Vorticity */
  ADType Omega = 0.0;
  calc_vort_mag(Omega, omega);
  dbl a1 = 0.31;
  ADType mu_t = rho * a1 * ad_fv.turb_k / (std::max(a1 * ad_fv.turb_omega, Omega * F2) + 1e-16);

  mu = mu_newt + mu_t;
  if (d_mu != NULL) {
    for (int j = 0; j < ei[pg->imtrx]->dof[TURB_OMEGA]; j++) {
      d_mu->turb_omega[j] = mu.dx(ad_fv.turb_omega_offset + j);
    }
    for (int j = 0; j < ei[pg->imtrx]->dof[TURB_K]; j++) {
      d_mu->turb_k[j] = mu.dx(ad_fv.turb_k_offset + j);
    }
    for (int b = 0; b < VIM; b++) {
      int var = VELOCITY1 + b;
      for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
        d_mu->v[b][j] = mu.dx(ad_fv.v_offset[b] + j);
      }
    }
  }
  return mu.val();
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
extern "C" int ad_assemble_turb_omega(dbl time_value, /* current time */
                                      dbl tt,         /* parameter to vary time integration from
                                                         explicit (tt = 1) to implicit (tt = 0)    */
                                      dbl dt, /* current time step size                    */
                                      const PG_DATA *pg_data) {

  //! WIM is the length of the velocity vector
  int i, j, b;
  int eqn, var, peqn, pvar;
  int *pdv = pd->v[pg->imtrx];

  int status = 0;

  eqn = TURB_OMEGA;

  dbl d_area = fv->wt * bf[eqn]->detJ * fv->h3;

  dbl mu = mp->viscosity;
  DENSITY_DEPENDENCE_STRUCT d_rho_struct;
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;
  dbl rho = density(d_rho, time_value);
  ADType F1 = 0;
  ADType F2 = 0;
  calc_blending_functions(rho, d_rho, F1, F2);

  ADType SI;
  ADType gamma_dot[DIM][DIM];
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      gamma_dot[i][j] = (ad_fv.grad_v[i][j] + ad_fv.grad_v[j][i]);
    }
  }

  ADType omega[DIM][DIM];
  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      omega[a][b] = (ad_fv.grad_v[a][b] - ad_fv.grad_v[b][a]);
    }
  }
  /* Vorticity */
  ADType Omega = 0.0;
  calc_vort_mag(Omega, omega);

  ad_calc_shearrate(SI, gamma_dot);
  dbl a1 = 0.31;

  dbl x_dot[DIM] = {0.};
  if (pd->gv[R_MESH1]) {
    for (int i = 0; i < DIM; i++) {
      x_dot[i] = fv_dot->x[i];
    }
  }

  dbl supg = 1.;
  ADType supg_tau;
  if (mp->SAwt_funcModel == GALERKIN) {
    supg = 0.;
  } else if (mp->SAwt_funcModel == SUPG || mp->SAwt_funcModel == SUPG_GP ||
             mp->SAwt_funcModel == SUPG_SHAKIB) {
    supg = mp->SAwt_func;
    ad_supg_tau_shakib(supg_tau, pd->Num_Dim, dt, mu, TURB_OMEGA);
  }

  dbl beta_star = 0.09;
  dbl sigma_k1 = 0.85;
  dbl sigma_k2 = 1.0;
  dbl sigma_omega1 = 0.5;
  dbl sigma_omega2 = 0.856;
  dbl beta1 = 0.075;
  dbl beta2 = 0.0828;
  dbl gamma1 = beta1 / beta_star;
  dbl gamma2 = beta2 / beta_star;

  // blended values
  ADType gamma = F1 * gamma1 + (1 - F1) * gamma2;
  ADType sigma_k = F1 * sigma_k1 + (1 - F1) * sigma_k2;
  ADType sigma_omega = F1 * sigma_omega1 + (1 - F1) * sigma_omega2;
  ADType beta = F1 * beta1 + (1 - F1) * beta2;

  ADType mu_t = rho * a1 * ad_fv.turb_k / (std::max(a1 * fv_old->turb_omega, Omega * F2) + 1e-16);
  ADType P = mu_t * SI * SI;
  ADType Plim = std::min(P, 10 * beta_star * rho * fv_old->turb_omega * ad_fv.turb_k);

  std::vector<ADType> resid(ei[pg->imtrx]->dof[eqn]);
  for (int i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
    resid[i] = 0;
  }
  /*
   * Residuals_________________________________________________________________
   */
  if (af->Assemble_Residual) {
    /*
     * Assemble residual for eddy viscosity
     */
    eqn = TURB_OMEGA;
    peqn = upd->ep[pg->imtrx][eqn];

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      ADType wt_func = bf[eqn]->phi[i];

      if (supg > 0) {
        if (supg != 0.0) {
          for (int p = 0; p < VIM; p++) {
            wt_func += supg * supg_tau * ad_fv.v[p] * bf[eqn]->grad_phi[i][p];
          }
        }
      }

      /* Assemble mass term */
      ADType mass = 0.0;
      if (pd->TimeIntegration != STEADY) {
        if (pd->e[pg->imtrx][eqn] & T_MASS) {
          mass += rho * ad_fv.turb_omega_dot * wt_func * d_area;
          mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
        }
      }

      /* Assemble advection term */
      ADType adv = 0;
      for (int p = 0; p < VIM; p++) {
        adv += rho * (ad_fv.v[p] - x_dot[p]) * ad_fv.grad_turb_omega[p];
      }
      adv *= wt_func * d_area;
      adv *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

      /* Assemble source terms */
      ADType src1 = (gamma * rho / (mu_t + 1e-16)) * Plim -
                    beta * rho * fv_old->turb_omega * fv_old->turb_omega;
      ADType src2 = 0;
      for (int p = 0; p < pd->Num_Dim; p++) {
        src2 += ad_fv.grad_turb_k[p] * fv_old->grad_turb_omega[p];
      }
      src2 *= 2 * (1 - F1) * rho * sigma_omega2 / (fv_old->turb_omega + 1e-16);
      ADType src = src1 + src2;
      src *= -wt_func * d_area;
      src *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

      /* Assemble diffusion terms */
      ADType diff = 0.0;
      for (int p = 0; p < VIM; p++) {
        diff += bf[eqn]->grad_phi[i][p] * (mu + mu_t * sigma_omega) * ad_fv.grad_turb_omega[p];
      }
      diff *= d_area;
      diff *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

      resid[i] += mass + adv + src + diff;
      lec->R[LEC_R_INDEX(peqn, i)] += mass.val() + adv.val() + src.val() + diff.val();
    } /* end of for (i=0,ei[pg->imtrx]->dofs...) */
  }   /* end of if assemble residual */

  /*
   * Jacobian terms...
   */

  if (af->Assemble_Jacobian) {
    eqn = TURB_OMEGA;
    peqn = upd->ep[pg->imtrx][eqn];

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      /* Sensitivity w.r.t. k */
      var = TURB_OMEGA;
      if (pdv[var]) {
        pvar = upd->vp[pg->imtrx][var];

        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += resid[i].dx(ad_fv.turb_omega_offset + j);
        } /* End of loop over j */
      }   /* End of if the variable is active */

      /* Sensitivity w.r.t. omega */
      var = TURB_K;
      if (pdv[var]) {
        pvar = upd->vp[pg->imtrx][var];
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += resid[i].dx(ad_fv.turb_k_offset + j);
        } /* End of loop over j */
      }   /* End of if the variable is active */

      /* Sensitivity w.r.t. velocity */
      for (b = 0; b < VIM; b++) {
        var = VELOCITY1 + b;
        if (pdv[var]) {
          pvar = upd->vp[pg->imtrx][var];

          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += resid[i].dx(ad_fv.v_offset[b] + j);

          } /* End of loop over j */
        }   /* End of if the variale is active */
      }     /* End of loop over velocity components */

    } /* End of loop over i */
  }   /* End of if assemble Jacobian */
  return (status);
}
#endif