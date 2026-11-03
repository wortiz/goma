/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) Goma Developers, National Technology & Engineering        *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/
#ifdef GOMA_ENABLE_SACADO
#include "ad/viscosity.h"
#include "ad/turbulence.h"
#include "ad/util.h"
/* GOMA include files */
#include "density.h"
#include "el_elm.h"
#include "mm_as.h"
#include "mm_as_structs.h"
#include "mm_eh.h"
#include "mm_fill_ls.h"
#include "mm_fill_stress.h"
#include "mm_mp.h"
#include "mm_mp_structs.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "std.h"

ADType sst_viscosity(const ADType &Omega, const ADType &F2) {
  ADType omega = std::max(1e-20, goma_ctx->fem.ad_fv->turb_omega);
  ADType k = std::max(1e-20, goma_ctx->fem.ad_fv->turb_k);
  dbl rho = density(NULL, goma_ctx->fem.tran->time_value);
  dbl a1 = 0.31;
  ADType mu_turb = goma_ctx->fem.mp->viscosity + rho * a1 * k / (std::max(a1 * omega, Omega * F2));
  return mu_turb;
}

ADType ad_sa_viscosity(struct Generalized_Newtonian *gn_local) {
  ADType mu = 0;
  dbl scale = 1.0;
  DENSITY_DEPENDENCE_STRUCT d_rho;
  if (gn_local->ConstitutiveEquation == TURBULENT_SA_DYNAMIC) {
    scale = density(&d_rho, goma_ctx->fem.tran->time_value);
  }
  int negative_mu_e = FALSE;
  if (goma_ctx->fem.fv_old->eddy_nu < 0) {
    negative_mu_e = TRUE;
  }

  double mu_newt = goma_ctx->fem.mp->viscosity;
  if (negative_mu_e) {
    mu = mu_newt;
  } else {

    ADType mu_e = goma_ctx->fem.ad_fv->eddy_nu;
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
    scale = density(&d_rho, goma_ctx->fem.tran->time_value);
  }
  int negative_mu_e = FALSE;
  if (goma_ctx->fem.fv_old->eddy_nu < 0) {
    negative_mu_e = TRUE;
  }

  double mu_newt = goma_ctx->fem.mp->viscosity;
  if (negative_mu_e) {
    mu = mu_newt;
  } else {

    ADType mu_e = goma_ctx->fem.ad_fv->eddy_nu;
    ADType cv1 = 7.1;
    ADType chi = mu_e / mu_newt;
    ADType fv1 = pow(chi, 3) / (pow(chi, 3) + pow(cv1, 3));

    mu = scale * (mu_newt + (mu_e * fv1));
    if (mu > 1e3 * mu_newt) {
      mu = 1e3 * mu_newt;
    }

    if (d_mu != NULL) {
      for (int j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[EDDY_NU]; j++) {
        d_mu->eddy_nu[j] = mu.dx(goma_ctx->fem.ad_fv->offset[EDDY_NU] + j);
      }
    }
  }

  return mu.val();
}
/* This routine calculates the adaptive viscosity from Sun et al., 1999.
 * The adaptive viscosity term multiplies the continuous and discontinuous
 * shear-rate, so it should cancel out and not affect the
 * solution, other than increasing the stability of the
 * algorithm in areas of high shear and stress.
 */
ADType ad_numerical_viscosity(ADType s[DIM][DIM], /* total stress */
                              ADType gamma_cont[DIM][DIM],
                              int sdim) /* continuous shear rate */
{
  int a, b;
  ADType s_dbl_dot_s;
  ADType g_dbl_dot_g;
  ADType eps2;
  ADType eps; /* should migrate this to input deck */

  ADType mun;

  eps = goma_ctx->fem.vn->eps;

  eps2 = eps / 2.;

  s_dbl_dot_s = 0.;
  for (a = 0; a < sdim; a++) {
    for (b = 0; b < sdim; b++) {
      s_dbl_dot_s += s[a][b] * s[a][b];
    }
  }

  g_dbl_dot_g = 0.;
  for (a = 0; a < goma_ctx->problem.vim; a++) {
    for (b = 0; b < goma_ctx->problem.vim; b++) {
      g_dbl_dot_g += gamma_cont[a][b] * gamma_cont[a][b];
    }
  }

  mun = (sqrt(1. + eps2 * s_dbl_dot_s)) / sqrt(1. + eps2 * g_dbl_dot_g);

  return (mun);
}

ADType ad_arrhenius_simple_viscosity(struct Generalized_Newtonian *gn_local,
                                     ADType gamma_dot[DIM][DIM]);

ADType ad_carreau_arrhenius_viscosity(struct Generalized_Newtonian *gn_local,
                                      ADType gamma_dot[DIM][DIM]);
ADType ad_arrhenius_viscosity(struct Generalized_Newtonian *gn_local, ADType gamma_dot[DIM][DIM]);

ADType ad_ls_modulate_property(
    const ADType &p1, const ADType &p2, double width, double pm_minus, double pm_plus) {
  ADType p_plus, p_minus, p;

  p_minus = p1 * pm_plus + p2 * pm_minus;
  p_plus = p1 * pm_minus + p2 * pm_plus;

  /* Fetch the level set interfacial functions. */
  load_lsi(width);

  /* Calculate the material property. */
  if (goma_ctx->fem.ls->Elem_Sign == -1)
    p = p_minus;
  else if (goma_ctx->fem.ls->Elem_Sign == 1)
    p = p_plus;
  else
    p = p_minus + (p_plus - p_minus) * goma_ctx->fem.lsi->H;

  return (p);
}

int ad_ls_modulate_viscosity(
    ADType &mu1, double mu2, double width, double pm_minus, double pm_plus, const int model) {

  if (model == RATIO) {
    GOMA_EH(GOMA_ERROR, "Invalid Viscosity Model ls_modulate");
  }
  mu1 = ad_ls_modulate_property(mu1, mu2, width, pm_minus, pm_plus);

  return (1);
}

ADType ad_bingham_viscosity(struct Generalized_Newtonian *gn_local,
                            ADType gamma_dot[DIM][DIM]) { /* strain rate tensor */

  ADType gammadot; /* strain rate invariant */

  ADType val1;
  ADType visc_cy;
  ADType yield, shear;
  ADType mu = 0.;
  dbl mu0;
  dbl muinf;
  dbl nexp;
  dbl atexp;
  dbl aexp;
  dbl at_shift;
  dbl lambda;
  dbl tau_y = 0.0;
  dbl fexp;
  dbl temp;
#if MELTING_BINGHAM
  dbl tmelt;
#endif

  ad_calc_shearrate(gammadot, gamma_dot);

  mu0 = gn_local->mu0;
  nexp = gn_local->nexp;
  muinf = gn_local->muinf;
  aexp = gn_local->aexp;
  atexp = gn_local->atexp;
  lambda = gn_local->lam;
  if (gn_local->tau_yModel == CONSTANT) {
    tau_y = gn_local->tau_y;
  } else {
    GOMA_EH(GOMA_ERROR, "Invalid Yield Stress Model");
  }
  fexp = gn_local->fexp;

  if (goma_ctx->fem.pd->gv[TEMPERATURE]) {
    temp = goma_ctx->fem.fv->T;
  } else {
    temp = goma_ctx->fem.upd->Process_Temperature;
  }

  if (DOUBLE_NONZERO(temp) && DOUBLE_NONZERO(goma_ctx->fem.mp->reference[TEMPERATURE])) {
    /* normal, non-melting version */
    at_shift = exp(atexp * (1. / temp - 1. / goma_ctx->fem.mp->reference[TEMPERATURE]));
    if (!isfinite(at_shift)) {
      at_shift = DBL_MAX;
    }
  } else {
    at_shift = 1.;
  }

  if (DOUBLE_NONZERO(at_shift * lambda * gammadot)) {
    shear = std::pow(at_shift * lambda * gammadot, aexp);
    val1 = std::pow(at_shift * lambda * gammadot, aexp - 1.);
  } else {
    shear = 0.;
  }

  if (DOUBLE_NONZERO(gammadot) && DOUBLE_NONZERO(at_shift)) {
    yield = tau_y * (1. - exp(-at_shift * fexp * gammadot)) / (at_shift * gammadot);
  } else {
    yield = tau_y * fexp;
  }

  visc_cy = pow(1. + shear, (nexp - 1.) / aexp);

  mu = at_shift * (muinf + (mu0 - muinf + yield) * visc_cy);

  return (mu);
}
extern "C" dbl ad_viscosity_wrap(struct Generalized_Newtonian *gn_local) {
  ADType gamma[DIM][DIM];
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      gamma[i][j] = goma_ctx->fem.ad_fv->grad_v[i][j] + goma_ctx->fem.ad_fv->grad_v[j][i];
    }
  }

  auto mu = ad_viscosity(goma_ctx->fem.gn, gamma);
  return mu.val();
}
ADType ad_viscosity(struct Generalized_Newtonian *gn_local, ADType gamma_dot[DIM][DIM]) {
  int err;
  ADType mu = 0.;

  /* this section is for all Newtonian models */
  if (gn_local->ConstitutiveEquation == NEWTONIAN) {
    if (goma_ctx->fem.mp->ViscosityModel == CONSTANT) {
      /*  mu   = gn_local->mu0; corrected for auto continuation 3/01 */
      if (gn_local->ConstitutiveEquation == CONSTANT) {
        mu = gn_local->mu0;
      } else {
        mu = goma_ctx->fem.mp->viscosity;
      }
      goma_ctx->fem.mp_old->viscosity = mu.val();
    } else {
      GOMA_EH(GOMA_ERROR, "Unrecognized viscosity model for Newtonian fluid");
    }
  } else if (gn_local->ConstitutiveEquation == CONSTANT) {
    mu = gn_local->mu0;
    goma_ctx->fem.mp_old->viscosity = mu.val();
    /*Sensitivities were already set to zero */
  } else if (gn_local->ConstitutiveEquation == TURBULENT_K_OMEGA) {
    ADType W[DIM][DIM];
    for (int i = 0; i < DIM; i++) {
      for (int j = 0; j < DIM; j++) {
        W[i][j] = 0.5 * (goma_ctx->fem.ad_fv->grad_v[i][j] - goma_ctx->fem.ad_fv->grad_v[j][i]);
      }
    }

    ADType Omega = 0.0;
    for (int i = 0; i < DIM; i++) {
      for (int j = 0; j < DIM; j++) {
        Omega += W[i][j] * W[i][j];
      }
    }
    Omega = sqrt(std::max(Omega, 1e-20));
    ADType F1, F2;
    compute_sst_blending(F1, F2);
    mu = sst_viscosity(Omega, F2);
    // mu = ad_only_turb_k_omega_viscosity();
  } else if (gn_local->ConstitutiveEquation == TURBULENT_SA ||
             gn_local->ConstitutiveEquation == TURBULENT_SA_DYNAMIC) {
    mu = ad_sa_viscosity(gn_local);
  } else if (gn_local->ConstitutiveEquation == BINGHAM) {
    mu = ad_bingham_viscosity(gn_local, gamma_dot);
  } else if (gn_local->ConstitutiveEquation == CARREAU_ARRHENIUS) {
    mu = ad_carreau_arrhenius_viscosity(gn_local, gamma_dot);
  } else if (gn_local->ConstitutiveEquation == ARRHENIUS_ADVANCED) {
    mu = ad_arrhenius_viscosity(gn_local, gamma_dot);
  } else if (gn_local->ConstitutiveEquation == ARRHENIUS_SIMPLE) {
    mu = ad_arrhenius_simple_viscosity(gn_local, gamma_dot);
  } else {
    GOMA_EH(GOMA_ERROR, "Unrecognized viscosity model for non-Newtonian fluid");
  }

  if (goma_ctx->fem.ls != NULL && gn_local->ConstitutiveEquation != VE_LEVEL_SET &&
      goma_ctx->fem.mp->ViscosityModel != LEVEL_SET &&
      goma_ctx->fem.mp->ViscosityModel != LS_QUADRATIC && goma_ctx->fem.mp->mp2nd != NULL &&
      (goma_ctx->fem.mp->mp2nd->ViscosityModel == CONSTANT ||
       goma_ctx->fem.mp->mp2nd->ViscosityModel == RATIO)) {
    err = ad_ls_modulate_viscosity(
        mu, goma_ctx->fem.mp->mp2nd->viscosity, goma_ctx->fem.ls->Length_Scale,
        (double)goma_ctx->fem.mp->mp2nd->viscositymask[0],
        (double)goma_ctx->fem.mp->mp2nd->viscositymask[1], goma_ctx->fem.mp->mp2nd->ViscosityModel);
    GOMA_EH(err, "ls_modulate_viscosity");
  }
  return (mu);
}
#endif // GOMA_ENABLE_SACADO