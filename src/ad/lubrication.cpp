#include "ad/lubrication.h"
#include "ad/level_set.h"
#include "ad/momentum.h"
#include "ad/structs.h"
#include <memory>
extern "C" {
#include "density.h"
#include "el_elm.h"
#include "el_geom.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_eh.h"
#include "mm_fill_energy.h"
#include "mm_fill_ptrs.h"
#include "mm_fill_rs.h"
#include "mm_fill_shell.h"
#include "mm_fill_solid.h"
#include "mm_fill_species.h"
#include "mm_fill_terms.h"
#include "mm_fill_util.h"
#include "mm_mp.h"
#include "mm_mp_const.h"
#include "mm_mp_structs.h"
#include "mm_ns_bc.h"
#include "mm_post_def.h"
#include "mm_shell_util.h"
#include "mm_std_models.h"
#include "mm_std_models_shell.h"
#include "mm_viscosity.h"
#include "rd_mesh.h"
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
#include "rf_node_const.h"
#include "rf_solver.h"
#include "rf_solver_const.h"
#include "rf_vars_const.h"
#include "shell_tfmp_struct.h"
#include "shell_tfmp_util.h"
#include "sl_util.h"
#include "user_mp.h"
}

std::unique_ptr<AD_Lubrication_Auxiliaries> AD_LubAux = nullptr;

void ADInn(ADType v[DIM], // Input vector
           ADType w[DIM]  // Output rotated vector
           )
/******************************************************************************
 *
 * Inn()
 *
 * Function to rotate into shell cordinates by the following transformation:
 *       w = (I-nn)*v
 *
 * Scott A Roberts (1514) sarober@sandia.gov
 *
 ******************************************************************************/
{
  int i, j;
  for (i = 0; i < DIM; i++) {
    w[i] = 0.0;
    for (j = 0; j < DIM; j++) {
      w[i] += (v[j] * delta(i, j) - v[j] * fv->snormal[i] * fv->snormal[j]);
    }
  }
  return;
} /* End of Inn */

void ad_calculate_lub_q_v(const int EQN, double time, double dt, double xi[DIM], const Exo_DB *exo)
/******************************************************************************
 *
 * calculate_lub_q_v()
 *
 * Function to calculate flow rate per unit width (q) and average velocity (v)
 * in lubrication flow
 *
 *
 * Kris Tjiptowidjojo tjiptowi@unm.edu
 *
 * EDITED:
 * 2010-11-30: Scott Roberts - sarober@sandia.gov
 *    Re-wrote the lubrication section to include more key physics.
 *    Added calculation of full Jacobian entries.
 * REVAMPED to allow for shear-thinning and other viscosity models
 *    Robert Secor - rbs@hirdeal.com - October 4, 2023
 ******************************************************************************/
{
  if (!AD_LubAux) {
    AD_LubAux = std::make_unique<AD_Lubrication_Auxiliaries>();
  }
  int i, j, k, jk, w;
  ADType H;
  dbl veloL[DIM], veloU[DIM];
  ADType mu, dmu_dc = 0., dmu_dT = 0., srate = 0.;
  ADType *dmu_df = NULL;
  ADType rho;
  VISCOSITY_DEPENDENCE_STRUCT d_mu_struct; /* viscosity dependence */
  VISCOSITY_DEPENDENCE_STRUCT *d_mu = &d_mu_struct;
  DENSITY_DEPENDENCE_STRUCT d_rho_struct;
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;
  int VAR;
  int err;

  /* Problem dimensions */
  int dim = pd->Num_Dim;
  int do_convection = (pd->v[pg->imtrx][VELOCITY1] && (mp->FSIModel > 0) &&
                       (mp->PorousMediaType == POROUS_BRINKMAN));
  int nonmoving_model =
      (gn->ConstitutiveEquation == BINGHAM || gn->ConstitutiveEquation == BINGHAM_WLF ||
       gn->ConstitutiveEquation == CARREAU || gn->ConstitutiveEquation == CARREAU_WLF);
  int movwall_model =
      (gn->ConstitutiveEquation == BINGHAM || gn->ConstitutiveEquation == BINGHAM_WLF ||
       gn->ConstitutiveEquation == CARREAU || gn->ConstitutiveEquation == CARREAU_WLF ||
       gn->ConstitutiveEquation == POWER_LAW || gn->ConstitutiveEquation == HERSCHEL_BULKLEY);

  /* Calculate flow rate and average velocity with their sensitivities
   * depending on the lubrication model employed
   */

  /* Confined lubrication flow - Newtonian */
  /* The next else block is for film flow) */
  VAR = FILL;
  if ((EQN == R_LUBP) || (EQN == R_LUBP_2)) {

    /* Set proper fill variable first.   If in lub_p layer, then use FILL,
     * but if in LUBP_2 layer use PHASE1.  This will have to be made more general
     * if you wanted to do both LS phases and R_phaseN fields in same layer.
     * We will leave that to the next sucker to develop
     */

    if (EQN == R_LUBP_2) {
      VAR = PHASE1;
    }

    /***** INITIALIZE LUBRICATION COMPONENTS AND LOAD IN VARIABLES *****/

    /* Setup lubrication shell constructs */
    dbl wt_old = fv->wt;
    int dof_map[MDE];
    int *n_dof = (int *)array_alloc(1, MAX_VARIABLE_TYPES, sizeof(int));
    lubrication_shell_initialize(n_dof, dof_map, -1, xi, exo, 0);

    /* Load viscosity and density */
    rho = density(d_rho, time);
    if (movwall_model || nonmoving_model) {
      mu = gn->mu0;
    } else {
      mu = ad_viscosity(gn, NULL); // This viscosity has already been modulated by H(F), fyi.
    }

    /* Extract wall velocities */
    velocity_function_model(veloU, veloL, time, dt);

    /* Extract wall heights */
    ADType H_U, dH_U_dtime, H_L, dH_L_dtime;
    ADType dH_U_dX[DIM], dH_L_dX[DIM], dH_U_dp, dH_U_ddh, dH_dF[MDE];
    H = ad_height_function_model(&H_U, &dH_U_dtime, &H_L, &dH_L_dtime, dH_U_dX, dH_L_dX, &dH_U_dp,
                                 &dH_U_ddh, dH_dF, time, dt);

    /***** DEFORM HEIGHT AND CALCULATE SENSITIVITIES *****/


    /* Deform height */
    switch (mp->FSIModel) {
    case FSI_MESH_CONTINUUM:
    case FSI_MESH_UNDEF:
    case FSI_SHELL_ONLY_UNDEF:
      for (i = 0; i < dim; i++) {
        H -= fv->snormal[i] * fv->d[i];
      }
      break;
    case FSI_SHELL_ONLY_MESH:
      if (pd->e[pg->imtrx][R_SHELL_NORMAL1] && pd->e[pg->imtrx][R_SHELL_NORMAL2] &&
          pd->e[pg->imtrx][R_SHELL_NORMAL3]) {
        for (i = 0; i < dim; i++) {
          H -= fv->n[i] * fv->d[i];
        }
      } else {
        for (i = 0; i < dim; i++) {
          H -= fv->snormal[i] * fv->d[i];
        }
      }
      break;
    case FSI_REALSOLID_CONTINUUM:
      for (i = 0; i < dim; i++) {
        H -= fv->snormal[i] * fv->d_rs[i];
      }
      break;
    }

    /* Calculate height sensitivity to mesh */

    /***** CALCULATE PRESSURE GRADIENT AND SENSITIVITIES *****/

    /* Define variables */
    ADType GRADP[DIM];

    if (EQN == R_LUBP) {
      ADInn(ad_fv->grad_lubp, GRADP);
    } else {
      GOMA_EH(GOMA_ERROR, "AD: Not a supported lubrication pressure equation");
    }

    /***** CALCULATE HEAVISIDE GRADIENT AND SENSITIVITIES *****/

    /* Define variables */
    dbl GRADH[DIM];
    dbl D_GRADH_DF[DIM][MDE], D_GRADH_DX[DIM][DIM][MDE];
    memset(GRADH, 0.0, sizeof(double) * DIM);
    memset(D_GRADH_DF, 0.0, sizeof(double) * DIM * MDE);
    memset(D_GRADH_DX, 0.0, sizeof(double) * DIM * DIM * MDE);

    /* Rotate and calculate mesh sensitivity */
    dbl d_grad_Hside_dmx[DIM][DIM][MDE];
    memset(d_grad_Hside_dmx, 0.0, sizeof(double) * DIM * DIM * MDE);
    if (pd->v[pg->imtrx][VAR]) {
      if (mp->Lub_Curv_NormalModel) {
        load_lsi(ls->Length_Scale);
        if (!mp->Lub_Curv_Modulation || lsi->near) {
          load_lsi_derivs();
          for (i = 0; i < dim; i++) {
            for (j = 0; j < dim; j++) {
              for (k = 0; k < ei[pg->imtrx]->dof[MESH_DISPLACEMENT1]; k++) {
                jk = dof_map[k];
                d_grad_Hside_dmx[i][j][jk] = lsi->d_gradHn_dmesh[i][j][k];
              }
            }
          }
          ShellRotate(lsi->gradHn, d_grad_Hside_dmx, GRADH, D_GRADH_DX, n_dof[MESH_DISPLACEMENT1]);

          /* Calculate F sensitivity */
          for (i = 0; i < dim; i++) {
            for (j = 0; j < dim; j++) {
              for (k = 0; k < ei[pg->imtrx]->dof[VAR]; k++) {
                D_GRADH_DF[i][k] += lsi->d_gradHn_dF[j][k] * delta(i, j);
                D_GRADH_DF[i][k] -= lsi->d_gradHn_dF[j][k] * fv->snormal[i] * fv->snormal[j];
              }
            }
          }
        }
      } else {
        double deltan[DIM];
        load_lsi(ls->Length_Scale);
        if (!mp->Lub_Curv_Modulation || lsi->near) {
          load_lsi_derivs();
          for (i = 0; i < dim; i++) {
            deltan[i] = lsi->delta * lsi->normal[i];
            for (j = 0; j < dim; j++) {
              for (k = 0; k < ei[pg->imtrx]->dof[MESH_DISPLACEMENT1]; k++) {
                jk = dof_map[k];
                d_grad_Hside_dmx[i][j][jk] = lsi->d_delta_dmesh[j][k] * lsi->normal[i] +
                                             lsi->delta * lsi->d_normal_dmesh[i][j][k];
              }
            }
          }
          ShellRotate(deltan, d_grad_Hside_dmx, GRADH, D_GRADH_DX, n_dof[MESH_DISPLACEMENT1]);

          /* Calculate F sensitivity */
          for (i = 0; i < dim; i++) {
            for (j = 0; j < dim; j++) {
              for (k = 0; k < ei[pg->imtrx]->dof[VAR]; k++) {
                D_GRADH_DF[i][k] +=
                    (lsi->delta * lsi->d_normal_dF[i][k] + lsi->d_delta_dF[k] * lsi->normal[i]) *
                    (delta(i, j) - fv->snormal[i] * fv->snormal[j]);
              }
            }
          }
        }
      }
    }

    /***** CALCULATE CURVATURE AND SENSITIVITIES *****/

    /* Define variables */
    ADType CURV = 0.0;
    ADType H_cap;

    if (mp->HeightUFunctionModel == WALL_DISTMOD || mp->HeightUFunctionModel == WALL_DISTURB) {
      H_cap = std::max(H_U - H_L, DBL_SEMI_SMALL);
      GOMA_WH(GOMA_ERROR, "Lubrication Wall Effect assumes constant capillary height for now...");
    } else {
      H_cap = H;
    }
    /* Curvature - analytic in the "z" direction  */
    dbl dcaU, dcaL, cos_dcaU, cos_dcaL;
    ADType slopeU, slopeL;
    dcaU = dcaL = 0.5 * M_PIE;
    cos_dcaU = cos_dcaL = 0;
    if (pd->gv[VAR]) {
      double d_dcaU_dV, d_dcaL_dV;
      double V = 0;
      if (mp->DcaUFunctionModel == CONSTANT && mp->DcaLFunctionModel == CONSTANT) {
        dcaU = mp->dcaU * M_PIE / 180.0;
        dcaL = mp->dcaL * M_PIE / 180.0;
      } else {
        /* Connecting up the DCA model routine ... no point in V-dependence at the moment*/
        dynamic_contact_angle_model(&cos_dcaU, &cos_dcaL, V, &d_dcaU_dV, &d_dcaL_dV, &dcaU, &dcaL);
      }
      slopeU = slopeL = 0.;
      for (i = 0; i < dim; i++) {
        slopeU += dH_U_dX[i] * ad_lsi->normal[i];
        slopeL += dH_L_dX[i] * ad_lsi->normal[i];
      }
      /*  Positive sign for convex meniscus, negative for concave meniscus,
            this sign convention is opposite of generally accepted one for curvature
          i.e., 2H = grad-dot-normal_vector vs. 2H = -grad-dot-normal_vector       */
      ADType ad_cos_dcaU = cos(dcaU + atan(slopeU));
      ADType ad_cos_dcaL = cos(dcaL + atan(-slopeL));
      CURV = -(ad_cos_dcaU + ad_cos_dcaL) / H_cap;
      AD_LubAux->op_curv = CURV;

      /* Curvature - numerical in planview direction */
      if (mp->Lub_Curv_Combine && pd->e[pg->imtrx][SHELL_LUB_CURV]) {
        CURV = fv->sh_l_curv;
      } else if (pd->e[pg->imtrx][SHELL_LUB_CURV]) {
        CURV += fv->sh_l_curv;
      }
      if (pd->e[pg->imtrx][SHELL_LUB_CURV_2]) {
        CURV += fv->sh_l_curv_2;
      }
      if (pd->e[pg->imtrx][NORMAL1]) {
        CURV += fv->div_n;
      }
    }

    /***** CALCULATE GRAVITY AND LORENTZ (OTHER lubmomsource) / BODY FORCE AND SENSITIVITIES *****/

    /* Define variables */
    dbl bodf[DIM], GRAV[DIM];

    dbl D_GRAV_DF[DIM][MDE], D_GRAV_DX[DIM][DIM][MDE];
    memset(bodf, 0.0, sizeof(double) * DIM);
    memset(GRAV, 0.0, sizeof(double) * DIM);
    memset(D_GRAV_DF, 0.0, sizeof(double) * DIM * MDE);
    memset(D_GRAV_DX, 0.0, sizeof(double) * DIM * DIM * MDE);

    /* Calculate and rotate body force, calculate mesh derivatives */
    dbl Bouss[DIM];
    memset(Bouss, 0.0, sizeof(double) * DIM);
    MOMENTUM_SOURCE_DEPENDENCE_STRUCT dBouss_struct; /* Body force dependence */
    MOMENTUM_SOURCE_DEPENDENCE_STRUCT *dBouss = &dBouss_struct;

    /* Calculate and rotate body force, calculate mesh derivatives */
    dbl d_bodf_dmx[DIM][DIM][MDE];
    memset(d_bodf_dmx, 0.0, sizeof(double) * DIM * DIM * MDE);
    for (i = 0; i < dim; i++) {
      bodf[i] = mp->momentum_source[i] * rho.val();
    }

    ShellRotate(bodf, d_bodf_dmx, GRAV, D_GRAV_DX, n_dof[MESH_DISPLACEMENT1]);

    /* Sensitivity to level set F, then rotate */
    dbl d_bodf_df[DIM][MDE];
    for (i = 0; i < dim; i++) {
      for (j = 0; j < ei[pg->imtrx]->dof[VAR]; j++) {
        d_bodf_df[i][j] = mp->momentum_source[i] * d_rho->F[j];
      }
    }
    for (k = 0; k < ei[pg->imtrx]->dof[VAR]; k++) {
      for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
          D_GRAV_DF[i][k] += d_bodf_df[j][k] * delta(i, j);
          D_GRAV_DF[i][k] -= d_bodf_df[j][k] * fv->snormal[i] * fv->snormal[j];
        }
      }
    }

    /* Sensitivity to species if buoyancy force matters */
    if (mp->MomentumSourceModel == BOUSSINESQ) {
      err = bouss_momentum_source(Bouss, dBouss, 0, TRUE);
      GOMA_EH(err, "Problems in bouss_momentum_source");
    }

    /***** CALCULATE CONVECTIVE (Inertial) TERMS if available *****/
    ADType convf[DIM], CONV[DIM];
    double D_CONV_DF[DIM][MDE], D_CONV_DX[DIM][DIM][MDE], D_CONV_DV[DIM][DIM][MDE];
    double d_conv_df[DIM][MDE], d_conv_dx[DIM][DIM][MDE];
    memset(convf, 0.0, sizeof(double) * DIM);
    memset(CONV, 0.0, sizeof(double) * DIM);
    memset(d_conv_df, 0.0, sizeof(double) * DIM * MDE);
    memset(D_CONV_DF, 0.0, sizeof(double) * DIM * MDE);
    memset(D_CONV_DX, 0.0, sizeof(double) * DIM * DIM * MDE);
    memset(d_conv_dx, 0.0, sizeof(double) * DIM * DIM * MDE);
    memset(D_CONV_DV, 0.0, sizeof(double) * DIM * DIM * MDE);

    if (do_convection) {
      for (i = 0; i < dim; i++) {
        for (j = 0; j < VIM; j++) {
          convf[i] += rho * ad_fv->v[j] * ad_fv->grad_v[j][i];
        }
      }
      ADInn(convf, CONV);
    }
    /********** PREPARE VISCOSITY DERIVATIVES **********/
    dbl D_MU_DX[DIM][MDE];
    memset(D_MU_DX, 0.0, sizeof(double) * DIM * MDE);
    for (i = 0; i < dim; i++) {
      for (k = 0; k < ei[pg->imtrx]->dof[MESH_DISPLACEMENT1]; k++) {
        jk = dof_map[k];
        D_MU_DX[i][jk] = d_mu->X[i][k];
      }
    }
    if (pd->v[pg->imtrx][SHELL_PARTC]) {
      dmu_dc = mp->d_viscosity[SHELL_PARTC];
    }

    /********** CALCULATE FLOW RATE AND AVERAGE VELOCITY **********/

    /* Set some coefficients */
    ADType k_turb = 12.;
    ADType vsqr, q_mag = 0., v_mag = 0., tau_w, vis_w = 1., pre_delP = 0., vpre_delP = 0.;
    ADType  H_inv = 1. / H;
    ADType factor, ratio = 0., q_mag2;
    ADType q[DIM], ev[DIM], pgrad, pg_cmp[DIM], dev_dpg[DIM][DIM];
    ADType v_avg[DIM];
    int movingwall = FALSE;

    for (i = 0; i < dim; i++) {
      pg_cmp[i] = GRADP[i] - GRAV[i] - Bouss[i] + CONV[i];
      if (pd->gv[VAR] && !isnan(CURV.val())) {
        pg_cmp[i] += GRADH[i] * CURV * mp->surface_tension;
        // DGRADP_DK += GRADH[i] * mp->surface_tension;
      }
    }
    pgrad = 0.;
    vsqr = 0.;
    for (i = 0; i < dim; i++) {
      pgrad += SQUARE(pg_cmp[i]);
      vsqr += SQUARE(veloL[i]);
      vsqr += SQUARE(veloU[i]);
    }
    movingwall = DOUBLE_NONZERO(vsqr);
    pgrad = sqrt(pgrad);
    if (pgrad > DBL_SEMI_SMALL) {
      for (i = 0; i < dim; i++) {
        ev[i] = pg_cmp[i] / pgrad;
      }
    } else {
      ev[0] = 1.;
    }
    dev_dpg[0][0] = ev[1] * ev[1] + ev[2] * ev[2];
    dev_dpg[1][1] = ev[0] * ev[0] + ev[2] * ev[2];
    dev_dpg[2][2] = ev[1] * ev[1] + ev[0] * ev[0];
    dev_dpg[0][1] = dev_dpg[1][0] = -ev[0] * ev[1];
    dev_dpg[0][2] = dev_dpg[2][0] = -ev[0] * ev[2];
    dev_dpg[1][2] = dev_dpg[2][1] = -ev[1] * ev[2];
    ADType mu_diss = 0.;

    tau_w = 0.5 * H * pgrad;
    if (!movingwall) {
      /*  First non-Newtonian models with analytical viscosity integration */
      if (gn->ConstitutiveEquation == POWER_LAW) {
        double nexp = gn->nexp;
        k_turb = 4. * (2. + 1. / nexp);
        tau_w = MAX(tau_w, DBL_SEMI_SMALL);
        q_mag = -2. * SQUARE(H) / k_turb * pow(tau_w / mu, 1. / nexp);
        pre_delP = -CUBE(H) / (k_turb * mu) * pow(tau_w / mu, 1. / nexp - 1.);
        srate = pow(fabs(tau_w) / mu, 1. / nexp);
        vis_w = tau_w / srate;
      } else if (gn->ConstitutiveEquation == HERSCHEL_BULKLEY) {
        double nexp = gn->nexp, yield = gn->tau_y, eps_rate = 0.000001;
        if (tau_w - yield > 0.) {
          ADType f, f_c, f_term, f_termd;
          f = yield / tau_w;
          f_c = 1. - f;
          f_term = 1. - SQUARE(f) - SQUARE(f_c) / (2. * nexp + 1.) - 2. * f * f_c / (2. + nexp);
          f_termd = SQUARE(f) + f * f_c / (2 * nexp + 1.) + f * (1. - 2 * f) / (2 + nexp);
          srate = pow((tau_w - yield) / mu, 1. / nexp);
          vis_w = tau_w / srate;
          q_mag = -0.25 * SQUARE(H) * srate * f_term;
          pre_delP = 0.5 * H * q_mag / tau_w;
        } else {
          srate = 0.;
          vis_w = yield / eps_rate;
          q_mag = 0.;
          pre_delP = 0.;
        }
        /*  Next  non-Newtonian models with numerical viscosity integration */
      } else if (nonmoving_model) {
        if (isnan(tau_w.val()))
          GOMA_WH(GOMA_ERROR, "Trouble, tau_w is nan...\n");
        err = lub_viscosity_integrate(tau_w, H, &q_mag, &dq_gradp, &dq_dH, &srate, &pre_delP,
                                      &vis_w, &dq_dT, &dq_dshrw);
        if (isnan(srate.val()))
          DPRINTF(stderr, "lub_srate isnan %g %g %g %g\n", tau_w, q_mag, srate, vis_w);
        if (err < 0) {
          GOMA_WH(GOMA_ERROR, "Some trouble with Numerical Lubrication...\n");
        }
      } else { /*  Newtonian type models - nonmoving wall part  */
        k_turb = 12.;
        q_mag = pre_delP * pgrad;
        srate = fabs(tau_w / mu);
        vis_w = mu;
      }
      /* modulate q (stationary wall part) if level-set interface present
         Newtonian models are modulated through the viscosity functions,
         so need to exclude those                                       */
      if (pd->v[pg->imtrx][VAR] && (nonmoving_model || gn->ConstitutiveEquation == POWER_LAW ||
                                    gn->ConstitutiveEquation == HERSCHEL_BULKLEY)) {
        ad_load_lsi(ls->Length_Scale);
        if (mp->mp2nd->ViscosityModel == RATIO) {
          ratio = 1. / mp->mp2nd->viscosity; /* Assuming model = RATIO for now */
          q_mag2 = q_mag * ratio;
          q_mag = ls_modulate_property(
              q_mag, q_mag2, ls->Length_Scale, (double)mp->mp2nd->viscositymask[0],
              (double)mp->mp2nd->viscositymask[1], dqmag_dF, &factor, LSI_INTERP_LINEAR);
          factor *= (1. - ratio);
          factor += ratio;
          dq_gradp *= factor;
          dq_dH *= factor;
          dq_dT *= factor;
          pre_delP *= factor;
          vis_w /= factor;
        } else if (mp->mp2nd->ViscosityModel == CONSTANT ||
                   mp->mp2nd->ViscosityModel == TIME_RAMP) {
          if (mp->Lub_LS_Interpolation == LOGARITHMIC) {
            if (lsi->near || (ad_fv->F > 0 && mp->mp2nd->viscositymask[1]) ||
                (ad_fv->F < 0 && mp->mp2nd->viscositymask[0])) {
              double dq_gradp2, pre_delP2, dq_dH2, srate2, qmag_log;
              k_turb = 12.;
              dq_gradp2 = pre_delP2 = -CUBE(H) / (k_turb * mp->mp2nd->viscosity);
              q_mag2 = pre_delP2 * pgrad;
              dq_dH2 = -3. * SQUARE(H) / (k_turb * mp->mp2nd->viscosity) * pgrad;
              srate2 = tau_w / mp->mp2nd->viscosity;
              qmag_log = (DOUBLE_NONZERO(q_mag) ? log(q_mag2 / q_mag) : 0.0);
              if (!lsi->near) {
                q_mag = q_mag2;
                dq_gradp = dq_gradp2;
                pre_delP = pre_delP2;
                dq_dH = dq_dH2;
                srate = srate2;
                vis_w = mp->mp2nd->viscosity;
              } else {
                double dfact_sign = (mp->mp2nd->viscositymask[1] ? -1.0 : 1.0);
                factor = (mp->mp2nd->viscositymask[1] ? (1.0 - lsi->H) : lsi->H);
                q_mag = -pow(-q_mag, factor) * pow(-q_mag2, 1.0 - factor);
                dq_gradp = -pow(-dq_gradp, factor) * pow(-dq_gradp2, 1.0 - factor);
                pre_delP = -pow(-pre_delP, factor) * pow(-pre_delP2, 1.0 - factor);
                dq_dH = -pow(-dq_dH, factor) * pow(-dq_dH2, 1.0 - factor);
                srate = pow(srate, factor) * pow(srate2, 1.0 - factor);
                vis_w = pow(vis_w, factor) * pow(mp->mp2nd->viscosity, 1.0 - factor);
                for (j = 0; j < ei[pg->imtrx]->dof[VAR]; j++) {
                  dqmag_dF[j] += q_mag * qmag_log * (-dfact_sign) * lsi->d_H_dF[j];
                }
              }
            }
          } else if (mp->Lub_LS_Interpolation == LINEAR) {
            double dq_gradp2, pre_delP2, dq_dH2, srate2;
            k_turb = 12.;
            dq_gradp2 = pre_delP2 = -CUBE(H) / (k_turb * mp->mp2nd->viscosity);
            q_mag2 = pre_delP2 * pgrad;
            dq_dH2 = -3. * SQUARE(H) / (k_turb * mp->mp2nd->viscosity) * pgrad;
            srate2 = tau_w / mp->mp2nd->viscosity;
            q_mag = ls_modulate_property(
                q_mag, q_mag2, ls->Length_Scale, (double)mp->mp2nd->viscositymask[0],
                (double)mp->mp2nd->viscositymask[1], dqmag_dF, &factor, LSI_INTERP_LINEAR);
            dq_gradp = dq_gradp * factor + dq_gradp2 * (1. - factor);
            pre_delP = pre_delP * factor + pre_delP2 * (1. - factor);
            dq_dH = dq_dH * factor + (1. - factor) * dq_dH2;
            dq_dT *= factor; // mp2nd->viscosity is independent of Temperature
            srate = srate * factor + (1. - factor) * srate2;
            vis_w = vis_w * factor + (1. - factor) * mp->mp2nd->viscosity;
          } else {
            GOMA_WH(GOMA_ERROR, "mp->Lub_LS_Interpolation needs to be LOG or LINEAR...\n");
          }
        } else {
          GOMA_WH(GOMA_ERROR, "mp2nd->ViscosityModel needs to be RATIO or CONSTANT...\n");
        }
      }
      if (pd->v[pg->imtrx][SHELL_TEMPERATURE]) {
        mu_diss = -q_mag * pgrad;
        dmu_diss_dT = -dq_dT * pgrad;
        dmu_diss_dpgrad = -q_mag - pgrad * dq_gradp;
      }
      memset(q, 0.0, sizeof(double) * DIM);
      for (i = 0; i < dim; i++) {
        q[i] += q_mag * ev[i];
      }
      v_mag = q_mag * H_inv;
      dv_gradp = dq_gradp * H_inv;
      dv_dH = dq_dH * H_inv - q_mag * SQUARE(H_inv);
      vpre_delP = pre_delP / H;
      /* Convert to more general nomenclature  */
      if (pd->v[pg->imtrx][SHELL_SHEAR_TOP]) {
        for (i = 0; i < dim; i++) {
          D_Q_DSHRW[i] = dq_dshrw * ev[i];
          D_V_DSHRW[i] = H_inv * dq_dshrw * ev[i];
          D_Q_DH[i] = dq_dH * ev[i];
          D_V_DH[i] = dq_dH * ev[i] * H_inv - q[i] * SQUARE(H_inv);
        }
      } else {
        for (i = 0; i < dim; i++) {
          for (j = 0; j < dim; j++) {
            D_Q_DGRADP[i][j] = dq_gradp * ev[i] * ev[j] + pre_delP * dev_dpg[i][j];
          }
          D_Q_DH[i] = dq_dH * ev[i];
          D_V_DH[i] = dq_dH * ev[i] * H_inv - q[i] * SQUARE(H_inv);
        }
      }
      /* moving wall parts  */
    } else {
      if (movwall_model) { /*  non-Newtonian models with a moving wall */
        double wstrs, relax = 0.5;
        int guess = 0;
        err = lub2D_flow2D(pg_cmp, ev, dev_dpg, q, D_Q_DGRADP, DQ_DH, H, &srate, veloL, veloU,
                           guess, &wstrs, relax);
        if (isnan(srate))
          DPRINTF(stderr, "lub_srate isnan %g %g %g %g\n", wstrs, q[0], q[1], srate);
        if (err < 0) {
          GOMA_WH(GOMA_ERROR, "Some trouble with Numerical Lubrication...\n");
        }
      } else { /*  moving wall part of Newtonian type models */
        k_turb = 12.;
        dq_gradp = pre_delP = -CUBE(H) / (k_turb * mu);
        q_mag = pre_delP * pgrad;
        dq_dH = -3. * SQUARE(H) / (k_turb * mu) * pgrad;
        srate = fabs(tau_w / mu);
        vis_w = mu;
        memset(q, 0.0, sizeof(double) * DIM);
        for (i = 0; i < dim; i++) {
          q[i] += q_mag * ev[i];
          q[i] += 0.5 * H * (veloL[i] + veloU[i]);
        }
        for (j = 0; j < ei[pg->imtrx]->dof[VAR]; j++) {
          dqmag_dF[j] += q_mag * (-d_k_turb_dmu * dmu_df[j] / k_turb - dmu_df[j] / mu);
        }
        /*  Only temperature-dependent Newtonian type model is THERMAL  */
        if (gn->ConstitutiveEquation == THERMAL || gn->ConstitutiveEquation == TABLE) {
          dq_dT = q_mag * (-dmu_dT / mu);
        }
        /* Convert to more general nomenclature  */
        for (i = 0; i < dim; i++) {
          for (j = 0; j < dim; j++) {
            D_Q_DGRADP[i][j] = dq_gradp * ev[i] * ev[j] + pre_delP * dev_dpg[i][j];
          }
          DQ_DH[i] = dq_dH * ev[i];
          DQ_DH[i] += 0.5 * (veloL[i] + veloU[i]);
        }
        v_mag = q_mag * H_inv;
        dv_gradp = dq_gradp * H_inv;
        dv_dH = dq_dH * H_inv - q_mag * SQUARE(H_inv);
        vpre_delP = pre_delP * H_inv;
      } /*  End of Viscosity Models **/

      /* modulate q (moving wall part) if level-set interface present */
      /* Newtonian models are modulated through the viscosity function*/
      if (pd->v[pg->imtrx][VAR] && movwall_model) {
        for (i = 0; i < dim; i++) {
          q_mag += q[i] * ev[i];
        } /** Retrieve pressure part for Non-Newtonian   */
        /*  q_mag is a sort of dummy variable as we apply the factor to whole array */
        if (mp->mp2nd->ViscosityModel == RATIO) {
          ratio = 1. / mp->mp2nd->viscosity; /* Assuming model = RATIO for now */
          q_mag2 = q_mag * ratio;
          q_mag = ls_modulate_property(
              q_mag, q_mag2, ls->Length_Scale, (double)mp->mp2nd->viscositymask[0],
              (double)mp->mp2nd->viscositymask[1], dqmag_dF, &factor, LSI_INTERP_LINEAR);
          factor *= (1. - ratio);
          factor += ratio;
          /* Possibly lots different here since dq_gradp not really used for moving wall */
          for (i = 0; i < dim; i++) {
            for (j = 0; j < dim; j++) {
              D_Q_DGRADP[i][j] *= factor;
            }
            DQ_DH[i] *= factor;
            q[i] *= factor;
          }
        } else if (mp->mp2nd->ViscosityModel == CONSTANT ||
                   mp->mp2nd->ViscosityModel == TIME_RAMP) {
          if (mp->Lub_LS_Interpolation == LOGARITHMIC) {
            if (lsi->near || (fv->F > 0 && mp->mp2nd->viscositymask[1]) ||
                (fv->F < 0 && mp->mp2nd->viscositymask[0])) {
              double dq_gradp2, pre_delP2, srate2, qmag_log;
              k_turb = 12.;
              dq_gradp2 = pre_delP2 = -CUBE(H) / (k_turb * mp->mp2nd->viscosity);
              srate2 = tau_w / mp->mp2nd->viscosity;
              if (fabs(fv->F) > 0.5 * ls->Length_Scale) {
                srate = srate2;
                vis_w = mp->mp2nd->viscosity;
                for (i = 0; i < dim; i++) {
                  for (j = 0; j < dim; j++) {
                    D_Q_DGRADP[i][j] = delta(i, j) * dq_gradp2;
                  }
                  DQ_DH[i] = 0.5 * (veloU[i] + veloL[i]) + (3. * H_inv * dq_gradp2) * pg_cmp[i];
                  q[i] = 0.5 * H * (veloU[i] + veloL[i]) + dq_gradp2 * pg_cmp[i];
                }
              } else {
                double dfact_sign = (mp->mp2nd->viscositymask[1] ? -1.0 : 1.0);
                factor = (mp->mp2nd->viscositymask[1] ? (1.0 - lsi->H) : lsi->H);
                /* Try modulating diagonal components only, since gas side is diagonal */
                for (i = 0; i < dim; i++) {
                  double q_gas, q_gas_dH, q_liq;
                  q_gas = 0.5 * H * (veloU[i] + veloL[i]) + dq_gradp2 * pg_cmp[i];
                  q_gas_dH = 0.5 * (veloU[i] + veloL[i]) + 3. * H_inv * dq_gradp2 * pg_cmp[i];
                  q_liq = q[i];
                  qmag_log = (DOUBLE_NONZERO(q_liq) ? log(q_gas / q_liq) : 0.0);
                  D_Q_DGRADP[i][i] = SGN(D_Q_DGRADP[i][i]) * pow(ABS(D_Q_DGRADP[i][i]), factor) *
                                     pow(ABS(dq_gradp2), 1.0 - factor);
                  DQ_DH[i] =
                      SGN(DQ_DH[i]) * pow(ABS(DQ_DH[i]), factor) * pow(ABS(q_gas_dH), 1.0 - factor);
                  q[i] = SGN(q_liq) * pow(ABS(q_liq), factor) * pow(ABS(q_gas), 1.0 - factor);
                  for (j = 0; j < ei[pg->imtrx]->dof[VAR]; j++) {
                    D_Q_DF[i][j] = q[i] * qmag_log * (-dfact_sign) * lsi->d_H_dF[j];
                  }
                }
                srate = pow(srate, factor) * pow(srate2, 1.0 - factor);
                vis_w = pow(vis_w, factor) * pow(mp->mp2nd->viscosity, 1.0 - factor);
              }
            }
          } else if (mp->Lub_LS_Interpolation == LINEAR) {
            double dq_gradp2, pre_delP2, srate2;
            double dfact_sign = (mp->mp2nd->viscositymask[1] ? -1.0 : 1.0);
            k_turb = 12.;
            dq_gradp2 = pre_delP2 = -CUBE(H) / (k_turb * mp->mp2nd->viscosity);
            q_mag2 = pre_delP2 * pgrad;
            srate2 = tau_w / mp->mp2nd->viscosity;
            q_mag = ls_modulate_property(
                q_mag, q_mag2, ls->Length_Scale, (double)mp->mp2nd->viscositymask[0],
                (double)mp->mp2nd->viscositymask[1], dqmag_dF, &factor, LSI_INTERP_LINEAR);
            for (i = 0; i < dim; i++) {
              double q_gas, q_gas_dH, q_liq;
              q_gas = 0.5 * H * (veloU[i] + veloL[i]) + dq_gradp2 * pg_cmp[i];
              q_gas_dH = 0.5 * (veloU[i] + veloL[i]) + 3. * H_inv * dq_gradp2 * pg_cmp[i];
              q_liq = q[i];
              for (j = 0; j < dim; j++) {
                D_Q_DGRADP[i][j] *= factor;
                D_Q_DGRADP[i][j] += (1. - factor) * delta(i, j) * dq_gradp2;
              }
              DQ_DH[i] *= factor;
              DQ_DH[i] += (1. - factor) * q_gas_dH;
              for (j = 0; j < ei[pg->imtrx]->dof[VAR]; j++) {
                D_Q_DF[i][j] = (q_liq - q_gas) * (-dfact_sign) * lsi->d_H_dF[j];
              }
              q[i] = q_liq * factor + (1. - factor) * q_gas;
            }
            dq_dT *= factor; // mp2nd->viscosity is independent of Temperature
            srate = srate * factor + (1. - factor) * srate2;
            vis_w = vis_w * factor + (1. - factor) * mp->mp2nd->viscosity;
          } else {
            GOMA_WH(GOMA_ERROR, "mp->Lub_LS_Interpolation needs to be LOG or LINEAR...\n");
          }
        } else {
          GOMA_WH(GOMA_ERROR, "mp2nd->ViscosityModel needs to be RATIO or CONSTANT...\n");
        }
      } /*  end of if (LS) block  */
      if (pd->gv[SHELL_TEMPERATURE]) {
        mu_diss = -q_mag * pgrad; /* Need to add the drag flow part yet */
      }

    } /* End of moving wall part  */

    for (i = 0; i < dim; i++) {
      v_avg[i] = q[i] * H_inv;
    }


    AD_LubAux->H = H;
    AD_LubAux->H_cap = H_cap;
    AD_LubAux->gradP_mag = 0;
    AD_LubAux->srate = srate;
    AD_LubAux->mu_star = vis_w;
    AD_LubAux->visc_diss = mu_diss;
    for (i = 0; i < dim; i++) {
      AD_LubAux->gradP[i] = pg_cmp[i];
      AD_LubAux->q[i] = q[i];
      AD_LubAux->v_avg[i] = v_avg[i];
      AD_LubAux->gradP_mag += SQUARE(pg_cmp[i]);

    }
    if (do_convection) {
      for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
          for (k = 0; k < ei[pg->imtrx]->dof[VELOCITY1]; k++) {
            AD_LubAux->dq_dv[i][j][k] = D_Q_DV[i][j][k];
          }
        }
      }
    }
    AD_LubAux->gradP_mag = sqrt(AD_LubAux->gradP_mag);

    // Cleanup
    fv->wt = wt_old;
    safe_free((void *)n_dof);

  } else if (EQN == R_SHELL_FILMP) {
    GOMA_EH(GOMA_ERROR, "Lubrication film pressure equation not yet implemented in AD.\n");
  }

  return;

} /* End of calculate_lub_q_v */
int ad_assemble_lubrication(const int EQN,  /* equation type: either R_LUBP or R_LUBP2 */
                            double time,    /* present time value */
                            double tt,      /* parameter to vary time integration from
                                             * explicit (tt = 1) to implicit (tt = 0)    */
                            double dt,      /* current time step size */
                            double xi[DIM], /* Local stu coordinates */
                            const Exo_DB *exo) {
  int eqn, var, peqn, pvar, p, q, a, b, k, jk, w;
  int i = -1, j, status; //, err;
  int *n_dof = NULL;
  int dof_map[MDE];

  // dbl toggle_dh_dependence = 0.;

  dbl H, dH_dtime;
  dbl H_U, dH_U_dtime, H_L, dH_L_dtime;
  dbl dH_U_dX[DIM], dH_L_dX[DIM], dH_dtime_dmesh[DIM][MDE];
  dbl dH_dtime_drealsolid[DIM][MDE];
  dbl dH_dtime_dnormal[DIM][MDE];
  dbl dH_U_dp, dH_U_ddh, dH_dF[MDE];
  dbl veloU[DIM], veloL[DIM];
  dbl diffusion, source;

  /*
   * Basis functions and derivatives
   */
  dbl phi_i, grad_phi_i[DIM], grad_II_phi_i[DIM], d_grad_II_phi_i_dmesh[DIM][DIM][MDE];
  dbl phi_j, grad_phi_j[DIM], grad_II_phi_j[DIM], d_grad_II_phi_j_dmesh[DIM][DIM][MDE];

  /*
   * Bail out fast if there's nothing to do...
   */
  status = 0;
  eqn = EQN;
  if (!pd->e[pg->imtrx][eqn])
    return (status);

  /*
   * Load Gauss point weights before looking for friends
   */
  dbl dim = pd->Num_Dim;
  dbl wt = fv->wt;
  dbl h3 = fv->h3;

  /*
   * Prepare geometry
   */
  n_dof = (int *)array_alloc(1, MAX_VARIABLE_TYPES, sizeof(int));
  lubrication_shell_initialize(n_dof, dof_map, -1, xi, exo, 0);

  /* Load proper FEM weights */
  dbl det_J = fv->sdet;

  /* Load up source models -- momentum*/
  // err = load_lubrication_momentum_source(time, dt);

  /* Load up source models -- mass */
  // No calls yet as only constat models exist. See mm_input_mp.c

  int err = -1;
  dbl flux = 0.0;
  dbl d_flux[MAX_VARIABLE_TYPES][MDE];
  memset(d_flux, 0.0, sizeof(double) * MAX_VARIABLE_TYPES * MDE);
  err = lubrication_fluid_source(&flux, d_flux, n_dof);
  GOMA_EH(err, "Error in loading lubrication_fluid_source");

  /* Time settings */
  if (pd->TimeIntegration != TRANSIENT) {
    tt = -0.5;
    dt = 1.0;
  }

  /*  Calculate non-constant surface tension, if needed  */
  if (mp->SurfaceTensionModel != CONSTANT) {
    double dsigma_dx[DIM][MDE];
    load_surface_tension(dsigma_dx);
    if (neg_elem_volume)
      return (status);
  }

  /*** CALCULATE FLOW RATE FROM FUNCTION **************************************/
  calculate_lub_q_v(EQN, time, dt, xi, exo); // PRS: NEED TO DO SOMETHING HERE

  /*** CALCULATE PHYSICAL PROPERTIES AND SENSITIVITIES ************************/

  /* Lubrication height from model */
  H = height_function_model(&H_U, &dH_U_dtime, &H_L, &dH_L_dtime, dH_U_dX, dH_L_dX, &dH_U_dp,
                            &dH_U_ddh, dH_dF, time, dt);
  dH_dtime = dH_U_dtime - dH_L_dtime;
  /*
  if (pd->v[pg->imtrx][SHELL_DELTAH] &&
      (mp->HeightUFunctionModel == CONSTANT_SPEED_DEFORM ||
       mp->HeightUFunctionModel == CONSTANT_SPEED_MELT ||
       mp->HeightUFunctionModel == FLAT_GRAD_FLAT_MELT ||
       mp->HeightUFunctionModel == CIRCLE_MELT )) toggle_dh_dependence = 1.;
  */

  /* Deform lubrication height for FSI interaction */
  switch (mp->FSIModel) {
  case FSI_MESH_CONTINUUM:
  case FSI_MESH_UNDEF:
  case FSI_SHELL_ONLY_UNDEF:
    for (i = 0; i < dim; i++) {
      H -= fv->snormal[i] * fv->d[i];
      if (pd->TimeIntegration == TRANSIENT) {
        dH_dtime -= fv->snormal[i] * fv_dot->d[i];
      }
    }
    break;

  case FSI_SHELL_ONLY_MESH:
    if ((pd->e[pg->imtrx][R_SHELL_NORMAL1]) && (pd->e[pg->imtrx][R_SHELL_NORMAL2]) &&
        (pd->e[pg->imtrx][R_SHELL_NORMAL3])) {
      for (i = 0; i < dim; i++) {
        H -= fv->n[i] * fv->d[i];
        if (pd->TimeIntegration == TRANSIENT) {
          dH_dtime -= fv->n[i] * fv_dot->d[i] + fv_dot->n[i] * fv->d[i];
        }
      }
    } else {
      for (i = 0; i < dim; i++) {
        H -= fv->snormal[i] * fv->d[i];
        if (pd->TimeIntegration == TRANSIENT) {
          dH_dtime -= fv->snormal[i] * fv_dot->d[i];
        }
      }
    }
    break;

  case FSI_REALSOLID_CONTINUUM:
    for (i = 0; i < dim; i++) {
      H -= fv->snormal[i] * fv->d_rs[i];
      if (pd->TimeIntegration == TRANSIENT) {
        dH_dtime -= fv->snormal[i] * fv_dot->d_rs[i];
      }
    }
    break;
  }

  /* Check for negative lubrication height, if so, get out */
  if (H <= 0.0) {
    neg_lub_height = TRUE;

#ifdef PARALLEL
    fprintf(stderr, "\nP_%d: Lubrication height =  %e\n", ProcID, H);
#else
    fprintf(stderr, "\n Lubrication height =  %e\n", H);
#endif

    status = 2;
    return (status);
  }

  /* Lubrication wall velocity from model */
  velocity_function_model(veloU, veloL, time, dt);

  /* Lubrication height - mesh sensitivity */
  memset(dH_dtime_dmesh, 0.0, sizeof(double) * DIM * MDE);
  memset(dH_dtime_drealsolid, 0.0, sizeof(double) * DIM * MDE);
  switch (mp->FSIModel) {
  case FSI_MESH_CONTINUUM:
  case FSI_MESH_UNDEF:
  case FSI_SHELL_ONLY_UNDEF:
    for (i = 0; i < VIM; i++) {
      for (b = 0; b < dim; b++) {
        for (k = 0; k < ei[pg->imtrx]->dof[MESH_DISPLACEMENT1]; k++) {
          jk = dof_map[k];
          dH_dtime_dmesh[b][k] -= fv->dsnormal_dx[i][b][jk] * fv_dot->d[i];
          dH_dtime_dmesh[b][k] -=
              fv->snormal[i] * delta(i, b) * bf[MESH_DISPLACEMENT1]->phi[k] * (1 + 2 * tt) / dt;
        }
      }
    }
    break;
  case FSI_SHELL_ONLY_MESH:
    if ((pd->e[pg->imtrx][R_SHELL_NORMAL1]) && (pd->e[pg->imtrx][R_SHELL_NORMAL2]) &&
        (pd->e[pg->imtrx][R_SHELL_NORMAL3])) {
      for (i = 0; i < VIM; i++) {
        for (b = 0; b < dim; b++) {
          for (k = 0; k < ei[pg->imtrx]->dof[MESH_DISPLACEMENT1]; k++) {
            dH_dtime_dmesh[b][k] -=
                fv->n[i] * delta(i, b) * bf[MESH_DISPLACEMENT1]->phi[k] * (1 + 2 * tt) / dt;
            dH_dtime_dmesh[b][k] -= fv_dot->n[i] * delta(i, b) * bf[MESH_DISPLACEMENT1]->phi[k];
          }
        }
      }
    } else {
      for (i = 0; i < VIM; i++) {
        for (b = 0; b < dim; b++) {
          for (k = 0; k < ei[pg->imtrx]->dof[MESH_DISPLACEMENT1]; k++) {
            jk = dof_map[k];
            dH_dtime_dmesh[b][k] -= fv->dsnormal_dx[i][b][jk] * fv_dot->d[i];
            dH_dtime_dmesh[b][k] -=
                fv->snormal[i] * delta(i, b) * bf[MESH_DISPLACEMENT1]->phi[k] * (1 + 2 * tt) / dt;
          }
        }
      }
    }
    break;
  case FSI_REALSOLID_CONTINUUM:
    for (i = 0; i < VIM; i++) {
      for (b = 0; b < dim; b++) {
        for (k = 0; k < ei[pg->imtrx]->dof[MESH_DISPLACEMENT1]; k++) {
          jk = dof_map[k];
          dH_dtime_dmesh[b][k] -= fv->dsnormal_dx[i][b][jk] * fv_dot->d_rs[i];
        }
        for (k = 0; k < ei[pg->imtrx]->dof[SOLID_DISPLACEMENT1]; k++) {
          jk = dof_map[k];
          dH_dtime_drealsolid[b][k] -=
              fv->snormal[i] * delta(i, b) * bf[SOLID_DISPLACEMENT1]->phi[jk] * (1 + 2 * tt) / dt;
        }
      }
    }
    break;
  }

  /* Lubrication height - shell normal sensitivity */
  memset(dH_dtime_dnormal, 0.0, sizeof(double) * DIM * MDE);
  switch (mp->FSIModel) {

  case FSI_SHELL_ONLY_MESH:
    if ((pd->e[pg->imtrx][R_SHELL_NORMAL1]) && (pd->e[pg->imtrx][R_SHELL_NORMAL2]) &&
        (pd->e[pg->imtrx][R_SHELL_NORMAL3])) {
      for (i = 0; i < VIM; i++) {
        for (b = 0; b < dim; b++) {
          for (k = 0; k < ei[pg->imtrx]->dof[SHELL_NORMAL1]; k++) {
            dH_dtime_dnormal[b][k] -= fv_dot->d[i] * delta(i, b) * bf[SHELL_NORMAL1]->phi[k];
            dH_dtime_dnormal[b][k] -=
                fv->d[i] * delta(i, b) * bf[SHELL_NORMAL1]->phi[k] * (1 + 2 * tt) / dt;
          }
        }
      }
    }

    break;
  }

  /*** RESIDUAL ASSEMBLY ******************************************************/
  if (af->Assemble_Residual) {
    peqn = upd->ep[pg->imtrx][eqn];

    /*** Loop over DOFs (i) ***/
    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

      /* this is an optimization for xfem */
      if (xfem != NULL) {
        int xfem_active, extended_dof, base_interp, base_dof;
        xfem_dof_state(i, pd->i[pg->imtrx][eqn], ei[pg->imtrx]->ielem_shape, &xfem_active,
                       &extended_dof, &base_interp, &base_dof);
        if (extended_dof && !xfem_active)
          continue;
      }

      /* Prepare basis funcitons */
      ShellBF(eqn, i, &phi_i, grad_phi_i, grad_II_phi_i, d_grad_II_phi_i_dmesh,
              n_dof[MESH_DISPLACEMENT1], dof_map);

      /* Assemble diffusion term */
      diffusion = 0.0;
      if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
        for (p = 0; p < dim; p++) {
          diffusion += LubAux->q[p] * grad_II_phi_i[p];
        }
        diffusion *= det_J * wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
      }

      /* Assemble source term */
      source = 0.0;
      if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
        source = flux;
        source += -dH_dtime;
        source += (veloU[0] * dH_U_dX[0] + veloU[1] * dH_U_dX[1] - veloU[2]);
        source -= (veloL[0] * dH_L_dX[0] + veloL[1] * dH_L_dX[1] - veloL[2]);
        source *= phi_i;
      }
      source *= det_J * wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

      lec->R[LEC_R_INDEX(peqn, i)] += diffusion + source;
    } /* end of loop over i */
  } /* end of Assemble_Residual */

  /*** JACOBIAN ASSEMBLY ******************************************************/

  if (af->Assemble_Jacobian) {
    peqn = upd->ep[pg->imtrx][eqn];

    /*** Loop over DOFs (i) ***/
    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

      /* this is an optimization for xfem */
      if (xfem != NULL) {
        int xfem_active, extended_dof, base_interp, base_dof;
        xfem_dof_state(i, pd->i[pg->imtrx][eqn], ei[pg->imtrx]->ielem_shape, &xfem_active,
                       &extended_dof, &base_interp, &base_dof);
        if (extended_dof && !xfem_active)
          continue;
      }

      /* Prepare basis functions (i) */
      ShellBF(eqn, i, &phi_i, grad_phi_i, grad_II_phi_i, d_grad_II_phi_i_dmesh,
              n_dof[MESH_DISPLACEMENT1], dof_map);

      /*
       * J_lubp_p or J_lubp2_p2  --the diagonal piece.
       */
      if (EQN == R_LUBP) {
        var = LUBP;
      } else if (EQN == R_LUBP_2) {
        var = LUBP_2;
      } else
        GOMA_EH(GOMA_ERROR, "Mucho problema: Shouldn't be here.");

      if (pd->v[pg->imtrx][var]) {
        pvar = upd->vp[pg->imtrx][var];

        /*** Loop over DOFs (j) ***/
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

          /* Load basis functions (j) */
          ShellBF(var, j, &phi_j, grad_phi_j, grad_II_phi_j, d_grad_II_phi_j_dmesh,
                  n_dof[MESH_DISPLACEMENT1], dof_map);

          /* Add diffusion term */
          diffusion = 0.0;
          if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
            for (a = 0; a < dim; a++) {
              diffusion += LubAux->dq_dp2[a] * phi_j * grad_II_phi_i[a];
              for (b = 0; b < dim; b++) {
                diffusion += LubAux->dq_dgradp[a][b] * grad_II_phi_j[b] * grad_II_phi_i[a];
              }
            }
          }
          diffusion *= det_J * wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          /* Add source term */
          source = 0.0;
          if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
            source += d_flux[var][j] * det_J;
            source *= phi_i;
          }
          source *= wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += diffusion + source;
        } // End of loop over j
      } // End of J_lubp_p

      /*
       * J_lubp_velocity
       */
      var = VELOCITY1;
      if (pd->v[pg->imtrx][var]) {
        pvar = upd->vp[pg->imtrx][var];

        /*** Loop over DOFs (j) ***/
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

          /* Add diffusion term */
          diffusion = 0.0;
          if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
            for (b = 0; b < dim; b++) {
              for (p = 0; p < dim; p++) {
                diffusion += LubAux->dq_dv[b][p][j] * grad_II_phi_i[b];
              }
            }
          }
          diffusion *= det_J * wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += diffusion;
        } // End of loop over j
      } // End of J_lubp_velocity

      /*
       * J_lubp_curv
       */
      var = SHELL_LUB_CURV;
      if (pd->v[pg->imtrx][var]) {
        pvar = upd->vp[pg->imtrx][var];

        /*** Loop over DOFs (j) ***/
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          phi_j = bf[var]->phi[j];

          /* Add diffusion term */
          diffusion = 0.0;
          if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
            for (b = 0; b < dim; b++) {
              diffusion += LubAux->dq_dk[b] * grad_II_phi_i[b] * phi_j;
            }
          }
          diffusion *= det_J * wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += diffusion;
        } // End of loop over j
      } // End of J_lubp_curv

      /*
       * J_lubp_curv_2
       */
      var = SHELL_LUB_CURV_2;
      if (pd->v[pg->imtrx][var]) {
        pvar = upd->vp[pg->imtrx][var];

        /*** Loop over DOFs (j) ***/
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          phi_j = bf[var]->phi[j];

          /* Add diffusion term */
          diffusion = 0.0;
          if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
            for (b = 0; b < dim; b++) {
              diffusion += LubAux->dq_dk[b] * grad_II_phi_i[b] * phi_j;
            }
          }
          diffusion *= det_J * wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += diffusion;
        } // End of loop over j
      } // End of J_lubp_curv_2

      /*
       * J_lubp_LS or J_lubp_phase1  depending on lubp or lubp2
       */
      var = LS;
      if (EQN == R_LUBP_2)
        var = PHASE1;

      if (pd->v[pg->imtrx][var]) {
        pvar = upd->vp[pg->imtrx][var];

        /*** Loop over DOFs (j) ***/
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

          /* Add diffusion term */
          diffusion = 0.0;
          if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
            for (b = 0; b < dim; b++) {
              diffusion += LubAux->dq_df[b][j] * grad_II_phi_i[b];
            }
          }
          diffusion *= det_J * wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += diffusion;
        } // End of loop over j
      } // End of J_lubp_LS

      /*
       * J_lubp_DMX
       */
      var = MESH_DISPLACEMENT1;
      if (pd->v[pg->imtrx][var] &&
          (mp->FSIModel == FSI_MESH_CONTINUUM || mp->FSIModel == FSI_REALSOLID_CONTINUUM ||
           mp->FSIModel == FSI_MESH_UNDEF || mp->FSIModel == FSI_SHELL_ONLY_MESH ||
           mp->FSIModel == FSI_SHELL_ONLY_UNDEF)) {
        pvar = upd->vp[pg->imtrx][var];

        /*** Loop over dimensions of mesh displacement ***/
        for (b = 0; b < dim; b++) {
          var = MESH_DISPLACEMENT1 + b;
          pvar = upd->vp[pg->imtrx][var];

          /*** Loop over DOFs (j) ***/
          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            jk = dof_map[j];

            /* Add diffusion term */
            diffusion = 0.0;
            if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
              for (p = 0; p < dim; p++) {
                diffusion += det_J * LubAux->dq_dx[p][b][j] * grad_II_phi_i[p];
                diffusion += det_J * LubAux->q[p] * d_grad_II_phi_i_dmesh[p][b][jk];
                diffusion += fv->dsurfdet_dx[b][jk] * LubAux->q[p] * grad_II_phi_i[p];
              }
            }
            diffusion *= wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

            /* Add source term */
            source = 0.0;
            if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
              source += -dH_dtime_dmesh[b][j] * det_J;
              source += (mp->lubsource - dH_dtime) * fv->dsurfdet_dx[b][jk];
              source += (veloU[0] * dH_U_dX[0] + veloU[1] * dH_U_dX[1] - veloU[2]) *
                        fv->dsurfdet_dx[b][jk];
              source -= (veloL[0] * dH_L_dX[0] + veloL[1] * dH_L_dX[1] - veloL[2]) *
                        fv->dsurfdet_dx[b][jk];
              source *= phi_i;
            }
            source *= wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

            lec->J[LEC_J_INDEX(peqn, pvar, i, jk)] += diffusion + source;
          } // End of loop over j
        } // End of loop over b
      } // End of J_lubp_mesh

      /*
       * J_lubp_DRS
       */
      var = SOLID_DISPLACEMENT1;
      if (upd->vp[pg->imtrx][var] >= 0 && (mp->FSIModel == FSI_REALSOLID_CONTINUUM)) {
        pvar = upd->vp[pg->imtrx][var];

        /*** Loop over dimensions of mesh displacement ***/
        for (b = 0; b < dim; b++) {
          var = SOLID_DISPLACEMENT1 + b;
          pvar = upd->vp[pg->imtrx][var];

          /*** Loop over DOFs (j) ***/
          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            jk = dof_map[j];

            /* Add diffusion term */
            diffusion = 0.0;
            if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
              for (p = 0; p < dim; p++) {
                diffusion += det_J * LubAux->dq_drs[p][b][j] * grad_II_phi_i[p];
              }
            }
            diffusion *= wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

            /* Add source term */
            source = 0.0;
            if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
              source += -dH_dtime_drealsolid[b][j] * det_J;
              source *= phi_i;
            }
            source *= wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

            lec->J[LEC_J_INDEX(peqn, pvar, i, jk)] += diffusion + source;
          } // End of loop over j
        } // End of loop over b
      } // End of J_lubp_drs

      /*
       * J_lubp_pressure
       */
      var = PRESSURE;
      if (upd->vp[pg->imtrx][var] >= 0) {
        pvar = upd->vp[pg->imtrx][var];

        /*** Loop over DOFs (j) ***/
        for (j = 0; j < n_dof[var]; j++) {
          jk = dof_map[j];

          /* Add source term */
          source = 0.0;
          if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
            source += d_flux[var][j] * det_J;
            source *= phi_i;
          }
          source *= wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, jk)] += source;
        } // End of loop over j
      } // End of J_lubp_pressure

      /*
       * J_lubp_shell_normal
       */
      var = SHELL_NORMAL1;
      if (pd->v[pg->imtrx][var] && mp->FSIModel == FSI_SHELL_ONLY_MESH) {
        pvar = upd->vp[pg->imtrx][var];

        /*** Loop over dimensions of shell normals ***/
        for (b = 0; b < dim; b++) {
          var = SHELL_NORMAL1 + b;
          pvar = upd->vp[pg->imtrx][var];

          /*** Loop over DOFs (j) ***/
          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

            /* Add diffusion term */
            diffusion = 0.0;
            if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
              for (p = 0; p < dim; p++) {
                diffusion += det_J * LubAux->dq_dnormal[p][b][j] * grad_II_phi_i[p];
              }
            }
            diffusion *= wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

            /* Add source term */
            source = 0.0;
            if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
              source += -dH_dtime_dnormal[b][j] * det_J;
              source *= phi_i;
            }
            source *= wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

            lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += diffusion + source;
          } // End of loop over j
        } // End of loop over b
      } // End of J_lubp_shell_normal

      /*
       * J_lubp_D_sh_dh
       */
      var = SHELL_DELTAH;
      if (pd->v[pg->imtrx][var]) {
        pvar = upd->vp[pg->imtrx][var];

        /*** Loop over DOFs (j) ***/
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          phi_j = bf[var]->phi[j];

          /* Add diffusion term */
          diffusion = 0.0;
          if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
            for (p = 0; p < dim; p++) {
              diffusion += det_J * LubAux->dq_ddh[p] * phi_j * grad_II_phi_i[p];
            }
          }
          diffusion *= wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          /* Add source term */
          source = 0.0;
          if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
            // dh_time no longer has dependence here, as of 4/11/2011. Talk to PRS.
            // If you wanted to add some volume expansion, however, there would be
            // a boost here.
            // source += -0.*toggle_dh_dependence*(1 + 2. * tt)*phi_j/dt;
            source *= phi_i;
          }
          source *= det_J * wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += diffusion + source;
        } // End of loop over j
      } // End of J_lubp_dDeltah

      /*
       * J_lubp_D_sh_pc
       */

      var = SHELL_PARTC;

      if (pd->v[pg->imtrx][var]) {
        pvar = upd->vp[pg->imtrx][var];
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          phi_j = bf[var]->phi[j];

          diffusion = 0.;
          if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
            for (p = 0; p < VIM; p++) {
              diffusion += LubAux->dq_dc[p][j] * phi_j * grad_II_phi_i[p];
            }

            diffusion *= det_J * wt;
            diffusion *= h3;
            diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
          }
          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += diffusion;
        } // End of loop over j
      } // End of J_lubp_dsh_pc

      /*
       * J_lubp_D_C
       */

      var = MASS_FRACTION;

      if (pd->v[pg->imtrx][var]) {
        for (w = 0; w < pd->Num_Species_Eqn; w++) {
          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

            diffusion = 0.;
            if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
              for (p = 0; p < VIM; p++) {
                for (q = 0; q < VIM; q++) {
                  diffusion += LubAux->dq_dconc[p][q][w][j] * grad_II_phi_i[q];
                }
              }

              diffusion *= det_J * wt;
              diffusion *= h3;
              diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
            }
            lec->J[LEC_J_INDEX(peqn, MAX_PROB_VAR + w, i, j)] += diffusion;
          } // End of loop over j
        } // loop over species
      } // End of J_lubp_d_C

      /*
       * J_lubp_shear_top
       */
      var = SHELL_SHEAR_TOP;
      if (pd->v[pg->imtrx][var]) {
        pvar = upd->vp[pg->imtrx][var];

        /*** Loop over DOFs (j) ***/
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          phi_j = bf[var]->phi[j];

          /* Add diffusion term */
          diffusion = 0.0;
          if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
            for (b = 0; b < dim; b++) {
              diffusion += LubAux->dq_dshrw[b] * grad_II_phi_i[b] * phi_j;
            }
          }
          diffusion *= det_J * wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += diffusion;
        } // End of loop over j
      } // End of J_lubp_shear_top

      /*
       * J_lubp_Temperature
       */
      var = SHELL_TEMPERATURE;
      if (pd->v[pg->imtrx][var]) {
        pvar = upd->vp[pg->imtrx][var];

        /*** Loop over DOFs (j) ***/
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          phi_j = bf[var]->phi[j];

          /* Add diffusion term */
          diffusion = 0.0;
          if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
            for (b = 0; b < dim; b++) {
              diffusion += LubAux->dq_dT[b] * grad_II_phi_i[b] * phi_j;
            }
          }
          diffusion *= det_J * wt * h3 * pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += diffusion;
        } // End of loop over j
      } // End of J_lubp_T

    } /* end of loop over i */
  } /* end of Assemble_Jacobian */

  /* clean-up */
  fv->wt = wt; /* load_neighbor_var_data screws this up */
  safe_free((void *)n_dof);
  return (status);
} /* end of assemble_lubrication */

void ad_lubrication_shell_initialize(int *n_dof,        // Degrees of freedom
                                     int *dof_map,      // Map of DOFs
                                     int id_side,       // Side ID
                                     double xi[DIM],    // Local STU coordinates
                                     const Exo_DB *exo, // Exodus database
                                     int use_def        // Use deformed normal anyway
                                     )
/******************************************************************************
 *
 *
 *
 *
 *    Routine to set up all of the necessary shell normals and heights for
 *    lubrication shells.  Ideally, anything that needs to be done in multiple
 *    locations will be put here.  This will include loading of the heights
 *    and velocities, proper calculations of FSI and any mesh derivatives.
 *
 * Scott A Roberts (1514) sarober@sandia.gov
 *
 ******************************************************************************/
{
  int a, b, i, j, k, p;
  int el1 = -1, el2 = -1, nf;
  int FSIModel = -1;
  int n_dofptr[MAX_VARIABLE_TYPES][MDE];
  dbl wt = fv->wt;

  int ShapeVar = pd->ShapeVar;
  int mdof = ei[pg->imtrx]->dof[ShapeVar];
  int edim = ei[pg->imtrx]->ielem_dim;
  int pdim = pd->Num_Dim;
  int node, index;
  double Jlocal[DIM][DIM];
  double T[DIM - 1][DIM], t[DIM - 1][DIM]; /* t = J . T */
  int siz;
  double nx, ny, nz;
  double r_det;

  /*** LOOK FOR NEIGHBOR ELEMENTS AND SELECT CORRECT MODEL ********************/

  /* Find friends */
  el1 = ei[pg->imtrx]->ielem;
  nf = num_elem_friends[el1];

  /* Deal with number of friends */
  switch (nf) {
  case 0:
    if ((mp->FSIModel != FSI_SHELL_ONLY) && (mp->FSIModel != FSI_SHELL_ONLY_MESH) &&
        (mp->FSIModel != FSI_SHELL_ONLY_UNDEF))
      GOMA_EH(GOMA_ERROR, "ERROR:  What happened to my little friend?");
    FSIModel = mp->FSIModel;
    break;
  case 1:
    el2 = elem_friends[el1][0];
    if (mp->FSIModel == 0) {
      a = find_elemblock_index(el2, exo);
      b = Matilda[a];
      FSIModel = mp_glob[b]->FSIModel;
    } else {
      FSIModel = mp->FSIModel;
    }
    break;
  default:
    GOMA_EH(GOMA_ERROR, "ERROR: Not set up for more than one element friend!");
    break;
  }

  /* Make sure we have a model */
  if (FSIModel == 0)
    GOMA_EH(GOMA_ERROR, "ERROR: Could not find FSI Model");

  /* Use deformed normal for boundary */
  if ((use_def == 1) && (FSIModel == FSI_MESH_UNDEF))
    FSIModel = FSI_MESH_CONTINUUM;

  /*** CALCULATE SHELL NORMALS AND POPULATE FV AND BF STRUCTURES **************/
  switch (FSIModel) {

    /*** SHELL ONLY ***/
  case FSI_SHELL_ONLY:

    shell_determinant_and_normal(ei[pg->imtrx]->ielem, ei[pg->imtrx]->iconnect_ptr,
                                 ei[pg->imtrx]->num_local_nodes, ei[pg->imtrx]->ielem_dim, 1);
    n_dof[MESH_DISPLACEMENT1] = 0;
    n_dof[MESH_DISPLACEMENT2] = 0;
    n_dof[MESH_DISPLACEMENT3] = 0;

    /* calc_surf_tangent (ei[pg->imtrx]->ielem, ei[pg->imtrx]->iconnect_ptr,
       ei[pg->imtrx]->num_local_nodes, ei[pg->imtrx]->ielem_dim, ei[pg->imtrx]->num_local_nodes,
       &temp);
    */
    break;

  case FSI_SHELL_ONLY_MESH:

    if (!pd->e[pg->imtrx][R_MESH1])
      GOMA_EH(GOMA_ERROR, "ERROR:  FSI_SHELL_ONLY_MESH requires mesh equation turned on!");

    shell_determinant_and_normal(ei[pg->imtrx]->ielem, ei[pg->imtrx]->iconnect_ptr,
                                 ei[pg->imtrx]->num_local_nodes, ei[pg->imtrx]->ielem_dim, 1);

    /* Populate ndof array */
    n_dof[MESH_DISPLACEMENT1] = ei[pg->imtrx]->dof[MESH_DISPLACEMENT1];
    n_dof[MESH_DISPLACEMENT2] = ei[pg->imtrx]->dof[MESH_DISPLACEMENT2];
    n_dof[MESH_DISPLACEMENT3] = ei[pg->imtrx]->dof[MESH_DISPLACEMENT3];
    if (pd->e[pg->imtrx][R_SHELL_NORMAL1]) {
      n_dof[SHELL_NORMAL1] = ei[pg->imtrx]->dof[SHELL_NORMAL1];
      n_dof[SHELL_NORMAL2] = ei[pg->imtrx]->dof[SHELL_NORMAL2];
      n_dof[SHELL_NORMAL3] = ei[pg->imtrx]->dof[SHELL_NORMAL3];
    }

    /* Populate a trivial dof_map array */
    for (i = 0; i < ei[pg->imtrx]->dof[pd->ShapeVar]; i++) {
      dof_map[i] = i;
    }
    break;

  case FSI_SHELL_ONLY_UNDEF:

    if (!pd->e[pg->imtrx][R_MESH1])
      GOMA_EH(GOMA_ERROR, "ERROR:  FSI_SHELL_ONLY_UNDEF requires mesh equation turned on!");

    /* Populate surface determinants (detJ) and its sensitivities first */

    shell_determinant_and_normal(ei[pg->imtrx]->ielem, ei[pg->imtrx]->iconnect_ptr,
                                 ei[pg->imtrx]->num_local_nodes, ei[pg->imtrx]->ielem_dim, 1);

    /* Then calculate normal using the original configuration */

    /* Calculate Jacobian of transformation */
    for (i = 0; i < edim; i++) {
      for (j = 0; j < pdim; j++) {
        Jlocal[i][j] = 0.0;
        for (k = 0; k < mdof; k++) {
          node = ei[pg->imtrx]->dof_list[ShapeVar][k];
          index = Proc_Elem_Connect[Proc_Connect_Ptr[ei[pg->imtrx]->ielem] + node];
          Jlocal[i][j] += Coor[j][index] * bf[ShapeVar]->dphidxi[k][i];
        }
        Jlocal[2][j] = (j + 1) * 1.0;
      }
    }

#if 0
    /* Big T calculation*/
    T[0][0] = 1.;
    T[0][1] = 0.;
    T[0][2] = 0.;
    T[1][0] = 0.;
    T[1][1] = 1.;
    T[1][2] = 0.;

#else
    siz = (DIM - 1) * DIM * sizeof(double);
    memset(T, 0, siz);
    memset(t, 0, siz);
    /*  since T & t are zeroed, only need to set nonzero elements */
    /* revert back by setting id_side = 6  */
    switch (ei[pg->imtrx]->ielem_shape) {
    case SHELL:
    case TRISHELL:
      switch (id_side) {
      case 1:
        T[0][0] = 1.;
        T[1][2] = 1.;
        break;
      case 2:
        T[0][1] = 1.;
        T[1][2] = 1.;
        break;
      case 3:
        T[0][0] = -1.;
        T[1][2] = 1.;
        break;
      case 4:
        T[0][1] = -1.;
        T[1][2] = 1.;
        break;
      case 5:
        T[0][0] = 1.;
        T[1][1] = -1.;
        break;
      case 6:
      case -1:
        T[0][0] = 1.;
        T[1][1] = 1.;
        break;
      default:
        GOMA_EH(GOMA_ERROR, "Incorrect side for SHELL");
        break;
      }
      break;
    }
#endif
    /* Little t calculation */
    for (p = 0; p < 2; p++) {
      for (a = 0; a < pd->Num_Dim; a++) {
        t[p][a] = 0.;
        for (b = 0; b < pd->Num_Dim; b++) {
          t[p][a] += Jlocal[b][a] * T[p][b] * fv->h[b];
        }
      }
    }

    /* N calculation */
    nx = t[0][1] * t[1][2] - t[0][2] * t[1][1];
    ny = t[0][2] * t[1][0] - t[0][0] * t[1][2];
    nz = t[0][0] * t[1][1] - t[0][1] * t[1][0];

    /* Dets */
    r_det = 1. / sqrt(nx * nx + ny * ny + nz * nz);

    /* Overwrite normals */
    ad_fv->snormal[0] = r_det * nx;
    ad_fv->snormal[1] = r_det * ny;
    ad_fv->snormal[2] = r_det * nz;

    /* Populate ndof array */
    n_dof[MESH_DISPLACEMENT1] = ei[pg->imtrx]->dof[MESH_DISPLACEMENT1];
    n_dof[MESH_DISPLACEMENT2] = ei[pg->imtrx]->dof[MESH_DISPLACEMENT2];
    n_dof[MESH_DISPLACEMENT3] = ei[pg->imtrx]->dof[MESH_DISPLACEMENT3];

    /* Populate a trivial dof_map array */
    for (i = 0; i < ei[pg->imtrx]->dof[pd->ShapeVar]; i++) {
      dof_map[i] = i;
    }

    /* Zero out sensitivities */
    int jk;
    for (a = 0; a < pdim; a++) {
      for (b = 0; b < pdim; b++) {
        for (k = 0; k < ei[pg->imtrx]->dof[MESH_DISPLACEMENT1]; k++) {
          jk = dof_map[k];
          fv->dsnormal_dx[a][b][jk] = 0.0;
        }
      }
    }

    break;

    /*** CONTINUUM DEFORMED MESH ***/
  case FSI_MESH_CONTINUUM:
  case FSI_MESH_ONEWAY:
  case FSI_REALSOLID_CONTINUUM:

    load_neighbor_var_data(el1, el2, n_dof, dof_map, n_dofptr, id_side, xi, exo);

    break;
    /*** CONTINUUM DEFORMED MESH ***/
  case FSI_MESH_UNDEF:

    // Load DOF count and repopulate fv structure from neighbor element
    load_neighbor_var_data(el1, el2, n_dof, dof_map, n_dofptr, id_side, xi, exo);

    // Calculate Jacobian of transformation
    for (i = 0; i < edim; i++) {
      for (j = 0; j < pdim; j++) {
        Jlocal[i][j] = 0.0;
        for (k = 0; k < mdof; k++) {
          node = ei[pg->imtrx]->dof_list[ShapeVar][k];
          index = Proc_Elem_Connect[Proc_Connect_Ptr[ei[pg->imtrx]->ielem] + node];
          Jlocal[i][j] += Coor[j][index] * bf[ShapeVar]->dphidxi[k][i];
        }
        Jlocal[2][j] = (j + 1) * 1.0;
      }
    }

    // Bit T calculation
    T[0][0] = 1.;
    T[0][1] = 0.;
    T[0][2] = 0.;
    T[1][0] = 0.;
    T[1][1] = 1.;
    T[1][2] = 0.;

    // Little t calculation
    for (p = 0; p < 2; p++) {
      for (a = 0; a < pd->Num_Dim; a++) {
        t[p][a] = 0.;
        for (b = 0; b < pd->Num_Dim; b++) {
          t[p][a] += Jlocal[b][a] * T[p][b] * fv->h[b];
        }
      }
    }

    // N calculation
    nx = t[0][1] * t[1][2] - t[0][2] * t[1][1];
    ny = t[0][2] * t[1][0] - t[0][0] * t[1][2];
    nz = t[0][0] * t[1][1] - t[0][1] * t[1][0];

    // Dets
    fv->sdet = sqrt(nx * nx + ny * ny + nz * nz);
    r_det = 1. / fv->sdet;

    // Calculate normals
    ad_fv->snormal[0] = r_det * nx;
    ad_fv->snormal[1] = r_det * ny;
    ad_fv->snormal[2] = r_det * nz;

    // Zero out sensitivities
    int ldof;
    for (i = 0; i < ei[pg->imtrx]->num_local_nodes; i++) {
      ldof = ei[pg->imtrx]->ln_to_dof[ShapeVar][2];
      for (a = 0; a < pdim; a++) {
        fv->dsurfdet_dx[a][ldof] = 0.0;
        for (j = 0; j < 3; j++) {
          fv->dsnormal_dx[j][a][ldof] = 0.0;
        }
      }
    }

    break;
    /*** UNIMPLEMENTED METHODS ***/
  case FSI_MESH_BOTH:
  case FSI_MESH_SHELL:

    GOMA_EH(GOMA_ERROR, "ERROR:  lubrication_shell_initialize() - Error in FSI Model");
    break;
  }

  /*** Reset weight ***/
  fv->wt = wt;
  return;
} /* End of lubrication_shell_initialize */

void ad_shell_determinant_and_normal(
    const int ielem,          /* current element number               */
    const int iconnect_ptr,   /* Pointer to beginning of connectivity
                               * list for current element             */
    const int nodes_per_elem, /* number of nodes in the element       */
    const int ielem_surf_dim, /* physical dimension of the element
                               * surface (0, 1, 2)                    */
    const int id_side)        /* shell side (bottom or top) (exo/patran convention)  */

/************************************************************************
 *
 * shell_determinant_and_normal()
 *
 *      Function which calculates the shell surface determinant and shell surface
 *      normal at a local surface quadrature point. The function also
 *      calculates the sensitivities of those quantities to the
 *      mesh positions, if the mesh positions are part of the solution
 *      vector.
 *
 *      Adapted from surface_determinant_and_normal by PR Schunk (9/2/2009)
 *      Looks like id_side is always 1, so pretty much not used - RBS
 *
 *  Returns:
 * ------------
 *   fv->sdet = surface determinant at the quadrature point
 *   fv->snormal[] = surface normal at the quadrature point
 *   fv->dsurfdet_dx[][] = sensitivity of fv->sdet wrt mesh displacements.
 *   fv->dsnormal_dx[][] = sensitivity of fv->snormal[]
 *                         wrt mesh displacements
 ***********************************************************************/
{
  int i, inode, a, b, p, q;
  int ShapeVar, ldof;
  int DeformingMesh;
  ADType r_det, det_h01, r_det_h01, d_det_h01_x;
  double phi_i;
  int siz;
  ADType T[DIM - 1][DIM], t[DIM - 1][DIM]; /* t = J . T */
  double dt_x[DIM - 1][DIM][DIM][MDE];     /* d(t) / d(x_j) */
  struct Basis_Functions *map_bf;

  DeformingMesh = pd->e[pg->imtrx][R_MESH1];
  DeformingMesh = (upd->ep[pg->imtrx][R_MESH1] != -1);
  ShapeVar = pd->ShapeVar;

  siz = MAX_PDIM * MDE * sizeof(double);
  memset(fv->dsurfdet_dx, 0, siz);
  siz = MAX_PDIM * MDE * MAX_PDIM * sizeof(double);
  memset(fv->dsnormal_dx, 0, siz);

  map_bf = bf[ShapeVar];

  /* Here ielem_surf_dim is the shell dimension, which is either 1 or 2 */
  /* It is set as ei[pg->imtrx]->ielem_dim (not pd->Num_Dim which is usual 1 dimension more) */

  if (ielem_surf_dim == 0) /* get out quickly */
  {
    ad_fv->sdet = 1.0;
    ad_fv->snormal[0] = 1.0;
    return;
  }

  /* define space of surface */
  siz = (DIM - 1) * DIM * sizeof(double);
  memset(T, 0, siz);
  memset(t, 0, siz);
  /*  since T & t are zeroed, only need to set nonzero elements */
  switch (ielem_surf_dim) {
  case 1:
    switch (ei[pg->imtrx]->ielem_shape) {
    case LINE_SEGMENT:
      T[0][0] = 1.;
      break;
    default:
      GOMA_EH(GOMA_ERROR, "Invalid shape");
      break;
    }
    break;
  case 2:
    switch (ei[pg->imtrx]->ielem_shape) {
    case SHELL:
    case TRISHELL:
      /* if (id_side == 5 )
        {
          T[0][0] =  1.; T[0][1] =  0.; T[0][2] =  0.;
          T[1][0] =  0.; T[1][1] = -1.; T[1][2] =  0.;
        }
      else if ( id_side == 6 )
        {
          T[0][0] =  1.; T[0][1] =  0.; T[0][2] =  0.;
          T[1][0] =  0.; T[1][1] =  1.; T[1][2] =  0.;
        }
      else
        {
          GOMA_EH(GOMA_ERROR, "Incorrect side for HEXAHEDRAL");
          } */

      /*Not sure how we want to handle this, viz. compatible with the
        bulk element surf or just define the isoparametric integration.  For
        now we will hardwire to "patran id_side 6" in which Zeta is fixed */

      /* Here we want to load up a 3D Jacobian matrix which we can use the T matrix to
       * for gradient conversion.  beer_belly loads up 2D jacobian for the shell-only case, so
       * we cannot use that.
       */

      T[0][0] = 1.;
      T[1][1] = 1.;
      break;

    default:
      GOMA_EH(GOMA_ERROR, "Invalid shape. Lubrication capability requires shell elements");
      break;
    }
    break;
  }

  /* Use T matrix which 2x3 and J which is 3x3 to transform grad operator */
  /* This is a load_bf_derivs equivalent for shells */

  /* transform from element to physical coords */
  /* NOTE: This does not work for 2D bars because map_bf->J
     is hardwired in beerbelly to already compute this */

  for (p = 0; p < ielem_surf_dim; p++) {
    for (a = 0; a < pd->Num_Dim; a++) {
      t[p][a] = 0.;
      for (b = 0; b < pd->Num_Dim; b++) {
        t[p][a] += ad_fv->J[b][a] * T[p][b] * fv->h[b];
      }
    }
  }

  if (ielem_surf_dim == 1) {

    /* N.B. NEXT Stage of Change: Correct Mapbf->J for 1D case in beer_belly */
    // GOMA_EH(GOMA_ERROR, "you should not be here until mapbf->J in beerbelly is corrected for 1D
    // case");
    /* calculate surface determinant using the coordinate scale factors
     * for orthogonal curvilinear coordinates */
    det_h01 = sqrt(t[0][0] * t[0][0] + t[0][1] * t[0][1]);
    r_det_h01 = 1. / det_h01;

    ad_fv->sdet = fv->h[2] * det_h01;
    r_det = 1. / ad_fv->sdet;

    /* calculate surface normal using the coordinate scale factors
     * for orthogonal curvilinear coordinates */
    ad_fv->snormal[0] = t[0][1] * r_det_h01;
    ad_fv->snormal[1] = -t[0][0] * r_det_h01;
    ad_fv->snormal[2] = 0.;

    /* Calculate sensitivity w.r.t. mesh, if applicable */

    if (mp->ehl_integration_kind == SIK_S) {
      double det_J;
      double d_det_J_dmeshkj[DIM][MDE];
      memset(d_det_J_dmeshkj, 0.0, sizeof(double) * DIM * MDE);
      detJ_2d_bar(&det_J, d_det_J_dmeshkj);
      ad_fv->sdet = det_J;
      for (int k = 0; k < DIM; k++) {
        for (int j = 0; j < ei[pg->imtrx]->dof[MESH_DISPLACEMENT1]; j++) {
          fv->dsurfdet_dx[k][j] = d_det_J_dmeshkj[k][j];
        }
      }
    }
  } else if (ielem_surf_dim == 2) {
    ADType nx = t[0][1] * t[1][2] - t[0][2] * t[1][1];
    ADType ny = t[0][2] * t[1][0] - t[0][0] * t[1][2];
    ADType nz = t[0][0] * t[1][1] - t[0][1] * t[1][0];

    /* calculate surface determinant using the coordinate scale factors
     * for orthogonal curvilinear coordinates */
    ad_fv->sdet = sqrt(nx * nx + ny * ny + nz * nz);
    r_det = 1. / ad_fv->sdet;

    /* calculate surface normal using the coordinate scale factors
     * for orthogonal curvilinear coordinates */
    ad_fv->snormal[0] = r_det * nx;
    ad_fv->snormal[1] = r_det * ny;
    ad_fv->snormal[2] = r_det * nz;

    /* Calculate sensitivity w.r.t. mesh, if applicable */
  }
}

ADType ad_height_function_model(ADType *H_U,
                                ADType *H_L,
                                ADType *dH_U_dtime,
                                ADType *dH_L_dtime,
                                ADType dH_U_dX[DIM],
                                ADType dH_L_dX[DIM],
                                ADType *dH_U_dp,
                                ADType *dH_U_ddh,
                                ADType dH_dF[MDE],
                                double time,    /* present time value           */
                                double delta_t) /* present time step             */

/******************************************************************************
 *
 *  A function which computes the height at the current time and the rate-of-change of
 *  height.  This model is used for the lubrication capability
 *
 *  P. Randall Schunk (March 2009, Somewhere over Texas)
 *
 *
 ******************************************************************************/

{
  ADType H = 0.;
  ADType H_dot, H_init, H_delta, H_low, x_0, Length, r;
  ADType R, origin[3], dir_angle[3], t, axis_pt[3], dist, cos_denom;

  // Initialize gradients, etc. so only need to add nonzero values
  dH_U_dX[0] = dH_U_dX[1] = dH_U_dX[2] = 0.0;
  dH_L_dX[0] = dH_L_dX[1] = dH_L_dX[2] = 0.0;
  *dH_U_dtime = 0.0;
  *dH_U_dp = 0.0;
  *dH_U_ddh = 0.0;
  *dH_L_dtime = 0.0;
  for (int i = 0; i < MDE; i++) {
    dH_dF[i] = 0.0;
  }

  if (pd->TimeIntegration == STEADY)
    time = 0.;

  if (mp->HeightUFunctionModel == CONSTANT) {
    *H_U = mp->heightU;
  }

  else if (mp->HeightUFunctionModel == CONSTANT_SPEED || mp->HeightUFunctionModel == WALL_DISTMOD ||
           mp->HeightUFunctionModel == WALL_DISTURB) {
    H_dot = mp->u_heightU_function_constants[0];
    H_init = mp->u_heightU_function_constants[1];
    *H_U = H_dot * time + H_init;

    /* add on external field height if there is one. The scale factor will be the third user const*/
    // It seems we should dispense with this option and just use the EXTERNAL_FIELD one...
    if (mp->heightU_ext_field_index >= 0 && mp->HeightUFunctionModel != WALL_DISTMOD &&
        mp->HeightUFunctionModel != WALL_DISTURB) {
      *H_U += mp->u_heightU_function_constants[2] * fv->external_field[mp->heightU_ext_field_index];
      dH_U_dX[0] =
          mp->u_heightU_function_constants[2] * fv->grad_ext_field[mp->heightU_ext_field_index][0];
      dH_U_dX[1] =
          mp->u_heightU_function_constants[2] * fv->grad_ext_field[mp->heightU_ext_field_index][1];
      dH_U_dX[2] =
          mp->u_heightU_function_constants[2] * fv->grad_ext_field[mp->heightU_ext_field_index][2];
    }
    *dH_U_dtime = H_dot;
  }

  else if (mp->HeightUFunctionModel == EXTERNAL_FIELD) {
    /* Note that this isthe same model as CONSTANT_SPEED, but allows for more generality on input */
    H_dot = mp->u_heightU_function_constants[0];
    H_init = mp->u_heightU_function_constants[1];
    *H_U = H_dot * time + H_init;

    *H_U += mp->u_heightU_function_constants[2] * fv->external_field[mp->heightU_ext_field_index];
    dH_U_dX[0] =
        mp->u_heightU_function_constants[2] * fv->grad_ext_field[mp->heightU_ext_field_index][0];
    dH_U_dX[1] =
        mp->u_heightU_function_constants[2] * fv->grad_ext_field[mp->heightU_ext_field_index][1];
    dH_U_dX[2] =
        mp->u_heightU_function_constants[2] * fv->grad_ext_field[mp->heightU_ext_field_index][2];

    *dH_U_dtime = H_dot;
  }

  else if (mp->HeightUFunctionModel == CONSTANT_SPEED_DEFORM) {
    double E_mod, L_0, Pext;
    H_dot = mp->u_heightU_function_constants[0];
    H_init = mp->u_heightU_function_constants[1];
    E_mod = mp->u_heightU_function_constants[2];
    L_0 = mp->u_heightU_function_constants[3];
    Pext = mp->u_heightU_function_constants[4];

    *H_U = H_dot * time + H_init + (fv->lubp - Pext) / (E_mod / L_0);
    // Right now this isn't complete because we need an augmenting condition
    // to give us a fv_dot->lubp kicker.
    *dH_U_dtime = H_dot + fv_dot->lubp / (E_mod / L_0);
    *dH_U_dp = 1. / (E_mod / L_0);
  }

  else if (mp->HeightUFunctionModel == CONSTANT_SPEED_MELT) {
    H_dot = mp->u_heightU_function_constants[0];
    H_init = mp->u_heightU_function_constants[1];

    *H_U = H_dot * time + H_init + fv->sh_dh; // 0.01*fv->external_field[0];

    // PRS: I'll leave this in here for now
    // because this is where you would add
    // a volume expansion effect upon phase change.
    // otherwise it is 0.*fv_dot->sh_dh

    *dH_U_dtime = H_dot - 0. * fv_dot->sh_dh;
    *dH_U_ddh = 1.0;
    if (*H_U <= 0.00001 * H_init) {
      *H_U = 0.00001 * H_init;
    }

  }

  else if (mp->HeightUFunctionModel == ROLL_ON) {
    Length = mp->u_heightU_function_constants[4];
    H_dot = mp->u_heightU_function_constants[3];
    H_delta = mp->u_heightU_function_constants[2];
    x_0 = mp->u_heightU_function_constants[0];
    H_low = mp->u_heightU_function_constants[1];

    *H_U = (H_dot * time + H_delta) * ((fv->x[0] - x_0) / Length) + H_low;

    /* add on external field height if there is one. The scale factor will be the third user const*/
    if (mp->heightU_ext_field_index >= 0)
      *H_U += mp->u_heightU_function_constants[5] * fv->external_field[mp->heightU_ext_field_index];

    *dH_U_dtime = H_dot * (fv->x[0] - x_0) / Length;
    dH_U_dX[0] = (H_dot * time + H_delta) / Length;
  }

  else if (mp->HeightUFunctionModel == ROLL_ON_MELT) {
    Length = mp->u_heightU_function_constants[4];
    H_dot = mp->u_heightU_function_constants[3];
    H_delta = mp->u_heightU_function_constants[2];
    x_0 = mp->u_heightU_function_constants[0];
    H_low = mp->u_heightU_function_constants[1];

    *H_U = (H_dot * time + H_delta) * ((fv->x[0] - x_0) / Length) + H_low + fv->sh_dh;
    *dH_U_dtime = H_dot * (fv->x[0] - x_0) / Length;
    dH_U_dX[0] = (H_dot * time + H_delta) / Length;
    *dH_U_ddh = 1.0;
  }

  else if (mp->HeightUFunctionModel == ROLL) {
    R = mp->u_heightU_function_constants[0];
    /*  origin and direction of rotation axis	*/
    origin[0] = mp->u_heightU_function_constants[1];
    origin[1] = mp->u_heightU_function_constants[2];
    origin[2] = mp->u_heightU_function_constants[3];
    dir_angle[0] = mp->u_heightU_function_constants[4];
    dir_angle[1] = mp->u_heightU_function_constants[5];
    dir_angle[2] = mp->u_heightU_function_constants[6];
    H_dot = mp->u_heightU_function_constants[7];
    origin[2] += H_dot * time;

    /*  find intersection of axis with normal plane - i.e., locate point on
            axis that intersects plane normal to axis that contains local point. */

    cos_denom = (SQUARE(dir_angle[0]) + SQUARE(dir_angle[1]) + SQUARE(dir_angle[2]));
    t = (dir_angle[0] * (fv->x[0] - origin[0]) + dir_angle[1] * (fv->x[1] - origin[1]) +
         dir_angle[2] * (fv->x[2] - origin[2])) /
        cos_denom;
    axis_pt[0] = origin[0] + dir_angle[0] * t;
    axis_pt[1] = origin[1] + dir_angle[1] * t;
    axis_pt[2] = origin[2] + dir_angle[2] * t;

    /*  compute radial direction	*/

    dist = sqrt(SQUARE(fv->x[0] - axis_pt[0]) + SQUARE(fv->x[1] - axis_pt[1]));
    if (dist > fabs(R)) {
      *H_U = fabs(R);
    } else {
      ADType sqrt_sR_sdist = sqrt(SQUARE(R) - SQUARE(dist));
      *H_U = SGN(R) * (axis_pt[2] - fv->x[2] - sqrt_sR_sdist);
      *dH_U_dtime = 0.; /* finish later  */

      dH_U_dX[0] = 0.;
      dH_U_dX[1] = 0.;
      dH_U_dX[2] = -1.;

      if (DOUBLE_NONZERO(sqrt_sR_sdist)) {
        dH_U_dX[0] += ((fv->x[0] - axis_pt[0]) * (1. - SQUARE(dir_angle[0]) / cos_denom) +
                       (fv->x[1] - axis_pt[1]) * (-dir_angle[0] * dir_angle[1] / cos_denom) +
                       (fv->x[2] - axis_pt[2]) * (-dir_angle[0] * dir_angle[2] / cos_denom)) /
                      sqrt_sR_sdist;
        dH_U_dX[1] += ((fv->x[0] - axis_pt[0]) * (-dir_angle[1] * dir_angle[0] / cos_denom) +
                       (fv->x[1] - axis_pt[1]) * (1. - SQUARE(dir_angle[1]) / cos_denom) +
                       (fv->x[2] - axis_pt[2]) * (-dir_angle[1] * dir_angle[2] / cos_denom)) /
                      sqrt_sR_sdist;

        dH_U_dX[2] += ((fv->x[0] - axis_pt[0]) * (-dir_angle[2] * dir_angle[0] / cos_denom) +
                       (fv->x[1] - axis_pt[1]) * (-dir_angle[2] * dir_angle[1] / cos_denom) +
                       (fv->x[2] - axis_pt[2]) * (-SQUARE(dir_angle[2]) / cos_denom)) /
                      sqrt_sR_sdist;
      }
    }
  }

  else if (mp->HeightUFunctionModel == CAP_SQUEEZE) {
    H_dot = mp->u_heightU_function_constants[0];
    H_low = mp->u_heightU_function_constants[1];
    R = mp->u_heightU_function_constants[2];
    dbl x_0 = mp->u_heightU_function_constants[3];
    dbl z_0 = mp->u_heightU_function_constants[4];

    *H_U = H_dot * time + H_low + R -
           sqrt(R * R - (fv->x[0] - x_0) * (fv->x[0] - x_0) - (fv->x[2] - z_0) * (fv->x[2] - z_0));
    *dH_U_dtime = H_dot;
    dH_U_dX[0] = (fv->x[0] - x_0) / sqrt(R * R - (fv->x[0] - x_0) * (fv->x[0] - x_0) -
                                         (fv->x[2] - z_0) * (fv->x[2] - z_0));
    dH_U_dX[2] = (fv->x[2] - z_0) / sqrt(R * R - (fv->x[0] - x_0) * (fv->x[0] - x_0) -
                                         (fv->x[2] - z_0) * (fv->x[2] - z_0));
  }

  else if ((mp->HeightUFunctionModel == FLAT_GRAD_FLAT) ||
           (mp->HeightUFunctionModel == FLAT_GRAD_FLAT_MELT)) {

    // Read in parameters
    dbl x1 = mp->u_heightU_function_constants[0];
    dbl h1 = mp->u_heightU_function_constants[1];
    dbl x2 = mp->u_heightU_function_constants[2];
    dbl h2 = mp->u_heightU_function_constants[3];
    dbl p = mp->u_heightU_function_constants[4];
    dbl x = fv->x[0];

    // Shortcuts
    dbl pp = pow(p, 2);
    dbl xx = pow(x, 2);
    dbl xx1 = pow(x1, 2);
    dbl xx2 = pow(x2, 2);

    // Define factors
    dbl n = (h2 - h1) / (x2 - x1);
    dbl f = h1 + n * (x - x1);
    dbl z1 = 0.5 + atan(p * (x - x1)) / PI;
    dbl z2 = 0.5 + atan(p * (x - x2)) / PI;
    dbl z1_x = p / (PI * (pp * xx - 2 * pp * x1 * x + pp * xx1 + 1));
    dbl z2_x = p / (PI * (pp * xx - 2 * pp * x2 * x + pp * xx2 + 1));

    // Assemble
    *H_U = (1 - z1) * h1 + z1 * (1 - z2) * f + z2 * h2;
    dH_U_dX[0] = -h1 * z1_x + z1_x * (1 - z2) * f - z1 * z2_x * f + z1 * (1 - z2) * n + z2_x * h2;

    // Add in any melting
    if (mp->HeightUFunctionModel == FLAT_GRAD_FLAT_MELT) {
      *H_U += fv->sh_dh;
    }
  }

  else if (mp->HeightUFunctionModel == POLY_TIME) {

    // Define variables and initialize
    int i;
    dbl np;

    // Read in parameters
    np = mp->len_u_heightU_function_constants;

    // Assemble
    *H_U = mp->u_heightU_function_constants[0];
    for (i = 1; i < np; i++) {
      *H_U += mp->u_heightU_function_constants[i] * pow(time, i);
      *dH_U_dtime += mp->u_heightU_function_constants[i] * pow(time, i - 1) * i;
    }
  }

  else if (mp->HeightUFunctionModel == JOURNAL) {

    // Define variables and initialize
    dbl C, ecc;
    dbl x, y;
    dbl Ri, theta;

    // Read in parameters
    C = mp->u_heightU_function_constants[0];
    ecc = mp->u_heightU_function_constants[1];

    // Read in current point
    x = fv->x[0];
    y = fv->x[1];

    // Calculate cylinder radius
    Ri = sqrt(x * x + y * y);

    // Calculate angle
    if (x > 0) {
      theta = acos(y / Ri);
    } else {
      theta = 2 * PI - acos(y / Ri);
    }

    // Calculate height and slopes
    *H_U = C * (1 + ecc * y / Ri);
    dH_U_dX[0] = -C * ecc / Ri * sin(theta) * cos(theta);
    dH_U_dX[1] = +C * ecc / Ri * sin(theta) * sin(theta);
  }

  else if (mp->HeightUFunctionModel == CIRCLE_MELT) {
    x_0 = mp->u_heightU_function_constants[0];
    r = mp->u_heightU_function_constants[1];
    H_low = mp->u_heightU_function_constants[2];
    Length = fv->x[0] - x_0;
    if (Length > 0.95 * r)
      GOMA_EH(GOMA_ERROR, "Problem in calculating height function model CIRCLE_MELT");

    *H_U = H_low + r - sqrt(r * r - Length * Length) + fv->sh_dh;
    *dH_U_dtime = Length / sqrt(r * r - Length * Length);
    *dH_U_dtime = 0.0;
    *dH_U_ddh = 1.0;
  }

  else if (mp->HeightUFunctionModel == TABLE) {
    struct Data_Table *table_local;
    table_local = MP_Tables[mp->heightU_function_constants_tableid];

    if (!strcmp(table_local->t_name[0], "LINEAR_TIME")) {
      dbl time_local[1];
      time_local[0] = time;
      double dbl_dH_U_dtime;

      *H_U = interpolate_table(table_local, time_local, &dbl_dH_U_dtime, NULL);
      *dH_U_dtime = dbl_dH_U_dtime;
    }
  }

  else if (mp->HeightUFunctionModel == ROLLER) {
    // implement for bar elements in 2d space for now
    double hmin = mp->u_heightU_function_constants[0];
    double r = mp->u_heightU_function_constants[1];
    double xc = mp->u_heightU_function_constants[2];
    double external_field_multiplier = mp->u_heightU_function_constants[3];
    double x = fv->x[0];

    // we're all efv Sherman! It's likely that gap thickness
    // should be defined radially for this problem
    if (external_field_multiplier != 1.0) {
      *H_U = hmin + r - sqrt(SQUARE(r) - SQUARE(x - xc));
    }

    /* add on external field height if there is one. The scale factor will be the fourth user
     * const*/
    if (mp->heightU_ext_field_index >= 0) {
      *H_U += mp->u_heightU_function_constants[3] * fv->external_field[mp->heightU_ext_field_index];
      if (*H_U < 0.0) {
        GOMA_WH(GOMA_ERROR, "read in a negative external field in height_function_model()");
      }
    }
    if (external_field_multiplier != 1.0) {
      dH_U_dX[0] = (x - xc) / sqrt(SQUARE(r) - SQUARE(x - xc));
    }
    // dH_U_DX[0] = dH_ds for my_normal == primitive_s
    // so handle the external field gradients
    if (mp->heightU_ext_field_index >= 0) {

      // load ds_dcsi, that is det_J here
      double det_J;
      double d_det_J_dmeshkj[DIM][MDE];
      memset(d_det_J_dmeshkj, 0.0, sizeof(double) * DIM * MDE);
      detJ_2d_bar(&det_J, d_det_J_dmeshkj);

      int i;
      double dHext_ds, dHext_dcsi;
      dHext_ds = 0.0;
      dHext_dcsi = 0.0;

      // assume that the height field has the same dof as displacement
      for (i = 0; i < ei[pg->imtrx]->dof[MESH_DISPLACEMENT1]; i++) {
        dHext_dcsi += mp->u_heightU_function_constants[3] *
                      *evp->external_field[mp->heightU_ext_field_index][i] *
                      bf[MESH_DISPLACEMENT1]->dphidxi[i][0];
      }

      dHext_ds = dHext_dcsi / det_J;
      dH_U_dX[0] += dHext_ds;
    } // end handling of the external field gradients
  } else {
    GOMA_EH(GOMA_ERROR, "Not a supported height-function model");
  }

  if (mp->HeightLFunctionModel == CONSTANT) {
    *H_L = mp->heightL;
  }

  else if (mp->HeightLFunctionModel == CONSTANT_SPEED || mp->HeightLFunctionModel == WALL_DISTMOD ||
           mp->HeightLFunctionModel == WALL_DISTURB) {
    H_dot = mp->u_heightL_function_constants[0];
    H_init = mp->u_heightL_function_constants[1];
    *H_L = H_dot * time + H_init;

    /* add on external field height if there is one. The scale factor will be the third user const*/
    if (mp->heightL_ext_field_index >= 0 && mp->HeightUFunctionModel != WALL_DISTMOD &&
        mp->HeightUFunctionModel != WALL_DISTURB) {
      *H_L += mp->u_heightL_function_constants[2] * fv->external_field[mp->heightL_ext_field_index];
      dH_L_dX[0] =
          mp->u_heightL_function_constants[2] * fv->grad_ext_field[mp->heightL_ext_field_index][0];
      dH_L_dX[1] =
          mp->u_heightL_function_constants[2] * fv->grad_ext_field[mp->heightL_ext_field_index][1];
      dH_L_dX[2] =
          mp->u_heightL_function_constants[2] * fv->grad_ext_field[mp->heightL_ext_field_index][2];
    }

    *dH_L_dtime = H_dot;
  }

  else if (mp->HeightLFunctionModel == EXTERNAL_FIELD) {
    H_dot = mp->u_heightL_function_constants[0];
    H_init = mp->u_heightL_function_constants[1];
    *H_L = H_dot * time + H_init;

    /* add on external field height if there is one. The scale factor will be the third user const*/
    *H_L += mp->u_heightL_function_constants[2] * fv->external_field[mp->heightL_ext_field_index];
    dH_L_dX[0] =
        mp->u_heightL_function_constants[2] * fv->grad_ext_field[mp->heightL_ext_field_index][0];
    dH_L_dX[1] =
        mp->u_heightL_function_constants[2] * fv->grad_ext_field[mp->heightL_ext_field_index][1];
    dH_L_dX[2] =
        mp->u_heightL_function_constants[2] * fv->grad_ext_field[mp->heightL_ext_field_index][2];
    *dH_L_dtime = H_dot;
  }

  else if (mp->HeightLFunctionModel == ROLL_ON) {
    Length = mp->u_heightL_function_constants[4];
    H_dot = mp->u_heightL_function_constants[3];
    H_delta = mp->u_heightL_function_constants[2];
    x_0 = mp->u_heightL_function_constants[0];
    H_low = mp->u_heightL_function_constants[1];

    *H_L = (H_dot * time + H_delta) * ((fv->x[0] - x_0) / Length) + H_low;
    *dH_L_dtime = H_dot * (fv->x[0] - x_0) / Length;
    dH_L_dX[0] = (H_dot * time + H_delta) / Length;
  }

  else if (mp->HeightLFunctionModel == ROLL) {
    R = mp->u_heightL_function_constants[0];
    /*  origin and direction of rotation axis	*/
    origin[0] = mp->u_heightL_function_constants[1];
    origin[1] = mp->u_heightL_function_constants[2];
    origin[2] = mp->u_heightL_function_constants[3];
    dir_angle[0] = mp->u_heightL_function_constants[4];
    dir_angle[1] = mp->u_heightL_function_constants[5];
    dir_angle[2] = mp->u_heightL_function_constants[6];
    H_dot = mp->u_heightL_function_constants[7];
    origin[2] += H_dot * time;

    /*  find intersection of axis with normal plane - i.e., locate point on
            axis that intersects plane normal to axis that contains local point. */

    cos_denom = (SQUARE(dir_angle[0]) + SQUARE(dir_angle[1]) + SQUARE(dir_angle[2]));
    t = (dir_angle[0] * (fv->x[0] - origin[0]) + dir_angle[1] * (fv->x[1] - origin[1]) +
         dir_angle[2] * (fv->x[2] - origin[2])) /
        cos_denom;
    axis_pt[0] = origin[0] + dir_angle[0] * t;
    axis_pt[1] = origin[1] + dir_angle[1] * t;
    axis_pt[2] = origin[2] + dir_angle[2] * t;

    /*  compute radial direction	*/

    dist = sqrt(SQUARE(fv->x[0] - axis_pt[0]) + SQUARE(fv->x[1] - axis_pt[1]));
    if (dist > fabs(R)) {
      *H_L = -fabs(R);
    } else {
      ADType sqrt_sR_sdist = sqrt(SQUARE(R) - SQUARE(dist));
      *H_L = SGN(R) * (axis_pt[2] - fv->x[2] - sqrt_sR_sdist);
      *dH_L_dtime = 0.; /* finish later  */
      dH_L_dX[0] = 0.;
      dH_L_dX[1] = 0.;
      dH_L_dX[2] = -1.;
      if (DOUBLE_NONZERO(sqrt_sR_sdist)) {
        dH_L_dX[0] += -((fv->x[0] - axis_pt[0]) * (1. - SQUARE(dir_angle[0]) / cos_denom) +
                        (fv->x[1] - axis_pt[1]) * (-dir_angle[0] * dir_angle[1] / cos_denom) +
                        (fv->x[2] - axis_pt[2]) * (-dir_angle[0] * dir_angle[2] / cos_denom)) /
                      sqrt_sR_sdist;
        dH_L_dX[1] += -((fv->x[0] - axis_pt[0]) * (-dir_angle[1] * dir_angle[0] / cos_denom) +
                        (fv->x[1] - axis_pt[1]) * (1. - SQUARE(dir_angle[1]) / cos_denom) +
                        (fv->x[2] - axis_pt[2]) * (-dir_angle[1] * dir_angle[2] / cos_denom)) /
                      sqrt_sR_sdist;
        dH_L_dX[2] += -((fv->x[0] - axis_pt[0]) * (-dir_angle[2] * dir_angle[0] / cos_denom) +
                        (fv->x[1] - axis_pt[1]) * (-dir_angle[2] * dir_angle[1] / cos_denom) +
                        (fv->x[2] - axis_pt[2]) * (-SQUARE(dir_angle[2]) / cos_denom)) /
                      sqrt_sR_sdist;
      }
    }
  }

  else if (mp->HeightLFunctionModel == TABLE) {
    struct Data_Table *table_local;
    table_local = MP_Tables[mp->heightL_function_constants_tableid];
    if (!strcmp(table_local->t_name[0], "LOWER_DISTANCE")) {
      /*
       * The LOWER_DISTANCE model looks at the lower velocity function to determine
       * how far the surface has traveled throughout the run.  It then
       * translates that into a position shift in the x direction.  This actual
       * function is a table look-up that has a lower height function model as a
       * function of distance.  However, that distance is shifted by the motion
       * of the lower surface.
       *
       * AUTHOR:    Scott A. Roberts, 1514
       * DATE:      May 2, 2012
       * LOCATION:  Somewhere over the US
       */

      // Calculate amount that the lower surface has shifted.
      int i;
      double np = 0, time_scale = 0, tn, disp = 0.0;
      if (mp->VeloLFunctionModel == SLIDER_POLY_TIME) {
        np = mp->len_u_veloL_function_constants;
        time_scale = mp->u_veloL_function_constants[0];
        tn = time_scale * time;
        for (i = 1; i < np; i++) {
          disp += mp->u_veloL_function_constants[i] / i * pow(tn, i);
        }
      } else {
        GOMA_EH(GOMA_ERROR, "To use LOWER_DISTANCE for Lower Height Function Model, "
                            "SLIDER_POLY_TIME must be used for the Lower Velocity Function.");
      }

      // Get height from table lookup
      double var1[1], slope;
      var1[0] = fv->x[0] - disp;
      *H_L = interpolate_table(table_local, var1, &slope, NULL);

      // Calculate spatial derivatives
      dH_L_dX[0] = -slope;

      // Calculate time derivative
      double H2;
      tn = time_scale * time * 1.0001;
      for (i = 1; i < np; i++) {
        disp += mp->u_veloL_function_constants[i] / i * pow(tn, i);
      }
      var1[0] = fv->x[0] - disp;
      H2 = interpolate_table(table_local, var1, &slope, NULL);
      *dH_L_dtime = (H2 - *H_L) / (time_scale * time * 0.0001);

    } else {
      GOMA_EH(GOMA_ERROR, "Lower Height Function does not know how to handle this type of TABLE.");
    }
  } else {
    GOMA_EH(GOMA_ERROR, "Not a supported height-function model");
  }

  ADType H_UminusH_L = *H_U - *H_L;
  if (H_UminusH_L < DBL_SEMI_SMALL) {
    H = DBL_SEMI_SMALL;
  } else {
    H = H_UminusH_L;
  }

  // Now would be a good time to implement sidewall effects

  if (mp->HeightUFunctionModel == WALL_DISTMOD || mp->HeightUFunctionModel == WALL_DISTURB ||
      mp->HeightLFunctionModel == WALL_DISTMOD || mp->HeightLFunctionModel == WALL_DISTURB) {
    ADType wall_d, alpha = 0., powerlaw = 1., H_orig = H, F_shift = 0., F_stretch = 1., beta = 0.;
    bool Fwall_model = false;

    if (mp->HeightUFunctionModel == WALL_DISTMOD) {
      wall_d = fv->external_field[mp->heightU_ext_field_index];
      if (mp->len_u_heightU_function_constants > 2)
        alpha = mp->u_heightU_function_constants[2];
    } else if (mp->HeightLFunctionModel == WALL_DISTMOD) {
      wall_d = fv->external_field[mp->heightL_ext_field_index];
      if (mp->len_u_heightL_function_constants > 2)
        alpha = mp->u_heightL_function_constants[2];
    } else {
      wall_d = fv->wall_distance;
      if (mp->len_u_heightU_function_constants > 2)
        alpha = mp->u_heightU_function_constants[2];
      if (mp->len_u_heightL_function_constants > 2)
        alpha = mp->u_heightL_function_constants[2];
    }

    if (mp->len_u_heightU_function_constants > 3) {
      Fwall_model = (bool)mp->u_heightU_function_constants[3];
    } else if (mp->len_u_heightL_function_constants > 3) {
      Fwall_model = (bool)mp->u_heightL_function_constants[3];
    }

    if (mp->len_u_heightU_function_constants > 4) {
      F_shift = mp->u_heightU_function_constants[4];
    } else if (mp->len_u_heightL_function_constants > 4) {
      F_shift = mp->u_heightL_function_constants[4];
    }

    if (mp->len_u_heightU_function_constants > 5) {
      F_stretch = mp->u_heightU_function_constants[5];
    } else if (mp->len_u_heightL_function_constants > 5) {
      F_stretch = mp->u_heightL_function_constants[5];
    }

    if (mp->len_u_heightU_function_constants > 6) {
      powerlaw = mp->u_heightU_function_constants[6];
    } else if (mp->len_u_heightL_function_constants > 6) {
      powerlaw = mp->u_heightL_function_constants[6];
    } else {
      powerlaw = gn->nexp;
    }

    if (mp->len_u_heightU_function_constants > 7) {
      beta = mp->u_heightU_function_constants[7];
    } else if (mp->len_u_heightL_function_constants > 7) {
      beta = mp->u_heightL_function_constants[7];
    }

    // Keep wall distance positive
    ADType rel_dist = std::max(wall_d, 0.0) / H_orig;
    // 3 decimal point accuracy on end of boundary layer
    if (rel_dist <= 3. / alpha * log(10.)) {
      ADType dh_grad = 0., exp_term, exp_term2, exp_termd, tmp = alpha * rel_dist;
      ADType pl_fact = 1. / (2. * powerlaw + 1.);
      int j;
      if (tmp < 0.1) {
        exp_term = tmp * (1. - 0.5 * tmp * (1. - tmp / 3. * (1. - 0.25 * tmp)));
        exp_term = MAX(DBL_SEMI_SMALL, exp_term);
        exp_termd = alpha * (1. - exp_term);
      } else {
        exp_term = 1. - exp(-tmp);
        exp_termd = alpha * exp(-tmp);
      }
      exp_term2 = pow(exp_term, pl_fact);
      if ((ls != NULL || pfd != NULL) && Fwall_model) {
        ADType inv_F_str = 1. / F_stretch;
        // Modified, shifted LS distance & Heaviside variable
        // Doesn´t look like we accomodate NEGATIVE sense yet ...
        ADType F_prime = (DOUBLE_NONZERO(ls->Length_Scale)
                              ? inv_F_str * (2 * ad_fv->F / ls->Length_Scale - F_shift)
                              : ad_fv->F);
        ADType H_prime, dH_prime = 0.0;
        ADType exp_plus = 1., exp_plus2 = 1., exp_plusd = 0.;
        // Modulate beta at the beginning of time
        if (tran->time_value < 10. * tran->Delta_t0) {
          beta *= tran->time_value / (10. * tran->Delta_t0);
        }
        if (tmp < 0.1) {
          exp_plus =
              1. + beta * (1. - tmp * (1. - 0.5 * tmp * (1. - tmp / 3. * (1. - 0.25 * tmp))));
        } else {
          exp_plus = 1. + beta * exp(-tmp);
        }
        exp_plusd = -alpha * (1. - exp_plus);
        exp_plus2 = pow(exp_plus, pl_fact);
        if (F_prime <= 1.) {
          H_prime = 0.5 * (1. + F_prime + sin(PI * F_prime) / PI);
          dH_prime = (1. + cos(PI * F_prime)) * inv_F_str / ls->Length_Scale;
        } else if (F_prime <= -1.) {
          H_prime = 0.0;
        } else {
          H_prime = 1.0;
        }
        ADType factor = (mp->mp2nd->viscositymask[1] ? (1.0 - H_prime) : H_prime);
        ADType dfact_dF = (mp->mp2nd->viscositymask[1] ? (-dH_prime) : dH_prime);
        if (mp->Lub_LS_Interpolation == LOGARITHMIC) {
          if (F_prime > -1. && F_prime < 1.) {
            /*  LS interface zone */
            ADType H_log = log(exp_term);
            ADType Hplus_log = log(exp_plus);
            H = H_orig * pow(exp_term2, factor) * pow(exp_plus2, 1.0 - factor);
            dh_grad = factor * exp_termd / exp_term;
            dh_grad += (1.0 - factor) * exp_plusd / exp_plus;
            dh_grad *= H * pl_fact;
            for (j = 0; j < ei[pg->imtrx]->dof[FILL]; j++) {
              dH_dF[j] = H * pl_fact * (H_log - Hplus_log) * dfact_dF * bf[FILL]->phi[j];
            }
          } else if ((F_prime >= 1. && mp->mp2nd->viscositymask[1]) ||
                     (F_prime <= -1. && mp->mp2nd->viscositymask[0])) {
            /*  In the gas phase  */
            H *= exp_plus2;
            dh_grad = H * pl_fact * exp_plusd / exp_plus;
          } else {
            /*  In the liquid phase  */
            H *= exp_term2;
            dh_grad = H * exp_termd * pl_fact / exp_term;
          }
        } else {
          H *= factor * exp_term2 + (1.0 - factor) * exp_plus2;
          dh_grad = factor * exp_termd * exp_term2 / exp_term;
          dh_grad += (1.0 - factor) * exp_plusd * exp_plus2 / exp_plus;
          dh_grad *= H_orig * pl_fact;
          for (j = 0; j < ei[pg->imtrx]->dof[FILL]; j++) {
            dH_dF[j] = H_orig * (exp_plus2 - exp_term2) * dfact_dF * bf[FILL]->phi[j];
          }
        }
      } else {
        H *= exp_term2;
        dh_grad = H * exp_termd * pl_fact / exp_term;
      }
      if (mp->HeightUFunctionModel == WALL_DISTMOD) {
        dH_U_dX[0] = dh_grad * fv->grad_ext_field[mp->heightU_ext_field_index][0];
        dH_U_dX[1] = dh_grad * fv->grad_ext_field[mp->heightU_ext_field_index][1];
        dH_U_dX[2] = dh_grad * fv->grad_ext_field[mp->heightU_ext_field_index][2];
      } else if (mp->HeightLFunctionModel == WALL_DISTMOD) {
        dH_L_dX[0] = -dh_grad * fv->grad_ext_field[mp->heightL_ext_field_index][0];
        dH_L_dX[1] = -dh_grad * fv->grad_ext_field[mp->heightL_ext_field_index][1];
        dH_L_dX[2] = -dh_grad * fv->grad_ext_field[mp->heightL_ext_field_index][2];
      } else if (mp->HeightUFunctionModel == WALL_DISTURB) {
        dH_U_dX[0] = dh_grad * fv->grad_wall_distance[0];
        dH_U_dX[1] = dh_grad * fv->grad_wall_distance[1];
        dH_U_dX[2] = dh_grad * fv->grad_wall_distance[2];
      } else if (mp->HeightLFunctionModel == WALL_DISTURB) {
        dH_L_dX[0] = -dh_grad * fv->grad_wall_distance[0];
        dH_L_dX[1] = -dh_grad * fv->grad_wall_distance[1];
        dH_L_dX[2] = -dh_grad * fv->grad_wall_distance[2];
      }
    }
    if (H < DBL_SEMI_SMALL) {
      H = DBL_SEMI_SMALL;
      dH_U_dX[0] = dH_U_dX[1] = dH_U_dX[2] = 0.;
      dH_L_dX[0] = dH_L_dX[1] = dH_L_dX[2] = 0.;
    }
  }
  return (H);
}