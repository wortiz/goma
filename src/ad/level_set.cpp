#include "ad/level_set.h"
#include "ad/structs.h"

extern "C" {
#include "exo_struct.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_fill_aux.h"
#include "mm_fill_fill.h"
#include "mm_fill_stabilization.h"
#include "mm_shell_util.h"
#include "rf_allo.h"
#include "rf_fem.h"
#include "std.h" /* This needs to be here. */
}

extern "C" int ad_assemble_fill(double tt,
                                double dt,
                                const PG_DATA *pg_data,
                                const int applied_eqn,
                                double xi[3],
                                Exo_DB *exo,
                                double time,
                                struct LS_Mass_Lumped_Penalty *mass_lumped_penalty) {
  /******************************************************************************
   *
   * assemble_fill -- integrate fill equation.  This routine assembles
   *                  the fill equations with the other physics equations.
   *
   * in:
   *      tt -- Time integration parameter.
   *      dt -- Current time step size.
   *
   * Created: Thu Mar  3 07:48:01 MST 1994 pasacki@sandia.gov
   *
   * Revised: 9/24/94 by RRR
   * Revised: 9/24/01 by PKN
   *
   *
   * Note: currently we do a "double load" into the addresses in the global
   *       "a" matrix and resid vector held in "esp", as well as into the
   *       local accumulators in "lec".
   *
   ******************************************************************************/
  int eqn, var, peqn, pvar, dim, status;
  int i, j, a, b, c;

  dbl F_dot;          /* Fill derivative wrt time. */
  dbl *grad_F;        /* Fill gradient. */
  dbl grad_II_F[DIM]; /* Fill surface gradient */
  dbl d_grad_II_F_dmesh[DIM][DIM][MDE];

  dbl *v;     /* Local velocity. */
  dbl *v_old; /* Old v[]. */

  dbl *xx;    /* Nodal coordinates. */
  dbl *x_old; /* Old xx[]. */

  dbl x_dot[DIM];         /* Time derivative of the mesh displacements. */
  /*dbl x_dot_old[DIM];*/ /* Old x_dot[]. */
  dbl *x_dot_old;         /* Old x_dot[]. */

  dbl v_rel[DIM];     /* Velocity relative to the mesh. */
  dbl v_rel_old[DIM]; /* Old v_rel[]. */

  dbl zero[3] = {0.0, 0.0, 0.0}; /* An array of zeros, for convienience. */

  dbl phi_i;       /* i-th basis function for the FILL equation. */
  dbl *grad_phi_i; /* Gradient of phi_i. */
  dbl grad_II_phi_i[DIM];
  dbl d_grad_II_phi_i_dmesh[DIM][DIM][MDE];

  dbl phi_j;       /* j-th basis function of a field variable. */
  dbl *grad_phi_j; /* Gradient of phi_j. */
  dbl grad_II_phi_j[DIM];
  dbl h3;        /* Volume element (scale factors). */
  dbl det_J;     /* Determinant of the Jacoabian of transformation. */
  dbl wt;        /* Gauss point weight. */
  dbl rmp[MDE];  /* Hold on to the integrands from the residuals. */
  dbl wfcn;      /* The weight function. */
  dbl d_wfcn_du; /* Deriv. of wfcn w.r.t. the fluid velocity. */
  dbl d_wfcn_dx; /* Deriv. of wfcn w.r.t. the mesh displacement. */
  dbl mass = 0.0, advection = 0.0;
  dbl v_dot_Dphi[MDE];  /* v.grad(phi) */
  dbl vc_dot_Dphi[MDE]; /* vcent.grad(phi) */
  dbl v_dot_DF;         /* v.grad(F) */
  dbl dtinv;            /* = 1 / dt */
  int Fill_Weight_Fcn;  /* Fill weight function. */

  /* See get_supg_stuff() for a better description of these variables */
  dbl supg_term;                    /* Major term for SUPG -- see get_supg_stuff(). */
  dbl vcent[DIM];                   /* Element centroid velocity (get_supg_stuff(). */
  dbl d_vcent_du[DIM][MDE][DIM];    /* deriv. of vcent[] w.r.t. nodal velocities.   */
  dbl d_supg_term_du[MDE][DIM];     /* deriv. of supg_term w.r.t. nodal velocities. */
  dbl d_supg_term_dx[MDE][DIM];     /* deriv. of supg_term w.r.t. mesh coords.      */
  dbl d_vrel_d_x_rs[DIM][DIM][MDE]; /* deriv of solid rel velo w.r.t. real-solid displ */

  double vmag_old, tau_gls;
  double h_elem;

  /* Alternative newer SUPG style */
  SUPG_terms supg_terms;

  status = 0;
  eqn = applied_eqn;

  /*
   * Bail out fast if there's nothing to do...
   */

  if (!pd->e[pg->imtrx][eqn]) {
    return (status);
  }

  /*
   * Unpack variables from structures for local convenience...
   */

  dim = pd->Num_Dim;                       /* Number of dimensions. */
  Fill_Weight_Fcn = tran->Fill_Weight_Fcn; /* Which weight function to use */
  wt = fv->wt;                             /* Gauss point weight. */
  h3 = fv->h3;                             /* Differential volume element. */
  det_J = bf[eqn]->detJ;                   /* Really, ought to be mesh eqn. */
  dtinv = 1.0 / dt;                        /* Ah, 1 / dt. */

  if (eqn == R_FILL) {
    if (pd->TimeIntegration != STEADY) {
      F_dot = fv_dot->F;
    } else {
      F_dot = 0.0;
    }
    grad_F = fv->grad_F;
  } else {
    if (pd->TimeIntegration != STEADY) {
      F_dot = fv_dot->pF[ls->var - PHASE1];
    } else {
      F_dot = 0.0;
    }
    grad_F = fv->grad_pF[ls->var - PHASE1];
    Fill_Weight_Fcn = FILL_WEIGHT_EXPLICIT;
  }

  h_elem = 0.;
  for (a = 0; a < dim; a++)
    h_elem += pg_data->hsquared[a];
  /* This is the size of the element */
  h_elem = sqrt(h_elem / ((double)dim));

  /* On 1/11/01, MMH changed these loops from dim's to VIM's.  This
   * was necessary for the PROJECTED_CARTESIAN coordinate system and
   * 3d stability of 2d flow.  The only coordinate system this could
   * possibly cause problems for is the CYLINDRICAL one.  In
   * CYLINDRICAL coordinates, VIM = 3, but there are not always 3
   * components to vectors (b/c the theta-velocity is assumed to be
   * zero).  It passed the test suite, though.
   */

  /*
   * Calculate lubrication velocity for direct integration
   */
  int lubon = 0;
  if (pd->e[pg->imtrx][R_LUBP]) {
    if (tran->Fill_Weight_Fcn == FILL_WEIGHT_G || tran->Fill_Weight_Fcn == FILL_WEIGHT_TG) {
      // if ( tran->Fill_Weight_Fcn == FILL_WEIGHT_G  ) {
      lubon = 1;
    } else {
      GOMA_WH(-1, "\n Multiphase lubrication should be run with Galerkin weighting to \n take "
                  "advantage of direct velocity calculations.  \n Talk to SAR.");
      lubon = 0;
    }
  }
  int *n_dof = NULL;
  int dof_map[MDE];
  if (pd->e[pg->imtrx][R_LUBP]) {

    /* Set up shells */
    n_dof = (int *)array_alloc(1, MAX_VARIABLE_TYPES, sizeof(int));
    lubrication_shell_initialize(n_dof, dof_map, -1, xi, exo, 0);

    /* Calculate velocity */
    calculate_lub_q_v(R_LUBP, time, dt, xi, exo);
    calculate_lub_q_v_old(R_LUBP, tran->time_value_old, tran->delta_t_old, xi, exo);

    /* Set up weights */
    wt = fv->wt;
    h3 = fv->h3;
    det_J = fv->sdet;

    Inn(grad_F, grad_II_F);
  }
  if (pd->e[pg->imtrx][R_LUBP_2]) {

    GOMA_EH(
        GOMA_ERROR,
        " if you have a fill equation turned on in the R_LUBP_2 phase, you are in the wrong place");
    /* Set up shells */
    n_dof = (int *)array_alloc(1, MAX_VARIABLE_TYPES, sizeof(int));
    lubrication_shell_initialize(n_dof, dof_map, -1, xi, exo, 0);

    /* Calculate velocity */
    calculate_lub_q_v(R_LUBP_2, time, dt, xi, exo);
    calculate_lub_q_v_old(R_LUBP_2, tran->time_value_old, tran->delta_t_old, xi, exo);

    /* Set up weights */
    wt = fv->wt;
    h3 = fv->h3;
    det_J = fv->sdet;

    Inn(grad_F, grad_II_F);
  }

  /* Use pointers unless we need to do algebra. */
  if (lubon) {
    v = LubAux->v_avg;
    v_old = LubAux_old->v_avg;
    xx = fv->x;
    x_old = fv_old->x;
  } else {
    v = fv->v;
    v_old = fv_old->v;
    xx = fv->x;
    x_old = fv_old->x;
  }
  if (eqn == R_PHASE1) {
    memset(v, 0, sizeof(double) * VIM);
    memset(v_old, 0, sizeof(double) * VIM);
  }

  v_dot_DF = 0.0;
  if (pd->TimeIntegration != STEADY && pd->gv[MESH_DISPLACEMENT1]) {
    x_dot_old = fv_dot_old->x;
    for (a = 0; a < VIM; a++) {
      x_dot[a] = (1. + 2. * tt) * (xx[a] - x_old[a]) * dtinv - 2. * tt * x_dot_old[a];
      if (lubon)
        x_dot[a] = (1 + 2 * tt) / dt * (xx[a] - x_old[a]);
      v_rel[a] = v[a] - x_dot[a];
      v_rel_old[a] = v_old[a] - x_dot_old[a];
      if (lubon) {
        v_dot_DF += v_rel[a] * grad_II_F[a]; /* v.gradII(F) */
      } else {
        v_dot_DF += v_rel[a] * grad_F[a]; /* v.grad(F) */
      }
    }
    if (lubon)
      ShellRotate(grad_F, fv->d_grad_F_dmesh, grad_II_F, d_grad_II_F_dmesh,
                  n_dof[MESH_DISPLACEMENT1]);
  } else if (pd->TimeIntegration != STEADY && pd->etm[pg->imtrx][R_SOLID1][(LOG2_MASS)] &&
             pd->MeshMotion == TOTAL_ALE) /*This is the Eulerian solid-mech case */
  {
    xx = fv->d_rs;
    x_old = fv_old->d_rs;
    x_dot_old = fv_dot_old->d_rs;
    for (a = 0; a < VIM; a++) {
      x_dot[a] = (1. + 2. * tt) * (xx[a] - x_old[a]) * dtinv - 2. * tt * x_dot_old[a];
    }
    for (a = 0; a < VIM; a++) {
      v_rel[a] = x_dot[a];
      v_rel_old[a] = x_dot_old[a];
      for (b = 0; b < VIM; b++) {
        v_rel[a] -= x_dot[b] * fv->grad_d_rs[b][a];
        v_rel_old[a] -= x_dot_old[b] * fv_old->grad_d_rs[b][a];
      }
    }
    for (a = 0; a < VIM; a++) {
      if (lubon) {
        v_dot_DF += v_rel[a] * grad_II_F[a]; /* v.gradII(F) */
      } else {
        v_dot_DF += v_rel[a] * grad_F[a]; /* v.grad(F) */
      }
    }

  } else {
    x_dot_old = zero;
    for (a = 0; a < VIM; a++) {
      x_dot[a] = 0.0;
      v_rel[a] = v[a];
      v_rel_old[a] = v_old[a];
      if (lubon) {
        v_dot_DF += v_rel[a] * grad_II_F[a]; /* v.gradII(F) */
      } else {
        v_dot_DF += v_rel[a] * grad_F[a]; /* v.grad(F) */
      }
    }
  }

  /* Get the SUPG stuff, if necessary. */
  if (Fill_Weight_Fcn == FILL_WEIGHT_SUPG) {
    memset(vcent, 0, sizeof(double) * DIM);
    memset(d_vcent_du, 0, sizeof(double) * DIM * MDE * DIM);
    memset(d_supg_term_du, 0, sizeof(double) * MDE * DIM);
    memset(d_supg_term_dx, 0, sizeof(double) * MDE * DIM);
    memset(vc_dot_Dphi, 0, sizeof(double) * MDE);
    supg_term = 0.;
    get_supg_stuff(&supg_term, vcent, d_vcent_du, d_supg_term_du, d_supg_term_dx,
                   pd->e[pg->imtrx][R_MESH1]);
  }

  if (Fill_Weight_Fcn == FILL_WEIGHT_SUPG_SHAKIB || Fill_Weight_Fcn == FILL_WEIGHT_SUPG_GP) {
    supg_tau(&supg_terms, dim, 0, pg_data, dt, Fill_Weight_Fcn == FILL_WEIGHT_SUPG_SHAKIB, R_FILL);
  }

  /* Compute and save v.grad(phi) and vcent.grad(phi). */
  memset(v_dot_Dphi, 0, sizeof(double) * MDE);

  for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
    /* So: grad_phi_i[a] == bf[var]->grad_phi[i][a] */
    grad_phi_i = bf[eqn]->grad_phi[i];

    if (lubon) {
      Inn(grad_phi_i, grad_II_phi_i);

      for (a = 0; a < VIM; a++) {
        v_dot_Dphi[i] += v_rel[a] * grad_II_phi_i[a];
      }
    } else {
      for (a = 0; a < VIM; a++) {
        v_dot_Dphi[i] += v_rel[a] * grad_phi_i[a];
        if (Fill_Weight_Fcn == FILL_WEIGHT_SUPG) {
          vc_dot_Dphi[i] += vcent[a] * grad_phi_i[a];
        }
      }
    }
  }

  vmag_old = 0.;
  for (a = 0; a < dim; a++) {
    vmag_old += fv_old->v[a] * fv_old->v[a];
  }
  vmag_old = sqrt(vmag_old);
  tau_gls = 0.0;
  if (Fill_Weight_Fcn == FILL_WEIGHT_EXPLICIT) {
    tau_gls =
        1. / sqrt((2. / dt) * (2. / dt) + (2. * vmag_old / h_elem) * (2. * vmag_old / h_elem));
  }

  if (ls != NULL && ls->Semi_Implicit_Integration &&
      (Fill_Weight_Fcn == FILL_WEIGHT_SUPG_SHAKIB || Fill_Weight_Fcn == FILL_WEIGHT_SUPG_GP)) {
    F_dot = (fv->F - fv_old->F) / dt;
    v_dot_DF = 0;
    for (int a = 0; a < VIM; a++) {
      v_dot_DF += (0.5 * (fv->v[a] + fv_old->v[a])) * (0.5 * (fv->grad_F[a] + fv_old->grad_F[a]));
    }

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      v_dot_Dphi[i] = 0;
      for (int a = 0; a < VIM; a++) {
        v_dot_Dphi[i] += (0.5 * (fv->v[a] + fv_old->v[a])) * bf[eqn]->grad_phi[i][a];
      }
    }
  } else if (ls != NULL && !ls->Semi_Implicit_Integration &&
             (Fill_Weight_Fcn == FILL_WEIGHT_SUPG_SHAKIB ||
              Fill_Weight_Fcn == FILL_WEIGHT_SUPG_GP)) {
    F_dot = fv_dot->F;
    v_dot_DF = 0;
    for (int a = 0; a < VIM; a++) {
      v_dot_DF += fv->v[a] * fv->grad_F[a];
    }

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      v_dot_Dphi[i] = 0;
      for (int a = 0; a < VIM; a++) {
        v_dot_Dphi[i] += fv->v[a] * bf[eqn]->grad_phi[i][a];
      }
    }
  } else if (ls != NULL && ls->Semi_Implicit_Integration) {
    GOMA_EH(GOMA_ERROR, "Error Level Set Semi-Implicit Time Integration can only be used "
                        "with SUPG_GP and SUPG_SHAKIB");
    return -1;
  }

  double p_e = 0;
  double d_p_e[MDE];
  double d_inv_F_norm[MDE];
  double inv_F_norm = 0;
  if (ls != NULL && ls->Toure_Penalty) {
    for (int i = 0; i < ei[pg->imtrx]->dof[R_FILL]; i++) {
      p_e += mass_lumped_penalty->penalty[i] * bf[R_FILL]->phi[i];
      if (ls->Semi_Implicit_Integration) {
        p_e += mass_lumped_penalty->penalty_old[i] * bf[R_FILL]->phi[i];
      }
    }
    if (ls->Semi_Implicit_Integration) {
      p_e *= 0.5;
    }

    for (int k = 0; k < ei[pg->imtrx]->dof[R_FILL]; k++) {
      d_p_e[k] = 0;
      for (int i = 0; i < ei[pg->imtrx]->dof[R_FILL]; i++) {
        d_p_e[k] += mass_lumped_penalty->d_penalty[i][k] * bf[R_FILL]->phi[i];
      }
      if (ls->Semi_Implicit_Integration) {
        d_p_e[k] *= 0.5;
      }
    }

    double F_norm = 0;
    for (int i = 0; i < dim; i++) {
      F_norm += fv->grad_F[i] * fv->grad_F[i];
    }
    F_norm = sqrt(F_norm);
    inv_F_norm = 1 / (F_norm + 1e-32);
    for (int j = 0; j < ei[pg->imtrx]->dof[eqn]; j++) {
      d_inv_F_norm[j] = 0;
      for (int i = 0; i < dim; i++) {
        d_inv_F_norm[j] += 2 * fv->grad_F[i] * bf[eqn]->grad_phi[j][i];
      }
      d_inv_F_norm[j] *= -0.5 * inv_F_norm * inv_F_norm * inv_F_norm;
    }
  }

  dbl k_dc = 0;
  // dbl d_k_dc[MDE] = {0};
  if (ls != NULL && ls->YZbeta != SC_NONE) {
    dbl strong_residual = 0;
    strong_residual = fv_dot_old->F;
    for (int p = 0; p < VIM; p++) {
      strong_residual += fv->v[p] * fv_old->grad_F[p];
    }
    // strong_residual -= s_terms.MassSource[w];
    dbl h_elem = 0;
    for (int a = 0; a < ei[pg->imtrx]->ielem_dim; a++) {
      h_elem += pg_data->hsquared[a];
    }
    /* This is the size of the element */
    h_elem = sqrt(h_elem / ((double)ei[pg->imtrx]->ielem_dim));

    dbl inner = 0;
    for (int i = 0; i < dim; i++) {
      inner += fv_old->grad_F[i] * fv_old->grad_F[i];
    }

    dbl yzbeta = 0;

    dbl inv_sqrt_inner = (1 / sqrt(inner + 1e-12));
    dbl dc1 = fabs(strong_residual) * inv_sqrt_inner * h_elem * 0.5;
    dbl dc2 = fabs(strong_residual) * h_elem * h_elem * 0.25;

    // dc1 = fmin(supg_terms.supg_tau,dc1);//0.5*(dc1 + dc2);
    yzbeta = fmin(supg_terms.supg_tau, 0.5 * (dc1 + dc2)); // 0.5*(dc1 + dc2);
    //    for (int k = 0; k <  ei[pg->imtrx]->dof[eqn]; k++) {
    //      d_k_dc[k] = 0;
    //    }

    k_dc = yzbeta;
  }

  dbl pspg[3] = {0.0, 0.0, 0.0};
  PSPG_DEPENDENCE_STRUCT d_pspg;
  if (upd->PSPG_advection_correction) {
    calc_pspg(pspg, &d_pspg, time, tt, dt, pg_data);
  }

  /**********************************************************************
   **********************************************************************
   ** Residuals
   **********************************************************************
   **********************************************************************/

  if (af->Assemble_Residual) {
    peqn = upd->ep[pg->imtrx][eqn];
    if (ls == NULL)
      var = FILL;
    else
      var = ls->var;
    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      phi_i = bf[eqn]->phi[i];

      /************************************************************
       * Assemble according to the weight function selected.
       ************************************************************/
      switch (Fill_Weight_Fcn) {
      case FILL_WEIGHT_TG: /* Taylor-Galerkin */

        mass = F_dot * phi_i;

        advection = 0.;

        if (lubon) {
          for (a = 0; a < dim; a++) {
            advection += phi_i * 0.5 * (v_rel[a] + v_rel_old[a]) * grad_II_F[a];
          }
        } else {
          for (a = 0; a < dim; a++) {
            advection += phi_i * 0.5 * (v_rel[a] + v_rel_old[a]) * grad_F[a];
          }
        }
        advection += v_dot_Dphi[i] * v_dot_DF * dt * 0.5;

        break;
      case FILL_WEIGHT_EXPLICIT:

        wfcn = phi_i;
        for (a = 0; a < dim; a++)
          wfcn += tau_gls * fv_old->v[a] * bf[eqn]->grad_phi[i][a];

        mass = F_dot * wfcn;

        advection = 0.;
        for (a = 0; a < dim; a++)
          advection += wfcn * v_rel_old[a] * fv->grad_F[a];
        if (pd->e[pg->imtrx][R_EXT_VELOCITY] && (pfd == NULL || eqn == R_PHASE1))
          advection += fv_old->ext_v * wfcn;

        break;
      case FILL_WEIGHT_G: /* Plain ol' Galerkin method */

        mass = F_dot * phi_i;
        advection = v_dot_DF * phi_i;
        if (pd->e[pg->imtrx][R_EXT_VELOCITY] && (pfd == NULL || eqn == R_PHASE1))
          advection += fv->ext_v * phi_i;

        break;

      case FILL_WEIGHT_SUPG: /* Streamline Upwind Petrov Galerkin (SUPG) */

        mass = F_dot * (vc_dot_Dphi[i] * supg_term + phi_i);
        advection = v_dot_DF * (vc_dot_Dphi[i] * supg_term + phi_i);

        break;

      case FILL_WEIGHT_SUPG_GP:
      case FILL_WEIGHT_SUPG_SHAKIB: {
        dbl wt_func = bf[eqn]->phi[i] + supg_terms.supg_tau * v_dot_Dphi[i];

        mass = F_dot * wt_func;
        advection = v_dot_DF;

        if (upd->PSPG_advection_correction) {
          advection -= pspg[0] * grad_F[0];
          advection -= pspg[1] * grad_F[1];
          if (VIM == 3)
            advection -= pspg[2] * grad_F[2];
        }

        if (ls != NULL && ls->Enable_Div_Term) {
          advection += fv->F * fv->div_v;
        }

        advection *= wt_func;
      } break;

      default:

        GOMA_EH(GOMA_ERROR, "Unknown Fill_Weight_Fcn");
      }
      mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
      advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

      dbl discontinuity_capturing = 0;
      for (int a = 0; a < dim; a++) {
        discontinuity_capturing += k_dc * fv->grad_F[a] * bf[eqn]->grad_phi[i][a];
      }

      dbl source = 0;
      if (ls != NULL && ls->Toure_Penalty) {
        for (int a = 0; a < dim; a++) {
          source += p_e * fv->grad_F[a] * inv_F_norm * bf[eqn]->grad_phi[i][a];
          //          double tmp = 1.0 / sqrt(fv->grad_F[0] * fv->grad_F[0] +
          //                                  fv->grad_F[1] * fv->grad_F[1]);
          //          source += p_e * fv->grad_F[a] * tmp *
          //          bf[eqn]->grad_phi[i][a];
        }
      }

      /* hang on to the integrand (without the "dV") for use below. */
      rmp[i] = mass + advection + source + discontinuity_capturing;

      lec->R[LEC_R_INDEX(peqn, i)] +=
          (mass + advection + source + discontinuity_capturing) * wt * det_J * h3;
    }
  }

  /**********************************************************************
   **********************************************************************
   * Jacobian terms...
   **********************************************************************
   **********************************************************************/

  if (af->Assemble_Jacobian) {
    peqn = upd->ep[pg->imtrx][eqn];
    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

      phi_i = bf[eqn]->phi[i];

      /* So: grad_phi_i[a] == bf[var]->grad_phi[i][a] */
      grad_phi_i = bf[eqn]->grad_phi[i];

      if (lubon) {
        ShellBF(eqn, i, &phi_i, grad_phi_i, grad_II_phi_i, d_grad_II_phi_i_dmesh,
                n_dof[MESH_DISPLACEMENT1], dof_map);
      }

      /*
       * Set up some preliminaries that are needed for the (a,i)
       * equation for bunches of (b,j) column variables...
       */

      /* The weight function for Galerkin and SUPG */
      wfcn = phi_i;
      if (Fill_Weight_Fcn == FILL_WEIGHT_SUPG) {
        wfcn += vc_dot_Dphi[i] * supg_term;
      }

      /* The weight function for SUPG */
      if (Fill_Weight_Fcn == FILL_WEIGHT_SUPG) {
        wfcn = 0.;

        /* vcent "dot" grad_phi */
        for (a = 0; a < dim; a++)
          wfcn += vcent[a] * bf[eqn]->grad_phi[i][a];

        wfcn *= supg_term;
        wfcn += bf[eqn]->phi[i];
      } else if (Fill_Weight_Fcn == FILL_WEIGHT_G) {
        wfcn = bf[eqn]->phi[i];
      }

      /*************************************************************
       *
       * Derivatives of fill equation w.r.t. to FILL variable
       *
       *************************************************************/

      if (ls == NULL)
        var = FILL;
      else
        var = ls->var;
      if (pd->v[pg->imtrx][var]) {
        pvar = upd->vp[pg->imtrx][var];
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          phi_j = bf[eqn]->phi[j];

          /* So: grad_phi_j[a] == bf[var]->grad_phi[j][a] */
          grad_phi_j = bf[eqn]->grad_phi[j];

          if (lubon)
            Inn(grad_phi_j, grad_II_phi_j);

          /*
           * Use the selected weight function
           */
          switch (Fill_Weight_Fcn) {
          case FILL_WEIGHT_TG: /* Taylor-Galerkin */

            mass = phi_i * phi_j * (1. + 2. * tt) * dtinv;
            advection = 0.;
            for (a = 0; a < VIM; a++) {
              if (lubon) {
                advection += 0.5 * (v_rel[a] + v_rel_old[a]) * grad_II_phi_j[a] * phi_i;
              } else {
                advection += 0.5 * (v_rel[a] + v_rel_old[a]) * grad_phi_j[a] * phi_i;
              }
            }
            advection += v_dot_Dphi[j] * v_dot_Dphi[i] * dt * 0.5;
            break;
#if 0
                    case FILL_WEIGHT_EXPLICIT:

		      mass = phi_i * phi_j * (1. + 2. * tt) * dtinv;
		      advection = 0.;
		      break;
#endif
#if 0
                    case FILL_WEIGHT_EXPLICIT:
		      wfcn = phi_i;
                      for (a = 0; a < dim; a++) wfcn += tau_gls * fv_old->v[a] * bf[eqn]->grad_phi[i][a];

		      mass = wfcn * phi_j * (1. + 2. * tt) * dtinv;
		      advection = 0.;
		      break;
#endif
#if 1
          case FILL_WEIGHT_EXPLICIT:
            wfcn = phi_i;
            for (a = 0; a < dim; a++)
              wfcn += tau_gls * fv_old->v[a] * bf[eqn]->grad_phi[i][a];

            mass = wfcn * phi_j * (1. + 2. * tt) * dtinv;
            advection = 0.;
            for (a = 0; a < dim; a++)
              advection += wfcn * v_rel_old[a] * bf[eqn]->grad_phi[j][a];
            break;
#endif
          case FILL_WEIGHT_G:    /* Plain ol' Galerkin method */
          case FILL_WEIGHT_SUPG: /* Streamline Upwind Petrov Galerkin (SUPG) */

            mass = (phi_j * (1. + 2. * tt) * dtinv) * wfcn;
            advection = v_dot_Dphi[j] * wfcn;
            if (lubon) {
              for (a = 0; a < dim; a++)
                advection += LubAux->dv_avg_df[a][j] * grad_II_F[a] * wfcn;
            }

            break;

          case FILL_WEIGHT_SUPG_GP:
          case FILL_WEIGHT_SUPG_SHAKIB: {
            dbl wt_func = bf[eqn]->phi[i] + supg_terms.supg_tau * v_dot_Dphi[i];
            if (ls != NULL && ls->Semi_Implicit_Integration) {
              mass = (phi_j) / dt * wt_func;
              advection = 0.5 * v_dot_Dphi[j] * wt_func;
            } else {

              mass = (phi_j * (1. + 2. * tt) * dtinv) * wt_func;
              advection = 0;
              for (int a = 0; a < dim; a++) {
                advection += fv->v[a] * bf[var]->grad_phi[j][a];
              }
              if (upd->PSPG_advection_correction) {
                advection -= pspg[0] * bf[var]->grad_phi[j][0];
                advection -= pspg[1] * bf[var]->grad_phi[j][1];
                if (VIM == 3)
                  advection -= pspg[2] * bf[var]->grad_phi[j][2];
              }

              if (ls != NULL && ls->Enable_Div_Term) {
                advection += bf[var]->phi[j] * fv->div_v;
              }
              advection *= wt_func;
            }
          } break;

          default:

            GOMA_EH(GOMA_ERROR, "Unknown Fill_Weight_Fcn");

          } /* switch(Fill_Weight_Fcn) */
          dbl source = 0;
          if (ls != NULL && ls->Toure_Penalty) {
            for (int a = 0; a < dim; a++) {
              source += p_e * fv->grad_F[a] * d_inv_F_norm[j] * bf[eqn]->grad_phi[i][a];
              source += p_e * bf[eqn]->grad_phi[j][a] * inv_F_norm * bf[eqn]->grad_phi[i][a];
              source += d_p_e[j] * fv->grad_F[a] * inv_F_norm * bf[eqn]->grad_phi[i][a];
              //              source +=
              //                  fv->grad_F[a] * d_inv_F_norm * bf[eqn]->grad_phi[i][a];
            }
          }

          mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
          advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

          dbl discontinuity_capturing = 0;
          for (int a = 0; a < dim; a++) {
            discontinuity_capturing += k_dc * bf[var]->grad_phi[j][a] * bf[eqn]->grad_phi[i][a];
          }

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] +=
              (mass + advection + source + discontinuity_capturing) * wt * h3 * det_J;

        } /* for: FILL DoFs */

      } /* if: FILL exisits */

      /*************************************************************
       *
       * Derivatives of fill equation w.r.t. to VELOCITY variables
       *
       *************************************************************/
      for (b = 0; b < VIM; b++) {
        var = VELOCITY1 + b;
        if (pd->v[pg->imtrx][var]) {

          pvar = upd->vp[pg->imtrx][var];
          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

            phi_j = bf[var]->phi[j];

            switch (Fill_Weight_Fcn) {
            case FILL_WEIGHT_TG: /* Taylor Galerkin */

              mass = 0.;
              advection = grad_phi_i[b] * v_dot_DF;
              advection += v_dot_Dphi[i] * grad_F[b];
              advection *= phi_j * dt * 0.5;
              advection += 0.5 * phi_j * phi_i * grad_F[b];
              if (lubon)
                advection = 0.0;

              break;

            case FILL_WEIGHT_EXPLICIT:

              mass = 0.;
              advection = 0.;

              break;

            case FILL_WEIGHT_G: /* Plain ol' Galerkin method */

              mass = 0.;
              advection = phi_j * grad_F[b] * phi_i;
              if (lubon)
                advection = 0.0;

              break;

            case FILL_WEIGHT_SUPG: /* Streamline Upwind Petrov Galerkin (SUPG) */

              d_wfcn_du = d_supg_term_du[j][b] * vc_dot_Dphi[i];
              for (a = 0; a < dim; a++) {
                d_wfcn_du += supg_term * d_vcent_du[a][j][b] * grad_phi_i[a];
              }

              mass = F_dot * d_wfcn_du;
              advection = v_dot_DF * d_wfcn_du;
              advection += phi_j * grad_F[b] * wfcn;

              break;
            case FILL_WEIGHT_SUPG_GP:
            case FILL_WEIGHT_SUPG_SHAKIB: {
              dbl wt_func = bf[eqn]->phi[i];
              for (int a = 0; a < dim; a++) {
                wt_func += supg_terms.supg_tau * fv->v[a] * bf[eqn]->grad_phi[i][a];
              }

              dbl d_wt_func = 0;
              for (int a = 0; a < dim; a++) {
                d_wt_func += supg_terms.supg_tau * bf[eqn]->phi[j] * bf[eqn]->grad_phi[i][a] +
                             supg_terms.d_supg_tau_dv[b][j] * fv->v[a] * bf[eqn]->grad_phi[i][a];
              }
              mass = fv_dot->F * d_wt_func;
              advection = 0;
              for (int a = 0; a < dim; a++) {
                advection += fv->v[a] * bf[eqn]->grad_phi[j][a];
              }
              advection *= d_wt_func;
              advection += bf[var]->phi[j] * bf[eqn]->grad_phi[j][b] * wt_func;
            } break;

            default:

              GOMA_EH(GOMA_ERROR, "Unknown Fill_Weight_Fcn");

            } /* switch(Fill_Weight_Fcn) */

            mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
            advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

            lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += (mass + advection) * wt * det_J * h3;
          }
        }
      }
      var = EXT_VELOCITY;
      if (pd->v[pg->imtrx][var]) {

        pvar = upd->vp[pg->imtrx][var];
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          phi_j = bf[var]->phi[j];
          switch (Fill_Weight_Fcn) {
          case FILL_WEIGHT_TG: /* Taylor Galerkin */
          case FILL_WEIGHT_EXPLICIT:

            mass = 0.;
            advection = 0.;

            break;

          case FILL_WEIGHT_G: /* Plain ol' Galerkin method */

            mass = 0.;
            advection = 0.;
            if (pd->e[pg->imtrx][R_EXT_VELOCITY] && (pfd == NULL || eqn == R_PHASE1))
              advection += phi_j * phi_i;

            break;

          case FILL_WEIGHT_SUPG: /* Streamline Upwind Petrov Galerkin (SUPG) */

            d_wfcn_du = d_supg_term_du[j][b] * vc_dot_Dphi[i];
            for (a = 0; a < dim; a++) {
              d_wfcn_du += supg_term * d_vcent_du[a][j][b] * grad_phi_i[a];
            }

            mass = F_dot * d_wfcn_du;
            advection = v_dot_DF * d_wfcn_du;
            advection += phi_j * grad_F[b] * wfcn;

            break;

          default:

            GOMA_EH(GOMA_ERROR, "Unknown Fill_Weight_Fcn");

          } /* switch(Fill_Weight_Fcn) */

          mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
          advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += (mass + advection) * wt * det_J * h3;
        }
      }

      /*************************************************************
       *
       * Derivatives of fill equation w.r.t. to MESH_DISPLACEMENT variables
       *
       *************************************************************/
      for (b = 0; b < VIM; b++) {
        var = MESH_DISPLACEMENT1 + b;
        if (pd->v[pg->imtrx][var]) {

          if (lubon) {

            pvar = upd->vp[pg->imtrx][var];
            for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
              c = dof_map[j];
              phi_j = bf[var]->phi[j];

              switch (Fill_Weight_Fcn) {

              case FILL_WEIGHT_TG: /* Taylor Galerkin */

                mass = 0.;
                mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];

                advection = 0.0;
                for (a = 0; a < dim; a++) {
                  advection += 0.5 * LubAux->dv_avg_dx[a][b][j] * grad_II_F[a] * phi_i;
                  advection -= 0.5 * (1 + 2 * tt) / dt * phi_j * delta(a, b) * grad_II_F[a] * phi_i;
                  advection += 0.5 * (v_rel[a] + v_rel_old[a]) * d_grad_II_F_dmesh[a][b][j] * phi_i;

                  advection += LubAux->dv_avg_dx[a][b][j] * grad_II_phi_i[a] * v_dot_DF * dt * 0.5;
                  advection -= (1 + 2 * tt) / dt * phi_j * delta(a, b) * grad_II_phi_i[a] *
                               v_dot_DF * dt * 0.5;
                  advection += v_rel[a] * d_grad_II_phi_i_dmesh[a][b][j] * v_dot_DF * dt * 0.5;

                  advection += v_dot_Dphi[i] * LubAux->dv_avg_dx[a][b][j] * grad_II_F[a] * dt * 0.5;
                  advection -= v_dot_Dphi[i] * (1 + 2 * tt) / dt * phi_j * delta(a, b) *
                               grad_II_F[a] * dt * 0.5;
                  advection += v_dot_Dphi[i] * v_rel[a] * d_grad_II_F_dmesh[a][b][j] * dt * 0.5;
                }
                advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                break;

              case FILL_WEIGHT_G: /* Plain ol' Galerkin */

                mass = 0.;
                mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];

                advection = 0.0;
                for (a = 0; a < dim; a++) {
                  //			    advection += LubAux->dv_avg_dx[a][b][j] * grad_F[a] *
                  // phi_i; 			    advection -= (1+2*tt)/dt * phi_j * delta(a,b) *
                  // grad_F[a]
                  // *
                  // phi_i; 			    advection += v_rel[a] *
                  // fv->d_grad_F_dmesh[a][b][j]
                  // * phi_i;

                  advection += LubAux->dv_avg_dx[a][b][j] * grad_II_F[a] * phi_i;
                  advection -= (1 + 2 * tt) / dt * phi_j * delta(a, b) * grad_II_F[a] * phi_i;
                  advection += v_rel[a] * d_grad_II_F_dmesh[a][b][j] * phi_i;
                }
                advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                break;
              }
              lec->J[LEC_J_INDEX(peqn, pvar, i, c)] += (mass + advection) * wt * h3 * fv->sdet;
              lec->J[LEC_J_INDEX(peqn, pvar, i, c)] += rmp[i] * wt * h3 * fv->dsurfdet_dx[b][c];
            } /* j loop */

          } else {

            pvar = upd->vp[pg->imtrx][var];
            for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

              phi_j = bf[var]->phi[j];

              /* So: grad_phi_j[a] == bf[var]->grad_phi[j][a] */
              grad_phi_j = bf[var]->grad_phi[j];

              switch (Fill_Weight_Fcn) {
              case FILL_WEIGHT_TG: /* Taylor Galerkin */

                mass = 0.;
                advection = -0.5 * phi_j * (1. + 2. * tt) * dtinv * phi_i * grad_F[b];
                advection += -0.5 * dt * phi_j * grad_phi_j[b] * v_dot_DF;

                break;

              case FILL_WEIGHT_EXPLICIT:

                mass = 0.;
                advection = 0.;

                break;

              case FILL_WEIGHT_G: /* Plain ol' Galerkin */

                mass = 0.;
                advection = -phi_j * (1. + 2. * tt) * dtinv * grad_F[b];

                break;

              case FILL_WEIGHT_SUPG: /* Streamline Upwind Petrov Galerkin (SUPG) */

                d_wfcn_dx = d_supg_term_dx[j][b] * vc_dot_Dphi[i];
                for (a = 0; a < dim; a++) {
                  d_wfcn_dx += supg_term * vcent[a] * bf[eqn]->d_grad_phi_dmesh[i][a][b][j];
                }

                mass = F_dot * d_wfcn_dx;
                advection = v_dot_DF * d_wfcn_dx;
                advection += -phi_j * (1. + 2. * tt) * dtinv * grad_F[b] * wfcn;

                break;
              case FILL_WEIGHT_SUPG_GP:
              case FILL_WEIGHT_SUPG_SHAKIB: {
                dbl wt_func = bf[eqn]->phi[i];
                for (int a = 0; a < dim; a++) {
                  wt_func += supg_terms.supg_tau * fv->v[a] * bf[eqn]->grad_phi[i][a];
                }

                dbl d_wt_func = 0;
                for (int a = 0; a < dim; a++) {
                  d_wt_func +=
                      supg_terms.supg_tau * fv->v[a] * bf[eqn]->d_grad_phi_dmesh[i][a][b][j] +
                      supg_terms.d_supg_tau_dX[b][j] * fv->v[a] * bf[eqn]->grad_phi[i][a];
                }
                mass = fv_dot->F * d_wt_func;
                advection = 0;
                for (int a = 0; a < dim; a++) {
                  advection += fv->v[a] * bf[eqn]->grad_phi[j][a];
                }
                advection *= d_wt_func;
                dbl advection_b = 0;
                for (int a = 0; a < dim; a++) {
                  advection_b += fv->v[a] * bf[eqn]->d_grad_phi_dmesh[j][a][b][j];
                }
                advection += advection_b * wt_func;
              } break;

              default:

                GOMA_EH(GOMA_ERROR, "Unknown Fill_Weight_Fcn");

              } /* switch(Fill_Weight_Fcn) */

              mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
              advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

              lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += (mass + advection) * wt * det_J * h3;

              /* Derivatives of the dV part. */
              /* rmp[i] holds the integrand without the "dV" part. */

              /* lec->J[LEC_J_INDEX(peqn,pvar,i,j)] += rmp[i] * wt * det_J * fv->dh3dq[b] *
               * bf[var]->phi[j]; */
              lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += rmp[i] * wt * det_J * fv->dh3dq[b] * phi_j;
              lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += rmp[i] * wt * h3 * bf[eqn]->d_det_J_dm[b][j];

            } /* for 'j': MESH DoFs */
          }

        } /* if: MESH exists? */

      } /* for 'b': MESH componenets */

      /*************************************************************
       *
       * Derivatives of fill equation w.r.t. to SHELL NORMAL variables
       *
       *************************************************************/
      for (b = 0; b < VIM; b++) {
        var = SHELL_NORMAL1 + b;
        if (pd->v[pg->imtrx][var]) {

          if (lubon) {

            pvar = upd->vp[pg->imtrx][var];
            for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

              advection = 0.0;
              for (a = 0; a < dim; a++) {
                advection += LubAux->dv_avg_dnormal[a][b][j] * grad_F[a] * phi_i;
              }
              advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

              lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += advection * wt * h3 * fv->sdet;
            } /* j loop */
          }

        } /* if: SHELL NORMAL exists? */

      } /* for 'b': SHELL NORMAL components */

      /*************************************************************
       *
       * Derivatives of fill equation w.r.t. to REAL_SOLID displacements
       * for Eulerian solid mechanics
       *
       *************************************************************/

      memset(d_vrel_d_x_rs, 0, sizeof(double) * DIM * DIM * MDE);
      for (a = 0; a < VIM; a++) {
        for (b = 0; b < VIM; b++) {
          var = SOLID_DISPLACEMENT1 + b;
          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            phi_j = bf[var]->phi[j];

            d_vrel_d_x_rs[a][b][j] += phi_j * (1. + 2. * tt) * dtinv * delta(a, b);

            grad_phi_j = bf[var]->grad_phi[j];

            d_vrel_d_x_rs[a][b][j] -= (1. + 2. * tt) * phi_j * dtinv * fv->grad_d_rs[b][a];

            for (c = 0; c < VIM; c++) {
              /*grad_phi_e[i][j][k][l] = e_k e_l : grad(phi_i e_j )*/
              d_vrel_d_x_rs[a][b][j] -= x_dot[c] * bf[var]->grad_phi_e[j][b][c][a];
            }
          }
        }
      }
      for (b = 0; b < VIM; b++) {
        var = SOLID_DISPLACEMENT1 + b;
        if (pd->v[pg->imtrx][var]) {

          pvar = upd->vp[pg->imtrx][var];
          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {

            phi_j = bf[var]->phi[j];

            /* So: grad_phi_j[a] == bf[var]->grad_phi[j][a] */
            grad_phi_j = bf[var]->grad_phi[j];

            switch (Fill_Weight_Fcn) {
            case FILL_WEIGHT_TG: /* Taylor Galerkin */

              for (a = 0; a < VIM; a++) {
                advection += phi_i * 0.5 * d_vrel_d_x_rs[a][b][j] * grad_F[a];
              }

              for (a = 0; a < VIM; a++) {
                advection += (v_dot_Dphi[i] * d_vrel_d_x_rs[a][b][j] * grad_F[a] +
                              d_vrel_d_x_rs[a][b][j] * grad_phi_i[a] * v_dot_DF) *
                             dt * 0.5;
              }

              break;

            case FILL_WEIGHT_EXPLICIT:

              advection = 0.;

              break;

            case FILL_WEIGHT_G: /* Plain ol' Galerkin */

              advection = phi_j * (1. + 2. * tt) * dtinv * grad_F[b] * phi_i;
              for (a = 0; a < VIM; a++) {
                GOMA_EH(GOMA_ERROR, "This is easy.  Need to copy essentials from TG case above");
              }

              break;

            case FILL_WEIGHT_SUPG: /* Streamline Upwind Petrov Galerkin (SUPG) */

              GOMA_EH(GOMA_ERROR,
                      "Level set fill for Eulerian solid mech incompatible for FILL_WEIGHT_SUPG");

              break;

            default:

              GOMA_EH(GOMA_ERROR, "Unknown Fill_Weight_Fcn");

            } /* switch(Fill_Weight_Fcn) */

            advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

            lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += advection * wt * det_J * h3;

          } /* for 'j': real-solid DoFs */

        } /* if: real-solid exists? */

      } /* for 'b': real solid componenets */

      /*************************************************************
       *
       * Derivatives of fill equation w.r.t. to LUBP variable
       *
       *************************************************************/

      var = LUBP;
      if (pd->v[pg->imtrx][var] && lubon) {
        pvar = upd->vp[pg->imtrx][var];
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          phi_j = bf[eqn]->phi[j];
          grad_phi_j = bf[eqn]->grad_phi[j];

          Inn(grad_phi_j, grad_II_phi_j);

          mass = 0.0;
          advection = 0.0;

          switch (Fill_Weight_Fcn) {

          case FILL_WEIGHT_TG: /* Taylor-Galerkin */
            for (a = 0; a < dim; a++) {
              advection +=
                  0.5 * LubAux->dv_avg_dp2[a] * phi_j *
                  (grad_II_F[a] * (phi_i + v_dot_Dphi[i] * dt) + grad_II_phi_i[a] * v_dot_DF * dt);
              for (b = 0; b < dim; b++) {
                advection += 0.5 * LubAux->dv_dgradp[a][b] * grad_II_phi_j[b] *
                             (grad_II_F[a] * (phi_i + v_dot_Dphi[i] * dt) +
                              grad_II_phi_i[a] * v_dot_DF * dt);
              }
            }

            break;

          case FILL_WEIGHT_G: /* Plain ol' Galerkin */
            for (a = 0; a < dim; a++) {
              advection += LubAux->dv_avg_dp2[a] * grad_II_F[a] * wfcn * grad_II_phi_j[a];
              for (b = 0; b < dim; b++) {
                advection += LubAux->dv_dgradp[a][b] * grad_II_F[a] * wfcn * grad_II_phi_j[b];
              }
            }

            break;
          }

          mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
          advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += (mass + advection) * wt * h3 * det_J;

        } /* for: LUBP DoFs */

      } /* if: LUBP exisits */

      /*************************************************************
       *
       * Derivatives of fill equation w.r.t. to SHELL_LUB_CURV variable
       *
       *************************************************************/

      var = SHELL_LUB_CURV;
      if (pd->v[pg->imtrx][var] && lubon) {
        pvar = upd->vp[pg->imtrx][var];
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          phi_j = bf[eqn]->phi[j];
          grad_phi_j = bf[eqn]->grad_phi[j];

          mass = 0.0;
          advection = 0.0;

          switch (Fill_Weight_Fcn) {

          case FILL_WEIGHT_TG: /* Taylor-Galerkin */
            for (a = 0; a < dim; a++) {

              advection +=
                  0.5 * LubAux->dv_avg_dk[a] * phi_j *
                  (grad_II_F[a] * (phi_i + v_dot_Dphi[i] * dt) + grad_II_phi_i[a] * v_dot_DF * dt);
            }

            break;

          case FILL_WEIGHT_G: /* Plain ol' Galerkin */
            for (a = 0; a < dim; a++)
              advection += LubAux->dv_avg_dk[a] * grad_II_F[a] * phi_i * phi_j;

            break;
          }

          mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
          advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += (mass + advection) * wt * h3 * det_J;

        } /* for: SHELL_LUB_CURV DoFs */

      } /* if: SHELL_LUB_CURV exisits */

      /*************************************************************
       *
       * Derivatives of fill equation w.r.t. to SHELL_DELTAH variable
       *
       *************************************************************/

      var = SHELL_DELTAH;
      if (pd->v[pg->imtrx][var] && lubon) {
        pvar = upd->vp[pg->imtrx][var];
        for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          phi_j = bf[eqn]->phi[j];
          grad_phi_j = bf[eqn]->grad_phi[j];

          mass = 0.0;
          advection = 0.0;
          for (a = 0; a < dim; a++)
            advection += LubAux->dv_avg_ddh[a] * grad_F[a] * phi_i * phi_j;

          mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
          advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

          lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += (mass + advection) * wt * h3 * det_J;

        } /* for: SHELL_DELTAH DoFs */

      } /* if: SHELL_DELTAH exisits */

    } /* for 'i': FILL DoFs */

  } /* if: af->Assemble_Jacobian */

  /* clean-up */
  fv->wt = wt; /* load_neighbor_var_data screws this up */
  safe_free((void *)n_dof);

  return (status);

} /* end of assemble_fill */
