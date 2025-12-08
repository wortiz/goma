#include "ad/level_set.h"
#include "ad/lubrication.h"
#include "ad/structs.h"
#include "ad/turbulence.h"
#include <memory>

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

int ad_level_set_property(
    const ADType p0, const ADType p1, const double width, ADType &pp) {

  /* Fetch the level set interfacial functions. */
  ad_load_lsi(width);

  /* Calculate the material property. */
  if (ls->Elem_Sign == -1)
    pp = p0;
  else if (ls->Elem_Sign == 1)
    pp = p1;
  else
    pp = p0 + (p1 - p0) * ad_lsi->H;

  return (0);
}

int ad_level_set_property_log(
    const ADType p0, const ADType p1, const double width, ADType &pp) {

  /* Fetch the level set interfacial functions. */
  ad_load_lsi(width);

  /* Calculate the material property. */
  if (ls->Elem_Sign == -1)
    pp = p0;
  else if (ls->Elem_Sign == 1)
    pp = p1;
  else
    pp = pow(p0, 1.0 - lsi->H) * pow(p1, lsi->H);

  return (0);
}


ADType ad_ls_modulate_property(ADType p1,
                            ADType p2,
                            double width,
                            double pm_minus,
                            double pm_plus,
                            ADType &factor,
                            const int interp_method) {
  ADType p_plus, p_minus, p;

  p_minus = p1 * pm_plus + p2 * pm_minus;
  p_plus = p1 * pm_minus + p2 * pm_plus;

  if (interp_method == LSI_INTERP_LINEAR)
    ad_level_set_property(p_minus, p_plus, width, p);
  else if (interp_method == LSI_INTERP_LOG)
    ad_level_set_property_log(p_minus, p_plus, width, p);
  else {
    GOMA_EH(-1, "Unknown level set interface interpolation method");
    return 0.0;
  }

  if (ls->Elem_Sign == -1) {
    factor = pm_plus;
  } else if (ls->Elem_Sign == 1) {
    factor = pm_minus;
  } else {
    if (interp_method == LSI_INTERP_LINEAR) {
      factor = pm_plus * (1.0 - lsi->H) + pm_minus * lsi->H;
    } else if (interp_method == LSI_INTERP_LOG) {
      factor = pm_minus * pow(p2, 1.0 - lsi->H) * lsi->H * pow(p1, lsi->H - 1.0) +
                pm_plus * (1.0 - lsi->H) * pow(p1, -lsi->H) * pow(p2, lsi->H);
    } else {
      GOMA_EH(-1, "Unknown level set interface interpolation method");
    }
  }

  return (p);
}

std::unique_ptr<AD_Level_Set_Interface> ad_lsi = nullptr;

void ad_get_convective_velocity(ADType v_conv[DIM],
                           dbl xi[DIM],
                          Exo_DB *exo) {
  int a;

  /* Start with fluid velocity */
  for (a = 0; a < DIM; a++) {
    v_conv[a] = ad_fv->v[a];
  }
  
  if (pd->gv[R_LUBP]) {
    ad_calculate_lub_q_v(R_LUBP, tran->time_value, tran->delta_t, xi, exo);
    /* Add in lubrication velocity */
    for (a = 0; a < DIM; a++) {
      v_conv[a] = AD_LubAux->v_avg[a];
    }
  }

  /* Subtract mesh velocity if deforming mesh */
  if (pd->gv[MESH_DISPLACEMENT1] && pd->TimeIntegration != STEADY) {
    for (a = 0; a < DIM; a++) {
      v_conv[a] -= ad_fv->x_dot[a];
    }
  }
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
  ADType grad_II_F[DIM]; /* Fill surface gradient */
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
 ADType grad_II_phi_i[DIM];
  dbl d_grad_II_phi_i_dmesh[DIM][DIM][MDE];

  dbl phi_j;       /* j-th basis function of a field variable. */
  dbl *grad_phi_j; /* Gradient of phi_j. */
  dbl grad_II_phi_j[DIM];
  dbl h3;        /* Volume element (scale factors). */
  ADType det_J;     /* Determinant of the Jacoabian of transformation. */
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
  det_J = ad_fv->detJ;                   /* Really, ought to be mesh eqn. */
  dtinv = 1.0 / dt;                        /* Ah, 1 / dt. */

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
  if (pd->gv[R_LUBP]) {
      // if ( tran->Fill_Weight_Fcn == FILL_WEIGHT_G  ) {
      lubon = 1;
  }
  int *n_dof = NULL;
  int dof_map[MDE];
  
  if (pd->gv[R_LUBP]) {
    n_dof = (int *)array_alloc(1, MAX_VARIABLE_TYPES, sizeof(int));
    lubrication_shell_initialize(n_dof, dof_map, -1, xi, exo, 0);
    ADInn(ad_fv->grad_F, grad_II_F);
    det_J = fv->sdet;
  }
  else if (pd->e[pg->imtrx][R_LUBP_2]) {
    GOMA_EH(
        GOMA_ERROR,
        " if you have a fill equation turned on in the R_LUBP_2 phase, you are in the wrong place");
  }

  ADType v_conv[DIM];
  ad_get_convective_velocity(v_conv, xi, exo);
  ADType supg_tau = 0.0;
      switch (Fill_Weight_Fcn) {
      case FILL_WEIGHT_SUPG: /* Streamline Upwind Petrov Galerkin (SUPG) */
      case FILL_WEIGHT_SUPG_GP:
      case FILL_WEIGHT_SUPG_SHAKIB: {
        ad_supg_tau_shakib(supg_tau, dim, dt, 1e-8, FILL);
      } break;
      default:
      break;
    }
      
  std::vector<ADType> resid(ei[pg->imtrx]->dof[eqn], ADType(0.0));
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
       ADType mass = 0.0;
        ADType advection = 0.0;
      switch (Fill_Weight_Fcn) {
      case FILL_WEIGHT_G: /* Plain ol' Galerkin method */
      case FILL_WEIGHT_SUPG: /* Streamline Upwind Petrov Galerkin (SUPG) */
      case FILL_WEIGHT_SUPG_GP:
      case FILL_WEIGHT_SUPG_SHAKIB: {
        ADType v_dot_Dphi = 0.0;
        ADType v_dot_gradF = 0.0;
        if (lubon) {
          ADInn( ad_fv->basis[var].grad_phi[i], grad_II_phi_i);
          for (int a = 0; a < dim; a++) {
            v_dot_Dphi += v_conv[a] * grad_II_phi_i[a];
            v_dot_gradF += v_conv[a] * grad_II_F[a];
          }

        } else {
          for (int a = 0; a < dim; a++) {
            v_dot_Dphi += v_conv[a] * bf[var]->grad_phi[i][a];
            v_dot_gradF += v_conv[a] * ad_fv->grad_F[a];
          }
        }
        ADType wt_func = bf[eqn]->phi[i] + supg_tau * v_dot_Dphi;

        mass = ad_fv->F_dot * wt_func;
        advection = v_dot_gradF * wt_func;

      } break;

      default:
        GOMA_EH(GOMA_ERROR, "Unknown Fill_Weight_Fcn ad_assemble_fill");
      }
      mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
      advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];


      resid[i] += (mass+advection) * wt * det_J * h3 ;
      lec->R[LEC_R_INDEX(peqn, i)] +=
          (mass.val() + advection.val()) * wt * det_J.val() * h3;
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
        for (int var = V_FIRST; var < V_LAST; var++) {

          /* Sensitivity w.r.t. velocity */
          if (pd->v[pg->imtrx][var]) {
            int pvar = upd->vp[pg->imtrx][var];

            for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
              // J = &(lec->J[LEC_J_INDEX(peqn, pvar, ii, 0)]);
              lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += resid[i].dx(ad_fv->offset[var] + j);

            } /* End of loop over j */
          } /* End of if the variale is active */
        }
    } /* for 'i': FILL DoFs */

  } /* if: af->Assemble_Jacobian */

  /* clean-up */
  fv->wt = wt; /* load_neighbor_var_data screws this up */
  safe_free((void *)n_dof);

  return (status);

} /* end of assemble_fill */
static inline ADType set_ad_or_dbl(dbl val, int eqn, int dof) {
  ADType tmp;
  if (ad_fv->total_ad_variables > 0 && af->Assemble_Jacobian == TRUE) {
    if (pd->gv[eqn]) {
      tmp = ADType(ad_fv->total_ad_variables, ad_fv->offset[eqn] + dof, val);
    } else {
      tmp = val;
    }
  } else {
    tmp = val;
  }
  return tmp;
}

int ad_load_lsi(const double width) {
  if (!ad_lsi) {
    ad_lsi = std::make_unique<AD_Level_Set_Interface>();
  }
  ADType F = ad_fv->F, alpha;
  int a;
  int i, j;

  /* Check if we're in the mushy zone. */
  ad_lsi->alpha = 0.5 * width;
  alpha = ad_lsi->alpha;

  ad_lsi->near = ls->on_sharp_surf || fabs(F) < alpha;
  lsi->near = ad_lsi->near;

  /* Calculate the interfacial functions we want to know even if not in mushy
   * zone. */

  ad_lsi->gfmag = 0.0;
  for (a = 0; a < VIM; a++) {
    ad_lsi->normal[a] = ad_fv->grad_F[a];
    ad_lsi->gfmag += ad_fv->grad_F[a] * ad_fv->grad_F[a];
  }
  ad_lsi->gfmag = sqrt(ad_lsi->gfmag);
  if (ad_lsi->gfmag.val() == 0.0) {
    ad_lsi->gfmaginv = 1.0;
  } else {
    ad_lsi->gfmaginv = 1.0 / ad_lsi->gfmag;
  }

  for (a = 0; a < VIM; a++) {
    ad_lsi->normal[a] *= ad_lsi->gfmaginv;
  }

  /* If we're not in the mushy zone: */
  if (ls->on_sharp_surf) {
    /*ad_lsi->H = ( F < 0.0) ? 0.0 : 1.0 ;*/
    ad_lsi->H = (ls->Elem_Sign < 0) ? 0.0 : 1.0;
    ad_lsi->delta = 1.;
  } else if (!ad_lsi->near) {
    ad_lsi->H = (F < 0.0) ? 0.0 : 1.0;
    ad_lsi->delta = 0.;
  } else {
    ad_lsi->H = 0.5 * (1. + F / alpha + sin(M_PIE * F / alpha) / M_PIE);
    ad_lsi->delta = 0.5 * (1. + cos(M_PIE * F / alpha)) * ad_lsi->gfmag / alpha;
  }

  /**** Shield the operations below since they are very expensive relative to
     the previous operations in the load_lsi routine. Add your variables as
     needed  ********/

  if (pd->gv[LUBP] || pd->gv[LUBP_2] || pd->gv[SHELL_SAT_CLOSED] || pd->gv[SHELL_PRESS_OPEN] ||
      pd->gv[SHELL_PRESS_OPEN_2] || pd->gv[SHELL_SAT_GASN]) {

    /* Evaluate heaviside using FEM basis functions */
    ADType Hni, d_Hni_dF, Fi;
    ADType Hni_old, Fi_old;
    int eqn = R_FILL;
    ad_lsi->Hn = 0.0;
    ad_lsi->Hn_old = 0.0;
    for (a = 0; a < DIM; a++) {
      ad_lsi->gradHn[a] = 0.0;
      ad_lsi->gradHn_old[a] = 0.0;
    }
    if (pd->gv[LUBP] || pd->gv[SHELL_SAT_CLOSED] || pd->gv[SHELL_PRESS_OPEN] ||
        pd->gv[SHELL_SAT_GASN]) {
      for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
        Fi = set_ad_or_dbl(*esp->F[i], FILL, i);
        if (fabs(Fi) > ad_lsi->alpha) {
          Hni = (Fi < 0.0) ? 0.0 : 1.0;
        } else {
          Hni = 0.5 * (1.0 + Fi / ad_lsi->alpha + sin(M_PIE * Fi / ad_lsi->alpha) / M_PIE);
        }
        ad_lsi->Hn += Hni * bf[eqn]->phi[i];
        for (j = 0; j < VIM; j++) {
          ad_lsi->gradHn[j] += Hni * ad_fv->basis[eqn].grad_phi[i][j];
        }

        Fi_old = *esp_old->F[i];
        if (fabs(Fi_old) > ad_lsi->alpha) {
          Hni_old = (Fi_old < 0.0) ? 0.0 : 1.0;
        } else {
          Hni_old = 0.5 * (1.0 + Fi_old / ad_lsi->alpha + sin(M_PIE * Fi_old / ad_lsi->alpha) / M_PIE);
        }
        ad_lsi->Hn_old += Hni_old * bf[eqn]->phi[i];
        for (j = 0; j < VIM; j++) {
          ad_lsi->gradHn_old[j] += Hni_old * bf[eqn]->grad_phi[i][j];
        }
      }
    } else if (pd->gv[LUBP_2] || pd->gv[SHELL_PRESS_OPEN_2]) {
      eqn = R_PHASE1;
      for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
        Fi = *esp->pF[0][i];
        if (fabs(Fi) > ad_lsi->alpha) {
          Hni = (Fi < 0.0) ? 0.0 : 1.0;
          d_Hni_dF = 0.0;
        } else {
          Hni = 0.5 * (1.0 + Fi / ad_lsi->alpha + sin(M_PIE * Fi / ad_lsi->alpha) / M_PIE);
          d_Hni_dF = 0.5 * (1 / ad_lsi->alpha + cos(M_PIE * Fi / ad_lsi->alpha) / ad_lsi->alpha);
        }
        ad_lsi->Hn += Hni * bf[eqn]->phi[i];
        for (j = 0; j < VIM; j++) {
          ad_lsi->gradHn[j] += Hni * bf[eqn]->grad_phi[i][j];
        }

        Fi_old = *esp_old->pF[0][i];
        if (fabs(Fi_old) > ad_lsi->alpha) {
          Hni_old = (Fi_old < 0.0) ? 0.0 : 1.0;
        } else {
          Hni_old = 0.5 * (1.0 + Fi_old / ad_lsi->alpha + sin(M_PIE * Fi_old / ad_lsi->alpha) / M_PIE);
        }
        ad_lsi->Hn_old += Hni_old * bf[eqn]->phi[i];
        for (j = 0; j < VIM; j++) {
          ad_lsi->gradHn_old[j] += Hni_old * bf[eqn]->grad_phi[i][j];
        }
      }
    }

  } /* end of if pd->v[LUBP] || ... etc */

  /************ End of shielding **************************/

  if (fabs(alpha) < 1e-32) {
    alpha = 1e-32;
  }
  ad_lsi->delta_max = ad_lsi->gfmag / alpha;

  return (0);
}