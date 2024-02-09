#define GOMA_AD_MOMENTUM_CPP
#include <Sacado.hpp>

#include "ad_momentum.h"
#include "ad_turbulence.h"

extern "C" {

/* GOMA include files */
#include "ad_turbulence.h"
#include "density.h"
#include "el_elm.h"
#include "exo_struct.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_eh.h"
#include "mm_fill_aux.h"
#include "mm_fill_ls.h"
#include "mm_fill_ptrs.h"
#include "mm_fill_rs.h"
#include "mm_fill_solid.h"
#include "mm_fill_species.h"
#include "mm_fill_stabilization.h"
#include "mm_fill_stress.h"
#include "mm_mp.h"
#include "mm_mp_const.h"
#include "mm_mp_structs.h"
#include "mm_qtensor_model.h"
#include "mm_viscosity.h"
#include "rf_allo.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_solver.h"
#include "std.h"
#include "user_mp.h"
}

/* ad_assemble_momentum -- assemble terms (Residual &| Jacobian) for momentum eqns
 *
 * in:
 * 	ei -- pointer to Element Indeces	structure
 *	pd -- pointer to Problem Description	structure
 *	af -- pointer to Action Flag		structure
 *	bf -- pointer to Basis Function		structure
 *	fv -- pointer to Field Variable		structure
 *	cr -- pointer to Constitutive Relation	structure
 *	md -- pointer to Mesh Derivative	structure
 *	me -- pointer to Material Entity	structure
 * 	ija -- vector of pointers into the a matrix
 * 	a  -- global Jacobian matrix
 * 	R  -- global residual vector
 *
 * out:
 *	a   -- gets loaded up with proper contribution
 *	lec -- gets loaded up with local contributions to resid, Jacobian
 * 	r  -- residual RHS vector
 *
 * Created:	Wed Dec  8 14:03:06 MST 1993 pasacki@sandia.gov
 *
 * Revised:	Sun Feb 27 06:53:12 MST 1994 pasacki@sandia.gov
 *
 *
 * Note: currently we do a "double load" into the addresses in the global
 *       "a" matrix and resid vector held in "esp", as well as into the
 *       local accumulators in "lec".
 *
 */

int ad_assemble_momentum(dbl time,       /* current time */
                         dbl tt,         /* parameter to vary time integration from
                                            explicit (tt = 1) to implicit (tt = 0) */
                         dbl dt,         /* current time step size */
                         dbl h_elem_avg, /* average global element size for PSPG*/
                         const PG_DATA *pg_data,
                         double xi[DIM], /* Local stu coordinates */
                         const Exo_DB *exo) {

  int dim;
  int i, p, q, a;

  int ledof, eqn, ii, peqn;

  int *pde = pd->e[pg->imtrx];
  int *pdv = pd->v[pg->imtrx];

  int status;
  struct Basis_Functions *bfm;

  dbl h3; /* Volume element (scale factors). */

  /* field variables */

  dbl rho; /* Density. */

  ADType f[DIM]; /* Body force. */

  dbl det_J; /* determinant of element Jacobian */
  dbl d_area;

  /*
   *
   * Note how carefully we avoid refering to d(phi[i])/dx[j] and refer instead
   * to the j-th component of grad_phi[j][i] so that this vector can be loaded
   * up with components that may be different in non Cartesian coordinate
   * systems.
   *
   * We will, however, insist on *orthogonal* coordinate systems, even if we
   * might permit them to be curvilinear.
   *
   * Assume all components of velocity are interpolated with the same kind
   * of basis function.
   */

  /*
   * Galerkin weighting functions for i-th and a-th momentum residuals
   * and some of their derivatives...
   */
  dbl phi_i;
  dbl(*grad_phi_i_e_a)[DIM] = NULL;
  /*
   * Interpolation functions for variables and some of their derivatives.
   */

  ADType Pi[DIM][DIM];

  dbl wt;

  int v_g[DIM][DIM];
  int v_s[MAX_MODES][DIM][DIM];

  int transient_run = pd->TimeIntegration != STEADY;
  int mass_on;
  int advection_on = 0;
  int source_on = 0;
  int diffusion_on = 0;

  dbl mass_etm, advection_etm, diffusion_etm, source_etm;

  double *R;

  /*
   * Petrov-Galerkin weighting functions for i-th residuals
   * and some of their derivatives...
   */

  int *n_dof = NULL;

  status = 0;

  /*
   * Unpack variables from structures for local convenience...
   */

  eqn = R_MOMENTUM1;

  /*
   * Bail out fast if there's nothing to do...
   */
  if (!pd->e[pg->imtrx][eqn]) {
    return (status);
  }

  dim = pd->Num_Dim;

  wt = fv->wt;

  det_J = bf[eqn]->detJ; /* Really, ought to be mesh eqn. */

  h3 = fv->h3; /* Differential volume element (scales). */

  d_area = det_J * wt * h3;

  dbl supg = 0.;

  if (mp->Mwt_funcModel == GALERKIN) {
    supg = 0.;
  } else if (mp->Mwt_funcModel == SUPG || mp->Mwt_funcModel == SUPG_GP ||
             mp->Mwt_funcModel == SUPG_SHAKIB) {
    supg = mp->Mwt_func;
  }

  /*** Density ***/
  rho = density(NULL, time);

  ADType tau;
  if (supg != 0.) {
    ad_only_tau_momentum_shakib(tau, dim, dt, FALSE);
  }
  /* end Petrov-Galerkin addition */

  if (pd->gv[POLYMER_STRESS11]) {
    (void)stress_eqn_pointer(v_s);
  }

  if (pd->gv[VELOCITY_GRADIENT11]) {
    v_g[0][0] = VELOCITY_GRADIENT11;
    v_g[0][1] = VELOCITY_GRADIENT12;
    v_g[1][0] = VELOCITY_GRADIENT21;
    v_g[1][1] = VELOCITY_GRADIENT22;
    v_g[0][2] = VELOCITY_GRADIENT13;
    v_g[1][2] = VELOCITY_GRADIENT23;
    v_g[2][0] = VELOCITY_GRADIENT31;
    v_g[2][1] = VELOCITY_GRADIENT32;
    v_g[2][2] = VELOCITY_GRADIENT33;
  }

  /*
   * Material property constants, etc. Any variations for this
   * Gauss point were evaluated in load_material_properties().
   */

  if (pd->e[pg->imtrx][eqn] & T_POROUS_BRINK) {
    GOMA_EH(GOMA_ERROR, "Porous Brinkmann term not enabled for autodiff momentum");
  }

  /*   eqn = R_MOMENTUM1; */
  /*
   * Field variables...
   */

  /*
   * Calculate the momentum stress tensor at the current gauss point
   */
  ad_fluid_stress(Pi);

  ad_momentum_source_term(f, time);

  /*
   * Residuals_________________________________________________________________
   */
  std::vector<std::vector<ADType>> resid(WIM);
  for (a = 0; a < WIM; a++) {
    resid[a].resize(ei[pg->imtrx]->dof[eqn + a]);
    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
      resid[a][i] = 0;
    }
  }

  if (af->Assemble_Residual) {
    /*
     * Assemble each component "a" of the momentum equation...
     */
    for (a = 0; a < WIM; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = upd->ep[pg->imtrx][eqn];
      bfm = bf[eqn];

      mass_on = pde[eqn] & T_MASS;
      advection_on = pde[eqn] & T_ADVECTION;
      diffusion_on = pde[eqn] & T_DIFFUSION;
      source_on = pde[eqn] & T_SOURCE;

      mass_etm = pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
      advection_etm = pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
      diffusion_etm = pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
      source_etm = pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

      /*
       * In the element, there will be contributions to this many equations
       * based on the number of degrees of freedom...
       */

      R = &(lec->R[LEC_R_INDEX(peqn, 0)]);

      for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
        ledof = ei[pg->imtrx]->lvdof_to_ledof[eqn][i];
        if (ei[pg->imtrx]->active_interp_ledof[ledof]) {
          /*
           *  Here is where we figure out whether the row is to placed in
           *  the normal spot (e.g., ii = i), or whether a boundary condition
           *  require that the volumetric contribution be stuck in another
           *  ldof pertaining to the same variable type.
           */
          ii = ei[pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          phi_i = bf[eqn]->phi[i];
          /* only use Petrov Galerkin on advective term - if required */
          ADType wt_func = phi_i;

          /* add Petrov-Galerkin terms as necessary */
          if (supg != 0.) {
            for (p = 0; p < dim; p++) {
              wt_func += supg * tau * ad_fv.v[p] * bfm->grad_phi[i][p];
            }
          }
          grad_phi_i_e_a = bfm->grad_phi_e[i][a];

          /* this is an optimization for xfem */
          if (xfem != NULL) {
            GOMA_EH(GOMA_ERROR, "xfem not configured for AD Momentum");
          }

          ADType mass = 0.;
          if (transient_run) {
            if (mass_on) {
              mass = ad_fv.v_dot[a] * rho;
              mass *= -wt_func * d_area;
              mass *= mass_etm;
            }
          }

          ADType advection = 0.;
          if (advection_on) {
#ifdef DO_NO_UNROLL
            for (p = 0; p < WIM; p++) {
              advection += (v[p] - x_dot[p]) * grad_v[p][a];
            }
#else
            advection += (ad_fv.v[0] - ad_fv.x_dot[0]) * ad_fv.grad_v[0][a];
            advection += (ad_fv.v[1] - ad_fv.x_dot[1]) * ad_fv.grad_v[1][a];
            if (WIM == 3)
              advection += (ad_fv.v[2] - ad_fv.x_dot[2]) * ad_fv.grad_v[2][a];
#endif

            advection *= rho;
            advection *= -wt_func * d_area;
            advection *= advection_etm;
          }

          ADType diffusion = 0.;
          if (diffusion_on) {
            for (p = 0; p < VIM; p++) {
              for (q = 0; q < VIM; q++) {
                diffusion += grad_phi_i_e_a[p][q] * Pi[q][p];
              }
            }
            diffusion *= -d_area;
            diffusion *= diffusion_etm;
          }

          /*
           * Source term...
           */
          ADType source = 0.0;
          if (source_on) {
            source += f[a];
            source *= wt_func * d_area;
            source *= source_etm;
          }

          /*
           * Add contributions to this residual (globally into Resid, and
           * locally into an accumulator)
           */

          /*lec->R[LEC_R_INDEX(peqn,ii)] += mass + advection + porous + diffusion + source;*/
          R[ii] += mass.val() + advection.val() + diffusion.val() + source.val();
          resid[a][ii] += mass + advection + diffusion + source;
        } /*end if (active_dofs) */
      }   /* end of for (i=0,ei[pg->imtrx]->dofs...) */
    }
  }

  /*
   * Jacobian terms...
   */

  if (af->Assemble_Jacobian) {
    for (a = 0; a < WIM; a++) {
      eqn = R_MOMENTUM1 + a;
      peqn = upd->ep[pg->imtrx][eqn];

      for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
        ii = ei[pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

        ledof = ei[pg->imtrx]->lvdof_to_ledof[eqn][i];
        if (ei[pg->imtrx]->active_interp_ledof[ledof]) {
          ii = ei[pg->imtrx]->lvdof_to_row_lvdof[eqn][i];

          for (int var = V_FIRST; var < V_LAST; var++) {

            /* Sensitivity w.r.t. velocity */
            if (pdv[var]) {
              int pvar = upd->vp[pg->imtrx][var];

              for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                lec->J[LEC_J_INDEX(peqn, pvar, ii, j)] += resid[a][ii].dx(ad_fv.offset[var] + j);

              } /* End of loop over j */
            }   /* End of if the variale is active */
          }

        } /* end of if(active_dofs) */
      }   /* End of loop over i */
    }     /* End of if assemble Jacobian */
  }
  safe_free((void *)n_dof);
  return (status);
}

ADType ad_vec_dot(const int n1, ADType *v1, ADType *v2) {
  int i;
  ADType rc = 0.0;

  for (i = 0; i < n1; i++) {
    rc += *v1 * *v2;
    v1++;
    v2++;
  }
  return (rc);
}
int ad_tensor_dot(ADType t1[DIM][DIM],
                  ADType t2[DIM][DIM],
                  ADType t1_dot_t2[DIM][DIM],
                  const int dim) {
  int i, j, k;
  int status;
  ADType v1[DIM];
  ADType v2[DIM];

  for (k = 0; k < dim; k++) {
    for (i = 0; i < dim; i++) {
      v1[i] = t1[k][i];
    }
    for (j = 0; j < dim; j++) {
      for (i = 0; i < dim; i++) {
        v2[i] = t2[i][j];
      }
      t1_dot_t2[k][j] = ad_vec_dot(dim, v1, v2);
    }
  }

  status = 1;
  return (status);
}
void ad_ve_polymer_stress(ADType gamma[DIM][DIM], ADType stress[DIM][DIM]) {
#if 1

  dbl dgamma[DIM][DIM];
  for (int i = 0; i < VIM; i++) {
    for (int j = 0; j < VIM; j++) {
      stress[i][j] = 0;
      dgamma[i][j] = gamma[i][j].val();
    }
  }
  switch (vn->evssModel) {
  case SQRT_CONF: {
    for (int mode = 0; mode < vn->modes; mode++) {
      /* get polymer viscosity */
      dbl mup = viscosity(ve[mode]->gn, dgamma, NULL);
      dbl lambda = ve[mode]->time_const;

      ADType bdotb[DIM][DIM];
      ADType b[DIM][DIM];
      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {
          if (ii <= jj) {
            b[ii][jj] = ad_fv.S[mode][ii][jj];
            b[jj][ii] = b[ii][jj];
          }
        }
      }

      ad_tensor_dot(b, b, bdotb, VIM);

      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {
          stress[ii][jj] += -(mup / lambda) * (delta(ii, jj) - bdotb[ii][jj]);
        }
      }
    }

  } break;
  default: // Regular stress formulations
  {
    for (int mode = 0; mode < vn->modes; mode++) {
      for (int i = 0; i < VIM; i++) {
        for (int j = 0; j < VIM; j++) {
          stress[i][j] += ad_fv.S[mode][i][j];
        }
      }
    }
  } break;
  }
#endif
}

/*
 * Calculate the total stress tensor for a fluid at a single gauss point
 *  This includes the diagonal pressure contribution
 *
 *  Pi = stress tensor
 *  d_Pi = dependence of the stress tensor on the independent variables
 */
void ad_fluid_stress(ADType Pi[DIM][DIM]) {

  /*
   * Variables for vicosity and derivative
   */
  ADType mu = 0.0;

  /* polymer viscosity and derivatives */

  ADType mup;

  /*  shift function */
  ADType at = 0.0;

  /* solvent viscosity and derivatives */

  ADType mus = 0.0;

  /* numerical "adaptive" viscosity and derivatives */

  ADType mu_num;
  if (pd->gv[TEMPERATURE]) {
    GOMA_EH(GOMA_ERROR, "Temperature not yet implemented ad_fluid_stress");
  }

  /*
   * Field variables...
   */

  dbl evss_f = 0;
  if (pd->gv[POLYMER_STRESS11] && is_evss_f_model(vn->evssModel)) {
    evss_f = 1.0;
  }

  double Heaviside = 1;
  if (ls != NULL && ls->ghost_stress) {
    load_lsi(ls->Length_Scale);
    switch (ls->ghost_stress) {
    case LS_OFF:
      Heaviside = 1;
      break;
    case LS_POSITIVE:
      Heaviside = lsi->H;
      break;
    case LS_NEGATIVE:
      Heaviside = 1 - lsi->H;
      break;
    default:
      GOMA_EH(GOMA_ERROR, "Unknown Level Set Ghost Stress value");
      break;
    }
  }

  ADType gamma_cont[DIM][DIM];
  ADType gamma[DIM][DIM];
  dbl dgamma[DIM][DIM];
  if (evss_f) {
    for (int a = 0; a < VIM; a++) {
      for (int b = 0; b < VIM; b++) {
        gamma_cont[a][b] = ad_fv.G[a][b] + ad_fv.G[b][a];
      }
    }
  } else {
    for (int a = 0; a < VIM; a++) {
      for (int b = 0; b < VIM; b++) {
        gamma_cont[a][b] = 0;
      }
    }
  }

  /* load up shear rate tensor based on velocity */
  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      dgamma[a][b] = fv->grad_v[a][b] + fv->grad_v[b][a];
      gamma[a][b] = ad_fv.grad_v[a][b] + ad_fv.grad_v[b][a];
    }
  }

  // TODO AD
  mu = viscosity(gn, dgamma, NULL);
  if (pd->gv[POLYMER_STRESS11]) {
    mus = viscosity(gn, dgamma, NULL);

    /* This is the adaptive viscosity from Sun et al., 1999.
     * The term multiplies the continuous and discontinuous
     * shear-rate, so it should cancel out and not affect the
     * solution, other than increasing the stability of the
     * algorithm in areas of high shear and stress.
     */

    mu_num = 1;
    // if (DOUBLE_NONZERO(vn->eps)) {
    //   for (int mode = 0; mode < vn->modes; mode++) {
    //     for (a = 0; a < VIM; a++) {
    //       for (b = 0; b < VIM; b++) {
    //         s[a][b] += fv->S[mode][a][b];
    //       }
    //     }
    //   }

    //   mu_num = numerical_viscosity(s, gamma_cont, d_mun_dS, d_mun_dG);

    mu = mu_num * mus;

    for (int mode = 0; mode < vn->modes; mode++) {
      /* get polymer viscosity */
      // TODO AD
      mup = viscosity(ve[mode]->gn, dgamma, NULL);

      mu += Heaviside * mu_num * at * mup;

    } // for mode
  }   // if POLYMER_STRESS

  /*
   * Calculate the dilational viscosity, if necessary
   */
  if (mp->DilationalViscosityModel != DILVISCM_KAPPAWIPESMU) {
    GOMA_EH(GOMA_ERROR, "Dilational viscosity not enabled AD fluid stress");
  }

  /*
   * Viscoelastic Stress contributions
   */
  ADType polymer_stress[DIM][DIM];
  if (pd->gv[POLYMER_STRESS11]) {
    ad_ve_polymer_stress(gamma, polymer_stress);
  }

  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      Pi[a][b] = -ad_fv.P * (double)delta(a, b) + mu * gamma[a][b];
    }
  }

  if (pd->gv[POLYMER_STRESS11]) {
    for (int a = 0; a < VIM; a++) {
      for (int b = 0; b < VIM; b++) {
        // TODO : derivative may be missing here...
        Pi[a][b] += -evss_f * (mu - mus) * gamma_cont[a][b] + Heaviside * polymer_stress[a][b];
      }
    }
  }
  if (gn->ConstitutiveEquation == BINGHAM_MIXED) {
    GOMA_EH(-1, "BINGHAM_MIXED not enabled for AD fluid stress");
  }
}
/******************************************************************************
 * momentum_source_term(): Computes the body force term for the momentum balance.
 *
 * Input
 * -----
 *   f    == Body force
 *   df->T == Derivative w.r.t. temperature
 *   df->X == Derivative w.r.t. mesh displacements
 *   df->C == Derivative w.r.t. concentration
 *   df->v == Derivative w.r.t. velocity
 *   df->F == Derivative w.r.t. FILL
 *   df->E == Derivative w.r.t. electric field
 *
 ******************************************************************************/
int ad_momentum_source_term(ADType f[DIM], /* Body force. */
                            dbl time) {
  int a;
  int eqn;
  const int dim = pd->Num_Dim;
  int status = 0;

  /* initialize everything to zero */
  for (int a = 0; a < VIM; a++) {
    f[a] = 0;
  }

  /****Momentum Source Model******/
  if (mp->MomentumSourceModel == CONSTANT) {
    int force_dim = dim;
    if (pd->CoordinateSystem == CARTESIAN_2pt5D) {
      force_dim = 3;
    }
    for (int a = 0; a < force_dim; a++) {
      eqn = R_MOMENTUM1 + a;
      if (pd->e[upd->matrix_index[eqn]][eqn] & T_SOURCE) {
        f[a] = mp->momentum_source[a];
      }
    }
  } else if (mp->MomentumSourceModel == VARIABLE_DENSITY) {
    if (mp->DensityModel == SOLVENT_POLYMER) {
      double rho = density(NULL, time);
      for (a = 0; a < dim; a++) {
        eqn = R_MOMENTUM1 + a;
        if (pd->e[upd->matrix_index[eqn]][eqn] & T_SOURCE) {
          f[a] = rho * mp->momentum_source[a];
        }
      }
    } else {
      GOMA_EH(GOMA_ERROR, "Unknown density model for variable density");
    }
  } else {
    GOMA_EH(GOMA_ERROR, "Unknown Navier-Stokes Source Model for AD");
  }

  if (ls != NULL && mp->mp2nd != NULL && mp->MomentumSourceModel != LEVEL_SET &&
      mp->mp2nd->MomentumSourceModel == CONSTANT && (pd->e[pg->imtrx][R_MOMENTUM1] & T_SOURCE)) {
    GOMA_EH(GOMA_ERROR, "Unknown Navier-Stokes Source Model for AD");
  }

  return (status);
}

int ad_assemble_continuity(dbl time_value, /* current time */
                           dbl tt,         /* parameter to vary time integration from
                                              explicit (tt = 1) to implicit (tt = 0)    */
                           dbl dt,         /* current time step size                    */
                           const PG_DATA *pg_data) {
  int dim;
  int p, q, a, b;

  int eqn, var;
  int peqn, pvar;
  int w;

  int i, j;
  int status, err;

  dbl time = 0.0; /*  RSL 6/6/02  */

  dbl epsilon = 0.0, derivative, sum; /*  RSL 7/24/00  */
  dbl sum1, sum2;                     /*  RSL 8/15/00  */
  dbl sum_a, sum_b;                   /*  RSL 9/28/01  */
  int jj;                             /*  RSL 7/25/00  */

  dbl advection;
  dbl source;
  dbl pressure_stabilization;

  dbl volsolvent = 0;         /* volume fraction of solvent                */
  dbl initial_volsolvent = 0; /* initial solvent volume fraction
                               * (in stress-free state) input as source
                               * constant from input file                  */

  dbl det_J;
  dbl h3;
  dbl wt;
  dbl d_area;

  dbl d_h3detJ_dmesh_bj; /* for specific (b,j) mesh dof */

  /*
   * Galerkin weighting functions...
   */

  dbl phi_i;
  dbl phi_j;
  dbl div_phi_j_e_b;
  dbl(*grad_phi)[DIM]; /* weight-function for PSPG term */

  dbl div_v_dmesh; /* for specific (b,j) mesh dof */

  /*
   * Variables for Pressure Stabilization Petrov-Galerkin...
   */
  int meqn;
  int v_s[MAX_MODES][DIM][DIM], v_g[DIM][DIM];
  int mode;

  int *pdv = pd->v[pg->imtrx];

  dbl pspg[DIM];
  PSPG_DEPENDENCE_STRUCT d_pspg_struct;
  PSPG_DEPENDENCE_STRUCT *d_pspg = &d_pspg_struct;

  dbl mass, mass_a;
  dbl source_a;
  dbl sourceBase = 0.0;

  dbl rho = 0;
  DENSITY_DEPENDENCE_STRUCT d_rho_struct; /* density dependence */
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;

  struct Species_Conservation_Terms s_terms;
  dbl rhos = 0, rhof = 0;
  dbl h_flux = 0;
  int w0 = 0;

  /* For particle momentum model.
   */
  int species;              /* species number for particle phase,  */
  dbl ompvf;                /* 1 - partical volume fraction */
  int particle_momentum_on; /* boolean. */

  /* Foaming model TAB */
  double dFVS_dv[DIM][MDE];
  double dFVS_dT[MDE];
  double dFVS_dx[DIM][MDE];
  double dFVS_dC[MAX_CONC][MDE];
  double dFVS_dF[MDE];
  double dFVS_drho[MDE];
  double dFVS_dMOM[MAX_MOMENTS][MDE];

  int transient_run = pd->TimeIntegration != STEADY;
  int advection_on = 0;
  int source_on = 0;
  int ion_reactions_on = 0, electrode_kinetics_on = 0;
  int lagrangian_mesh_motion = 0, total_ale_on = 0;
  int hydromassflux_on = 0, suspensionsource_on = 0;
  int foam_volume_source_on = 0;
  int total_ale_and_velo_off = 0;

  dbl advection_etm, source_etm;

  double *J;

  status = 0;

  /*
   * Unpack variables from structures for local convenience...
   */

  eqn = R_PRESSURE;
  peqn = upd->ep[pg->imtrx][eqn];

  ADType div_v = 0;

  for (a = 0; a < VIM; a++) {
    div_v += ad_fv.grad_v[a][a];
  }

  /*
   * Bail out fast if there's nothing to do...
   */

  if (!pd->e[pg->imtrx][eqn]) {
    return (status);
  }

  dim = pd->Num_Dim;

  if (pd->gv[POLYMER_STRESS11]) {
    err = stress_eqn_pointer(v_s);

    v_g[0][0] = VELOCITY_GRADIENT11;
    v_g[0][1] = VELOCITY_GRADIENT12;
    v_g[1][0] = VELOCITY_GRADIENT21;
    v_g[1][1] = VELOCITY_GRADIENT22;
    v_g[0][2] = VELOCITY_GRADIENT13;
    v_g[1][2] = VELOCITY_GRADIENT23;
    v_g[2][0] = VELOCITY_GRADIENT31;
    v_g[2][1] = VELOCITY_GRADIENT32;
    v_g[2][2] = VELOCITY_GRADIENT33;
  }

  wt = fv->wt;
  det_J = bf[eqn]->detJ; /* Really, ought to be mesh eqn. */
  h3 = fv->h3;           /* Differential volume element (scales). */

  d_area = wt * det_J * h3;

  grad_phi = bf[eqn]->grad_phi;

  /*
   * Get the deformation gradients and tensors if needed
   */

  lagrangian_mesh_motion = (cr->MeshMotion == LAGRANGIAN || cr->MeshMotion == DYNAMIC_LAGRANGIAN);
  electrode_kinetics_on = (mp->SpeciesSourceModel[0] == ELECTRODE_KINETICS);
  ion_reactions_on = (mp->SpeciesSourceModel[0] == ION_REACTIONS);
  total_ale_on = (cr->MeshMotion == TOTAL_ALE);
  hydromassflux_on = (cr->MassFluxModel == HYDRODYNAMIC);
  suspensionsource_on = (mp->MomentumSourceModel == SUSPENSION);

  if (lagrangian_mesh_motion && pd->gv[R_MESH1]) {
    err = belly_flop(elc->lame_mu);
    GOMA_EH(err, "error in belly flop");
    if (err == 2)
      return (err);
  }

  if (total_ale_on && !pd->gv[VELOCITY1]) {
    total_ale_and_velo_off = 1;
  }

  if (total_ale_and_velo_off && pd->gv[R_SOLID1]) {
    err = belly_flop_rs(elc_rs->lame_mu);
    GOMA_EH(err, "error in belly flop for real solid");
    if (err == 2)
      return (err);
  }

  particle_momentum_on = 0;
  species = -1;
  ompvf = 1.0;

  if (pd->gv[R_PMOMENTUM1]) {
    particle_momentum_on = 1;
    species = (int)mp->u_density[0];
    ompvf = 1.0 - fv->c[species];
  }

  if (PSPG) {
    calc_pspg(pspg, d_pspg, time_value, tt, dt, pg_data);
  }

  if ((lagrangian_mesh_motion || total_ale_and_velo_off) && (mp->PorousMediaType == CONTINUOUS)) {
    initial_volsolvent = elc->Strss_fr_sol_vol_frac;
    volsolvent = 0.;
    for (w = 0; w < pd->Num_Species_Eqn; w++)
      volsolvent += fv->c[w];
    if (particle_momentum_on)
      volsolvent -= fv->c[species];
  }

  if (electrode_kinetics_on || ion_reactions_on) {
    if (mp->PorosityModel == CONSTANT) {
      epsilon = mp->porosity;
    } else if (mp->PorosityModel == THERMAL_BATTERY) {
      epsilon = mp->u_porosity[0];
    } else {
      GOMA_EH(GOMA_ERROR, "invalid porosity model");
    }
  }

  if (mp->MomentumSourceModel == SUSPENSION_PM || electrode_kinetics_on) /*  RSL 7/25/00  */
  {
    err = get_continuous_species_terms(&s_terms, 0.0, tt, dt, pg_data->hsquared);
    GOMA_EH(err, "problem in getting the species terms");
  }

  if (ion_reactions_on) /*  RSL 3/19/01 and 6/6/02  */
  {
    zero_structure(&s_terms, sizeof(struct Species_Conservation_Terms), 1);
    err = get_continuous_species_terms(&s_terms, time, tt, dt, pg_data->hsquared);
    GOMA_EH(err, "problem in getting the species terms");
  }

  if ((hydromassflux_on) && (mp->DensityModel == SUSPENSION) && (suspensionsource_on)) {
    /*
     * Compute hydrodynamic/sedimentation flux and sensitivities.
     */

    w0 =
        (int)mp->u_density[0]; /* This is the species number that is transported HYDRODYNAMICally */

    hydro_flux(&s_terms, w0, tt, dt, pg_data->hsquared);

    rhof = mp->u_density[1];
    rhos = mp->u_density[2];
  }
  rho = density(d_rho, time_value);

  advection_on = pd->e[pg->imtrx][eqn] & T_ADVECTION;
  source_on = pd->e[pg->imtrx][eqn] & T_SOURCE;

  advection_etm = pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
  source_etm = pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];

  dbl ls_disable_pspg = 1;

  std::vector<ADType> resid(ei[pg->imtrx]->dof[eqn]);
  for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
    resid[i] = 0;
  }
  if (af->Assemble_Residual) {
    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

      phi_i = bf[eqn]->phi[i];

      /*
       *  Mass Terms: drhodt terms (usually though problem dependent)
       */
      mass = 0.0;

      /*
       *  Advection:
       *    This term refers to the standard del dot v .
       *
       *    int (phi_i div_v d_omega)
       *
       *   Note density is not multiplied into this term normally
       */
      ADType advection = 0.0;
      if (advection_on) {
        if (pd->gv[VELOCITY1]) /* then must be solving fluid mechanics in this material */
        {

          /*
           * Standard incompressibility constraint means we have
           * a solenoidal velocity field
           */

          advection = div_v;

          advection *= phi_i * d_area;
          advection *= advection_etm;
        }
      }

      source = 0.0;
      pressure_stabilization = 0.0;
      if (PSPG) {
        GOMA_EH(GOMA_ERROR, "Error");
        for (a = 0; a < WIM; a++) {
          meqn = R_MOMENTUM1 + a;
          if (pd->gv[meqn]) {
            pressure_stabilization += grad_phi[i][a] * pspg[a];
          }
        }
        pressure_stabilization *= d_area * ls_disable_pspg;
      }

      /*
       *  Add up the individual contributions and sum them into the local element
       *  contribution for the total continuity equation for the ith local unknown
       */
      lec->R[LEC_R_INDEX(peqn, i)] += advection.val() + pressure_stabilization;
      resid[i] = advection + pressure_stabilization;
    }
  }
  if (af->Assemble_Jacobian) {
    eqn = PRESSURE;
    peqn = upd->ep[pg->imtrx][eqn];

    for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

      /* Sensitivity w.r.t. velocity */
      for (b = 0; b < VIM; b++) {
        var = VELOCITY1 + b;
        if (pdv[var]) {
          pvar = upd->vp[pg->imtrx][var];

          for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
            lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += resid[i].dx(ad_fv.offset[var] + j);

          } /* End of loop over j */
        }   /* End of if the variale is active */
      }     /* End of loop over velocity components */

    } /* End of loop over i */
  }   /* End of if assemble Jacobian */
  return 0;
}

void ad_compute_a_dot_b(ADType b[DIM][DIM], ADType G[DIM][DIM], ADType a_dot_b[DIM][DIM]) {

  if (VIM == 2) {

    ADType a12 =
        ((b[0][1] * G[0][0] - b[0][0] * G[0][1]) + (b[1][1] * G[1][0] - b[1][0] * G[1][1])) /
        (b[0][0] + b[1][1] + 1e-16);

    ADType a[DIM][DIM] = {{0., a12, 0.}, {-a12, 0., 0.}, {0., 0., 0.}};

    ad_tensor_dot(a, b, a_dot_b, VIM);

  } else { // VIM = 3
    ADType D =
        -b[0][1] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) -
        b[0][2] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
        (b[1][1] + b[2][2]) * (-b[1][2] * b[1][2] + (b[0][0] + b[1][1]) * (b[0][0] + b[2][2])) +
        1e-16;
    ADType invD = 1.0 / D;

    ADType a12 = invD * (-pow(b[0][1], 2) + (b[0][0] + b[2][2]) * (b[1][1] + b[2][2])) *
                     (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] -
                      G[1][1] * b[0][1] + G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
                 (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) *
                     (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] -
                      G[1][2] * b[0][1] + G[2][0] * b[2][2] - G[2][2] * b[0][2]) +
                 (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) *
                     (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] -
                      G[1][2] * b[1][1] + G[2][1] * b[2][2] - G[2][2] * b[1][2]);

    ADType a13 =
        invD *
        ((-pow(b[0][2], 2) + (b[0][0] + b[1][1]) * (b[1][1] + b[2][2])) *
             (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
              G[2][0] * b[2][2] - G[2][2] * b[0][2]) +
         (-b[0][1] * b[0][2] - b[1][2] * (b[1][1] + b[2][2])) *
             (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
              G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
         (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) *
             (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
              G[2][1] * b[2][2] - G[2][2] * b[1][2])) /
        (-b[0][1] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) -
         b[0][2] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
         (b[1][1] + b[2][2]) * (-pow(b[1][2], 2) + (b[0][0] + b[1][1]) * (b[0][0] + b[2][2])));

    ADType a23 = invD * (-pow(b[1][2], 2) + (b[0][0] + b[1][1]) * (b[0][0] + b[2][2])) *
                     (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] -
                      G[1][2] * b[1][1] + G[2][1] * b[2][2] - G[2][2] * b[1][2]) +
                 (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) *
                     (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] -
                      G[1][1] * b[0][1] + G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
                 (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) *
                     (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] -
                      G[1][2] * b[0][1] + G[2][0] * b[2][2] - G[2][2] * b[0][2]);

    ADType a[DIM][DIM] = {
        {0.0, a12, a13},
        {-a12, 0.0, a23},
        {-a13, -a23, 0.0},
    };

    ad_tensor_dot(a, b, a_dot_b, VIM);
  }
}

int ad_sqrt_conf_source(int mode, ADType b[DIM][DIM], ADType source_term[DIM][DIM]) {
  ADType binv[DIM][DIM];
  ADType d_binv_db[DIM][DIM][DIM][DIM];
  if (VIM == 2) {
    ADType det = b[0][0] * b[1][1] - b[0][1] * b[0][1] + 1e-16;
    binv[0][0] = b[1][1] / det;
    binv[0][1] = -b[0][1] / det;
    binv[1][0] = -b[0][1] / det;
    binv[1][1] = b[0][0] / det;

  } else if (VIM == 3) {
    ADType det = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1]) -
                 b[0][1] * (b[1][0] * b[2][2] - b[2][0] * b[1][2]) +
                 b[0][2] * (b[1][0] * b[2][1] - b[2][0] * b[1][1]) + 1e-16;

    binv[0][0] = (b[1][1] * b[2][2] - b[2][1] * b[1][2]) / (det);

    binv[0][1] = -(b[0][1] * b[2][2] - b[2][1] * b[0][2]) / (det);

    binv[0][2] = (b[0][1] * b[1][2] - b[1][1] * b[0][2]) / (det);

    binv[1][0] = -(b[1][0] * b[2][2] - b[2][0] * b[1][2]) / (det);

    binv[1][1] = (b[0][0] * b[2][2] - b[2][0] * b[0][2]) / (det);

    binv[1][2] = -(b[0][0] * b[1][2] - b[1][0] * b[0][2]) / (det);

    binv[2][0] = (b[1][0] * b[2][1] - b[1][1] * b[2][0]) / (det);

    binv[2][1] = -(b[0][0] * b[2][1] - b[2][0] * b[0][1]) / (det);

    binv[2][2] = (b[0][0] * b[1][1] - b[1][0] * b[0][1]) / (det);

  } else {
    GOMA_EH(GOMA_ERROR, "Unknown VIM = %d for SQRT conformation tensor", VIM);
  }

  switch (vn->ConstitutiveEquation) {
  case OLDROYDB: {
    for (int ii = 0; ii < VIM; ii++) {
      for (int jj = 0; jj < VIM; jj++) {
        source_term[ii][jj] = -0.5 * (binv[ii][jj] - b[ii][jj]);
      }
    }
  } break;
  case PTT: {

    ADType trace = 0;
    for (int i = 0; i < VIM; i++) {
      for (int j = 0; j < VIM; j++) {
        trace += b[i][j] * b[i][j];
      }
    }

    ADType Z = 1.0;
    ADType dZ_dtrace = 0;

    // PTT exponent
    eps = ve[mode]->eps;

    if (vn->ptt_type == PTT_LINEAR) {
      Z = 1 + eps * (trace - (double)VIM);
    } else if (vn->ptt_type == PTT_EXPONENTIAL) {
      const double exp_max = 700;
      ADType inner = eps * (trace - (double)VIM);
      if ((inner > exp_max) || (inner < -exp_max)) {
        GOMA_WH_MANY(GOMA_ERROR, "Exponential overflow in PTT_EXPONENTIAL");
        return GOMA_ERROR;
      }
      Z = exp(eps * (trace - (double)VIM));
    } else {
      GOMA_EH(GOMA_ERROR, "Unrecognized PTT Form %d", vn->ptt_type);
    }

    for (int ii = 0; ii < VIM; ii++) {
      for (int jj = 0; jj < VIM; jj++) {
        source_term[ii][jj] = -0.5 * Z * (binv[ii][jj] - b[ii][jj]);
      }
    }
  } break;
  default:
    GOMA_EH(GOMA_ERROR, "Unknown Constitutive equation form for SQRT_CONF");
    break;
  }

  return GOMA_SUCCESS;
}
void ad_load_modal_pointers(
    int ve_mode, /* mode number */
    dbl tt,
    dbl dt,
    ADType s[DIM][DIM],           /* stress tensor for mode ve_mode */
    ADType s_dot[DIM][DIM],       /* stress tensor time derivative for mode ve_mode */
    ADType grad_s[DIM][DIM][DIM]) /* grad of stress tensor for mode ve_mode */

{
  int a, b, p; /* indeces for dimensions */
  /* load up things we need in the assembly routine for each mode in turn*/

  /* put stress in a nice working array */

  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      s[a][b] = ad_fv.S[ve_mode][a][b];
      if (pd->TimeIntegration != STEADY) {
        s_dot[a][b] = ad_fv.S_dot[ve_mode][a][b];
      } else {
        s_dot[a][b] = 0.;
      }
    }
  }

  for (p = 0; p < VIM; p++) {
    for (a = 0; a < VIM; a++) {
      for (b = 0; b < VIM; b++) {
        grad_s[p][a][b] = ad_fv.grad_S[ve_mode][p][a][b];
      }
    }
  }
}

int ad_assemble_stress_sqrt_conf(dbl tt, /* parameter to vary time integration from
                                          * explicit (tt = 1) to implicit (tt = 0) */
                                 dbl dt, /* current time step size */
                                 PG_DATA *pg_data) {
  int dim, p, q, r, w;

  int eqn, var;
  int peqn, pvar;
  int evss_gradv = 0;

  int i, j, status, mode;
  ADType v[DIM];     /* Velocity field. */
  ADType x_dot[DIM]; /* current position field derivative wrt time. */
  dbl h3;            /* Volume element (scale factors). */
  dbl dh3dmesh_pj;   /* Sensitivity to (p,j) mesh dof. */

  ADType grad_v[DIM][DIM];
  ADType gamma[DIM][DIM]; /* Shear-rate tensor based on velocity */
  dbl dgamma[DIM][DIM];   /* Shear-rate tensor based on velocity */
  dbl det_J;              /* determinant of element Jacobian */

  dbl d_det_J_dmesh_pj; /* for specific (p,j) mesh dof */

  dbl mass; /* For terms and their derivatives */
  dbl mass_a, mass_b;
  dbl advection;
  dbl advection_a, advection_b, advection_c, advection_d;
  dbl diffusion;
  dbl source;
  dbl source_a = 0, source_b = 0, source_c = 0;
  int err;
  dbl alpha = 0;  /* This is the Geisekus mobility parameter */
  dbl lambda = 0; /* polymer relaxation constant */
  dbl d_lambda_dF[MDE];
  double xi;
  double d_xi_dF[MDE];
  dbl eps = 0; /* This is the PTT elongation parameter */
  double d_eps_dF[MDE];
  /*
   *
   * Note how carefully we avoid refering to d(phi[i])/dx[j] and refer instead
   * to the j-th component of grad_phi[j][i] so that this vector can be loaded
   * up with components that may be different in non Cartesian coordinate
   * systems.
   *
   * We will, however, insist on *orthogonal* coordinate systems, even if we
   * might permit them to be curvilinear.
   *
   * Assume all components of velocity are interpolated with the same kind
   * of basis function.
   */

  /*
   * Petrov-Galerkin weighting functions for i-th and ab-th stress residuals
   * and some of their derivatives...
   */

  dbl wt_func;

  /*
   * Interpolation functions for variables and some of their derivatives.
   */

  dbl phi_j;
  dbl wt;

  /* Variables for stress */

  int R_s[MAX_MODES][DIM][DIM];
  int v_s[MAX_MODES][DIM][DIM];
  int v_g[DIM][DIM];

  ADType b[DIM][DIM];     /* stress tensor */
  ADType b_dot[DIM][DIM]; /* stress tensor from last time step */
  ADType grad_b[DIM][DIM][DIM];

  ADType g[DIM][DIM];  /* velocity gradient tensor */
  ADType gt[DIM][DIM]; /* transpose of velocity gradient tensor */

  /* dot product tensors */

  ADType b_dot_g[DIM][DIM];

  /* polymer viscosity and derivatives */
  dbl mup;

  const bool saramitoEnabled =
      (vn->ConstitutiveEquation == SARAMITO_OLDROYDB || vn->ConstitutiveEquation == SARAMITO_PTT ||
       vn->ConstitutiveEquation == SARAMITO_GIESEKUS);

  if (saramitoEnabled) {
    GOMA_EH(GOMA_ERROR, "Saramito not available for SQRT_CONF");
  }

  /*  shift function */
  dbl at = 0.0;
  dbl d_at_dT[MDE];
  dbl wlf_denom;

  /* advective terms are precalculated */
  ADType v_dot_del_b[DIM][DIM];
  ADType x_dot_del_b[DIM][DIM];

  dbl d_xdotdels_dm;

  dbl d_vdotdels_dm;

  /* SUPG variables */
  dbl supg = 0;

  if (vn->evssModel == EVSS_GRADV) {
    evss_gradv = 1;
  }

  status = 0;

  eqn = R_STRESS11;

  /*
   * Bail out fast if there's nothing to do...
   */

  if (!pd->e[pg->imtrx][eqn]) {
    return (status);
  }

  /*
   * Unpack variables from structures for local convenience...
   */

  dim = pd->Num_Dim;

  wt = fv->wt;

  det_J = bf[eqn]->detJ; /* Really, ought to be mesh eqn. */

  h3 = fv->h3; /* Differential volume element (scales). */

  /* load eqn and variable number in tensor form */
  (void)stress_eqn_pointer(v_s);
  (void)stress_eqn_pointer(R_s);

  v_g[0][0] = VELOCITY_GRADIENT11;
  v_g[0][1] = VELOCITY_GRADIENT12;
  v_g[1][0] = VELOCITY_GRADIENT21;
  v_g[1][1] = VELOCITY_GRADIENT22;
  v_g[0][2] = VELOCITY_GRADIENT13;
  v_g[1][2] = VELOCITY_GRADIENT23;
  v_g[2][0] = VELOCITY_GRADIENT31;
  v_g[2][1] = VELOCITY_GRADIENT32;
  v_g[2][2] = VELOCITY_GRADIENT33;

  /*
   * Field variables...
   */
  for (int a = 0; a < WIM; a++) {
    v[a] = ad_fv.v[a];

    /* note, these are zero for steady calculations */
    x_dot[a] = 0.0;
    if (pd->TimeIntegration != STEADY && pd->gv[MESH_DISPLACEMENT1 + a]) {
      x_dot[a] = ad_fv.x_dot[a];
    }
  }

  /*
   * In Cartesian coordinates, this velocity gradient tensor will
   * have components that are...
   *
   * 			grad_v[a][b] = d v_b
   *				       -----
   *				       d x_a
   */

  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      grad_v[a][b] = ad_fv.grad_v[a][b];
    }
  }

  /* load up shearrate tensor based on velocity */
  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      gamma[a][b] = grad_v[a][b] + grad_v[b][a];
      dgamma[a][b] = fv->grad_v[a][b] + fv->grad_v[b][a];
    }
  }

  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      if (evss_gradv) {
        g[a][b] = ad_fv.grad_v[a][b];
        gt[a][b] = ad_fv.grad_v[b][a];
      } else {
        g[a][b] = ad_fv.G[a][b];
        gt[b][a] = g[a][b];
      }
    }
  }

  if (vn->wt_funcModel == GALERKIN) {
    supg = 0.;
  } else if (vn->wt_funcModel == SUPG) {
    supg = vn->wt_func;
  }

  ADType supg_tau;
  if (supg != 0.) {
    ad_supg_tau_shakib(supg_tau, dim, dt, 1e-8, eqn);
  }
  /* end Petrov-Galerkin addition */
  dbl yzbeta_factor = 0.0;
  dbl beta[2] = {1.0, 2.0};
  if (vn->shockcaptureModel == SC_YZBETA) {
    yzbeta_factor = vn->shockcapture;
  } else if (vn->shockcaptureModel == SC_DCDD) {
  } else if (vn->shockcaptureModel != SC_NONE) {
    GOMA_EH(GOMA_ERROR, "Unknown shock capture model, only YZBETA supported for SQRT_CONF");
  }

  /*  shift factor  */
  if (pd->gv[TEMPERATURE]) {
    if (vn->shiftModel == CONSTANT) {
      at = vn->shift[0];
      for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
        d_at_dT[j] = 0.;
      }
    } else if (vn->shiftModel == MODIFIED_WLF) {
      wlf_denom = vn->shift[1] + fv->T - mp->reference[TEMPERATURE];
      if (wlf_denom != 0.) {
        at = exp(vn->shift[0] * (mp->reference[TEMPERATURE] - fv->T) / wlf_denom);
        for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
          d_at_dT[j] =
              -at * vn->shift[0] * vn->shift[1] / (wlf_denom * wlf_denom) * bf[TEMPERATURE]->phi[j];
        }
      } else {
        at = 1.;
      }
      for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
        d_at_dT[j] = 0.;
      }
    }
  } else {
    at = 1.;
  }

  /* Begin loop over modes */
  for (mode = 0; mode < vn->modes; mode++) {

    ad_load_modal_pointers(mode, tt, dt, b, b_dot, grad_b);

    /* precalculate advective terms of form (v dot del tensor)*/

    /*
     * Stress tensor...(Note "anti-BSL" sign convention on deviatoric stress)
     */
    for (int ii = 0; ii < VIM; ii++) {
      for (int jj = 0; jj < VIM; jj++) {
        v_dot_del_b[ii][jj] = 0.;
        x_dot_del_b[ii][jj] = 0.;
        for (q = 0; q < WIM; q++) {
          v_dot_del_b[ii][jj] += v[q] * grad_b[q][ii][jj];
          x_dot_del_b[ii][jj] += x_dot[q] * grad_b[q][ii][jj];
        }
      }
    }

    /* get polymer viscosity */
    mup = viscosity(ve[mode]->gn, dgamma, NULL);

    if (saramitoEnabled == TRUE) {
      GOMA_EH(GOMA_ERROR, "Saramito not enabled sqrt");
    }

    double d_alpha_dF[MDE];
    /* get Geisekus mobility parameter */
    if (ve[mode]->alphaModel == CONSTANT) {
      alpha = ve[mode]->alpha;
    } else if (ls != NULL && ve[mode]->alphaModel == VE_LEVEL_SET) {
      double pos_alpha = ve[mode]->pos_ls.alpha;
      double neg_alpha = ve[mode]->alpha;
      double width = ls->Length_Scale;
      err = level_set_property(neg_alpha, pos_alpha, width, &alpha, d_alpha_dF);
      GOMA_EH(err, "level_set_property() failed for mobility parameter.");
    } else {
      GOMA_EH(GOMA_ERROR, "Unknown mobility parameter model");
    }

    /* get time constant */
    if (ve[mode]->time_constModel == CONSTANT) {
      lambda = ve[mode]->time_const;
    } else if (ve[mode]->time_constModel == CARREAU || ve[mode]->time_constModel == POWER_LAW) {
      lambda = mup / ve[mode]->time_const;
    } else if (ls != NULL && ve[mode]->time_constModel == VE_LEVEL_SET) {
      double pos_lambda = ve[mode]->pos_ls.time_const;
      double neg_lambda = ve[mode]->time_const;
      double width = ls->Length_Scale;
      err = level_set_property(neg_lambda, pos_lambda, width, &lambda, d_lambda_dF);
      GOMA_EH(err, "level_set_property() failed for polymer time constant.");
    }

    xi = 0;
    if (ve[mode]->xiModel == CONSTANT) {
      xi = ve[mode]->xi;
    } else if (ls != NULL && ve[mode]->xiModel == VE_LEVEL_SET) {
      double pos_xi = ve[mode]->pos_ls.xi;
      double neg_xi = ve[mode]->xi;
      double width = ls->Length_Scale;
      err = level_set_property(neg_xi, pos_xi, width, &xi, d_xi_dF);
      GOMA_EH(err, "level_set_property() failed for ptt xi parameter.");
    } else {
      GOMA_EH(GOMA_ERROR, "Unknown PTT Xi parameter model");
    }

    if (DOUBLE_NONZERO(xi)) {
      GOMA_EH(GOMA_ERROR, "PTT Xi parameter currently required to be 0 for SQRT_CONF");
    }

    if (ve[mode]->epsModel == CONSTANT) {
      eps = ve[mode]->eps;
    } else if (ls != NULL && ve[mode]->epsModel == VE_LEVEL_SET) {
      double pos_eps = ve[mode]->pos_ls.eps;
      double neg_eps = ve[mode]->eps;
      double width = ls->Length_Scale;
      err = level_set_property(neg_eps, pos_eps, width, &eps, d_eps_dF);
      GOMA_EH(err, "level_set_property() failed for ptt epsilon parameter.");
    } else {
      GOMA_EH(GOMA_ERROR, "Unknown PTT Epsilon parameter model");
    }

    (void)ad_tensor_dot(b, g, b_dot_g, VIM);

    ADType a_dot_b[DIM][DIM];

    ad_compute_a_dot_b(b, g, a_dot_b);

    ADType source_term[DIM][DIM];
    goma_error err = ad_sqrt_conf_source(mode, b, source_term);
    if (err) {
      return 1;
    }
    std::vector<std::vector<std::vector<ADType>>> resid(VIM);
    for (int a = 0; a < VIM; a++) {
      resid[a].resize(VIM);
      for (int b = 0; b < VIM; b++) {
        resid[a][b].resize(ei[pg->imtrx]->dof[v_s[mode][a][b]]);
        for (i = 0; i < ei[pg->imtrx]->dof[v_s[mode][a][b]]; i++) {
          resid[a][b][i] = 0;
        }
      }
    }
    /*
     * Residuals_________________________________________________________________
     */

    if (af->Assemble_Residual) {
      /*
       * Assemble each component "ab" of the polymer stress equation...
       */
      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {

          if (ii <= jj) /* since the stress tensor is symmetric, only assemble the upper half */
          {
            eqn = R_s[mode][ii][jj];

            /*
             * In the element, there will be contributions to this many equations
             * based on the number of degrees of freedom...
             */

            for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
              ADType wt_func = bf[eqn]->phi[i];
              /* add Petrov-Galerkin terms as necessary */
              if (supg != 0.) {
                for (w = 0; w < dim; w++) {
                  wt_func += supg * supg_tau * v[w] * bf[eqn]->grad_phi[i][w];
                }
              }

              ADType mass = 0.;

              if (pd->TimeIntegration != STEADY) {
                if (pd->e[pg->imtrx][eqn] & T_MASS) {
                  mass = b_dot[ii][jj];
                  mass *= wt_func * at * lambda * det_J * wt;
                  mass *= h3;
                  mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                }
              }

              ADType advection = 0.;
              if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                if (DOUBLE_NONZERO(lambda)) {

                  advection += v_dot_del_b[ii][jj] - x_dot_del_b[ii][jj];
                  advection -= b_dot_g[ii][jj];
                  advection -= a_dot_b[ii][jj];
                  advection *= wt_func * at * lambda * det_J * wt * h3;
                  advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                }
              }

              ADType diffusion = 0.;
              if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
                if (vn->shockcaptureModel == SC_YZBETA) {
                  ADType Z = b_dot[ii][jj];
                  Z += 1e-32 + v_dot_del_b[ii][jj] - x_dot_del_b[ii][jj];
                  Z -= b_dot_g[ii][jj];
                  Z -= a_dot_b[ii][jj];
                  Z *= at * lambda;
                  Z += source_term[ii][jj];

                  ADType Y_inv = 1.0;
                  ADType hdc = 0;
                  ADType js = 0;
                  for (int k = 0; k < ei[pg->imtrx]->dof[eqn]; k++) {
                    for (int p = 0; p < VIM; p++) {
                      js += fabs(grad_b[p][ii][jj] * bf[eqn]->grad_phi[k][p]);
                    }
                  }
                  hdc = 1 / (js + 1e-16);

                  ADType inner = 1e-32;
                  for (int p = 0; p < VIM; p++) {
                    inner += Y_inv * grad_b[p][ii][jj] * grad_b[p][ii][jj];
                  }

                  ADType kdc = 0;
                  for (int ib = 0; ib < 1; ib++) {
                    ADType bt = beta[ib];
                    if (bt > 1.0) {
                    kdc += fabs(Y_inv * Z) * pow(hdc, bt) * pow(fabs(b[ii][jj]), 1 - bt);
                    } else {
                    kdc += fabs(Y_inv * Z) * hdc *
                           pow(inner, bt / 2 - 1);

                    }
                    // kdc += pow(inner, bt / 2 - 1);
                  }
                  kdc *= 0.5;
                  for (int r = 0; r < VIM; r++) {
                    diffusion += kdc * grad_b[r][ii][jj] * bf[eqn]->grad_phi[i][r];
                  }

                  diffusion *= yzbeta_factor * det_J * wt * h3;
                  diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
                } else if (vn->shockcaptureModel == SC_DCDD) {
                  dbl dcdd_factor = vn->shockcapture;
                  ADType tmp = 0.0;
                  ADType s[DIM] = {0.0};
                  ADType r[DIM] = {0.0};
                  for (int w = 0; w < dim; w++) {
                    tmp += (v[w] - x_dot[w]) * (v[w] - x_dot[w]);
                  }
                  tmp = 1.0 / (sqrt(tmp + 1e-32));
                  for (int w = 0; w < dim; w++) {
                    s[w] = (v[w] - x_dot[w]) * tmp;
                  }
                  ADType mags = 0;
                  for (int w = 0; w < dim; w++) {
                    mags += (grad_b[w][ii][jj] * grad_b[w][ii][jj]);
                  }
                  mags = 1.0 / (sqrt(mags + 1e-32));
                  for (int w = 0; w < dim; w++) {
                    r[w] = grad_b[w][ii][jj] * mags;
                  }

                  ADType he = 0.0;
                  for (int q = 0; q < ei[pg->imtrx]->dof[eqn]; q++) {
                    ADType tmp = 0;
                    for (int w = 0; w < dim; w++) {
                      tmp += bf[eqn]->grad_phi[q][w] * bf[eqn]->grad_phi[q][w];
                    }
                    he += 1.0 / sqrt(tmp);
                  }

                  tmp = 0;
                  for (int q = 0; q < ei[pg->imtrx]->dof[eqn]; q++) {
                    for (int w = 0; w < dim; w++) {
                      tmp += fabs(r[w] * bf[eqn]->grad_phi[q][w]);
                    }
                  }
                  ADType hrgn = 1.0 / (tmp + 1e-14);

                  ADType magv = 0.0;
                  for (int q = 0; q < VIM; q++) {
                    magv += v[q] * v[q];
                  }
                  magv = sqrt(magv + 1e-32);

                  ADType tau_dcdd =
                      0.5 * he * magv * magv * pow((1.0 / (mags + 1e-16)) * hrgn, 1.0);
                  // ADType tau_dcdd = (1.0 / mags) * hrgn * hrgn;
                  tau_dcdd = 1 / sqrt(1.0 / (supg_tau * supg_tau + 1e-32) +
                                      (supg_tau * supg_tau * supg_tau * supg_tau + 1e-32) +
                                      1.0 / (tau_dcdd * tau_dcdd + 1e-32));
                  ADType ss[DIM][DIM] = {{0.0}};
                  ADType rr[DIM][DIM] = {{0.0}};
                  ADType rdots = 0.0;
                  for (int w = 0; w < dim; w++) {
                    for (int z = 0; z < dim; z++) {
                      ss[w][z] = s[w] * s[z];
                      rr[w][z] = r[w] * r[z];
                    }
                    rdots += r[w] * s[w];
                  }

                  ADType inner_tensor[DIM][DIM] = {{0.0}};
                  for (int w = 0; w < dim; w++) {
                    for (int z = 0; z < dim; z++) {
                      inner_tensor[w][z] = rr[w][z] - rdots * rdots * ss[w][z];
                    }
                  }

                  ADType gs_inner_dot[DIM] = {0.0};
                  for (int w = 0; w < dim; w++) {
                    ADType tmp = 0.;
                    for (int z = 0; z < dim; z++) {
                      tmp += grad_b[w][ii][jj] * inner_tensor[w][z];
                    }
                    gs_inner_dot[w] = tmp;
                    // gs_inner_dot[w] = grad_s[w][ii][jj];
                  }

                  for (int w = 0; w < dim; w++) {
                    diffusion -= tau_dcdd * gs_inner_dot[w] * bf[eqn]->grad_phi[i][w];
                  }
                  diffusion *= dcdd_factor * det_J * wt * h3;
                }
              }

              /*
               * Source term...
               */

              ADType source = 0.;
              if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                // consider whether saramitoCoeff should multiply here
                source = source_term[ii][jj];
                source *= wt_func * det_J * h3 * wt;

                source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
              }

              /*
               * Add contributions to this residual (globally into Resid, and
               * locally into an accumulator)
               */

              lec->R[LEC_R_INDEX(upd->ep[pg->imtrx][eqn], i)] +=
                  mass.val() + advection.val() + diffusion.val() + source.val();
              resid[ii][jj][i] += mass + advection + diffusion + source;
            }
          }
        }
      }
    }

    /*
     * Jacobian terms...
     */

    if (af->Assemble_Jacobian) {
      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {
          if (ii <= jj) /* since the stress tensor is symmetric, only assemble the upper half */
          {
            eqn = R_s[mode][ii][jj];
            peqn = upd->ep[pg->imtrx][eqn];

            for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
              for (int var = V_FIRST; var < V_LAST; var++) {
                if (pd->v[pg->imtrx][var]) {
                  pvar = upd->vp[pg->imtrx][var];
                  for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                    lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += resid[ii][jj][i].dx(ad_fv.offset[var] + j);
                  }
                }
              }
            }
          }
        }
      }
    }

  } /* End loop over modes */

  return (status);
}