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
#include "mm_fill_stress_log_conf.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "az_aztec.h"
#include "el_elm.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_eh.h"
#include "mm_fill_ls.h"
#include "mm_fill_stabilization.h"
#include "mm_fill_stress_legacy.h"
#include "mm_mp.h"
#include "mm_mp_const.h"
#include "mm_mp_structs.h"
#include "mm_viscosity.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "std.h"

// Epsilon to help with repeated eigenvalues in the decompositon
#ifndef ANALYTIC_EIG_EPSILON
#define ANALYTIC_EIG_EPSILON 1e-14
#endif

// direct call to a fortran LAPACK eigenvalue routine
extern FSUB_TYPE dsyev_(char *JOBZ,
                        char *UPLO,
                        int *N,
                        double *A,
                        int *LDA,
                        double *W,
                        double *WORK,
                        int *LWORK,
                        int *INFO,
                        int len_jobz,
                        int len_uplo);

#define ASSEMBLE_STRESS_LOG_CONF_NEW
#ifdef ASSEMBLE_STRESS_LOG_CONF_NEW

static void advective_decomposition(dbl grad_v[DIM][DIM],
                                    dbl xi,
                                    dbl s[DIM][DIM],
                                    dbl R[DIM][DIM],
                                    dbl R_T[DIM][DIM],
                                    dbl eig_values[DIM],
                                    dbl d_R[DIM][DIM][DIM][DIM],
                                    dbl d_R_T[DIM][DIM][DIM][DIM],
                                    dbl d_eig_values[DIM][DIM][DIM],
                                    bool compute_jacobian_entries,
                                    dbl advective_term[DIM][DIM],
                                    dbl d_advective_term_ds[DIM][DIM][DIM][DIM]) {
  dbl inner[DIM][DIM] = {{0.}};
  dbl M[DIM][DIM] = {{0.}};
  dbl w[DIM][DIM] = {{0.}};
  dbl M_eye[DIM][DIM] = {{0.}};
  dbl tmp[DIM][DIM];
  dbl omega[DIM][DIM];
  dbl B[DIM][DIM];

  if (DOUBLE_NONZERO(xi)) {
    for (int i = 0; i < VIM; i++) {
      for (int j = 0; j < VIM; j++) {
        inner[i][j] = grad_v[j][i] - 0.5 * xi * (grad_v[i][j] + grad_v[j][i]);
      }
    }
  } else {
    for (int i = 0; i < VIM; i++) {
      for (int j = 0; j < VIM; j++) {
        inner[i][j] = grad_v[j][i];
      }
    }
  }

  tensor_dot(R_T, inner, tmp, VIM);
  tensor_dot(tmp, R, M, VIM);

  for (int i = 0; i < VIM; i++) {
    M_eye[i][i] = M[i][i];
    for (int j = 0; j < VIM; j++) {
      if (j > i) {
        w[i][j] = (eig_values[j] * M[i][j] + eig_values[i] * M[j][i]) /
                  (eig_values[j] - eig_values[i] + 1e-16);
        w[j][i] = -w[i][j];
      }
    }
  }

  tensor_dot(R, w, tmp, VIM);
  tensor_dot(tmp, R_T, omega, VIM);

  tensor_dot(R, M_eye, tmp, VIM);
  tensor_dot(tmp, R_T, B, VIM);

  dbl omega_s[DIM][DIM];
  dbl s_omega[DIM][DIM];
  tensor_dot(omega, s, omega_s, VIM);
  tensor_dot(s, omega, s_omega, VIM);
  for (int i = 0; i < VIM; i++) {
    for (int j = 0; j < VIM; j++) {
      advective_term[i][j] = -(omega_s[i][j] - s_omega[i][j]) - 2 * B[i][j];
    }
  }
  if (compute_jacobian_entries) {
    dbl d_tmp[DIM][DIM];
    for (int p = 0; p < VIM; p++) {
      for (int q = 0; q < VIM; q++) {
        dbl d_M[DIM][DIM] = {{0.}};
        tensor_dot(d_R_T[p][q], inner, tmp, VIM);
        tensor_dot(tmp, R, d_tmp, VIM);

        for (int i = 0; i < VIM; i++) {
          for (int j = 0; j < VIM; j++) {
            d_M[i][j] += d_tmp[i][j];
          }
        }

        tensor_dot(R_T, inner, tmp, VIM);
        tensor_dot(tmp, d_R[p][q], d_tmp, VIM);

        for (int i = 0; i < VIM; i++) {
          for (int j = 0; j < VIM; j++) {
            d_M[i][j] += d_tmp[i][j];
          }
        }

        for (int i = 0; i < VIM; i++) {
          M_eye[i][i] = M[i][i];
          for (int j = 0; j < VIM; j++) {
            if (i != j) {
              w[i][j] = (eig_values[j] * M[i][j] - eig_values[i] * M[j][i]) /
                        (eig_values[j] - eig_values[i] + 1e-16);
              if (j < i) {
                w[i][j] *= -1.;
              }
            }
          }
        }

        tensor_dot(R_T, w, tmp, VIM);
        tensor_dot(tmp, R, omega, VIM);

        tensor_dot(R_T, M_eye, tmp, VIM);
        tensor_dot(tmp, R, B, VIM);
      }
    }
  }
}

static void source_term_logc(int mode,
                             dbl eig_values[DIM],
                             dbl R[DIM][DIM],
                             dbl R_T[DIM][DIM],
                             dbl d_eig_values[DIM][DIM][DIM],
                             dbl d_R[DIM][DIM][DIM][DIM],
                             dbl d_R_T[DIM][DIM][DIM][DIM],
                             dbl source_term[DIM][DIM],
                             dbl d_source_term[DIM][DIM][DIM][DIM]) {
  dbl lambda = 0;
  if (ve[mode]->time_constModel == CONSTANT) {
    lambda = ve[mode]->time_const;
  } else {
    GOMA_EH(GOMA_ERROR, "Unknown time constant model for log conformation");
  }
  switch (vn->ConstitutiveEquation) {
  case OLDROYDB: {
    dbl tmp[DIM][DIM];
    dbl inner[DIM][DIM] = {{0.}};
    for (int i = 0; i < VIM; i++) {
      inner[i][i] = (1.0 / lambda) * (1.0 / (eig_values[i] + 1e-16) - 1.0);
    }

    tensor_dot(R, inner, tmp, VIM);
    tensor_dot(tmp, R_T, source_term, VIM);

    if (af->Assemble_Jacobian && d_source_term != NULL) {
      memset(d_source_term, 0, sizeof(dbl) * DIM * DIM * DIM * DIM);

      for (int p = 0; p < VIM; p++) {
        for (int q = 0; q < VIM; q++) {
          if (q >= p) {
            dbl d_inner[DIM][DIM] = {{0.}};
            dbl d_tmp[DIM][DIM] = {{0.}};
            for (int i = 0; i < VIM; i++) {
              d_inner[i][i] = (1.0 / lambda) * (1.0 / (d_eig_values[p][q][i] + 1e-16) - 1.0);
            }

            tensor_dot(R, d_inner, tmp, VIM);
            tensor_dot(tmp, R_T, d_tmp, VIM);
            for (int i = 0; i < VIM; i++) {
              for (int j = 0; j < VIM; j++) {
                d_source_term[p][q][i][j] += d_tmp[i][j];
              }
            }
            tensor_dot(d_R[p][q], inner, tmp, VIM);
            tensor_dot(tmp, R_T, d_tmp, VIM);
            for (int i = 0; i < VIM; i++) {
              for (int j = 0; j < VIM; j++) {
                d_source_term[p][q][i][j] += d_tmp[i][j];
              }
            }
            tensor_dot(R, inner, tmp, VIM);
            tensor_dot(tmp, d_R_T[p][q], d_tmp, VIM);
            for (int i = 0; i < VIM; i++) {
              for (int j = 0; j < VIM; j++) {
                d_source_term[p][q][i][j] += d_tmp[i][j];
              }
            }
          }
        }
      }
    }
  } break;
  default:
    GOMA_EH(GOMA_ERROR, "Unknown constitutive equation for log conformation");
  }
}

/*
 * This routine assembles the stress with a log-conformation tensor formulation.
 */
int assemble_stress_log_conf(dbl tt, dbl dt, PG_DATA *pg_data) {

  int dim = pd->Num_Dim;
  int R_s[MAX_MODES][DIM][DIM];
  int v_s[MAX_MODES][DIM][DIM];

  int eqn = R_STRESS11;
  // Check if we are actually needed
  if (!pd->e[pg->imtrx][eqn]) {
    return 0;
  }

  // Load pointers
  (void)stress_eqn_pointer(v_s);
  (void)stress_eqn_pointer(R_s);

  dbl wt = fv->wt;
  dbl det_J = bf[eqn]->detJ;
  dbl h3 = fv->h3;

  dbl supg = 0;
  if (vn->wt_funcModel == GALERKIN) {
    supg = 0.0;
  } else if (vn->wt_funcModel == SUPG) {
    supg = vn->wt_func;
  }

  dbl xi = 0;

  dbl g[DIM][DIM];
  if (vn->evssModel == LOG_CONF_GRADV) {
    for (int i = 0; i < VIM; i++) {
      for (int j = 0; j < VIM; j++) {
        g[i][j] = fv->grad_v[i][j];
      }
    }
  } else {
    for (int i = 0; i < VIM; i++) {
      for (int j = 0; j < VIM; j++) {
        g[i][j] = fv->G[i][j];
      }
    }
  }

  SUPG_terms supg_terms;
  if (supg != 0.0) {
    supg_tau(&supg_terms, dim, 1e-8, pg_data, dt, true, eqn);
  }
  // Loop over modes
  for (int mode = 0; mode < vn->modes; mode++) {
    // Load up constants and some pointers
    dbl s[DIM][DIM], exp_s[DIM][DIM];
    dbl R[DIM][DIM], R_T[DIM][DIM];
    dbl eig_values[DIM];
    dbl s_dot[DIM][DIM];
    dbl grad_s[DIM][DIM][DIM];
    dbl d_grad_s_dmesh[DIM][DIM][DIM][DIM][MDE];
    load_modal_pointers(mode, tt, dt, s, s_dot, grad_s, d_grad_s_dmesh);
    // compute_exp_s(s, exp_s, eig_values, R);
    analytical_exp_s(s, exp_s, eig_values, R, NULL, NULL, NULL);
    for (int i = 0; i < VIM; i++) {
      for (int j = 0; j < VIM; j++) {
        R_T[i][j] = R[j][i];
      }
    }

    dbl source_term[DIM][DIM];
    dbl advective_term[DIM][DIM];
    advective_decomposition(g, xi, s, R, R_T, eig_values, NULL, NULL, NULL, false, advective_term,
                            NULL);
    source_term_logc(mode, eig_values, R, R_T, NULL, NULL, NULL, source_term, NULL);
    if (af->Assemble_Residual) {
      for (int a = 0; a < VIM; a++) {
        for (int b = 0; b < VIM; b++) {
          if (a <= b) {
            eqn = R_s[mode][a][b];

            for (int i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
              dbl wt_func = bf[eqn]->phi[i];

              // SUPG weighting, this is SUPG with s, not e^s
              if (DOUBLE_NONZERO(supg)) {
                for (int w = 0; w < dim; w++) {
                  wt_func += supg * supg_terms.supg_tau * fv->v[w] * bf[eqn]->grad_phi[i][w];
                }
              }

              dbl mass = 0.0;
              if (pd->TimeIntegration != STEADY) {
                if (pd->e[pg->imtrx][eqn] & T_MASS) {
                  mass = s_dot[a][b];
                  mass *= wt_func * det_J * wt * h3;
                  mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                }
              }

              dbl advection = 0.;
              if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                for (int k = 0; k < VIM; k++) {
                  advection += fv->v[k] * grad_s[k][a][b];
                  if (pd->gv[R_MESH1] && pd->TimeIntegration != STEADY)
                    advection -= fv_dot->x[k] * grad_s[k][a][b];
                }
                advection += advective_term[a][b];
                advection *= wt_func * det_J * wt * h3;
                advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
              }

              dbl diffusion = 0.;
              if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
                diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
              }

              dbl source = 0.0;
              if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                source -= source_term[a][b];
                source *= wt_func * det_J * h3 * wt;
                source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
              }
              lec->R[LEC_R_INDEX(upd->ep[pg->imtrx][eqn], i)] +=
                  mass + advection + diffusion + source;
            } // i loop
          }   // if a<=b
        }     // b loop
      }       // a loop
    }         // if Residual
  }
  return 0;
}
#else
/*
 * This routine assembles the stress with a log-conformation tensor formulation.
 */
int assemble_stress_log_conf(dbl tt, dbl dt, PG_DATA *pg_data) {
  int dim, q, a, b, w;
  int eqn, siz;

  int i, j, status, mode;
  int logc_gradv = 0;
  dbl v[DIM];
  dbl x_dot[DIM];
  dbl h3;

  dbl grad_v[DIM][DIM];
  dbl gamma[DIM][DIM];
  dbl det_J;

  dbl mass;
  dbl advection;
  dbl source;
  dbl source_term1[DIM][DIM];

  dbl wt_func;
  dbl wt;
  dbl tmp1[DIM][DIM], tmp2[DIM][DIM], tmp3[DIM][DIM];
  dbl advection_term1[DIM][DIM];

  // Variables for stress, velocity gradient
  int R_s[MAX_MODES][DIM][DIM];
  int v_s[MAX_MODES][DIM][DIM];
  dbl s[DIM][DIM], exp_s[DIM][DIM];
  dbl s_dot[DIM][DIM];
  dbl grad_s[DIM][DIM][DIM];
  dbl d_grad_s_dmesh[DIM][DIM][DIM][DIM][MDE];
  dbl gt[DIM][DIM];

  // Polymer viscosity
  dbl mup;
  VISCOSITY_DEPENDENCE_STRUCT d_mup_struct;
  VISCOSITY_DEPENDENCE_STRUCT *d_mup = &d_mup_struct;

  // Temperature shift
  dbl at = 0.0;
  dbl wlf_denom;

  // Consitutive prameters
  dbl alpha;
  dbl lambda = 0;
  dbl d_lambda;
  dbl eps;
  dbl Z = 1.0;

  // Decomposition of velocity vector
  dbl M1[DIM][DIM];
  dbl eig_values[DIM];
  dbl R1[DIM][DIM];
  dbl R1_T[DIM][DIM];
  dbl Rt_dot_gradv[DIM][DIM];
  dbl D[DIM][DIM];
  dbl D_dot_D[DIM][DIM];

  // Advective terms
  dbl v_dot_del_s[DIM][DIM];
  dbl x_dot_del_s[DIM][DIM];

  // Trace of stress
  dbl trace = 0.0;

  // SUPG terms
  dbl supg = 0;

  status = 0;
  if (vn->evssModel == LOG_CONF_GRADV) {
    logc_gradv = 1;
  }

  eqn = R_STRESS11;
  // Check if we are actually needed
  if (!pd->e[pg->imtrx][eqn]) {
    return (status);
  }

  dim = pd->Num_Dim;
  wt = fv->wt;
  det_J = bf[eqn]->detJ;
  h3 = fv->h3;

  // Load pointers
  (void)stress_eqn_pointer(v_s);
  (void)stress_eqn_pointer(R_s);

  memset(s, 0, sizeof(double) * DIM * DIM);
  memset(exp_s, 0, sizeof(double) * DIM * DIM);

  // Load up field variables
  for (a = 0; a < WIM; a++) {
    // Velocity
    v[a] = fv->v[a];
    //
    if (pd->TimeIntegration != STEADY && pd->gv[MESH_DISPLACEMENT1 + a]) {
      x_dot[a] = fv_dot->x[a];
    } else {
      x_dot[a] = 0.0;
    }
  }

  // Velocity gradient
  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      grad_v[a][b] = fv->grad_v[a][b];
    }
  }

  // Shear rate
  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      gamma[a][b] = grad_v[a][b] + grad_v[b][a];
    }
  }

  // Velocity gradient projection
  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      gt[a][b] = fv->G[b][a];
    }
  }

  if (vn->wt_funcModel == GALERKIN) {
    supg = 0.0;
  } else if (vn->wt_funcModel == SUPG) {
    supg = vn->wt_func;
  }

  SUPG_terms supg_terms;
  if (supg != 0.0) {
    supg_tau(&supg_terms, dim, 1e-8, pg_data, dt, true, eqn);
  }

  dbl dcdd_factor = 0.0;
  if (vn->shockcaptureModel == SC_DCDD) {
    dcdd_factor = vn->shockcapture;
  } else if (vn->shockcaptureModel != SC_NONE) {
    GOMA_EH(GOMA_ERROR, "Unknown shock capture model, only DCDD supported for LOG_CONF");
  }

  // Shift factor
  if (pd->gv[TEMPERATURE]) {
    if (vn->shiftModel == CONSTANT) {
      at = vn->shift[0];
    } else if (vn->shiftModel == MODIFIED_WLF) {
      wlf_denom = vn->shift[1] + fv->T - mp->reference[TEMPERATURE];
      if (wlf_denom != 0.0) {
        at = exp(vn->shift[0] * (mp->reference[TEMPERATURE] - fv->T) / wlf_denom);
      } else {
        at = 1.0;
      }
    }
  } else {
    at = 1.0;
  }

  // Loop over modes
  for (mode = 0; mode < vn->modes; mode++) {
    // Load up constants and some pointers
    load_modal_pointers(mode, tt, dt, s, s_dot, grad_s, d_grad_s_dmesh);

    // Polymer viscosity
    mup = viscosity(ve[mode]->gn, gamma, d_mup);

    // Giesekus mobility parameter
    alpha = ve[mode]->alpha;

    // Polymer time constant
    if (ve[mode]->time_constModel == CONSTANT) {
      lambda = ve[mode]->time_const;
    } else if (ve[mode]->time_constModel == CARREAU || ve[mode]->time_constModel == POWER_LAW) {
      lambda = mup / ve[mode]->time_const;
    }

    dbl xi = 0;
    if (ve[mode]->xiModel == CONSTANT) {
      xi = ve[mode]->xi;
    } else if (ls != NULL && ve[mode]->xiModel == VE_LEVEL_SET) {
      double pos_xi = ve[mode]->pos_ls.xi;
      double neg_xi = ve[mode]->xi;
      double width = ls->Length_Scale;
      int err = level_set_property(neg_xi, pos_xi, width, &xi, NULL);
      GOMA_EH(err, "level_set_property() failed for ptt xi parameter.");
    } else {
      GOMA_EH(GOMA_ERROR, "Unknown PTT Xi parameter model");
    }

    if (lambda <= 0.) {
      GOMA_WH(-1, "Trouble: Zero relaxation time with LOG_CONF");
      return -1;
    }

#ifdef ANALEIG_PLEASE
    analytical_exp_s(s, exp_s, eig_values, R1, NULL, NULL, NULL);
#else
    compute_exp_s(s, exp_s, eig_values, R1);
#endif

    /* Check to make sure eigenvalues are positive (negative eigenvalues will not
       work for log-conformation formulation). These eigenvalues are for the
       conformation tensor, not the log-conformation tensor. */
    if (eig_values[0] < 0. || eig_values[1] < 0. || (VIM > 2 && eig_values[2] < 0.)) {
      GOMA_WH(-1, "Error: Negative eigenvalue for conformation tensor");
      return -1;
    }

    memset(D, 0, sizeof(double) * DIM * DIM);
    D[0][0] = eig_values[0];
    D[1][1] = eig_values[1];
    if (VIM > 2) {
      D[2][2] = eig_values[2];
    }
    (void)tensor_dot(D, D, D_dot_D, VIM);

    // Decompose velocity gradient

    memset(M1, 0, sizeof(double) * DIM * DIM);
    memset(R1_T, 0, sizeof(double) * DIM * DIM);

    for (i = 0; i < VIM; i++) {
      for (j = 0; j < VIM; j++) {
        R1_T[i][j] = R1[j][i];
      }
    }

    for (i = 0; i < VIM; i++) {
      for (j = 0; j < VIM; j++) {
        Rt_dot_gradv[i][j] = 0.;
        for (w = 0; w < VIM; w++) {
          if (DOUBLE_NONZERO(xi)) {
            if (logc_gradv) {
              Rt_dot_gradv[i][j] +=
                  R1_T[i][w] * (grad_v[j][w] - 0.5 * xi * (grad_v[j][w] + grad_v[w][j]));
            } else {
              Rt_dot_gradv[i][j] += R1_T[i][w] * (gt[w][j] - 0.5 * xi * (gt[j][w] + gt[w][j]));
            }
          } else {
            if (logc_gradv) {
              Rt_dot_gradv[i][j] += R1_T[i][w] * grad_v[j][w];
            } else {
              Rt_dot_gradv[i][j] += R1_T[i][w] * gt[w][j];
            }
          }
        }
      }
    }

    for (i = 0; i < VIM; i++) {
      for (j = 0; j < VIM; j++) {
        M1[i][j] = 0.;
        for (w = 0; w < VIM; w++) {
          M1[i][j] += Rt_dot_gradv[i][w] * R1[w][j];
        }
      }
    }

    // Predetermine advective terms
    trace = exp_s[0][0] + exp_s[1][1];
    if (VIM > 2) {
      trace += exp_s[2][2];
    }

    for (a = 0; a < VIM; a++) {
      for (b = 0; b < VIM; b++) {
        v_dot_del_s[a][b] = 0.0;
        x_dot_del_s[a][b] = 0.0;
        for (q = 0; q < dim; q++) {
          v_dot_del_s[a][b] += v[q] * grad_s[q][a][b];
          x_dot_del_s[a][b] += x_dot[q] * grad_s[q][a][b];
        }
      }
    }

    // PTT exponent
    eps = ve[mode]->eps;

    // PTT
    Z = 1;
    if (vn->ConstitutiveEquation == PTT) {
      if (vn->ptt_type == PTT_LINEAR) {
        Z = 1 + eps * (trace - (double)VIM);
      } else if (vn->ptt_type == PTT_EXPONENTIAL) {
        Z = exp(eps * (trace - (double)VIM));
      } else {
        GOMA_EH(GOMA_ERROR, "Unrecognized PTT Form %d", vn->ptt_type);
      }
    }

    siz = sizeof(double) * DIM * DIM;
    memset(tmp1, 0, siz);
    memset(tmp2, 0, siz);
    memset(tmp3, 0, siz);
    memset(advection_term1, 0, siz);
    memset(source_term1, 0, siz);

    for (a = 0; a < VIM; a++) {
      for (b = 0; b < VIM; b++) {
        if (a != b) {
          d_lambda = eig_values[b] - eig_values[a];
          if (DOUBLE_NONZERO(d_lambda)) {
            double eiglog_a = log(DBL_SMALL), eiglog_b = log(DBL_SMALL);
            if (DOUBLE_NONZERO(eig_values[b])) {
              eiglog_b = fmax(eiglog_b, log(eig_values[b]));
            }
            if (DOUBLE_NONZERO(eig_values[a])) {
              eiglog_a = fmax(eiglog_a, log(eig_values[a]));
            }
            tmp1[a][b] += (eiglog_b - eiglog_a) / d_lambda;
            tmp1[a][b] *= (eig_values[a] * M1[b][a] + eig_values[b] * M1[a][b]);
          } else {
            tmp1[a][b] += eig_values[b] * (M1[a][b] + M1[b][a]);
          }
        }
        if (a == b) {
          source_term1[a][b] += Z * (1.0 - D[a][a]) / lambda;
          if (DOUBLE_NONZERO(alpha)) {
            source_term1[a][b] += alpha * (2.0 * D[a][a] - 1.0 - D_dot_D[a][a]) / lambda;
          }
          source_term1[a][b] /= fmax(DBL_SMALL, eig_values[a]);
          source_term1[a][b] += 2.0 * M1[a][a];
        }
      }
    }

    (void)tensor_dot(R1, tmp1, tmp2, VIM);
    (void)tensor_dot(tmp2, R1_T, advection_term1, VIM);
    (void)tensor_dot(R1, source_term1, tmp3, VIM);
    (void)tensor_dot(tmp3, R1_T, source_term1, VIM);

    if (af->Assemble_Residual) {
      for (a = 0; a < VIM; a++) {
        for (b = 0; b < VIM; b++) {
          if (a <= b) {
            eqn = R_s[mode][a][b];

            for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
              wt_func = bf[eqn]->phi[i];

              // SUPG weighting, this is SUPG with s, not e^s
              if (DOUBLE_NONZERO(supg)) {
                for (w = 0; w < dim; w++) {
                  wt_func += supg * supg_terms.supg_tau * v[w] * bf[eqn]->grad_phi[i][w];
                }
              }

              mass = 0.0;
              if (pd->TimeIntegration != STEADY) {
                if (pd->e[pg->imtrx][eqn] & T_MASS) {
                  mass = s_dot[a][b];
                  mass *= wt_func * at * det_J * wt * h3;
                  mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                }
              }

              advection = 0.;
              if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                advection += v_dot_del_s[a][b] - x_dot_del_s[a][b];
                advection -= advection_term1[a][b];
                advection *= wt_func * at * det_J * wt * h3;
                advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
              }

              dbl diffusion = 0.;
              if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
                if (DOUBLE_NONZERO(dcdd_factor)) {
                  dbl tmp = 0.0;
                  dbl s[DIM] = {0.0};
                  dbl r[DIM] = {0.0};
                  for (int w = 0; w < dim; w++) {
                    tmp += (v[w] - x_dot[w]) * (v[w] - x_dot[w]);
                  }
                  tmp = 1.0 / (sqrt(tmp) + 1e-16);
                  for (int w = 0; w < dim; w++) {
                    s[w] = (v[w] - x_dot[w]) * tmp;
                  }
                  dbl mags = 0;
                  for (int w = 0; w < dim; w++) {
                    mags += (grad_s[w][a][b] * grad_s[w][a][b]);
                  }
                  mags = 1.0 / (sqrt(mags) + 1e-14);
                  for (int w = 0; w < dim; w++) {
                    r[w] = grad_s[w][a][b] * mags;
                  }

                  dbl he = 0.0;
                  for (int q = 0; q < ei[pg->imtrx]->dof[eqn]; q++) {
                    dbl tmp = 0;
                    for (int w = 0; w < dim; w++) {
                      tmp += bf[eqn]->grad_phi[q][w] * bf[eqn]->grad_phi[q][w];
                    }
                    he += 1.0 / sqrt(tmp);
                  }

                  dbl G[DIM][DIM];
                  get_metric_tensor(bf[eqn]->B, dim, ei[pg->imtrx]->ielem_type, G);

                  dbl hrgn = 0.0;
                  // for (int w = 0; w < dim; w++) {
                  //   for (int z = 0; z < dim; z++) {
                  //     //tmp += fabs(r[w] * G[w][z] * r[z]);
                  //   }
                  // }
                  tmp = 0;
                  for (int q = 0; q < ei[pg->imtrx]->dof[eqn]; q++) {
                    for (int w = 0; w < dim; w++) {
                      tmp += fabs(r[w] * bf[eqn]->grad_phi[q][w]);
                    }
                  }
                  hrgn = 1.0 / (tmp + 1e-14);

                  dbl magv = 0.0;
                  for (int q = 0; q < VIM; q++) {
                    magv += v[q] * v[q];
                  }
                  magv = sqrt(magv);

                  dbl tau_dcdd = 0.5 * he * magv * magv * pow((1.0 / (mags + 1e-16)) * hrgn, 1.0);
                  // dbl tau_dcdd = (1.0 / mags) * hrgn * hrgn;
                  tau_dcdd = 1 / sqrt(1.0 / (supg_terms.supg_tau * supg_terms.supg_tau + 1e-32) +
                                      (supg_terms.supg_tau * supg_terms.supg_tau *
                                           supg_terms.supg_tau * supg_terms.supg_tau +
                                       1e-32) +
                                      1.0 / (tau_dcdd * tau_dcdd + 1e-32));
                  dbl ss[DIM][DIM] = {{0.0}};
                  dbl rr[DIM][DIM] = {{0.0}};
                  dbl rdots = 0.0;
                  for (int w = 0; w < dim; w++) {
                    for (int z = 0; z < dim; z++) {
                      ss[w][z] = s[w] * s[z];
                      rr[w][z] = r[w] * r[z];
                    }
                    rdots += r[w] * s[w];
                  }

                  dbl inner_tensor[DIM][DIM] = {{0.0}};
                  for (int w = 0; w < dim; w++) {
                    for (int z = 0; z < dim; z++) {
                      inner_tensor[w][z] = rr[w][z] - rdots * rdots * ss[w][z];
                    }
                  }

                  dbl gs_inner_dot[DIM] = {0.0};
                  for (int w = 0; w < dim; w++) {
                    dbl tmp = 0.;
                    for (int z = 0; z < dim; z++) {
                      tmp += grad_s[w][a][b] * inner_tensor[w][z];
                    }
                    gs_inner_dot[w] = tmp;
                    // gs_inner_dot[w] = grad_s[w][a][b];
                  }

                  for (int w = 0; w < dim; w++) {
                    diffusion += tau_dcdd * gs_inner_dot[w] * bf[eqn]->grad_phi[i][w];
                  }
                  diffusion *= dcdd_factor * det_J * wt * h3;
                }
                diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
              }

              source = 0.0;
              if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                source -= source_term1[a][b];
                source *= wt_func * det_J * h3 * wt;
                source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
              }
              lec->R[LEC_R_INDEX(upd->ep[pg->imtrx][eqn], i)] +=
                  mass + advection + diffusion + source;
            } // i loop
          }   // if a<=b
        }     // b loop
      }       // a loop
    }         // if Residual
  }
  return (status);
}
#endif

int assemble_stress_log_conf_transient(dbl tt, dbl dt, PG_DATA *pg_data) {
  int dim, p, q, a, b, w;
  int eqn, siz;

  int i, j, status, mode;
  int logc_gradv = 0;
  dbl v[DIM];
  dbl x_dot[DIM];
  dbl h3;

  dbl grad_v[DIM][DIM];
  dbl gamma[DIM][DIM];
  dbl det_J;

  dbl mass;
  dbl advection;
  dbl source;
  dbl source_term1[DIM][DIM];

  dbl wt_func;
  dbl wt;
  dbl tmp1[DIM][DIM], tmp2[DIM][DIM], tmp3[DIM][DIM];
  dbl advection_term1[DIM][DIM];

  // Variables for stress, velocity gradient
  int R_s[MAX_MODES][DIM][DIM];
  int v_s[MAX_MODES][DIM][DIM];
  dbl s[DIM][DIM], exp_s[DIM][DIM];
  dbl s_dot[DIM][DIM];
  dbl grad_s[DIM][DIM][DIM];
  dbl d_grad_s_dmesh[DIM][DIM][DIM][DIM][MDE];
  dbl gt[DIM][DIM];

  // Polymer viscosity
  dbl mup;
  VISCOSITY_DEPENDENCE_STRUCT d_mup_struct;
  VISCOSITY_DEPENDENCE_STRUCT *d_mup = &d_mup_struct;

  // Temperature shift
  dbl at = 0.0;
  dbl wlf_denom;

  // Consitutive prameters
  dbl alpha;
  dbl lambda = 0;
  dbl d_lambda;
  dbl eps;
  dbl Z = 1.0;

  // Decomposition of velocity vector
  dbl M1[DIM][DIM];
  dbl eig_values[DIM];
  dbl R1[DIM][DIM];
  dbl R1_T[DIM][DIM];
  dbl Rt_dot_gradv[DIM][DIM];
  dbl D[DIM][DIM];
  dbl D_dot_D[DIM][DIM];

  // Advective terms
  dbl v_dot_del_s[DIM][DIM];
  dbl x_dot_del_s[DIM][DIM];

  // Trace of stress
  dbl trace = 0.0;

  // SUPG terms
  dbl supg = 0;

  status = 0;
  if (vn->evssModel == LOG_CONF_TRANSIENT_GRADV) {
    logc_gradv = 1;
  }

  eqn = R_STRESS11;
  // Check if we are actually needed
  if (!pd->e[pg->imtrx][eqn]) {
    return (status);
  }

  dim = pd->Num_Dim;
  wt = fv->wt;
  det_J = bf[eqn]->detJ;
  h3 = fv->h3;

  // Load pointers
  (void)stress_eqn_pointer(v_s);
  (void)stress_eqn_pointer(R_s);

  memset(s, 0, sizeof(double) * DIM * DIM);
  memset(exp_s, 0, sizeof(double) * DIM * DIM);

  // Load up field variables
  for (a = 0; a < dim; a++) {
    // Velocity
    v[a] = fv->v[a];
    //
    if (pd->TimeIntegration != STEADY && pd->gv[MESH_DISPLACEMENT1 + a]) {
      x_dot[a] = fv_dot->x[a];
    } else {
      x_dot[a] = 0.0;
    }
  }

  // Velocity gradient
  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      grad_v[a][b] = fv->grad_v[a][b];
    }
  }

  // Shear rate
  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      gamma[a][b] = grad_v[a][b] + grad_v[b][a];
    }
  }

  // Velocity gradient projection
  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      gt[a][b] = fv->G[b][a];
    }
  }

  if (vn->wt_funcModel == GALERKIN) {
    supg = 0.0;
  } else if (vn->wt_funcModel == SUPG) {
    supg = vn->wt_func;
  }

  const bool saramitoEnabled =
      (vn->ConstitutiveEquation == SARAMITO_OLDROYDB || vn->ConstitutiveEquation == SARAMITO_PTT ||
       vn->ConstitutiveEquation == SARAMITO_GIESEKUS);

  dbl saramitoCoeff = 1.;

  SUPG_terms supg_terms;
  if (supg != 0.0) {
    supg_tau_shakib(&supg_terms, dim, dt, 1e-6, POLYMER_STRESS11);
  }

  // Shift factor
  if (pd->gv[TEMPERATURE]) {
    if (vn->shiftModel == CONSTANT) {
      at = vn->shift[0];
    } else if (vn->shiftModel == MODIFIED_WLF) {
      wlf_denom = vn->shift[1] + fv->T - mp->reference[TEMPERATURE];
      if (wlf_denom != 0.0) {
        at = exp(vn->shift[0] * (mp->reference[TEMPERATURE] - fv->T) / wlf_denom);
      } else {
        at = 1.0;
      }
    }
  } else {
    at = 1.0;
  }

  // Loop over modes
  for (mode = 0; mode < vn->modes; mode++) {
    // Load up constants and some pointers
    load_modal_pointers(mode, tt, dt, s, s_dot, grad_s, d_grad_s_dmesh);

    // Polymer viscosity
    mup = viscosity(ve[mode]->gn, gamma, d_mup);

    // Giesekus mobility parameter
    alpha = ve[mode]->alpha;

    // Polymer time constant
    if (ve[mode]->time_constModel == CONSTANT) {
      lambda = ve[mode]->time_const;
    } else if (ve[mode]->time_constModel == CARREAU || ve[mode]->time_constModel == POWER_LAW) {
      lambda = mup / ve[mode]->time_const;
    }

#ifdef ANALEIG_PLEASE
    analytical_exp_s(fv_old->S[mode], exp_s, eig_values, R1, NULL, NULL, NULL);
#else
    compute_exp_s(fv_old->S[mode], exp_s, eig_values, R1);
#endif

    dbl tau[DIM][DIM] = {{0.0}};
    if (saramitoEnabled == TRUE) {
      for (int i = 0; i < VIM; i++) {
        for (int j = 0; j < VIM; j++) {
          tau[i][j] = mup / lambda * (exp_s[i][j] - delta(i, j));
        }
      }
      compute_saramito_model_terms(&saramitoCoeff, NULL, tau, ve[mode]->gn, FALSE);
    } else {
      saramitoCoeff = 1.;
    }

    /* Check to make sure eigenvalues are positive (negative eigenvalues will not
       work for log-conformation formulation). These eigenvalues are for the
       conformation tensor, not the log-conformation tensor. */
    if (eig_values[0] < 0. || eig_values[1] < 0. || (VIM > 2 && eig_values[2] < 0.)) {
      GOMA_WH(-1, "Error: Negative eigenvalue for conformation tensor");
      return -1;
    }

    memset(D, 0, sizeof(double) * DIM * DIM);
    D[0][0] = eig_values[0];
    D[1][1] = eig_values[1];
    if (VIM > 2) {
      D[2][2] = eig_values[2];
    }
    (void)tensor_dot(D, D, D_dot_D, VIM);

    // Decompose velocity gradient

    memset(M1, 0, sizeof(double) * DIM * DIM);
    memset(R1_T, 0, sizeof(double) * DIM * DIM);

    for (i = 0; i < VIM; i++) {
      for (j = 0; j < VIM; j++) {
        R1_T[i][j] = R1[j][i];
      }
    }

    for (i = 0; i < VIM; i++) {
      for (j = 0; j < VIM; j++) {
        Rt_dot_gradv[i][j] = 0.;
        for (w = 0; w < VIM; w++) {
          if (logc_gradv) {
            Rt_dot_gradv[i][j] += R1_T[i][w] * grad_v[j][w];
          } else {
            Rt_dot_gradv[i][j] += R1_T[i][w] * gt[w][j];
          }
        }
      }
    }

    for (i = 0; i < VIM; i++) {
      for (j = 0; j < VIM; j++) {
        M1[i][j] = 0.;
        for (w = 0; w < VIM; w++) {
          M1[i][j] += Rt_dot_gradv[i][w] * R1[w][j];
        }
      }
    }

    // Predetermine advective terms
    trace = exp_s[0][0] + exp_s[1][1];
    if (VIM > 2) {
      trace += exp_s[2][2];
    }

    for (a = 0; a < VIM; a++) {
      for (b = 0; b < VIM; b++) {
        v_dot_del_s[a][b] = 0.0;
        x_dot_del_s[a][b] = 0.0;
        for (q = 0; q < dim; q++) {
          v_dot_del_s[a][b] += v[q] * grad_s[q][a][b];
          x_dot_del_s[a][b] += x_dot[q] * grad_s[q][a][b];
        }
      }
    }

    // PTT exponent
    eps = ve[mode]->eps;

    // Exponential term for PTT
    Z = exp(eps * (trace - (double)VIM));

    siz = sizeof(double) * DIM * DIM;
    memset(tmp1, 0, siz);
    memset(tmp2, 0, siz);
    memset(tmp3, 0, siz);
    memset(advection_term1, 0, siz);
    memset(source_term1, 0, siz);

    for (a = 0; a < VIM; a++) {
      for (b = 0; b < VIM; b++) {
        if (a != b) {
          d_lambda = eig_values[b] - eig_values[a];
          if (DOUBLE_NONZERO(d_lambda)) {
            double eiglog_a = log(DBL_SMALL), eiglog_b = log(DBL_SMALL);
            if (DOUBLE_NONZERO(eig_values[b])) {
              eiglog_b = fmax(eiglog_b, log(eig_values[b]));
            }
            if (DOUBLE_NONZERO(eig_values[a])) {
              eiglog_a = fmax(eiglog_a, log(eig_values[a]));
            }
            tmp1[a][b] += (eiglog_b - eiglog_a) / d_lambda;
            tmp1[a][b] *= (eig_values[a] * M1[b][a] + eig_values[b] * M1[a][b]);
          } else {
            tmp1[a][b] += eig_values[b] * (M1[a][b] + M1[b][a]);
          }
        }
        if (a == b) {
          source_term1[a][b] += saramitoCoeff * Z * (1.0 - D[a][a]) / lambda;
          if (DOUBLE_NONZERO(alpha)) {
            source_term1[a][b] += alpha * (2.0 * D[a][a] - 1.0 - D_dot_D[a][a]) / lambda;
          }
          source_term1[a][b] /= fmax(DBL_SMALL, eig_values[a]);
          source_term1[a][b] += 2.0 * M1[a][a];
        }
      }
    }

    (void)tensor_dot(R1, tmp1, tmp2, VIM);
    (void)tensor_dot(tmp2, R1_T, advection_term1, VIM);
    (void)tensor_dot(R1, source_term1, tmp3, VIM);
    (void)tensor_dot(tmp3, R1_T, source_term1, VIM);

    if (af->Assemble_Residual) {
      for (a = 0; a < VIM; a++) {
        for (b = 0; b < VIM; b++) {
          if (a <= b) {
            eqn = R_s[mode][a][b];

            for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
              wt_func = bf[eqn]->phi[i];

              // SUPG weighting, this is SUPG with s, not e^s
              if (DOUBLE_NONZERO(supg)) {
                for (w = 0; w < dim; w++) {
                  wt_func += supg * supg_terms.supg_tau * v[w] * bf[eqn]->grad_phi[i][w];
                }
              }

              mass = 0.0;
              if (pd->TimeIntegration != STEADY) {
                if (pd->e[pg->imtrx][eqn] & T_MASS) {
                  mass = fv_dot->S[mode][a][b];
                  mass *= wt_func * at * det_J * wt * h3;
                  mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                }
              }

              advection = 0.;
              if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                advection += v_dot_del_s[a][b] - x_dot_del_s[a][b];
                advection -= advection_term1[a][b];
                advection *= wt_func * at * det_J * wt * h3;
                advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
              }

              source = 0.0;
              if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                source -= source_term1[a][b];
                source *= wt_func * det_J * h3 * wt;
                source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
              }
              lec->R[LEC_R_INDEX(upd->ep[pg->imtrx][eqn], i)] += mass + advection + source;
            } // i loop
          }   // if a<=b
        }     // b loop
      }       // a loop
    }         // if Residual
    if (af->Assemble_Jacobian) {
      for (a = 0; a < VIM; a++) {
        for (b = 0; b < VIM; b++) {
          if (a <= b) {
            eqn = R_s[mode][a][b];
            int peqn = upd->ep[pg->imtrx][eqn];

            for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
              wt_func = bf[eqn]->phi[i];

              // SUPG weighting, this is SUPG with s, not e^s
              if (DOUBLE_NONZERO(supg)) {
                for (w = 0; w < dim; w++) {
                  wt_func += supg * supg_terms.supg_tau * v[w] * bf[eqn]->grad_phi[i][w];
                }
              }

              /*
               * J_S_S
               */
              for (p = 0; p < VIM; p++) {
                for (q = 0; q < VIM; q++) {
                  int var = v_s[mode][p][q];

                  if (pd->v[pg->imtrx][var]) {
                    int pvar = upd->vp[pg->imtrx][var];
                    for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                      double phi_j = bf[var]->phi[j];
                      mass = 0.;
                      if (pd->TimeIntegration != STEADY) {
                        if (pd->e[pg->imtrx][eqn] & T_MASS) {
                          mass = (1. + 2. * tt) * phi_j / dt * (double)delta(a, p) *
                                 (double)delta(b, q);
                          mass *= h3 * det_J;
                          mass *= wt_func * wt * pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                        }
                      }

                      advection = 0.;

                      if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                        if ((a == p) && (b == q)) {
                          for (int r = 0; r < dim; r++) {
                            advection += (v[r] - x_dot[r]) * bf[var]->grad_phi[j][r];
                          }
                        }

                        advection *= h3 * det_J;

                        advection *= wt_func * wt * pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                      }

                      source = 0;
                      lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += mass + advection + source;
                    }
                  }
                }
              }
            } // i loop
          }   // if a<=b
        }     // b loop
      }       // a loop
    }         // if Residual
  }
  return (status);
}

void compute_exp_s(double s[DIM][DIM],
                   double exp_s[DIM][DIM],
                   double eig_values[DIM],
                   double R[DIM][DIM]) {

  int N = VIM;
  int LDA = N;
  int i, j, k;

  int INFO;
  int LWORK = 20;
  double WORK[LWORK];
  double A[VIM * VIM];
  double EIGEN_MAX = sqrt(sqrt(DBL_MAX));
  double eig_S[DIM];
  memset(eig_values, 0.0, sizeof(double) * DIM);
  memset(eig_S, 0.0, sizeof(double) * DIM);
  memset(WORK, 0, sizeof(double) * LWORK);

  // convert to column major
  for (i = 0; i < VIM; i++) {
    for (j = 0; j < VIM; j++) {
      A[i * VIM + j] = s[j][i];
    }
  }

  // eig solver
  dsyev_("V", "U", &N, A, &LDA, eig_S, WORK, &LWORK, &INFO, 1, 1);
  if (INFO > 0)
    fprintf(stderr, "eigsolver not converged %d\n", INFO);
  if (INFO < 0)
    fprintf(stderr, "eigsolver illegal entry %d\n", INFO);

  // transpose (revert to row major)
  for (i = 0; i < VIM; i++) {
    for (j = 0; j < VIM; j++) {
      R[i][j] = A[j * VIM + i];
    }
  }

  // exponentiate diagonal
  for (i = 0; i < VIM; i++) {
    eig_values[i] = MIN(exp(eig_S[i]), EIGEN_MAX);
  }

  memset(exp_s, 0, sizeof(double) * DIM * DIM);
  for (i = 0; i < VIM; i++) {
    for (j = 0; j < VIM; j++) {
      for (k = 0; k < VIM; k++) {
        exp_s[i][j] += R[i][k] * eig_values[k] * R[j][k];
      }
    }
  }

} // End compute_exp_s

void analytical_exp_s(dbl s[DIM][DIM],
                      dbl exp_s[DIM][DIM],
                      dbl eig_values[DIM],
                      dbl R[DIM][DIM],
                      dbl d_exp_s_ds[DIM][DIM][DIM][DIM],
                      dbl d_eig_values_ds[DIM][DIM][DIM],
                      dbl dR_ds[DIM][DIM][DIM][DIM]) {

  double EIGEN_MAX = sqrt(sqrt(DBL_MAX));
  if (pd->Num_Dim == 3) {
    GOMA_EH(GOMA_ERROR,
            "Unable to compute analytic eigenvalues in 3D must use numerical jacobians");
  }

  if (fabs(s[0][1]) < ANALYTIC_EIG_EPSILON) {
    for (int i = 0; i < VIM; i++) {
      eig_values[i] = s[i][i];
    }
    for (int a = 0; a < VIM; a++) {
      for (int b = 0; b < VIM; b++) {
        if (a == b) {
          R[a][b] = 1.;
        } else {
          R[a][b] = 0.;
        }
      }
    }
  } else {
    dbl m = 0.5 * (s[0][0] + s[1][1]);
    dbl p = s[0][0] * s[1][1] - s[0][1] * s[0][1];
    if ((m + sqrt(m * m - p) < (m - sqrt(m * m - p)))) {
      eig_values[0] = m - sqrt(m * m - p);
      eig_values[1] = m + sqrt(m * m - p);
    } else {
      eig_values[0] = m + sqrt(m * m - p);
      eig_values[1] = m - sqrt(m * m - p);
    }
    if (VIM == 3)
      eig_values[2] = s[2][2];

    dbl norm = 0;

    // first eigenvector
    R[0][0] = eig_values[0] - s[1][1];
    R[0][1] = s[0][1];
    R[0][2] = 0.;

    norm = sqrt(R[0][0] * R[0][0] + R[0][1] * R[0][1]);
    R[0][0] /= norm;
    R[0][1] /= norm;

    // second eigenvector
    R[1][0] = eig_values[1] - s[1][1];
    R[1][1] = s[0][1];
    R[1][2] = 0.;

    norm = sqrt(R[1][0] * R[1][0] + R[1][1] * R[1][1]);
    R[1][0] /= norm;
    R[1][1] /= norm;

    // third eigenvector
    if (VIM == 3) {
      R[2][0] = 0.;
      R[2][1] = 0.;
      R[2][2] = 1.0;
    }
  }

  // exponentiate diagonal
  for (int i = 0; i < VIM; i++) {
    eig_values[i] = MIN(exp(eig_values[i]), EIGEN_MAX);
  }

  memset(exp_s, 0, sizeof(double) * DIM * DIM);
  for (int i = 0; i < VIM; i++) {
    for (int j = 0; j < VIM; j++) {
      for (int k = 0; k < VIM; k++) {
        exp_s[i][j] += R[i][k] * eig_values[k] * R[j][k];
      }
    }
  }
}

void compute_d_exp_s_ds(dbl s[DIM][DIM], // s - stress
                        dbl exp_s[DIM][DIM],
                        dbl d_exp_s_ds[DIM][DIM][DIM][DIM]) {
  double s_p[DIM][DIM], s_n[DIM][DIM];
  double exp_s_p[DIM][DIM], exp_s_n[DIM][DIM];
  double eig_values[DIM];
  double R1[DIM][DIM];
  int i, j, p, q;
  double ds, ds_den, fd = FD_FACTOR;

  memset(d_exp_s_ds, 0, sizeof(double) * DIM * DIM * DIM * DIM);

  for (i = 0; i < VIM; i++) {
    for (j = 0; j < VIM; j++) {
      s_p[i][j] = s_n[i][j] = s[i][j];
    }
  }

  for (i = 0; i < VIM; i++) {
    for (j = i; j < VIM; j++) {

      // perturb s
      ds = fd * s[i][j];
      ds = (fabs(ds) < fd ? fd : ds);
      s_p[i][j] += ds;
      s_n[i][j] -= ds;
      if (i != j) {
        s_p[j][i] = s_p[i][j];
        s_n[j][i] = s_n[i][j];
      }

      ds_den = 0.5 / ds;

// find exp_s at perturbed value
#ifdef ANALEIG_PLEASE
      analytical_exp_s(s_p, exp_s_p, eig_values, R1, NULL, NULL, NULL);
      analytical_exp_s(s_n, exp_s_n, eig_values, R1, NULL, NULL, NULL);

#else
      compute_exp_s(s_p, exp_s_p, eig_values, R1);
      compute_exp_s(s_n, exp_s_n, eig_values, R1);
#endif

      // approximate derivative
      for (p = 0; p < VIM; p++) {
        for (q = 0; q < VIM; q++) {
          d_exp_s_ds[p][q][i][j] = ds_den * (exp_s_p[p][q] - exp_s_n[p][q]);
          if (i != j)
            d_exp_s_ds[p][q][j][i] = d_exp_s_ds[p][q][i][j];
        }
      }
      s_p[i][j] = s_n[i][j] = s[i][j];
      if (i != j) {
        s_p[j][i] = s_n[j][i] = s[j][i];
      }
    }
  }
}

void compute_log_c_with_derivs(dbl s[DIM][DIM], // s - stress
                               dbl exp_s[DIM][DIM],
                               dbl eig_values[DIM],
                               dbl R1[DIM][DIM],
                               dbl d_exp_s_ds[DIM][DIM][DIM][DIM],
                               dbl d_R1_ds[DIM][DIM][DIM][DIM],
                               dbl d_eig_values_ds[DIM][DIM][DIM]) {
  double s_p[DIM][DIM], s_n[DIM][DIM];
  double exp_s_p[DIM][DIM], exp_s_n[DIM][DIM];
  double R1_p[DIM][DIM], R1_n[DIM][DIM];
  double eig_values_p[DIM], eig_values_n[DIM];
  int i, j, p, q;
  double ds, ds_den, fd = FD_FACTOR;

  memset(d_exp_s_ds, 0, sizeof(double) * DIM * DIM * DIM * DIM);
  memset(d_R1_ds, 0, sizeof(double) * DIM * DIM * DIM * DIM);
  memset(d_eig_values_ds, 0, sizeof(double) * DIM * DIM * DIM);

  for (i = 0; i < VIM; i++) {
    for (j = 0; j < VIM; j++) {
      s_p[i][j] = s_n[i][j] = s[i][j];
    }
  }

  for (i = 0; i < VIM; i++) {
    for (j = i; j < VIM; j++) {
      if (j >= i) {

        // perturb s
        ds = fd * s[i][j];
        ds = (fabs(ds) < fd ? fd : ds);
        s_p[i][j] += ds;
        s_n[i][j] -= ds;
        if (i != j) {
          s_p[j][i] = s_p[i][j];
          s_n[j][i] = s_n[i][j];
        }

        ds_den = 0.5 / ds;

        compute_exp_s(s_p, exp_s_p, eig_values_p, R1_p);
        compute_exp_s(s_n, exp_s_n, eig_values_n, R1_n);

        // approximate derivatives
        for (p = 0; p < VIM; p++) {
          d_eig_values_ds[i][j][p] = ds_den * (exp_s_p[p][q] - exp_s_n[p][q]);
          if (i != j)
            d_eig_values_ds[j][i][p] = d_eig_values_ds[i][j][p];
          for (q = 0; q < VIM; q++) {
            d_exp_s_ds[i][j][p][q] = ds_den * (exp_s_p[p][q] - exp_s_n[p][q]);
            d_R1_ds[i][j][p][q] = ds_den * (R1_p[p][q] - R1_n[p][q]);
            if (i != j) {
              d_exp_s_ds[j][i][p][q] = d_exp_s_ds[i][j][p][q];
              d_R1_ds[j][i][p][q] = d_R1_ds[i][j][p][q];
            }
          }
        }
        s_p[i][j] = s_n[i][j] = s[i][j];
        if (i != j) {
          s_p[j][i] = s_n[j][i] = s[j][i];
        }
      }
    }
  }
}